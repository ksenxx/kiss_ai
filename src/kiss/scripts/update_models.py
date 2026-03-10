#!/usr/bin/env python3
"""Fetch latest model pricing/context from vendor APIs, test new models,
and update model_info.py.

Usage:
    uv run python scripts/update_models.py [OPTIONS]

Options:
    --dry-run        Show what would change without modifying files
    --skip-test      Skip model capability testing for new models
    --test-existing  Re-test capabilities of existing models too
    --verbose        Print detailed progress
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import ssl
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_INFO_PATH = PROJECT_ROOT / "src" / "kiss" / "core" / "models" / "model_info.py"

_SSL_CTX = ssl.create_default_context()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def api_get(url: str, headers: dict[str, str] | None = None) -> Any:
    req = Request(url, headers=headers or {})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                return json.loads(resp.read())
        except Exception:
            logger.debug("Exception caught", exc_info=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fmt_price(p: float) -> str:
    if p == 0:
        return "0.00"
    if p == int(p):
        return f"{int(p):.2f}"
    s = f"{p:.3f}"
    if s[-1] == "0" and len(s.split(".")[1]) > 2:
        s = s[:-1]
    return s


# ---------------------------------------------------------------------------
# API Fetchers — each returns dict[model_name, dict] with ctx/pricing info
# ---------------------------------------------------------------------------


def fetch_openrouter(verbose: bool = False) -> dict[str, dict]:
    """Fetch all models from OpenRouter (public API, no auth).

    Models with an expiration_date in the past are filtered out.
    """
    if verbose:
        print("  Fetching OpenRouter models...")
    data = api_get("https://openrouter.ai/api/v1/models")
    today = datetime.date.today().isoformat()
    models: dict[str, dict] = {}
    skipped_deprecated = 0
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if not model_id:
            continue
        expiration = m.get("expiration_date")
        if expiration and expiration <= today:
            skipped_deprecated += 1
            continue
        pricing = m.get("pricing", {})
        prompt_per_tok = float(pricing.get("prompt") or "0")
        completion_per_tok = float(pricing.get("completion") or "0")
        ctx = m.get("context_length", 0)
        name = f"openrouter/{model_id}"
        models[name] = {
            "context_length": ctx,
            "input_price_per_1M": round(prompt_per_tok * 1_000_000, 3),
            "output_price_per_1M": round(completion_per_tok * 1_000_000, 3),
            "source": "openrouter",
        }
    if verbose:
        print(f"    Found {len(models)} models ({skipped_deprecated} deprecated filtered out)")
    return models


def fetch_together(verbose: bool = False) -> dict[str, dict]:
    """Fetch models from Together AI API (pricing is per-1M already)."""
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        print("  WARNING: TOGETHER_API_KEY not set, skipping Together AI")
        return {}
    if verbose:
        print("  Fetching Together AI models...")
    data = api_get(
        "https://api.together.xyz/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "kiss-update-models/1.0",
        },
    )
    from kiss.core.models.model_info import _TOGETHER_PREFIXES

    models: dict[str, dict] = {}
    for m in data:
        model_id = m.get("id", "")
        model_type = m.get("type", "")
        ctx = m.get("context_length", 0) or 0
        pricing = m.get("pricing", {})
        inp = float(pricing.get("input", 0) or 0)
        out = float(pricing.get("output", 0) or 0)
        if not model_id or not model_id.startswith(_TOGETHER_PREFIXES):
            continue
        if model_type not in ("chat", "embedding", "language"):
            continue
        is_emb = model_type == "embedding"
        models[model_id] = {
            "context_length": ctx,
            "input_price_per_1M": round(inp, 3),
            "output_price_per_1M": round(out, 3),
            "source": "together",
            "is_embedding": is_emb,
            "type": model_type,
        }
    if verbose:
        print(f"    Found {len(models)} relevant models")
    return models


def fetch_gemini(verbose: bool = False) -> dict[str, dict]:
    """Fetch models from Google Gemini API (context lengths, no pricing)."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("  WARNING: GEMINI_API_KEY not set, skipping Gemini")
        return {}
    if verbose:
        print("  Fetching Gemini models...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    data = api_get(url)
    skip_fragments = (
        "-latest",
        "-preview-tts",
        "-image-generation",
        "-image-preview",
        "-customtools",
        "-native-audio",
        "-computer-use",
        "-robotics",
    )
    models: dict[str, dict] = {}
    for m in data.get("models", []):
        raw_name = m.get("name", "")
        model_id = raw_name.replace("models/", "")
        if not model_id.startswith("gemini-"):
            continue
        if any(s in model_id for s in skip_fragments):
            continue
        ctx = m.get("inputTokenLimit", 0)
        methods = m.get("supportedGenerationMethods", [])
        is_emb = "embedContent" in methods
        is_gen = "generateContent" in methods
        models[model_id] = {
            "context_length": ctx,
            "source": "gemini",
            "is_embedding": is_emb,
            "is_generation": is_gen,
        }
    if verbose:
        print(f"    Found {len(models)} models")
    return models


def fetch_anthropic(verbose: bool = False) -> dict[str, dict]:
    """Fetch model list from Anthropic API (IDs only, no pricing/context)."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  WARNING: ANTHROPIC_API_KEY not set, skipping Anthropic")
        return {}
    if verbose:
        print("  Fetching Anthropic models...")
    data = api_get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
    )
    models: dict[str, dict] = {}
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if not model_id.startswith("claude-"):
            continue
        models[model_id] = {"source": "anthropic"}
    if verbose:
        print(f"    Found {len(models)} models")
    return models


def fetch_openai(verbose: bool = False) -> dict[str, dict]:
    """Fetch model list from OpenAI API (IDs and context, no pricing).

    Filters to models matching _OPENAI_PREFIXES so we only pick up chat /
    embedding models, not internal fine-tune artefacts.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("  WARNING: OPENAI_API_KEY not set, skipping OpenAI")
        return {}
    if verbose:
        print("  Fetching OpenAI models...")
    data = api_get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    from kiss.core.models.model_info import _OPENAI_PREFIXES

    skip_fragments = (
        "realtime",
        "audio",
        "transcribe",
        "tts",
        "whisper",
        "dall-e",
        "davinci",
        "babbage",
        "instruct",
        "search-api",
    )
    models: dict[str, dict] = {}
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if not model_id or not model_id.startswith(_OPENAI_PREFIXES):
            continue
        if any(f in model_id for f in skip_fragments):
            continue
        models[model_id] = {"source": "openai"}
    if verbose:
        print(f"    Found {len(models)} models")
    return models


# ---------------------------------------------------------------------------
# Get current MODEL_INFO
# ---------------------------------------------------------------------------


def get_current_model_info() -> dict[str, dict]:
    from kiss.core.models.model_info import MODEL_INFO

    return {
        name: {
            "context_length": info.context_length,
            "input_price_per_1M": info.input_price_per_1M,
            "output_price_per_1M": info.output_price_per_1M,
            "fc": info.is_function_calling_supported,
            "emb": info.is_embedding_supported,
            "gen": info.is_generation_supported,
        }
        for name, info in MODEL_INFO.items()
    }


# ---------------------------------------------------------------------------
# Model capability testing
# ---------------------------------------------------------------------------


def test_generate(model_name: str) -> bool:
    from kiss.core.models.model_info import model as create_model

    try:
        m = create_model(model_name)
        m.initialize("Say hello in one word.")
        text, _ = m.generate()
        return bool(text and text.strip())
    except Exception:
        logger.debug("Exception caught", exc_info=True)
        return False


def test_embedding(model_name: str) -> bool:
    from kiss.core.models.model_info import model as create_model

    try:
        m = create_model(model_name)
        m.initialize("")
        vec = m.get_embedding("Hello world")
        return isinstance(vec, list) and len(vec) > 0
    except Exception:
        logger.debug("Exception caught", exc_info=True)
        return False


def test_function_calling(model_name: str) -> bool:
    from kiss.core.models.model_info import model as create_model

    def calculator(expression: str = "") -> str:
        """Compute a math expression.

        Args:
            expression: A math expression string like '2+3'.
        """
        try:
            return str(eval(expression))
        except Exception:
            logger.debug("Exception caught", exc_info=True)
            return "error"

    try:
        m = create_model(model_name)
        m.initialize("What is 2+3? Use the calculator tool.")
        calls, _, _ = m.generate_and_process_with_tools({"calculator": calculator})
        return len(calls) > 0
    except Exception:
        logger.debug("Exception caught", exc_info=True)
        return False


def test_model_capabilities(
    model_name: str,
    verbose: bool = False,
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    if verbose:
        print(f"    Testing {model_name}...", end="", flush=True)

    results["gen"] = test_generate(model_name)
    time.sleep(0.5)

    results["emb"] = test_embedding(model_name)
    time.sleep(0.5)

    if results["gen"]:
        results["fc"] = test_function_calling(model_name)
        time.sleep(0.5)
    else:
        results["fc"] = False

    if verbose:
        flags = " ".join(f"{k}={'Y' if v else 'N'}" for k, v in results.items())
        print(f" {flags}")
    return results


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------


def find_deprecated_models(
    current: dict[str, dict],
    openrouter: dict[str, dict],
    anthropic: dict[str, dict],
    gemini: dict[str, dict],
    openai: dict[str, dict],
) -> list[dict]:
    """Identify models in current MODEL_INFO that are deprecated upstream.

    A model is considered deprecated if:
    - It's an openrouter/ model not present in the fetched OpenRouter list
      (which already filters out expired models).
    - It's a claude- model not returned by the Anthropic models API and not an
      alias (aliases don't have date suffixes and resolve to snapshot versions).
    - It's a gemini- model not returned by the Gemini models API.
    - It's an OpenAI model (gpt-/o1-/o3-/o4-/codex-) with a date suffix not
      returned by the OpenAI models API.
    """
    from kiss.core.models.model_info import _OPENAI_PREFIXES

    deprecated: list[dict] = []

    for name in current:
        if name.startswith("openrouter/"):
            if openrouter and name not in openrouter:
                base_name = name.split("/")[-1]
                if ":" in base_name:
                    continue
                deprecated.append({"name": name, "reason": "not in OpenRouter API"})
        elif name.startswith("claude-"):
            if anthropic and name not in anthropic:
                has_date = bool(re.search(r"\d{8}$", name))
                if has_date:
                    deprecated.append({"name": name, "reason": "not in Anthropic API"})
        elif name.startswith("gemini-") and not name.startswith("gemini-embedding"):
            if gemini and name not in gemini:
                deprecated.append({"name": name, "reason": "not in Gemini API"})
        elif name.startswith(_OPENAI_PREFIXES):
            if openai and name not in openai:
                has_date = bool(re.search(r"\d{4}-\d{2}-\d{2}$|\d{8}$", name))
                if has_date:
                    deprecated.append({"name": name, "reason": "not in OpenAI API"})

    return deprecated


def _strip_date_suffix(name: str) -> str:
    """Remove trailing date suffixes (YYYYMMDD or YYYY-MM-DD) for fuzzy lookup."""
    stripped = re.sub(r"-\d{8}$", "", name)
    if stripped != name:
        return stripped
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)


_VENDOR_OR_PREFIX: dict[str, str] = {
    "openai": "openrouter/openai/",
    "anthropic": "openrouter/anthropic/",
    "gemini": "openrouter/google/",
}


def _lookup_openrouter_pricing(
    model_name: str,
    source: str,
    openrouter: dict[str, dict],
) -> dict | None:
    """Cross-reference a vendor model name against OpenRouter for pricing/context.

    Tries an exact match first (e.g. ``gpt-5.4`` → ``openrouter/openai/gpt-5.4``),
    then falls back to the base name with date suffixes stripped (e.g.
    ``gpt-5.4-2026-03-05`` → ``openrouter/openai/gpt-5.4``).
    """
    prefix = _VENDOR_OR_PREFIX.get(source)
    if not prefix:
        return None
    or_key = f"{prefix}{model_name}"
    if or_key in openrouter:
        return openrouter[or_key]
    base = _strip_date_suffix(model_name)
    if base != model_name:
        or_key = f"{prefix}{base}"
        if or_key in openrouter:
            return openrouter[or_key]
    return None


def compute_changes(
    current: dict[str, dict],
    openrouter: dict[str, dict],
    together: dict[str, dict],
    gemini: dict[str, dict],
    anthropic: dict[str, dict],
    openai: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Compare fetched data with current MODEL_INFO.

    Returns (updates, new_models) where each is a list of dicts with model info.
    """
    updates: list[dict] = []
    new_models: list[dict] = []

    # --- OpenRouter models ---
    for name, fetched in openrouter.items():
        # Skip variant endpoints (:free, :exacto, :thinking, :extended, etc.)
        if ":" in name.split("/")[-1]:
            continue
        if name in current:
            cur = current[name]
            changed = {}
            if fetched["context_length"] and fetched["context_length"] != cur["context_length"]:
                changed["context_length"] = fetched["context_length"]
            if abs(fetched["input_price_per_1M"] - cur["input_price_per_1M"]) > 0.005:
                changed["input_price_per_1M"] = fetched["input_price_per_1M"]
            if abs(fetched["output_price_per_1M"] - cur["output_price_per_1M"]) > 0.005:
                changed["output_price_per_1M"] = fetched["output_price_per_1M"]
            if changed:
                updates.append({"name": name, "changes": changed, "source": "openrouter"})
        else:
            is_preview = "preview" in name.split("/")[-1]
            has_pricing = fetched["input_price_per_1M"] > 0
            if fetched["context_length"] and (has_pricing or is_preview):
                new_models.append(
                    {
                        "name": name,
                        "context_length": fetched["context_length"],
                        "input_price_per_1M": fetched["input_price_per_1M"],
                        "output_price_per_1M": fetched["output_price_per_1M"],
                        "source": "openrouter",
                        "needs_pricing": not has_pricing,
                    }
                )

    # --- Together AI models ---
    for name, fetched in together.items():
        if name in current:
            cur = current[name]
            changed = {}
            if fetched["context_length"] and fetched["context_length"] != cur["context_length"]:
                changed["context_length"] = fetched["context_length"]
            inp_diff = abs(fetched["input_price_per_1M"] - cur["input_price_per_1M"])
            out_diff = abs(fetched["output_price_per_1M"] - cur["output_price_per_1M"])
            if inp_diff > 0.005 and not cur["emb"]:
                changed["input_price_per_1M"] = fetched["input_price_per_1M"]
            if out_diff > 0.005 and not cur["emb"]:
                changed["output_price_per_1M"] = fetched["output_price_per_1M"]
            if changed:
                updates.append({"name": name, "changes": changed, "source": "together"})
        else:
            is_preview = "preview" in name.split("/")[-1]
            has_pricing = fetched["input_price_per_1M"] > 0
            if (
                fetched["context_length"]
                and fetched.get("type") in ("chat", "embedding")
                and (has_pricing or is_preview)
            ):
                new_models.append(
                    {
                        "name": name,
                        "context_length": fetched["context_length"],
                        "input_price_per_1M": fetched["input_price_per_1M"],
                        "output_price_per_1M": fetched["output_price_per_1M"],
                        "source": "together",
                        "is_embedding": fetched.get("is_embedding", False),
                        "needs_pricing": not has_pricing,
                    }
                )

    # --- Gemini models (context from API, pricing from OpenRouter cross-ref) ---
    for name, fetched in gemini.items():
        if name in current:
            cur = current[name]
            if fetched["context_length"] and fetched["context_length"] != cur["context_length"]:
                updates.append(
                    {
                        "name": name,
                        "changes": {"context_length": fetched["context_length"]},
                        "source": "gemini",
                    }
                )
        else:
            or_info = _lookup_openrouter_pricing(name, "gemini", openrouter)
            inp = or_info["input_price_per_1M"] if or_info else 0.0
            out = or_info["output_price_per_1M"] if or_info else 0.0
            new_models.append(
                {
                    "name": name,
                    "context_length": fetched["context_length"],
                    "input_price_per_1M": inp,
                    "output_price_per_1M": out,
                    "source": "gemini",
                    "needs_pricing": inp == 0,
                }
            )

    # --- Anthropic models (pricing/context from OpenRouter cross-ref) ---
    for name in anthropic:
        if name not in current:
            or_info = _lookup_openrouter_pricing(name, "anthropic", openrouter)
            ctx = or_info["context_length"] if or_info and or_info.get("context_length") else 200000
            inp = or_info["input_price_per_1M"] if or_info else 0.0
            out = or_info["output_price_per_1M"] if or_info else 0.0
            new_models.append(
                {
                    "name": name,
                    "context_length": ctx,
                    "input_price_per_1M": inp,
                    "output_price_per_1M": out,
                    "source": "anthropic",
                    "needs_pricing": inp == 0,
                }
            )

    # --- OpenAI models (pricing/context from OpenRouter cross-ref) ---
    for name in openai:
        if name not in current:
            or_info = _lookup_openrouter_pricing(name, "openai", openrouter)
            ctx = or_info["context_length"] if or_info and or_info.get("context_length") else 0
            inp = or_info["input_price_per_1M"] if or_info else 0.0
            out = or_info["output_price_per_1M"] if or_info else 0.0
            new_models.append(
                {
                    "name": name,
                    "context_length": ctx,
                    "input_price_per_1M": inp,
                    "output_price_per_1M": out,
                    "source": "openai",
                    "needs_pricing": inp == 0,
                }
            )

    # --- Backfill pricing/context for existing zero-priced models via OpenRouter ---
    from kiss.core.models.model_info import _OPENAI_PREFIXES

    update_by_name = {upd["name"]: upd for upd in updates}
    for name, cur in current.items():
        if name.startswith("openrouter/"):
            continue
        has_pricing = cur["input_price_per_1M"] > 0
        has_context = cur["context_length"] > 0
        if has_pricing and has_context:
            continue
        source = None
        if name.startswith(_OPENAI_PREFIXES):
            source = "openai"
        elif name.startswith("claude"):
            source = "anthropic"
        elif name.startswith("gemini-"):
            source = "gemini"
        if not source:
            continue
        or_info = _lookup_openrouter_pricing(name, source, openrouter)
        if not or_info:
            continue
        changed: dict[str, Any] = {}
        if not has_pricing and or_info.get("input_price_per_1M", 0) > 0:
            changed["input_price_per_1M"] = or_info["input_price_per_1M"]
            changed["output_price_per_1M"] = or_info["output_price_per_1M"]
        if not has_context and or_info.get("context_length", 0) > 0:
            changed["context_length"] = or_info["context_length"]
        if not changed:
            continue
        if name in update_by_name:
            update_by_name[name]["changes"].update(changed)
        else:
            updates.append({"name": name, "changes": changed, "source": "openrouter-xref"})

    return updates, new_models


# ---------------------------------------------------------------------------
# File update
# ---------------------------------------------------------------------------


def _make_entry_line(
    name: str,
    ctx: int,
    inp: float,
    out: float,
    fc: bool = True,
    emb: bool = False,
    gen: bool = True,
    comment: str = "",
) -> str:
    if emb and not gen:
        line = f'    "{name}": _emb({ctx}, {fmt_price(inp)}),'
    else:
        args = f"{ctx}, {fmt_price(inp)}, {fmt_price(out)}"
        extras = []
        if not fc:
            extras.append("fc=False")
        if emb:
            extras.append("emb=True")
        if not gen:
            extras.append("gen=False")
        if extras:
            args += ", " + ", ".join(extras)
        line = f'    "{name}": _mi({args}),'
    if comment and len(line) + len(comment) + 4 <= 100:
        line += f"  # {comment}"
    return line


def apply_updates_to_file(
    updates: list[dict],
    new_models: list[dict],
    deprecated: list[dict],
    current: dict[str, dict],
    dry_run: bool = False,
) -> None:
    content = MODEL_INFO_PATH.read_text()
    lines = content.split("\n")

    # --- Helpers for multi-line entries ---
    _key_pat = re.compile(r'^\s+"[^"]+"\s*:')

    def _find_entry_span(lines: list[str], name: str) -> tuple[int, int]:
        """Return (start, end) indices of a model entry, handling multi-line spans."""
        pat = re.compile(rf'^\s+"{re.escape(name)}"\s*:')
        for i, line in enumerate(lines):
            if pat.match(line):
                if line.rstrip().endswith(","):
                    return i, i + 1
                for j in range(i + 1, len(lines)):
                    if lines[j].rstrip().endswith(",") or _key_pat.match(lines[j]):
                        if _key_pat.match(lines[j]):
                            return i, j
                        return i, j + 1
                return i, i + 1
        return -1, -1

    # --- Remove deprecated models ---
    deprecated_names = {d["name"] for d in deprecated}
    removed = 0
    if deprecated_names:
        spans: list[tuple[int, int]] = []
        for name in deprecated_names:
            start, end = _find_entry_span(lines, name)
            if start >= 0:
                spans.append((start, end))
        for start, end in sorted(spans, reverse=True):
            del lines[start:end]
            removed += 1

    # --- Apply pricing/context updates ---
    applied_updates = 0
    for upd in updates:
        name = upd["name"]
        cur = current[name]
        new_ctx = upd["changes"].get("context_length", cur["context_length"])
        new_inp = upd["changes"].get("input_price_per_1M", cur["input_price_per_1M"])
        new_out = upd["changes"].get("output_price_per_1M", cur["output_price_per_1M"])
        new_line = _make_entry_line(
            name,
            new_ctx,
            new_inp,
            new_out,
            fc=cur["fc"],
            emb=cur["emb"],
            gen=cur["gen"],
        )
        start, end = _find_entry_span(lines, name)
        if start >= 0:
            old_first = lines[start]
            old_comment = ""
            if "#" in old_first:
                old_comment = old_first[old_first.index("#") + 1 :].strip()
            if old_comment and len(new_line) + len(old_comment) + 4 <= 100:
                new_line += f"  # {old_comment}"
            lines[start:end] = [new_line]
            applied_updates += 1

    # --- Add new models ---
    # Find the closing "}" of MODEL_INFO dict (first "}" on its own line
    # after the MODEL_INFO opening)
    added = 0
    insert_before_closing = -1
    in_model_info = False
    for i, line in enumerate(lines):
        if "MODEL_INFO" in line and "{" in line:
            in_model_info = True
        if in_model_info and line.strip() == "}":
            insert_before_closing = i
            break

    new_lines_to_add: list[str] = []
    for nm in new_models:
        name = nm["name"]
        if nm.get("needs_pricing"):
            comment = "NEW: needs pricing"
        else:
            comment = "NEW"
        entry_line = _make_entry_line(
            name,
            nm["context_length"],
            nm["input_price_per_1M"],
            nm["output_price_per_1M"],
            fc=nm.get("fc", True),
            emb=nm.get("emb", False),
            gen=nm.get("gen", True),
            comment=comment,
        )
        new_lines_to_add.append(entry_line)
        added += 1

    if new_lines_to_add and insert_before_closing >= 0:
        for line in reversed(new_lines_to_add):
            lines.insert(insert_before_closing, line)

    # --- Sort all MODEL_INFO entries alphabetically ---
    # Find the MODEL_INFO dict boundaries, extract entries, sort, replace.
    # Entries may span multiple lines (e.g. long _mi() calls), so we group
    # continuation lines with the preceding key line before sorting.
    dict_start = -1
    dict_end = -1
    in_mi = False
    for i, line in enumerate(lines):
        if "MODEL_INFO" in line and "{" in line:
            dict_start = i + 1
            in_mi = True
        if in_mi and line.strip() == "}":
            dict_end = i
            break

    if dict_start >= 0 and dict_end > dict_start:
        standalone_comments: list[str] = []
        entry_blocks: list[list[str]] = []
        current_block: list[str] = []

        for line in lines[dict_start:dict_end]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if current_block:
                    current_block.append(line)
                elif not stripped.startswith("# ==="):
                    standalone_comments.append(line)
                continue
            if re.match(r'\s+"[^"]+"\s*:', line):
                if current_block:
                    entry_blocks.append(current_block)
                current_block = [line]
            else:
                current_block.append(line)

        if current_block:
            entry_blocks.append(current_block)

        def _sort_key(block: list[str]) -> str:
            m = re.search(r'"([^"]+)"', block[0])
            return m.group(1).lower() if m else block[0]

        entry_blocks.sort(key=_sort_key)
        sorted_lines: list[str] = standalone_comments[:]
        for block in entry_blocks:
            sorted_lines.extend(block)
        lines[dict_start:dict_end] = sorted_lines

    print(f"\n  Removed {removed} deprecated, applied {applied_updates} updates, added {added} new")
    if not dry_run:
        MODEL_INFO_PATH.write_text("\n".join(lines))
        print(f"  Written to {MODEL_INFO_PATH}")
    else:
        print("  (dry-run, no files modified)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Update model_info.py from vendor APIs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't modify files",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip capability testing")
    parser.add_argument("--test-existing", action="store_true", help="Re-test existing models")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Model Info Updater")
    print("=" * 60)

    # 1. Load current MODEL_INFO
    print("\n[1/6] Loading current MODEL_INFO...")
    current = get_current_model_info()
    print(f"  {len(current)} models loaded")

    # 2. Fetch from vendor APIs
    print("\n[2/6] Fetching from vendor APIs...")
    openrouter_models = fetch_openrouter(verbose=args.verbose)
    together_models = fetch_together(verbose=args.verbose)
    gemini_models = fetch_gemini(verbose=args.verbose)
    anthropic_models = fetch_anthropic(verbose=args.verbose)
    openai_models = fetch_openai(verbose=args.verbose)

    # 3. Detect deprecated models
    print("\n[3/6] Detecting deprecated models...")
    deprecated = find_deprecated_models(
        current,
        openrouter_models,
        anthropic_models,
        gemini_models,
        openai_models,
    )
    if deprecated:
        print(f"\n  Deprecated models in MODEL_INFO ({len(deprecated)}):")
        for dep in deprecated:
            print(f"    {dep['name']} ({dep['reason']})")
    else:
        print("  No deprecated models found")

    # 4. Compute diff
    print("\n[4/6] Computing changes...")
    updates, new_models = compute_changes(
        current,
        openrouter_models,
        together_models,
        gemini_models,
        anthropic_models,
        openai_models,
    )

    # Print summary
    if updates:
        print(f"\n  Pricing/context updates ({len(updates)}):")
        for upd in updates:
            changes_str = ", ".join(
                f"{k}: {current[upd['name']].get(k, '?')} -> {v}" for k, v in upd["changes"].items()
            )
            print(f"    {upd['name']}: {changes_str}")
    else:
        print("\n  No pricing/context updates needed")

    if new_models:
        print(f"\n  New models discovered ({len(new_models)}):")
        for nm in new_models[:50]:
            pricing = ""
            if not nm.get("needs_pricing"):
                pricing = f" ${nm['input_price_per_1M']}/{nm['output_price_per_1M']}"
            print(f"    {nm['name']} (ctx={nm['context_length']}{pricing}) [{nm['source']}]")
        if len(new_models) > 50:
            print(f"    ... and {len(new_models) - 50} more")
    else:
        print("\n  No new models discovered")

    # Filter out models that are deprecated in any vendor from the new list
    deprecated_names = {d["name"] for d in deprecated}
    new_models = [nm for nm in new_models if nm["name"] not in deprecated_names]

    if not updates and not new_models and not deprecated:
        print("\nEverything is up to date!")
        return

    # 5. Test new models
    if new_models and not args.skip_test:
        print(f"\n[5/6] Testing {len(new_models)} new models...")
        for nm in new_models:
            caps = test_model_capabilities(nm["name"], verbose=args.verbose)
            nm["gen"] = caps["gen"]
            nm["emb"] = caps["emb"]
            nm["fc"] = caps["fc"]
            if not caps["gen"] and not caps["emb"]:
                nm["_skip"] = True
        new_models = [nm for nm in new_models if not nm.get("_skip")]
        print(f"  {len(new_models)} models passed testing")
    elif new_models and args.skip_test:
        print("\n[5/6] Skipping model testing (--skip-test)")
        for nm in new_models:
            nm["fc"] = True
            nm["gen"] = not nm.get("is_embedding", False)
            nm["emb"] = nm.get("is_embedding", False)
    else:
        print("\n[5/6] No new models to test")

    # Optionally re-test existing models
    if args.test_existing:
        print("\n  Re-testing existing models...")
        for upd in updates:
            name = upd["name"]
            caps = test_model_capabilities(name, verbose=args.verbose)
            cur = current[name]
            if caps["fc"] != cur["fc"]:
                upd["changes"]["fc"] = caps["fc"]
                print(f"    {name}: fc changed {cur['fc']} -> {caps['fc']}")

    # 6. Apply changes
    print("\n[6/6] Applying changes...")
    apply_updates_to_file(updates, new_models, deprecated, current, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
