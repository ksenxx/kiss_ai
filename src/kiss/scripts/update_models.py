#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fetch latest model pricing/context from vendor APIs, test new models,
and update MODEL_INFO.json.

The script writes the source-of-truth ``src/kiss/core/models/MODEL_INFO.json``
in the repo. When the user-local copy at ``~/.kiss/MODEL_INFO.json`` exists,
it is also refreshed so a running KISS install picks up the changes on its
next ``model_info`` reload without waiting for a reinstall.

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

_EXPECTED_SUBPATH = Path("src") / "kiss" / "core" / "models" / "MODEL_INFO.json"


def _find_project_root() -> Path:
    """Find the project root directory for writing MODEL_INFO.json.

    Checks in order:
    1. KISS_WORKDIR environment variable (set by the KISS agent runtime)
    2. Current working directory if it contains a .git directory and the
       expected source structure
    3. __file__-based resolution (fallback for direct invocation)

    This avoids a bug where running from the VS Code extension's bundled
    copy of the project would write to the extension directory instead of
    the actual source repository.
    """
    # 1. KISS_WORKDIR env var — set by the agent runtime
    workdir = os.environ.get("KISS_WORKDIR", "")
    if workdir:
        p = Path(workdir)
        if (p / _EXPECTED_SUBPATH).exists():
            return p

    # 2. CWD with .git marker (a real git checkout, not a bundled copy)
    cwd = Path.cwd()
    if (cwd / ".git").exists() and (cwd / _EXPECTED_SUBPATH).exists():
        return cwd

    # 3. Fallback: derive from script location
    return Path(__file__).resolve().parent.parent.parent.parent


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_INFO_PATH = PROJECT_ROOT / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
USER_MODEL_INFO_PATH = Path.home() / ".kiss" / "MODEL_INFO.json"

_SSL_CTX = ssl.create_default_context()


def api_get(url: str, headers: dict[str, str] | None = None) -> Any:
    req = Request(url, headers=headers or {})
    for attempt in range(3):  # pragma: no branch
        try:
            with urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                return json.loads(resp.read())
        except Exception:
            logger.debug("Exception caught", exc_info=True)
            if attempt == 2:  # pragma: no branch
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_openrouter(verbose: bool = False) -> dict[str, dict]:
    """Fetch all models from OpenRouter (public API, no auth).

    Models with an expiration_date in the past are filtered out.
    """
    if verbose:  # pragma: no branch
        print("  Fetching OpenRouter models...")
    data = api_get("https://openrouter.ai/api/v1/models")
    today = datetime.date.today().isoformat()
    models: dict[str, dict] = {}
    skipped_deprecated = 0
    for m in data.get("data", []):  # pragma: no branch
        model_id = m.get("id", "")
        if not model_id:  # pragma: no branch
            continue
        expiration = m.get("expiration_date")
        if expiration and expiration <= today:  # pragma: no branch
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
    if verbose:  # pragma: no branch
        print(f"    Found {len(models)} models ({skipped_deprecated} deprecated filtered out)")
    return models


def fetch_together(verbose: bool = False) -> dict[str, dict]:
    """Fetch models from Together AI API (pricing is per-1M already)."""
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:  # pragma: no branch
        print("  WARNING: TOGETHER_API_KEY not set, skipping Together AI")
        return {}
    if verbose:  # pragma: no branch
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
    for m in data:  # pragma: no branch
        model_id = m.get("id", "")
        model_type = m.get("type", "")
        ctx = m.get("context_length", 0) or 0
        pricing = m.get("pricing", {})
        inp = float(pricing.get("input", 0) or 0)
        out = float(pricing.get("output", 0) or 0)
        if not model_id or not model_id.startswith(_TOGETHER_PREFIXES):  # pragma: no branch
            continue
        if model_type not in ("chat", "embedding", "language"):  # pragma: no branch
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
    if verbose:  # pragma: no branch
        print(f"    Found {len(models)} relevant models")
    return models


def fetch_gemini(verbose: bool = False) -> dict[str, dict]:
    """Fetch models from Google Gemini API (context lengths, no pricing)."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:  # pragma: no branch
        print("  WARNING: GEMINI_API_KEY not set, skipping Gemini")
        return {}
    if verbose:  # pragma: no branch
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
    for m in data.get("models", []):  # pragma: no branch
        raw_name = m.get("name", "")
        model_id = raw_name.replace("models/", "")
        if not model_id.startswith("gemini-"):  # pragma: no branch
            continue
        if any(s in model_id for s in skip_fragments):  # pragma: no branch
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
    if verbose:  # pragma: no branch
        print(f"    Found {len(models)} models")
    return models


def fetch_anthropic(verbose: bool = False) -> dict[str, dict]:
    """Fetch model list from Anthropic API (IDs only, no pricing/context)."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:  # pragma: no branch
        print("  WARNING: ANTHROPIC_API_KEY not set, skipping Anthropic")
        return {}
    if verbose:  # pragma: no branch
        print("  Fetching Anthropic models...")
    data = api_get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
    )
    models: dict[str, dict] = {}
    for m in data.get("data", []):  # pragma: no branch
        model_id = m.get("id", "")
        if not model_id.startswith("claude-"):  # pragma: no branch
            continue
        models[model_id] = {"source": "anthropic"}
    if verbose:  # pragma: no branch
        print(f"    Found {len(models)} models")
    return models


def fetch_openai(verbose: bool = False) -> dict[str, dict]:
    """Fetch model list from OpenAI API (IDs and context, no pricing).

    Filters to models matching _OPENAI_PREFIXES so we only pick up chat /
    embedding models, not internal fine-tune artefacts.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:  # pragma: no branch
        print("  WARNING: OPENAI_API_KEY not set, skipping OpenAI")
        return {}
    if verbose:  # pragma: no branch
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
    for m in data.get("data", []):  # pragma: no branch
        model_id = m.get("id", "")
        if not model_id or not model_id.startswith(_OPENAI_PREFIXES):  # pragma: no branch
            continue
        if any(f in model_id for f in skip_fragments):  # pragma: no branch
            continue
        models[model_id] = {"source": "openai"}
    if verbose:  # pragma: no branch
        print(f"    Found {len(models)} models")
    return models


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
            "thinking": info.thinking,
        }
        for name, info in MODEL_INFO.items()
    }


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


_THINKING_LEVELS_TO_PROBE: tuple[str, ...] = ("xhigh",)


def detect_thinking_level(model_name: str) -> str | None:
    """Detect the highest ``reasoning_effort`` level the model accepts.

    Probes each level in :data:`_THINKING_LEVELS_TO_PROBE` (currently just
    ``"xhigh"``) by issuing a minimal generate call with
    ``model_config={"reasoning_effort": <level>}`` explicitly so that the
    OpenAI Chat Completions API itself decides the verdict, regardless of
    whether the model is already flagged in ``MODEL_INFO``. Returns the
    first level that succeeds, or ``None`` if none did.

    Returns ``None`` (without making any API call) for backends that don't
    accept ``reasoning_effort`` at all:

    * ``codex/*`` — routed through the Codex CLI, which controls reasoning
      via its own ``model_reasoning_effort`` config rather than per-call.
    * ``claude-*``, ``gemini-*`` — non-OpenAI providers that don't accept
      ``reasoning_effort``.
    * Variants known to reject ``reasoning_effort`` entirely (``-pro``,
      ``-chat-latest``, ``-image``).
    """
    from kiss.core.models.model_info import _OPENAI_PREFIXES

    if model_name.startswith(("codex/", "claude-", "gemini-")):
        return None
    if any(marker in model_name for marker in ("-pro", "chat-latest", "-image")):
        return None
    is_openai = model_name.startswith(_OPENAI_PREFIXES) and not model_name.startswith(
        "text-embedding"
    )
    is_openrouter_openai = model_name.startswith(
        ("openrouter/openai/", "openrouter/~openai/")
    )
    if not (is_openai or is_openrouter_openai):
        return None

    from kiss.core.models.model_info import model as create_model

    for level in _THINKING_LEVELS_TO_PROBE:
        try:
            m = create_model(model_name, model_config={"reasoning_effort": level})
            m.initialize("Say hello in one word.")
            text, _ = m.generate()
            if text and text.strip():
                return level
        except Exception:
            logger.debug("Exception caught", exc_info=True)
            continue
    return None


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
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if verbose:  # pragma: no branch
        print(f"    Testing {model_name}...", end="", flush=True)

    results["gen"] = test_generate(model_name)
    time.sleep(0.5)

    results["emb"] = test_embedding(model_name)
    time.sleep(0.5)

    if results["gen"]:  # pragma: no branch
        results["fc"] = test_function_calling(model_name)
        time.sleep(0.5)
    else:
        results["fc"] = False

    if results["gen"]:  # pragma: no branch
        results["thinking"] = detect_thinking_level(model_name)
        time.sleep(0.5)
    else:
        results["thinking"] = None

    if verbose:  # pragma: no branch
        flags = " ".join(
            f"{k}={v if isinstance(v, str) else ('Y' if v else 'N')}"
            for k, v in results.items()
        )
        print(f" {flags}")
    return results


def find_deprecated_models(
    current: dict[str, dict],
    openrouter: dict[str, dict],
    anthropic: dict[str, dict],
    gemini: dict[str, dict],
    openai: dict[str, dict],
    codex_slugs: set[str] | None = None,
) -> list[dict]:
    """Identify models in current MODEL_INFO that are deprecated upstream.

    A model is considered deprecated if:
    - It's a codex/ model whose slug is not in the Codex CLI's official
      models.json (except ``codex/default`` which is always kept).
    - It's an openrouter/ model not present in the fetched OpenRouter list
      (which already filters out expired models).
    - It's a claude- model not returned by the Anthropic models API and not an
      alias (aliases don't have date suffixes and resolve to snapshot versions).
    - It's a gemini- model not returned by the Gemini models API.
    - It's an OpenAI model (gpt-/o1-/o3-/o4-/codex-) not returned by the
      OpenAI models API and not an alias whose dated snapshots still exist.
    """
    from kiss.core.models.model_info import _OPENAI_PREFIXES

    deprecated: list[dict] = []

    for name in current:  # pragma: no branch
        if name.startswith("codex/"):  # pragma: no branch
            if codex_slugs and name != "codex/default":
                slug = name.removeprefix("codex/")
                if slug not in codex_slugs:
                    deprecated.append(
                        {"name": name, "reason": "not in Codex CLI models.json"}
                    )
            continue
        if name.startswith("openrouter/"):  # pragma: no branch
            if openrouter and name not in openrouter:  # pragma: no branch
                base_name = name.split("/")[-1]
                if ":" in base_name:  # pragma: no branch
                    continue
                deprecated.append({"name": name, "reason": "not in OpenRouter API"})
        elif name.startswith("claude-"):  # pragma: no branch
            if anthropic and name not in anthropic:  # pragma: no branch
                has_date = bool(re.search(r"\d{8}$", name))
                if has_date:  # pragma: no branch
                    deprecated.append({"name": name, "reason": "not in Anthropic API"})
                else:
                    alias_re = re.compile(rf"^{re.escape(name)}-\d{{8}}$")
                    if not any(alias_re.match(n) for n in anthropic):
                        deprecated.append(
                            {"name": name, "reason": "alias with no snapshot in Anthropic API"}
                        )
        elif (  # pragma: no branch
            name.startswith("gemini-") and not name.startswith("gemini-embedding")
        ):
            if gemini and name not in gemini:  # pragma: no branch
                deprecated.append({"name": name, "reason": "not in Gemini API"})
        elif name.startswith(_OPENAI_PREFIXES):  # pragma: no branch
            if openai and name not in openai:  # pragma: no branch
                has_date = bool(re.search(r"\d{4}-\d{2}-\d{2}$|\d{8}$", name))
                if has_date:  # pragma: no branch
                    deprecated.append({"name": name, "reason": "not in OpenAI API"})
                else:
                    alias_re = re.compile(
                        rf"^{re.escape(name)}-(\d{{8}}|\d{{4}}-\d{{2}}-\d{{2}})$"
                    )
                    if not any(alias_re.match(n) for n in openai):
                        deprecated.append(
                            {"name": name, "reason": "alias with no snapshot in OpenAI API"}
                        )

    return deprecated


_GPT_PRO_OR_CODEX_RE = re.compile(r"-(pro|codex)(-|$)")


def _is_excluded_gpt_pro_or_codex(name: str) -> bool:
    """Return True for GPT ``-pro`` / ``-codex`` variants we never auto-add.

    GPT ``-pro`` slugs (``gpt-5-pro``, ``gpt-5.5-pro-2026-04-23``, ...) reject
    the ``reasoning_effort`` parameter, are billed at premium tiers, and the
    Codex CLI rejects them at runtime for ChatGPT-account users. GPT
    ``-codex`` slugs (``gpt-5-codex``, ``gpt-5.1-codex-max``, ...) are
    intended to be exercised through the Codex CLI backend rather than
    direct Chat Completions. In both cases the discovery flow must silently
    skip those names so they are never appended to ``model_info.py``.

    The match is scoped to the GPT family by checking that the base name
    (last ``/``-separated segment) starts with ``gpt-``; this keeps unrelated
    vendor models (e.g. a hypothetical ``acme/super-pro``) eligible for
    addition. Matches both undated forms (``gpt-5-pro``, ``gpt-5.3-codex``)
    and dated snapshots (``gpt-5-pro-2025-10-06``), and applies equally to
    OpenRouter passthroughs (``openrouter/openai/gpt-5-pro``) and Codex CLI
    keys (``codex/gpt-5.3-codex``).
    """
    base = name.rsplit("/", 1)[-1]
    if not base.startswith("gpt-"):
        return False
    return bool(_GPT_PRO_OR_CODEX_RE.search(base))


def _strip_date_suffix(name: str) -> str:
    """Remove trailing date suffixes (YYYYMMDD or YYYY-MM-DD) for fuzzy lookup."""
    stripped = re.sub(r"-\d{8}$", "", name)
    if stripped != name:  # pragma: no branch
        return stripped
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)


_VENDOR_OR_PREFIX: dict[str, str] = {
    "openai": "openrouter/openai/",
    "anthropic": "openrouter/anthropic/",
    "gemini": "openrouter/google/",
}

_CODEX_MODELS_JSON_URL = (
    "https://raw.githubusercontent.com/openai/codex/main/"
    "codex-rs/models-manager/models.json"
)


def fetch_codex_supported_slugs(verbose: bool = False) -> set[str]:
    """Fetch the list of model slugs the Codex CLI actually supports.

    Reads the official ``models.json`` from the openai/codex repository on
    GitHub. Returns a set of slug strings (e.g. ``{"gpt-5.5", "gpt-5.4"}``).
    Returns an empty set on network failure so the caller can skip codex
    candidate generation rather than adding unsupported models.
    """
    if verbose:  # pragma: no branch
        print("  Fetching Codex supported models...")
    try:
        data = api_get(_CODEX_MODELS_JSON_URL)
        slugs = {m["slug"] for m in data.get("models", []) if m.get("slug")}
        if verbose:  # pragma: no branch
            print(f"    Found {len(slugs)} supported Codex slugs")
        return slugs
    except Exception:
        logger.debug("Failed to fetch Codex models.json", exc_info=True)
        if verbose:  # pragma: no branch
            print("    WARNING: Could not fetch Codex models.json, skipping codex candidates")
        return set()


def _add_codex_candidates(
    codex_slugs: set[str],
    current: dict[str, dict],
    openrouter: dict[str, dict],
    new_models: list[dict],
) -> None:
    """Add ``codex/<slug>`` entries for models the Codex CLI supports.

    Only models whose slug appears in the official Codex CLI ``models.json``
    are added. This avoids adding models that the Codex CLI rejects at
    runtime (e.g. ``gpt-5.5-pro`` is not supported with a ChatGPT account).

    Context length is taken from the matching OpenRouter entry when
    available, falling back to 400000 (the default for codex/* models).
    All entries get $0/0 pricing since Codex is billed via the user's
    ChatGPT subscription.
    """
    for slug in codex_slugs:  # pragma: no branch
        codex_name = f"codex/{slug}"
        if codex_name in current:  # pragma: no branch
            continue
        if _is_excluded_gpt_pro_or_codex(codex_name):  # pragma: no branch
            continue
        or_info = _lookup_openrouter_pricing(slug, "openai", openrouter)
        ctx = or_info["context_length"] if or_info and or_info.get("context_length") else 400000
        new_models.append(
            {
                "name": codex_name,
                "context_length": ctx,
                "input_price_per_1M": 0.0,
                "output_price_per_1M": 0.0,
                "source": "codex",
                "needs_pricing": False,
                "gen": True,
                "fc": True,
                "emb": False,
            }
        )


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
    if not prefix:  # pragma: no branch
        return None
    or_key = f"{prefix}{model_name}"
    if or_key in openrouter:  # pragma: no branch
        return openrouter[or_key]
    base = _strip_date_suffix(model_name)
    if base != model_name:  # pragma: no branch
        or_key = f"{prefix}{base}"
        if or_key in openrouter:  # pragma: no branch
            return openrouter[or_key]
    return None


def compute_changes(
    current: dict[str, dict],
    openrouter: dict[str, dict],
    together: dict[str, dict],
    gemini: dict[str, dict],
    anthropic: dict[str, dict],
    openai: dict[str, dict],
    codex_slugs: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Compare fetched data with current MODEL_INFO.

    Returns (updates, new_models) where each is a list of dicts with model info.
    """
    updates: list[dict] = []
    new_models: list[dict] = []

    for name, fetched in openrouter.items():
        if ":" in name.split("/")[-1]:
            continue
        if name in current:
            cur = current[name]
            changed = {}
            ctx = fetched["context_length"]
            if (  # pragma: no branch
                ctx and ctx != cur["context_length"]
            ):
                changed["context_length"] = ctx
            inp_delta = abs(fetched["input_price_per_1M"] - cur["input_price_per_1M"])
            if inp_delta > 0.005:  # pragma: no branch
                changed["input_price_per_1M"] = fetched["input_price_per_1M"]
            out_delta = abs(fetched["output_price_per_1M"] - cur["output_price_per_1M"])
            if out_delta > 0.005:  # pragma: no branch
                changed["output_price_per_1M"] = fetched["output_price_per_1M"]
            if changed:  # pragma: no branch
                updates.append({"name": name, "changes": changed, "source": "openrouter"})
        else:
            if _is_excluded_gpt_pro_or_codex(name):
                continue
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

    for name, fetched in together.items():
        if name in current:  # pragma: no branch
            cur = current[name]
            changed = {}
            if (  # pragma: no branch
                fetched["context_length"] and fetched["context_length"] != cur["context_length"]
            ):
                changed["context_length"] = fetched["context_length"]
            inp_diff = abs(fetched["input_price_per_1M"] - cur["input_price_per_1M"])
            out_diff = abs(fetched["output_price_per_1M"] - cur["output_price_per_1M"])
            if inp_diff > 0.005 and not cur["emb"]:  # pragma: no branch
                changed["input_price_per_1M"] = fetched["input_price_per_1M"]
            if out_diff > 0.005 and not cur["emb"]:  # pragma: no branch
                changed["output_price_per_1M"] = fetched["output_price_per_1M"]
            if changed:  # pragma: no branch
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

    for name, fetched in gemini.items():
        if name in current:  # pragma: no branch
            cur = current[name]
            if (  # pragma: no branch
                fetched["context_length"] and fetched["context_length"] != cur["context_length"]
            ):
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

    for name in anthropic:  # pragma: no branch
        if name not in current:  # pragma: no branch
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

    for name in openai:  # pragma: no branch
        if name not in current:  # pragma: no branch
            if _is_excluded_gpt_pro_or_codex(name):  # pragma: no branch
                continue
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

    if codex_slugs:  # pragma: no branch
        _add_codex_candidates(codex_slugs, current, openrouter, new_models)

    from kiss.core.models.model_info import _OPENAI_PREFIXES

    update_by_name = {upd["name"]: upd for upd in updates}
    for name, cur in current.items():
        if name.startswith("openrouter/"):  # pragma: no branch
            continue
        has_pricing = cur["input_price_per_1M"] > 0
        has_context = cur["context_length"] > 0
        if has_pricing and has_context:  # pragma: no branch
            continue
        source = None
        if name.startswith(_OPENAI_PREFIXES):  # pragma: no branch
            source = "openai"
        elif name.startswith("claude"):  # pragma: no branch
            source = "anthropic"
        elif name.startswith("gemini-"):  # pragma: no branch
            source = "gemini"
        if not source:  # pragma: no branch
            continue
        or_info = _lookup_openrouter_pricing(name, source, openrouter)
        if not or_info:  # pragma: no branch
            continue
        changed = {}
        if not has_pricing and or_info.get("input_price_per_1M", 0) > 0:  # pragma: no branch
            changed["input_price_per_1M"] = or_info["input_price_per_1M"]
            changed["output_price_per_1M"] = or_info["output_price_per_1M"]
        if not has_context and or_info.get("context_length", 0) > 0:  # pragma: no branch
            changed["context_length"] = or_info["context_length"]
        if not changed:  # pragma: no branch
            continue
        if name in update_by_name:  # pragma: no branch
            update_by_name[name]["changes"].update(changed)
        else:
            updates.append({"name": name, "changes": changed, "source": "openrouter-xref"})

    return updates, new_models


def _build_entry(
    ctx: int,
    inp: float,
    out: float,
    fc: bool = True,
    emb: bool = False,
    gen: bool = True,
    thinking: str | None = None,
    comment: str = "",
) -> dict[str, Any]:
    """Build a MODEL_INFO.json entry dict for one model.

    Optional fields (``thinking``, ``comment``) are only included when set
    so the on-disk JSON stays compact and reviewable; required fields
    (``context_length``, prices, ``fc``/``emb``/``gen``) are always present.

    Args:
        ctx: Maximum context length in tokens.
        inp: Input price per 1M tokens (USD).
        out: Output price per 1M tokens (USD).
        fc: Whether the model supports function calling.
        emb: Whether the model is an embedding model.
        gen: Whether the model supports text generation.
        thinking: Highest ``reasoning_effort`` level the model accepts.
        comment: Free-form annotation (e.g. ``"NEW"`` /
            ``"NEW: needs pricing"``). Omitted when empty.

    Returns:
        A dict suitable for serialization to MODEL_INFO.json.
    """
    entry: dict[str, Any] = {
        "context_length": ctx,
        "input_price_per_1M": inp,
        "output_price_per_1M": out,
        "fc": fc,
        "emb": emb,
        "gen": gen,
    }
    if thinking:  # pragma: no branch
        entry["thinking"] = thinking
    if comment:  # pragma: no branch
        entry["comment"] = comment
    return entry


def _read_model_info_json(path: Path) -> dict[str, dict]:
    """Read MODEL_INFO.json into a name → entry-dict mapping.

    Returns an empty dict when the file doesn't exist yet (lets the script
    bootstrap a brand new JSON file from scratch).
    """
    if not path.exists():  # pragma: no branch
        return {}
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _write_model_info_json(path: Path, data: dict[str, dict]) -> None:
    """Write ``data`` to ``path`` as sorted, pretty-printed JSON."""
    sorted_data = dict(sorted(data.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted_data, indent=2) + "\n")


def apply_updates_to_file(
    updates: list[dict],
    new_models: list[dict],
    deprecated: list[dict],
    current: dict[str, dict],
    dry_run: bool = False,
) -> None:
    """Apply MODEL_INFO updates/additions/removals to the JSON source of truth.

    Mutates ``MODEL_INFO_PATH`` (the in-repo ``MODEL_INFO.json``) and, when
    it already exists, also syncs the user-local copy at
    ``~/.kiss/MODEL_INFO.json`` so that a running KISS install picks up the
    changes on next ``model_info`` reload without waiting for a reinstall.

    Args:
        updates: ``[{"name": str, "changes": {field: value, ...}}]``.
            ``changes`` may target ``context_length``, ``input_price_per_1M``,
            ``output_price_per_1M``, ``fc``, ``emb``, ``gen``, ``thinking``.
            A ``thinking`` value of ``None`` removes the field.
        new_models: Each entry must carry at minimum ``name``,
            ``context_length``, ``input_price_per_1M``, ``output_price_per_1M``.
            Optional flags: ``fc`` (default True), ``emb`` (False),
            ``gen`` (True), ``thinking``, ``needs_pricing``.
        deprecated: ``[{"name": str, "reason": str}]``; removed by name.
        current: Snapshot of the pre-update ``MODEL_INFO`` (used to
            preserve unchanged fields when applying updates to models that
            don't yet have a JSON entry).
        dry_run: When True, print what would change and return without
            touching disk.
    """
    data = _read_model_info_json(MODEL_INFO_PATH)

    deprecated_names = {d["name"] for d in deprecated}
    removed = sum(1 for name in deprecated_names if data.pop(name, None) is not None)

    applied = 0
    for upd in updates:  # pragma: no branch
        name = upd["name"]
        cur = current.get(name, {})
        entry = data.get(name) or _build_entry(
            ctx=cur.get("context_length", 0),
            inp=cur.get("input_price_per_1M", 0.0),
            out=cur.get("output_price_per_1M", 0.0),
            fc=cur.get("fc", True),
            emb=cur.get("emb", False),
            gen=cur.get("gen", True),
            thinking=cur.get("thinking"),
        )
        for field, value in upd["changes"].items():  # pragma: no branch
            if field == "thinking" and value is None:  # pragma: no branch
                entry.pop("thinking", None)
            else:
                entry[field] = value
        data[name] = entry
        applied += 1

    added = 0
    for nm in new_models:  # pragma: no branch
        comment = "NEW: needs pricing" if nm.get("needs_pricing") else "NEW"
        data[nm["name"]] = _build_entry(
            ctx=nm["context_length"],
            inp=nm["input_price_per_1M"],
            out=nm["output_price_per_1M"],
            fc=nm.get("fc", True),
            emb=nm.get("emb", False),
            gen=nm.get("gen", True),
            thinking=nm.get("thinking"),
            comment=comment,
        )
        added += 1

    print(f"\n  Removed {removed} deprecated, applied {applied} updates, added {added} new")
    if dry_run:
        print("  (dry-run, no files modified)")
        return
    _write_model_info_json(MODEL_INFO_PATH, data)
    print(f"  Written to {MODEL_INFO_PATH}")
    if USER_MODEL_INFO_PATH.exists():  # pragma: no branch
        _write_model_info_json(USER_MODEL_INFO_PATH, data)
        print(f"  Also synced to {USER_MODEL_INFO_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update MODEL_INFO.json from vendor APIs")
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

    print("\n[1/6] Loading current MODEL_INFO...")
    current = get_current_model_info()
    print(f"  {len(current)} models loaded")

    print("\n[2/6] Fetching from vendor APIs...")
    openrouter_models = fetch_openrouter(verbose=args.verbose)
    together_models = fetch_together(verbose=args.verbose)
    gemini_models = fetch_gemini(verbose=args.verbose)
    anthropic_models = fetch_anthropic(verbose=args.verbose)
    openai_models = fetch_openai(verbose=args.verbose)
    codex_slugs = fetch_codex_supported_slugs(verbose=args.verbose)

    print("\n[3/6] Detecting deprecated models...")
    deprecated = find_deprecated_models(
        current,
        openrouter_models,
        anthropic_models,
        gemini_models,
        openai_models,
        codex_slugs=codex_slugs,
    )
    if deprecated:  # pragma: no branch
        print(f"\n  Deprecated models in MODEL_INFO ({len(deprecated)}):")
        for dep in deprecated:  # pragma: no branch
            print(f"    {dep['name']} ({dep['reason']})")
    else:
        print("  No deprecated models found")

    print("\n[4/6] Computing changes...")
    updates, new_models = compute_changes(
        current,
        openrouter_models,
        together_models,
        gemini_models,
        anthropic_models,
        openai_models,
        codex_slugs=codex_slugs,
    )

    if updates:  # pragma: no branch
        print(f"\n  Pricing/context updates ({len(updates)}):")
        for upd in updates:  # pragma: no branch
            changes_str = ", ".join(
                f"{k}: {current[upd['name']].get(k, '?')} -> {v}" for k, v in upd["changes"].items()
            )
            print(f"    {upd['name']}: {changes_str}")
    else:
        print("\n  No pricing/context updates needed")

    if new_models:  # pragma: no branch
        print(f"\n  New models discovered ({len(new_models)}):")
        for nm in new_models[:50]:  # pragma: no branch
            pricing = ""
            if not nm.get("needs_pricing"):  # pragma: no branch
                pricing = f" ${nm['input_price_per_1M']}/{nm['output_price_per_1M']}"
            print(f"    {nm['name']} (ctx={nm['context_length']}{pricing}) [{nm['source']}]")
        if len(new_models) > 50:  # pragma: no branch
            print(f"    ... and {len(new_models) - 50} more")
    else:
        print("\n  No new models discovered")

    deprecated_names = {d["name"] for d in deprecated}
    new_models = [nm for nm in new_models if nm["name"] not in deprecated_names]

    if not updates and not new_models and not deprecated:  # pragma: no branch
        print("\nEverything is up to date!")
        return

    if new_models and not args.skip_test:  # pragma: no branch
        print(f"\n[5/6] Testing {len(new_models)} new models...")
        for nm in new_models:  # pragma: no branch
            if nm["name"].startswith("codex/"):  # pragma: no branch
                continue
            caps = test_model_capabilities(nm["name"], verbose=args.verbose)
            nm["gen"] = caps["gen"]
            nm["emb"] = caps["emb"]
            nm["fc"] = caps["fc"]
            nm["thinking"] = caps["thinking"]
            if not caps["gen"] and not caps["emb"]:  # pragma: no branch
                nm["_skip"] = True
        new_models = [nm for nm in new_models if not nm.get("_skip")]
        print(f"  {len(new_models)} models passed testing")
    elif new_models and args.skip_test:  # pragma: no branch
        print("\n[5/6] Skipping model testing (--skip-test)")
        for nm in new_models:  # pragma: no branch
            nm["fc"] = True
            nm["gen"] = not nm.get("is_embedding", False)
            nm["emb"] = nm.get("is_embedding", False)
            nm["thinking"] = None
    else:
        print("\n[5/6] No new models to test")

    if args.test_existing:  # pragma: no branch
        print("\n  Re-testing existing models...")
        update_by_name = {upd["name"]: upd for upd in updates}
        for name, cur in current.items():  # pragma: no branch
            caps = test_model_capabilities(name, verbose=args.verbose)
            fc_changed = caps["fc"] != cur["fc"]
            thinking_changed = caps["thinking"] != cur.get("thinking")
            if not (fc_changed or thinking_changed):  # pragma: no branch
                continue
            existing = update_by_name.get(name)
            if existing is None:  # pragma: no branch
                existing = {"name": name, "changes": {}, "source": "retest"}
                updates.append(existing)
                update_by_name[name] = existing
            if fc_changed:  # pragma: no branch
                existing["changes"]["fc"] = caps["fc"]
                print(f"    {name}: fc changed {cur['fc']} -> {caps['fc']}")
            if thinking_changed:  # pragma: no branch
                existing["changes"]["thinking"] = caps["thinking"]
                print(
                    f"    {name}: thinking changed "
                    f"{cur.get('thinking')!r} -> {caps['thinking']!r}"
                )

    print("\n[6/6] Applying changes...")
    apply_updates_to_file(updates, new_models, deprecated, current, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
