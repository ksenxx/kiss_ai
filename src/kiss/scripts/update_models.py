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
import os
import re
import ssl
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

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
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
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
    skip_suffixes = (
        "-exp", "-latest", "-preview-tts", "-image-generation",
        "-image-preview", "-customtools", "-native-audio",
        "-computer-use", "-robotics",
    )
    models: dict[str, dict] = {}
    for m in data.get("models", []):
        raw_name = m.get("name", "")
        model_id = raw_name.replace("models/", "")
        if not model_id.startswith("gemini-"):
            continue
        if any(s in model_id for s in skip_suffixes):
            continue
        if model_id.endswith("-latest"):
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
        return False


def test_embedding(model_name: str) -> bool:
    from kiss.core.models.model_info import model as create_model

    try:
        m = create_model(model_name)
        m.initialize("")
        vec = m.get_embedding("Hello world")
        return isinstance(vec, list) and len(vec) > 0
    except Exception:
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
            return "error"

    try:
        m = create_model(model_name)
        m.initialize("What is 2+3? Use the calculator tool.")
        calls, _, _ = m.generate_and_process_with_tools({"calculator": calculator})
        return len(calls) > 0
    except Exception:
        return False


def test_model_capabilities(
    model_name: str, verbose: bool = False,
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
) -> list[dict]:
    """Identify models in current MODEL_INFO that are deprecated upstream.

    A model is considered deprecated if:
    - It's an openrouter/ model not present in the fetched OpenRouter list
      (which already filters out expired models).
    - It's a claude- model not returned by the Anthropic models API and not an
      alias (aliases don't have date suffixes and resolve to snapshot versions).
    - It's a gemini- model not returned by the Gemini models API.
    """
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

    return deprecated


def compute_changes(
    current: dict[str, dict],
    openrouter: dict[str, dict],
    together: dict[str, dict],
    gemini: dict[str, dict],
    anthropic: dict[str, dict],
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
            if fetched["context_length"] and fetched["input_price_per_1M"] > 0:
                new_models.append({
                    "name": name,
                    "context_length": fetched["context_length"],
                    "input_price_per_1M": fetched["input_price_per_1M"],
                    "output_price_per_1M": fetched["output_price_per_1M"],
                    "source": "openrouter",
                })

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
            if (
                fetched["context_length"]
                and fetched.get("type") in ("chat", "embedding")
                and fetched["input_price_per_1M"] > 0
            ):
                new_models.append({
                    "name": name,
                    "context_length": fetched["context_length"],
                    "input_price_per_1M": fetched["input_price_per_1M"],
                    "output_price_per_1M": fetched["output_price_per_1M"],
                    "source": "together",
                    "is_embedding": fetched.get("is_embedding", False),
                })

    # --- Gemini models (context length updates only, no pricing in API) ---
    for name, fetched in gemini.items():
        if name in current:
            cur = current[name]
            if fetched["context_length"] and fetched["context_length"] != cur["context_length"]:
                updates.append({
                    "name": name,
                    "changes": {"context_length": fetched["context_length"]},
                    "source": "gemini",
                })
        else:
            new_models.append({
                "name": name,
                "context_length": fetched["context_length"],
                "input_price_per_1M": 0.0,
                "output_price_per_1M": 0.0,
                "source": "gemini",
                "needs_pricing": True,
            })

    # --- Anthropic models (new model detection only) ---
    for name in anthropic:
        if name not in current:
            new_models.append({
                "name": name,
                "context_length": 200000,
                "input_price_per_1M": 0.0,
                "output_price_per_1M": 0.0,
                "source": "anthropic",
                "needs_pricing": True,
            })

    return updates, new_models


# ---------------------------------------------------------------------------
# File update
# ---------------------------------------------------------------------------

def _make_entry_line(
    name: str, ctx: int, inp: float, out: float,
    fc: bool = True, emb: bool = False, gen: bool = True,
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
    current: dict[str, dict],
    dry_run: bool = False,
) -> None:
    content = MODEL_INFO_PATH.read_text()
    lines = content.split("\n")

    applied_updates = 0
    for upd in updates:
        name = upd["name"]
        cur = current[name]
        new_ctx = upd["changes"].get("context_length", cur["context_length"])
        new_inp = upd["changes"].get("input_price_per_1M", cur["input_price_per_1M"])
        new_out = upd["changes"].get("output_price_per_1M", cur["output_price_per_1M"])
        new_line = _make_entry_line(
            name, new_ctx, new_inp, new_out,
            fc=cur["fc"], emb=cur["emb"], gen=cur["gen"],
        )
        pattern = re.compile(rf'^\s+"{re.escape(name)}"\s*:')
        for i, line in enumerate(lines):
            if pattern.match(line):
                old_comment = ""
                if "#" in line:
                    old_comment = line[line.index("#") + 1 :].strip()
                if old_comment and len(new_line) + len(old_comment) + 4 <= 100:
                    new_line += f"  # {old_comment}"
                lines[i] = new_line
                applied_updates += 1
                break

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
            name, nm["context_length"],
            nm["input_price_per_1M"], nm["output_price_per_1M"],
            fc=nm.get("fc", True), emb=nm.get("emb", False),
            gen=nm.get("gen", True), comment=comment,
        )
        new_lines_to_add.append(entry_line)
        added += 1

    if new_lines_to_add and insert_before_closing >= 0:
        header = [
            "    # ==========================================================================",
            "    # Auto-discovered models (verify pricing and capabilities)",
            "    # ==========================================================================",
        ]
        for line in reversed(header + new_lines_to_add):
            lines.insert(insert_before_closing, line)

    print(f"\n  Applied {applied_updates} updates, added {added} new models")
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
        "--dry-run", action="store_true", help="Don't modify files",
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

    # 3. Detect deprecated models
    print("\n[3/6] Detecting deprecated models...")
    deprecated = find_deprecated_models(
        current, openrouter_models, anthropic_models, gemini_models,
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
        current, openrouter_models, together_models, gemini_models, anthropic_models,
    )

    # Print summary
    if updates:
        print(f"\n  Pricing/context updates ({len(updates)}):")
        for upd in updates:
            changes_str = ", ".join(
                f"{k}: {current[upd['name']].get(k, '?')} -> {v}"
                for k, v in upd["changes"].items()
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

    if not updates and not new_models:
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
    apply_updates_to_file(updates, new_models, current, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
