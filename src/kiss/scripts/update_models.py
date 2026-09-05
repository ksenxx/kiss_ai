#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fetch latest model pricing/context from vendor APIs, test new models,
and update MODEL_INFO.json.

Capability testing covers generation, embeddings, function calling, the
highest accepted ``reasoning_effort`` level, and OpenAI v2 (Responses API)
support: every tested model is live-probed through ``/v1/responses`` (see
``update_responses_api_support.probe_responses_support``) and gets
``"use_responses_api": true`` only when the probe passes, which makes the
``model()`` factory build it on the v2 transport.

By default the script writes the source-of-truth
``src/kiss/core/models/MODEL_INFO.json`` in the repo.  The ``--model-info``
option points it at a different catalog file instead — most importantly the
user-local ``~/.kiss/MODEL_INFO.json``, which the installer seeds from the
bundled catalog and which an installed KISS Sorcar reads at runtime (see
``kiss.core.models.model_info``); the settings panel's "Update Models"
button runs this script against that copy.  When a non-default target is
updated, the repo's ``README.md`` catalog totals are left untouched.  The
write is atomic (temp file + ``os.replace``) because ``model_info`` loads
the catalog at import time, so a truncating rewrite would break every
process that starts while the script is running.

Usage:
    uv run python scripts/update_models.py [OPTIONS]

Options:
    --model-info PATH  Location of the MODEL_INFO.json catalog to update
                       (default: the repo's bundled
                       src/kiss/core/models/MODEL_INFO.json)
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
import tempfile
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
    workdir = os.environ.get("KISS_WORKDIR", "")
    if workdir:
        p = Path(workdir)
        if (p / _EXPECTED_SUBPATH).exists():
            return p

    cwd = Path.cwd()
    if (cwd / ".git").exists() and (cwd / _EXPECTED_SUBPATH).exists():
        return cwd

    return Path(__file__).resolve().parent.parent.parent.parent


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_MODEL_INFO_PATH = (
    PROJECT_ROOT / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
)
MODEL_INFO_PATH = DEFAULT_MODEL_INFO_PATH
README_PATH = PROJECT_ROOT / "README.md"


def _writes_default_catalog() -> bool:
    """Return True when the update targets the repo's bundled catalog.

    The README's "Models Supported" totals describe the bundled
    ``src/kiss/core/models/MODEL_INFO.json`` only, so a run pointed at a
    different catalog via ``--model-info`` (e.g. the user-local
    ``~/.kiss/MODEL_INFO.json``) must not rewrite them.  Both sides are
    resolved so a symlinked spelling of the default still counts as the
    default.
    """
    try:
        return MODEL_INFO_PATH.resolve() == DEFAULT_MODEL_INFO_PATH.resolve()
    except OSError:
        logger.debug("Exception caught", exc_info=True)
        return MODEL_INFO_PATH == DEFAULT_MODEL_INFO_PATH

_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "minimax-",
    "MiniMaxAI/",
    "openrouter/minimax/",
)

_SSL_CTX = ssl.create_default_context()

_CONTEXT_CAP_THRESHOLD = 1_000_000
_CAPPED_CONTEXT_LENGTH = 500_000


def _cap_context_length(ctx: int) -> int:
    """Return ``ctx`` capped at 500000 when it is 1000000 or above.

    Applied to every context length fetched from vendor APIs and to every
    entry written to ``MODEL_INFO.json`` so the catalog never advertises a
    context window of one million tokens or more.
    """
    if ctx >= _CONTEXT_CAP_THRESHOLD:
        return _CAPPED_CONTEXT_LENGTH
    return ctx


def _is_excluded_provider(name: str) -> bool:
    """Return True if ``name`` belongs to a permanently excluded provider.

    Used to drop entries from upstream fetches and to mark any leftover
    catalog entries as deprecated so they are removed from
    ``MODEL_INFO.json`` on the next run.
    """
    return name.startswith(_EXCLUDED_PREFIXES)


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
        ctx = _cap_context_length(m.get("context_length", 0) or 0)
        name = f"openrouter/{model_id}"
        if _is_excluded_provider(name):  # pragma: no branch
            continue
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
        ctx = _cap_context_length(m.get("context_length", 0) or 0)
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
        ctx = _cap_context_length(m.get("inputTokenLimit", 0) or 0)
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
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
    if workspace_id:  # pragma: no branch
        # Required when the key is identity-linked; harmless otherwise.
        headers["anthropic-workspace-id"] = workspace_id
    data = api_get("https://api.anthropic.com/v1/models", headers=headers)
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

    Audio-capable chat models (e.g. ``gpt-audio``, ``gpt-audio-mini``,
    ``gpt-4o-audio-preview``) are intentionally *not* filtered out: they
    accept plain text requests on ``/v1/chat/completions`` (audio is an
    optional input/output modality), so they work with KISS's
    ``OpenAICompatibleModel`` and pass the capability probes. Models that
    require entirely different endpoints (``/v1/realtime``,
    ``/v1/audio/transcriptions``, ``/v1/audio/speech``) remain excluded.
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
            "alias_of": info.alias_of,
            "use_responses_api": info.use_responses_api,
        }
        for name, info in MODEL_INFO.items()
    }


def _noop_token_callback(_token: str) -> None:
    """No-op token callback used during capability probes.

    Some vendor models (e.g. ``Qwen/Qwen3.7-Max`` on Together AI) reject
    Chat Completions requests without ``stream=true``. KISS's
    ``OpenAICompatibleModel._stream_text`` only switches on streaming when
    a ``token_callback`` is registered, so probes must register one even
    if they don't care about the streamed deltas — otherwise the request
    is sent non-streaming and the vendor returns HTTP 400.
    """


def _tiny_wav_bytes() -> bytes:
    """Return a minimal valid WAV file (0.1s of 16-bit mono silence at 8kHz).

    Used as probe input for audio-capable chat models, which reject
    text-only requests. Built in-process so the script has no test-asset
    dependency.
    """
    import struct

    sample_rate = 8000
    data = b"\x00\x00" * (sample_rate // 10)
    fmt_chunk = b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16
    )
    data_chunk = b"data" + struct.pack("<I", len(data)) + data
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk


def _probe_attachments(model_name: str) -> list[Any] | None:
    """Return the attachments needed to capability-probe ``model_name``.

    Audio-capable chat models (``gpt-audio``, ``gpt-audio-mini``,
    ``gpt-4o-audio-preview``, and their dated snapshots / OpenRouter
    passthroughs) reject text-only requests on ``/v1/chat/completions``
    with HTTP 400: "This model requires that either input content or
    output modality contain audio." Probes for those models must therefore
    include an audio input part; this returns a single tiny silent WAV
    attachment for them and ``None`` for every other model.
    """
    base = model_name.rsplit("/", 1)[-1]
    if "audio" not in base:
        return None
    from kiss.core.models.model import Attachment

    return [Attachment(data=_tiny_wav_bytes(), mime_type="audio/wav")]


def test_generate(model_name: str) -> bool:
    from kiss.core.models.model_info import model as create_model

    try:
        m = create_model(model_name, token_callback=_noop_token_callback)
        m.initialize("Say hello in one word.", attachments=_probe_attachments(model_name))
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


_THINKING_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh")
"""The OpenAI-family ``reasoning_effort`` scale, in ascending order of effort.

This is the default scale for every vendor without an entry in
:func:`_thinking_scale_for`."""

_MOONSHOT_THINKING_LEVELS: tuple[str, ...] = ("low", "high", "max")
"""The Moonshot/Kimi ``reasoning_effort`` scale, in ascending order.

Kimi K3 accepts ``low`` / ``high`` / ``max`` (vendor default ``max``);
``medium`` and ``xhigh`` are rejected with HTTP 400, and thinking cannot
be disabled. Source:
https://platform.kimi.ai/docs/guide/use-reasoning-effort."""

_ALL_THINKING_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
"""Union of every vendor scale, used when sweeping generated ``-{level}``
aliases regardless of which scale produced them."""

_MOONSHOT_MODEL_PREFIXES: tuple[str, ...] = (
    "kimi-",
    "moonshot-",
    "moonshotai/",
    "openrouter/moonshotai/",
    "openrouter/~moonshotai/",
)
"""Catalog-key prefixes of Moonshot/Kimi models across every routing path:
direct (``kimi-*`` / ``moonshot-*``), Together (``moonshotai/*``), and
OpenRouter (``openrouter/moonshotai/*``)."""

_GROK_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high")
"""The xAI Grok ``reasoning_effort`` scale for the ``grok-4.3`` / ``grok-4.5``
family. xAI docs (https://docs.x.ai/developers/model-capabilities/text/reasoning)
document three levels; ``xhigh`` and ``max`` are rejected and thinking cannot
be disabled entirely for these models."""

_GROK_3_MINI_EFFORT_LEVELS: tuple[str, ...] = ("low", "high")
"""The xAI ``grok-3-mini`` / ``grok-3-mini-beta`` ``reasoning_effort`` scale.
Unlike ``grok-4.5``/``grok-4.3``, the mini family accepts ONLY ``low`` and
``high`` — ``medium`` returns HTTP 400 (docs.x.ai). The alias-writer emits
``-low`` and ``-high`` siblings only; a bogus ``-medium`` alias would be a
correctness bug."""

_XAI_MODEL_PREFIXES: tuple[str, ...] = (
    "openrouter/x-ai/",
    "openrouter/~x-ai/",
)
"""Catalog-key prefixes for xAI Grok models. Only routed through OpenRouter
today; a direct-xAI namespace can be added here alongside if/when KISS adds
a native xAI backend."""

_GLM_5_2_EFFORT_LEVELS: tuple[str, ...] = ("high", "max")
"""The z-ai ``GLM-5.2`` ``reasoning_effort`` scale. Per Zhipu docs the native
levels are ``high`` and ``max``; OpenRouter's Zhipu routes accept both plus
``xhigh`` as an alias for ``max``, so the source-of-truth 2-level ladder is
preserved and ``xhigh`` is not materialized. Every other GLM (4.5 / 4.6 /
4.7 / 5.0 / 5.1 / 5v-turbo) uses the older ``thinking.type`` boolean surface
and is deliberately kept behind the probe gate."""

_GLM_5_2_MODEL_PREFIXES: tuple[str, ...] = (
    "zai-org/",
    "openrouter/z-ai/",
    "openrouter/~z-ai/",
)
"""Catalog-key prefixes under which a ``GLM-5.2`` model may appear. The
narrower :func:`_is_glm_5_2_family` predicate refines the check to the
5.2 base name; other GLMs sharing these prefixes stay gated out."""


def _is_grok_effort_family(model_name: str) -> bool:
    """Return True for xAI Grok models that accept ``reasoning_effort``.

    Admits only the three ``reasoning_effort``-capable Grok families:
    ``grok-4.5``, ``grok-4.3`` (three-level ladder), and ``grok-3-mini`` /
    ``grok-3-mini-beta`` (two-level ladder). Every other Grok slug —
    including the boolean-thinking ``grok-4`` / ``grok-4-fast`` /
    ``grok-4.20`` / ``grok-4.20-multi-agent`` / ``grok-4.1-fast`` and every
    non-reasoning legacy Grok (``grok-3``, ``grok-2*``, ``grok-code-*``,
    ``grok-build-*``, ``grok-beta``, ``grok-vision-*``, ``grok-latest``) —
    stays behind the probe gate: those models either don't accept the
    parameter at all or repurpose it for agent count rather than depth.

    The match is on the last ``/``-separated segment, case-insensitively,
    so it covers both ``openrouter/x-ai/grok-4.5`` and any future direct
    xAI routing.

    Args:
        model_name: The catalog key of the model.

    Returns:
        True when the model belongs to the Grok reasoning-effort family.
    """
    base = model_name.rsplit("/", 1)[-1].lower()
    if base in ("grok-3-mini", "grok-3-mini-beta"):
        return True
    return base in ("grok-4.5", "grok-4.3")


def _is_grok_3_mini_family(model_name: str) -> bool:
    """Return True when ``model_name`` is a ``grok-3-mini`` variant.

    ``grok-3-mini`` and ``grok-3-mini-beta`` accept only ``low`` and
    ``high`` (no ``medium``), unlike ``grok-4.5``/``grok-4.3`` which
    accept ``low``/``medium``/``high``. Distinguishing them is critical
    to :func:`_thinking_scale_for`: emitting a ``-medium`` alias for a
    ``grok-3-mini`` model would fabricate a level the API rejects.
    """
    base = model_name.rsplit("/", 1)[-1].lower()
    return base in ("grok-3-mini", "grok-3-mini-beta")


def _is_glm_5_2_family(model_name: str) -> bool:
    """Return True when ``model_name`` is exactly a z-ai GLM-5.2 model.

    Per the November 2026 audit, ``zai-org/GLM-5.2`` and
    ``openrouter/z-ai/glm-5.2`` are the **only** Zhipu / GLM entries that
    accept the ``reasoning_effort`` API surface (native values ``high`` and
    ``max``). Every other GLM (4.5 / 4.6 / 4.7 / 5.0 / 5.1 / 5v-turbo /
    ``glm-4.5-air`` etc.) uses the older ``thinking.type`` boolean and
    must stay behind the probe gate.

    The match requires:

    * a recognized GLM-5.2 route prefix (``zai-org/``, ``openrouter/z-ai/``,
      ``openrouter/~z-ai/``), AND
    * a last-segment name equal to ``glm-5.2`` (case-insensitive) — so a
      hypothetical ``glm-5.2v-turbo`` or ``glm-5.2-air`` stays gated out
      unless it is separately verified to accept ``reasoning_effort``.

    Args:
        model_name: The catalog key of the model.

    Returns:
        True when the model is exactly a GLM-5.2 base.
    """
    if not model_name.startswith(_GLM_5_2_MODEL_PREFIXES):
        return False
    base = model_name.rsplit("/", 1)[-1].lower()
    return base == "glm-5.2"


def _is_together_gpt_oss(model_name: str) -> bool:
    """Return True for the Together-route OpenAI gpt-oss family.

    Together AI hosts the open-weight gpt-oss models under the ``openai/``
    catalog namespace (``openai/gpt-oss-120b``, ``openai/gpt-oss-20b``).
    That prefix does not overlap ``_OPENAI_PREFIXES`` (which starts with
    ``gpt``, ``o1``, ``o3``, ``o4``, ``codex``, ``computer-use``), so
    these entries would otherwise slip through the probe gate. They are
    OpenAI-compatible chat-completions models and accept the OpenAI
    ``reasoning_effort`` ladder (topping at ``high``); the standard
    OpenAI scale returned by :func:`_thinking_scale_for` is correct for
    them — the probe naturally lands at ``high`` and stops.

    The check is scoped to ``openai/gpt-oss-*`` slugs specifically so
    that no unrelated Together model is affected.
    """
    return model_name.startswith("openai/gpt-oss-")


def _is_kimi_k3_family(model_name: str) -> bool:
    """Return True when ``model_name`` is a Kimi K3-generation model.

    ``reasoning_effort`` is a K3-introduced API surface: Moonshot's docs
    define it (``low``/``high``/``max``) for ``kimi-k3*`` only, while
    K2.x models control thinking via the separate ``thinking.type``
    request field and ``moonshot-v1-*`` models have no thinking at all.
    The probe gate in :func:`detect_thinking_level` therefore admits only
    the K3 family — probing older Moonshot models through a gateway that
    silently drops unknown parameters (e.g. a passthrough that ignores
    ``reasoning_effort``) would otherwise fabricate alias levels the
    model does not honor.

    The check is on the last ``/``-separated segment, case-insensitively,
    so it covers ``kimi-k3``, ``moonshotai/Kimi-K3`` (Together) and
    ``openrouter/moonshotai/kimi-k3`` alike.

    Args:
        model_name: The catalog key of the model.

    Returns:
        True when the model belongs to the Kimi K3 family.
    """
    base = model_name.rsplit("/", 1)[-1].lower()
    return base.startswith("kimi-k3")


def _thinking_scale_for(model_name: str) -> tuple[str, ...]:
    """Return the ascending ``reasoning_effort`` scale for ``model_name``.

    The scale is vendor-specific:

    * Moonshot/Kimi models use :data:`_MOONSHOT_THINKING_LEVELS`
      (``low``/``high``/``max``).
    * xAI ``grok-3-mini`` / ``grok-3-mini-beta`` use
      :data:`_GROK_3_MINI_EFFORT_LEVELS` (``low``/``high``).
    * xAI ``grok-4.5`` / ``grok-4.3`` use :data:`_GROK_EFFORT_LEVELS`
      (``low``/``medium``/``high``).
    * z-ai ``GLM-5.2`` uses :data:`_GLM_5_2_EFFORT_LEVELS`
      (``high``/``max``).
    * Every other model uses the OpenAI ladder :data:`_THINKING_LEVELS`
      (``low``/``medium``/``high``/``xhigh``) — including Together's
      OpenAI-compatible ``openai/gpt-oss-*`` entries, which naturally top
      at ``high``.

    Every scale is required to contain ``"high"`` — the level stored on
    base entries when the detected maximum is higher (see
    :func:`_write_entry_with_thinking_split`).

    Args:
        model_name: The catalog key of the model.

    Returns:
        The ordered tuple of levels the vendor accepts.
    """
    if model_name.startswith(_MOONSHOT_MODEL_PREFIXES):
        return _MOONSHOT_THINKING_LEVELS
    if model_name.startswith(_XAI_MODEL_PREFIXES) and _is_grok_effort_family(model_name):
        if _is_grok_3_mini_family(model_name):
            return _GROK_3_MINI_EFFORT_LEVELS
        return _GROK_EFFORT_LEVELS
    if _is_glm_5_2_family(model_name):
        return _GLM_5_2_EFFORT_LEVELS
    return _THINKING_LEVELS


def detect_thinking_level(model_name: str) -> str | None:
    """Detect the highest ``reasoning_effort`` level the model accepts.

    Probes each level of the model's vendor scale (see
    :func:`_thinking_scale_for`) in descending order — ``xhigh``,
    ``high``, ``medium``, ``low`` for the OpenAI family; ``max``,
    ``high``, ``low`` for Moonshot/Kimi — by issuing a minimal generate
    call with ``model_config={"reasoning_effort": <level>}`` explicitly so
    that the vendor's Chat Completions API itself decides the verdict,
    regardless of whether the model is already flagged in ``MODEL_INFO``.
    Returns the first (highest) level that succeeds, or ``None`` if none
    did. Levels below the returned one are assumed supported too — the
    probed vendors accept every level of their scale once they accept any.

    Returns ``None`` (without making any API call) for backends that don't
    accept ``reasoning_effort`` at all:

    * ``codex/*`` — routed through the Codex CLI, which controls reasoning
      via its own ``model_reasoning_effort`` config rather than per-call.
    * ``cc/*`` — routed through the Claude Code CLI, which has no
      ``reasoning_effort`` surface at all.
    * ``claude-*``, ``gemini-*`` — non-OpenAI providers that don't accept
      ``reasoning_effort``.
    * Variants known to reject ``reasoning_effort`` entirely (``-pro``,
      ``-chat-latest``, ``-image``).
    * Moonshot models outside the Kimi K3 family (K2.x controls thinking
      via ``thinking.type``, ``moonshot-v1-*`` has none; see
      :func:`_is_kimi_k3_family`).
    * xAI Grok models outside the reasoning-effort family: only
      ``grok-4.5``, ``grok-4.3``, ``grok-3-mini``, and ``grok-3-mini-beta``
      are probed (see :func:`_is_grok_effort_family`). The boolean-only
      variants (``grok-4``, ``grok-4-fast``, ``grok-4.20*``, ``grok-4.1*``,
      etc.) either don't accept ``reasoning_effort`` or repurpose the
      levels for agent count rather than depth.
    * GLMs other than exactly ``GLM-5.2`` (see :func:`_is_glm_5_2_family`).
    * Every other vendor not yet verified to support the parameter (only
      the OpenAI family — direct and via OpenRouter, plus Together's
      ``openai/gpt-oss-*`` — Kimi K3, the xAI Grok effort family, and
      z-ai GLM-5.2 are probed).
    """
    from kiss.core.models.model_info import _OPENAI_PREFIXES

    if model_name.startswith(("codex/", "cc/", "claude-", "gemini-")):
        return None
    if any(marker in model_name for marker in ("-pro", "chat-latest", "-image")):
        return None
    is_openai = model_name.startswith(_OPENAI_PREFIXES) and not model_name.startswith(
        "text-embedding"
    )
    is_openrouter_openai = model_name.startswith(("openrouter/openai/", "openrouter/~openai/"))
    is_together_gpt_oss = _is_together_gpt_oss(model_name)
    is_moonshot_k3 = model_name.startswith(_MOONSHOT_MODEL_PREFIXES) and _is_kimi_k3_family(
        model_name
    )
    is_grok_effort = model_name.startswith(_XAI_MODEL_PREFIXES) and _is_grok_effort_family(
        model_name
    )
    is_glm_5_2 = _is_glm_5_2_family(model_name)
    if not (
        is_openai
        or is_openrouter_openai
        or is_together_gpt_oss
        or is_moonshot_k3
        or is_grok_effort
        or is_glm_5_2
    ):
        return None

    from kiss.core.models.model_info import model as create_model

    for level in reversed(_thinking_scale_for(model_name)):
        try:
            m = create_model(
                model_name,
                model_config={"reasoning_effort": level},
                token_callback=_noop_token_callback,
            )
            m.initialize("Say hello in one word.", attachments=_probe_attachments(model_name))
            text, _ = m.generate()
            if text and text.strip():
                return level
        except Exception:
            logger.debug("Exception caught", exc_info=True)
            continue
    return None


_ALLOWED_ARITHMETIC_NODES = (
    "Expression",
    "BinOp",
    "UnaryOp",
    "Constant",
    "Add",
    "Sub",
    "Mult",
    "Div",
    "FloorDiv",
    "Mod",
    "USub",
    "UAdd",
)
# ``**`` is deliberately NOT allowed: a hostile expression like
# ``9**9**9**9`` would pin the CPU and exhaust memory before any
# whitelist of node types could help.


def safe_arithmetic(expression: str) -> str:
    """Evaluate a plain arithmetic expression without executing code.

    Capability probes hand this to live third-party models as a
    ``calculator`` tool, so the argument is **attacker-controlled**: a
    model can return ``__import__('os').system(...)`` as the expression.
    A bare ``eval`` would execute it with the developer's privileges.
    This evaluator therefore parses the expression with :mod:`ast` and
    accepts only numeric literals combined with ``+ - * / // %`` and
    unary sign — anything else (names, calls, attributes, subscripts,
    strings, and ``**``, which enables CPU/memory exhaustion) is rejected.

    Args:
        expression: A math expression string like ``'25*4'``.

    Returns:
        The numeric result as a string, or ``"error"`` when the
        expression is not plain arithmetic (a probe only needs the happy
        path; models that send garbage simply get an error result back).
    """
    import ast

    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if type(node).__name__ not in _ALLOWED_ARITHMETIC_NODES:
                return "error"
            if isinstance(node, ast.Constant) and not isinstance(
                node.value, (int, float)
            ):
                return "error"
        return str(eval(compile(tree, "<arithmetic>", "eval"), {"__builtins__": {}}))
    except Exception:
        logger.debug("Exception caught", exc_info=True)
        return "error"


def test_function_calling(model_name: str) -> bool:
    from kiss.core.models.model_info import model as create_model

    def calculator(expression: str = "") -> str:
        """Compute a math expression.

        Args:
            expression: A math expression string like '2+3'.
        """
        return safe_arithmetic(expression)

    try:
        m = create_model(model_name, token_callback=_noop_token_callback)
        m.initialize(
            "What is 2+3? Use the calculator tool.",
            attachments=_probe_attachments(model_name),
        )
        calls, _, _ = m.generate_and_process_with_tools({"calculator": calculator})
        return len(calls) > 0
    except Exception:
        logger.debug("Exception caught", exc_info=True)
        return False


def test_responses_api(model_name: str, fc: bool) -> bool | None:
    """Live-probe whether *model_name* works over the OpenAI v2 Responses API.

    Delegates to ``update_responses_api_support.probe_responses_support``
    (lazy import — that module imports helpers from this one): a plain
    streaming generation through ``/v1/responses``, plus a full tool
    round-trip when *fc* is true.  Models without an OpenAI-compatible
    endpoint (Anthropic/Gemini native, CLIs) report a definitive ``False``,
    keeping them on the Chat Completions v1 adapter.

    Args:
        model_name: The model name (with any routing prefix).
        fc: Whether the model supports function calling.

    Returns:
        True when every applicable probe passed, False when the endpoint
        definitively rejected the model, and ``None`` when nothing could
        be verified (missing vendor key, exhausted transient endpoint
        errors) — callers must not flip a stored flag on ``None``.
    """
    from kiss.scripts.update_responses_api_support import probe_responses_support

    result = probe_responses_support(model_name, fc=fc)
    if not result.conclusive:
        return None
    return result.supported


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

    if results["gen"]:  # pragma: no branch
        results["use_responses_api"] = test_responses_api(model_name, results["fc"])
        time.sleep(0.5)
    else:
        # The generation probe itself failed (endpoint outage, missing
        # key, deprecated model): nothing was verified either way, so the
        # verdict is inconclusive rather than a negative.
        results["use_responses_api"] = None

    if verbose:  # pragma: no branch
        flags = " ".join(
            f"{k}={v if isinstance(v, str) else ('Y' if v else 'N')}" for k, v in results.items()
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
    - It's a cc/ (Claude Code CLI) model whose ``claude-*`` slug is gone from
      the Anthropic models API, using the same dated-snapshot/alias rules as
      direct ``claude-*`` entries. The short aliases (``cc/haiku``,
      ``cc/opus``, ``cc/sonnet``) are always kept.
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
        if _is_excluded_provider(name):  # pragma: no branch
            deprecated.append({"name": name, "reason": "excluded provider"})
            continue
        if name.endswith(_XHIGH_SUFFIX) or current[name].get("alias_of"):
            continue
        if name.startswith("codex/"):  # pragma: no branch
            slug = name.removeprefix("codex/")
            if slug in SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS:
                deprecated.append({
                    "name": name,
                    "reason": "rejected by Codex CLI on ChatGPT subscriptions",
                })
            elif codex_slugs and name != "codex/default":
                if slug not in codex_slugs:
                    deprecated.append({"name": name, "reason": "not in Codex CLI models.json"})
            continue
        if name.startswith("cc/"):
            slug = name.removeprefix("cc/")
            if anthropic and slug.startswith("claude-") and slug not in anthropic:
                if re.search(r"\d{8}$", slug):
                    deprecated.append({"name": name, "reason": "not in Anthropic API"})
                else:
                    alias_re = re.compile(rf"^{re.escape(slug)}-\d{{8}}$")
                    if not any(alias_re.match(n) for n in anthropic):
                        deprecated.append(
                            {"name": name, "reason": "alias with no snapshot in Anthropic API"}
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
                    alias_re = re.compile(rf"^{re.escape(name)}-(\d{{8}}|\d{{4}}-\d{{2}}-\d{{2}})$")
                    if not any(alias_re.match(n) for n in openai):
                        deprecated.append(
                            {"name": name, "reason": "alias with no snapshot in OpenAI API"}
                        )

    return deprecated


_GPT_PRO_OR_CODEX_RE = re.compile(r"-(pro|codex)(-|$)")

_OPENAI_RESPONSES_ONLY_RE = re.compile(r"^o\d+(?:\.\d+)?-pro(?:-|$)")


def _is_excluded_openai_responses_only(name: str) -> bool:
    """Return True for OpenAI reasoning models that only work via ``/v1/responses``.

    Models in the ``o1-pro`` / ``o3-pro`` family (e.g. ``o1-pro``,
    ``o1-pro-2025-03-19``, ``o3-pro``, ``o3-pro-2025-06-10``) are routed
    exclusively through OpenAI's ``/v1/responses`` endpoint. KISS's
    ``OpenAICompatibleModel`` invokes ``client.chat.completions.create``
    (``/v1/chat/completions``), so probing or running these models there
    returns HTTP 404 — there is no way for the discovery flow to test
    them, and they would fail at runtime for KISS users anyway. Skipping
    them upfront avoids a wasted (and billable) probe each ``update_models``
    run and prevents broken entries from leaking into ``MODEL_INFO.json``.

    Matches both bare OpenAI names (``o1-pro``, ``o3-pro``), their dated
    snapshots (``o1-pro-2025-03-19``), and OpenRouter passthroughs
    (``openrouter/openai/o1-pro``). The match is scoped to the
    ``o<digits>-pro`` shape (last ``/``-separated segment) so unrelated
    names that happen to contain ``o-pro`` aren't caught.
    """
    base = name.rsplit("/", 1)[-1]
    return bool(_OPENAI_RESPONSES_ONLY_RE.match(base))


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


def _dot_version(name: str) -> str:
    """Convert a hyphenated minor version to dotted form for OpenRouter lookup.

    Anthropic model IDs separate major and minor versions with a hyphen
    (``claude-fable-5-1``, ``claude-opus-4-6``) while OpenRouter slugs use a
    dot (``claude-fable-5.1``, ``claude-opus-4.6``). Rewrites every
    ``<digits>-<digits>`` pair to ``<digits>.<digits>`` so
    ``claude-3-5-sonnet`` also maps to ``claude-3.5-sonnet``. Names without
    hyphen-separated version pairs are returned unchanged.
    """
    return re.sub(r"(\d)-(\d)", r"\1.\2", name)


_VENDOR_OR_PREFIX: dict[str, str] = {
    "openai": "openrouter/openai/",
    "anthropic": "openrouter/anthropic/",
    "gemini": "openrouter/google/",
}

_CODEX_MODELS_JSON_URL = (
    "https://raw.githubusercontent.com/openai/codex/main/codex-rs/models-manager/models.json"
)

SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS = frozenset({
    "gpt-5.2",
    "gpt-daybreak-blue-latest",
    "gpt-daybreak-red-latest",
})
"""Codex CLI slugs rejected under a ChatGPT subscription.

Verified live on 2026-09-02: the Codex CLI returns HTTP 400 ("not
supported when using Codex with a ChatGPT account") for these slugs
even though the upstream ``models.json`` lists them.  They must stay
out of ``MODEL_INFO.json``: ``_add_codex_candidates`` never adds them
and :func:`find_deprecated_models` flags existing entries.  The
``test_codex_model.py`` guard test imports this set, so there is a
single source of truth.
"""


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
    Slugs in :data:`SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS` are skipped even
    when the upstream ``models.json`` lists them: they were verified live to
    fail on ChatGPT subscriptions.

    Context length is taken from the matching OpenRouter entry when
    available, falling back to 400000 (the default for codex/* models).
    All entries get $0/0 pricing since Codex is billed via the user's
    ChatGPT subscription.
    """
    for slug in codex_slugs:  # pragma: no branch
        if slug in SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS:
            continue
        codex_name = f"codex/{slug}"
        if codex_name in current:  # pragma: no branch
            continue
        if _is_excluded_gpt_pro_or_codex(codex_name):  # pragma: no branch
            continue
        if _is_excluded_openai_responses_only(codex_name):  # pragma: no branch
            continue
        or_info = _lookup_openrouter_pricing(slug, "openai", openrouter)
        ctx = or_info["context_length"] if or_info and or_info.get("context_length") else 400000
        ctx = _cap_context_length(ctx)
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


_CLAUDE_CODE_ALIASES: tuple[str, ...] = ("haiku", "opus", "sonnet")
"""Short model aliases the Claude Code CLI resolves itself (``--model opus``
picks the newest Opus tier available to the subscription). They never appear
in the Anthropic ``/v1/models`` list, so they are seeded here and — like
``codex/default`` — never marked deprecated."""


def _add_claude_code_candidates(
    anthropic: dict[str, dict],
    current: dict[str, dict],
    openrouter: dict[str, dict],
    new_models: list[dict],
) -> None:
    """Add ``cc/<model>`` entries for models the Claude Code CLI supports.

    The Claude Code backend passes the part after ``cc/`` verbatim as the
    ``claude`` CLI's ``--model`` flag, which accepts any Anthropic model ID
    plus the short aliases in :data:`_CLAUDE_CODE_ALIASES`. Every
    ``claude-*`` model returned by the Anthropic models API therefore
    becomes a ``cc/claude-*`` candidate, alongside the always-present
    alias entries.

    Context length is taken from the matching OpenRouter entry when
    available, then from the direct ``claude-*`` catalog entry, falling
    back to 200000 (the Anthropic default). All entries get $0/0 pricing
    since Claude Code is billed via the user's Claude subscription.
    """
    for slug in list(_CLAUDE_CODE_ALIASES) + sorted(anthropic):
        cc_name = f"cc/{slug}"
        if cc_name in current:
            continue
        or_info = _lookup_openrouter_pricing(slug, "anthropic", openrouter)
        if or_info and or_info.get("context_length"):
            ctx = or_info["context_length"]
        elif current.get(slug, {}).get("context_length"):
            ctx = current[slug]["context_length"]
        else:
            ctx = 200000
        new_models.append(
            {
                "name": cc_name,
                "context_length": _cap_context_length(ctx),
                "input_price_per_1M": 0.0,
                "output_price_per_1M": 0.0,
                "source": "claude-code",
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
    then the base name with date suffixes stripped (e.g.
    ``gpt-5.4-2026-03-05`` → ``openrouter/openai/gpt-5.4``), and finally the
    base name with hyphenated minor versions rewritten to dotted form, since
    Anthropic IDs use hyphens where OpenRouter slugs use dots (e.g.
    ``claude-fable-5-1`` → ``openrouter/anthropic/claude-fable-5.1``).
    """
    prefix = _VENDOR_OR_PREFIX.get(source)
    if not prefix:  # pragma: no branch
        return None
    base = _strip_date_suffix(model_name)
    for candidate in dict.fromkeys((model_name, base, _dot_version(base))):
        or_key = f"{prefix}{candidate}"
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
            if _is_excluded_openai_responses_only(name):
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
            if _is_excluded_openai_responses_only(name):  # pragma: no branch
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

    if anthropic:
        _add_claude_code_candidates(anthropic, current, openrouter, new_models)

    from kiss.core.models.model_info import _OPENAI_PREFIXES

    update_by_name = {upd["name"]: upd for upd in updates}
    for name, cur in current.items():
        if name.startswith("openrouter/"):  # pragma: no branch
            continue
        if name.startswith(("cc/", "codex/")):
            # Subscription-billed CLI backends: pricing stays $0/0 forever,
            # even though their slugs have priced OpenRouter twins.
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
    use_responses_api: bool = False,
) -> dict[str, Any]:
    """Build a MODEL_INFO.json entry dict for one model.

    Optional fields (``thinking``, ``comment``, ``use_responses_api``) are
    only included when set so the on-disk JSON stays compact and
    reviewable; required fields (``context_length``, prices,
    ``fc``/``emb``/``gen``) are always present.

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
        use_responses_api: Whether the model passed the live OpenAI v2
            (``/v1/responses``) probe and should use that transport.

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
    if use_responses_api:
        entry["use_responses_api"] = True
    return entry


_XHIGH_SUFFIX = "-xhigh"


def _alias_base_name(name: str, entry: dict[str, Any]) -> str | None:
    """Return the base model name when ``entry`` is a generated thinking alias.

    A generated alias is recognized either by its explicit ``alias_of``
    marker (written by :func:`_write_entry_with_thinking_split` on every
    generated ``-{level}`` sibling) or — for catalogs written before the
    marker existed — by the legacy ``-xhigh`` name suffix. The marker is
    required for ``-low`` / ``-medium`` / ``-high`` because real upstream
    models can end in those suffixes (e.g. ``openrouter/openai/o3-mini-high``)
    and must never be mistaken for synthetic aliases.

    Args:
        name: The catalog key of the entry.
        entry: The entry dict (may or may not carry ``alias_of``).

    Returns:
        The base model name, or ``None`` when ``entry`` is a real model.
    """
    alias_of = entry.get("alias_of")
    if alias_of:
        return str(alias_of)
    if name.endswith(_XHIGH_SUFFIX):
        return name.removesuffix(_XHIGH_SUFFIX)
    return None


def _expected_alias_entry(base_name: str, base: dict[str, Any], level: str) -> dict[str, Any]:
    """Return the generated ``-{level}`` sibling expected for ``base``.

    The sibling mirrors every field of ``base`` (context length, pricing,
    ``fc`` / ``emb`` / ``gen``, ``comment``) except ``thinking``, which is
    pinned to ``level``, and ``alias_of``, which records the base name so
    runtime code and later runs can tell synthetic aliases apart from real
    upstream models.
    """
    sibling = dict(base)
    sibling["thinking"] = level
    sibling["alias_of"] = base_name
    return sibling


def _pop_generated_aliases(
    data: dict[str, dict],
    name: str,
    keep_levels: tuple[str, ...] = (),
) -> None:
    """Remove ``name``'s generated ``-{level}`` aliases not in ``keep_levels``.

    Only entries recognized as generated aliases of ``name`` (via
    :func:`_alias_base_name`) are removed; a real upstream model that
    happens to be called ``{name}-high`` (e.g. ``o3-mini-high``) is left
    untouched. The sweep covers :data:`_ALL_THINKING_LEVELS` (the union
    of every vendor scale) so aliases left behind by a scale change are
    cleaned up too.
    """
    for level in _ALL_THINKING_LEVELS:
        if level in keep_levels:
            continue
        sibling_name = f"{name}-{level}"
        sibling = data.get(sibling_name)
        if sibling is not None and _alias_base_name(sibling_name, sibling) == name:
            data.pop(sibling_name)


def _write_entry_with_thinking_split(
    data: dict[str, dict],
    name: str,
    entry: dict[str, Any],
    *,
    remove_stale_siblings: bool = True,
) -> None:
    """Write ``entry`` under ``name``, materializing one alias per thinking level.

    When ``entry["thinking"]`` is a level on the model's vendor scale
    (see :func:`_thinking_scale_for`), the catalog emits the base entry
    **plus one generated sibling per supported level** — every level from
    ``low`` up to the detected maximum:

    * OpenAI family, ``thinking="xhigh"`` → base (downgraded to
      ``thinking="high"``) and ``-low`` / ``-medium`` / ``-high`` /
      ``-xhigh`` siblings;
    * OpenAI family, ``thinking="high"`` → base and ``-low`` /
      ``-medium`` / ``-high``;
    * Moonshot/Kimi, ``thinking="max"`` → base (downgraded to
      ``thinking="high"``) and ``-low`` / ``-high`` / ``-max`` siblings;
    * lower levels analogously.

    Each sibling inherits every other field on ``entry`` (context length,
    pricing, ``fc`` / ``emb`` / ``gen`` flags, ``comment``) and carries an
    ``alias_of`` marker naming the base, so its pricing / capability
    signature matches the base byte for byte aside from ``thinking`` and
    ``alias_of``.

    When ``entry`` has no recognized ``thinking`` level this is a plain
    ``data[name] = entry`` write. When ``remove_stale_siblings`` is true,
    pre-existing generated aliases at unsupported levels are removed so the
    catalog stays consistent with the latest probe results. Set it false
    for routine updates that did not re-test ``thinking``: those are not
    evidence that a level was lost, so an existing generated top-level
    sibling (``-xhigh`` on the OpenAI scale, ``-max`` on the Moonshot
    scale) is trusted as proof of top-level support and the full alias
    set is regenerated (and synchronized) from it.

    The helper short-circuits with a plain write when ``name`` itself is a
    generated alias (marker on the incoming or on-disk entry, or the legacy
    ``-xhigh`` suffix), preserving the marker; we never produce nested
    aliases like ``foo-xhigh-xhigh``.
    """
    alias_base = entry.get("alias_of") or (data.get(name) or {}).get("alias_of")
    if alias_base or name.endswith(_XHIGH_SUFFIX):
        if alias_base:
            entry = dict(entry)
            entry["alias_of"] = alias_base
        data[name] = entry
        return
    entry = dict(entry)
    entry.pop("alias_of", None)
    scale = _thinking_scale_for(name)
    stored_level = entry.get("thinking")
    if stored_level not in scale:
        data[name] = entry
        if remove_stale_siblings:
            _pop_generated_aliases(data, name)
        return
    top_level = scale[-1]
    max_level = stored_level
    if not remove_stale_siblings and stored_level != top_level:
        sibling_name = f"{name}-{top_level}"
        sibling = data.get(sibling_name)
        if sibling is not None and _alias_base_name(sibling_name, sibling) == name:
            max_level = top_level
    base = dict(entry)
    high_rank = scale.index("high")
    max_rank = scale.index(max_level)
    base["thinking"] = "high" if max_rank > high_rank else max_level
    data[name] = base
    supported = scale[: max_rank + 1]
    for level in supported:
        sibling_name = f"{name}-{level}"
        existing = data.get(sibling_name)
        if existing is not None and _alias_base_name(sibling_name, existing) != name:
            # A real upstream model (e.g. o3-mini-high) or a foreign alias
            # occupies this name; never clobber it with a generated alias.
            continue
        data[sibling_name] = _expected_alias_entry(name, base, level)
    _pop_generated_aliases(data, name, keep_levels=supported)


def _stored_max_thinking_level(
    data: dict[str, dict],
    name: str,
    entry: dict[str, Any],
) -> str | None:
    """Return the max reasoning level recorded on disk for base entry ``name``.

    The base entry stores at most ``"high"`` (see
    :func:`_write_entry_with_thinking_split`), so a generated top-level
    sibling — ``-xhigh`` on the OpenAI scale, ``-max`` on the Moonshot
    scale — promotes the stored maximum to that top level.
    """
    scale = _thinking_scale_for(name)
    level = entry.get("thinking")
    if level not in scale:
        return None
    top_level = scale[-1]
    if level != top_level:
        sibling_name = f"{name}-{top_level}"
        sibling = data.get(sibling_name)
        if sibling is not None and _alias_base_name(sibling_name, sibling) == name:
            return top_level
    return str(level)


def _normalize_thinking_splits(data: dict[str, dict]) -> None:
    """Normalize entries to the base plus ``-{level}`` alias convention.

    Two passes:

    1. Drop orphan generated aliases whose base entry no longer exists.
    2. For every base entry with a recognized ``thinking`` level, regenerate
       the full set of ``-{level}`` siblings (repairing malformed ones and
       adding the ``alias_of`` marker to legacy ``-xhigh`` siblings).
    """
    for name, entry in list(data.items()):
        base_name = _alias_base_name(name, entry)
        if base_name is not None and base_name not in data:
            data.pop(name)
    for name, entry in list(data.items()):
        if _alias_base_name(name, entry) is not None:
            continue
        max_level = _stored_max_thinking_level(data, name, entry)
        if max_level is None:
            continue
        normalized = dict(entry)
        normalized["thinking"] = max_level
        _write_entry_with_thinking_split(data, name, normalized)


def _has_thinking_normalization_changes(data: dict[str, dict]) -> bool:
    """Return True when ``data`` needs generated thinking-alias normalization."""
    normalized = {name: dict(entry) for name, entry in data.items()}
    _normalize_thinking_splits(normalized)
    return normalized != data


def _read_model_info_json(path: Path) -> dict[str, dict]:
    """Read MODEL_INFO.json into a name → entry-dict mapping.

    Returns an empty dict when the file doesn't exist yet (lets the script
    bootstrap a brand new JSON file from scratch).
    """
    if not path.exists():  # pragma: no branch
        return {}
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _has_context_cap_changes(data: dict[str, dict]) -> bool:
    """Return True when any entry's context length still needs capping."""
    return any(
        entry.get("context_length", 0) >= _CONTEXT_CAP_THRESHOLD for entry in data.values()
    )


def _normalize_context_caps(data: dict[str, dict]) -> None:
    """Cap every entry's context length at 500000 when it is 1000000 or above."""
    for entry in data.values():
        ctx = entry.get("context_length", 0)
        if ctx >= _CONTEXT_CAP_THRESHOLD:
            entry["context_length"] = _CAPPED_CONTEXT_LENGTH


def _write_model_info_json(path: Path, data: dict[str, dict]) -> None:
    """Write ``data`` to ``path`` as sorted, pretty-printed JSON.

    Context lengths of 1000000 or above are capped at 500000 before
    writing so the on-disk catalog never advertises a >=1M context window.

    The publish is **atomic**: the ~200 KB catalog is written to a
    sibling temp file and then ``os.replace``d over the target.  A plain
    ``write_text`` truncates first, and ``model_info`` reads this file at
    **import time** — so any kiss process starting during the write (a
    ``run_parallel`` fan-out can start dozens per second) used to read a
    truncated prefix and die with a ``JSONDecodeError`` out of the import
    itself.

    Args:
        path: The catalog file to publish.
        data: The catalog contents, mutated in place by the context cap.
    """
    _normalize_context_caps(data)
    sorted_data = dict(sorted(data.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=f".{path.name}-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(sorted_data, indent=2) + "\n")
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


def apply_updates_to_file(
    updates: list[dict],
    new_models: list[dict],
    deprecated: list[dict],
    current: dict[str, dict],
    dry_run: bool = False,
) -> None:
    """Apply MODEL_INFO updates/additions/removals to the target catalog.

    Mutates ``MODEL_INFO_PATH`` — the repo's bundled ``MODEL_INFO.json``
    by default, or the catalog selected with ``--model-info`` (see the
    module docstring) — and nothing else.

    Args:
        updates: ``[{"name": str, "changes": {field: value, ...}}]``.
            ``changes`` may target ``context_length``, ``input_price_per_1M``,
            ``output_price_per_1M``, ``fc``, ``emb``, ``gen``, ``thinking``,
            ``use_responses_api``.  A ``thinking`` value of ``None`` and a
            falsy ``use_responses_api`` value remove their field.
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
    _normalize_thinking_splits(data)

    deprecated_names = {d["name"] for d in deprecated}
    removed = 0
    for name in deprecated_names:
        if data.pop(name, None) is not None:
            removed += 1
        for level in _ALL_THINKING_LEVELS:
            sibling_name = f"{name}-{level}"
            sibling = data.get(sibling_name)
            if sibling is not None and _alias_base_name(sibling_name, sibling) == name:
                data.pop(sibling_name)
                removed += 1

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
            use_responses_api=bool(cur.get("use_responses_api")),
        )
        changes = upd["changes"]
        for field, value in changes.items():  # pragma: no branch
            if field == "thinking" and value is None:  # pragma: no branch
                entry.pop("thinking", None)
            elif field == "use_responses_api" and not value:
                entry.pop("use_responses_api", None)
            else:
                entry[field] = value
        if entry.get("comment") == "NEW: needs pricing" and entry["input_price_per_1M"] > 0:
            entry["comment"] = "NEW"
        _write_entry_with_thinking_split(
            data,
            name,
            entry,
            remove_stale_siblings="thinking" in changes,
        )
        applied += 1

    added = 0
    for nm in new_models:  # pragma: no branch
        comment = "NEW: needs pricing" if nm.get("needs_pricing") else "NEW"
        entry = _build_entry(
            ctx=nm["context_length"],
            inp=nm["input_price_per_1M"],
            out=nm["output_price_per_1M"],
            fc=nm.get("fc", True),
            emb=nm.get("emb", False),
            gen=nm.get("gen", True),
            thinking=nm.get("thinking"),
            comment=comment,
            use_responses_api=nm.get("use_responses_api", False),
        )
        _write_entry_with_thinking_split(data, nm["name"], entry)
        added += 1

    print(f"\n  Removed {removed} deprecated, applied {applied} updates, added {added} new")
    if dry_run:
        print("  (dry-run, no files modified)")
        return
    _write_model_info_json(MODEL_INFO_PATH, data)
    print(f"  Written to {MODEL_INFO_PATH}")


def _readme_provider_category(model_name: str) -> str:
    """Return the README provider-category label for ``model_name``.

    Mirrors ``tests/test_readme_zai_moonshot._provider_category`` so that
    counts emitted into ``README.md`` are guaranteed to agree with the
    test's expectations after :func:`sync_readme_catalog` rewrites the
    "Models Supported" section. The ``cc/*`` and ``codex/*`` labels keep
    the backticks (and matching parentheses) so they can be matched
    verbatim by the test's ``| Provider | count |`` regex.
    """
    if model_name.startswith("openrouter/"):
        return "OpenRouter"
    if model_name.startswith("cc/"):
        return "Claude Code CLI (`cc/*`)"
    if model_name.startswith("codex/"):
        return "Codex CLI (`codex/*`)"
    if model_name.startswith("claude-"):
        return "Anthropic"
    if model_name.startswith("glm-"):
        return "Z.AI"
    if model_name.startswith(("kimi-", "moonshot-")):
        return "Moonshot AI"
    if model_name.startswith(("gemini-", "google/")):
        return "Gemini / Google"
    if model_name.startswith(("gpt-", "o", "computer-use-preview", "text-embedding-")):
        return "OpenAI"
    return "Together AI"


def _summary_label(category: str) -> str:
    """Map a README category label to its ``<summary>`` form (no backticks).

    The table rows keep the backticks (``| Claude Code CLI (`cc/*`) | 3 |``)
    but the collapsible-section headers strip them
    (``<summary><strong>Claude Code CLI (cc/*) (3)</strong></summary>``),
    so the rewriter must produce both spellings.
    """
    return category.replace("`", "")


def sync_readme_catalog(readme_path: Path, model_info_path: Path) -> bool:
    """Rewrite the catalog totals in ``README.md`` to match MODEL_INFO.json.

    Updates the catalog totals, capability counts, per-provider table
    counts, and ``<summary>`` headers in place using targeted regex
    substitutions, leaving the rest of the file untouched. Returns
    ``True`` when the file was modified.
    """
    data: dict[str, dict[str, Any]] = json.loads(model_info_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for name in data:
        category = _readme_provider_category(name)
        counts[category] = counts.get(category, 0) + 1
    total = sum(counts.values())
    cat_count = len(counts)
    generation = sum(1 for entry in data.values() if entry.get("gen"))
    function_calling = sum(1 for entry in data.values() if entry.get("fc"))
    embedding = sum(1 for entry in data.values() if entry.get("emb"))

    text = readme_path.read_text(encoding="utf-8")
    original = text

    text = re.sub(
        r"(\| \*\*Models in bundled catalog\*\* \| )\d+ across \d+ provider categories",
        rf"\g<1>{total} across {cat_count} provider categories",
        text,
    )
    text = re.sub(
        r"(ships a catalog of \*\*)\d+( models\*\* across \*\*)\d+( provider categories\*\*)",
        rf"\g<1>{total}\g<2>{cat_count}\g<3>",
        text,
    )
    text = re.sub(
        r"- \*\*\d+\*\* generation-capable models",
        f"- **{generation}** generation-capable models",
        text,
    )
    text = re.sub(
        r"- \*\*\d+\*\* function-calling-capable models",
        f"- **{function_calling}** function-calling-capable models",
        text,
    )
    text = re.sub(
        r"- \*\*\d+\*\* embedding models",
        f"- **{embedding}** embedding models",
        text,
    )

    for category, count in counts.items():
        table_pat = rf"(\| {re.escape(category)} \| )\d+( \|)"
        text = re.sub(table_pat, rf"\g<1>{count}\g<2>", text)
        summary = _summary_label(category)
        details_pat = rf"(<summary><strong>{re.escape(summary)} \()\d+(\)</strong></summary>)"
        text = re.sub(details_pat, rf"\g<1>{count}\g<2>", text)

    if text == original:
        return False
    readme_path.write_text(text, encoding="utf-8")
    return True


def _run_scrub_only(dry_run: bool = False) -> None:
    """Offline path: drop excluded-provider entries and resync README.

    Reads ``MODEL_INFO.json`` directly (rather than going through
    ``get_current_model_info``) so the script can run without importing
    any provider backends, then writes the trimmed JSON and refreshes the
    README's catalog totals. Used to apply provider-removal fixes (e.g.
    purging MiniMax) without requiring any vendor API keys.
    """
    print("=" * 60)
    print("Model Info Updater (scrub-only mode)")
    print("=" * 60)
    data = _read_model_info_json(MODEL_INFO_PATH)
    print(f"\n[1/3] Loaded {len(data)} entries from {MODEL_INFO_PATH}")
    removed = [name for name in list(data) if _is_excluded_provider(name)]
    print(f"\n[2/3] {len(removed)} excluded-provider entries to remove:")
    for name in removed:
        print(f"    {name}")
    if dry_run:
        print("  (dry-run, no files modified)")
        return
    for name in removed:
        data.pop(name, None)
    _write_model_info_json(MODEL_INFO_PATH, data)
    print(f"  Written to {MODEL_INFO_PATH}")
    if _writes_default_catalog():
        print("\n[3/3] Syncing README catalog totals...")
        changed = sync_readme_catalog(README_PATH, MODEL_INFO_PATH)
        print(f"  README updated: {changed} ({README_PATH})")
    else:
        print("\n[3/3] Non-default catalog target: README left untouched")
    print("\nDone!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update MODEL_INFO.json from vendor APIs")
    parser.add_argument(
        "--model-info",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Location of the MODEL_INFO.json catalog to read and update "
            f"(default: {DEFAULT_MODEL_INFO_PATH})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't modify files",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip capability testing")
    parser.add_argument("--test-existing", action="store_true", help="Re-test existing models")
    parser.add_argument(
        "--scrub-only",
        action="store_true",
        help=(
            "Offline mode: skip vendor API fetches, drop catalog entries "
            "belonging to permanently excluded providers, and resync "
            "README.md catalog totals. Requires no API keys."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.model_info is not None:
        # An explicit --model-info retargets the module-global write path
        # (left untouched otherwise, so a caller that pre-set the global
        # keeps its target).  The env override makes
        # kiss.core.models.model_info — imported lazily by
        # get_current_model_info() and the capability probes — read the
        # SAME catalog this run updates, so "current" and "target" never
        # diverge; a target that does not exist yet falls back to the
        # bundled catalog inside model_info.
        global MODEL_INFO_PATH
        MODEL_INFO_PATH = Path(args.model_info).expanduser().resolve()
        os.environ["KISS_MODEL_INFO_PATH"] = str(MODEL_INFO_PATH)
        if (
            not args.dry_run
            and not _writes_default_catalog()
            and not MODEL_INFO_PATH.exists()
            and DEFAULT_MODEL_INFO_PATH.exists()
        ):
            # A brand-new non-default target starts as a copy of the
            # bundled catalog; without the seed, apply_updates_to_file
            # would begin from an empty table and publish a catalog
            # holding only the entries this very run happened to touch.
            _write_model_info_json(
                MODEL_INFO_PATH, _read_model_info_json(DEFAULT_MODEL_INFO_PATH)
            )
            print(f"Seeded {MODEL_INFO_PATH} from {DEFAULT_MODEL_INFO_PATH}")

    if args.scrub_only:
        _run_scrub_only(dry_run=args.dry_run)
        return

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

    on_disk = _read_model_info_json(MODEL_INFO_PATH)
    needs_thinking_normalization = _has_thinking_normalization_changes(on_disk)
    needs_context_cap = _has_context_cap_changes(on_disk)
    if (  # pragma: no branch
        not updates
        and not new_models
        and not deprecated
        and not args.test_existing
        and not needs_thinking_normalization
        and not needs_context_cap
    ):
        print("\nEverything is up to date!")
        return

    if new_models and not args.skip_test:  # pragma: no branch
        print(f"\n[5/6] Testing {len(new_models)} new models...")
        for nm in new_models:  # pragma: no branch
            if nm["name"].startswith(("codex/", "cc/")):  # pragma: no branch
                continue
            caps = test_model_capabilities(nm["name"], verbose=args.verbose)
            nm["gen"] = caps["gen"]
            nm["emb"] = caps["emb"]
            nm["fc"] = caps["fc"]
            nm["thinking"] = caps["thinking"]
            # A brand-new entry is flagged only on a verified pass; an
            # inconclusive (None) verdict stays on Chat Completions.
            nm["use_responses_api"] = bool(caps.get("use_responses_api"))
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
            # Unverified: the v2 transport flag is only ever written after
            # a live probe, so --skip-test models stay on Chat Completions.
            nm["use_responses_api"] = False
    else:
        print("\n[5/6] No new models to test")

    if args.test_existing:  # pragma: no branch
        print("\n  Re-testing existing models...")
        update_by_name = {upd["name"]: upd for upd in updates}
        for name, cur in current.items():  # pragma: no branch
            if name.endswith(_XHIGH_SUFFIX) or cur.get("alias_of"):
                continue
            caps = test_model_capabilities(name, verbose=args.verbose)
            fc_changed = caps["fc"] != cur["fc"]
            stored_thinking = cur.get("thinking")
            top_level = _thinking_scale_for(name)[-1]
            sibling_thinking = current.get(f"{name}-{top_level}", {}).get("thinking")
            if stored_thinking == "high" and sibling_thinking == top_level:
                stored_thinking = top_level
            thinking_changed = caps["thinking"] != stored_thinking
            responses_verdict = caps.get("use_responses_api")
            # None means the probe never ran or died on transient endpoint
            # errors — a stored flag must survive an unexecuted probe.
            responses_changed = responses_verdict is not None and bool(
                responses_verdict
            ) != bool(cur.get("use_responses_api"))
            if not (fc_changed or thinking_changed or responses_changed):
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
                    f"    {name}: thinking changed {cur.get('thinking')!r} -> {caps['thinking']!r}"
                )
            if responses_changed:
                existing["changes"]["use_responses_api"] = bool(responses_verdict)
                print(
                    f"    {name}: use_responses_api changed "
                    f"{bool(cur.get('use_responses_api'))} -> "
                    f"{bool(responses_verdict)}"
                )

    print("\n[6/6] Applying changes...")
    apply_updates_to_file(updates, new_models, deprecated, current, dry_run=args.dry_run)

    if not args.dry_run and _writes_default_catalog() and README_PATH.exists():
        print("\n  Syncing README catalog totals...")
        changed = sync_readme_catalog(README_PATH, MODEL_INFO_PATH)
        print(f"  README updated: {changed} ({README_PATH})")

    print("\nDone!")


if __name__ == "__main__":
    main()
