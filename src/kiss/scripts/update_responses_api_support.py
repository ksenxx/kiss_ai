# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Live-verify OpenAI v2 (Responses API) support and flag it in MODEL_INFO.json.

For every generation-capable, non-alias catalog model that the ``model()``
factory routes to an OpenAI-compatible vendor (OpenAI, OpenRouter, Together,
Z.AI, Moonshot), this script probes the vendor's ``/v1/responses`` endpoint
**through the framework's own v2 adapter** (``OpenAICompatibleModel2``):

1. A plain streaming generation probe ("Reply with the single word: ok").
2. For function-calling models (``fc=true``), additionally a full tool
   round-trip: the model must emit a ``function_call`` for a calculator
   tool, accept the ``function_call_output``, and produce a final answer.

Models that pass every applicable probe get ``"use_responses_api": true``
written into their ``MODEL_INFO.json`` entry (and into their generated
``-{level}`` thinking aliases).  Models that fail lose any stale flag, so
re-runs reconcile the catalog with live reality.  The ``model()`` factory
then builds those models as ``OpenAICompatibleModel2`` — every request goes
to ``/v1/responses`` instead of ``/v1/chat/completions``.

Skipped (never probed, flag never written):

* alias entries (``alias_of`` marker or legacy ``-xhigh`` suffix) — they
  mirror their base entry, including this flag;
* embedding-only / non-generation entries;
* models routed to native SDKs or CLIs (``claude-*``, ``gemini-*``,
  ``cc/*``, ``codex/*``) — they have no OpenAI-compatible endpoint;
* vendors whose API key is not configured (probe result would be
  meaningless) — existing flags on their models are left untouched.

Usage:
    uv run python -m kiss.scripts.update_responses_api_support [--dry-run]
        [--only PREFIX] [--workers N]

Run it after ``update_models.py`` discovers new models so newly added
entries get probed too.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiss.core import config as config_module
from kiss.core.models.model_info import (
    OpenAICompatibleProvider,
    _match_openai_compatible_provider,
)
from kiss.scripts.update_models import (
    MODEL_INFO_PATH,
    _alias_base_name,
    _read_model_info_json,
    _write_model_info_json,
)

logger = logging.getLogger(__name__)

_GENERATION_PROMPT = "Reply with the single word: ok"

_TOOL_PROMPT = "What is 25 times 4? Use the calculator tool to compute it."

_PRINT_LOCK = threading.Lock()

_PROBE_ATTEMPTS = 3
_RETRY_DELAYS_SECONDS = (5.0, 15.0)

# Hard wall-clock bound for the WHOLE probe batch.  A healthy full-catalog
# run finishes in 15-30 minutes; the bound only exists to guarantee the
# batch terminates even when a vendor stream never ends.  Probes cut off by
# the deadline are recorded as inconclusive, and the script exits with
# status 2 (partial) after writing everything it learned.
_BATCH_WALL_CLOCK_SECONDS = 3600.0


def _is_transient_error(exc: Exception) -> bool:
    """Return True for failures that say nothing about Responses support.

    Rate limits, 5xx server errors, connection drops and timeouts are
    endpoint load/health artifacts: a model must not lose (or miss) its
    ``use_responses_api`` flag because the vendor happened to be overloaded
    during the probe.  Definitive 4xx rejections ("not supported with the
    Responses API", "model_not_found") are NOT transient.

    Args:
        exc: The exception raised by a probe attempt.

    Returns:
        True when the probe should be retried.
    """
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    if isinstance(
        exc,
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
    ):
        return True
    # The adapter's stall watchdog raises a builtin TimeoutError ("Model
    # stream stalled ... it will be retried"), and OpenRouter surfaces
    # upstream provider blips as an in-stream ``response.failed`` with the
    # generic "Provider returned error" message.  Both say nothing about
    # Responses-API support.
    if isinstance(exc, TimeoutError):
        return True
    return "Provider returned error" in str(exc)


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of probing one catalog model against ``/v1/responses``.

    Attributes:
        name: The catalog key that was probed.
        supported: True when every applicable probe passed.
        conclusive: False when the failure says nothing about the model's
            Responses-API support (exhausted transient errors, missing
            vendor key, wall-clock cutoff).  Inconclusive verdicts never
            change an existing catalog flag.
        detail: Human-readable outcome ("ok", "generation failed: ...").
    """

    name: str
    supported: bool
    detail: str
    conclusive: bool = True


def calculator(expression: str = "") -> str:
    """Compute a math expression.

    The wire tool name is this function's ``__name__`` (the schema builder
    uses it), so it must stay ``calculator`` — the round-trip probe checks
    the model called exactly that tool.

    Args:
        expression: A math expression string like '25*4'.
    """
    from kiss.scripts.update_models import safe_arithmetic

    return safe_arithmetic(expression)


def _noop_token_callback(_token: str) -> None:
    """No-op streaming callback so probes exercise the SSE path agents use."""


def _build_v2_model(name: str, provider: OpenAICompatibleProvider) -> Any:
    """Return a fresh ``OpenAICompatibleModel2`` aimed at *provider*'s endpoint.

    Args:
        name: The catalog model name (may carry an ``openrouter/`` prefix).
        provider: The registry entry whose ``base_url`` and API key to use.

    Returns:
        A v2 adapter ready for ``initialize``/``generate``.
    """
    from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

    api_key = getattr(config_module.DEFAULT_CONFIG, provider.api_key_name)
    return OpenAICompatibleModel2(
        name,
        base_url=provider.base_url,
        api_key=api_key,
        # A tight stall watchdog plus an output cap keep a single
        # misbehaving vendor stream (endless slow reasoning tokens) from
        # hanging the whole batch; a model that cannot finish a trivial
        # task within these bounds does not usably support the transport.
        model_config={"stream_stall_timeout": 120.0, "max_output_tokens": 32000},
        token_callback=_noop_token_callback,
    )


def _probe_generation(name: str, provider: OpenAICompatibleProvider) -> str | None:
    """Run the plain-generation probe for *name* via ``/v1/responses``.

    Args:
        name: The catalog model name.
        provider: The vendor endpoint to probe.

    Returns:
        None on success, otherwise a short failure description.

    Raises:
        Exception: Whatever the SDK raised; classified by the caller.
    """
    m = _build_v2_model(name, provider)
    m.initialize(_GENERATION_PROMPT)
    text, _ = m.generate()
    if not (text and text.strip()):
        return "generation failed: empty response text"
    return None


def _probe_tool_round_trip(name: str, provider: OpenAICompatibleProvider) -> str | None:
    """Run the tool-calling round-trip probe for *name* via ``/v1/responses``.

    The model must (a) emit a ``function_call`` for the calculator tool and
    (b) accept the ``function_call_output`` on the follow-up request and
    produce a final assistant message — the exact sequence KISS's agentic
    loop performs on every tool turn.

    Args:
        name: The catalog model name.
        provider: The vendor endpoint to probe.

    Returns:
        None on success, otherwise a short failure description.

    Raises:
        Exception: Whatever the SDK raised; classified by the caller.
    """
    m = _build_v2_model(name, provider)
    m.initialize(_TOOL_PROMPT)
    function_map = {"calculator": calculator}
    calls, content, _ = m.generate_and_process_with_tools(function_map)
    if not calls:
        return "tool probe failed: no function call emitted"
    for _turn in range(3):
        if any(call["name"] != "calculator" for call in calls):
            return "tool probe failed: called a tool that was not offered"
        results = []
        for call in calls:
            arguments = call.get("arguments") or {}
            expression = str(arguments.get("expression", "25*4"))
            results.append((call["name"], {"result": calculator(expression)}))
        m.add_function_results_to_conversation_and_return(results)
        calls, content, _ = m.generate_and_process_with_tools(function_map)
        if not calls:
            break
    if calls:
        return "tool probe failed: model never stopped calling tools"
    if not (content and content.strip()):
        return "tool probe failed: empty final answer after tool results"
    return None


def _run_probe_with_retries(
    probe: Any, label: str, name: str, provider: OpenAICompatibleProvider
) -> tuple[str | None, bool]:
    """Run one probe function, retrying transient endpoint failures.

    Args:
        probe: ``_probe_generation`` or ``_probe_tool_round_trip``.
        label: Failure-message prefix ("generation" / "tool probe").
        name: The catalog model name.
        provider: The vendor endpoint to probe.

    Returns:
        ``(failure, conclusive)``: ``failure`` is None on success,
        otherwise a short description; ``conclusive`` is False when every
        attempt died on a transient endpoint error, so the verdict must
        not overwrite an existing catalog flag.
    """
    import time

    last: str | None = None
    for attempt in range(_PROBE_ATTEMPTS):
        try:
            return probe(name, provider), True
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            last = f"{label} failed: {type(e).__name__}: {e}"
            if not _is_transient_error(e):
                return last, True
            if attempt + 1 >= _PROBE_ATTEMPTS:
                return last, False
            time.sleep(_RETRY_DELAYS_SECONDS[attempt])
    return last, False


def probe_responses_support(name: str, fc: bool = True) -> ProbeResult:
    """Live-probe whether *name* works over its vendor's ``/v1/responses``.

    The single entry point shared by this script's batch mode and by
    ``update_models.py``'s per-model capability testing: a plain streaming
    generation probe, plus a full tool round-trip when *fc* is true, both
    through the framework's v2 adapter with transient-error retries.

    Args:
        name: The model name (with any ``openrouter/`` routing prefix).
        fc: Whether the model claims function-calling support (adds the
            tool round-trip requirement).

    Returns:
        The probe outcome; ``supported`` is False when the model has no
        OpenAI-compatible endpoint or its vendor key is not configured.
    """
    provider = _match_openai_compatible_provider(name)
    if provider is None:
        # Definitive: the framework has no OpenAI-compatible endpoint for
        # this model, so the v2 transport can never apply.
        return ProbeResult(
            name=name, supported=False, detail="no OpenAI-compatible endpoint"
        )
    if not getattr(config_module.DEFAULT_CONFIG, provider.api_key_name, ""):
        # Inconclusive: nothing was probed, so an existing flag must survive.
        return ProbeResult(
            name=name,
            supported=False,
            detail=f"no {provider.api_key_name} configured",
            conclusive=False,
        )
    failure, conclusive = _run_probe_with_retries(
        _probe_generation, "generation", name, provider
    )
    if failure is None and fc:
        failure, conclusive = _run_probe_with_retries(
            _probe_tool_round_trip, "tool probe", name, provider
        )
    if failure is not None:
        return ProbeResult(
            name=name, supported=False, detail=failure, conclusive=conclusive
        )
    return ProbeResult(name=name, supported=True, detail="ok")


def _probe_model(name: str, entry: dict[str, Any]) -> ProbeResult:
    """Probe one catalog model and report whether v2 works for it.

    Args:
        name: The catalog key.
        entry: The model's MODEL_INFO.json entry.

    Returns:
        The probe outcome.
    """
    return probe_responses_support(name, fc=entry.get("fc", True))


def _is_alias(name: str, entry: dict[str, Any]) -> bool:
    """Return True when *name*/*entry* is a generated thinking alias."""
    return _alias_base_name(name, entry) is not None


def _candidate_names(
    data: dict[str, dict[str, Any]], only_prefix: str | None
) -> tuple[list[str], list[str]]:
    """Split catalog keys into probe candidates and skipped-with-reason names.

    Args:
        data: The full MODEL_INFO.json mapping.
        only_prefix: When set, restrict candidates to keys matching one of
            these comma-separated prefixes.

    Returns:
        ``(candidates, skipped)`` where ``skipped`` holds
        ``"name (reason)"`` strings for reporting.
    """
    keys = config_module.DEFAULT_CONFIG
    candidates: list[str] = []
    skipped: list[str] = []
    prefixes = (
        tuple(p for p in only_prefix.split(",") if p) if only_prefix else None
    )
    for name, entry in data.items():
        if prefixes is not None and not name.startswith(prefixes):
            continue
        if _is_alias(name, entry):
            skipped.append(f"{name} (alias of {_alias_base_name(name, entry)})")
            continue
        if not entry.get("gen", True):
            skipped.append(f"{name} (not a generation model)")
            continue
        provider = _match_openai_compatible_provider(name)
        if provider is None:
            skipped.append(f"{name} (no OpenAI-compatible endpoint)")
            continue
        if not getattr(keys, provider.api_key_name, ""):
            skipped.append(f"{name} (no {provider.api_key_name} configured)")
            continue
        candidates.append(name)
    return candidates, skipped


def _apply_results(
    data: dict[str, dict[str, Any]], results: list[ProbeResult]
) -> tuple[int, int]:
    """Write probe verdicts into the catalog mapping, aliases included.

    Only conclusive verdicts change the catalog: an inconclusive result
    (exhausted transient errors, missing key, wall-clock cutoff) says
    nothing about Responses support, so the entry's existing flag — set
    by an earlier successful probe — is preserved.

    Args:
        data: The full MODEL_INFO.json mapping, mutated in place.
        results: Probe outcomes for base (non-alias) models.

    Returns:
        ``(flagged, unflagged)`` counts of entries whose flag changed.
    """
    verdicts = {r.name: r for r in results}
    flagged = 0
    unflagged = 0
    for name, entry in data.items():
        base = _alias_base_name(name, entry) or name
        verdict = verdicts.get(base)
        if verdict is None or not verdict.conclusive:
            continue
        if verdict.supported and entry.get("use_responses_api") is not True:
            entry["use_responses_api"] = True
            flagged += 1
        elif not verdict.supported and "use_responses_api" in entry:
            entry.pop("use_responses_api")
            unflagged += 1
    return flagged, unflagged


def main() -> None:
    """Probe the catalog and update ``use_responses_api`` flags on disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Probe and report, but do not write."
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Probe only catalog keys matching one of these comma-separated prefixes.",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Concurrent probe workers."
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Optional path to dump per-model probe results as JSON.",
    )
    args = parser.parse_args()

    data = _read_model_info_json(MODEL_INFO_PATH)
    candidates, skipped = _candidate_names(data, args.only)
    print(f"Probing {len(candidates)} models ({len(skipped)} skipped) ...")

    results: list[ProbeResult] = []

    def run_one(name: str) -> ProbeResult:
        result = _probe_model(name, data[name])
        with _PRINT_LOCK:
            status = "PASS" if result.supported else "fail"
            print(f"  [{status}] {name}: {result.detail}", flush=True)
        return result

    # A vendor stream that ignores both max_output_tokens and the stall
    # watchdog (tokens keep trickling forever) would otherwise hang the
    # whole batch on pool shutdown, so the batch runs against one shared
    # monotonic deadline; probes still unfinished at the deadline get an
    # INCONCLUSIVE verdict (their existing catalog flag survives) and the
    # pool is abandoned (not joined).
    hung = False
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(run_one, name): name for name in candidates}
    done, not_done = concurrent.futures.wait(
        futures, timeout=_BATCH_WALL_CLOCK_SECONDS
    )
    for future in done:
        results.append(future.result())
    for future in not_done:
        hung = True
        future.cancel()
        name = futures[future]
        results.append(
            ProbeResult(
                name=name,
                supported=False,
                detail="probe timed out (batch wall-clock deadline)",
                conclusive=False,
            )
        )
        with _PRINT_LOCK:
            print(f"  [fail] {name}: probe timed out (inconclusive)", flush=True)
    if not hung:
        pool.shutdown(wait=True)

    passed = sorted(r.name for r in results if r.supported)
    failed = sorted(r.name for r in results if not r.supported)
    print(f"\nResponses API works for {len(passed)}/{len(results)} probed models.")

    if args.results_json:
        Path(args.results_json).write_text(
            json.dumps(
                {r.name: {"supported": r.supported, "detail": r.detail} for r in results},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    flagged, unflagged = _apply_results(data, results)
    print(f"Entries flagged: {flagged}, flags removed: {unflagged} (aliases included).")
    if args.dry_run:
        print("(dry-run, no files modified)")
        if hung:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(2)
        return
    _write_model_info_json(MODEL_INFO_PATH, data)
    print(f"Written to {MODEL_INFO_PATH}")
    if failed:
        print("\nModels staying on Chat Completions (probe failed):")
        for name in failed:
            print(f"  - {name}")
    if hung:
        # A stuck vendor stream keeps its (non-daemon) worker thread alive
        # forever; every result is already written, so exit hard instead of
        # letting interpreter shutdown join the abandoned pool.  os._exit
        # skips interpreter cleanup, so flush what print() buffered first.
        # Status 2 tells automation the run was PARTIAL (some probes were
        # cut off as inconclusive), while the catalog write did succeed.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)


if __name__ == "__main__":
    main()
