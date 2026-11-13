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
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kiss.scripts.update_models as update_models_module
from kiss.core import config as config_module
from kiss.core.models.model_info import (
    OpenAICompatibleProvider,
    _match_openai_compatible_provider,
)
from kiss.scripts.update_models import (
    _alias_base_name,
    _noop_token_callback,
    _read_model_info_json,
    _write_model_info_json,
    calculator,
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

# ``thinking`` of a probe that is given no catalog row: the adapter then
# applies the default effort the runtime catalog has for the model.
_RUNTIME_DEFAULT: Any = object()


def _is_transient_error(exc: Exception) -> bool:
    """Return True for failures that say nothing about Responses support.

    Rate limits, 5xx server errors, connection drops and timeouts are
    endpoint load/health artifacts: a model must not lose (or miss) its
    ``use_responses_api`` flag because the vendor happened to be overloaded
    during the probe.  Definitive 4xx rejections ("not supported with the
    Responses API", "model_not_found") are NOT transient.  A
    ``server_error`` or a rate limit reported inside the stream is the
    HTTP 5xx / 429 of a request that had already been accepted, and is
    known by the provider's code on the error: the adapter's ``KISSError``
    carries it for a ``response.failed`` and for the bare ``error`` event,
    the SDK's ``APIError`` for a gateway's ``error`` object.

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
    # The last two of those again, reported inside a stream that had answered 200.
    if getattr(exc, "code", None) in ("server_error", "rate_limit_exceeded"):
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
            Responses-API support (exhausted transient errors, missing or
            rejected vendor key, wall-clock cutoff).  Inconclusive verdicts never
            change an existing catalog flag.
        detail: Human-readable outcome ("ok", "generation failed: ...").
    """

    name: str
    supported: bool
    detail: str
    conclusive: bool = True


def _build_v2_model(name: str, provider: OpenAICompatibleProvider, thinking: str | None) -> Any:
    """Return a fresh ``OpenAICompatibleModel2`` aimed at *provider*'s endpoint.

    Args:
        name: The catalog model name (may carry an ``openrouter/`` prefix).
        provider: The registry entry whose ``base_url`` and API key to use.
        thinking: The reasoning effort to request (``None``: none), or
            :data:`_RUNTIME_DEFAULT`.

    Returns:
        A v2 adapter ready for ``initialize``/``generate``.
    """
    from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

    api_key = getattr(config_module.DEFAULT_CONFIG, provider.api_key_name)
    # A tight stall watchdog plus an output cap keep a single
    # misbehaving vendor stream (endless slow reasoning tokens) from
    # hanging the whole batch; a model that cannot finish a trivial
    # task within these bounds does not usably support the transport.
    model_config: dict[str, Any] = {"stream_stall_timeout": 120.0, "max_output_tokens": 32000}
    if thinking is not _RUNTIME_DEFAULT:
        # Present, even as None, the key keeps the adapter from filling in the
        # effort the runtime catalog (not the probed one) has for the name.
        model_config["reasoning_effort"] = thinking
    return OpenAICompatibleModel2(
        name,
        base_url=provider.base_url,
        api_key=api_key,
        model_config=model_config,
        token_callback=_noop_token_callback,
    )


def _probe_generation(m: Any) -> str | None:
    """Run the plain-generation probe via ``/v1/responses``.

    Args:
        m: A fresh adapter for the model (:func:`_build_v2_model`).

    Returns:
        None on success, otherwise a short failure description.

    Raises:
        Exception: Whatever the SDK raised; classified by the caller.
    """
    m.initialize(_GENERATION_PROMPT)
    text, _ = m.generate()
    if not (text and text.strip()):
        return "generation failed: empty response text"
    return None


def _probe_tool_round_trip(m: Any) -> str | None:
    """Run the tool-calling round-trip probe via ``/v1/responses``.

    The model must (a) emit a ``function_call`` for the calculator tool and
    (b) accept the ``function_call_output`` on the follow-up request and
    produce a final assistant message — the exact sequence KISS's agentic
    loop performs on every tool turn.

    Args:
        m: A fresh adapter for the model (:func:`_build_v2_model`).

    Returns:
        None on success, otherwise a short failure description.

    Raises:
        Exception: Whatever the SDK raised; classified by the caller.
    """
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
    probe: Any, label: str, name: str, provider: OpenAICompatibleProvider, thinking: str | None
) -> tuple[str | None, bool]:
    """Run one probe function, retrying transient endpoint failures.

    Args:
        probe: ``_probe_generation`` or ``_probe_tool_round_trip``.
        label: Failure-message prefix ("generation" / "tool probe").
        name: The catalog model name.
        provider: The vendor endpoint to probe.
        thinking: The reasoning effort to request (see :func:`_build_v2_model`).

    Returns:
        ``(failure, conclusive)``: ``failure`` is None on success,
        otherwise a short description; ``conclusive`` is False when every
        attempt died on a transient endpoint error or the vendor rejected
        the API key or the account, so the verdict must not overwrite an
        existing catalog flag.
    """
    last: str | None = None
    for attempt in range(_PROBE_ATTEMPTS):
        try:
            return probe(_build_v2_model(name, provider, thinking)), True
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            last = f"{label} failed: {type(e).__name__}: {e}"
            if getattr(e, "status_code", None) in (401, 402, 403):
                # A refused key (401 / 403) or an account out of credits (402): the
                # answer does not show whether the model would have been served, so
                # the stored flag is kept and the request is not retried.
                return last, False
            if not _is_transient_error(e):
                return last, True
            if attempt + 1 < _PROBE_ATTEMPTS:
                time.sleep(_RETRY_DELAYS_SECONDS[attempt])
    return last, False


def probe_responses_support(
    name: str, fc: bool = True, thinking: str | None = _RUNTIME_DEFAULT
) -> ProbeResult:
    """Live-probe whether *name* works over its vendor's ``/v1/responses``.

    The single entry point shared by this script's batch mode and by
    ``update_models.py``'s per-model capability testing: a plain streaming
    generation probe, plus a full tool round-trip when *fc* is true, both
    through the framework's v2 adapter with transient-error retries.

    Args:
        name: The model name (with any ``openrouter/`` routing prefix).
        fc: Whether the model claims function-calling support (adds the
            tool round-trip requirement).
        thinking: The default reasoning effort of the catalog row being
            probed (``None``: the row has none); every probe request asks
            for exactly that.  Left out, the runtime catalog's default
            applies (``update_models.py`` probes the catalog it runs on).

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
        _probe_generation, "generation", name, provider, thinking
    )
    if failure is None and fc:
        failure, conclusive = _run_probe_with_retries(
            _probe_tool_round_trip, "tool probe", name, provider, thinking
        )
    if failure is not None:
        return ProbeResult(
            name=name, supported=False, detail=failure, conclusive=conclusive
        )
    return ProbeResult(name=name, supported=True, detail="ok")


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
        alias_base = _alias_base_name(name, entry)
        if alias_base is not None:
            skipped.append(f"{name} (alias of {alias_base})")
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
    (exhausted transient errors, missing or rejected key, wall-clock cutoff) says
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
        "--model-info",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Location of the MODEL_INFO.json catalog to read and update "
            "(default: the repo's bundled catalog, "
            f"{update_models_module.DEFAULT_MODEL_INFO_PATH})"
        ),
    )
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

    # Read the (possibly retargeted) global at call time — a from-import
    # would freeze the repo default even after update_models.main()
    # rebinds it or a test monkeypatches it.
    catalog_path = (
        Path(args.model_info).expanduser().resolve()
        if args.model_info is not None
        else update_models_module.MODEL_INFO_PATH
    )
    if not catalog_path.exists():
        # Not parser.error: its status 2 is this script's status for a partial run.
        sys.exit(f"--model-info {catalog_path} does not exist")
    data = _read_model_info_json(catalog_path)
    candidates, skipped = _candidate_names(data, args.only)
    print(f"Probing {len(candidates)} models ({len(skipped)} skipped) ...")

    def run_one(name: str) -> ProbeResult:
        row = data[name]
        result = probe_responses_support(name, fc=row.get("fc", True), thinking=row.get("thinking"))
        with _PRINT_LOCK:
            status = "PASS" if result.supported else "fail"
            print(f"  [{status}] {name}: {result.detail}", flush=True)
        return result

    # A vendor stream that ignores both max_output_tokens and the stall
    # watchdog (tokens keep trickling forever) would otherwise hang the
    # whole batch on pool shutdown, so the batch runs against one shared
    # monotonic deadline; probes still unfinished at the deadline get an
    # INCONCLUSIVE verdict (their existing catalog flag survives) and the
    # pool is abandoned (not joined) on every exit path, including a
    # failed publication and Ctrl-C.  Submission and shutdown are inside
    # the guard too: submission starts the workers, so an interrupt that
    # lands there already has a running probe to abandon, and one that
    # lands after publication, before or inside the shutdown, still has
    # the stuck worker to abandon.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {pool.submit(run_one, name): name for name in candidates}
        partial = _publish_batch(args, catalog_path, data, futures)
        _shutdown_pool(pool, futures)
    except BaseException:
        # No join can be trusted here: a submit() the interrupt cut short
        # may have queued, even started, a probe whose future never came
        # back, and a worker inside a vendor stream is a non-daemon thread
        # that interpreter exit would wait on forever.  os._exit skips
        # that; what print() buffered is flushed first.  The diagnostics
        # themselves may fail (a stdout pipe whose reader went away keeps
        # raising BrokenPipeError on every flush), so the exit is in a
        # finally: it happens whatever they do.
        try:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(1)
    # Status 2 tells automation the run was PARTIAL (some probes were cut
    # off as inconclusive), while the publication did succeed.
    if partial:
        sys.exit(2)


def _publish_batch(
    args: argparse.Namespace,
    catalog_path: Path,
    data: dict[str, dict[str, Any]],
    futures: dict[concurrent.futures.Future[ProbeResult], str],
) -> bool:
    """Wait for the probe batch, report it and publish the verdicts.

    The catalog is rewritten only when a verdict changed a flag, and before
    the ``--results-json`` dump: a dump that cannot be written costs the
    dump, not the verdicts.

    Args:
        args: The parsed command line.
        catalog_path: The catalog the verdicts are written to.
        data: The catalog as loaded before the probes started; used for
            the ``--dry-run`` report only.  A real publication re-reads
            the catalog under its writer lock so that entries and fields
            another updater committed while the probes ran survive.
        futures: The submitted probes, keyed to their catalog names.

    Returns:
        True when the batch deadline cut probes off (a partial run).
    """
    done, not_done = concurrent.futures.wait(futures, timeout=_BATCH_WALL_CLOCK_SECONDS)
    results: list[ProbeResult] = []
    for future in not_done:
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
    results.extend(future.result() for future in done)

    passed = sum(r.supported for r in results)
    print(f"\nResponses API works for {passed}/{len(results)} probed models.")

    if args.dry_run:
        flagged, unflagged = _apply_results(data, results)
    else:
        with update_models_module.catalog_lock(catalog_path):
            data = _read_model_info_json(catalog_path)
            flagged, unflagged = _apply_results(data, results)
            if flagged or unflagged:
                _write_model_info_json(catalog_path, data)

    if args.results_json:
        dump = {
            r.name: {"supported": r.supported, "conclusive": r.conclusive, "detail": r.detail}
            for r in results
        }
        Path(args.results_json).write_text(
            json.dumps(dump, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(f"Entries flagged: {flagged}, flags removed: {unflagged} (aliases included).")
    if args.dry_run:
        print("(dry-run, no files modified)")
        return bool(not_done)
    if flagged or unflagged:
        print(f"Written to {catalog_path}")
    # An inconclusive probe changed nothing: its model stays on whichever
    # transport its flag already named.
    for heading, conclusive in (
        ("Models staying on Chat Completions (probe failed):", True),
        ("Probes inconclusive (flags unchanged):", False),
    ):
        names = sorted(r.name for r in results if not r.supported and r.conclusive is conclusive)
        if names:
            print(f"\n{heading}")
            for name in names:
                print(f"  - {name}")
    return bool(not_done)


def _shutdown_pool(
    pool: concurrent.futures.ThreadPoolExecutor,
    futures: dict[concurrent.futures.Future[ProbeResult], str],
) -> None:
    """Shut the published batch's pool down, or exit hard when a worker cannot be stopped.

    Pending probes are cancelled.  A worker still inside a vendor stream
    (a probe the deadline cut off) is a non-daemon thread that
    ``shutdown(wait=True)`` -- and interpreter exit, which joins such
    threads -- would wait on forever, so the process exits with status 2
    (PARTIAL) instead; ``os._exit`` skips interpreter cleanup, so what
    print() buffered is flushed first.

    Args:
        pool: The batch's executor.
        futures: The submitted probes, keyed to their catalog names.
    """
    stuck = [f for f in futures if not f.done() and not f.cancel()]
    if stuck:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)
    pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
