# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pre-run task classification for Sorcar agents.

Before a Sorcar agent starts a task, :func:`classify_task` answers two
questions about the task — with one typed question to a decisions model
when it can, else with one lightweight NON-AGENTIC
:class:`~kiss.core.kiss_agent.KISSAgent` call on the run's own model:

- ``is_simple``: the task involves neither software development nor
  Internet search.  A simple task runs with the lite system prompt
  (``SYSTEM_LITE.md``) instead of the full ``SYSTEM.md``.
- ``is_development``: the task may create or modify files in the
  project — software development, documentation, or producing an
  artifact such as a report, presentation, notebook or data file.
  Any such task must run in a worktree, since every file it writes
  may end up git-tracked.  Tasks that create or edit no file, and
  tasks that only request git operations (commit, merge, rebase,
  resolving merge conflicts, ...), are NOT development.  The verdict
  decides the run's effective
  ``is_worktree`` value (worktree isolation on/off) without ever
  touching the persisted ``is_worktree`` setting — but it can only
  DEMOTE a run that asked for a worktree to direct execution, never
  promote a run whose caller pinned ``use_worktree=False`` (a channel
  dispatch classifies for its system prompt while running in a
  scratch directory that must never get a worktree).

Two classifiers implement the verdict; :func:`classify_task` picks one
per call:

- The **decisions classifier** (default whenever it can run) asks
  OpenRouter's ``~typesafe/jev-latest`` decisions model ONE typed
  ``choice`` question — which of five kinds of task this is
  (:data:`_DECISIONS_KIND_CRITERIA`) — through the same ``decide`` tool
  the agents use (:mod:`kiss.agents.sorcar.decide_tool`), and maps the
  chosen kind to the verdict (:data:`_KIND_VERDICTS`).  It is a
  non-generative call: ~0.2 s, ~$0.00003 per task, and it cannot execute
  the task, so it also classifies for the run-to-completion CLI models
  (``cc/*``, ``codex/*``) the LLM classifier must skip.  Measured on
  ``benchmarkings/task_classifier/`` (415 real and synthetic prompts)
  its verdicts agreed with the hand labels more often than any of the
  four LLM classifiers measured (89% against 84% for the best), at
  about 1/200 of that LLM's cost and 1/15 of its latency.  It runs only
  when
  :func:`decisions_classification_enabled` says so: the
  ``classify_with_decisions`` config key (default on) and an
  ``OPENROUTER_API_KEY`` with the model in the catalog.  When it is off,
  unavailable, or its call fails, the LLM classifier below runs
  instead, so a missing key or an OpenRouter outage never changes the
  classification path's behaviour from what it was before.
- The **LLM classifier** runs a lightweight NON-AGENTIC
  :class:`~kiss.core.kiss_agent.KISSAgent` on the run's own model.

The LLM classification is a single non-agentic ``generate()`` call — no
tool loop, no tools — that returns the STRUCTURED verdict
``{"is_simple": <bool>, "is_development": <bool>}``.  The structure is
enforced twice over: the provider's own structured-output mechanism
pins :data:`_VERDICT_JSON_SCHEMA` on the request (``response_json_schema``
on Gemini, ``response_format`` on the OpenAI-compatible vendors), and
the prompt demands the same bare JSON object so the verdict survives
even on an endpoint without schema support.  Should a provider reject
the structured-output parameters, one plain retry (same prompt, no
schema knobs) recovers.  :func:`_parse_verdict` extracts the object
even when a lax model wraps it in code fences or prose, and accepts
only genuinely boolean values.

Anthropic models get the plain prompt only.  Measured on the live API,
``output_format`` adds ~280 input tokens of grammar and 1.0–1.4 s of
latency to a call that otherwise takes 0.6 s (haiku) / 2.1 s (fable),
and forcing ``thinking.type=disabled`` is rejected outright (HTTP 400)
by the adaptive-thinking generation (``claude-fable-5``,
``claude-opus-5``, Opus >= 4.6) — every such run used to pay a wasted
round trip before the plain fallback.  The Anthropic adapter picks the
right thinking mode for the model on its own, and the 1000-token
output cap keeps a fixed-budget model's thinking off.

The call is kept fast on purpose: one generation, a hard
:data:`CLASSIFIER_MAX_TOKENS` output cap (which also keeps extended
thinking off on models where a thinking budget would not fit), and a
terse JSON-only answer.  Verdicts of both classifiers are also
memoised per ``(criteria, model, endpoint, task text)`` in a small JSON
file under the KISS home (:func:`_cache_path`): a re-submitted prompt
— a retry, a template, a "hi" — skips the model call and launches
immediately.

Classification is best-effort and optional.  It runs only when
:func:`classification_enabled` says so (the ``classify_tasks`` config
key, toggleable in the settings panel, and the
``KISS_DISABLE_TASK_CLASSIFIER`` environment kill switch the test
suite sets).  Any failure — model error, budget, a run-to-completion
CLI model, an unparseable verdict — yields ``classification=None`` and
the agent behaves exactly as it did without a classifier.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.decide_tool import (
    DEFAULT_DECISIONS_MODEL,
    decisions_tool_available,
    make_decide_tool,
)
from kiss.core.config import kiss_home
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.decisions_model import choice

logger = logging.getLogger(__name__)

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"

# Overrides the API root the decisions classifier posts to (the tests
# point it at a local replay of ``/alpha/decisions``); unset means
# OpenRouter.
_DECISIONS_BASE_URL_ENV = "KISS_DECISIONS_BASE_URL"

# The decisions classifier waits at most this long for Jev.  On the
# benchmark a live call took 0.22 s at the median and never more than
# 1.2 s when it answered at all, while about 1% of requests hung until
# the timeout; anything past this bound is treated as an outage and the
# LLM classifier takes over, so the worst case adds these seconds to
# the LLM classifier's own latency rather than replacing it.
CLASSIFIER_DECISIONS_TIMEOUT_SECONDS = 5.0

_GIT_OPERATIONS = (
    "status, diff, log, add, commit, push, pull, fetch, checkout, branch, "
    "merge, squash, rebase, cherry-pick, tag, stash, resolving merge "
    "conflicts, rewriting commit messages, worktree management"
)

# The five kinds of task the decisions classifier chooses between.  The
# descriptions are contrastive on purpose: Jev reads them literally and
# picks the best-matching option.  Selected on
# ``benchmarkings/task_classifier/`` against four-``noul`` and hybrid
# designs (see ``run_benchmark.py`` there).
_DECISIONS_KIND_CRITERIA: dict[str, str] = {
    "development": (
        "any work that may create or modify files in the project: software "
        "development (implementing or changing features, fixing bugs or "
        "reported errors, refactoring, writing tests), updating "
        "documentation, README files or scripts, and producing a file such "
        "as a report, document, presentation, notebook, database, data or "
        "image file, or writing results into a file; includes requests "
        "phrased as required behaviour of the software ('when X happens, "
        "the app must do Y')"
    ),
    "git_only": (
        f"only git version-control operations ({_GIT_OPERATIONS}) with no "
        "code or file changes"
    ),
    "internet": (
        "answering a question, in the reply only, that requires searching "
        "or browsing the Internet or reading a web page, without creating "
        "or editing any file"
    ),
    "simple": (
        "a question, explanation, shell command or chore that creates or "
        "edits no file and needs neither software development nor Internet "
        "search: explaining or reading code, running a command or tests, "
        "reading messages or email, authenticating a service, managing "
        "servers, virtual machines or remote services, querying a database "
        "without changing it"
    ),
    "ambiguous": (
        "a fragment or follow-up that continues earlier work not shown here, "
        "such as 'continue', 'fix it', or a bare path"
    ),
}

_DECISIONS_QUESTIONS: dict[str, dict[str, Any]] = {
    "kind": choice("Which kind of task is this?", _DECISIONS_KIND_CRITERIA),
}
# The question set as the ``decide`` tool takes it.
_DECISIONS_QUESTIONS_JSON = json.dumps(_DECISIONS_QUESTIONS, sort_keys=True)

# A chosen kind whose probability is below this floor is treated as
# ``ambiguous`` — the conservative verdict — rather than trusted.  On
# the benchmark's held-out half this halves the harmful errors (a
# non-simple task on the lite prompt, a development task without a
# worktree) at unchanged accuracy.
CLASSIFIER_DECISIONS_MIN_PROBABILITY = 0.6

# Kind -> (is_simple, is_development).  ``ambiguous`` gets the same
# conservative verdict the LLM prompt demands for follow-ups it cannot
# see the context of.
_KIND_VERDICTS: dict[str, tuple[bool, bool]] = {
    "development": (False, True),
    "ambiguous": (False, True),
    "git_only": (True, False),
    "simple": (True, False),
    "internet": (False, False),
}

# The decisions classifier's memo criteria: everything that turns a task
# into its verdict — the questions, the kind -> verdict mapping and the
# probability floor — so changing any of them retires the old memos.
_DECISIONS_CRITERIA = json.dumps(
    {
        "questions": _DECISIONS_QUESTIONS,
        "verdicts": _KIND_VERDICTS,
        "min_probability": CLASSIFIER_DECISIONS_MIN_PROBABILITY,
    },
    sort_keys=True,
)

# Verdict cache: file name under the KISS home and the maximum number
# of entries kept (oldest dropped first).  Each entry is ~100 bytes.
CLASSIFIER_CACHE_FILENAME = "task_classifier_cache.json"
CLASSIFIER_CACHE_MAX_ENTRIES = 2000
# A memo is reused for at most this long.  A verdict is the model's
# reading of the prompt; re-asking once a week keeps a stale or
# unlucky first reading from sticking forever while still sparing
# every near-term repeat (retries, templates, follow-ups) the call.
CLASSIFIER_CACHE_TTL_SECONDS = 7 * 24 * 3600.0

# In-process mirror of the cache file, keyed by :func:`_cache_key`, in
# insertion order.  ``None`` until first use.  Guarded by
# ``_cache_lock`` together with the file it mirrors.
_cache_lock = threading.Lock()
_cache: dict[str, dict[str, Any]] | None = None
_cache_loaded_from: Path | None = None

# Hard output cap for the single classification generation, applied over
# any caller-configured output limit.  The verdict is a one-line JSON
# object, so this is generous headroom; on models whose extended
# thinking needs a >=1024-token budget inside ``max_tokens``, a cap this
# small keeps thinking off entirely, which is exactly what a quick
# classification wants.
CLASSIFIER_MAX_TOKENS = 1000

# Spend cap passed to the classification run.  A non-agentic run is a
# single generation, so the framework has no mid-run point at which to
# stop it; the effective cost bound comes from the inputs instead — the
# task is truncated to ``CLASSIFIER_TASK_MAX_CHARS`` and the output
# capped at ``CLASSIFIER_MAX_TOKENS``, keeping each attempt far under
# this figure.  The spend is folded into the main run's totals only
# after the run ends (see ``SorcarAgent._fold_classifier_usage``).
CLASSIFIER_MAX_BUDGET = 1.0

# Classification reads the gist of a task, not every byte: a prompt
# longer than this is truncated before it is sent, which bounds each
# attempt's input cost and latency no matter how large the main run's
# prompt is.
CLASSIFIER_TASK_MAX_CHARS = 20_000

# Fail fast when a provider stalls: a streamed classification that has
# produced no output for this long is abandoned (and the plain fallback
# attempt, if any, gets its turn) instead of waiting out the
# framework's default stall timeout.  Imposed over any caller value —
# the caller's patience budget sizes the MAIN run, not this one-line
# verdict.  Unary (non-streaming) transports are bounded by their SDK
# client's own request timeout instead; the input/output caps above
# keep the normal case to a few seconds either way.
CLASSIFIER_STALL_TIMEOUT_SECONDS = 60.0

# The verdict's JSON schema, enforced on the request by every provider
# with a structured-output mechanism (see ``_structured_output_config``)
# and demanded verbatim by the prompt for everything else.
_VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_simple": {
            "type": "boolean",
            "description": (
                "True if the task involves neither software development "
                "nor searching the internet."
            ),
        },
        "is_development": {
            "type": "boolean",
            "description": (
                "True if the task may create or modify any file in the "
                "project: software development, documentation, or "
                "producing a report, document, presentation, notebook, "
                "database or data file. False for tasks that create or "
                "edit no file (questions, explanations, running commands, "
                "answers given in the reply) and for git-only tasks."
            ),
        },
    },
    "required": ["is_simple", "is_development"],
    "additionalProperties": False,
}

_CLASSIFIER_PROMPT_PREFIX = (
    "You are a task classifier. Read the task below and respond with "
    "ONLY one JSON object — no prose, no code fences, no explanation — "
    "exactly matching this schema:\n"
    '{"is_simple": <true|false>, "is_development": <true|false>}\n\n'
    "- is_simple: true if the task involves NEITHER software "
    "development NOR searching the internet. false otherwise.\n"
    "- is_development: true if the task may create or modify ANY file in "
    "the project: software development (implementing or changing "
    "features, fixing bugs, refactoring, writing tests), updating "
    "documentation or scripts, and producing a file such as a report, "
    "document, presentation, notebook, database, data or image file. "
    "false only for tasks that create or edit no file: questions, "
    "explanations, running commands or tests, reading messages, or "
    "answers given in the reply. A "
    "task that only requests git operations (e.g. status, diff, log, "
    "add, commit, push, pull, fetch, checkout, branch, merge, squash, "
    "rebase, resolving merge conflicts, tagging, worktree management) "
    "is NOT software development and MUST get is_development=false.\n\n"
    "If the task is ambiguous, or is a follow-up that continues earlier "
    "work you cannot see (e.g. 'continue', 'fix it', 'do the same for "
    "the rest'), be conservative: respond with "
    '{"is_simple": false, "is_development": true}.\n\n'
    "Do NOT attempt to perform the task. Answer immediately with the "
    "single JSON object.\n\n"
    "# Task\n"
)


@dataclass(frozen=True)
class TaskClassification:
    """The classifier's verdict about one task.

    Attributes:
        is_simple: The task involves neither software development nor
            Internet search.
        is_development: The task may create or modify files in the
            project — code, documentation, or artifacts such as
            reports, presentations, notebooks or data files — and so
            must run in a worktree (git-only tasks are not
            development).
    """

    is_simple: bool
    is_development: bool


@dataclass(frozen=True)
class ClassifierRun:
    """Outcome and usage of one classification attempt.

    Attributes:
        classification: The parsed verdict, or ``None`` when the
            attempt failed for any reason.
        budget_used: USD spent by the classification call(s).
        tokens_used: Total tokens consumed by the classification
            call(s).
        steps: LLM steps the classification took (1, or 2 when the
            structured-output attempt needed the plain fallback; 0 for
            a decisions verdict, which is not an LLM step, and for a
            memo hit).
    """

    classification: TaskClassification | None
    budget_used: float
    tokens_used: int
    steps: int


def classification_enabled(override: bool | None = None) -> bool:
    """Whether pre-run task classification should run.

    Args:
        override: Per-run override of the persisted ``classify_tasks``
            setting — the ``classifyTasks`` wire field of the ``run``
            command (the *classify_tasks* parameter of
            :func:`kiss.server.sorcar.run`).  ``None`` (the default)
            means "no override": the config key decides.

    Returns:
        ``False`` when the ``KISS_DISABLE_TASK_CLASSIFIER`` environment
        variable is set to ``"1"`` (the test suite's kill switch, which
        wins over everything).  Otherwise *override* when it is not
        ``None``, else the ``classify_tasks`` config key — persisted in
        ``~/.kiss/config.json`` and toggleable in the settings panel —
        defaulting to ``True``.
    """
    if os.environ.get(_DISABLE_ENV, "") == "1":
        return False
    if override is not None:
        return override
    from kiss.core.vscode_config import load_config

    try:
        return bool(load_config().get("classify_tasks", True))
    except Exception:  # pragma: no cover — unreadable config
        logger.debug("Could not read classify_tasks", exc_info=True)
        return True


def decisions_classification_enabled() -> bool:
    """Whether :func:`classify_task` should try the decisions classifier first.

    Returns:
        ``True`` when the ``classify_with_decisions`` config key (persisted
        in ``~/.kiss/config.json``, default ``True``; the settings panel's
        "Classify with Jev" checkbox) is on AND the
        ``decide`` tool can work in this process — an ``OPENROUTER_API_KEY``
        is configured and :data:`DEFAULT_DECISIONS_MODEL` is in the catalog
        (see :func:`~kiss.agents.sorcar.decide_tool.decisions_tool_available`).
        ``False`` sends every classification to the LLM classifier.
    """
    if not decisions_tool_available():
        return False
    from kiss.core.vscode_config import load_config

    try:
        return bool(load_config().get("classify_with_decisions", True))
    except Exception:  # pragma: no cover — unreadable config
        logger.debug("Could not read classify_with_decisions", exc_info=True)
        return True


def _verdict_bool(value: Any) -> bool | None:
    """Interpret one verdict value as a strict boolean.

    Args:
        value: A candidate ``is_simple`` / ``is_development`` value from
            the model's JSON.

    Returns:
        The boolean it denotes — a real JSON boolean, or one of the lax
        string spellings ``"true"/"false"``, ``"yes"/"no"``, ``"1"/"0"``
        — or ``None`` for every other type or spelling (numbers, null,
        arrays, objects), which must NOT silently become a verdict that
        flips worktree isolation.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    return None


def _parse_verdict(result: str) -> TaskClassification | None:
    """Extract the structured verdict from the model's response text.

    The schema demands a bare JSON object, but a lax model (or one on an
    endpoint without structured-output enforcement) may wrap it in a
    code fence or surround it with prose, so every JSON object embedded
    in *result* is tried, in order, until one carries both verdict keys
    with genuinely boolean values.

    Args:
        result: The response text of the non-agentic classification
            call.

    Returns:
        The parsed :class:`TaskClassification`, or ``None`` when no
        JSON object holding boolean ``is_simple`` and ``is_development``
        values can be found in *result*.
    """
    if not isinstance(result, str):
        return None
    decoder = json.JSONDecoder()
    idx = result.find("{")
    while idx != -1:
        try:
            verdict, _end = decoder.raw_decode(result, idx)
        except ValueError:
            idx = result.find("{", idx + 1)
            continue
        if isinstance(verdict, dict):
            is_simple = _verdict_bool(verdict.get("is_simple"))
            is_development = _verdict_bool(verdict.get("is_development"))
            if is_simple is not None and is_development is not None:
                return TaskClassification(
                    is_simple=is_simple, is_development=is_development,
                )
        idx = result.find("{", idx + 1)
    return None


def _classifier_model_config(
    model_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the classification call's base model configuration.

    Starts from a copy of the main run's *model_config* (custom
    endpoint, headers) and imposes the classifier's own bounds: the
    output is capped at :data:`CLASSIFIER_MAX_TOKENS` — overriding any
    caller output limit under every portable or native spelling, since
    the caller's limit sizes the MAIN run's answers, not this one-line
    verdict — the caller's thinking configuration is dropped, and a
    stalled streaming provider is abandoned after
    :data:`CLASSIFIER_STALL_TIMEOUT_SECONDS`.

    Args:
        model_config: The model configuration the Sorcar agent will run
            with, or ``None``.  Never mutated.

    Returns:
        The base configuration for the classifier's model.
    """
    config: dict[str, Any] = dict(model_config) if model_config else {}
    config.pop("max_completion_tokens", None)
    config.pop("max_output_tokens", None)
    config["max_tokens"] = CLASSIFIER_MAX_TOKENS
    # The main run's thinking configuration is dropped, not inherited: a
    # caller thinking budget sized for the main run (e.g. 10000 tokens)
    # cannot fit inside this call's 1000-token output cap — Anthropic
    # rejects budget_tokens >= max_tokens — and a one-line verdict needs
    # no reasoning budget anyway.
    config.pop("thinking", None)
    config["stream_stall_timeout"] = CLASSIFIER_STALL_TIMEOUT_SECONDS
    return config


def _structured_output_config(
    base_config: dict[str, Any], model_name: str
) -> dict[str, Any] | None:
    """Add the provider's structured-output enforcement to *base_config*.

    Args:
        base_config: The classifier's base model configuration (see
            :func:`_classifier_model_config`).  Never mutated.
        model_name: The classifying model, used to pick the provider's
            structured-output mechanism.

    Returns:
        A copy of *base_config* that pins :data:`_VERDICT_JSON_SCHEMA`
        on the request — ``response_json_schema`` for Gemini,
        Chat-Completions ``response_format`` for the registered
        OpenAI-compatible vendors — or ``None`` when the provider has no
        structured-output mechanism worth carrying: unknown providers,
        and Anthropic, whose ``output_format`` grammar measurably slows
        the call down (see the module docstring) while the prompt alone
        already yields the bare JSON object.  The caller then relies on
        the prompt alone.
    """
    from kiss.core.models.model_info import (
        OPENAI_COMPATIBLE_PROVIDERS,
        _strip_provider_prefix,
        get_model_provider,
    )

    # The model factory strips harbor-style prefixes before routing
    # ("anthropic/claude-...", "google/gemini-..."), so provider lookup
    # must too or every prefixed name would land on "Unknown" and lose
    # its schema enforcement.
    provider = get_model_provider(_strip_provider_prefix(model_name))
    config = dict(base_config)
    if provider == "Gemini":
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = _VERDICT_JSON_SCHEMA
        return config
    if provider in {p.label for p in OPENAI_COMPATIBLE_PROVIDERS}:
        config["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "task_classification",
                "strict": True,
                "schema": _VERDICT_JSON_SCHEMA,
            },
        }
        return config
    return None


def _cache_path() -> Path:
    """Return the verdict cache file: ``<KISS home>/task_classifier_cache.json``.

    Resolved on every call so a ``KISS_HOME`` change (the test suite
    isolates one per test) switches files.
    """
    return kiss_home() / CLASSIFIER_CACHE_FILENAME


def _cache_key(
    task: str,
    model_name: str,
    model_config: dict[str, Any] | None,
    criteria: str = _CLASSIFIER_PROMPT_PREFIX,
) -> str:
    """Return the cache key for one classification request.

    Args:
        task: The (already truncated) task text sent to the classifier.
        model_name: The classifying model.
        model_config: The run's model configuration.  The WHOLE mapping
            takes part in the key (canonical JSON, so key order is
            irrelevant): endpoint, credentials, headers and provider
            routing can all send the same model name to a different
            backend, and two requests that may be answered differently
            must never share a memo.  Only the digest is stored, never
            the values.
        criteria: The classification criteria that turn the model's
            answer into a verdict: the LLM classifier's prompt prefix
            (the default) or the decisions classifier's questions,
            verdict mapping and probability floor
            (:data:`_DECISIONS_CRITERIA`).  The two classifiers therefore
            never share a memo.

    Returns:
        A hex SHA-256 digest of the criteria, model name, model
        configuration and task text.  Including the criteria retires
        every memo automatically when they change in a later release.
    """
    config_json = json.dumps(model_config or {}, sort_keys=True, default=str)
    digest = hashlib.sha256()
    for part in (criteria, model_name, config_json):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    digest.update(task.encode("utf-8"))
    return digest.hexdigest()


def _read_cache_file(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the cache file at *path*, dropping invalid or expired entries.

    Args:
        path: The cache file.

    Returns:
        Valid, unexpired entries (``is_simple``/``is_development``
        booleans plus the ``ts`` write time) in file order.  A missing,
        unreadable or malformed file yields an empty mapping.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    oldest_valid = time.time() - CLASSIFIER_CACHE_TTL_SECONDS
    entries: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not (isinstance(key, str) and isinstance(value, dict)):
            continue
        is_simple = value.get("is_simple")
        is_development = value.get("is_development")
        ts = value.get("ts")
        if not (isinstance(is_simple, bool) and isinstance(is_development, bool)):
            continue
        if not isinstance(ts, (int, float)) or ts < oldest_valid:
            continue
        entries[key] = {
            "is_simple": is_simple, "is_development": is_development, "ts": ts,
        }
    return entries


def _load_cache_locked() -> dict[str, dict[str, Any]]:
    """Return the in-process cache, (re)loading it from disk when needed.

    Caller holds ``_cache_lock``.  The mirror is (re)read whenever the
    cache path changed (``KISS_HOME`` switched) or nothing is loaded.
    """
    global _cache, _cache_loaded_from
    path = _cache_path()
    if _cache is None or _cache_loaded_from != path:
        _cache = _read_cache_file(path)
        _cache_loaded_from = path
    return _cache


def cached_classification(
    task: str,
    model_name: str,
    model_config: dict[str, Any] | None = None,
    criteria: str = _CLASSIFIER_PROMPT_PREFIX,
) -> TaskClassification | None:
    """Return the memoised verdict for *task* on *model_name*, if any.

    Args:
        task: The task text exactly as :func:`classify_task` would send
            it (after truncation).
        model_name: The classifying model.
        model_config: The run's model configuration, or ``None``.
        criteria: The classifier's criteria text, see :func:`_cache_key`.

    Returns:
        The cached :class:`TaskClassification`, or ``None`` on a miss
        or when the memo is older than :data:`CLASSIFIER_CACHE_TTL_SECONDS`.
    """
    key = _cache_key(task, model_name, model_config, criteria)
    with _cache_lock:
        entry = _load_cache_locked().get(key)
    if entry is None or entry["ts"] < time.time() - CLASSIFIER_CACHE_TTL_SECONDS:
        return None
    return TaskClassification(
        is_simple=entry["is_simple"], is_development=entry["is_development"],
    )


def remember_classification(
    task: str,
    model_name: str,
    model_config: dict[str, Any] | None,
    classification: TaskClassification,
    criteria: str = _CLASSIFIER_PROMPT_PREFIX,
) -> None:
    """Memoise *classification* for *task* on *model_name* and persist it.

    Memos another process wrote to the file since this process last
    read it are merged in first (newer write wins per key), so
    concurrent daemon and CLI processes only ever lose a memo to a
    same-instant race, never to a stale mirror.  The cache is bounded
    at :data:`CLASSIFIER_CACHE_MAX_ENTRIES` (oldest first) and written
    atomically; a write failure only loses the memo, never the verdict.

    Args:
        task: The task text exactly as sent to the classifier.
        model_name: The classifying model.
        model_config: The run's model configuration, or ``None``.
        classification: The verdict to remember.
        criteria: The classifier's criteria text, see :func:`_cache_key`.
    """
    key = _cache_key(task, model_name, model_config, criteria)
    with _cache_lock:
        cache = _load_cache_locked()
        path = _cache_path()
        for other_key, other in _read_cache_file(path).items():
            mine = cache.get(other_key)
            if mine is None or mine["ts"] < other["ts"]:
                cache[other_key] = other
        cache.pop(key, None)
        cache[key] = {
            "is_simple": classification.is_simple,
            "is_development": classification.is_development,
            "ts": time.time(),
        }
        while len(cache) > CLASSIFIER_CACHE_MAX_ENTRIES:
            del cache[next(iter(cache))]
        tmp_path = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(cache), encoding="utf-8")
            os.replace(tmp_path, path)
        except OSError:
            logger.debug("Could not persist classifier cache", exc_info=True)
            try:
                tmp_path.unlink()
            except OSError:
                pass


def clear_classification_cache() -> None:
    """Forget every memoised verdict, in memory and on disk.

    Used by tests and by callers that change the classifier prompt
    semantics.
    """
    global _cache, _cache_loaded_from
    with _cache_lock:
        _cache = None
        _cache_loaded_from = None
        try:
            _cache_path().unlink()
        except OSError:
            pass


def _attempt_classification(
    task: str, model_name: str, model_config: dict[str, Any]
) -> tuple[TaskClassification | None, float, int, int]:
    """Run one non-agentic classification generation.

    Args:
        task: The task prompt to classify.
        model_name: The classifying model.
        model_config: The full model configuration for this attempt.

    Returns:
        Tuple of the parsed verdict (or ``None``), and the attempt's
        spend in USD, tokens, and steps.
    """
    agent = KISSAgent("Task Classifier")
    result = ""
    try:
        result = agent.run(
            model_name=model_name,
            prompt_template=_CLASSIFIER_PROMPT_PREFIX + task,
            is_agentic=False,
            max_budget=CLASSIFIER_MAX_BUDGET,
            model_config=model_config,
            verbose=False,
            print_prompts=False,
        )
    except Exception:
        logger.warning("Task classification attempt failed", exc_info=True)
    classification = _parse_verdict(result)
    if classification is None and result:
        logger.warning(
            "Task classifier returned an unparseable verdict: %.200s",
            result,
        )
    return (
        classification,
        float(getattr(agent, "budget_used", 0.0) or 0.0),
        int(getattr(agent, "total_tokens_used", 0) or 0),
        int(getattr(agent, "step_count", 0) or 0),
    )


def _verdict_from_decision(answers: Any) -> TaskClassification | None:
    """Map the decisions classifier's answers to a verdict.

    Args:
        answers: The ``answers`` object of a ``decide`` tool result.

    Returns:
        The :class:`TaskClassification` for the chosen kind
        (:data:`_KIND_VERDICTS`) — or the conservative ``ambiguous``
        verdict when the chosen kind's probability is below
        :data:`CLASSIFIER_DECISIONS_MIN_PROBABILITY` — or ``None`` when
        the ``kind`` answer is missing or names an option that was not
        asked for.
    """
    if not isinstance(answers, dict) or not isinstance(answers.get("kind"), dict):
        return None
    kind = answers["kind"].get("choice")
    if not isinstance(kind, str) or kind not in _KIND_VERDICTS:
        return None
    verdict = _KIND_VERDICTS[kind]
    probabilities = answers["kind"].get("probabilities")
    probability = probabilities.get(kind) if isinstance(probabilities, dict) else None
    if isinstance(probability, (int, float)) and probability < CLASSIFIER_DECISIONS_MIN_PROBABILITY:
        verdict = _KIND_VERDICTS["ambiguous"]
    return TaskClassification(is_simple=verdict[0], is_development=verdict[1])


def _decisions_model_config() -> dict[str, Any]:
    """Return the ``decide`` tool's model settings for the classifier.

    The HTTP timeout is :data:`CLASSIFIER_DECISIONS_TIMEOUT_SECONDS`; the
    API root is OpenRouter unless ``KISS_DECISIONS_BASE_URL`` is set.
    """
    config: dict[str, Any] = {"timeout": CLASSIFIER_DECISIONS_TIMEOUT_SECONDS}
    base_url = os.environ.get(_DECISIONS_BASE_URL_ENV, "")
    if base_url:
        config["base_url"] = base_url
    return config


def _attempt_decisions_classification(
    task: str,
) -> tuple[TaskClassification | None, float, int]:
    """Classify *task* once with the ``decide`` tool on the decisions model.

    Args:
        task: The (truncated) task prompt to classify; it is the
            ``state`` the single ``kind`` question is asked about.

    Returns:
        Tuple of the verdict (or ``None`` when the tool reported an
        error, the endpoint was unreachable, or the answer did not name
        one of the asked-for kinds) and the attempt's spend in USD and
        tokens, as the tool priced them from the catalog.
    """
    try:
        decide = make_decide_tool(None, DEFAULT_DECISIONS_MODEL, _decisions_model_config())
        result = decide(task, _DECISIONS_QUESTIONS_JSON)
        if result.startswith("Error:"):
            logger.warning("Decisions task classification failed: %.300s", result)
            return None, 0.0, 0
        parsed = json.loads(result)
        usage = parsed.get("usage") or {}
        budget = float(usage.get("cost_usd") or 0.0)
        tokens = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        classification = _verdict_from_decision(parsed.get("answers"))
    except Exception:
        # The classifier is best-effort: nothing here may raise into the
        # agent's launch path, whatever the endpoint or the tool returns.
        logger.warning("Decisions task classification attempt failed", exc_info=True)
        return None, 0.0, 0
    if classification is None:
        logger.warning("Decisions task classifier returned no usable kind: %.300s", result)
    return classification, budget, tokens


def _model_runs_task_to_completion(model_name: str) -> bool:
    """Whether *model_name* is a run-to-completion CLI model (cc/codex).

    Args:
        model_name: The classifying model.

    Returns:
        ``True`` for models classification must skip; ``False`` for
        every other model and for names the registry cannot resolve.
    """
    from kiss.core.models.model_info import model_runs_task_to_completion

    try:
        return bool(model_runs_task_to_completion(model_name))
    except Exception:
        return False


def _truncate_task(task: str) -> str:
    """Return *task* cut to :data:`CLASSIFIER_TASK_MAX_CHARS` for the classifier.

    The gist suffices for classification; the full prompt belongs to
    the main run.  Truncation bounds the call's input cost and latency
    no matter how large the task is, and the same text keys the
    verdict cache.

    Args:
        task: The task prompt about to be classified.

    Returns:
        *task* unchanged when it fits, else its prefix plus a marker.
    """
    if len(task) <= CLASSIFIER_TASK_MAX_CHARS:
        return task
    return task[:CLASSIFIER_TASK_MAX_CHARS] + "\n... [task truncated for classification]"


def classification_will_call_model(
    task: str,
    model_name: str,
    model_config: dict[str, Any] | None = None,
    enabled_override: bool | None = None,
) -> bool:
    """Whether :func:`classify_task` would make a model round trip now.

    Lets callers overlap other launch work (e.g. preparing a spare git
    worktree) with the classifier's wait — and skip that overlap when
    there is nothing to wait for.  Applies exactly the gates
    :func:`classify_task` applies before calling a model: the classifier
    must be enabled (see :func:`classification_enabled`), the verdict
    must not be memoised already, and — on the LLM route only — the
    model must not be a run-to-completion CLI model.  Best-effort by
    nature: the answer is a snapshot, and a concurrent memo write or
    settings change between this call
    and :func:`classify_task` can make the two disagree.  Callers must
    only use it to schedule work that is harmless either way.

    Args:
        task: The task prompt as it will be passed to
            :func:`classify_task` (before truncation).
        model_name: The classifying model.
        model_config: The run's model configuration (endpoint), or
            ``None``.
        enabled_override: Per-run ``classify_tasks`` override, see
            :func:`classification_enabled`.

    Returns:
        ``True`` when a classification model call is imminent — the
        decisions classifier's, or the LLM classifier's when the
        decisions classifier is off (see
        :func:`decisions_classification_enabled`).
    """
    if not classification_enabled(enabled_override):
        return False
    task = _truncate_task(task)
    if decisions_classification_enabled():
        return _cached_decision(task) is None
    if _model_runs_task_to_completion(model_name):
        return False
    return cached_classification(task, model_name, model_config) is None


def _cached_decision(task: str) -> TaskClassification | None:
    """Return the memoised decisions-classifier verdict for *task*, if any.

    Keyed by :data:`_DECISIONS_CRITERIA`, the decisions model and its
    endpoint settings (:func:`_decisions_model_config`), so a verdict
    from one endpoint is never served for another.
    """
    return cached_classification(
        task, DEFAULT_DECISIONS_MODEL, _decisions_model_config(), _DECISIONS_CRITERIA,
    )


def classify_task(
    task: str,
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> ClassifierRun:
    """Classify *task*: with the decisions classifier when it can run, else the LLM.

    When :func:`decisions_classification_enabled` allows it, one
    ``decide`` tool call on :data:`DEFAULT_DECISIONS_MODEL` produces the
    verdict.  If that classifier is off, unavailable (no
    ``OPENROUTER_API_KEY``), or its call fails for any reason, the LLM
    classifier on *model_name* runs instead — the classification path's
    behaviour without a key is exactly what it was before the decisions
    classifier existed — and a failed decisions attempt's spend is
    included in the returned usage.

    Args:
        task: The task prompt about to be handed to the Sorcar agent.
        model_name: The model the Sorcar agent will run with; the LLM
            classifier uses the same one.
        model_config: The model configuration the Sorcar agent will run
            with (custom endpoint, headers), forwarded to the LLM
            classifier on a copy that imposes its own output cap.

    Returns:
        A :class:`ClassifierRun` whose ``classification`` is ``None``
        on any failure, and whose usage fields always report what the
        attempt(s) actually spent so callers can fold it into the
        task's totals.  A verdict served from the cache reports zero
        usage and zero steps; a decisions verdict reports its cost and
        tokens and zero steps (it is not an LLM step).
    """
    task = _truncate_task(task)
    if not decisions_classification_enabled():
        return _classify_with_llm(task, model_name, model_config)
    decided = _classify_with_decisions(task)
    if decided.classification is not None:
        return decided
    llm = _classify_with_llm(task, model_name, model_config)
    return ClassifierRun(
        classification=llm.classification,
        budget_used=llm.budget_used + decided.budget_used,
        tokens_used=llm.tokens_used + decided.tokens_used,
        steps=llm.steps,
    )


def _classify_with_decisions(task: str) -> ClassifierRun:
    """Classify the (truncated) *task* with the decisions classifier.

    Args:
        task: The task text, already cut by :func:`_truncate_task`.

    Returns:
        The memoised verdict at zero usage when there is one, else the
        outcome of one :func:`_attempt_decisions_classification`; a
        fresh verdict is memoised under the decisions criteria.
    """
    cached = _cached_decision(task)
    if cached is not None:
        logger.info(
            "Task classification served from cache: is_simple=%s "
            "is_development=%s",
            cached.is_simple,
            cached.is_development,
        )
        return ClassifierRun(
            classification=cached, budget_used=0.0, tokens_used=0, steps=0,
        )
    classification, budget, tokens = _attempt_decisions_classification(task)
    if classification is not None:
        remember_classification(
            task,
            DEFAULT_DECISIONS_MODEL,
            _decisions_model_config(),
            classification,
            _DECISIONS_CRITERIA,
        )
    return ClassifierRun(
        classification=classification, budget_used=budget, tokens_used=tokens, steps=0,
    )


def _classify_with_llm(
    task: str,
    model_name: str,
    model_config: dict[str, Any] | None,
) -> ClassifierRun:
    """Classify the (truncated) *task* with one non-agentic KISSAgent call.

    The first (and normally only) generation carries the provider's
    structured-output enforcement of :data:`_VERDICT_JSON_SCHEMA`; if
    that attempt fails — an endpoint that rejects the structured-output
    parameters, or output the schema could not save — one plain attempt
    with the prompt alone recovers.

    Args:
        task: The task text, already cut by :func:`_truncate_task`.
        model_name: The classifying model (the run's own).
        model_config: The run's model configuration, or ``None``.

    Returns:
        The :class:`ClassifierRun`; see :func:`classify_task`.
    """
    if _model_runs_task_to_completion(model_name):
        # cc/* and codex/* models are full coding agents with native
        # host tools: KISSAgent hands them the whole prompt in one CLI
        # invocation, so such a model could actually EXECUTE the
        # embedded task (editing files outside any worktree) instead of
        # classifying it.  Classification is skipped for them entirely.
        logger.info(
            "Skipping task classification: %s runs tasks to completion",
            model_name,
        )
        return ClassifierRun(
            classification=None, budget_used=0.0, tokens_used=0, steps=0,
        )
    cached = cached_classification(task, model_name, model_config)
    if cached is not None:
        logger.info(
            "Task classification served from cache: is_simple=%s "
            "is_development=%s",
            cached.is_simple,
            cached.is_development,
        )
        return ClassifierRun(
            classification=cached, budget_used=0.0, tokens_used=0, steps=0,
        )
    base_config = _classifier_model_config(model_config)
    structured_config = _structured_output_config(base_config, model_name)
    attempts = [structured_config] if structured_config is not None else []
    attempts.append(base_config)

    classification: TaskClassification | None = None
    budget_used = 0.0
    tokens_used = 0
    steps = 0
    for attempt_config in attempts:
        classification, budget, tokens, attempt_steps = (
            _attempt_classification(task, model_name, attempt_config)
        )
        budget_used += budget
        tokens_used += tokens
        steps += attempt_steps
        if classification is not None:
            break
    if classification is not None:
        remember_classification(task, model_name, model_config, classification)
    return ClassifierRun(
        classification=classification,
        budget_used=budget_used,
        tokens_used=tokens_used,
        steps=steps,
    )
