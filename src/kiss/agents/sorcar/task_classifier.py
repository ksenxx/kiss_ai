# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pre-run task classification for Sorcar agents.

Before a Sorcar agent starts a task, :func:`classify_task` runs a
lightweight NON-AGENTIC :class:`~kiss.core.kiss_agent.KISSAgent` on the
SAME model to answer two questions about the task:

- ``is_simple``: the task involves neither software development nor
  Internet search.  A simple task runs with the lite system prompt
  (``SYSTEM_LITE.md``) instead of the full ``SYSTEM.md``.
- ``is_development``: the task is a software development task that
  requires creating or editing files.  Tasks that only request git
  operations (commit, merge, rebase, resolving merge conflicts, ...)
  are NOT development.  The verdict becomes the run's effective
  ``is_worktree`` value (worktree isolation on/off) without ever
  touching the persisted ``is_worktree`` setting.

The classification is a single non-agentic ``generate()`` call — no
tool loop, no tools — that returns the STRUCTURED verdict
``{"is_simple": <bool>, "is_development": <bool>}``.  The structure is
enforced twice over: the provider's own structured-output mechanism
pins :data:`_VERDICT_JSON_SCHEMA` on the request (``output_format`` on
Anthropic, ``response_json_schema`` on Gemini, ``response_format`` on
the OpenAI-compatible vendors), and the prompt demands the same bare
JSON object so the verdict survives even on an endpoint without schema
support.  Should a provider reject the structured-output parameters,
one plain retry (same prompt, no schema knobs) recovers.
:func:`_parse_verdict` extracts the object even when a lax model wraps
it in code fences or prose, and accepts only genuinely boolean values.

The call is kept fast on purpose: one generation, a hard
:data:`CLASSIFIER_MAX_TOKENS` output cap (which also keeps extended
thinking off on models where a thinking budget would not fit), thinking
explicitly disabled on Anthropic models, and a terse JSON-only answer.

Classification is best-effort and optional.  It runs only when
:func:`classification_enabled` says so (the ``classify_tasks`` config
key, toggleable in the settings panel, and the
``KISS_DISABLE_TASK_CLASSIFIER`` environment kill switch the test
suite sets).  Any failure — model error, budget, a run-to-completion
CLI model, an unparseable verdict — yields ``classification=None`` and
the agent behaves exactly as it did without a classifier.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from kiss.core.kiss_agent import KISSAgent

logger = logging.getLogger(__name__)

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"

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
                "True only if the task is a software development or research task "
                "that requires creating or editing files; git-only "
                "tasks are false."
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
    "- is_development: true ONLY if the task is a software development "
    "task that requires creating or editing files. false otherwise. A "
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
        is_development: The task is a software development task that
            requires creating or editing files (git-only tasks are not
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
            structured-output attempt needed the plain fallback).
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
        on the request — ``output_format`` for Anthropic (with thinking
        disabled: a one-line verdict needs no reasoning budget, and
        skipping it keeps the call fast), ``response_json_schema`` for
        Gemini, Chat-Completions ``response_format`` for the registered
        OpenAI-compatible vendors — or ``None`` when the provider has no
        structured-output mechanism the adapters can carry (the caller
        then relies on the prompt alone).
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
    if provider == "Anthropic":
        config["output_format"] = {
            "type": "json_schema",
            "schema": _VERDICT_JSON_SCHEMA,
        }
        config["thinking"] = {"type": "disabled"}
        return config
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


def classify_task(
    task: str,
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> ClassifierRun:
    """Classify *task* with one non-agentic KISSAgent call on *model_name*.

    The first (and normally only) generation carries the provider's
    structured-output enforcement of :data:`_VERDICT_JSON_SCHEMA`; if
    that attempt fails — an endpoint that rejects the structured-output
    parameters, or output the schema could not save — one plain attempt
    with the prompt alone recovers.

    Args:
        task: The task prompt about to be handed to the Sorcar agent.
        model_name: The model the Sorcar agent will run with; the
            classifier uses the same one.
        model_config: The model configuration the Sorcar agent will run
            with (custom endpoint, headers), forwarded on a copy that
            imposes the classifier's own output cap.

    Returns:
        A :class:`ClassifierRun` whose ``classification`` is ``None``
        on any failure, and whose usage fields always report what the
        attempt(s) actually spent so callers can fold it into the
        task's totals.
    """
    from kiss.core.models.model_info import model_runs_task_to_completion

    try:
        runs_to_completion = model_runs_task_to_completion(model_name)
    except Exception:
        runs_to_completion = False
    if runs_to_completion:
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
    if len(task) > CLASSIFIER_TASK_MAX_CHARS:
        # The gist suffices for classification; the full prompt belongs
        # to the main run.  Truncation bounds this call's input cost and
        # latency no matter how large the task is.
        task = (
            task[:CLASSIFIER_TASK_MAX_CHARS]
            + "\n... [task truncated for classification]"
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
    return ClassifierRun(
        classification=classification,
        budget_used=budget_used,
        tokens_used=tokens_used,
        steps=steps,
    )
