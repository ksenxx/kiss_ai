# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pre-run task classification for Sorcar agents.

Before a Sorcar agent starts a task, :func:`classify_task` runs a
lightweight :class:`~kiss.core.kiss_agent.KISSAgent` on the SAME model
to answer two questions about the task:

- ``is_simple``: the task involves neither software development nor
  Internet search.  A simple task runs with the lite system prompt
  (``SYSTEM_LITE.md``) instead of the full ``SYSTEM.md``.
- ``is_development``: the task is a software development task that
  requires creating or editing files.  The verdict becomes the run's
  effective ``is_worktree`` value (worktree isolation on/off) without
  ever touching the persisted ``is_worktree`` setting.

The classifier is as close to a non-agentic run as the framework
allows: :meth:`KISSAgent.run` refuses tools in non-agentic mode, and
the classification contract REQUIRES a ``finish`` tool carrying the
two boolean arguments — so the classifier is a tool-forced single-step
run whose only registered tool is :func:`finish`.  The model is
instructed to make exactly one call, and nothing else is possible: no
other tool exists.

Classification is best-effort and optional.  It runs only when
:func:`classification_enabled` says so (the ``classify_tasks`` config
key, toggleable in the settings panel, and the
``KISS_DISABLE_TASK_CLASSIFIER`` environment kill switch the test
suite sets).  Any failure — model error, budget, a run-to-completion
CLI model that never calls KISS tools, an unparseable verdict — yields
``classification=None`` and the agent behaves exactly as it did
without a classifier.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from kiss.core.kiss_agent import KISSAgent
from kiss.core.utils import _coerce_bool

logger = logging.getLogger(__name__)

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"

# One generation is enough for a model that follows the instruction; a
# second step absorbs a single text-only or malformed turn (the
# framework's "your response MUST have a function call" nudge) before
# the run is abandoned.  Kept minimal on purpose: the classification is
# meant to be a single non-agentic-style call, not an agentic loop.
CLASSIFIER_MAX_STEPS = 2

# Hard spend cap for the classification call.  The classifier reads the
# task and emits one tool call, so a dollar is already generous.  This
# allowance is on top of the main run's ``max_budget``: the spend is
# folded into the run's totals only after the run ends (see
# ``SorcarAgent._fold_classifier_usage``), so a run can exceed its cap
# by at most this amount.
CLASSIFIER_MAX_BUDGET = 1.0

_CLASSIFIER_PROMPT_PREFIX = (
    "You are a task classifier. Read the task below and classify it by "
    "calling the `finish` tool exactly once, immediately, with two "
    "boolean arguments:\n"
    "- is_simple: True if the task involves NEITHER software development "
    "NOR searching the internet. False otherwise.\n"
    "- is_development: True if the task is a software development task "
    "that requires creating or editing files. False otherwise.\n\n"
    "If the task is ambiguous, or is a follow-up that continues earlier "
    "work you cannot see (e.g. 'continue', 'fix it', 'do the same for "
    "the rest'), be conservative: classify it with is_simple=False and "
    "is_development=True.\n\n"
    "Do NOT attempt to perform the task. Do NOT reply with prose. Your "
    "first and only action MUST be the single `finish` tool call.\n\n"
    "# Task\n"
)


@dataclass(frozen=True)
class TaskClassification:
    """The classifier's verdict about one task.

    Attributes:
        is_simple: The task involves neither software development nor
            Internet search.
        is_development: The task is a software development task that
            requires creating or editing files.
    """

    is_simple: bool
    is_development: bool


@dataclass(frozen=True)
class ClassifierRun:
    """Outcome and usage of one classification attempt.

    Attributes:
        classification: The parsed verdict, or ``None`` when the
            attempt failed for any reason.
        budget_used: USD spent by the classification call.
        tokens_used: Total tokens consumed by the classification call.
        steps: LLM steps the classification call took.
    """

    classification: TaskClassification | None
    budget_used: float
    tokens_used: int
    steps: int


def classification_enabled() -> bool:
    """Whether pre-run task classification should run.

    Returns:
        ``False`` when the ``KISS_DISABLE_TASK_CLASSIFIER`` environment
        variable is set to ``"1"`` (the test suite's kill switch) or
        when the ``classify_tasks`` config key — persisted in
        ``~/.kiss/config.json`` and toggleable in the settings panel —
        is off; ``True`` otherwise (the default).
    """
    if os.environ.get(_DISABLE_ENV, "") == "1":
        return False
    from kiss.core.vscode_config import load_config

    try:
        return bool(load_config().get("classify_tasks", True))
    except Exception:  # pragma: no cover — unreadable config
        logger.debug("Could not read classify_tasks", exc_info=True)
        return True


def finish(is_simple: bool, is_development: bool) -> str:
    """Record the task classification verdict.

    The classifier agent must call this tool exactly once, immediately,
    with its verdict about the task.

    Args:
        is_simple: True if the task involves neither software
            development nor searching the internet.
        is_development: True if the task is a software development task
            that requires creating or editing files.

    Returns:
        str: The verdict as a JSON object string.
    """
    return json.dumps(
        {
            "is_simple": _coerce_bool(is_simple),
            "is_development": _coerce_bool(is_development),
        }
    )


def _parse_verdict(result: str) -> TaskClassification | None:
    """Parse the classifier run's result string into a verdict.

    Args:
        result: The string :meth:`KISSAgent.run` returned — the JSON
            object :func:`finish` produced on success, or arbitrary
            model text when the model never called the tool.

    Returns:
        The parsed :class:`TaskClassification`, or ``None`` when
        *result* is not a JSON object holding both keys.
    """
    try:
        verdict = json.loads(result)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(verdict, dict)
        or "is_simple" not in verdict
        or "is_development" not in verdict
    ):
        return None
    return TaskClassification(
        is_simple=_coerce_bool(verdict["is_simple"]),
        is_development=_coerce_bool(verdict["is_development"]),
    )


def classify_task(
    task: str,
    model_name: str,
    model_config: dict[str, Any] | None = None,
) -> ClassifierRun:
    """Classify *task* with a single-step KISSAgent run on *model_name*.

    Args:
        task: The task prompt about to be handed to the Sorcar agent.
        model_name: The model the Sorcar agent will run with; the
            classifier uses the same one.
        model_config: The model configuration the Sorcar agent will run
            with (custom endpoint, headers), forwarded unchanged.

    Returns:
        A :class:`ClassifierRun` whose ``classification`` is ``None``
        on any failure, and whose usage fields always report what the
        attempt actually spent so callers can fold it into the task's
        totals.
    """
    from kiss.core.models.model_info import model_runs_task_to_completion

    try:
        runs_to_completion = model_runs_task_to_completion(model_name)
    except Exception:
        runs_to_completion = False
    if runs_to_completion:
        # cc/* and codex/* models are full coding agents with native
        # host tools: KISSAgent hands them the whole prompt in one CLI
        # invocation and never exposes the ``finish`` tool.  Such a
        # model could actually EXECUTE the embedded task (editing
        # files outside any worktree) instead of classifying it, so
        # classification is skipped for them entirely.
        logger.info(
            "Skipping task classification: %s runs tasks to completion",
            model_name,
        )
        return ClassifierRun(
            classification=None, budget_used=0.0, tokens_used=0, steps=0,
        )
    agent = KISSAgent("Task Classifier")
    result = ""
    try:
        result = agent.run(
            model_name=model_name,
            prompt_template=_CLASSIFIER_PROMPT_PREFIX + task,
            tools=[finish],
            max_steps=CLASSIFIER_MAX_STEPS,
            max_budget=CLASSIFIER_MAX_BUDGET,
            model_config=model_config,
            verbose=False,
            print_prompts=False,
        )
    except Exception:
        logger.warning(
            "Task classification failed; running with defaults",
            exc_info=True,
        )
    classification = _parse_verdict(result)
    if classification is None and result:
        logger.warning(
            "Task classifier returned an unparseable verdict: %.200s",
            result,
        )
    return ClassifierRun(
        classification=classification,
        budget_used=float(getattr(agent, "budget_used", 0.0) or 0.0),
        tokens_used=int(getattr(agent, "total_tokens_used", 0) or 0),
        steps=int(getattr(agent, "step_count", 0) or 0),
    )
