# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for pre-run task classification.

``kiss.agents.sorcar.task_classifier`` runs a single-step KISSAgent on
the run's own model BEFORE a Sorcar agent starts a task.  Its
``finish(is_simple, is_development)`` verdict selects the system prompt
(``SYSTEM_LITE.md`` for simple tasks, ``SYSTEM.md`` otherwise) and
decides worktree isolation for that run (``is_development`` becomes the
effective ``use_worktree``) without touching the persisted
``is_worktree`` setting.

The classification tests call real LLMs (a cheap model) — no mocks.
The wiring tests drive real ``SorcarAgent`` / ``WorktreeSorcarAgent``
runs inside an isolated KISS_HOME with a scratch git repo; where a full
agent run is unnecessary, the run is stopped early by a real server
printer whose task-allocation hook raises (the same technique as
``test_audit0903_worktree_run_wrapping``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.task_classifier import (
    TaskClassification,
    _parse_verdict,
    classification_enabled,
    classify_task,
    finish,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.tests.server.parallel_agent_harness import (
    CapturePrinter,
    IsolatedKissHome,
)

MODEL = "claude-haiku-4-5"

# Real-LLM tests follow the repo's live_api convention: marked, and
# skipped outright when the provider key is absent.
requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; classifier live tests need it",
)
live_api = pytest.mark.live_api

_SIMPLE_TASK = (
    "What is 2 + 2? Answer in text only. This does not involve any "
    "software development, any file edits, or any internet search."
)

_DEV_TASK = (
    "Software development task: edit the file src/utils.py in this "
    "repository to fix the off-by-one bug in the pagination helper, "
    "and update its unit test. This requires editing source files."
)

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"


class _RaisingPrinter(CapturePrinter):
    """A real server printer whose task-allocation hook raises.

    Stops an agent run right after classification and the worktree
    decision, so worktree-gating tests spend exactly one LLM call (the
    classifier's) instead of a whole agent run.
    """

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self._exc = exc

    def agent_task_allocated(
        self, agent: Any, task_id: Any, chat_id: str = ""
    ) -> None:
        super().agent_task_allocated(agent, task_id, chat_id)
        raise self._exc


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """Isolated KISS_HOME + scratch repo, with the classifier enabled.

    The root conftest disables the classifier suite-wide via
    ``KISS_DISABLE_TASK_CLASSIFIER=1``; these are the classifier's own
    tests, so the kill switch is lifted and restored afterwards.
    """
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "0"
    isolated = IsolatedKissHome("kiss-task-classifier-")
    try:
        yield isolated
    finally:
        if saved is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved
        isolated.cleanup()


# ---------------------------------------------------------------------------
# classify_task — real LLM
# ---------------------------------------------------------------------------


@live_api
@requires_anthropic
def test_classify_simple_task(env: IsolatedKissHome) -> None:
    """A pure-arithmetic task is simple and not development."""
    outcome = classify_task(task=_SIMPLE_TASK, model_name=MODEL)
    assert outcome.classification is not None
    assert outcome.classification.is_simple is True
    assert outcome.classification.is_development is False
    # The attempt's real usage is reported for budget folding.
    assert outcome.budget_used > 0.0
    assert outcome.tokens_used > 0
    assert outcome.steps >= 1


@live_api
@requires_anthropic
def test_classify_development_task(env: IsolatedKissHome) -> None:
    """A file-editing coding task is development and not simple."""
    outcome = classify_task(task=_DEV_TASK, model_name=MODEL)
    assert outcome.classification is not None
    assert outcome.classification.is_simple is False
    assert outcome.classification.is_development is True


def test_classify_task_bad_model_fails_soft() -> None:
    """An unknown model yields classification=None, not an exception."""
    outcome = classify_task(
        task="anything", model_name="no-such-model-xyz",
    )
    assert outcome.classification is None
    assert outcome.budget_used == 0.0


# ---------------------------------------------------------------------------
# Verdict parsing and the finish tool — no LLM
# ---------------------------------------------------------------------------


def test_finish_tool_normalizes_string_booleans() -> None:
    """String-typed booleans from lax models are interpreted by content."""
    assert json.loads(finish(True, False)) == {
        "is_simple": True,
        "is_development": False,
    }
    assert json.loads(finish("false", "TRUE")) == {  # type: ignore[arg-type]
        "is_simple": False,
        "is_development": True,
    }


def test_parse_verdict_accepts_finish_output() -> None:
    """The finish tool's JSON round-trips into a TaskClassification."""
    assert _parse_verdict(finish(True, False)) == TaskClassification(
        is_simple=True, is_development=False,
    )
    assert _parse_verdict(
        '{"is_simple": "yes", "is_development": "no"}'
    ) == TaskClassification(is_simple=True, is_development=False)


def test_parse_verdict_rejects_junk() -> None:
    """Prose, non-object JSON, and partial objects all yield None."""
    assert _parse_verdict("I think this task is simple.") is None
    assert _parse_verdict("[1, 2]") is None
    assert _parse_verdict('{"is_simple": true}') is None
    assert _parse_verdict("") is None


# ---------------------------------------------------------------------------
# classification_enabled — env kill switch and config key
# ---------------------------------------------------------------------------


def test_classification_enabled_env_kill_switch(
    env: IsolatedKissHome,
) -> None:
    """KISS_DISABLE_TASK_CLASSIFIER=1 wins over any config value."""
    env.write_config(classify_tasks=True)
    os.environ[_DISABLE_ENV] = "1"
    assert classification_enabled() is False


def test_classification_enabled_follows_config(
    env: IsolatedKissHome,
) -> None:
    """With the kill switch lifted, the classify_tasks key decides."""
    assert classification_enabled() is True  # default (key absent)
    env.write_config(classify_tasks=False)
    assert classification_enabled() is False
    env.write_config(classify_tasks=True)
    assert classification_enabled() is True


# ---------------------------------------------------------------------------
# SorcarAgent._classify_task_once — caching and the disabled path
# ---------------------------------------------------------------------------


def test_classify_once_returns_none_when_disabled(
    env: IsolatedKissHome,
) -> None:
    """A disabled classifier attempts nothing and reports no verdict."""
    env.write_config(classify_tasks=False)
    agent = SorcarAgent("clf-disabled")
    assert agent._classify_task_once(MODEL, _SIMPLE_TASK, None) is None
    assert agent._classification_attempted is True
    # The attempt is cached: later calls do not re-consult the config.
    env.write_config(classify_tasks=True)
    assert agent._classify_task_once(MODEL, _SIMPLE_TASK, None) is None


def test_classify_once_returns_cached_verdict(
    env: IsolatedKissHome,
) -> None:
    """A verdict already computed for this run is returned as-is."""
    agent = SorcarAgent("clf-cached")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(
        is_simple=False, is_development=True,
    )
    verdict = agent._classify_task_once(MODEL, _SIMPLE_TASK, None)
    assert verdict == TaskClassification(is_simple=False, is_development=True)


def test_fold_classifier_usage_banks_and_resets(
    env: IsolatedKissHome,
) -> None:
    """Classifier spend lands in the run totals exactly once."""
    agent = SorcarAgent("clf-fold")
    agent._classifier_budget_used = 0.25
    agent._classifier_tokens_used = 123
    agent._classifier_steps = 2
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(True, False)
    agent._fold_classifier_usage()
    assert agent.budget_used == pytest.approx(0.25)
    assert agent.total_tokens_used == 123
    assert agent.total_steps == 2
    assert agent._classifier_budget_used == 0.0
    assert agent._classifier_tokens_used == 0
    assert agent._classifier_steps == 0
    assert agent._classification_attempted is False
    assert agent._task_classification is None
    # Folding again is a no-op: the counters were reset.
    agent._fold_classifier_usage()
    assert agent.budget_used == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# System prompt selection — full SorcarAgent runs (real LLM)
# ---------------------------------------------------------------------------


def _run_minimal_sorcar(agent: SorcarAgent, task: str, **kwargs: Any) -> str:
    """Run *agent* on *task* with the smallest possible tool surface."""
    return agent.run(
        model_name=MODEL,
        prompt_template=task,
        append_basic_tools=False,
        web_tools=False,
        is_parallel=False,
        max_steps=8,
        max_budget=2.0,
        verbose=False,
        **kwargs,
    )


@live_api
@requires_anthropic
def test_simple_task_runs_with_lite_prompt(env: IsolatedKissHome) -> None:
    """A simple task's run uses SYSTEM_LITE.md as its base prompt."""
    agent = SorcarAgent("clf-lite-prompt")
    result = _run_minimal_sorcar(
        agent, _SIMPLE_TASK + " Call the finish tool with the answer.",
    )
    assert yaml.safe_load(result)["success"] is True
    assert agent.system_prompt.startswith(SYSTEM_PROMPT_LITE)
    assert not agent.system_prompt.startswith(SYSTEM_PROMPT)
    # The classifier's own spend was folded into the run totals.
    assert agent.budget_used > 0.0
    # Classification state was cleared for the next run.
    assert agent._classification_attempted is False
    assert agent._task_classification is None


@live_api
@requires_anthropic
def test_non_simple_verdict_keeps_full_prompt(
    env: IsolatedKissHome,
) -> None:
    """An is_simple=False verdict keeps the full SYSTEM.md prompt."""
    agent = SorcarAgent("clf-full-prompt")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(
        is_simple=False, is_development=True,
    )
    result = _run_minimal_sorcar(
        agent, "Call the finish tool with the word DONE.",
    )
    assert yaml.safe_load(result)["success"] is True
    assert agent.system_prompt.startswith(SYSTEM_PROMPT)


@live_api
@requires_anthropic
def test_disabled_classifier_keeps_full_prompt(
    env: IsolatedKissHome,
) -> None:
    """With classify_tasks off, every task keeps the full prompt."""
    env.write_config(classify_tasks=False)
    agent = SorcarAgent("clf-off-prompt")
    result = _run_minimal_sorcar(
        agent, _SIMPLE_TASK + " Call the finish tool with the answer.",
    )
    assert yaml.safe_load(result)["success"] is True
    assert agent.system_prompt.startswith(SYSTEM_PROMPT)


@live_api
@requires_anthropic
def test_base_system_prompt_override_beats_verdict(
    env: IsolatedKissHome,
) -> None:
    """A caller-supplied base prompt wins over the lite selection."""
    agent = SorcarAgent("clf-custom-prompt")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(
        is_simple=True, is_development=False,
    )
    custom = "You are a terse assistant. Use the finish tool to answer."
    result = _run_minimal_sorcar(
        agent,
        "Call the finish tool with the word DONE.",
        base_system_prompt=custom,
    )
    assert yaml.safe_load(result)["success"] is True
    assert agent.system_prompt.startswith(custom)


# ---------------------------------------------------------------------------
# Worktree gating — WorktreeSorcarAgent runs stopped after the decision
# ---------------------------------------------------------------------------


@live_api
@requires_anthropic
def test_non_development_task_skips_worktree(
    env: IsolatedKissHome,
) -> None:
    """is_development=False turns worktree isolation off for the run."""
    agent = WorktreeSorcarAgent("clf-wt-skip")
    printer = _RaisingPrinter(RuntimeError("stop-after-decision"))
    result = agent.run(
        prompt_template=_SIMPLE_TASK,
        model_name=MODEL,
        work_dir=str(env.repo),
        printer=printer,
    )
    payload = yaml.safe_load(result)
    assert payload["success"] is False
    assert "stop-after-decision" in payload["summary"]
    assert agent._wt is None
    assert printer.events_of_type("worktree_created") == []


@live_api
@requires_anthropic
def test_development_task_forces_worktree(env: IsolatedKissHome) -> None:
    """is_development=True turns worktree isolation on for the run,
    even when the caller asked for none."""
    agent = WorktreeSorcarAgent("clf-wt-force")
    printer = _RaisingPrinter(RuntimeError("stop-after-decision"))
    result = agent.run(
        prompt_template=_DEV_TASK,
        model_name=MODEL,
        use_worktree=False,
        work_dir=str(env.repo),
        printer=printer,
    )
    payload = yaml.safe_load(result)
    assert payload["success"] is False
    assert len(printer.events_of_type("worktree_created")) == 1
    assert agent._wt is not None
    # The persisted setting was never touched by the runtime override.
    from kiss.core.vscode_config import load_config

    assert load_config()["is_worktree"] is True
    agent.discard()
    assert agent._wt is None


@live_api
@requires_anthropic
def test_disabled_classifier_keeps_callers_worktree_choice(
    env: IsolatedKissHome,
) -> None:
    """With the classifier off, use_worktree=False is respected."""
    env.write_config(classify_tasks=False)
    agent = WorktreeSorcarAgent("clf-wt-off")
    printer = _RaisingPrinter(RuntimeError("stop-after-decision"))
    result = agent.run(
        prompt_template=_DEV_TASK,
        model_name=MODEL,
        use_worktree=False,
        work_dir=str(env.repo),
        printer=printer,
    )
    assert yaml.safe_load(result)["success"] is False
    assert agent._wt is None
    assert printer.events_of_type("worktree_created") == []


# ---------------------------------------------------------------------------
# Review-driven fixes: CLI models, pre-seeding, substitution, stale usage
# ---------------------------------------------------------------------------


def test_classify_task_skips_run_to_completion_models() -> None:
    """cc/* models could execute the embedded task with native tools,
    so classification is skipped for them without any spend."""
    outcome = classify_task(task=_DEV_TASK, model_name="cc/claude-opus-4-6")
    assert outcome.classification is None
    assert outcome.budget_used == 0.0
    assert outcome.tokens_used == 0
    assert outcome.steps == 0


def test_reset_drops_stale_unfolded_usage(env: IsolatedKissHome) -> None:
    """A crashed run's never-folded classifier spend is dropped, not
    misattributed to the next unrelated run."""
    agent = SorcarAgent("clf-stale-usage")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(True, False)
    agent._classifier_budget_used = 0.5
    agent._classifier_tokens_used = 42
    agent._classifier_steps = 1
    agent._reset_task_classification()
    assert agent._classification_attempted is False
    assert agent._task_classification is None
    assert agent._classifier_budget_used == 0.0
    assert agent._classifier_tokens_used == 0
    assert agent._classifier_steps == 0


def test_reset_keeps_preseeded_verdict(env: IsolatedKissHome) -> None:
    """A verdict pre-seeded by an external driver survives the
    per-run reset so every subtask of a submission reuses it."""
    agent = SorcarAgent("clf-preseed-keep")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(False, True)
    agent._classification_preseeded = True
    agent._reset_task_classification()
    assert agent._task_classification == TaskClassification(False, True)
    assert agent._classification_attempted is True


@live_api
@requires_anthropic
def test_classify_task_for_run_preseeds_worktree_decision(
    env: IsolatedKissHome,
) -> None:
    """The server-style pre-classification drives the run's worktree
    mode; the run reuses the seeded verdict instead of re-classifying
    the (contradictory) prompt it is given."""
    agent = WorktreeSorcarAgent("clf-preseed-run")
    verdict = agent.classify_task_for_run(model_name=MODEL, task=_DEV_TASK)
    assert verdict is not None
    assert verdict.is_development is True
    printer = _RaisingPrinter(RuntimeError("stop-after-decision"))
    # The run's own prompt is a NON-development task: if the run
    # re-classified it, no worktree would be created.  The seeded
    # development verdict must win.
    result = agent.run(
        prompt_template=_SIMPLE_TASK,
        model_name=MODEL,
        work_dir=str(env.repo),
        printer=printer,
    )
    assert yaml.safe_load(result)["success"] is False
    assert len(printer.events_of_type("worktree_created")) == 1
    assert agent._wt is not None
    agent.discard()
    # The next server-style classification replaces the seed.
    env.write_config(classify_tasks=False)
    assert agent.classify_task_for_run(model_name=MODEL, task=_DEV_TASK) is None
    assert agent._task_classification is None


@live_api
@requires_anthropic
def test_classifier_sees_substituted_template_arguments(
    env: IsolatedKissHome,
) -> None:
    """{placeholder} templates are substituted before classification."""
    agent = SorcarAgent("clf-substitution")
    verdict = agent._classify_task_once(
        MODEL, "{task}", None, arguments={"task": _DEV_TASK},
    )
    assert verdict is not None
    assert verdict.is_development is True
