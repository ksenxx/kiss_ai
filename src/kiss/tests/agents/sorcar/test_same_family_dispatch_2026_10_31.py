# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Inside a task, a sub-agent on the caller's own model and effort is refused unless the
work is parallel, and a review is sent to another family for a second opinion.

The rule (:func:`~kiss.agents.sorcar.fanout_guard.same_model_conflict`, wired into
:func:`~kiss.agents.sorcar.agent_dispatch.plan_run_agent`): a child on the caller's
model at the caller's effort is the same model with less context.  It starts only when
the call is a ``run_parallel`` over disjoint named pieces of work, the caller's own task
asks for sub-agents, another model or effort is named (whatever its vendor), or the
child is a review -- which then runs on ``other_family_model(parent)`` and says so on
its ``ran:`` line.

The refusal, the fan-out over distinct files, the shared-file refusal, the parent
task that asks for sub-agents, the named budget capped at the caller's remainder
and the review that has to stay on its family run END TO END: a real local daemon
runs a parent whose scripted stand-in model calls ``run_agent`` / ``run_parallel``;
the children are real daemon sub-tasks on the same stand-in.  The re-route itself,
an explicitly pinned other-family model and a pinned same-family reviewer are
asserted on the plan ``plan_run_agent`` returns:
running such a child would send it to the real vendor of the other family (the
caller's ``model_config`` is never inherited by another model), which no test may do.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import agent_dispatch
from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar.agent_dispatch import (
    NO_OTHER_FAMILY_NOTE,
    PIN_TASK_MODEL_ENV,
    SECOND_OPINION_NOTE,
    plan_run_agent,
    run_agents_parallel,
    sibling_plan,
)
from kiss.agents.sorcar.cron_agent import CHAT_TASK_HEADING
from kiss.agents.sorcar.daemon_client import TaskResult
from kiss.agents.sorcar.fanout_guard import named_pieces, same_model_conflict, undivided_work
from kiss.core import config as config_module
from kiss.core.models.model_families import family_of, other_family_model
from kiss.core.models.model_info import delete_custom_model, save_custom_model
from kiss.server import sorcar
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    OfflineFastModel,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)

WAYS_OUT = (
    "Ways out: do the work inline in this task; or name the disjoint files/commands in each "
    "task of a run_parallel; or pass another model or effort (model="
)
PARENT_MARK = "PARENT-TASK"


# --- the daemon ---------------------------------------------------------------------


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    home = IsolatedKissHome(prefix="kiss-same-family-dispatch-")
    home.write_config(classify_tasks=False)
    try:
        yield home
    finally:
        home.cleanup()


@pytest.fixture
def daemon(env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A real local daemon; yields its endpoint file, which the parent's dispatches use."""
    endpoint_file = str(env.tmpdir / "sorcar-local.json")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", endpoint_file)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server = RemoteAccessServer(local_endpoint_file=endpoint_file, work_dir=str(env.repo))
    asyncio.run_coroutine_threadsafe(server.start_private_async(), loop).result(timeout=30)
    try:
        yield endpoint_file
    finally:
        asyncio.run_coroutine_threadsafe(server.stop_async(), loop).result(timeout=15)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


class _ScriptedParent:
    """The stand-in model of a parent and its children.

    The parent's first step is *call* (a ``run_agent`` or ``run_parallel``
    tool call), its second step finishes; a request whose current task
    (the text after the last chat task heading: a child's prompt carries
    the parent's task as chat history) lacks the parent marker is a
    child's, which finishes at once with ``done-<marker>``.
    """

    def __init__(self, call: dict[str, Any], child_markers: tuple[str, ...]) -> None:
        self.call = call
        self.child_markers = child_markers
        self.parent_requests: list[dict[str, Any]] = []
        self.child_requests: list[dict[str, Any]] = []
        self.children: list[str] = []

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        current_task = text.rsplit(CHAT_TASK_HEADING, 1)[-1]
        if PARENT_MARK not in current_task:
            marker = next((m for m in self.child_markers if m in current_task), "KID-?")
            self.children.append(marker)
            self.child_requests.append(request)
            return finish_response(f"done-{marker}")
        self.parent_requests.append(request)
        if len(self.parent_requests) == 1:
            return self.call
        return finish_response("parent-done")

    def tool_result(self) -> str:
        """The text the parent read after its tool call (its second request), one-spaced.

        A result is YAML, whose long ``ran:`` scalar the dumper folds at
        80 columns; collapsing white space makes it one line again.
        """
        assert len(self.parent_requests) >= 2, self.parent_requests
        return " ".join(request_text(self.parent_requests[1]).split())


def _run_parent(
    env: IsolatedKissHome, daemon: str, parent_task: str, script: _ScriptedParent,
) -> TaskResult:
    """Run *parent_task* on the daemon with *script* as its model; return the result."""
    model = StandInModelServer(script)
    try:
        return sorcar.run(
            parent_task,
            work_dir=str(env.repo),
            model=STANDIN_MODEL,
            model_config=model.model_config,
            max_budget=8.0,
            endpoint_file=daemon,
            timeout=300,
        )
    finally:
        model.stop()


def _child_rows(env: IsolatedKissHome) -> list[str]:
    """The task texts of every persisted sub-task of the parent."""
    with sqlite3.connect(env.kiss_home / "history.db") as db:
        rows = [
            (str(a), str(b or ""), str(c))
            for a, b, c in db.execute("SELECT id, parent_task_id, task FROM task_history")
        ]
    parents = [row for row in rows if PARENT_MARK in row[2]]
    assert len(parents) == 1, rows
    return sorted(row[2] for row in rows if row[1] == parents[0][0])


def test_same_family_run_agent_without_named_pieces_is_refused(
    env: IsolatedKissHome, daemon: str,
) -> None:
    """One child on the caller's family, no pieces, no ask: refused before anything starts."""
    script = _ScriptedParent(
        tool_call_response("run_agent", {"task": "KID-1 say done", "timeout": "120"}),
        ("KID-1",),
    )
    result = _run_parent(env, daemon, f"{PARENT_MARK} do the work", script)
    assert result.success is True, result
    text = script.tool_result()
    assert (
        f"Error: sorcar: a sub-agent on {STANDIN_MODEL} is this task's own model at its own "
        "effort, the same model with less context, and one sub-task "
        "is no fan-out over disjoint work. " + WAYS_OUT
    ) in text, text[-1500:]
    assert "ran:" not in text
    assert script.children == []
    assert _child_rows(env) == []


def test_fan_out_over_distinct_files_runs_on_the_callers_family(
    env: IsolatedKissHome, daemon: str,
) -> None:
    """Two tasks naming different files are parallel work: both children run, no note."""
    tasks = '["KID-1 edit src/a.py then say done", "KID-2 edit src/b.py then say done"]'
    script = _ScriptedParent(
        tool_call_response("run_parallel", {"tasks": tasks}), ("KID-1", "KID-2"),
    )
    result = _run_parent(env, daemon, f"{PARENT_MARK} do the work", script)
    assert result.success is True, result
    text = script.tool_result()
    assert sorted(script.children) == ["KID-1", "KID-2"], text[-1500:]
    assert text.count(f"ran: sorcar model={STANDIN_MODEL} tools=") == 2, text[-1500:]
    assert "another family" not in text and "no other family" not in text
    assert "done-KID-1" in text and "done-KID-2" in text
    assert [row[:5] for row in _child_rows(env)] == ["KID-1", "KID-2"]


def test_fan_out_naming_the_same_file_is_refused(env: IsolatedKissHome, daemon: str) -> None:
    """Two tasks that both name src/a.py are not disjoint: nothing starts."""
    tasks = '["KID-1 edit src/a.py", "KID-2 edit src/a.py and src/b.py"]'
    script = _ScriptedParent(
        tool_call_response("run_parallel", {"tasks": tasks}), ("KID-1", "KID-2"),
    )
    result = _run_parent(env, daemon, f"{PARENT_MARK} do the work", script)
    assert result.success is True, result
    text = script.tool_result()
    assert "and tasks 1 and 2 of the fan-out both name src/a.py. " + WAYS_OUT in text, text[-1500:]
    assert script.children == [] and _child_rows(env) == []


def test_parent_task_asking_for_sub_agents_may_use_its_own_family(
    env: IsolatedKissHome, daemon: str,
) -> None:
    """The user asked for sub-agents: pieceless same-family children run."""
    tasks = '["KID-1 say done", "KID-2 say done"]'
    script = _ScriptedParent(
        tool_call_response("run_parallel", {"tasks": tasks}), ("KID-1", "KID-2"),
    )
    result = _run_parent(env, daemon, f"{PARENT_MARK} use two sub-agents for this", script)
    assert result.success is True, result
    assert sorted(script.children) == ["KID-1", "KID-2"]
    assert script.tool_result().count(f"ran: sorcar model={STANDIN_MODEL} tools=") == 2


def test_named_budget_above_the_callers_remainder_is_capped_at_it(
    env: IsolatedKissHome, daemon: str,
) -> None:
    """An inheriting child whose call names $16 of a caller with $8 left runs with $8.

    The cap (``agent_dispatch.budget_cap_of``, applied by the daemon's
    ``sea_apply.cap_budget``) is the caller's remainder: the child's own
    Task Settings say $8.00, where the uncapped call would say $16.00 and
    the default share of one child $4.00.
    """
    script = _ScriptedParent(
        tool_call_response(
            "run_agent", {"task": "KID-1 say done", "max_budget": "16", "timeout": "120"},
        ),
        ("KID-1",),
    )
    result = _run_parent(env, daemon, f"{PARENT_MARK} use a sub-agent for this", script)
    assert result.success is True, result
    assert script.children == ["KID-1"]
    (child,) = script.child_requests
    text = request_text(child)
    assert text.count("Max budget (USD): $") == 1, text[:2000]
    assert "Max budget (USD): $8.00" in text, text[:2000]
    assert "done-KID-1" in script.tool_result()


def test_review_with_no_other_family_stays_on_the_callers_model_and_says_so(
    env: IsolatedKissHome, daemon: str,
) -> None:
    """No credential for another family: the review runs inherited; its ran: line says why."""
    script = _ScriptedParent(
        tool_call_response(
            "run_agent",
            {"task": "KID-1 review the diff for bugs, then say done", "timeout": "120"},
        ),
        ("KID-1",),
    )
    with OfflineFastModel():
        assert other_family_model(STANDIN_MODEL) is None
        result = _run_parent(env, daemon, f"{PARENT_MARK} do the work", script)
    assert result.success is True, result
    text = script.tool_result()
    assert script.children == ["KID-1"]
    assert f"ran: sorcar model={STANDIN_MODEL} {NO_OTHER_FAMILY_NOTE} tools=review" in text, (
        text[-1500:]
    )


# --- the plan -----------------------------------------------------------------------


class _Caller:
    """A calling agent on the stand-in model whose current task text is *task*."""

    work_dir = ""
    model_name = STANDIN_MODEL
    model_config: dict[str, Any] | None = None

    def __init__(self, task: str = "PARENT do the work") -> None:
        self.task_description = task


def _plan(task: str, caller: Any = None, **kwargs: Any) -> dict[str, Any] | str:
    return plan_run_agent("", task, "", parent_agent=caller or _Caller(), **kwargs)


def _only_another_family_configured() -> tuple[OfflineFastModel, str]:
    """Enter a context with one credential, for a family other than the stand-in's.

    Returns:
        The entered context (the caller exits it) and the second-opinion
        model it makes :func:`other_family_model` choose.
    """
    offline = OfflineFastModel()
    offline.__enter__()
    config_module.DEFAULT_CONFIG.ANTHROPIC_API_KEY = "kiss-test-key"
    chosen = other_family_model(STANDIN_MODEL)
    assert chosen and family_of(chosen) == "anthropic", chosen
    return offline, chosen


def test_review_on_the_inherited_family_is_rerouted_to_another_family() -> None:
    """A review the call did not pin runs on other_family_model(parent), with the note."""
    offline, chosen = _only_another_family_configured()
    try:
        for task, profile in (
            ("Review the diff for bugs", ""),  # inferred from the text
            ("Verify the listed changes", "review"),  # explicit profile
            ("Verify the listed changes", "readonly"),  # the profile's alias
        ):
            plan = _plan(task, tool_profile=profile)
            assert isinstance(plan, dict), plan
            assert plan["model_name"] == chosen, (task, profile, plan["model_name"])
            assert plan["model_note"] == SECOND_OPINION_NOTE
        # A pinned same-family reviewer is honoured, without a note.
        plan = _plan("Review the diff for bugs", model="gpt-4o")
        assert isinstance(plan, dict), plan
        assert plan["model_name"] == "gpt-4o" and plan["model_note"] == ""
        # A reviewer pinned to another family by the call needs no re-route.
        plan = _plan("Review the diff for bugs", model=chosen)
        assert isinstance(plan, dict), plan
        assert plan["model_name"] == chosen and plan["model_note"] == ""
    finally:
        offline.__exit__(None, None, None)


def test_a_fan_out_is_judged_task_by_task_from_one_plan() -> None:
    """``run_parallel`` plans once; each other task re-makes the prompt and the verdict."""
    offline, chosen = _only_another_family_configured()
    try:
        tasks = ["Verify src/a.py", "Verify src/b.py"]
        first = _plan(tasks[0], tool_profile="review", sibling_tasks=tasks)
        assert isinstance(first, dict), first
        second = sibling_plan(first, tasks[1], "", _Caller(), tasks)
        assert isinstance(second, dict), second
        assert tasks[1] in second["prompt"] and tasks[0] not in second["prompt"]
        assert second["model_name"] == chosen and second["model_note"] == SECOND_OPINION_NOTE
        assert {k: v for k, v in second.items() if k != "prompt"} == {
            k: v for k, v in first.items() if k != "prompt"
        }
        # A review sibling may start, an implementation sibling of the same
        # call may not: the call reports the refusal and starts nothing.
        mixed = ["Review the diff for bugs", "Implement the feature"]
        first = _plan(mixed[0], sibling_tasks=mixed)
        assert isinstance(first, dict), first
        refused = sibling_plan(first, mixed[1], "", _Caller(), mixed)
        assert isinstance(refused, str) and "names no file" in refused, refused
        assert run_agents_parallel("", mixed, parent_agent=_Caller()) == refused
    finally:
        offline.__exit__(None, None, None)


def test_sea_that_pins_its_model_is_a_pinned_reviewer(tmp_path: Path) -> None:
    """A SEA whose settings name a same-family model keeps it: the daemon applies that setting."""
    sea = tmp_path / "pinned_sea.py"
    sea.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n"
        "    def settings(self, settings):\n"
        "        return settings | {'model': 'gpt-4o'}\n",
        encoding="utf-8",
    )
    offline, _chosen = _only_another_family_configured()
    try:
        plan = plan_run_agent(str(tmp_path), "Review the diff for bugs", str(sea),
                              parent_agent=_Caller())
        assert isinstance(plan, dict), plan
        assert plan["model_name"] == "" and plan["model_note"] == ""
        # The same SEA running a non-review child: another model than the caller's may start,
        # the caller's own model at its own effort is the same model with less context.
        allowed = plan_run_agent(str(tmp_path), "probe the host", str(sea), parent_agent=_Caller())
        assert isinstance(allowed, dict), allowed
        caller = _Caller()
        caller.model_name = "gpt-4o"
        refused = plan_run_agent(str(tmp_path), "probe the host", str(sea), parent_agent=caller)
        assert isinstance(refused, str) and "a sub-agent on gpt-4o is this task's own" in refused
    finally:
        offline.__exit__(None, None, None)


def test_daemon_that_pins_its_model_keeps_the_review_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A benchmark daemon pins every task to its model: no re-route, and the note says so."""
    monkeypatch.setenv(PIN_TASK_MODEL_ENV, "1")
    offline, _chosen = _only_another_family_configured()
    try:
        plan = _plan("Review the diff for bugs")
        assert isinstance(plan, dict), plan
        assert plan["model_name"] == ""
        assert plan["model_note"] == (
            "(no other family configured: this daemon pins every task to its model)"
        )
    finally:
        offline.__exit__(None, None, None)


def test_daemon_that_pins_its_model_refuses_a_sea_that_names_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Under the pin a SEA whose settings name a model or a model configuration is refused.

    The call names neither, so the pin's check of the call's arguments passes;
    the daemon would then apply the SEA's ``model`` and the sub-task would run
    on it, off the pinned model.  The same SEA is admitted without the pin.
    """
    other = tmp_path / "other_model_sea.py"
    other.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n"
        "    def settings(self, settings):\n"
        "        return settings | {'model': 'gpt-4o'}\n",
        encoding="utf-8",
    )
    configured = tmp_path / "configured_sea.py"
    configured.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n"
        "    def settings(self, settings):\n"
        "        return settings | {'model_config': {'temperature': 0}}\n",
        encoding="utf-8",
    )
    same = tmp_path / "same_model_sea.py"
    same.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n"
        "    def settings(self, settings):\n"
        f"        return settings | {{'model': {STANDIN_MODEL!r}}}\n",
        encoding="utf-8",
    )
    caller = _Caller("PARENT do the work with sub-agents")  # the family rule lets these pass
    monkeypatch.setenv(PIN_TASK_MODEL_ENV, "1")
    refused = plan_run_agent(str(tmp_path), "do the thing", str(other), parent_agent=caller)
    assert isinstance(refused, str) and "pins every task to its model" in refused, refused
    assert "'gpt-4o'" in refused, refused
    refused = plan_run_agent(str(tmp_path), "do the thing", str(configured), parent_agent=caller)
    assert isinstance(refused, str) and "model_config" in refused, refused
    # A SEA that names the pinned model itself moves nothing and is admitted.
    plan = plan_run_agent(str(tmp_path), "do the thing", str(same), parent_agent=caller)
    assert isinstance(plan, dict), plan
    # A standalone call (no calling task) is never pinned.
    plan = plan_run_agent(str(tmp_path), "do the thing", str(other))
    assert isinstance(plan, dict), plan
    monkeypatch.setenv(PIN_TASK_MODEL_ENV, "")
    plan = plan_run_agent(str(tmp_path), "do the thing", str(other), parent_agent=caller)
    assert isinstance(plan, dict), plan


def test_another_model_or_effort_is_allowed_without_pieces() -> None:
    """The guard compares model and effort, not vendor: only the caller's own pair is refused."""
    for other in ("claude-opus-5-5", "gpt-6-astra", "openrouter/openai/gpt-6-astra"):
        plan = _plan("probe the host", model=other)  # another model, of any family
        assert isinstance(plan, dict), plan
        assert plan["model_name"] == other and plan["model_note"] == ""
    # The caller's own model through another route is still the caller's own model.
    refused = _plan("probe the host", model=f"openrouter/openai/{STANDIN_MODEL}")
    assert isinstance(refused, str), refused
    assert (
        f"a sub-agent on openrouter/openai/{STANDIN_MODEL} is this task's own model at its "
        "own effort, the same model with less context"
    ) in refused
    # The launch the family rule refused 29 times on 2026-11-11: Opus medium -> Fable max.
    caller = _Caller()
    caller.model_name = "claude-opus-5-5-medium"
    plan = _plan("probe the host", caller=caller, model="claude-fable-5-2-max")
    assert isinstance(plan, dict), plan
    # One model at two efforts is two workers; a bare name runs at its catalog effort (high).
    caller.model_name = "claude-fable-5-2-max"
    assert isinstance(_plan("probe the host", caller=caller, model="claude-fable-5-2-medium"), dict)
    assert isinstance(_plan("probe the host", caller=caller, model="claude-fable-5-2"), dict)
    for same in ("claude-fable-5-2-max", "openrouter/anthropic/claude-fable-5.2-max"):
        assert isinstance(_plan("probe the host", caller=caller, model=same), str), same
    caller.model_name = "claude-fable-5-2-high"
    assert isinstance(_plan("probe the host", caller=caller, model="claude-fable-5-2"), str)
    # A name outside the catalog carries its effort in its suffix only.
    caller.model_name = "codex/gpt-6-astra-xhigh"
    assert isinstance(_plan("probe the host", caller=caller, model="codex/gpt-6-astra"), dict)
    assert isinstance(_plan("probe the host", caller=caller, model="codex/gpt-6-astra-xhigh"), str)


def test_the_effort_a_model_configuration_asks_for_outranks_the_catalogs(tmp_path: Path) -> None:
    """A ``reasoning_effort`` in the model configuration is the effort the guard compares.

    The caller's configuration is its own; the child's is the call's ``model_config``
    option, else the SEA's setting, else the caller's when the child inherits it.
    """
    probe = "probe the host"
    at_high = '{"model_config": {"reasoning_effort": "high"}}'
    at_max = '{"model_config": {"reasoning_effort": "max"}}'
    caller = _Caller()
    caller.model_name = "claude-fable-5-2"  # the catalog runs it at high
    assert isinstance(_plan(probe, caller=caller), str)
    # The same name at another effort is another worker; the max alias at high is the caller.
    assert isinstance(_plan(probe, caller=caller, options=at_max), dict)
    fable_max = "claude-fable-5-2-max"
    assert isinstance(_plan(probe, caller=caller, model=fable_max, options=at_high), str)
    # A caller its configuration runs at max: the max alias is its own model and effort, and so
    # is the child that inherits the configuration; one the call configures itself is not.
    caller.model_config = {"reasoning_effort": "max"}
    assert isinstance(_plan(probe, caller=caller, model=fable_max), str)
    assert isinstance(_plan(probe, caller=caller), str)
    assert isinstance(_plan(probe, caller=caller, options='{"model_config": {}}'), dict)
    tasks = [probe, "probe the guest"]
    first = _plan(tasks[0], caller=caller, options=at_high, sibling_tasks=tasks)
    assert isinstance(first, dict), first
    assert isinstance(sibling_plan(first, tasks[1], "", caller, tasks), dict)
    # A SEA's model_config outranks the inherited one, and the call's outranks the SEA's.
    sea = tmp_path / "high_effort_sea.py"
    sea.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n"
        "    def settings(self, settings):\n"
        "        return settings | {'model_config': {'reasoning_effort': 'high'}}\n",
        encoding="utf-8",
    )
    assert isinstance(plan_run_agent(str(tmp_path), probe, str(sea), parent_agent=caller), dict)
    refused = plan_run_agent(str(tmp_path), probe, str(sea), parent_agent=caller, options=at_max)
    assert isinstance(refused, str) and "is this task's own model at its own effort" in refused


def test_a_bedrock_id_is_the_claude_model_behind_it() -> None:
    """A Bedrock model or inference-profile id is one more route to its Claude model."""
    probe = "probe the host"
    us = "bedrock/us.anthropic.claude-opus-4-6-v1:0"
    world = "bedrock/global.anthropic.claude-opus-4-6-v1:0"
    caller = _Caller()
    caller.model_name = "claude-opus-4-6"  # the catalog runs it at high
    for same in (us, world, f"{us}-high"):
        refused = _plan(probe, caller=caller, model=same)
        assert isinstance(refused, str), same
        assert f"a sub-agent on {same} is this task's own model at its own effort" in refused
    for other in (f"{us}-medium", "bedrock/global.anthropic.claude-opus-4-7"):
        assert isinstance(_plan(probe, caller=caller, model=other), dict), other
    caller.model_name = us  # two profiles of one model
    assert isinstance(_plan(probe, caller=caller, model=world), str)
    # A Claude id the catalog does not describe is one model on every profile too; an id
    # that is no Claude id stays a name of its own.
    caller.model_name = "bedrock/us.anthropic.claude-zzz-9-v1:0"
    other_profile = "bedrock/global.anthropic.claude-zzz-9-v1:0"
    assert isinstance(_plan(probe, caller=caller, model=other_profile), str)
    assert isinstance(_plan(probe, caller=caller, model="bedrock/us.meta.llama9-v1:0"), dict)
    # A model the catalog lists under Bedrock alone is priced as that entry, and is the
    # Claude model behind it all the same.
    caller.model_name = "claude-sonnet-4-20250514"
    sonnet = "bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0"
    assert isinstance(_plan(probe, caller=caller, model=sonnet), str)


def test_a_provider_prefix_comes_off_before_the_effort_alias_is_read() -> None:
    """``anthropic/claude-opus-5-5-max`` is ``claude-opus-5-5-max``, as the factory runs it."""
    probe = "probe the host"
    caller = _Caller()
    caller.model_name = "claude-opus-5-5-max"
    assert isinstance(_plan(probe, caller=caller, model="anthropic/claude-opus-5-5-max"), str)
    assert isinstance(_plan(probe, caller=caller, model="anthropic/claude-opus-5-5-medium"), dict)
    # A MY_MODELS.json model with an endpoint of its own is the name its gateway serves: the
    # same name is the bundled model before it is registered, and only itself afterwards.
    gateway = "anthropic/claude-opus-5-5"
    caller.model_name = "claude-opus-5-5-high"
    assert isinstance(_plan(probe, caller=caller, model=gateway), str)
    assert save_custom_model(gateway, endpoint="http://127.0.0.1:9/v1") is None
    try:
        assert isinstance(_plan(probe, caller=caller, model=gateway), dict)
        caller.model_name = gateway
        assert isinstance(_plan(probe, caller=caller), str)
    finally:
        assert delete_custom_model(gateway) is None


def test_refusal_covers_every_shortfall_and_names_the_pin_to_pass() -> None:
    """One task, a pieceless task of a fan-out, a shared piece; the pin is a name to pass.

    The strongest model of another family, else the caller's own model at another effort,
    else a strong model of its own family: the first the rule lets through as the
    sub-task would run it.
    """
    offline, chosen = _only_another_family_configured()
    try:
        refused = _plan("probe the host")
        assert isinstance(refused, str), refused
        assert refused.endswith(f'{WAYS_OUT}"{chosen}").'), refused
        assert "and one sub-task is no fan-out over disjoint work." in refused
        tasks = ["KID-1 edit src/a.py", "KID-2 say done"]
        refused = _plan(tasks[0], sibling_tasks=tasks)
        assert isinstance(refused, str), refused
        assert (
            "and task 2 of the fan-out names no file, directory, module, URL or `command` "
            "of its own."
        ) in refused
        tasks = ["edit src/a.py and kiss.core", "edit src/b.py", "edit kiss.core"]
        refused = _plan(tasks[0], sibling_tasks=tasks)
        assert isinstance(refused, str), refused
        assert "and tasks 1 and 3 of the fan-out both name kiss.core." in refused
        allowed = _plan(tasks[0], sibling_tasks=tasks[:2])
        assert isinstance(allowed, dict), allowed
        assert allowed["model_name"] == "" and allowed["model_note"] == ""
        # The one credential is the caller's own vendor's: the pin is its model at the nearest
        # effort below its own, else the nearest above, and it passes when passed.
        caller = _Caller()
        for model, config, pin in (
            ("claude-fable-5-2", None, "claude-fable-5-2-medium"),
            ("claude-fable-5-2-max", None, "claude-fable-5-2-xhigh"),
            ("claude-fable-5-2-low", None, "claude-fable-5-2-medium"),
            ("claude-fable-5-2", {"reasoning_effort": "turbo"}, "claude-fable-5-2-max"),
            ("claude-fable-5-2-xhigh", {"reasoning_effort": "max"}, "claude-fable-5-2-high"),
        ):
            caller.model_name, caller.model_config = model, config
            refused = _plan("probe the host", caller=caller)
            assert isinstance(refused, str) and refused.endswith(f'{WAYS_OUT}"{pin}").'), refused
            assert isinstance(_plan("probe the host", caller=caller, model=pin), dict), pin
        # The sub-task's own configuration sets the effort, which outranks an alias's: no
        # alias is a way out, another model of the family is, with that configuration.
        caller.model_name, caller.model_config = "claude-fable-5-2", None
        at_high = '{"model_config": {"reasoning_effort": "high"}}'
        refused = _plan("probe the host", caller=caller, options=at_high)
        assert isinstance(refused, str), refused
        assert refused.endswith(f'{WAYS_OUT}"claude-opus-5-5").'), refused
        alias = _plan("probe the host", caller=caller, model="claude-fable-5-2-medium",
                      options=at_high)
        assert isinstance(alias, str), alias
        pinned = _plan("probe the host", caller=caller, model="claude-opus-5-5", options=at_high)
        assert isinstance(pinned, dict), pinned
    finally:
        offline.__exit__(None, None, None)
    with OfflineFastModel():  # a model without effort aliases, its vendor's the one credential
        config_module.DEFAULT_CONFIG.OPENAI_API_KEY = "kiss-test-key"
        caller.model_name, caller.model_config = "gpt-4o", None
        refused = _plan("probe the host", caller=caller)
        assert isinstance(refused, str) and refused.endswith(f'{WAYS_OUT}"gpt-6-astra").'), refused
        assert isinstance(_plan("probe the host", caller=caller, model="gpt-6-astra"), dict)
    with OfflineFastModel():  # the caller's own name inherits its effort: no way out, not named
        config_module.DEFAULT_CONFIG.GEMINI_API_KEY = "kiss-test-key"
        caller.model_name = "gemini-3.1-pro-preview"
        caller.model_config = {"reasoning_effort": "low"}
        refused = _plan("probe the host", caller=caller)
        assert isinstance(refused, str), refused
        assert refused.endswith("or pass another model or effort."), refused
        assert _plan("probe the host", caller=caller, model=caller.model_name) == refused
    with OfflineFastModel():  # no credential at all: the third way out has no name to give
        for model in (STANDIN_MODEL, "claude-fable-5-2"):
            caller.model_name, caller.model_config = model, None
            refused = _plan("probe the host", caller=caller)
            assert isinstance(refused, str), refused
            assert refused.endswith("or pass another model or effort."), refused


def test_review_that_asks_for_changes_is_implementation_work_and_is_refused() -> None:
    """"Review ... and fix" keeps the full toolset, so it is no review: the rule applies."""
    refused = _plan("Review the diff and fix every bug you find")
    assert isinstance(refused, str) and WAYS_OUT in refused, refused


def test_only_the_current_chat_task_counts_as_the_parents_ask() -> None:
    """An earlier task of the chat that asked for sub-agents does not license a later one."""
    history = f"use sub-agents here\n\n{CHAT_TASK_HEADING} (work on it now)\n\nprobe the host"
    refused = _plan("say done", caller=_Caller(history))
    assert isinstance(refused, str) and WAYS_OUT in refused, refused
    allowed = _plan("say done", caller=_Caller(f"{history} with a second opinion"))
    assert isinstance(allowed, dict), allowed
    for ask in ("Subagents", "a sub-task", "subtasks", "run_parallel", "run_agent",
                "parallel agents", "adversarial testing", "adversarial training", "a reviewer"):
        assert isinstance(_plan("say done", caller=_Caller(f"do it with {ask}")), dict), ask


def test_children_that_inherit_nothing_are_left_alone(env: IsolatedKissHome) -> None:
    """inherit: false (as a channel) takes no model from the caller: nothing to compare.

    Nor does a sub-task on another machine: that is work the caller cannot
    do inline here (the ``/sync`` agent syncs every registered machine
    through one), so neither the refusal nor the second opinion applies.
    """
    plan = _plan("say done", options='{"inherit": false}')
    assert isinstance(plan, dict), plan
    assert plan["inherit"] is False and plan["model_name"] == "" and plan["model_note"] == ""
    mach.save_machines([mach.Machine(name="box", url="https://203.0.113.7:8787", password="x")])
    for task in ("say done", "Review the diff for bugs"):
        plan = _plan(task, machine="box")
        assert isinstance(plan, dict), plan
        assert plan["machine"] == "box" and plan["model_name"] == "" and plan["model_note"] == ""
    # A caller without a model (standalone use) is not compared either.
    plan = plan_run_agent("", "say done", "")
    assert isinstance(plan, dict), plan
    assert same_model_conflict("", STANDIN_MODEL, ["x"], "", False) == ""
    assert same_model_conflict(STANDIN_MODEL, "", ["x"], "", False) == ""


def test_the_ran_line_carries_the_note_after_the_model() -> None:
    """format_dispatch_result puts the note right after model=<name>; nothing else moves."""
    result = TaskResult(
        text="ok", success=True, cost=0.0, tokens=0, steps=0,
        settings={"model": "claude-opus-5-5", "tool_profile": "review", "max_budget": 5},
    )
    plain = yaml.safe_load(agent_dispatch.format_dispatch_result(result, 60.0))
    noted = yaml.safe_load(
        agent_dispatch.format_dispatch_result(result, 60.0, model_note=SECOND_OPINION_NOTE),
    )
    assert f"model=claude-opus-5-5 {SECOND_OPINION_NOTE} tools=review" in noted["ran"], noted
    assert noted["ran"].replace(f" {SECOND_OPINION_NOTE}", "") == plain["ran"]
    assert {k: v for k, v in noted.items() if k != "ran"} == {"success": True, "summary": "ok"}
    assert agent_dispatch.format_dispatch_result("Error: x", 60.0, model_note="(n)") == "Error: x"


# --- the pieces ---------------------------------------------------------------------


def test_named_pieces_finds_paths_modules_urls_and_commands() -> None:
    assert named_pieces(
        "Edit src/kiss/agents/sorcar/fanout_guard.py and/or fanout_guard.py; see e.g. "
        "kiss.agents.sorcar (Python 3.12, gpt-4.1, v1.2.3, I/O, yes/no, 50/50)."
    ) == {"src/kiss/agents/sorcar/fanout_guard.py", "fanout_guard.py", "kiss.agents.sorcar"}
    assert named_pieces(
        "Run `uv run pytest -q tests/x.py` then open https://example.com/a/b. and ./tmp/, "
        "also /etc/hosts, ~/.kiss/config.json, tests/agents/ and README.md."
    ) == {
        "uv run pytest -q tests/x.py", "tests/x.py", "https://example.com/a/b", "./tmp/",
        "/etc/hosts", "~/.kiss/config.json", "tests/agents/", "README.md",
    }
    assert named_pieces("KID-1 say done") == set()
    assert named_pieces("i.e. a.m. `` ` `") == set()
    # An abbreviation that opens a sentence is prose too.
    assert named_pieces("E.g. do it. I.e. at 5 P.M. or 9 A.M.") == set()


def test_undivided_work_wants_two_tasks_each_with_a_piece_and_no_shared_piece() -> None:
    assert undivided_work(["fix a.py"]) == "one sub-task is no fan-out over disjoint work"
    assert undivided_work(["fix a.py", "fix b.py"]) == ""
    assert undivided_work(["Fix a.py. E.g. rename x.", "Fix b.py. E.g. rename y."]) == ""
    assert undivided_work(["fix a.py", "fix a.py and b.py"]) == (
        "tasks 1 and 2 of the fan-out both name a.py"
    )
    assert undivided_work(["fix a.py", "nothing here"]) == (
        "task 2 of the fan-out names no file, directory, module, URL or `command` of its own"
    )


def test_undivided_work_reports_every_overlap_and_ignores_shared_instructions() -> None:
    """Audit 2026-11-01 (task 4aa14638): a 24-task fan-out was refused four times, one
    overlap per refusal (a shared instructions file, then a shared module, then a shared
    test directory), the 15 KB task list re-emitted each time."""
    # every overlapping pair in one clause, a few shared pieces each
    clause = undivided_work(["fix a.py and b.py", "fix a.py and c.py", "fix b.py, c.py and d.py"])
    assert clause == ("tasks 1 and 2 of the fan-out both name a.py; "
                      "tasks 1 and 3 of the fan-out both name b.py; "
                      "tasks 2 and 3 of the fan-out both name c.py")
    many = undivided_work(["fix a.py b.py c.py d.py e.py", "fix a.py b.py c.py d.py e.py f.py"])
    assert many == "tasks 1 and 2 of the fan-out both name a.py, b.py, c.py (+2 more)"
    # pieces every task of three or more names are shared instructions, not claims
    shared = "Read tmp/INSTRUCTIONS.md and README.md first, run `uv run pytest -q` after; "
    assert undivided_work([shared + "fix a.py", shared + "fix b.py", shared + "fix c.py"]) == ""
    # ...but with two tasks a shared README.md is an overlap (every task is both tasks)
    assert undivided_work(["Read README.md; fix a.py", "Read README.md; fix b.py"]) == (
        "tasks 1 and 2 of the fan-out both name README.md")
    # tmp/ paths never count, however many tasks name them
    assert undivided_work(["fix a.py, notes in tmp/x/notes.md", "fix b.py; log to ./tmp/x/log"
                           ", see /w/tmp/x/notes.md"]) == ""
    # nor do paths in a sub-agent's private folder, wherever the temp directory is
    plan = "/var/folders/T/kiss-sub-ab12cd34/plan.md"
    assert undivided_work([f"fix a.py as {plan} says", f"fix b.py as {plan} says"]) == ""
    assert undivided_work(["work in tmp/x/", "work in tmp/y/"]) == (
        "task 1 of the fan-out names no file, directory, module, URL or `command` of its own "
        "(pieces every task names, and tmp/ paths, do not count)")
    # a directory both tasks name is no overlap when each names its own file under it
    assert undivided_work(["tests in src/kiss/webapp/test/: fix src/kiss/webapp/test/a.js",
                           "tests in src/kiss/webapp/test/: fix src/kiss/webapp/test/b.js"]) == ""
    # one task on the whole directory, the other on a file in it: an overlap,
    # whether or not the second task repeats the directory (audit 2026-11-04)
    assert undivided_work(["refactor src/x/", "fix src/x/b.py in src/x/"]) == (
        "tasks 1 and 2 of the fan-out both name src/x/")
    assert undivided_work(["refactor src/x/", "fix src/x/b.py"]) == (
        "tasks 1 and 2 of the fan-out both name src/x/")
    assert undivided_work(["refactor src/x/", "refactor src/x/nested/"]) == (
        "tasks 1 and 2 of the fan-out both name src/x/")
    assert undivided_work(["refactor src/x/", "refactor src/x/"]) == (
        "tasks 1 and 2 of the fan-out both name src/x/")
    # ...while files of their own under a directory only one task names are disjoint
    assert undivided_work(["fix src/x/a.py in src/x/", "fix src/x/b.py"]) == ""
