# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``BaseSea`` (``base_sea.py``) is the root layer of every run.

A plain chat run, a ``/xxx`` SEA run and the daemon-launched side
channels (task update, merge resolver) go through the methods of
:class:`kiss.agents.seas.base.base_sea.BaseSea` first, so editing that
one file customizes every run.  The tests
stand in for such an edit by rebinding the methods on the class for
their duration (what a daemon restart after editing the file does)
and drive the real pipeline: the daemon's ``run`` command through
:class:`DaemonRunApiHarness` (only the executor's LLM loop is swapped
for a recorder), and the launcher's :func:`apply_sea` /
:func:`evaluate_sea` directly.
"""

from __future__ import annotations

import subprocess
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.base.base_sea import (
    ALLOW,
    SUMMARY_DUE_REFUSAL,
    SUMMARY_EVERY_STEPS,
    BaseSea,
    Verdict,
    refuse,
)
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.run_config import PROVENANCE_EXPLICIT
from kiss.agents.sorcar.sea_apply import RUN_CONFIG_FIELD, apply_sea, load_layers
from kiss.agents.sorcar.sea_commands import (
    SeaError,
    base_prompt,
    base_system_prompt,
    base_tool_call_hook,
    evaluate_sea,
    sea_layers,
    sea_name,
)
from kiss.server import sorcar
from kiss.server.merge_conflict_resolver import run_merge_sea
from kiss.server.task_update import run_task_update_sea
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve, tool_call_body
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness

HOUSE_RULE = "\n\nHOUSE RULE: answer in British English."

SEA_WITH_RULE = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return system_prompt + "\\n\\nSEA RULE"

    def prompt(self, task):
        return f"[sea] {task}"
"""


def house_tool(text: str) -> str:
    """Return *text* unchanged (a tool added by the customized base)."""
    return text


def _custom_system_prompt(self: BaseSea, system_prompt: str) -> str:
    return system_prompt + HOUSE_RULE


def _custom_tools(self: BaseSea, tools: list[Any]) -> list[Any]:
    return tools + [house_tool]


def _custom_tool_call_hook(self: BaseSea, name: str, args: dict[str, Any]) -> Verdict:
    if name == "task_context" or (name == "Bash" and "rm -rf" in str(args.get("command", ""))):
        return refuse("Blocked by base")
    return ALLOW


def test_tool_call_hook_fold_stops_at_the_first_refusing_verdict() -> None:
    """``base_tool_call_hook`` returns the first refusing ``Verdict`` of the chain, else ``ALLOW``.

    Only a ``Verdict`` is accepted: the former ``None`` (allow) and
    string (refuse) spellings, ``"OK"`` included, are script errors.
    """

    class Allows(BaseSea):
        def tool_call_hook(self, name: str, args: dict[str, Any]) -> Verdict:
            return ALLOW

    class FormerAllows(BaseSea):
        def tool_call_hook(self, name: str, args: dict[str, Any]) -> Any:
            return "OK"

    class FormerAllowsNone(BaseSea):
        def tool_call_hook(self, name: str, args: dict[str, Any]) -> Any:
            return None

    class Refuses(BaseSea):
        def tool_call_hook(self, name: str, args: dict[str, Any]) -> Verdict:
            return refuse("ok") if name == "Bash" else ALLOW

    class Broken(BaseSea):
        def tool_call_hook(self, name: str, args: dict[str, Any]) -> Any:
            return 1

    assert base_tool_call_hook([Allows(), Refuses()], "Bash", {}) == refuse("ok")
    assert base_tool_call_hook([Refuses(), Allows()], "Read", {}) == ALLOW
    assert base_tool_call_hook([Refuses(), Allows()], "Read", {}).allowed
    for broken in (FormerAllows(), FormerAllowsNone(), Broken()):
        with pytest.raises(SeaError, match=r"tool_call_hook\(\) of SEA .* must return a Verdict"):
            base_tool_call_hook([Allows(), broken], "Bash", {})
    with pytest.raises(ValueError, match="non-blank"):
        refuse("  ")


def _custom_llm_call_hook(self: BaseSea, new_messages: list[Any]) -> list[Any]:
    return list(reversed(new_messages))


def _custom_settings(self: BaseSea, settings: dict[str, Any]) -> dict[str, Any]:
    return settings | {"use_memory": True}


def _custom_prompt(self: BaseSea, task: str) -> str:
    return f"{task} (be brief)"


_CUSTOM = {
    "system_prompt": _custom_system_prompt,
    "tools": _custom_tools,
    "tool_call_hook": _custom_tool_call_hook,
    "llm_call_hook": _custom_llm_call_hook,
}


@pytest.fixture
def customized_base() -> Iterator[None]:
    """Rebind the four hook methods of ``BaseSea`` for the test, as an edit of the file would."""
    originals = {name: vars(BaseSea)[name] for name in _CUSTOM}
    for name, method in _CUSTOM.items():
        setattr(BaseSea, name, method)
    try:
        yield
    finally:
        for name, method in originals.items():
            setattr(BaseSea, name, method)


class BaseSeaRootLayerDaemonTest(DaemonRunApiHarness):
    """The daemon applies the customized ``BaseSea`` to a run that names no SEA."""

    def test_plain_run_goes_through_the_customized_base(self) -> None:
        originals = {name: vars(BaseSea)[name] for name in _CUSTOM}
        for name, method in _CUSTOM.items():
            setattr(BaseSea, name, method)
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        try:
            result = sorcar.run(
                "plain task",
                work_dir=self.repo,
                use_worktree=False,
                endpoint_file=self.endpoint_file,
                timeout=60,
            )
            assert result.success is True, result.text
            executor_calls = [c for c in calls if "task_description" in c["arguments"]]
            assert len(executor_calls) == 1, calls
            call = executor_calls[0]
            assert call["system_prompt"].count(HOUSE_RULE) == 1, call["system_prompt"]
            assert "house_tool" in call["tool_names"]
            # The hooks look the base's methods up when called, so they
            # are exercised while the customization is still in place.
            assert call["tool_call_hook"]("Bash", {"command": "rm -rf /"}) == refuse(
                "Blocked by base"
            )
            assert call["tool_call_hook"]("Bash", {"command": "ls"}) == ALLOW
            assert call["llm_call_hook"]([1, 2, 3]) == [3, 2, 1]
        finally:
            for name, method in originals.items():
                setattr(BaseSea, name, method)

    def test_plain_run_with_the_stock_base_passes_identity_hooks(self) -> None:
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "plain task",
            work_dir=self.repo,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True, result.text
        executor_calls = [c for c in calls if "task_description" in c["arguments"]]
        assert len(executor_calls) == 1, calls
        call = executor_calls[0]
        assert call["llm_call_hook"]([1, 2]) == [1, 2]
        assert call["tool_call_hook"]("Bash", {"command": "ls"}) == ALLOW
        assert "house_tool" not in call["tool_names"]
        assert HOUSE_RULE not in call["system_prompt"]
        # The run's toolset holds ``summary``, so the stock base's
        # cadence rule is armed; ``finish`` is never held up by it.
        assert "summary" in call["tool_names"]
        assert call["tool_call_hook"]("finish", {}) == ALLOW
        assert call["tool_call_hook"]("finish", {"success": True}) == ALLOW
        assert call["tool_call_hook"]("summary", {"description": "- did x"}) == ALLOW
        assert call["tool_call_hook"]("finish", {"success": True}) == ALLOW


def summary(description: str) -> str:
    """Stand in for the run's ``summary`` tool (the root layer reads only its name)."""
    return description


def test_stock_base_requires_a_summary_at_every_tenth_step() -> None:
    """At a 10th step every tool but ``summary`` and ``finish`` waits for a summary.

    ``finish`` is exempt at every step: a one-step run (or the agent's
    implicit-finish probe) must be able to end without a summary first.
    """
    sea = BaseSea()
    sea.tools([summary, house_tool])
    assert sea.has_summary_tool
    due = SUMMARY_DUE_REFUSAL.format(step=10, every=SUMMARY_EVERY_STEPS, name="Bash")
    # Nothing ran yet: both the probe and a real finish pass.
    assert base_tool_call_hook([sea], "finish", {}) == ALLOW
    assert base_tool_call_hook([sea], "finish", {"success": True}) == ALLOW
    for _ in range(9):
        sea.llm_call_hook([])
        assert base_tool_call_hook([sea], "Bash", {"command": "ls"}) == ALLOW
    sea.llm_call_hook([])
    assert sea.step == 10 and sea.summary_due
    assert base_tool_call_hook([sea], "Bash", {"command": "ls"}) == refuse(due)
    assert base_tool_call_hook([sea], "finish", {"success": True}) == ALLOW
    assert base_tool_call_hook([sea], "finish", {}) == ALLOW
    assert base_tool_call_hook([sea], "summary", {"description": "- ran ls"}) == ALLOW
    assert not sea.summary_due
    assert base_tool_call_hook([sea], "Bash", {"command": "ls"}) == ALLOW
    assert base_tool_call_hook([sea], "finish", {"success": True}) == ALLOW
    assert base_tool_call_hook([sea], "finish", {"success": True, "is_continue": True}) == ALLOW
    # A toolset without ``summary`` (tool_profile "none", a SEA that
    # drops it) is under no rule at all; ``tools`` restarts the count.
    sea.tools([house_tool])
    assert not sea.has_summary_tool and sea.step == 0
    for _ in range(10):
        sea.llm_call_hook([])
    assert base_tool_call_hook([sea], "Bash", {"command": "ls"}) == ALLOW
    assert base_tool_call_hook([sea], "finish", {"success": True}) == ALLOW


def test_a_plain_command_runs_the_bare_base_and_names_no_sea() -> None:
    layers = load_layers({"prompt": "p"})
    assert [type(layer) for layer in layers] == [BaseSea]
    assert layers[0].path == Path(sea_commands.__file__).parents[1] / "seas/base/base_sea.py"
    assert sea_name(layers) == ""
    cmd: dict[str, Any] = {"prompt": "p", "useMemory": False}
    assert apply_sea(cmd) == set()
    assert cmd.pop(RUN_CONFIG_FIELD) == {"sea": "", "channel": False, "pinned": {}}
    assert cmd["prompt"] == "p" and cmd["useMemory"] is False
    assert cmd["systemPromptHook"]("X") == "X"
    assert cmd["toolsHook"]([house_tool]) == [house_tool]
    assert cmd["llmCallHook"]([1]) == [1]
    assert cmd["toolCallHook"]("Bash", {}) == ALLOW
    # ``defines`` still asks what a SEA adds on top of the base.
    assert sea_commands.defines(layers, "system_prompt") is False


def test_customized_base_applies_to_a_plain_command(customized_base: None) -> None:
    cmd: dict[str, Any] = {"prompt": "p"}
    assert apply_sea(cmd) == set()
    assert cmd["systemPromptHook"]("X") == "X" + HOUSE_RULE
    assert cmd["toolsHook"]([]) == [house_tool]
    assert cmd["llmCallHook"]([1, 2]) == [2, 1]
    assert cmd["toolCallHook"]("Bash", {"command": "rm -rf x"}) == refuse(
        "Blocked by base"
    )


def test_customized_base_runs_first_under_a_sea(customized_base: None, tmp_path: Path) -> None:
    sea = tmp_path / "rule_sea.py"
    sea.write_text(textwrap.dedent(SEA_WITH_RULE))
    cmd: dict[str, Any] = {"seaPath": str(sea), "prompt": "do it"}
    assert apply_sea(cmd) == {"prompt"}
    assert cmd["prompt"] == "[sea] do it"
    assert cmd["systemPromptHook"]("X") == "X" + HOUSE_RULE + "\n\nSEA RULE"
    assert cmd["toolsHook"]([]) == [house_tool]
    assert cmd[RUN_CONFIG_FIELD]["sea"] == "rule"
    # ``/rule check`` reports the SEA's own methods, not the base's.
    layers = sea_layers(sea)
    assert sea_commands.defines(layers, "system_prompt") is True
    assert sea_commands.defines(layers, "tools") is False


def test_system_prompt_is_folded_like_prompt(customized_base: None, tmp_path: Path) -> None:
    # Each layer's return is the next layer's input and the last return
    # is the run's prompt: no append/replace inference, no deduplication
    # (a sub-agent re-runs its own layers instead of inheriting the text).
    assert base_system_prompt([BaseSea()], "X") == "X" + HOUSE_RULE
    assert base_system_prompt([BaseSea()], "X" + HOUSE_RULE) == "X" + HOUSE_RULE + HOUSE_RULE
    sea = tmp_path / "rule_sea.py"
    sea.write_text(textwrap.dedent(SEA_WITH_RULE))
    assert base_system_prompt(sea_layers(sea), "X") == "X" + HOUSE_RULE + "\n\nSEA RULE"
    replacing = tmp_path / "replace_sea.py"
    replacing.write_text(textwrap.dedent("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return "ONLY THIS"
"""))
    assert base_system_prompt(sea_layers(replacing), "X") == "ONLY THIS"
    assert base_system_prompt(sea_layers(replacing), "ONLY THIS") == "ONLY THIS"


def test_base_settings_pin_a_plain_run_unless_the_caller_chose_explicitly() -> None:
    original = vars(BaseSea)["settings"]
    BaseSea.settings = _custom_settings  # type: ignore[method-assign]
    try:
        persisted: dict[str, Any] = {"prompt": "p", "useMemory": False}
        assert apply_sea(persisted) == {"useMemory"}
        assert persisted["useMemory"] is True
        assert persisted[RUN_CONFIG_FIELD]["pinned"] == {"use_memory": [False, True]}
        explicit: dict[str, Any] = {
            "prompt": "p", "useMemory": False,
            "provenance": {"use_memory": PROVENANCE_EXPLICIT},
        }
        assert apply_sea(explicit) == set()
        assert explicit["useMemory"] is False
    finally:
        BaseSea.settings = original  # type: ignore[method-assign]


def test_base_prompt_rewrites_every_task_and_an_identity_passes_an_empty_one() -> None:
    assert base_prompt([BaseSea()], "") == ""  # the stock base leaves an empty task alone
    # ``{task_id}`` is filled in whether or not a method changed the text:
    # the caller's own placeholder gets the id too (``""`` without one).
    assert base_prompt([BaseSea()], "about {task_id}", "T-1") == "about T-1"
    assert base_prompt([BaseSea()], "about {task_id}") == "about "
    original = vars(BaseSea)["prompt"]
    BaseSea.prompt = _custom_prompt  # type: ignore[method-assign]
    try:
        cmd: dict[str, Any] = {"prompt": "p"}
        assert apply_sea(cmd) == {"prompt"}
        assert cmd["prompt"] == "p (be brief)"
        run = evaluate_sea([BaseSea()], "t {task_id}", task_id="T-1")
        assert run.prompt == "t T-1 (be brief)"
    finally:
        BaseSea.prompt = original  # type: ignore[method-assign]
    BaseSea.prompt = lambda self, task: ""  # type: ignore[method-assign]
    try:
        with pytest.raises(SeaError, match="must return a non-empty string"):
            base_prompt([BaseSea()], "t")
    finally:
        BaseSea.prompt = original  # type: ignore[method-assign]


def _tool_results(requests: list[dict[str, Any]]) -> list[str]:
    """Return the ``tool`` message contents of the scripted model's last agentic request."""
    last = [request for request in requests if request.get("tools")][-1]
    return [str(m["content"]) for m in last["messages"] if m["role"] == "tool"]


def test_task_update_side_channel_goes_through_the_base(
    customized_base: None, tmp_path: Path,
) -> None:
    # The task-update child (the /ask SEA asked about the parent) is
    # launched by the daemon itself, not through a ``run`` command; its
    # tool calls still pass the base's ``tool_call_hook``, whose refusal
    # reaches the model as the tool's result.
    parent = ChatSorcarAgent("task-update-parent")
    parent.work_dir = str(tmp_path)
    task_id, chat_id = _add_task(
        "Parent task prompt", chat_id="", extra={"model": MODEL, "work_dir": str(tmp_path)},
    )
    parent.resume_chat_by_id(chat_id)
    with parent._task_id_lock:
        parent._last_task_id = task_id
    script = [
        tool_call_body("task_context", {"task_id": task_id}, prompt_tokens=500),
        finish_body("<p>report</p>", prompt_tokens=800),
    ]
    with serve(script) as (url, requests):
        parent.model_name = MODEL
        parent.model_config = {"base_url": url, "api_key": "local"}
        text, _cost = run_task_update_sea(parent, task_id)
    assert text == "<p>report</p>"
    results = _tool_results(requests)
    assert len(results) == 1 and "Blocked by base" in results[0], results
    # The ask SEA's ``system_prompt`` replaces the whole prompt, so the
    # base's appended rule is (correctly) gone from this child.
    system = next(m for m in requests[0]["messages"] if m["role"] == "system")
    assert HOUSE_RULE not in str(system["content"])


def test_merge_resolver_side_channel_goes_through_the_base(
    customized_base: None, tmp_path: Path,
) -> None:
    # The merge-conflict resolver (the /merge SEA) is launched by the
    # daemon after a conflicted auto-merge; it gets the base's tools and
    # hooks like every other run.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    parent = ChatSorcarAgent("merge-parent")
    parent.work_dir = str(tmp_path)
    script = [
        tool_call_body("Bash", {"command": "rm -rf build"}, prompt_tokens=500),
        tool_call_body("house_tool", {"text": "echo"}, prompt_tokens=600),
        finish_body("<p>merged</p>", prompt_tokens=800),
    ]
    with serve(script) as (url, requests):
        parent.model_name = MODEL
        parent.model_config = {"base_url": url, "api_key": "local"}
        run_merge_sea(parent, "resolve f.txt", tmp_path)
    results = _tool_results(requests)
    assert len(results) == 2, results
    assert "Blocked by base" in results[0]
    assert results[1].startswith("echo")  # the base's tool ran (a step footer follows)
    names = {t["function"]["name"] for t in requests[0]["tools"]}
    assert "house_tool" in names
    # The merge SEA's ``system_prompt`` replaces the whole prompt, so the
    # base's appended rule is (correctly) gone from this child.
    system = next(m for m in requests[0]["messages"] if m["role"] == "system")
    assert HOUSE_RULE not in str(system["content"])
