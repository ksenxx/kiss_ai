# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.server.sorcar.run``'s append parameters.

Drive ``kiss.server.sorcar.run(append_to_system_prompt=...,
append_to_prompt=...)`` against a real daemon on a temporary
Unix-domain socket, with only the executor LLM stubbed (see
:class:`kiss.tests.server.test_append_basic_tools.DaemonRunApiHarness`).

Contract under test: ``append_to_system_prompt`` is appended to the
run's system prompt — right after the default ``SYSTEM.md`` prompt or
the ``system_prompt`` replacement — and ``append_to_prompt`` is
appended to the executed task prompt (to EACH subtask of a
multi-``<task>`` prompt).  Both default to ``""`` (append nothing),
are overridable by an agent script's ``get_append_to_system_prompt()``
/ ``get_append_to_prompt()`` getters, and are treated as untrusted
wire input by the daemon (non-string appends nothing).
"""

from __future__ import annotations

import unittest
from typing import Any, cast

from kiss.core.base import SYSTEM_PROMPT
from kiss.server import sorcar
from kiss.tests.server.test_append_basic_tools import (
    DaemonRunApiHarness,
)

_SYS_MARKER = "\n\nUNIQUE-APPENDED-SYSTEM-SUFFIX-9317"
_PROMPT_MARKER = "\n\nUNIQUE-APPENDED-PROMPT-SUFFIX-4620"


class AppendToPromptsApiTest(DaemonRunApiHarness):
    """Drive ``sorcar.run(append_to_*_prompt=...)`` against a real daemon."""

    def _executor_calls(
        self, calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return the task-executor ``KISSAgent`` calls, in order.

        Args:
            calls: The record list filled by the executor stub.

        Returns:
            The calls whose arguments carry ``task_description`` — the
            executor sessions ``RelentlessAgent.perform_task`` spawned,
            one per ``<task>`` subtask.
        """
        return [c for c in calls if "task_description" in c["arguments"]]

    def test_append_to_system_prompt_appended_after_default(self) -> None:
        """The suffix lands in the executor's system prompt, after the base.

        The executed system prompt is ``SYSTEM.md`` + suffix + the
        per-run operational instructions, so the suffix must appear
        exactly once, after the default prompt's content.  The task
        prompt must NOT carry it.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with a system prompt suffix",
            work_dir=self.repo,
            append_to_system_prompt=_SYS_MARKER,
            use_worktree=False,
            web_tools=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        (call,) = self._executor_calls(calls)
        sp = call["system_prompt"]
        assert sp.count(_SYS_MARKER) == 1
        # After the whole default prompt: SYSTEM.md's first line comes
        # before the suffix.
        assert sp.index(SYSTEM_PROMPT[:80]) < sp.index(_SYS_MARKER)
        assert _SYS_MARKER not in call["arguments"]["task_description"]

    def test_append_to_prompt_appended(self) -> None:
        """The suffix lands in the executed task prompt, not the system prompt."""
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with a prompt suffix",
            work_dir=self.repo,
            append_to_prompt=_PROMPT_MARKER,
            use_worktree=False,
            web_tools=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        (call,) = self._executor_calls(calls)
        task_description = call["arguments"]["task_description"]
        assert task_description.count(_PROMPT_MARKER) == 1
        assert task_description.index("task with a prompt suffix") < (
            task_description.index(_PROMPT_MARKER)
        )
        assert _PROMPT_MARKER not in call["system_prompt"]

    def test_defaults_append_nothing(self) -> None:
        """Without the parameters, neither prompt carries the markers."""
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task without suffixes",
            work_dir=self.repo,
            use_worktree=False,
            web_tools=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        (call,) = self._executor_calls(calls)
        assert _SYS_MARKER not in call["system_prompt"]
        assert _PROMPT_MARKER not in call["arguments"]["task_description"]

    def test_append_to_prompt_reaches_every_subtask(self) -> None:
        """A multi-``<task>`` prompt gets the suffix on EACH subtask."""
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "<task>first subtask body</task><task>second subtask body</task>",
            work_dir=self.repo,
            append_to_prompt=_PROMPT_MARKER,
            use_worktree=False,
            web_tools=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        executor_calls = self._executor_calls(calls)
        assert len(executor_calls) == 2, calls
        for call, body in zip(
            executor_calls,
            ("first subtask body", "second subtask body"),
            strict=True,
        ):
            # The chat context of a later subtask quotes the earlier
            # subtask's (already suffixed) prompt, so assert on the
            # executed prompt's tail rather than a global count.
            task_description = call["arguments"]["task_description"]
            assert task_description.endswith(body + _PROMPT_MARKER)

    def test_append_to_system_prompt_after_custom_base(self) -> None:
        """With ``system_prompt`` set, the suffix follows the replacement."""
        custom_base = "CUSTOM-BASE-SYSTEM-PROMPT-7805: call finish when done."
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with custom base and suffix",
            work_dir=self.repo,
            system_prompt=custom_base,
            append_to_system_prompt=_SYS_MARKER,
            use_worktree=False,
            web_tools=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        (call,) = self._executor_calls(calls)
        sp = call["system_prompt"]
        assert sp.index(custom_base) < sp.index(_SYS_MARKER)
        assert SYSTEM_PROMPT[:80] not in sp

    def test_agent_script_getters_override(self) -> None:
        """Script ``get_append_to_*()`` getters override the client values."""
        agent_path = self._write_py(
            "append_prompts_agent.py",
            f'''
            """Agent script appending to both prompts."""


            def get_append_to_system_prompt() -> str:
                """Append a system prompt suffix."""
                return {_SYS_MARKER!r}


            def get_append_to_prompt() -> str:
                """Append a prompt suffix."""
                return {_PROMPT_MARKER!r}
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "script appends to both prompts",
            work_dir=self.repo,
            extension_agent_path=agent_path,
            use_worktree=False,
            web_tools=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        (call,) = self._executor_calls(calls)
        assert _SYS_MARKER in call["system_prompt"]
        assert _PROMPT_MARKER in call["arguments"]["task_description"]

    def test_agent_script_getter_wrong_type_fails_task(self) -> None:
        """A non-string ``get_append_to_prompt()`` stops the task loudly."""
        agent_path = self._write_py(
            "bad_append_prompt_agent.py",
            '''
            """Agent script with a wrong-typed getter."""


            def get_append_to_prompt() -> int:
                """Return the wrong type."""
                return 5
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "script with broken append getter",
            work_dir=self.repo,
            extension_agent_path=agent_path,
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is False
        assert "get_append_to_prompt" in result.text
        assert "string" in result.text
        assert calls == [], "no executor session may start for a broken script"

    def test_malformed_wire_fields_append_nothing(self) -> None:
        """Non-string wire fields are ignored, not applied or crashing.

        The daemon treats the ``run`` command as untrusted input: a
        dict/list where a string is expected falls back to the default
        ``""`` instead of appending garbage or killing the task
        thread.  The absent-fields case (every pre-existing client) is
        exercised by every other suite of this harness.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        self._raw_daemon_run({
            "appendToSystemPrompt": {"marker": "MALFORMED-SYS-3341"},
            "appendToPrompt": ["MALFORMED-PROMPT-3341"],
        })
        (call,) = self._executor_calls(calls)
        assert "MALFORMED-SYS-3341" not in call["system_prompt"]
        assert "MALFORMED-PROMPT-3341" not in (
            call["arguments"]["task_description"]
        )

    def test_append_to_system_prompt_reaches_subagents(self) -> None:
        """The fan-out engine passes the suffix to every sub-agent.

        Covers both halves of the sub-agent wiring: the engine's
        ``system_prompt_suffix`` parameter (called directly) and the
        parent-agent forwarding of its stored ``_system_prompt_suffix``
        (``SorcarAgent._run_tasks_parallel``) — mirroring the existing
        ``base_system_prompt`` inheritance, so a run's extra system
        instructions constrain its whole task tree.
        """
        import threading as _threading

        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
        from kiss.agents.sorcar.sorcar_agent import (
            SorcarAgent,
            run_tasks_parallel,
        )

        parent_class = cast(Any, SorcarAgent.__mro__[1])
        original_run = parent_class.run
        lock = _threading.Lock()
        composed_prompts: list[str] = []

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            with lock:
                composed_prompts.append(str(kwargs.get("system_prompt")))
            return "success: true\nis_continue: false\nsummary: ok\n"

        parent_class.run = stub_run
        try:
            # Half 1: the engine parameter, as forwarded by a parent.
            results = run_tasks_parallel(
                ["child task one", "child task two"],
                work_dir=self.repo,
                system_prompt_suffix=_SYS_MARKER,
            )
            assert len(results) == 2
            assert len(composed_prompts) == 2
            for composed in composed_prompts:
                assert composed.startswith(SYSTEM_PROMPT)
                assert _SYS_MARKER in composed

            # Half 2: a parent agent that ran with the suffix stores it
            # and forwards it through its own fan-out.
            composed_prompts.clear()
            parent = ChatSorcarAgent("suffix-parent")
            parent._system_prompt_suffix = _SYS_MARKER
            results = parent._run_tasks_parallel(["nested child task"])
            assert len(results) == 1
            assert len(composed_prompts) == 1
            assert _SYS_MARKER in composed_prompts[0]

            # A parent WITHOUT a suffix spawns suffix-free children.
            composed_prompts.clear()
            plain_parent = ChatSorcarAgent("plain-parent")
            plain_parent._run_tasks_parallel(["plain child task"])
            assert len(composed_prompts) == 1
            assert _SYS_MARKER not in composed_prompts[0]
        finally:
            parent_class.run = original_run

    def test_early_panels_mirror_the_suffixes(self) -> None:
        """The optimistic panels show the suffixes the run executes with.

        ``_broadcast_early_prompts`` mirrors ``SorcarAgent.run``'s
        ``system_instructions`` and the executed prompt, so both early
        events must carry the corresponding suffix.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        events: list[dict[str, Any]] = []
        self._raw_daemon_run(
            {
                "appendToSystemPrompt": _SYS_MARKER,
                "appendToPrompt": _PROMPT_MARKER,
            },
            events_out=events,
        )
        early = {
            e["type"]: e["text"]
            for e in events
            if e.get("early") and e.get("type") in ("system_prompt", "prompt")
        }
        assert _SYS_MARKER in early["system_prompt"]
        assert early["prompt"].endswith(_PROMPT_MARKER)


if __name__ == "__main__":
    unittest.main()
