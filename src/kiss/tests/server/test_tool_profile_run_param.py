# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``tool_profile`` run parameter on the daemon.

Spin up a real :class:`kiss.server.web_server.RemoteAccessServer` on a
temporary Unix-domain socket (the :class:`DaemonRunApiHarness` from
``test_append_basic_tools``) and drive ``kiss.server.sorcar.run`` /
the ``run_agent`` tool against it.  The only replaced boundary is the
LLM itself: the per-session executor's
:meth:`kiss.core.kiss_agent.KISSAgent.run` is swapped for a stub that
records the tools it was handed, so the daemon's full run pipeline —
wire ``toolProfile`` field → ``apply_agent_overrides`` (the SEA's
``tool_profile()`` getter) → ``task_runner`` validation →
``SorcarAgent.run(tool_profile=...)`` → ``_get_tools`` filtering —
executes for real without any model API calls.

Contract under test: ``tool_profile`` (the parameter, the wire field
and the agent-script getter) cuts the built-in toolset down to the
named ``TOOL_PROFILES`` entry; the bundled ``sh_sea.py`` therefore
runs with ``Bash`` + ``finish`` only, in the caller's work directory
(no worktree), with its own system prompt; an unknown name fails the
task before any executor session starts; a malformed wire value means
"no profile"; the ``run_agent`` tool validates and forwards its
``tool_profile`` argument; and the OUTER relay run of a ``/xxx``
command honours the SEA's ``use_worktree()`` / ``auto_commit()``
verdicts (a broken SEA fails that relay with a diagnostic).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

from kiss.agents.seas import sh_sea
from kiss.agents.sorcar import cron_agent, sea_commands
from kiss.agents.sorcar.agent_dispatch import _run_agent, make_run_agent_tool
from kiss.core.config import kiss_home
from kiss.core.kiss_agent import KISSAgent
from kiss.server import sorcar
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness

pytestmark = requires_unix_sockets

_SH_SEA_PATH = str(Path(sh_sea.__file__).resolve())


class ToolProfileRunParamTest(DaemonRunApiHarness):
    """``tool_profile`` must reach the executor's tool list through the daemon."""

    def _install_recording_stub(self, calls: list[dict[str, Any]]) -> None:
        """Swap the executor LLM loop for a stub recording tools and work dir.

        Like the harness stub, but also records the work directory the
        built-in ``Bash`` tool is bound to (``UsefulTools.work_dir``),
        which tells a direct run apart from a worktree run.

        Args:
            calls: List receiving, per task-executor ``KISSAgent.run``
                invocation, the ``tool_names`` in order, the
                ``system_prompt``, and ``bash_work_dir`` (``None`` when
                no ``Bash`` tool was handed to the session).
        """

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""
            arguments = dict(kwargs.get("arguments") or {})
            if "task_description" not in arguments:
                self_agent.step_count = 1
                return "result: prior progress\n"
            tools = kwargs.get("tools") or []
            bash = next((t for t in tools if getattr(t, "__name__", "") == "Bash"), None)
            calls.append({
                "tool_names": [getattr(t, "__name__", "?") for t in tools],
                "system_prompt": str(kwargs.get("system_prompt") or ""),
                "bash_work_dir": (
                    getattr(getattr(bash, "__self__", None), "work_dir", None)
                    if bash is not None else None
                ),
            })
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0001
            self_agent.step_count = 1
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            printer = kwargs.get("printer")
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1, total_tokens=1,
                    cost="$0.0001",
                )
            return raw

        cast(Any, KISSAgent).run = stub_run

    def _single_call(self, calls: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the one recorded executor call."""
        assert len(calls) == 1, calls
        return calls[0]

    def _worktree_count(self) -> int:
        """Return the number of git worktrees of the test repo (1 = none added)."""
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=self.repo, capture_output=True, text=True, check=True,
        ).stdout
        return sum(1 for line in out.splitlines() if line.startswith("worktree "))

    def _git(self, *args: str) -> str:
        """Run ``git *args`` in the test repo and return its stripped stdout."""
        return subprocess.run(
            ["git", *args], cwd=self.repo, capture_output=True, text=True, check=True,
        ).stdout.strip()

    def test_slash_sh_relay_runs_without_worktree_or_auto_commit(self) -> None:
        """The OUTER ``/sh`` run honours the SEA's no-worktree / no-auto-commit verdicts.

        A ``/sh <command>`` prompt becomes a relay run whose agent calls
        ``run_agent`` with its own work directory.  With the tab's
        worktree and auto-commit toggles ON, the relay must still run
        in the real checkout (or the SEA would inherit the relay's
        worktree) and must not commit what the command left behind.
        """
        (Path(self.repo) / "sentinel.txt").write_text("left by the command\n")
        commits_before = self._git("rev-list", "--count", "HEAD")
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        result = sorcar.run(
            "/sh printf hi",
            work_dir=self.repo,
            use_worktree=True,
            auto_commit=True,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True, result
        call = self._single_call(calls)
        assert call["bash_work_dir"] == self.repo, call
        assert self._worktree_count() == 1
        assert self._git("status", "--porcelain") == "?? sentinel.txt"
        assert self._git("rev-list", "--count", "HEAD") == commits_before

    def test_broken_user_sea_fails_the_relay_instead_of_stopping_it(self) -> None:
        """An SEA raising ``KeyboardInterrupt`` at import is a task ERROR, not a stop.

        The relay demotion imports the SEA on the task thread; whatever
        the script raises is normalised into ``SeaScriptError`` (with
        the original raise as the cause), so the runner reports a
        failed task with the diagnostic rather than "stopped by user".
        """
        folder = Path(self.tmpdir) / "user-seas"
        folder.mkdir()
        (folder / "boom_sea.py").write_text(
            'raise KeyboardInterrupt("boom at import")\n', encoding="utf-8",
        )
        home = kiss_home()
        home.mkdir(parents=True, exist_ok=True)
        seas_md = home / "SEAS.md"
        seas_md.write_text(str(folder) + "\n", encoding="utf-8")
        sea_commands.refresh_registry()
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        try:
            assert sea_commands.get_command("boom") is not None
            result = sorcar.run(
                "/boom anything",
                work_dir=self.repo,
                use_worktree=True,
                sock_path=self.sock_path,
                timeout=60,
            )
        finally:
            seas_md.unlink()
            sea_commands._reset_for_tests()
        assert result.success is False, result
        assert "Task failed: SeaScriptError" in result.text, result
        assert "boom_sea.py" in result.text and "use_worktree()" in result.text, result
        assert "KeyboardInterrupt: boom at import" in result.text, result
        assert "stopped" not in result.text.lower(), result
        assert calls == []

    def test_plain_prompt_control_commits_the_dirty_tree(self) -> None:
        """Control for the relay test: a plain direct run with auto-commit on commits."""
        (Path(self.repo) / "sentinel.txt").write_text("left by the task\n")
        commits_before = self._git("rev-list", "--count", "HEAD")
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        result = sorcar.run(
            "plain task on the main tree",
            work_dir=self.repo,
            use_worktree=False,
            auto_commit=True,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True, result
        assert self._git("status", "--porcelain") == ""
        assert int(self._git("rev-list", "--count", "HEAD")) == int(commits_before) + 1

    def test_sh_sea_runs_bash_only_in_the_callers_directory(self) -> None:
        """The bundled ``/sh`` agent: Bash + finish, its own prompt, no worktree.

        The run is dispatched exactly as the ``/sh`` slash command
        dispatches it — the SEA path as ``extension_agent_path`` with
        the daemon defaults (``use_worktree=True``) that the script's
        getters must override.
        """
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        result = sorcar.run(
            "printf 'hello from sh'",
            work_dir=self.repo,
            extension_agent_path=_SH_SEA_PATH,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True, result
        call = self._single_call(calls)
        assert call["tool_names"] == ["finish", "Bash"], call
        assert call["system_prompt"].startswith(sh_sea.SYSTEM_PROMPT), call
        assert "# Restricted tool profile: bash" in call["system_prompt"]
        assert call["bash_work_dir"] == self.repo, call
        assert self._worktree_count() == 1

    def test_shell_profile_parameter_cuts_the_toolset(self) -> None:
        """``sorcar.run(tool_profile="shell")`` offers the shell profile's tools."""
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        result = sorcar.run(
            "task with the shell profile",
            work_dir=self.repo,
            tool_profile="shell",
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True, result
        call = self._single_call(calls)
        assert set(call["tool_names"]) == {
            "finish", "Bash", "bash_job", "Read", "run_commands_parallel",
        }, call
        assert "# Restricted tool profile: shell" in call["system_prompt"]

    def test_no_profile_keeps_the_full_toolset(self) -> None:
        """Without ``tool_profile`` the executor gets the full built-in set."""
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        result = sorcar.run(
            "task with the full toolset",
            work_dir=self.repo,
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True, result
        names = set(self._single_call(calls)["tool_names"])
        assert {"finish", "Bash", "Read", "Edit", "Write", "run_agent"} <= names

    def test_unknown_profile_fails_the_task_before_any_session(self) -> None:
        """An unknown ``tool_profile`` stops the run with a diagnostic."""
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        result = sorcar.run(
            "task with a bogus profile",
            work_dir=self.repo,
            tool_profile="bogus",
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is False, result
        assert "tool_profile must be one of" in result.text, result
        assert "'bogus'" in result.text, result
        assert calls == [], "no executor session may start for an unknown profile"

    def test_agent_script_profile_getter_wins_over_the_wire_value(self) -> None:
        """A script's ``tool_profile()`` overrides the client's ``tool_profile``."""
        agent_path = self._write_py(
            "shell_profile_agent.py",
            '''
            """Agent script choosing the shell profile."""


            def tool_profile() -> str:
                """Return the profile name."""
                return "shell"
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        result = sorcar.run(
            "task whose script picks the profile",
            work_dir=self.repo,
            extension_agent_path=agent_path,
            tool_profile="bash",
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True, result
        assert "bash_job" in self._single_call(calls)["tool_names"]

    def test_wrong_typed_profile_getter_fails_task(self) -> None:
        """A non-string ``tool_profile()`` result stops the task."""
        agent_path = self._write_py(
            "bad_profile_agent.py",
            '''
            """Agent script with a wrong-typed profile getter."""


            def tool_profile() -> int:
                """Return the wrong type."""
                return 7
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        result = sorcar.run(
            "task with a broken profile getter",
            work_dir=self.repo,
            extension_agent_path=agent_path,
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is False
        assert "tool_profile" in result.text
        assert "a string" in result.text
        assert calls == []

    def test_malformed_wire_profile_means_no_profile(self) -> None:
        """A non-string ``toolProfile`` wire value is ignored, not fatal."""
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        events: list[dict[str, Any]] = []
        self._raw_daemon_run({"toolProfile": 123}, events_out=events)
        names = set(self._single_call(calls)["tool_names"])
        assert {"finish", "Bash", "Read", "Edit", "Write"} <= names
        results = [e for e in events if e.get("type") == "result"]
        assert results and results[-1].get("success") is True, events

    def test_run_agent_tool_forwards_tool_profile(self) -> None:
        """The ``run_agent`` tool's ``tool_profile`` argument reaches the daemon."""
        script = self._write_py(
            "plain_agent.py",
            '''
            """Agent script with no getters: the tool's arguments decide."""
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        saved_sock = cron_agent._daemon_sock_path
        cron_agent._daemon_sock_path = self.sock_path
        try:
            tool = make_run_agent_tool(self.repo, None)
            text = tool(
                "run with the bash profile", script,
                use_worktree="false", tool_profile="bash",
            )
        finally:
            cron_agent._daemon_sock_path = saved_sock
        assert yaml.safe_load(text)["success"] is True, text
        assert self._single_call(calls)["tool_names"] == ["finish", "Bash"]

    def test_run_agent_tool_rejects_unknown_profile_locally(self) -> None:
        """``run_agent(tool_profile="bogus")`` is refused without a daemon round trip."""
        calls: list[dict[str, Any]] = []
        self._install_recording_stub(calls)
        text = _run_agent(
            self.repo, _SH_SEA_PATH, "printf hi", "", "", "", "",
            tool_profile="bogus",
        )
        assert text.startswith("Error: tool_profile must be one of"), text
        assert "'bogus'" in text
        assert calls == []
