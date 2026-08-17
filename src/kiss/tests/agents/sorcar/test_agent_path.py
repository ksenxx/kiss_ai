# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.server.sorcar.run``'s ``extension_agent_path``.

Spin up a real :class:`kiss.server.web_server.RemoteAccessServer` on a
temporary Unix-domain socket and drive ``kiss.server.sorcar.run`` with
an ``extension_agent_path`` script against it.  The only replaced boundary is the
LLM itself: like the other task-runner suites in this directory,
``SorcarAgent``'s parent ``run`` is swapped for a stub so the daemon's
full run pipeline (``run`` command dispatch → worker thread →
agent-script overrides → agent wiring → event broadcast → status end)
executes for real without any model API calls.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import vscode_config
from kiss.core.models.model_info import get_available_models
from kiss.server import sorcar
from kiss.server.web_server import RemoteAccessServer


def _task_chat_id(task_id: str) -> str:
    """Return the persisted chat_id of *task_id* via ``_load_history``."""
    for row in _persistence._load_history():
        if row["id"] == task_id:
            return str(row["chat_id"] or "")
    return ""


def _init_repo(repo: str) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            check=False,
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")
    git("config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    git("add", "seed.txt")
    git("commit", "-q", "-m", "seed")


class AgentPathApiTest(unittest.TestCase):
    """Drive ``sorcar.run(extension_agent_path=...)`` against a real daemon over UDS."""

    def setUp(self) -> None:
        # Resolved: macOS mkdtemp returns a symlinked /var/... path while
        # the worktree machinery canonicalizes the repo (git_worktree
        # resolves it), so un-resolved paths break startswith checks.
        self.tmpdir = str(Path(tempfile.mkdtemp(prefix="sorcar_agent_path_")).resolve())
        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None
        self._saved_config_override = (
            vars(vscode_config).get("CONFIG_DIR"),
            vars(vscode_config).get("CONFIG_PATH"),
        )
        vscode_config.CONFIG_DIR = kiss_dir
        vscode_config.CONFIG_PATH = kiss_dir / "config.json"

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path, work_dir=self.repo,
        )
        self.server._printer._loop = self.loop
        self.server._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=5)

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        from kiss.server import agent_state

        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — best-effort cleanup
                    pass
        agent_state.agent_states.clear()

        async def _shutdown() -> None:
            with self.server._printer._ws_lock:
                writers = list(self.server._printer._uds_writers)
            for writer in writers:
                try:
                    writer.close()
                except Exception:
                    pass
            self.uds_server.close()
            await self.uds_server.wait_closed()
            pending = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task()
            ]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(
                _shutdown(), self.loop,
            ).result(timeout=5)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence
        saved_dir, saved_path = self._saved_config_override
        if saved_dir is None:
            if "CONFIG_DIR" in vars(vscode_config):
                delattr(vscode_config, "CONFIG_DIR")
        else:
            vscode_config.CONFIG_DIR = saved_dir
        if saved_path is None:
            if "CONFIG_PATH" in vars(vscode_config):
                delattr(vscode_config, "CONFIG_PATH")
        else:
            vscode_config.CONFIG_PATH = saved_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_py(self, name: str, content: str) -> str:
        """Write a Python file under the test tmpdir and return its path.

        Args:
            name: File name (e.g. ``"my_agent.py"``).
            content: Python source for the file.

        Returns:
            The absolute path of the written file.
        """
        path = Path(self.tmpdir) / name
        path.write_text(textwrap.dedent(content))
        return str(path)

    def _install_recording_stub(self, seen: dict[str, Any]) -> None:
        """Swap the agent's LLM run for a stub recording its kwargs.

        Args:
            seen: Dict that receives the kwargs the daemon-built agent
                was invoked with (plus a few agent attributes).
        """

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen.update(kwargs)
            seen["_web_tools_attr"] = getattr(
                self_agent, "_use_web_tools", None,
            )
            seen["_is_parallel_attr"] = getattr(
                self_agent, "_is_parallel", None,
            )
            seen["_base_system_prompt_attr"] = getattr(
                self_agent, "_base_system_prompt", None,
            )
            seen["_auto_commit_attr"] = getattr(
                self_agent, "auto_commit_enabled", None,
            )
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            kwargs["printer"].print(
                raw, type="result", step_count=1, total_tokens=1,
                cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run

    def test_getters_override_every_supported_parameter(self) -> None:
        """Each defined ``get_X()`` replaces the value passed for X.

        The client passes explicit values for every overridable
        parameter and the script defines a getter for each — the
        daemon-built agent must see the SCRIPT's values, proving the
        getters ran on the daemon and won over the passed arguments.
        """
        available = get_available_models()
        assert available, "test needs at least one available model"
        script_model = available[-1]
        repo2 = str(Path(self.tmpdir) / "repo2")
        Path(repo2).mkdir(parents=True, exist_ok=True)
        _init_repo(repo2)
        tools_path = self._write_py(
            "script_tools.py",
            '''
            """Tools the agent script picks."""


            def scripted_tool(x: int) -> int:
                """Double.

                Args:
                    x: Value to double.
                """
                return 2 * x


            def get_tools():
                """Return the tools the agent may call."""
                return [scripted_tool]
            ''',
        )
        agent_path = self._write_py(
            "my_agent.py",
            f'''
            """Agent script overriding every supported parameter."""

            import pathlib


            def get_prompt():
                return "scripted prompt marker"


            def get_work_dir():
                return {repo2!r}


            def get_model():
                return {script_model!r}


            def get_system_prompt():
                return "scripted system prompt"


            def get_tools():
                # A pathlib.Path is accepted like run(tools=...) does.
                return pathlib.Path({tools_path!r})


            def get_use_worktree():
                return False


            def get_auto_commit():
                return False


            def get_max_budget():
                return 1.25


            def get_model_config():
                return {{"base_url": "http://localhost:1234/v1"}}


            def get_web_tools():
                return False


            def get_is_parallel():
                return False
            ''',
        )
        client_tools = self._write_py(
            "client_tools.py",
            '''
            """Tools the client passes (the script's must win)."""


            def client_tool() -> str:
                """Return a marker."""
                return "client"


            def get_tools():
                """Return the tools the agent may call."""
                return [client_tool]
            ''',
        )
        seen: dict[str, Any] = {}
        self._install_recording_stub(seen)
        result = sorcar.run(
            "client prompt",
            work_dir=self.repo,
            model=available[0],
            system_prompt="client system prompt",
            tools=client_tools,
            extension_agent_path=agent_path,
            use_worktree=True,
            auto_commit=True,
            max_budget=9.5,
            model_config={"base_url": "http://client:1/v1"},
            web_tools=True,
            is_parallel=True,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        assert result.text == "agent ok"
        assert "scripted prompt marker" in str(seen["prompt_template"])
        assert "client prompt" not in str(seen["prompt_template"])
        # ``get_use_worktree() -> False`` took effect: the agent runs in
        # the scripted work dir ITSELF, not in a worktree carved under
        # the client-passed repo.
        assert seen["work_dir"] == repo2
        assert seen["model_name"] == script_model
        assert seen["_base_system_prompt_attr"] == "scripted system prompt"
        assert str(seen["system_prompt"]).startswith("scripted system prompt")
        assert seen["_auto_commit_attr"] is False
        assert seen["max_budget"] == 1.25
        assert seen["model_config"] == {"base_url": "http://localhost:1234/v1"}
        assert seen["_web_tools_attr"] is False
        assert seen["_is_parallel_attr"] is False
        assert [t.__name__ for t in seen["tools"]] == ["scripted_tool"]
        assert seen["tools"][0](x=21) == 42

    def test_missing_getters_keep_passed_and_default_values(self) -> None:
        """Parameters without a ``get_X()`` keep the caller's values.

        The script defines only ``get_prompt()``; every other parameter
        must arrive exactly as passed to :func:`sorcar.run` — and
        parameters the caller did not pass must keep their defaults
        (``use_worktree=True``, ``is_parallel=True``, daemon-config
        budget).
        """
        agent_path = self._write_py(
            "prompt_only_agent.py",
            '''
            """Agent script overriding only the prompt."""


            def get_prompt():
                return "prompt from script"
            ''',
        )
        seen: dict[str, Any] = {}
        self._install_recording_stub(seen)
        result = sorcar.run(
            "original prompt",
            work_dir=self.repo,
            system_prompt="kept system prompt",
            extension_agent_path=agent_path,
            max_budget=3.5,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        assert "prompt from script" in str(seen["prompt_template"])
        # Default ``use_worktree=True`` kept: the agent runs in a
        # worktree carved under the passed repo, not in the repo itself.
        assert seen["work_dir"] != self.repo
        assert ".kiss-worktrees" in str(seen["work_dir"])
        assert str(seen["work_dir"]).startswith(self.repo)
        assert seen["_base_system_prompt_attr"] == "kept system prompt"
        assert seen["max_budget"] == 3.5
        assert seen["_is_parallel_attr"] is True
        assert seen["tools"] == []

    def test_get_tools_none_drops_client_tools(self) -> None:
        """A ``get_tools()`` returning ``None`` overrides to no tools."""
        client_tools = self._write_py(
            "dropped_tools.py",
            '''
            """Tools the client passes (dropped by the script)."""


            def client_tool() -> str:
                """Return a marker."""
                return "client"


            def get_tools():
                """Return the tools the agent may call."""
                return [client_tool]
            ''',
        )
        agent_path = self._write_py(
            "no_tools_agent.py",
            '''
            """Agent script clearing the tools."""


            def get_tools():
                return None
            ''',
        )
        seen: dict[str, Any] = {}
        self._install_recording_stub(seen)
        result = sorcar.run(
            "run without tools",
            work_dir=self.repo,
            tools=client_tools,
            extension_agent_path=agent_path,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        assert seen["tools"] == []

    def test_get_chat_id_continues_existing_chat(self) -> None:
        """A ``get_chat_id()`` override continues that chat's context.

        The client passes NO ``chat_id``; the script's ``get_chat_id()``
        returns the first run's chat id.  The second run must persist
        under that chat and see the first task in its prompt context —
        proving the override reached ``state.chat_id`` on the daemon.
        """
        prompts_seen: list[str] = []

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            prompts_seen.append(str(kwargs.get("prompt_template", "")))
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = (
                "success: true\n"
                "is_continue: false\n"
                "summary: chat marker answer\n"
            )
            kwargs["printer"].print(
                raw, type="result", step_count=1, total_tokens=1,
                cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run
        first = sorcar.run(
            "remember the word plugh",
            work_dir=self.repo,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert first.success is True
        assert first.chat_id
        agent_path = self._write_py(
            "chat_agent.py",
            f'''
            """Agent script pinning the chat id."""


            def get_chat_id():
                return {first.chat_id!r}
            ''',
        )
        second = sorcar.run(
            "what was the word?",
            work_dir=self.repo,
            extension_agent_path=agent_path,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert second.success is True
        assert second.chat_id == first.chat_id
        assert second.task_id and second.task_id != first.task_id
        assert _task_chat_id(second.task_id) == first.chat_id
        assert len(prompts_seen) == 2
        assert "remember the word plugh" in prompts_seen[1]
        assert "chat marker answer" in prompts_seen[1]

    def test_broken_agent_scripts_fail_the_task_with_diagnostics(self) -> None:
        """Import errors, raising getters, and bad returns stop the task."""
        seen: dict[str, Any] = {}
        self._install_recording_stub(seen)
        cases = [
            (
                "importfail_agent.py",
                'raise RuntimeError("boom at import")\n',
                ["agent script", "failed to import", "boom at import"],
            ),
            (
                "raising_getter_agent.py",
                "def get_model():\n    raise ValueError('no model today')\n",
                ["get_model()", "raised", "no model today"],
            ),
            (
                "badtype_agent.py",
                "def get_max_budget():\n    return 'lots'\n",
                ["get_max_budget()", "must return a finite number or None", "str"],
            ),
            (
                "noncallable_agent.py",
                "get_prompt = 'not a function'\n",
                ["get_prompt", "must be a callable", "str"],
            ),
            (
                "empty_prompt_agent.py",
                "def get_prompt():\n    return '  '\n",
                ["get_prompt()", "must return a non-empty string"],
            ),
            (
                "none_getter_agent.py",
                "get_model = None\n",
                ["get_model", "must be a callable", "NoneType"],
            ),
            (
                "nan_budget_agent.py",
                "def get_max_budget():\n    return float('nan')\n",
                [
                    "get_max_budget()",
                    "must return a finite number or None",
                ],
            ),
            (
                "inf_budget_agent.py",
                "def get_max_budget():\n    return float('inf')\n",
                [
                    "get_max_budget()",
                    "must return a finite number or None",
                ],
            ),
            (
                "huge_budget_agent.py",
                "def get_max_budget():\n    return 10 ** 400\n",
                [
                    "get_max_budget()",
                    "must return a finite number or None",
                ],
            ),
            (
                "evil_value_agent.py",
                "class _Evil(str):\n"
                "    def strip(self, *args):\n"
                "        raise RuntimeError('evil strip')\n"
                "def get_prompt():\n"
                "    return _Evil('x')\n",
                ["get_prompt()", "returned a broken value", "evil strip"],
            ),
        ]
        for name, source, expected_parts in cases:
            with self.subTest(script=name):
                agent_path = self._write_py(name, source)
                result = sorcar.run(
                    "should not run the agent",
                    work_dir=self.repo,
                    extension_agent_path=agent_path,
                    sock_path=self.sock_path,
                    timeout=60,
                )
                assert result.success is False
                assert "AgentFileError" in result.text
                for part in expected_parts:
                    assert part in result.text, (part, result.text)
        assert "prompt_template" not in seen, (
            "a broken agent script must stop the task before the agent runs"
        )

    def test_broken_script_leaves_command_untouched(self) -> None:
        """A later failing getter must not apply earlier overrides.

        ``apply_agent_overrides`` is the daemon-side loader; drive it
        directly with a real script whose ``get_chat_id()`` succeeds
        and whose LATER ``get_web_tools()`` raises: the command must
        come out exactly as it went in — a direct ``_run_task`` caller
        seeds its run state from the command, so a partial override
        surviving the failure would leak the broken script's chat id
        into a later run.
        """
        from kiss.server.agent_file import (
            AgentFileError,
            apply_agent_overrides,
        )

        agent_path = self._write_py(
            "partial_agent.py",
            '''
            """Agent script whose later getter fails."""


            def get_chat_id():
                return "hijacked-chat"


            def get_web_tools():
                raise RuntimeError("late failure")
            ''',
        )
        cmd = {
            "type": "run",
            "prompt": "hi",
            "chatId": "original-chat",
            "webTools": None,
            "agentPath": agent_path,
        }
        original = dict(cmd)
        with self.assertRaises(AgentFileError) as ctx:
            apply_agent_overrides(cmd)
        assert "get_web_tools()" in str(ctx.exception)
        assert "late failure" in str(ctx.exception)
        assert cmd == original

    def test_invalid_agent_path_raises_value_error(self) -> None:
        """The client rejects a bad ``extension_agent_path`` before connecting."""
        with self.assertRaises(ValueError):
            sorcar.run(
                "hi", extension_agent_path=str(Path(self.tmpdir) / "missing.py"),
                sock_path=self.sock_path,
            )
        not_py = Path(self.tmpdir) / "agent.txt"
        not_py.write_text("def get_model():\n    return 'x'\n")
        with self.assertRaises(ValueError):
            sorcar.run("hi", extension_agent_path=str(not_py), sock_path=self.sock_path)
        with self.assertRaises(ValueError):
            sorcar.run(
                "hi", extension_agent_path=cast(Any, 123), sock_path=self.sock_path,
            )


if __name__ == "__main__":
    unittest.main()
