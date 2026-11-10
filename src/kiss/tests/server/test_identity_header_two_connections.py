# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Who typed what in a shared workspace: ``KISS_IDENTITY_HEADER`` end to end.

A daemon behind an authenticating proxy serves several people under one
OS user.  With ``KISS_IDENTITY_HEADER`` naming the header the proxy sets
(oauth2-proxy's ``X-Forwarded-Email``), the daemon remembers the header
of every WebSocket connection and stamps it on the task's
``task_settings`` event (the person who started the run) and on each
steer ``prompt`` echo (the person who typed into the running task).

The test runs one real daemon on a loopback WSS listener and connects
real ``websockets`` clients with real upgrade headers.  The LLM boundary
is the only stand-in: ``SorcarAgent``'s base ``run`` is replaced by a
function that holds the task open until the second person's message has
arrived, then prints a result through the real printer.

Branches covered: header named and present (two different people on one
task), header named but absent (no key), a client-supplied ``identity``
field (dropped by ``dispatch``), variable unset (nothing recorded).  The
agent itself is never a stub: the only thing replaced is the model loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import TASK_ID_ENV, UsefulTools
from kiss.core import config as config_module
from kiss.core import user_config
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.web_server import IDENTITY_HEADER_ENV, RemoteAccessServer
from kiss.tests.server.conftest import (
    find_free_port,
    init_repo,
    no_verify_ssl,
    redirect_persistence,
    restore_persistence,
)

HEADER = "X-Forwarded-Email"
ALICE = "alice@example.com"
BOB = "bob@example.com"
MALLORY = "mallory@example.com"
HELD_MARK = "held run q4x"
SHELL_MARK = "shell run k7p"


class _Client:
    """One real WSS connection, every event it received, and its tab."""

    def __init__(self, socket: Any, name: str) -> None:
        self.socket = socket
        self.events: list[dict[str, Any]] = []
        self.tab_id = f"{name}-tab"
        self._drain = asyncio.ensure_future(self._pump())

    async def _pump(self) -> None:
        try:
            async for raw in self.socket:
                try:
                    self.events.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        except Exception:
            return

    async def send(self, cmd: dict[str, Any]) -> None:
        await self.socket.send(json.dumps(cmd))

    def of_type(self, event_type: str, **fields: Any) -> list[dict[str, Any]]:
        return [
            e for e in list(self.events)
            if e.get("type") == event_type
            and all(e.get(k) == v for k, v in fields.items())
        ]

    async def close(self) -> None:
        self._drain.cancel()
        try:
            await self.socket.close()
        except Exception:
            pass


class IdentityHeaderTwoConnectionsTest(IsolatedAsyncioTestCase):
    """Two people, two connections, one task: both emails in its event log."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss_identity_")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True)
        init_repo(self.repo)
        self._saved_persistence = redirect_persistence(self.tmpdir)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        self._saved_config = (
            vars(user_config).get("CONFIG_DIR"), vars(user_config).get("CONFIG_PATH"),
        )
        user_config.CONFIG_DIR = kiss_dir
        user_config.CONFIG_PATH = kiss_dir / "config.json"
        user_config.save_config({"remote_password": ""})
        self._saved_env = os.environ.get(IDENTITY_HEADER_ENV)
        os.environ[IDENTITY_HEADER_ENV] = HEADER
        # The daemon admits a run only when some provider has a key, and it
        # re-reads the keys from the environment on every run; the stubbed
        # model loop never uses this one.
        self._saved_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        config_module.DEFAULT_CONFIG.OPENROUTER_API_KEY = "test-key"
        self.model = get_available_models()[0]
        self.port = find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=self.repo, use_tunnel=False,
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self.clients: list[_Client] = []
        self.release = threading.Event()
        # Task id -> what its foreground Bash, background job and
        # run_commands_parallel printed.
        self.shell: dict[str, tuple[str, str, str]] = {}
        # A daemon started from some task's shell inherits that task's id.
        self._saved_task_id = os.environ.get(TASK_ID_ENV)
        os.environ[TASK_ID_ENV] = "stale-task"
        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run
        self._parent_class.run = self._stub_run()

    async def asyncTearDown(self) -> None:
        self.release.set()
        self._parent_class.run = self._original_run
        if self._saved_task_id is None:
            os.environ.pop(TASK_ID_ENV, None)
        else:
            os.environ[TASK_ID_ENV] = self._saved_task_id
        if self._saved_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self._saved_key
        config_module.DEFAULT_CONFIG.OPENROUTER_API_KEY = self._saved_key or ""
        for client in self.clients:
            await client.close()
        await self.server.stop_async()
        agent_state.agent_states.clear()
        if self._saved_env is None:
            os.environ.pop(IDENTITY_HEADER_ENV, None)
        else:
            os.environ[IDENTITY_HEADER_ENV] = self._saved_env
        saved_dir, saved_path = self._saved_config
        if saved_dir is None:
            user_config.__dict__.pop("CONFIG_DIR", None)
        else:
            user_config.CONFIG_DIR = saved_dir
        if saved_path is None:
            user_config.__dict__.pop("CONFIG_PATH", None)
        else:
            user_config.CONFIG_PATH = saved_path
        restore_persistence(self._saved_persistence)

    def _stub_run(self) -> Any:
        """The model loop: a ``HELD_MARK`` run waits for :attr:`release`, then finishes."""
        release = self.release
        shell = self.shell

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
            assert printer is not None, "a daemon run always has a printer"
            self_agent.printer = printer
            if SHELL_MARK in str(kwargs.get("prompt_template", "")):
                # The toolset the model loop would get, built by the agent.
                tools = {tool.__name__: tool for tool in self_agent._get_tools()}
                command = 'echo "task=$KISS_TASK_ID"'
                started = tools["Bash"](command, "task id", background=True)
                job_id = re.search(r"\bj[0-9a-f]{8}\b", started)
                assert job_id is not None, started
                shell[self_agent.last_task_id] = (
                    tools["Bash"](command, "task id"),
                    tools["bash_job"](job_id[0], "wait", timeout_seconds=20),
                    tools["run_commands_parallel"](json.dumps([command])),
                )
            if HELD_MARK in str(kwargs.get("prompt_template", "")):
                release.wait(timeout=30)
            self_agent.total_tokens_used = 5
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            printer.print(raw, type="result", step_count=1, total_tokens=5, cost="$0.0010")
            return raw

        return stub_run

    async def _connect(self, name: str, **headers: str) -> _Client:
        """Open a browser-like connection with *headers* on its upgrade request."""
        socket = await connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl(), max_size=None,
            additional_headers=headers,
        )
        client = _Client(socket, name)
        self.clients.append(client)
        await client.send({"type": "auth", "password": ""})
        await self._wait(lambda: client.of_type("auth_ok"), f"{name}: auth_ok")
        await client.send({"type": "ready", "tabId": client.tab_id, "restoredTabs": []})
        return client

    async def _wait(self, predicate: Any, what: str, timeout: float = 30.0) -> Any:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            found = predicate()
            if found:
                return found
            await asyncio.sleep(0.05)
        seen = [
            {k: v for k, v in e.items() if k in ("type", "text", "tabId", "code", "message")}
            for client in self.clients for e in client.events[-8:]
        ]
        raise AssertionError(f"timed out waiting for {what}; last events: {seen}")

    async def _submit(self, client: _Client, prompt: str, tab_id: str, **extra: Any) -> None:
        await client.send({
            "type": "submit", "prompt": prompt, "model": self.model, "attachments": [],
            "workDir": self.repo, "tabId": tab_id, **extra,
        })

    async def _settings_of(self, client: _Client, tab_id: str) -> dict[str, Any]:
        """The ``task_settings`` event a run on *tab_id* broadcast to *client*."""
        found = await self._wait(
            lambda: client.of_type("task_settings", tabId=tab_id),
            f"{client.tab_id}: task_settings",
        )
        return dict(found[0]["settings"])

    async def _finished(self, client: _Client, tab_id: str) -> None:
        await self._wait(
            lambda: client.of_type("status", running=False, tabId=tab_id),
            f"{tab_id}: task finished",
        )

    async def test_two_people_on_one_task_are_both_recorded(self) -> None:
        alice = await self._connect("alice", **{HEADER: ALICE})
        bob = await self._connect("bob", **{HEADER: BOB})

        # 1. Alice starts a task: the run's settings say she did, and all
        #    its shells get the task's own id.
        await self._submit(alice, f"{HELD_MARK} {SHELL_MARK}", alice.tab_id)
        settings = await self._settings_of(alice, alice.tab_id)
        self.assertEqual(settings.get("identity"), ALICE, settings)
        task_id = str(settings["task_id"])
        shells = await self._wait(lambda: self.shell.get(task_id), "alice: the task's shells")
        for output in shells:
            self.assertRegex(output, rf"task={task_id}(\n|$)")

        # 1b. A run_agent sub-task arrives over a connection that forwards
        #     no one; it records the person its caller works for, and its
        #     shells get the sub-task's id, not the caller's.
        relay = await self._connect("relay")
        await relay.send({
            "type": "run", "prompt": f"child {SHELL_MARK}", "tabId": relay.tab_id,
            "model": self.model, "workDir": self.repo, "parentTaskId": task_id,
        })
        child = await self._settings_of(relay, relay.tab_id)
        self.assertEqual(child.get("identity"), ALICE, child)
        child_id = str(child["task_id"])
        self.assertNotEqual(child_id, task_id)
        shells = await self._wait(lambda: self.shell.get(child_id), "child shells")
        for output in shells:
            self.assertRegex(output, rf"task={child_id}(\n|$)")

        # 2. Bob types into the running task from his own browser: the
        #    echo every viewer gets names him, not her.
        await self._submit(bob, "bob says hello", alice.tab_id)
        echo = (await self._wait(
            lambda: alice.of_type("prompt", steer=True, taskId=task_id),
            "alice: bob's steer echo",
        ))[0]
        self.assertEqual(echo.get("identity"), BOB, echo)
        self.release.set()
        await self._finished(alice, alice.tab_id)

        # 3. The persisted event log carries the two different emails.
        _persistence._flush_chat_events(task_id)
        session = _persistence._load_chat_events_by_task_id(task_id)
        assert session is not None
        events = cast(list[dict[str, Any]], session["events"])
        recorded = [
            e["settings"]["identity"] for e in events if e.get("type") == "task_settings"
        ] + [e["identity"] for e in events if e.get("type") == "prompt" and e.get("steer")]
        self.assertEqual(recorded, [ALICE, BOB], events)

        # 4. A connection without the header records nothing, and a
        #    client cannot name itself: the field it sends is dropped.
        nobody = await self._connect("nobody")
        await self._submit(nobody, "plain run", nobody.tab_id, identity=MALLORY)
        settings = await self._settings_of(nobody, nobody.tab_id)
        self.assertNotIn("identity", settings, settings)
        await self._finished(nobody, nobody.tab_id)

        # 5. Variable unset: the header is not even read.
        os.environ.pop(IDENTITY_HEADER_ENV, None)
        quiet = await self._connect("quiet", **{HEADER: ALICE})
        await self._submit(quiet, "quiet run", quiet.tab_id)
        settings = await self._settings_of(quiet, quiet.tab_id)
        self.assertNotIn("identity", settings, settings)
        await self._finished(quiet, quiet.tab_id)

        # 6. Tools that no task runs export no id, not the daemon's stale one.
        output = UsefulTools(work_dir=self.repo).Bash('echo "task=$KISS_TASK_ID"', "no task")
        self.assertIn("task=\n", output)
