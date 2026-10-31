# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Terminal-Bench trial whose remote sandbox dies ends with ``EnvironmentLostError``.

On 2026-10-30 Modal stopped nine running GPU sandboxes at once; every later
command of their trials failed with ``exit code 255`` and the trial agents
spent hours working around it (even dispatching host-side sub-agents).  Now
the exec bridge server records the environment as *lost* when a command
fails and a ``true`` probe fails too, and ``SeaHarborAgent.run_sea`` stops
the daemon task and raises, so Harbor records an error the job re-runs
(``harbor jobs resume -f EnvironmentLostError``).

Real pieces throughout: an ``ExecBridgeServer`` on a loopback port, the
daemon-side ``bridge_exec`` client, and a stand-in daemon over ``wss://``
(:func:`kiss.tests.local_ws.fake_daemon`) speaking the ``run`` / ``stop``
protocol ``daemon_client.run`` expects.  Only the Harbor environment is a
small in-process object whose ``exec`` can be switched off, since a real
sandbox cannot be killed from a test.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import exec_bridge
from kiss.benchmarks.terminal_bench import exec_bridge_server
from kiss.benchmarks.terminal_bench.exec_bridge_server import ExecBridgeServer
from kiss.tests.local_ws import fake_daemon


@dataclass
class ExecResult:
    """The shape of Harbor's ``ExecResult``."""

    stdout: str | None
    stderr: str | None
    return_code: int


class SwitchableEnvironment:
    """A Harbor-like environment that answers every command until it is switched off.

    ``alive`` false makes ``exec`` raise the way Harbor's Modal environment
    does once the sandbox is gone (``NotFoundError``); ``fail_next`` makes the
    next *n* commands fail while the sandbox is otherwise fine (a transient
    provider error).  ``commands`` records what was asked.
    """

    def __init__(self, workdir: str) -> None:
        self.workdir = workdir
        self.alive = True
        self.fail_next = 0
        self.commands: list[str] = []

    def type(self) -> str:
        """Harbor's environment type: not ``docker``, so the agent bridges it."""
        return "modal"

    async def exec(self, command: str, timeout_sec: int | None = None, **_: Any) -> ExecResult:
        self.commands.append(command)
        if not self.alive:
            raise RuntimeError("Modal Sandbox with container ID ta-01 not found")
        if self.fail_next:
            self.fail_next -= 1
            raise ConnectionError("provider hiccup")
        return ExecResult(self.workdir + "\n" if command.strip() == "pwd" else "", "", 0)


def _probes(env: SwitchableEnvironment) -> int:
    """How many ``true`` probes the bridge sent *env* (each tagged, so matched by content)."""
    return sum("true" in c and "ls" not in c for c in env.commands)


class _Loop:
    """An asyncio loop on its own thread, as Harbor's loop is to the bridge thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


def test_bridge_records_a_lost_environment_but_not_a_transient_failure() -> None:
    """``lost`` is set when a command and the probe both fail, never for a one-off error."""
    loop = _Loop()
    env = SwitchableEnvironment("/app")
    bridge = ExecBridgeServer(env, loop.loop)
    try:
        assert exec_bridge.bridge_exec(bridge.url, "ls", 5) == (0, "")
        assert bridge.lost == "" and _probes(env) == 0

        env.fail_next = 1  # the command fails, the probe (``true``) succeeds
        code, output = exec_bridge.bridge_exec(bridge.url, "ls", 5)
        assert code == 255 and "ConnectionError: provider hiccup" in output
        assert bridge.lost == ""
        assert _probes(env) == 1  # the probe ran (tagged like every bridged command)

        env.fail_next = 2  # the command and the probe fail, then the sandbox answers again
        assert exec_bridge.bridge_exec(bridge.url, "ls", 5)[0] == 255
        assert "ConnectionError" in bridge.lost and _probes(env) == 2
        assert exec_bridge.bridge_exec(bridge.url, "ls", 5) == (0, "")
        assert bridge.lost == ""  # a command that ran clears the verdict

        env.alive = False  # the sandbox is gone: the command and the probe fail
        code, output = exec_bridge.bridge_exec(bridge.url, "ls", 5)
        assert code == 255 and "RuntimeError: Modal Sandbox" in output
        assert bridge.lost == output
        # once lost, later commands fail the same way without another probe
        assert _probes(env) == 3
        assert exec_bridge.bridge_exec(bridge.url, "ls", 5)[0] == 255
        assert _probes(env) == 3 and bridge.lost == output  # a failure keeps the verdict
        assert exec_bridge_server.PROBE_TIMEOUT_S < 30  # fits the client's reply slack

        # the trial runner's confirmation: still gone -> confirmed; answering -> cleared
        t0 = time.monotonic()
        assert _confirm(bridge, loop) is True and bridge.lost == output
        assert time.monotonic() - t0 >= exec_bridge_server.LOST_CONFIRM_S
        env.alive = True
        assert _confirm(bridge, loop) is False and bridge.lost == ""
        assert _confirm(bridge, loop) is False  # nothing recorded: nothing to confirm
    finally:
        bridge.close()
        loop.close()


def _confirm(bridge: ExecBridgeServer, loop: _Loop) -> bool:
    """Run ``bridge.confirm_lost()`` on the environment's loop, as the trial runner does."""
    return asyncio.run_coroutine_threadsafe(bridge.confirm_lost(), loop.loop).result(timeout=60)


class _TrialDaemon:
    """A stand-in daemon running a task until the client's ``stop`` arrives or ``finish`` is set."""

    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []
        self.run_cmd: dict[str, Any] | None = None
        self.started = asyncio.Event()
        self.finish = asyncio.Event()  # set by a test: the task completes on its own

    async def handle(self, ws: ServerConnection) -> None:
        """Serve one trial: ``running``, deltas, then the terminal result and status."""
        try:
            run_cmd: dict[str, Any] = json.loads(await ws.recv())
        except (ConnectionClosed, json.JSONDecodeError):
            return
        self.run_cmd = run_cmd
        tab_id = run_cmd.get("tabId", "")
        await ws.send(json.dumps({"type": "status", "running": True, "tabId": tab_id}))
        self.started.set()
        try:
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.05)
                except TimeoutError:
                    if self.finish.is_set():
                        await self._end(ws, tab_id, success=True)
                        return
                    await ws.send(json.dumps({"type": "text_delta", "text": "…", "tabId": tab_id}))
                    continue
                command = json.loads(message)
                self.commands.append(command)
                if command.get("type") == "stop":
                    await self._end(ws, tab_id, success=False)
                    return
        except ConnectionClosed:
            return

    async def _end(self, ws: ServerConnection, tab_id: str, success: bool) -> None:
        """Send the task's result (stopped, or finished on its own) and its terminal status."""
        await ws.send(json.dumps({
            "type": "result", "tabId": tab_id, "taskId": "task-lost-1",
            "success": success, "text": "done" if success else "stopped", "cost": "$0.0500",
            "total_tokens": 50, "step_count": 3,
        }))
        await ws.send(json.dumps({"type": "status", "running": False, "tabId": tab_id}))


def test_harbor_agent_ends_the_trial_when_the_sandbox_is_lost(tmp_path: Path) -> None:
    """The agent's command fails on a dead sandbox: the trial is stopped and ``run`` raises."""
    harbor_agent = pytest.importorskip("kiss.benchmarks.terminal_bench.harbor_agent")
    from harbor.models.agent.context import AgentContext  # type: ignore[import-untyped]

    from kiss.benchmarks.terminal_bench.daemon import ENDPOINT_FILE

    home = tmp_path / "home"
    home.mkdir()
    logs = tmp_path / "logs"
    env = SwitchableEnvironment("/app")
    daemon = _TrialDaemon()
    context = AgentContext()
    agent = harbor_agent.SeaHarborAgent(logs_dir=logs, model_name="m", home=str(home),
                                        reviewer="none")

    async def agent_command() -> tuple[int | None, str]:
        """What the trial agent's Bash tool does: one command through the bridge, after the
        sandbox died."""
        await asyncio.wait_for(daemon.started.wait(), 10)
        config = json.loads((logs / "config.json").read_text())
        url = config["container"][len("exec:"):]
        env.alive = False
        return await asyncio.to_thread(exec_bridge.bridge_exec, url, "ls /app", 5)

    async def main() -> tuple[tuple[int | None, str], float]:
        async with fake_daemon(home, daemon.handle, endpoint_file=home / ENDPOINT_FILE):
            command = asyncio.ensure_future(agent_command())
            t0 = time.monotonic()
            with pytest.raises(harbor_agent.EnvironmentLostError, match="Modal Sandbox"):
                await agent.run("do it", env, context)  # type: ignore[arg-type]
            return await command, time.monotonic() - t0

    (code, output), seconds = asyncio.run(main())
    assert code == 255 and "Modal Sandbox" in output
    # the daemon task was stopped and its spend recorded before the error was raised
    assert [c["type"] for c in daemon.commands] == ["stop"]
    assert context.cost_usd == 0.05
    assert context.metadata is not None
    assert context.metadata["error"].startswith("CancelledError")
    assert context.metadata["tokens"] == 50 and context.metadata["steps"] == 3
    result = json.loads((logs / "result.json").read_text())
    assert result["agent_success"] is False and result["cost_usd"] == 0.05
    # noticed within one poll interval plus the confirmation, and the bridge is closed afterwards
    assert seconds < harbor_agent.LOST_POLL_S + exec_bridge_server.LOST_CONFIRM_S + 15
    assert seconds >= exec_bridge_server.LOST_CONFIRM_S
    config = json.loads((logs / "config.json").read_text())
    assert not exec_bridge.bridge_alive(config["container"][len("exec:"):])
    assert daemon.run_cmd is not None and daemon.run_cmd["type"] == "run"


def test_harbor_agent_confirms_a_loss_found_just_before_the_task_finished(tmp_path: Path) -> None:
    """The task ends on its own right after the sandbox died: still an error, not a result."""
    harbor_agent = pytest.importorskip("kiss.benchmarks.terminal_bench.harbor_agent")
    from harbor.models.agent.context import AgentContext  # type: ignore[import-untyped]

    from kiss.benchmarks.terminal_bench.daemon import ENDPOINT_FILE

    home = tmp_path / "home"
    home.mkdir()
    logs = tmp_path / "logs"
    env = SwitchableEnvironment("/app")
    daemon = _TrialDaemon()
    context = AgentContext()
    agent = harbor_agent.SeaHarborAgent(logs_dir=logs, model_name="m", home=str(home),
                                        reviewer="none")

    async def last_command_then_finish() -> None:
        """The agent's last command fails on the dead sandbox; the agent then finishes."""
        await asyncio.wait_for(daemon.started.wait(), 10)
        config = json.loads((logs / "config.json").read_text())
        env.alive = False
        code, _ = await asyncio.to_thread(
            exec_bridge.bridge_exec, config["container"][len("exec:"):], "ls /app", 5)
        assert code == 255
        daemon.finish.set()

    async def main() -> None:
        async with fake_daemon(home, daemon.handle, endpoint_file=home / ENDPOINT_FILE):
            command = asyncio.ensure_future(last_command_then_finish())
            with pytest.raises(harbor_agent.EnvironmentLostError, match="Modal Sandbox"):
                await agent.run("do it", env, context)  # type: ignore[arg-type]
            await command

    asyncio.run(main())
    assert daemon.commands == []  # the task was never stopped: it had finished
    assert context.cost_usd == 0.05 and context.metadata is not None
    assert context.metadata["error"] == "" and context.metadata["agent_success"] is True
    assert json.loads((logs / "result.json").read_text())["cost_usd"] == 0.05
