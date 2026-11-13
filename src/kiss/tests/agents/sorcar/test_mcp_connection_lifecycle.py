# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""MCP connection lifecycle: pooling, eviction, reconnect and locking.

F7 (a) ``connect()`` only stopped a connection under the *same* key, and the
key is a digest of the whole config, so any config change spawned a fresh
stdio child and orphaned the previous one for the life of the process.
``shutdown()`` is reachable only from ``atexit``, so the long-lived
``kiss-web`` daemon leaked one child per (project x config revision).
(c) ``call_tool`` never reconnected, so after a mid-task server crash every
remaining call to that server returned "is not connected" for the rest of
the (possibly hour-long) task.

F8 ``FileTokenStorage.set_tokens`` performed a blocking inter-process
``flock`` from inside a coroutine on the manager's single shared event loop,
so another process holding that lock (an interactive OAuth login) stalled
every other MCP tool call in this process for up to the 305 s call timeout.

Real stdio MCP servers (real FastMCP subprocesses over the real protocol),
a real FastMCP server over SSE under uvicorn on a loopback port, real
signals, real file locks held by a real second process.  No mocks.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import gc
import os
import socket
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

from kiss.agents.sorcar import mcp_servers
from kiss.agents.sorcar.mcp_servers import (
    FileTokenStorage,
    MCPManager,
    MCPServerConfig,
    _child_errlog,
    _connection_key,
)
from kiss.core import tool_interrupt
from kiss.core.locked import locked_call
from kiss.core.processes import pid_alive

_SERVER_SCRIPT = '''
import os, sys, time

from mcp.server.fastmcp import FastMCP

open(sys.argv[1], "w").write(str(os.getpid()))

mcp = FastMCP("gsrv")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool()
def block_until(started: str, release: str) -> str:
    """Announce *started*, wait for *release* to appear, then answer."""
    open(started, "w").close()
    deadline = time.time() + 120
    while time.time() < deadline:
        if os.path.exists(release):
            return "released"
        time.sleep(0.02)
    return "timeout"


if __name__ == "__main__":
    mcp.run()
'''

#: Holds the shared token lock for 5 s so the parent can measure whether an
#: unrelated MCP tool call is blocked behind it.
_LOCK_HOLDER = '''
import sys, time
from pathlib import Path

from kiss.core.file_lock import exclusive_file_lock

lock_path, ready = Path(sys.argv[1]), Path(sys.argv[2])
with exclusive_file_lock(lock_path):
    ready.write_text("held")
    time.sleep(5)
'''


@pytest.fixture
def real_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[None]:
    """Give ``sys.stdin`` and the MCP errlog real file descriptors.

    Under pytest the std streams are capture objects whose ``fileno()``
    raises, which stops the MCP stdio transport from spawning a server at
    all.  ``os.devnull`` and a real file restore them for the test.

    Only ``sys.stderr`` needs replacing: ``_enter_transport`` resolves the
    child's stderr through ``_child_errlog()`` on every spawn, so it picks
    this file up without the transport's import-time default being touched.
    """
    # ``os.devnull`` has a real descriptor on every platform (Windows has
    # no pty) and nothing reads stdin while the client talks to the child.
    stdin_stream = open(os.devnull, encoding="utf-8")
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    """Write the real FastMCP stdio server used by these tests."""
    script = tmp_path / "gsrv.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return script


def _config(script: Path, pid_file: Path, *extra: str) -> MCPServerConfig:
    """Build a stdio config that records the server's pid in *pid_file*."""
    return MCPServerConfig(
        name="gsrv",
        transport="stdio",
        command=sys.executable,
        args=(str(script), str(pid_file), *extra),
    )


def _wait_for_pid_file(pid_file: Path, timeout: float = 30) -> int:
    """Return the pid the freshly spawned server wrote, waiting for it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return int(pid_file.read_text())
        except (OSError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"server never wrote {pid_file}")


def _alive(pid: int) -> bool:
    """Return True while *pid* still exists."""
    return pid_alive(pid)


def _wait_until_dead(pid: int, timeout: float = 20) -> bool:
    """Poll until *pid* exits, returning whether it did."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def manager() -> Iterator[MCPManager]:
    """A private manager (never the process-wide singleton), always closed."""
    mgr = MCPManager(idle_timeout_s=0.3, max_connections=2, health_interval_s=0.5)
    try:
        yield mgr
    finally:
        mgr.shutdown()


def test_superseded_config_does_not_orphan_its_child(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F7a: editing the config must not leave the old stdio child running."""
    first_pid_file = tmp_path / "pid1"
    conn = manager.connect(_config(server_script, first_pid_file))
    assert conn.session is not None, conn.error
    first_pid = _wait_for_pid_file(first_pid_file)

    time.sleep(0.5)  # exceed the idle timeout
    second_pid_file = tmp_path / "pid2"
    conn2 = manager.connect(_config(server_script, second_pid_file, "--verbose"))
    assert conn2.session is not None, conn2.error

    assert _wait_until_dead(first_pid), (
        "the superseded server process was orphaned and is still running"
    )
    assert len(manager._connections) == 1


def test_pool_is_capped(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F7a: the live pool never grows past the cap, oldest evicted first."""
    pids = []
    for i in range(3):
        pid_file = tmp_path / f"cap{i}"
        conn = manager.connect(_config(server_script, pid_file, f"--n={i}"))
        assert conn.session is not None, conn.error
        pids.append(_wait_for_pid_file(pid_file))

    assert len(manager._connections) <= 2
    assert _wait_until_dead(pids[0]), "least recently used child was not reaped"
    assert _alive(pids[2])


def _stop_as_the_eviction_pops(frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: inject the task Stop as ``_pop_surplus`` returns having popped a connection."""
    if (
        event == "return" and frame.f_code is MCPManager._pop_surplus.__code__
        and frame.f_locals["doomed"]
    ):
        sys.setprofile(None)
        tool_interrupt.inject_async_exception(threading.get_ident(), KeyboardInterrupt)


def _run_under_hook(
    hook: Callable[[FrameType, str, Any], Any] | None, work: Callable[[], Any],
    errors: list[BaseException],
) -> None:
    """Thread body: run *work* with the profile *hook* (if any) installed; record what it raised."""
    sys.setprofile(hook)
    try:
        work()
    except BaseException as exc:  # noqa: BLE001 — the thread records any outcome
        errors.append(exc)
    finally:
        sys.setprofile(None)


def test_stop_during_eviction_still_tears_the_evicted_child_down(
    server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """A Stop landing as the eviction pops a connection must not orphan its child.

    The eviction removed the surplus connections from the pool under
    the lock and only then, on the task thread, told each one to stop:
    a Stop delivered in between left the popped connection in no
    registry, its maintainer parked and its child alive, invisible to
    ``disconnect_all``.  Now each connection is told to stop before it
    leaves the pool.
    """
    mgr = MCPManager(idle_timeout_s=3600, max_connections=1, health_interval_s=30)
    try:
        first = mgr.connect(_config(server_script, tmp_path / "old", "--n=0"))
        assert first.session is not None, first.error
        first_pid = _wait_for_pid_file(tmp_path / "old")
        errors: list[BaseException] = []
        worker = threading.Thread(
            target=_run_under_hook,
            args=(
                _stop_as_the_eviction_pops,
                functools.partial(mgr.connect, _config(server_script, tmp_path / "new", "--n=1")),
                errors,
            ),
        )
        worker.start()
        worker.join(30)
        assert not worker.is_alive(), "connect() did not return"
        assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt), errors
        assert first.finished.wait(20) and _wait_until_dead(first_pid), (
            "the evicted connection's child survived the Stop"
        )
        assert first.stop.is_set() and first.error
    finally:
        mgr.shutdown()


def _stop_as_the_maintainer_is_submitted(frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: inject the task Stop as ``run_coroutine_threadsafe`` is entered."""
    if event == "call" and frame.f_code is asyncio.run_coroutine_threadsafe.__code__:
        sys.setprofile(None)
        tool_interrupt.inject_async_exception(threading.get_ident(), KeyboardInterrupt)


def _stop_as_the_submission_returns(frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: inject the task Stop as ``run_coroutine_threadsafe`` returns the handle."""
    if event == "return" and frame.f_code is asyncio.run_coroutine_threadsafe.__code__:
        sys.setprofile(None)
        tool_interrupt.inject_async_exception(threading.get_ident(), KeyboardInterrupt)


@pytest.mark.parametrize(
    "hook", [_stop_as_the_maintainer_is_submitted, _stop_as_the_submission_returns],
    ids=["before-queueing", "before-the-handle-is-stored"],
)
def test_stop_during_the_connection_start_does_not_poison_the_server(
    hook: Callable[[FrameType, str, Any], Any], server_script: Path, tmp_path: Path,
    real_stdin: None,
) -> None:
    """A Stop landing while a connection is started must not leave a dead record in the pool.

    ``_connection_for`` published the new record in the pool and only
    then queued its maintainer and stored the handle.  A Stop landing
    in between left a record with no task, never ready and no error:
    every later connect reused it (only ``error`` is checked), its wait
    returned at once (no task), eviction never touched it (never ready)
    and every tool call answered "not connected" for the life of the
    daemon.  When the maintainer had already been queued, it connected
    and parked for ever with its child alive, invisible to the
    shutdown.  Now an interrupted start unpublishes the record and
    stops the maintainer it may have queued.
    """
    mgr = MCPManager(idle_timeout_s=3600, max_connections=2, health_interval_s=30)
    pid_file = tmp_path / "pid"
    config = _config(server_script, pid_file)
    key = _connection_key(config)
    errors: list[BaseException] = []
    try:
        worker = threading.Thread(
            target=_run_under_hook, args=(hook, functools.partial(mgr.connect, config), errors),
        )
        worker.start()
        worker.join(30)
        assert not worker.is_alive(), "connect() did not return"
        assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt), errors
        assert key not in mgr._connections, "the interrupted start left its record in the pool"
        if pid_file.exists():
            # The maintainer was queued before the Stop: it must be stopped.
            assert _wait_until_dead(_wait_for_pid_file(pid_file)), (
                "the maintainer queued by the interrupted start parked for ever"
            )
            pid_file.unlink()
        conn = mgr.connect(config)
        assert conn.session is not None, conn.error
        assert mgr.call_tool(key, "add", {"a": 1, "b": 2}) == "3"
        assert list(mgr._connections) == [key]
    finally:
        mgr.shutdown()


def _stop_as_the_loop_thread_starts(frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: inject the task Stop as ``Thread.start`` returns to ``MCPManager.__init__``."""
    if (
        event == "return" and frame.f_code is threading.Thread.start.__code__
        and frame.f_back is not None and frame.f_back.f_code is MCPManager.__init__.__code__
    ):
        sys.setprofile(None)
        tool_interrupt.inject_async_exception(threading.get_ident(), KeyboardInterrupt)


def test_stop_as_the_loop_thread_starts_leaves_no_loop_thread_behind() -> None:
    """A Stop landing as the manager's loop thread starts must not orphan that thread.

    ``__init__`` started the thread before anything owned it: a Stop
    delivered at ``Thread.start``'s return unwound the construction,
    ``instance()`` never published the manager, no ``atexit`` hook
    held it, and the daemon thread with its running loop (selector
    and self-pipe descriptors) lived on for the rest of the process
    while the next task built another manager.
    """
    before = set(threading.enumerate())
    errors: list[BaseException] = []
    _run_under_hook(_stop_as_the_loop_thread_starts, MCPManager, errors)
    assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt), errors
    leaked = [t for t in threading.enumerate() if t.name == "mcp-manager" and t not in before]
    for thread in leaked:
        loop = thread._target.__self__  # type: ignore[attr-defined]
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
    assert not leaked, "the interrupted construction left its loop thread running"


class _PauseAtReturn:
    """Profile hook parking the thread as *code* returns, until the test resumes it.

    With *fn*, only a return of *code* whose local ``fn`` is that
    callable: the ``locked_call`` of one critical section, after its
    lock is released.
    """

    def __init__(self, code: Any, fn: Any = None) -> None:
        self.code = code
        self.fn = fn
        self.reached = threading.Event()
        self.resume = threading.Event()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if (
            event == "return" and frame.f_code is self.code
            and (self.fn is None or frame.f_locals.get("fn") == self.fn)
        ):
            sys.setprofile(None)
            self.reached.set()
            self.resume.wait(60)


def test_least_recently_used_is_evicted_at_the_cap(
    server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F7a: with nothing idle, the cap still bounds the pool by LRU."""
    mgr = MCPManager(idle_timeout_s=3600, max_connections=2, health_interval_s=30)
    try:
        pids = []
        for i in range(3):
            pid_file = tmp_path / f"lru{i}"
            conn = mgr.connect(_config(server_script, pid_file, f"--n={i}"))
            assert conn.session is not None, conn.error
            pids.append(_wait_for_pid_file(pid_file))

        assert len(mgr._connections) == 2
        assert _wait_until_dead(pids[0]), "least recently used child survived"
        assert _alive(pids[1]) and _alive(pids[2])
    finally:
        mgr.shutdown()


def test_a_running_tool_call_is_not_evicted_by_the_cap(
    server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """A server executing a tool survives other agents connecting.

    The pool is capped, and every ``connect()`` evicts the oldest
    connection past the cap.  ``call_tool`` blocks for as long as the
    tool runs — up to five minutes — so the connection it is using is
    frequently the oldest one.  Tearing it down turns a valid, still
    running call into an error the agent can do nothing about.
    """
    mgr = MCPManager(idle_timeout_s=3600, max_connections=2, health_interval_s=30)
    started = tmp_path / "tool-started"
    release = tmp_path / "tool-release"
    answer: list[str] = []
    try:
        busy_pid_file = tmp_path / "busy_pid"
        busy = _config(server_script, busy_pid_file, "--busy")
        busy_key = _connection_key(busy)
        assert mgr.connect(busy).session is not None
        busy_pid = _wait_for_pid_file(busy_pid_file)

        caller = threading.Thread(
            target=lambda: answer.append(
                mgr.call_tool(
                    busy_key,
                    "block_until",
                    {"started": str(started), "release": str(release)},
                )
            ),
            name="mcp-busy-caller",
            daemon=True,
        )
        caller.start()
        deadline = time.time() + 60
        while time.time() < deadline and not started.exists():
            time.sleep(0.02)
        assert started.exists(), "the tool call never reached the server"

        # Two more agents connect, taking the pool past its cap while
        # the first server is still executing its tool.
        for i in range(2):
            other = _config(server_script, tmp_path / f"other{i}", f"--o={i}")
            assert mgr.connect(other).session is not None

        assert _alive(busy_pid), (
            "the server executing a tool call was torn down to make "
            "room in the pool"
        )
        release.write_text("go", encoding="utf-8")
        caller.join(timeout=120)
        assert not caller.is_alive()
        assert answer == ["released"], (
            f"the in-flight tool call was broken by pool eviction: {answer}"
        )

        # Once the call is done the lease is gone and the connection is
        # an ordinary eviction candidate again — the freshest one, since
        # it was in use most recently, so it takes two more connects to
        # push it out.  Without that the cap would leak a slot forever.
        for name in ("last-a", "last-b"):
            nxt = _config(server_script, tmp_path / name, f"--{name}")
            assert mgr.connect(nxt).session is not None
        assert _wait_until_dead(busy_pid), (
            "a finished connection was never evicted, so the cap leaks"
        )
        assert len(mgr._connections) <= 2
    finally:
        release.write_text("go", encoding="utf-8")
        mgr.shutdown()


def test_call_tool_on_an_unconfigured_server_reports_it(
    manager: MCPManager,
) -> None:
    """A server the manager never saw cannot be rebuilt, and says so."""
    result = manager.call_tool("nosuchserver", "add", {"a": 1, "b": 2})
    assert "is not connected" in result
    assert "nosuchserver" in result


def test_call_tool_names_the_deadline_when_the_loop_never_answers(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call the loop never settles reports its deadline, not an empty reason.

    The outer deadline re-raised a bare ``TimeoutError`` whose text is
    empty, so the model was told ``Error: MCP tool call failed: `` and
    nothing else, on the one path (a wedged manager loop) where the
    reason matters.
    """
    monkeypatch.setattr(mcp_servers, "CALL_TIMEOUT", 0.1)
    config = _config(server_script, tmp_path / "pid")
    assert manager.connect(config).session is not None
    blocked = threading.Event()
    # The loop is held: the call queued behind this never starts.
    manager._loop.call_soon_threadsafe(blocked.wait, 30)
    try:
        result = manager.call_tool(_connection_key(config), "add", {"a": 1, "b": 2})
    finally:
        blocked.set()
    assert result == "Error: MCP tool call failed: no answer within 5.1s"


def test_call_tool_reconnects_after_the_server_dies(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F7c: a crashed stdio server is reconnected on the next tool call."""
    pid_file = tmp_path / "crash_pid"
    config = _config(server_script, pid_file)
    key = _connection_key(config)
    conn = manager.connect(config)
    assert conn.session is not None, conn.error
    assert manager.call_tool(key, "add", {"a": 1, "b": 2}) == "3"

    pid = _wait_for_pid_file(pid_file)
    pid_file.unlink()
    os.kill(pid, 9)
    assert _wait_until_dead(pid)

    deadline = time.time() + 20
    while time.time() < deadline and conn.session is not None:
        time.sleep(0.05)
    assert conn.session is None, "connection did not notice the dead server"

    assert manager.call_tool(key, "add", {"a": 2, "b": 3}) == "5"
    assert _wait_for_pid_file(pid_file) != pid
    assert len(manager._connections) == 1


def test_a_closed_transport_retires_the_connection_before_any_health_ping(
    server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """A dead server is rebuilt by the call after the ones that found it dead.

    No health ping is due (ten minutes): the calls' own closed-transport
    error must take the connection out of the pool, and once only,
    though two calls were leased on it (the loop is held until both
    are).  A server that dies under a call fails that call alone.
    """
    manager = MCPManager(health_interval_s=600)
    blocked = threading.Event()
    started = tmp_path / "started"
    try:
        pid_file = tmp_path / "closed_pid"
        config = _config(server_script, pid_file)
        key = _connection_key(config)
        conn = manager.connect(config)
        assert conn.session is not None, conn.error
        pid = _wait_for_pid_file(pid_file)
        pid_file.unlink()
        os.kill(pid, 9)
        assert _wait_until_dead(pid)
        manager._loop.call_soon_threadsafe(blocked.wait, 60)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            calls = [
                pool.submit(manager.call_tool, key, "add", {"a": 1, "b": 2}) for _ in range(2)
            ]
            deadline = time.time() + 30
            while time.time() < deadline and conn.in_flight < 2:
                time.sleep(0.02)
            blocked.set()
            found_dead = [call.result(60) for call in calls]
            assert manager.call_tool(key, "add", {"a": 2, "b": 3}) == "5", found_dead
            dying = pool.submit(
                manager.call_tool, key, "block_until",
                {"started": str(started), "release": str(tmp_path / "never")},
            )
            deadline = time.time() + 30
            while time.time() < deadline and not started.exists():
                time.sleep(0.02)
            rebuilt = _wait_for_pid_file(pid_file)
            pid_file.unlink()
            os.kill(rebuilt, 9)
            found_dead.append(dying.result(60))
        assert all(a.startswith("Error: MCP tool call failed: ") for a in found_dead), found_dead
        assert manager.call_tool(key, "add", {"a": 2, "b": 3}) == "5"
        assert _wait_for_pid_file(pid_file) not in (pid, rebuilt)
        deadline = time.time() + 20
        while time.time() < deadline and manager._orphans:
            time.sleep(0.02)
        assert manager._orphans == [] and conn.finished.is_set() and conn.in_flight == 0
        assert len(manager._connections) == 1
    finally:
        blocked.set()
        manager.shutdown()


def test_a_call_in_flight_ends_with_its_connection(
    server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """A tool call still running when its connection is closed is failed with it.

    ``disconnect_all`` unwinds the session under the call.  The SDK
    answers its pending requests only when the server closes the
    transport, so the call waited out its 300 s read timeout for an
    answer from a session that no longer existed.
    """
    manager = MCPManager(health_interval_s=600)
    started = tmp_path / "started"
    answer: list[str] = []
    try:
        config = _config(server_script, tmp_path / "ending_pid")
        key = _connection_key(config)
        assert manager.connect(config).session is not None
        caller = threading.Thread(
            target=lambda: answer.append(
                manager.call_tool(
                    key, "block_until",
                    {"started": str(started), "release": str(tmp_path / "never")},
                )
            ),
            name="mcp-ending-caller",
            daemon=True,
        )
        caller.start()
        deadline = time.time() + 30
        while time.time() < deadline and not started.exists():
            time.sleep(0.02)
        assert started.exists(), "the tool call never reached the server"
        manager.disconnect_all()
        caller.join(timeout=60)
        assert answer == ["Error: MCP tool call failed: connection closed"], answer
    finally:
        manager.shutdown()


class _SSEServer:
    """A real FastMCP server over SSE, under uvicorn on a loopback port, that can refuse POSTs.

    While ``outage`` is set every POST (a message of the client) is
    answered 503 and counted in ``refused``.  ``ran`` names the tools
    whose bodies ran.
    """

    def __init__(self) -> None:
        self.outage = threading.Event()
        self.holding = threading.Event()
        self.refused = 0
        self.ran: list[str] = []
        mcp = FastMCP("ssesrv")
        mcp.add_tool(self.echo)
        mcp.add_tool(self.hold)
        self._app = mcp.sse_app()
        listener = socket.create_server(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/sse"
        self.config = MCPServerConfig(name="ssesrv", transport="sse", url=url)
        self._server = uvicorn.Server(uvicorn.Config(self, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, args=([listener],), daemon=True)
        self._thread.start()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Serve one ASGI request: a POST during the outage is refused, the rest is FastMCP's."""
        if self.outage.is_set() and scope.get("method") == "POST":
            self.refused += 1
            await JSONResponse({"error": "outage"}, 503)(scope, receive, send)
        else:
            await self._app(scope, receive, send)

    def echo(self, text: str) -> str:
        """Return *text*."""
        self.ran.append("echo")
        return text

    async def hold(self) -> str:
        """Announce the call, then keep it in flight until the server stops."""
        self.ran.append("hold")
        self.holding.set()
        while not self._server.should_exit:
            await asyncio.sleep(0.02)
        return "stopped"

    def stop(self) -> None:
        """Stop the server and wait for its thread."""
        self._server.should_exit = True
        self._thread.join(timeout=10)


def test_a_refused_post_ends_an_sse_connection_and_its_calls_at_once() -> None:
    """A POST an SSE server refuses ends the connection with that refusal, and its calls with it.

    The SDK's SSE writer logged a failed POST and stopped, its reader
    left running: the connection looked alive and could send nothing more.
    The refused tool call waited out its 300 s read timeout for the
    answer to a request the server never got, a refused handshake the
    60 s connect deadline.  No call is sent again (the server saw one
    refused POST and ran no ``echo``): the next one starts over on a new
    connection, whose handshake fails as fast while the outage lasts.
    """
    server, manager = _SSEServer(), MCPManager(health_interval_s=600)
    pool = concurrent.futures.ThreadPoolExecutor(2)
    try:
        key = _connection_key(server.config)
        conn = manager.connect(server.config)
        assert conn.session is not None, conn.error
        held = pool.submit(manager.call_tool, key, "hold", {})
        assert server.holding.wait(30), "the tool call never reached the server"
        server.outage.set()
        refused = pool.submit(manager.call_tool, key, "echo", {"text": "hi"})
        answers = [refused.result(30), held.result(30)]
        assert conn.finished.wait(10) and "503 Service Unavailable" in conn.error, conn.error
        assert answers == [f"Error: MCP tool call failed: {conn.error}"] * 2, answers
        assert (server.refused, server.ran) == (1, ["hold"]), "the refused call was sent again"
        started = time.monotonic()
        answer = manager.call_tool(key, "echo", {"text": "hi"})
        assert "503 Service Unavailable" in answer and time.monotonic() - started < 30, answer
        server.outage.clear()
        assert manager.call_tool(key, "echo", {"text": "hi"}) == "hi"
    finally:
        pool.shutdown(wait=False)
        manager.shutdown()
        server.stop()


def test_health_ping_does_not_tear_down_a_busy_connection(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """A long sync tool call outlives several health intervals unharmed.

    FastMCP runs a sync tool handler inline on the server's loop, so the
    server cannot answer a ping while ``block_until`` runs.  With the
    manager's 0.5 s health interval a 1.5 s call spans three ping slots;
    a ping sent during the call times out, unwinds the session under
    the pending ``call_tool`` and kills a healthy, busy server.  The
    ping must be skipped while a call is in flight.
    """
    pid_file = tmp_path / "busy_pid"
    config = _config(server_script, pid_file)
    key = _connection_key(config)
    conn = manager.connect(config)
    assert conn.session is not None, conn.error
    pid = _wait_for_pid_file(pid_file)

    started = tmp_path / "ping-started"
    release = tmp_path / "ping-release"
    answer: list[str] = []
    caller = threading.Thread(
        target=lambda: answer.append(
            manager.call_tool(
                key, "block_until", {"started": str(started), "release": str(release)},
            )
        ),
        name="mcp-ping-caller",
        daemon=True,
    )
    caller.start()
    try:
        deadline = time.time() + 60
        while time.time() < deadline and not started.exists():
            time.sleep(0.02)
        assert started.exists(), "the tool call never reached the server"
        time.sleep(1.5)  # three health intervals with the server's loop blocked
    finally:
        release.write_text("go", encoding="utf-8")
    caller.join(timeout=120)
    assert not caller.is_alive()
    assert answer == ["released"], f"the in-flight call was broken by a ping: {answer}"
    assert not conn.error, conn.error
    assert conn.session is not None, "the busy connection was torn down"
    assert manager._connections.get(key) is conn
    assert _alive(pid), "the busy server was killed"
    # The connection is idle again: it must still answer and keep pinging fine.
    assert manager.call_tool(key, "add", {"a": 1, "b": 2}) == "3"


#: A stdio server written against the wire protocol alone, as servers outside
#: the SDKs are: it answers what a tool call needs and "Method not found" to
#: everything else, ``ping`` included.
_NO_PING_SERVER_SCRIPT = '''
import json, os, sys

open(sys.argv[1], "w").write(str(os.getpid()))
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue  # a notification
    reply = {"jsonrpc": "2.0", "id": msg["id"]}
    if msg["method"] == "initialize":
        reply["result"] = {
            "protocolVersion": msg["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "noping", "version": "1"},
        }
    elif msg["method"] == "tools/list":
        reply["result"] = {"tools": [{"name": "pid", "inputSchema": {"type": "object"}}]}
    elif msg["method"] == "tools/call":
        reply["result"] = {"content": [{"type": "text", "text": str(os.getpid())}]}
    else:
        reply["error"] = {"code": -32601, "message": "Method not found"}
    print(json.dumps(reply), flush=True)
'''


def test_a_server_without_ping_keeps_its_idle_connection(
    manager: MCPManager, tmp_path: Path, real_stdin: None,
) -> None:
    """A "Method not found" answer to the idle health ping proves the server alive.

    Only the SDKs' servers are sure to implement ``ping``.  The error
    answer unwound the connection every health interval: the child was
    killed and spawned again for the next call, and whatever state it
    kept (a browser page, a login) went with it.
    """
    script = tmp_path / "noping.py"
    script.write_text(_NO_PING_SERVER_SCRIPT, encoding="utf-8")
    pid_file = tmp_path / "noping_pid"
    config = _config(script, pid_file)
    key = _connection_key(config)
    conn = manager.connect(config)
    assert conn.session is not None, conn.error
    pid = _wait_for_pid_file(pid_file)
    time.sleep(1.6)  # three health intervals, idle
    assert (conn.error, conn.session is not None) == ("", True), "the ping's answer tore it down"
    assert manager.call_tool(key, "pid", {}) == str(pid)


def test_shutdown_right_after_construction_stops_the_loop_thread() -> None:
    """``shutdown()`` must stop the loop even before ``run_forever`` began.

    ``__init__`` starts the loop thread and returns; the loop reports
    ``is_running()`` only once the thread has entered ``run_forever``.
    A shutdown that returned early in that window left the daemon thread
    spinning for the rest of the process.  Many back-to-back cycles make
    the window likely to be hit at least once.

    The stopped loop is also closed (its selector and self-pipe
    descriptors released) and the manager's ``atexit`` hook withdrawn,
    so a manager rebuilt after every shutdown leaks neither
    descriptors nor the old manager object.
    """
    for _ in range(49):
        _shut_down_new_manager()
    ref = weakref.ref(_shut_down_new_manager())
    gc.collect()
    assert ref() is None, "shutdown() left its atexit hook holding the manager"


def _shut_down_new_manager() -> MCPManager:
    """Build a manager, shut it down at once and check its loop thread and loop are gone."""
    mgr = MCPManager(idle_timeout_s=0.3, max_connections=2, health_interval_s=0.5)
    mgr.shutdown()
    mgr._thread.join(timeout=10)
    assert not mgr._thread.is_alive(), "manager loop thread outlived shutdown()"
    assert not mgr._loop.is_running()
    assert mgr._loop.is_closed(), "shutdown() left the stopped loop open"
    return mgr


#: The server above, kept alive after stdin EOF by a non-daemon thread so
#: its teardown needs the SDK's SIGTERM instead of ending at once.
_LINGERING_SERVER_SCRIPT = _SERVER_SCRIPT.replace(
    "import os, sys, time", "import os, sys, threading, time",
).replace(
    "    mcp.run()", "    threading.Thread(target=time.sleep, args=(60,)).start()\n    mcp.run()",
)


def test_shutdown_waits_for_an_evicted_connection_to_unwind(
    server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """``shutdown()`` right after an eviction must still reap the evicted child.

    The eviction popped the connection from the pool and told it to
    stop but registered it nowhere, so a shutdown that followed while
    its transport was still closing found nothing to wait for and
    stopped the loop underneath that teardown: a stdio child that
    lingers after stdin EOF (it needs the SDK's SIGTERM) outlived the
    manager.
    """
    lingering = tmp_path / "lsrv.py"
    lingering.write_text(_LINGERING_SERVER_SCRIPT, encoding="utf-8")
    mgr = MCPManager(idle_timeout_s=3600, max_connections=1, health_interval_s=30)
    old_pid = 0
    try:
        first = mgr.connect(_config(lingering, tmp_path / "old", "--n=0"))
        assert first.session is not None, first.error
        old_pid = _wait_for_pid_file(tmp_path / "old")
        second = mgr.connect(_config(server_script, tmp_path / "new", "--n=1"))
        assert second.session is not None, second.error
        assert _connection_key(first.config) not in mgr._connections, "not evicted"
        mgr.shutdown()
        assert first.finished.is_set(), (
            "shutdown() returned while the evicted connection was still unwinding"
        )
        assert not _alive(old_pid), "the evicted child outlived the manager"
    finally:
        mgr.shutdown()
        if _alive(old_pid):
            os.kill(old_pid, 9)


def test_shutdown_waits_for_a_concurrent_disconnect_to_finish(
    tmp_path: Path, real_stdin: None,
) -> None:
    """``shutdown()`` during another thread's ``disconnect_all()`` must not close the loop under it.

    ``disconnect_all`` took every connection out of the registries and
    then, outside the lock, told each to stop and waited for the
    teardown.  A ``shutdown`` meanwhile (the daemon's shutdown path
    against the ``atexit`` hook, or two shutdown paths) found empty
    registries, stopped and closed the loop at once: the disconnect
    failed with "Event loop is closed" and the stdio children whose
    transports were still closing outlived the manager.
    """
    lingering = tmp_path / "lsrv.py"
    lingering.write_text(_LINGERING_SERVER_SCRIPT, encoding="utf-8")
    mgr = MCPManager(idle_timeout_s=3600, max_connections=2, health_interval_s=30)
    pid = 0
    pause = _PauseAtReturn(locked_call.__code__, fn=mgr._take_all)
    errors: list[BaseException] = []
    try:
        conn = mgr.connect(_config(lingering, tmp_path / "pid"))
        assert conn.session is not None, conn.error
        pid = _wait_for_pid_file(tmp_path / "pid")
        drainer = threading.Thread(target=_run_under_hook, args=(pause, mgr.disconnect_all, errors))
        drainer.start()
        assert pause.reached.wait(30), "disconnect_all never drained the pool"
        closer = threading.Thread(target=mgr.shutdown)
        closer.start()
        # The shutdown has nothing to find in the registries: it must
        # wait for the drain instead of returning with the loop closed.
        closer.join(2)
        pause.resume.set()
        drainer.join(60)
        closer.join(60)
        assert not drainer.is_alive() and not closer.is_alive()
        assert errors == [], errors
        assert conn.finished.is_set(), "the loop was stopped underneath the disconnect"
        assert not _alive(pid), "the child outlived the manager"
        assert mgr._loop.is_closed()
    finally:
        pause.resume.set()
        mgr.shutdown()
        if _alive(pid):
            os.kill(pid, 9)


def test_token_lock_does_not_stall_other_mcp_calls(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F8: a foreign process holding the token lock must not block the loop."""
    import asyncio

    from mcp.shared.auth import OAuthToken

    config = _config(server_script, tmp_path / "tok_pid")
    key = _connection_key(config)
    assert manager.connect(config).session is not None

    storage = FileTokenStorage("gsrv")
    holder_script = tmp_path / "holder.py"
    holder_script.write_text(_LOCK_HOLDER, encoding="utf-8")
    ready = tmp_path / "held"
    holder = subprocess.Popen(
        [sys.executable, str(holder_script), str(storage._lock_path), str(ready)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 60
        while time.time() < deadline and not ready.exists():
            time.sleep(0.02)
        assert ready.exists(), "lock holder never started"

        future = asyncio.run_coroutine_threadsafe(
            storage.set_tokens(OAuthToken(access_token="a", token_type="Bearer")),
            manager._loop,
        )
        time.sleep(0.3)

        result: list[str] = []
        start = time.monotonic()
        thread = threading.Thread(
            target=lambda: result.append(
                manager.call_tool(key, "add", {"a": 4, "b": 4}),
            ),
        )
        thread.start()
        thread.join(timeout=30)
        elapsed = time.monotonic() - start
        assert result == ["8"], result
        assert elapsed < 2.5, (
            f"an unrelated MCP call waited {elapsed:.1f}s behind a foreign "
            f"process's token lock"
        )
        future.result(timeout=30)
    finally:
        holder.wait(timeout=30)


def test_spawn_survives_a_closed_stderr(
    manager: MCPManager,
    server_script: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed ``sys.stderr`` must not stop an MCP server from starting.

    ``mcp.client.stdio.stdio_client`` declares ``errlog: TextIO =
    sys.stderr``.  A default argument is evaluated once, when the module
    is imported, and the SDK passes it straight to the child's
    ``stderr=``.  So the very first import pins one stream object for the
    life of the process, and anything that later closes or replaces it --
    a log redirect in the long-lived ``kiss-web`` daemon, or pytest's
    per-test capture being torn down -- makes the next spawn die with
    ``ValueError: I/O operation on closed file``.

    This drives the real transport with a genuinely closed
    ``sys.stderr`` and requires a real FastMCP child to come up anyway.
    """
    # ``os.devnull`` has a real descriptor on every platform (Windows has
    # no pty) and nothing reads stdin while the client talks to the child.
    stdin_stream = open(os.devnull, encoding="utf-8")
    dead = (tmp_path / "dead_errlog.txt").open("w", encoding="utf-8")
    dead.close()
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", dead)
    try:
        pid_file = tmp_path / "closed-stderr-pid"
        conn = manager.connect(_config(server_script, pid_file))
        assert conn.session is not None, conn.error
        assert _alive(_wait_for_pid_file(pid_file))
    finally:
        stdin_stream.close()


def _settled_fd_count(timeout: float = 5) -> int:
    """This process's open descriptor count once it has stopped changing."""
    last, stable_since = -1, time.monotonic()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        count = len(os.listdir("/proc/self/fd"))
        if count != last:
            last, stable_since = count, time.monotonic()
        elif time.monotonic() - stable_since > 0.5:
            break
        time.sleep(0.05)
    return last


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="counts fds through /proc")
def test_respawns_with_a_closed_stderr_do_not_leak_descriptors(
    manager: MCPManager,
    server_script: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stderr fallback is one stream per process, not one per spawn.

    With ``sys.stderr`` closed every spawn used to ``dup`` the real
    stderr into a new file object that nothing ever closed; every
    reconnect of the daemon (idle eviction, a crashed server) then cost
    one descriptor for good.  Real children are crashed and respawned
    through ``call_tool``; the descriptor count must come back to where
    it was after the first spawn.
    """
    stdin_stream = open(os.devnull, encoding="utf-8")
    dead = (tmp_path / "dead_errlog.txt").open("w", encoding="utf-8")
    dead.close()
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", dead)
    try:
        assert _child_errlog() is _child_errlog()
        pid_file = tmp_path / "respawn-pid"
        config = _config(server_script, pid_file)
        key = _connection_key(config)
        conn = manager.connect(config)
        assert conn.session is not None, conn.error
        assert manager.call_tool(key, "add", {"a": 1, "b": 2}) == "3"
        baseline = _settled_fd_count()
        for _ in range(4):
            conn = manager._connections[key]  # a reconnect replaces the record
            pid = _wait_for_pid_file(pid_file)
            pid_file.unlink()
            os.kill(pid, 9)
            assert _wait_until_dead(pid)
            deadline = time.time() + 20
            while time.time() < deadline and conn.session is not None:
                time.sleep(0.05)
            assert conn.session is None, "connection did not notice the dead server"
            assert manager.call_tool(key, "add", {"a": 2, "b": 3}) == "5"
        assert _settled_fd_count() <= baseline
    finally:
        stdin_stream.close()
