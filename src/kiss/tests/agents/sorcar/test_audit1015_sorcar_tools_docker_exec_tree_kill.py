# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real-container regressions of the unified docker ``Bash`` engine.

Audit 2026-10-15 (docker-fix).  Unifying the two ``DockerManager.Bash``
engines into one ``_exec`` lost two behaviours and exposed a third:

1. A timed-out or stopped command handed its kill to the reaper, which
   returned as soon as the exec's own process had finished; a command
   whose shell exits at once but leaves background children behind
   (``sleep 71 &``) kept those children alive, since only a sweep of the
   token-tagged process tree kills them.  The cancel path now sweeps the
   tree at once (the reaper's first ``exec_inspect`` can block for up to
   2 s while the daemon waits for the exec's streams to close) and the
   reaper sweeps again once the exec has started, finished or not.
2. ``_drain_exec_stream`` logged and swallowed a stream-read failure, so
   the worker reported a clean end of stream.  It propagates again.  No
   test drives that branch: the SDK's ``CancellableStream.__next__``
   (docker 7.2, ``docker/types/daemon.py``) turns every ``OSError`` and
   ``ProtocolError`` into ``StopIteration``, so with real objects a
   dropped daemon connection is a plain end of stream — faulting the
   stream would need a fake, which these tests do not use.
3. What a dropped connection really does: the stream ends while the exec
   is still running, and ``_exec`` turned ``exec_inspect``'s
   ``ExitCode: None`` into ``TypeError: int() ... NoneType``.  ``_exec``
   now waits for the exit code within the deadline and reports it with
   the output received before the drop; the daemon records 126 for an
   exec whose client went away, so the result is a failure, not a
   truncated success.  The connection is dropped for real here by a
   unix-socket proxy between docker-py and the daemon.

Every test runs a real exec in a real container; nothing is mocked.
"""

import os
import re
import socket
import tempfile
import threading
import time
import unittest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available
from kiss.tests.agents.sorcar.test_docker_nonstream_timeout_kill import (
    _count_processes_script,
)

_DOCKER_SOCKET = "/var/run/docker.sock"
_EXEC_START = re.compile(rb"POST /[^ ]*/exec/[^/ ]+/start HTTP/1\.1\r\n")


class _DockerSocketProxy:
    """A unix-socket proxy to the docker daemon that can drop exec streams.

    Each client connection is forwarded byte for byte on its own
    threads.  When :attr:`cut_exec_stream_after` is set, a connection
    whose request line is an exec start (``POST /.../exec/<id>/start``)
    is shut down that many seconds after it was opened, the way a
    daemon restart or a dying network path would end it.
    """

    def __init__(self) -> None:
        self.cut_exec_stream_after: float | None = None
        self.directory = tempfile.mkdtemp(prefix="kiss-docker-proxy-")
        self.path = os.path.join(self.directory, "docker.sock")
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(32)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def close(self) -> None:
        """Stop accepting connections and remove the socket file."""
        self._server.close()
        os.unlink(self.path)
        os.rmdir(self.directory)

    def _accept_loop(self) -> None:
        while True:
            try:
                client, _ = self._server.accept()
            except OSError:  # closed
                return
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client: socket.socket) -> None:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        upstream.connect(_DOCKER_SOCKET)
        threading.Thread(target=_pump, args=(upstream, client), daemon=True).start()
        try:
            while True:
                data = client.recv(65536)
                if not data:
                    break
                upstream.sendall(data)
                # Inspect every request, not just the first: docker-py
                # may send the exec start on a pooled keep-alive connection.
                if self.cut_exec_stream_after is not None and _EXEC_START.match(data):
                    threading.Timer(
                        self.cut_exec_stream_after, _shutdown_both, args=(client, upstream),
                    ).start()
        except OSError:
            pass
        finally:
            _shutdown_both(client, upstream)


def _pump(source: socket.socket, sink: socket.socket) -> None:
    """Copy bytes from *source* to *sink* until either side ends."""
    try:
        while True:
            data = source.recv(65536)
            if not data:
                break
            sink.sendall(data)
    except OSError:
        pass
    finally:
        _shutdown_both(source, sink)


def _shutdown_both(first: socket.socket, second: socket.socket) -> None:
    """Shut down both sockets, ignoring ones that are already closed."""
    for sock in (first, second):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestExecTreeKill(unittest.TestCase):
    """Timeout and Stop kill the whole token-tagged tree, not just the shell."""

    env: DockerManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("python:3.11-slim")
        cls.env.open()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.streamed: list[str] = []

    def tearDown(self) -> None:
        self.env.stream_callback = None
        self.env.stop_event = None

    def _wait_until_no_process(self, needle_head: str, needle_tail: str) -> None:
        """Poll the container until no process matches the needle."""
        script = _count_processes_script(needle_head, needle_tail)
        deadline = time.monotonic() + 20
        count = ""
        while time.monotonic() < deadline:
            count = self.env.Bash(script, "count leftover processes")
            count = count.strip().splitlines()[-1]
            if count == "0":
                return
            time.sleep(0.5)
        self.fail(f"command still running in the container ({count} processes)")

    def _marker_absent_after_timeout(self, marker: str) -> None:
        """The background child must not reach ``touch`` after the timeout.

        The child sleeps 4 s (not 2 as in the reviewer's reproduction):
        the sweep is one ``exec_run``, which a daemon busy with other
        test containers can take over a second to run, and the point
        is whether the child dies, not how fast the daemon answers.
        For the same reason ``started`` is not required in the output:
        the 1 s deadline includes the exec's startup, and on a busy
        daemon the shell may not have printed yet.
        """
        result = self.env.Bash(
            f"rm -f {marker}; (sleep 4; touch {marker}) & echo started; exit 0",
            "background child outlives the shell",
            timeout_seconds=1,
        )
        self.assertIn("timed out", result)
        time.sleep(5)
        check = self.env.Bash(f"test -e {marker} && echo present || echo absent", "check")
        self.assertIn("absent", check)

    def test_timeout_kills_background_child_of_finished_shell_streaming(self) -> None:
        """Streaming ``Bash``: the orphaned child dies with the timeout."""
        self.env.stream_callback = self.streamed.append
        self._marker_absent_after_timeout("/tmp/after-timeout-stream")

    def test_timeout_kills_background_child_of_finished_shell_plain(self) -> None:
        """Non-streaming ``Bash``: the orphaned child dies with the timeout."""
        self._marker_absent_after_timeout("/tmp/after-timeout-plain")

    def test_timeout_kills_orphaned_sleep(self) -> None:
        """``sleep 71 &`` left behind by an exited shell is killed, not left for 71 s.

        The echo is not required in the output: the 1 s deadline includes
        the exec's startup, which on a busy daemon can take longer than
        that, and the assertion that matters is that the orphan dies.
        """
        self.env.stream_callback = self.streamed.append
        result = self.env.Bash("sleep 71 & echo CHILD=$!; exit 0", "orphan", timeout_seconds=1)
        self.assertIn("timed out", result)
        self._wait_until_no_process("sle", "ep 71")

    def test_stop_kills_background_child_of_finished_shell(self) -> None:
        """A task Stop sweeps the tree the same way."""
        self.env.stream_callback = self.streamed.append
        stop = threading.Event()
        self.env.stop_event = stop
        threading.Timer(1.0, stop.set).start()
        result = self.env.Bash("sleep 74 & echo CHILD=$!; exit 0", "orphan", timeout_seconds=60)
        self.env.stop_event = None
        self.assertIn("Killed: the task was stopped.", result)
        self._wait_until_no_process("sle", "ep 74")


@unittest.skipUnless(
    is_docker_available() and os.path.exists(_DOCKER_SOCKET),
    "Docker daemon is not reachable through /var/run/docker.sock",
)
class TestDroppedExecStream(unittest.TestCase):
    """The daemon connection drops mid-stream; the exec keeps running."""

    env: DockerManager
    proxy: _DockerSocketProxy
    saved_docker_host: str | None

    @classmethod
    def setUpClass(cls) -> None:
        cls.proxy = _DockerSocketProxy()
        cls.saved_docker_host = os.environ.get("DOCKER_HOST")
        os.environ["DOCKER_HOST"] = "unix://" + cls.proxy.path
        try:
            cls.env = DockerManager("python:3.11-slim")
            cls.env.open()
        finally:
            if cls.saved_docker_host is None:
                del os.environ["DOCKER_HOST"]
            else:
                os.environ["DOCKER_HOST"] = cls.saved_docker_host

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()
        cls.proxy.close()

    def tearDown(self) -> None:
        self.proxy.cut_exec_stream_after = None
        self.env.stream_callback = None

    def test_dropped_stream_waits_for_exit_code(self) -> None:
        """Output before the drop plus the real exit code, no crash, no early return."""
        streamed: list[str] = []
        self.env.stream_callback = streamed.append
        self.proxy.cut_exec_stream_after = 1.0
        start = time.monotonic()
        result = self.env.Bash("echo first; sleep 3; echo last; exit 5", "drop", timeout_seconds=10)
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 2.5)  # waited for the process, not the stream
        self.assertEqual(streamed, ["first\n"])
        # The daemon (moby ``daemon/exec.go``, ``ContainerExecStart``)
        # records exit code 126 for an exec whose attached client went
        # away unless the process's own exit code was recorded first:
        # it only notices the dead client when it writes "last", which
        # races the exit, so Docker 29 reports 5 or 126 (older daemons
        # always 126).  Either way the exec is not reported as success.
        lines = result.strip().splitlines()
        self.assertEqual(lines[:2], ["first", ""])
        self.assertIn(lines[2:], (["[exit code: 5]"], ["[exit code: 126]"]))
        self.proxy.cut_exec_stream_after = None
        self.assertEqual(self.env.Bash("echo back", "proxy still forwards").strip(), "back")

    def test_dropped_stream_in_run_commands_parallel(self) -> None:
        """The parallel runner reports a command whose stream dropped as failed, after waiting."""
        self.proxy.cut_exec_stream_after = 1.0
        # ``exit 7`` because a daemon that records the process's own exit
        # code before it notices the dead client (Docker 29, see above)
        # would otherwise report the truncated command as exit 0.
        report = self.env.run_commands_parallel(
            '["echo p; sleep 3; echo q; exit 7", "echo quick"]', timeout_seconds=10,
        )
        self.assertIn("2 commands: 1 succeeded, 1 failed, 0 timed out.", report)
        # The wall time includes the exec's startup and the exit-code
        # inspection, so it is "3.x s" on an idle daemon and more on a
        # busy one; what matters is that it waited for the process.
        seconds = re.search(r"exit (?:7|126) in ([0-9.]+)s", report)
        self.assertIsNotNone(seconds, report)
        assert seconds is not None
        self.assertGreaterEqual(float(seconds.group(1)), 2.5)
        self.assertIn("$ echo p; sleep 3; echo q; exit 7\np\n", report)


if __name__ == "__main__":
    unittest.main()
