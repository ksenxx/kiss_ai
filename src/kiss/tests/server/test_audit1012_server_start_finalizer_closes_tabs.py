# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The blocking ``start()`` finalizer closes the Browser tab and remote streams.

``stop_async`` (embedders, tests) shuts the streamed browser down and
closes every remote-machine stream; the daemon's real exit path, the
``finally`` of the blocking ``start()`` (SIGTERM, Ctrl-C), hung the
terminal shells up but left both behind: the ``kiss-browser-tab`` loop
thread was never stopped (so Playwright's driver killed Chromium
instead of closing its persistent context) and the relay sockets of
tabs bound to a remote machine stayed open until the process died.

The server is started with ``start()`` on a thread, as the lifecycle
tests do; the browser service's private loop is warmed the way the
first open does, and one real stream to a real remote daemon is
registered; stopping the loop runs the finalizer.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar import machines as mach
from kiss.core.vscode_config import save_config
from kiss.server.remote_tabs import RemoteTab
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server._blocking_start import close_leaked_listeners
from kiss.tests.server.conftest import find_free_port
from kiss.tests.server.test_blocking_start_releases_port import (
    _make_server,
    _start_on_thread,
    _stop_thread,
)


def test_stopped_start_shuts_browser_and_remote_tabs(tmp_path: Path) -> None:
    save_config({"remote_password": ""})
    mach.save_machines([])
    remote = start_remote_daemon(tmp_path / "remote-home", password="pw-remote")
    try:
        machine = mach.add_machine("box", remote.url, "pw-remote")
        server = _make_server(tmp_path, "finalized", find_free_port())
        thread = _start_on_thread(server)
        browser_tabs = server._vscode_server.browser_tabs
        browser_tabs._ensure_loop()
        browser_thread = browser_tabs._thread
        assert browser_thread is not None and browser_thread.is_alive()
        remote_tabs = server._server_api.remote_tabs
        tab = RemoteTab("tab-1", machine, remote_tabs)
        with remote_tabs._lock:
            remote_tabs._tabs["tab-1"] = tab
        try:
            _stop_thread(server, thread)
        finally:
            close_leaked_listeners(server)

        assert browser_tabs._closed, "start() returned without shutting the browser service"
        assert not browser_thread.is_alive(), "the kiss-browser-tab loop thread outlived start()"
        assert tab._closed, "start() returned without closing the remote stream"
        tab._thread.join(timeout=10)
        assert not tab._thread.is_alive(), "the remote relay thread outlived start()"
        with remote_tabs._lock:
            assert remote_tabs._tabs == {}
    finally:
        remote.stop()
        mach.close_links()
        mach.save_machines([])
