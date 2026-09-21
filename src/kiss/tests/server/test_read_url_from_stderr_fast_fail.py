"""``_read_url_from_stderr`` returns as soon as the process exits.

Before the fix the stderr drain thread only signalled the waiter when a
URL was found; a ``cloudflared`` that died at once (bad binary, exit 1,
missing config) still made the caller sit out the full URL timeout of
30 s before it could print "tunnel failed to start" and carry on.
"""

from __future__ import annotations

import subprocess
import sys
import time

from kiss.server.web_server import _parse_quick_tunnel_url, _read_url_from_stderr


def _spawn(code: str) -> subprocess.Popen[str]:
    """Start a Python child running *code* with a piped text stderr."""
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
    )


def test_returns_none_promptly_when_process_exits_without_url() -> None:
    proc = _spawn("import sys; print('ERR connection refused', file=sys.stderr)")
    try:
        started = time.monotonic()
        flag = [False]
        url = _read_url_from_stderr(
            proc, _parse_quick_tunnel_url, timeout=30.0, rate_limit_flag=flag,
        )
        elapsed = time.monotonic() - started
    finally:
        proc.wait(timeout=10)
    assert url is None
    assert flag[0] is False
    assert elapsed < 10.0, f"waited {elapsed:.1f}s for a process that had exited"


def test_rate_limit_flag_is_final_when_process_exits_without_url() -> None:
    proc = _spawn(
        "import sys; print('ERR error code: 1015 status_code=\"429\"', file=sys.stderr)"
    )
    try:
        flag = [False]
        url = _read_url_from_stderr(
            proc, _parse_quick_tunnel_url, timeout=30.0, rate_limit_flag=flag,
        )
    finally:
        proc.wait(timeout=10)
    assert url is None
    assert flag[0] is True


def test_url_found_before_exit_is_still_returned() -> None:
    proc = _spawn(
        "import sys, time; "
        "print('INF https://fast-fail.trycloudflare.com', file=sys.stderr); "
        "sys.stderr.flush(); time.sleep(0.2)"
    )
    try:
        url = _read_url_from_stderr(proc, _parse_quick_tunnel_url, timeout=30.0)
    finally:
        proc.wait(timeout=10)
    assert url == "https://fast-fail.trycloudflare.com"
