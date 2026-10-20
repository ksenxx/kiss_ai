# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: hanging a terminal up never signals the daemon's own process group.

``TerminalService._hang_up`` sends ``SIGHUP`` to the shell's process
group.  ``pty.fork()`` returns to the parent before the child has run
``setsid()`` (glibc's ``forkpty`` does it in the child, on the child's
own schedule), so for a moment ``os.getpgid(child)`` is the DAEMON's
group: a ``terminalClose`` or a ``shutdown`` landing in that window
sent ``SIGHUP`` to the daemon itself and to every task subprocess in
its group, which the daemon treats as a shutdown signal (audit 2026-10-20
C5-F1; the probe ``tmp/pgid_probe.py`` saw the stale group in 32 of 300
forks on this machine).

The scenario runs in a child Python process that first puts itself in
a session and process group of its own (``os.setsid()``): a group-wide
``SIGHUP`` then lands on that process (where a handler counts it) and
never on the pytest runner.  Two windows are exercised against the real
``close`` path (session registered, then hung up):

* a deterministic one: a plain ``os.fork()`` child that waits on a
  pipe before its ``setsid()`` and is released only once the hang-up
  returned, so the group is the parent's for the whole hang-up;
* the real one: ``pty.fork()`` children hung up right after the fork,
  300 times, as the probe did.

Before the fix the counter was non-zero (every iteration of the first
window, about one in ten of the second); after it the child's own pid
is signalled while its group is still ours, and its group once it is
its own.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from kiss.tests.conftest import posix_only

pytestmark = posix_only("pty.fork and process groups")

_CHILD = textwrap.dedent(
    """
    import os, pty, signal
    from kiss.server.terminal_tab import TerminalService, _Session

    hits = 0

    def on_hup(signum, frame):
        global hits
        hits += 1

    signal.signal(signal.SIGHUP, on_hup)
    os.setsid()  # our own group: a group-wide SIGHUP is counted here only
    service = TerminalService(printer=None)

    def hang_up(tab, pid, fd, release=-1):
        session = _Session(tab_id=tab, conn_id="c", pid=pid, fd=fd, shell="sh", cwd="/")
        with service._lock:
            service._sessions[tab] = session
        service.close(tab)
        if release >= 0:
            # The hang-up is done: let the child leave our group and exit.
            os.write(release, b"go")
            os.close(release)
        with service._lock:
            del service._sessions[tab]
        os.close(fd)
        os.waitpid(pid, 0)
        session.reaped = True  # the hang-up's kill timer has nothing left to do

    # Window 1: the child is still in our group for the whole hang-up,
    # deterministically: it calls setsid() only once released through
    # the pipe, after ``service.close`` returned.  (The SIGHUP it is
    # sent meanwhile runs the inherited handler; its blocked read
    # resumes.)
    master, slave = pty.openpty()
    os.close(slave)
    gate_r, gate_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(gate_w)
        os.read(gate_r, 1)
        os.setsid()
        os._exit(0)
    os.close(gate_r)
    hang_up("slow", pid, master, release=gate_w)
    slow_hits = hits

    # Window 2: real pty.fork children, hung up right after the fork.
    for i in range(300):
        pid, fd = pty.fork()
        if pid == 0:
            os._exit(0)
        hang_up(f"pty-{i}", pid, fd)
    print(slow_hits, hits - slow_hits)
    """
)


def test_hang_up_signals_only_the_shell_until_its_group_is_its_own() -> None:
    """Neither window delivers a ``SIGHUP`` to the service's own process."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    slow_hits, pty_hits = (int(n) for n in proc.stdout.split())
    assert slow_hits == 0, f"SIGHUP reached the daemon's group {slow_hits} times (slow setsid)"
    assert pty_hits == 0, f"SIGHUP reached the daemon's group {pty_hits} times (pty.fork)"
