# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A name lookup holds no call of ``machines`` past its deadline.

``is_local_machine`` asks reverse DNS last: ``socket.getfqdn()`` used
to run before the cheap exact matches, so even
``run_agent(..., machine="localhost")`` paid for it.  Now a name that
is not ``LOCAL_MACHINE`` / ``localhost`` / the host's own short name,
and is not an IP address, is the only input that reaches the lookup.

The resolver takes no timeout and cannot be interrupted, and a lookup
comes before any socket a timeout could run on (audit 2026-11-12, B4):
while a resolver did not answer, ``is_local_machine`` held every
``run_agent(machine=...)`` before its job existed, ``pin_certificate``
held ``/machine add``, and a link's dial held its request (and the
link's lock) for the resolver's retry schedule instead of the
request's timeout.  The stall is real here: in a throwaway namespace
the one nameserver is a socket that never answers, and this host's own
name is in no hosts file.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.tests.conftest import posix_only


def test_every_spelling_of_this_host_and_two_strangers() -> None:
    """Exact names, the FQDN, loopback and own addresses are local; strangers are not."""
    hostname = socket.gethostname()
    for text in (mach.LOCAL_MACHINE, "LOCALHOST", hostname, hostname.upper(),
                 mach.local_machine_name(), socket.getfqdn(), socket.getfqdn().upper(),
                 "127.0.0.1", "[::1]"):
        assert mach.is_local_machine(text), text
    for text in ("", "nowhere.example.org", "203.0.113.9", "2001:db8::9"):
        assert not mach.is_local_machine(text), text


_STALLED_RESOLVER = """\
import fcntl, os, socket, struct, subprocess, sys, threading, time
from functools import partial
from pathlib import Path
from kiss.agents.sorcar import machines as mach
from kiss.tests.local_ws import make_test_tls
scratch = Path(sys.argv[1])
# This namespace's resolver asks a nameserver that never answers, for 10 s a lookup.
conf = Path(os.path.realpath("/etc/resolv.conf"))
ours = scratch / "resolv.conf"
ours.write_text("nameserver 127.0.0.1\\noptions timeout:5 attempts:2\\n")
switch = scratch / "nsswitch.conf"
switch.write_text("hosts: files dns\\n")  # no module that answers for this host's own name
subprocess.run(["mount", "-t", "tmpfs", "tmpfs", "/run"], check=True)  # no resolver daemon to ask
if conf.is_relative_to("/run"):
    conf.parent.mkdir(parents=True)
    conf.write_text(ours.read_text())
else:
    subprocess.run(["mount", "--bind", str(ours), str(conf)], check=True)
subprocess.run(["mount", "--bind", str(switch), "/etc/nsswitch.conf"], check=True)
socket.sethostname("stalled-host")
fcntl.ioctl(socket.socket(), 0x8914, struct.pack("16sH", b"lo", 1))  # SIOCSIFFLAGS: lo up
silent = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
silent.bind(("127.0.0.1", 53))
for lookup in (socket.getfqdn, partial(socket.getaddrinfo, "stalled.invalid", 80)):
    probe = threading.Thread(target=lookup, daemon=True)
    probe.start()
    probe.join(2)
    if not probe.is_alive():
        sys.exit("the resolver does not stall here")
mach.DNS_TIMEOUT = mach.PIN_TIMEOUT = 1.0
mach.DIAL_GRACE = 0.5
link = mach.MachineLink(
    mach.Machine("box", "https://stalled.invalid:9787"), cert_file=make_test_tls(scratch)[2],
)
for call in (
    partial(link.request, {"type": "ping"}, "pong", 1.0),
    partial(mach.pin_certificate, "https://stalled.invalid:9787"),
    partial(mach.is_local_machine, "stranger.invalid"),  # is it this host's full name?
    partial(mach.is_local_machine, "203.0.113.9"),  # does this host's name resolve to it?
):
    started = time.monotonic()
    try:
        answer = call()
    except mach.MachineError as error:
        answer = str(error)
    print(time.monotonic() - started < 5, answer)
"""


@posix_only("a resolver stalled in a network namespace of its own")
def test_a_stalled_resolver_holds_no_call_past_its_deadline(tmp_path: Path) -> None:
    """The dial, the certificate fetch and the local-name check end on time without an answer.

    Before: 10 s each, the resolver's retry schedule here (minutes with
    another), whatever the call's timeout.
    """
    in_namespace = ["unshare", "-Urnmu"]
    if (
        shutil.which("unshare") is None
        or subprocess.run([*in_namespace, "true"], capture_output=True).returncode
    ):
        pytest.skip("no unprivileged namespaces here")
    child = subprocess.run(
        [*in_namespace, sys.executable, "-c", _STALLED_RESOLVER, str(tmp_path)],
        env={**os.environ, "KISS_HOME": str(tmp_path)},
        capture_output=True, text=True, timeout=120,
    )
    if "does not stall" in child.stderr:
        pytest.skip(child.stderr.strip())
    where = "https://stalled.invalid:9787"
    assert child.stdout.splitlines() == [
        f"True box ({where}) is unreachable: timed out",
        f"True Cannot reach {where}: timed out",
        "True False",
        "True False",
    ], child.stdout + child.stderr
