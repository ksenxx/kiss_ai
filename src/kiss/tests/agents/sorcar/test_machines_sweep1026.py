# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``is_local_machine`` asks reverse DNS last.

``socket.getfqdn()`` is an untimed C-level lookup (a slow resolver stalls
it for its full timeout, and a Stop injected on the task thread is not
delivered while it runs).  It used to run before the cheap exact matches,
so even ``run_agent(..., machine="localhost")`` paid for it.  Now a name
that is not ``LOCAL_MACHINE`` / ``localhost`` / the host's own short name,
and is not an IP address, is the only input that reaches the lookup.

The stall itself cannot be reproduced end-to-end: it needs a resolver that
hangs on the host's own reverse lookup, which means replacing
``socket.getfqdn`` (a test double) or re-pointing the system resolver
(``/etc/resolv.conf``, root).  Round 2 (review R3-sorcar-b P3) asked for a
patched lookup that fails when called for a local name or a loopback; the
project's test rules forbid test doubles, so this file pins the return
value of every branch and the ordering is documented here instead: only
the ``except ValueError`` branch of ``is_local_machine`` (a non-address
that is none of the exact names) calls ``socket.getfqdn``.
"""

from __future__ import annotations

import socket

from kiss.agents.sorcar import machines as mach


def test_every_spelling_of_this_host_and_two_strangers() -> None:
    """Exact names, the FQDN, loopback and own addresses are local; strangers are not."""
    hostname = socket.gethostname()
    for text in (mach.LOCAL_MACHINE, "LOCALHOST", hostname, hostname.upper(),
                 mach.local_machine_name(), socket.getfqdn(), socket.getfqdn().upper(),
                 "127.0.0.1", "[::1]"):
        assert mach.is_local_machine(text), text
    for text in ("", "nowhere.example.org", "203.0.113.9", "2001:db8::9"):
        assert not mach.is_local_machine(text), text
