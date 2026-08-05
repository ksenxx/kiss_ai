# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared pytest fixtures for VS Code / kiss-web web_server tests.

These tests instantiate the real :class:`RemoteAccessServer` and call
the module-level helpers that read/write ``remote-url.json``.  The
default file lives at ``~/.kiss/remote-url.json`` (or
``$KISS_HOME/remote-url.json``), which is watched every 10 seconds by:

* the running ``kiss-web`` daemon (it re-reads the file to publish the
  active URL), and
* the VS Code extension (it polls the file and, on certain transitions,
  invokes ``restartKissWebDaemon`` which kills the daemon).

When the test process and a live ``kiss-web`` daemon / VS Code extension
share the same path, writes and unlinks from tests can sever the live
agent's transport.  The root ``tests/conftest.py`` already isolates the
process by setting ``KISS_HOME`` to a per-process ``tempfile.mkdtemp``
so the shared path is no longer the live one.

For per-test isolation beyond that, individual tests can pass an
explicit ``url_file=tmp_path / "remote-url.json"`` to
:class:`RemoteAccessServer` (added by the constructor refactor) and use
that same path for direct file inspection.

Orphan-sweep join guard
-----------------------

Many tests in this folder construct a ``VSCodeServer`` — which starts
the ``orphan-task-sweep`` daemon thread — and then close
``persistence._db_conn`` in their teardown, which used to crash the
interpreter with SIGSEGV whenever the sweep was still running SQL on
that connection.  The guard against that race lives in the ROOT
``tests/conftest.py`` (``join_orphan_sweeps``), so it protects every
test directory rather than this one only; see its module docstring for
the mechanism.
"""

from __future__ import annotations
