# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (S5 F1): a redirected ``>>`` copy appends to the file as it is at Write time.

``DockerManager.Bash`` and ``ExecBridge.Bash`` redirect ``cat >> f <<'EOF'``
to ``Write(f, <current text> + body)``, the Write the agent then makes
through its hook and events.  The text was read when the redirect was
raised and the hook runs in between, so an append another writer made
there (a background job, a sibling task in the same container) was
overwritten by the stale snapshot.  The manager now remembers the
pending append and ``DockerTools.Write`` appends the body to the file
through the shell's ``>>`` (round 2, F2: in append mode, so what lands
between the hook and the store is kept as well); the pending append
belongs to that Write alone and is dropped by any other call (round 2,
F1: a refused redirect never turns a later Write into an append).

Every test runs against a real container (or a real exec bridge server
over a local environment) through the real ``SorcarAgent`` tool set and
a real ``KISSAgent`` dispatch; the concurrent writer is a sibling
task's own manager on the same container (a Bash of the agent's own
manager is a tool call, which drops a pending append) making a real
shell append from the agent's tool-call hook, where the redirect's hook
runs, or right before the Write stores (a trace of the test thread
chooses that interleaving).  Nothing is mocked.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any, cast

import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.agents.sorcar.exec_bridge import ExecBridge
from kiss.benchmarks.terminal_bench.exec_bridge_server import ExecBridgeServer
from kiss.core.tool_verdict import ALLOW, Verdict, refuse
from kiss.tests.agents.seas.test_coding_sea_evidence_gate import LocalEnvironment, _close, _Loop
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available
from kiss.tests.agents.sorcar.test_audit1101r2_sorcar_a_append_mode import AppendWhen
from kiss.tests.agents.sorcar.test_docker_parity_2026_10_31 import _Harness

pytestmark = pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")

#: The bridge's fetch and store scripts (``base64 -w0``, ``mv -T``, ``stat -c``) are those of
#: the GNU/busybox coreutils of the Linux sandboxes it serves; macOS's BSD tools reject them,
#: so a bridged local environment there refuses the append instead of redirecting it.
gnu_coreutils_only = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="the bridge's fetch and store scripts need GNU/busybox base64 -w0, mv -T and stat -c",
)

APPEND = "cat >> {path} <<'EOF'\nrequested\nEOF"


@pytest.fixture(scope="module")
def env() -> Iterator[DockerManager]:
    """One container of :data:`IMAGE` working in ``/tmp`` for the whole module."""
    manager = DockerManager(IMAGE, workdir="/tmp", mount_shared_volume=False)
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture(scope="module")
def sibling(env: DockerManager) -> Iterator[DockerManager]:
    """A second manager attached to the same container: the concurrent writer's own."""
    assert env.container is not None
    manager = DockerManager(f"container:{env.container.id}")
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


#: Runs a shell command in the environment as a sibling task would: ``Bash`` of its
#: own manager (command, description, timeout, output budget).
_Shell = Callable[[str, str, int, int], str]


class _AppendingHook:
    """The harness hook that makes a shell append to *path* when the redirected Write arrives.

    The hook is the agent's own tool-call hook: it runs between the
    redirect (the snapshot of the file) and the Write, which is where a
    concurrent writer lands.

    Args:
        h: The harness whose hook this replaces.
        path: The file the redirected Write targets.
        shell: Runs a shell command in the environment (:data:`_Shell`).
    """

    def __init__(self, h: _Harness, path: str, shell: _Shell) -> None:
        self.h = h
        self.path = path
        self.shell = shell
        self.appends = 0

    def __call__(self, name: str, args: dict[str, Any]) -> Verdict:
        """Append ``concurrent`` to the file on the Write of the redirect, then judge as usual."""
        if name == "Write" and args.get("file_path") == self.path:
            self.shell(f"printf 'concurrent\\n' >> {self.path}", "concurrent append", 30, 500)
            self.appends += 1
        return self.h.hook(name, args)


def _enters_store(frame: FrameType) -> bool:
    """Whether *frame* is ``DockerTools._store``, the Write about to store its bytes."""
    return frame.f_code is DockerTools._store.__code__


def _check_append_race(h: _Harness, env: Any, shell: _Shell, path: str) -> None:
    """The shared scenario: an append landing in the hook survives the redirected Write.

    Args:
        h: The harness dispatching the agent's calls to *env*.
        env: The agent's own manager.
        shell: The concurrent writer's way of running a shell command.
        path: The file appended to.
    """
    env.Bash(f"printf 'base\\n' > {path}", "seed", 30, 500)
    hook = _AppendingHook(h, path, shell)
    h.agent.tool_call_hook = hook
    out = h.call("Bash", command=APPEND.format(path=path), description="append")
    assert hook.appends == 1
    # The file holds every append, as it would after the shell's own ``>>``...
    assert env.read_bytes(path) == b"base\nconcurrent\nrequested\n"
    # ...and the hook saw the Write the command stood for when it was issued.
    assert out.startswith(
        f'Appended through Write(file_path="{path}", content="base\\nrequested\\n"), whose result '
        f"follows:\nSuccessfully wrote 15 characters to {path}"
    ), out
    assert h.hook_calls[-1] == ("Write", {"file_path": path, "content": "base\nrequested\n"})
    assert env.pending_append is None, "the pending append is forgotten once settled"
    # An append landing after the hook, right before the Write stores: kept as well.
    h.agent.tool_call_hook = h.hook
    late = functools.partial(shell, f"printf 'late\\n' >> {path}", "late append", 30, 500)
    with AppendWhen(_enters_store, late) as landed:
        out = h.call("Bash", command=APPEND.format(path=path), description="append")
    assert landed.made == 1, "the late append did not land before the Write stored"
    assert out.startswith("Appended through Write(") and "Successfully wrote 36 characters" in out
    assert env.read_bytes(path) == b"base\nconcurrent\nrequested\nlate\nrequested\n"
    assert env.pending_append is None
    # Nothing landed in between: the Write stores the content the hook saw.
    out = h.call("Bash", command=APPEND.format(path=path), description="append")
    assert out.startswith("Appended through Write(") and "Successfully wrote 51 characters" in out
    assert env.read_bytes(path) == b"base\nconcurrent\nrequested\nlate\nrequested\nrequested\n"
    assert env.pending_append is None


def test_a_concurrent_append_survives_the_redirected_write(
    env: DockerManager, sibling: DockerManager,
) -> None:
    """Docker: the file's current text, not the redirect-time snapshot, precedes the body."""
    _check_append_race(_Harness(env), env, sibling.Bash, "/tmp/append.txt")


def _refused_append(h: _Harness, env: DockerManager, path: str) -> None:
    """Issue the append into *path* with the hook refusing its Write; the file is unchanged."""
    h.verdict = refuse("not now")
    out = h.call("Bash", command=APPEND.format(path=path), description="append")
    assert out.endswith("REFUSED: the Write call did NOT run and had no effect. not now"), out
    assert env.read_bytes(path) == b"base\n"
    assert env.pending_append is not None, "the refused redirect's append is pending"
    h.verdict = ALLOW


def test_a_refused_redirect_leaves_no_append_for_a_later_write(env: DockerManager) -> None:
    """The pending append belongs to the redirected Write alone: the next call drops it."""
    h = _Harness(env)
    path = "/tmp/append2.txt"
    env.Bash(f"printf 'base\\n' > {path}", "seed", 30, 500)
    _refused_append(h, env, path)
    # The model's own Write of other content to the same file is made as issued...
    assert h.call("Write", file_path=path, content="fresh\n") == (
        f"Successfully wrote 6 characters to {path}"
    )
    assert env.read_bytes(path) == b"fresh\n"
    assert env.pending_append is None
    # ...and so is a later Write of the very content the refused redirect named: the
    # overwrite it says, not an append of the stale body after what the file holds now.
    assert h.call("Write", file_path=path, content="base\nrequested\n") == (
        f"Successfully wrote 15 characters to {path}"
    )
    assert env.read_bytes(path) == b"base\nrequested\n"
    # Every other call after the refusal drops it too: a Bash command, a Read, an Edit, a
    # Write of another file, a run_commands_parallel (A08 claude-7: it kept it).
    env.Bash(f"printf 'base\\n' > {path}", "seed", 30, 500)
    for name, args in (
        ("Bash", {"command": "true", "description": "nothing"}),
        ("run_commands_parallel", {"commands": '["true"]'}),
        ("Read", {"file_path": path}),
        ("Edit", {"file_path": path, "old_string": "base", "new_string": "base"}),
        ("Write", {"file_path": "/tmp/other.txt", "content": "base\nrequested\n"}),
    ):
        _refused_append(h, env, path)
        if name == "run_commands_parallel":
            # The agent keeps the prompt cache warm around a fan-out with
            # the model the harness's agent never had: the tool is called
            # as dispatched (the hook has no say over a fan-out).
            h.tools[name](**args)
        else:
            h.call(name, **args)
        assert env.pending_append is None, name
        assert h.call("Write", file_path=path, content="base\nrequested\n").startswith(
            "Successfully wrote 15 characters"
        )
        assert env.read_bytes(path) == b"base\nrequested\n", name
        env.Bash(f"printf 'base\\n' > {path}", "seed", 30, 500)
    assert env.read_bytes("/tmp/other.txt") == b"base\nrequested\n"
    # The same command again is redirected anew and goes through.
    _refused_append(h, env, path)
    out = h.call("Bash", command=APPEND.format(path=path), description="append")
    assert out.startswith("Appended through Write(") and env.read_bytes(path) == (
        b"base\nrequested\n"
    )
    assert env.pending_append is None


def test_an_append_whose_file_vanished_recreates_it_with_the_body(
    env: DockerManager, sibling: DockerManager,
) -> None:
    """The file removed while the hook ran: the body lands in a new file, as after ``>>``."""
    h = _Harness(env)
    path = "/tmp/append3.txt"
    env.Bash(f"printf 'base\\n' > {path}", "seed", 30, 500)

    def remove_in_hook(name: str, args: dict[str, Any]) -> Verdict:
        if name == "Write" and args.get("file_path") == path:
            sibling.Bash(f"rm -f {path}", "remove", 30, 500)
        return h.hook(name, args)

    h.agent.tool_call_hook = remove_in_hook
    out = h.call("Bash", command=APPEND.format(path=path), description="append")
    assert out.startswith(
        f'Appended through Write(file_path="{path}", content="base\\nrequested\\n"), whose result '
        f"follows:\nSuccessfully wrote 15 characters to {path}"
    ), out
    assert env.read_bytes(path) == b"requested\n"
    assert env.pending_append is None


@gnu_coreutils_only
def test_the_bridge_settles_its_pending_append_the_same_way(tmp_path: Path) -> None:
    """``ExecBridge`` (a remote environment through the real bridge server) keeps every append."""
    loop = _Loop()
    server = ExecBridgeServer(LocalEnvironment(str(tmp_path)), loop.loop)
    try:
        with ExecBridge(f"exec:{server.url}") as bridge, ExecBridge(f"exec:{server.url}") as other:
            h = _Harness(cast(Any, bridge))
            path = str(tmp_path / "append.txt")
            _check_append_race(h, bridge, other.Bash, path)
            # Its run_commands_parallel drops a refused redirect's append too (A08 claude-7).
            bridge.Bash(f"printf 'base\\n' > {path}", "seed", 30, 500)
            _refused_append(h, cast(Any, bridge), path)
            bridge.run_commands_parallel('["true"]')
            assert bridge.pending_append is None
    finally:
        _close(server, loop)
