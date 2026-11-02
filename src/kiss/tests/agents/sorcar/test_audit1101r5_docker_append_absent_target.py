# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A ``>>`` copy into an absent container file keeps what another writer creates meanwhile.

``DockerManager.Bash`` redirects ``cat >> path <<EOF`` to the ``Write``
it stands for and remembers the append (``PendingAppend``) so that
``DockerTools.Write`` appends the body through the shell's ``>>`` when
it lands.  The append used to be remembered only for a file that
already existed: for an absent target the redirected ``Write`` was a
plain store, so a file another writer (a sibling task sharing the
container) created between the redirect and the ``Write`` was replaced
and its bytes lost.  The append is now remembered for an absent target
too, with an empty snapshot; ``>>`` creates a missing file and keeps an
existing one.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.core.tool_verdict import ALLOW, Verdict
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available
from kiss.tests.agents.sorcar.test_docker_parity_2026_10_31 import _Harness

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running"),
]

TARGET = "/tmp/probe_absent_append.txt"
COMMAND = f"cat >> {TARGET} <<'EOF'\nmine\nEOF"


@pytest.fixture(scope="module")
def env() -> Iterator[DockerManager]:
    """A manager on a fresh container of :data:`IMAGE`."""
    manager = DockerManager(IMAGE)
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


def test_an_append_into_an_absent_file_keeps_what_another_writer_created_meanwhile(
    env: DockerManager,
) -> None:
    """The file appears between the redirect and the Write: the body goes after its line."""
    h = _Harness(env)
    env.Bash(f"rm -f {TARGET}", "clean")
    container = env.container
    assert container is not None

    def hook(name: str, args: dict[str, Any]) -> Verdict:
        h.hook_calls.append((name, dict(args)))
        if name == "Write":
            # Another writer (its own exec session, not this manager's
            # ``Bash``, which would drop the pending append as any call of
            # the agent's does) creates the file while the hook runs.
            exit_code, _ = container.exec_run(["sh", "-c", f"printf 'theirs\\n' > {TARGET}"])
            assert exit_code == 0
        return ALLOW

    h.agent.tool_call_hook = hook
    result = h.call("Bash", command=COMMAND, description="append")
    assert result == (
        f'Appended through Write(file_path="{TARGET}", content="mine\\n"), whose result '
        f"follows:\nSuccessfully wrote 5 characters to {TARGET}"
    ), result
    assert h.hook_calls == [
        ("Bash", {"command": COMMAND, "description": "append"}),
        ("Write", {"file_path": TARGET, "content": "mine\n"}),
    ]
    assert env.read_bytes(TARGET) == b"theirs\nmine\n"


def test_an_append_into_an_absent_file_nobody_else_touches_creates_it(env: DockerManager) -> None:
    """Without another writer the redirected append creates the file with the body alone."""
    h = _Harness(env)
    env.Bash(f"rm -f {TARGET}", "clean")
    result = h.call("Bash", command=COMMAND, description="append")
    assert result.endswith(f"Successfully wrote 5 characters to {TARGET}"), result
    assert env.read_bytes(TARGET) == b"mine\n"
    # A second append of the same body goes after the first, as ``>>`` does.
    result = h.call("Bash", command=COMMAND, description="append again")
    assert result.endswith(f"Successfully wrote 10 characters to {TARGET}"), result
    assert env.read_bytes(TARGET) == b"mine\nmine\n"
