# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``DockerTools`` stores a payload in one piece, whatever happens around it.

A payload above the argv limit travels in several ``docker exec``
commands.  Landing them straight on the destination let a second
writer truncate the file between two pieces: both writes reported
success and the file held ten bytes of one followed by the tail of
the other.  A piece that failed (no space left, the destination is a
directory) left the partial file behind; for an ``Edit`` that was its
staging payload under ``/tmp``.

The payload is now staged beside the destination under a unique name
and published with one ``mv`` once every piece landed, so a reader
sees either the previous file or the whole new one, and the staging
file is removed on every exit path that did not publish it.  The
rename changes nothing but the content: the file keeps its owner, a
file its user may not write is refused, and a name of the full length
is staged like any other.  All tests run against real containers; the
race is pinned with the settrace ``Pause`` on ``DockerManager.Bash``.
"""

import functools
import hashlib
import threading
from collections.abc import Iterator
from typing import Any

import docker
import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.core.kiss_error import KISSError
from kiss.core.tool_verdict import ToolRedirect
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available
from kiss.tests.server.pause import Pause

IMAGE = "python:3.11-slim"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running"),
]


@pytest.fixture
def env() -> Iterator[DockerManager]:
    """A manager on a fresh container of :data:`IMAGE`."""
    manager = DockerManager(IMAGE)
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


def _attached(**options: Any) -> Iterator[DockerManager]:
    """A manager attached to a container of :data:`IMAGE` started with *options*."""
    client = docker.from_env()
    container = client.containers.run(IMAGE, command="sleep infinity", detach=True, **options)
    try:
        manager = DockerManager(f"container:{container.id}")
        manager.open()
        try:
            yield manager
        finally:
            manager.close()
    finally:
        container.remove(force=True)
        client.close()


@pytest.fixture
def small_tmp_env() -> Iterator[DockerManager]:
    """A manager attached to a container whose ``/tmp`` is a 128 KiB tmpfs."""
    yield from _attached(tmpfs={"/tmp": "size=131072"})


@pytest.fixture
def user_env() -> Iterator[DockerManager]:
    """A manager attached to a container whose user is not root."""
    yield from _attached(user="1000:1000")


def _without_python(env: DockerManager, command: str, description: str) -> str:
    """``env.Bash`` with a ``command -v`` that finds nothing: an Edit takes its Perl editor."""
    return env.Bash(f"command() {{ return 1; }}\n{command}", description)


def _both_editors(env: DockerManager) -> list[DockerTools]:
    """Tools whose Edit runs the Python script, and tools whose Edit runs the Perl one."""
    return [DockerTools(env.Bash), DockerTools(functools.partial(_without_python, env))]


def _sha256_in(env: DockerManager, path: str) -> str:
    """The SHA-256 the container computes for *path*."""
    return env.Bash(f"sha256sum {path}", "hash").split()[0]


def _listing(env: DockerManager, directory: str) -> list[str]:
    """The names in *directory*, dotfiles included, without the exit-code line."""
    output = env.Bash(f"ls -A {directory}", "ls")
    return [line for line in output.splitlines() if not line.startswith("[exit code:")]


def _is_append_piece(frame_locals: dict[str, Any]) -> bool:
    """Whether a ``DockerManager.Bash`` call carries a second or later payload piece."""
    return "base64 -d >>" in frame_locals["command"]


def _is_second_part(frame_locals: dict[str, Any]) -> bool:
    """Whether a ``DockerManager.Bash`` call runs the second of a store's commands."""
    return "(part 2/" in frame_locals["description"]


def _write_through(results: list[str], tools: DockerTools, path: str, content: str) -> None:
    """Thread body: ``Write`` *content* to *path* and keep the result."""
    results.append(tools.Write(path, content))


def test_concurrent_writes_never_mix_and_readers_never_see_a_partial_file(
    env: DockerManager,
) -> None:
    """Two writers of one path leave the file one of the two contents, never a mixture.

    The second writer's publication waits for the one in flight (the
    tools' ``_FILE_WRITE_LOCK``, as on the host: A08 astra-2 / fable-1),
    so it lands last.
    """
    tools = DockerTools(env.Bash)
    path = "/tmp/race/target.txt"
    assert "Successfully wrote" in tools.Write(path, "previous\n")
    big = "A" * 80_000
    results: list[str] = []
    writer = threading.Thread(target=_write_through, args=(results, tools, path, big))
    second = threading.Thread(target=_write_through, args=(results, tools, path, "B" * 10))
    with Pause(DockerManager.Bash.__code__, _is_append_piece) as pause:
        writer.start()
        assert pause.paused.wait(60), "the big write never reached its second piece"
        # The first piece has landed somewhere, but a reader still sees
        # the previous file whole.
        assert env.Bash(f"cat {path}", "cat").split("\n")[0] == "previous"
        # A second writer waits for the publication in flight.
        second.start()
        second.join(2)
        assert second.is_alive(), "the second write did not wait for the first's publication"
        assert env.Bash(f"cat {path}", "cat").split("\n")[0] == "previous"
        pause.release()
    writer.join(60)
    second.join(60)
    assert not writer.is_alive() and not second.is_alive()
    assert all("Successfully wrote" in result for result in results), results
    assert _sha256_in(env, path) == hashlib.sha256(b"B" * 10).hexdigest()
    assert _listing(env, "/tmp/race") == ["target.txt"]


def test_a_big_append_lands_whole_in_the_one_file_it_opens(env: DockerManager) -> None:
    """Audit 2026-11-10 (B/S4-F2): an append above one command's payload opened its target
    once per piece, so a link repointed or a file replaced in between got the tail, the
    earlier file the head, and the Write reported success.  The pieces are staged and one
    command appends them through one open of the target, as ``cat >> target`` would."""
    tools = DockerTools(env.Bash, env.read_bytes)
    body = ("A" * 72_017 + "\n").encode()
    env.Bash(
        "mkdir /tmp/app && cd /tmp/app && : > first && : > second && ln -s first link"
        " && echo old > file && ln file kept && echo new > fresh && : > gone",
        "seed",
    )
    for name, swap, after in (
        ("link", "ln -sfn second link", {"first": b"", "second": body}),
        ("file", "mv fresh file", {"kept": b"old\n", "file": b"new\n" + body}),
        ("gone", "rm gone && mkdir gone", {}),  # the append fails: nothing staged is left
    ):
        with pytest.raises(ToolRedirect) as redirect:  # the Write the agent then makes
            env.Bash(f"cat >> /tmp/app/{name} <<'EOF'\n{body.decode()}EOF", "append")
        results: list[str] = []
        write = redirect.value.arguments
        writer = threading.Thread(
            target=_write_through, args=(results, tools, write["file_path"], write["content"])
        )
        with Pause(DockerManager.Bash.__code__, _is_second_part) as pause:
            writer.start()
            assert pause.paused.wait(60), "the append never reached its second command"
            env.Bash(f"cd /tmp/app && {swap}", "swap")
        writer.join(60)
        assert not writer.is_alive() and ("Successfully wrote" in results[0]) == bool(after)
        for stored, data in after.items():
            assert env.read_bytes(f"/tmp/app/{stored}") == data, stored
    assert _listing(env, "/tmp") == ["app"]


def test_failed_edit_leaves_no_staging_file(small_tmp_env: DockerManager) -> None:
    """An Edit whose payload does not fit the disk reports it and leaves nothing behind."""
    tools = DockerTools(small_tmp_env.Bash)
    assert "Successfully wrote" in tools.Write("/tmp/target.txt", "old\n")
    result = tools.Edit("/tmp/target.txt", "old", "X" * 200_000)
    assert "No space left on device" in result, result
    assert small_tmp_env.Bash("cat /tmp/target.txt", "cat").split("\n")[0] == "old"
    assert _listing(small_tmp_env, "/tmp") == ["target.txt"]
    # A file with a second name is written in place, and the disk takes the
    # payload but not the new text: the write fails when the file is closed.
    # The Python editor never closed it and reported the cut file as edited.
    seed = (
        "rm -f /tmp/fill /tmp/alias.txt; printf 'old\\n' > /tmp/target.txt && "
        "ln /tmp/target.txt /tmp/alias.txt; dd if=/dev/zero of=/tmp/fill bs=4096 count=64 "
        "2>/dev/null; truncate -s -8192 /tmp/fill"
    )
    for editor in _both_editors(small_tmp_env):
        small_tmp_env.Bash(seed, "seed")
        result = editor.Edit("/tmp/target.txt", "old", "X" * 6000)
        assert result.startswith("Error") and "No space left on device" in result, result
        assert "Successfully replaced" not in result, result
        assert _listing(small_tmp_env, "/tmp") == ["alias.txt", "fill", "target.txt"]


def test_write_onto_a_directory_fails_and_leaves_nothing(env: DockerManager) -> None:
    """A multi-piece Write whose destination is a directory fails without a leftover.

    So does one onto a FIFO or a device, and an Edit of the FIFO answers
    at once (S11 re-fix round 2): the rename turned the FIFO and
    ``/dev/null`` into regular files, and the editors blocked on the
    FIFO until the call timed out.
    """
    tools = DockerTools(env.Bash)
    env.Bash("mkdir -p /tmp/onto/adir && mkfifo /tmp/onto/fifo", "mkdir")
    for path, kind in (
        ("/tmp/onto/adir", "directory"),
        ("/tmp/onto/fifo", "fifo"),
        ("/dev/null", "character special file"),
    ):
        result = tools.Write(path, "C" * 200_000)
        assert "Successfully wrote" not in result
        assert "directory" in result, result
        assert env.Bash(f"stat -c %F {path}", "stat").split("\n")[0] == kind
    for editor in _both_editors(env):
        edited = editor.Edit("/tmp/onto/fifo", "a", "b")
        assert edited.startswith("Error: File not found: /tmp/onto/fifo"), edited
    assert _listing(env, "/tmp/onto") == ["adir", "fifo"]


def test_a_file_mounted_into_the_container_is_written_in_place(env: DockerManager) -> None:
    """S11 re-fix round 2: a rename over a mount point is refused (``EBUSY``).

    Docker mounts ``/etc/hosts`` into every container, as it does a
    single file given with ``-v``: Write and both editors answered
    "Device or resource busy" there, where ``cat > file`` writes
    through.
    """
    writer, (python, perl) = DockerTools(env.Bash), _both_editors(env)
    for content in ("one\n", "one\n" + "#" * 100_000):
        assert "Successfully wrote" in writer.Write("/etc/hosts", content)
    assert "Successfully replaced 1" in python.Edit("/etc/hosts", "one", "two")
    assert "Successfully replaced 1" in perl.Edit("/etc/hosts", "two", "three")
    assert env.read_bytes("/etc/hosts") == b"three\n" + b"#" * 100_000
    assert [n for n in _listing(env, "/etc") if "kiss" in n or n.endswith(".tmp")] == []


def test_write_keeps_the_mode_of_the_replaced_file(env: DockerManager) -> None:
    """Overwriting an executable keeps it executable."""
    tools = DockerTools(env.Bash)
    tools.Write("/tmp/mode/run.sh", "#!/bin/sh\necho one\n")
    env.Bash("chmod 755 /tmp/mode/run.sh", "chmod")
    assert "Successfully wrote" in tools.Write("/tmp/mode/run.sh", "#!/bin/sh\necho two\n")
    assert env.Bash("stat -c %a /tmp/mode/run.sh", "stat").split("\n")[0] == "755"
    assert env.Bash("/tmp/mode/run.sh", "run").split("\n")[0] == "two"


def test_a_replaced_file_keeps_its_owner_and_a_full_length_name_is_staged(
    env: DockerManager,
) -> None:
    """S11 re-fix round 2: what the rename over a file changed besides its text.

    The tools run as root here: another user's file came out root's
    (``0:0``), the staged file never being given its owner.  A 255-byte
    name left no room for a staging name made from it ("File name too
    long").  The line a one-line ``old`` adds to a CRLF file was written
    with a bare LF.
    """
    path = "/tmp/own/" + "n" * 255
    seed = f"printf 'old\\r\\ntail\\r\\n' > {path} && chown 1000:1000 {path} && chmod 640 {path}"
    env.Bash("mkdir /tmp/own", "mkdir")
    for tools in _both_editors(env):
        env.Bash(seed, "seed")
        assert "Successfully replaced 1" in tools.Edit(path, "old", "new\nmore")
        assert env.read_bytes(path) == b"new\r\nmore\r\ntail\r\n"
        assert env.Bash(f"stat -c '%u:%g %a' {path}", "stat").split("\n")[0] == "1000:1000 640"
    for content in ("small\n", "x" * 100_000):
        assert "Successfully wrote" in DockerTools(env.Bash).Write(path, content)
        assert env.Bash(f"stat -c '%u:%g %a' {path}", "stat").split("\n")[0] == "1000:1000 640"
    assert _listing(env, "/tmp/own") == ["n" * 255]


def test_a_read_only_file_is_refused_to_a_plain_user(user_env: DockerManager) -> None:
    """S11 re-fix rounds 2 and 3: the rename took the directory's permission only.

    In a container whose user is not root, Write and Edit replaced a
    0444 file and reported success; the host tools and the shell's ``>``
    refuse it.  The ``exact`` Edit still replaced it (round 3): made
    directly, or as the Edit a python rewrite is applied as, which
    itself is refused.  A ``sed -i`` does replace such a file, so it
    runs as written, as on the host, instead of as an Edit.
    """
    path = "/tmp/ro/locked.txt"
    user_env.Bash(f"mkdir /tmp/ro && printf 'old\\n' > {path} && chmod 444 {path}", "seed")
    python = (
        f"python3 - <<'EOF'\np = '{path}'\ns = open(p).read()\n"
        "s = s.replace('old', 'new')\nopen(p, 'w').write(s)\nEOF"
    )
    with pytest.raises(ToolRedirect) as redirect:  # the Edit the agent then makes
        user_env.Bash(python, "python rewrite")
    rewrite = redirect.value.arguments
    assert (redirect.value.name, rewrite["exact"]) == ("Edit", True)
    for tools in _both_editors(user_env):
        for refused in (
            tools.Edit(path, "old", "new"),
            tools.Edit(**rewrite),
            tools.Write(path, "new\n"),
            tools.Write(path, "x" * 100_000),
        ):
            assert refused.startswith("Error") and "Permission denied" in refused, refused
        assert user_env.read_bytes(path) == b"old\n"
    sed = user_env.Bash(f"sed -i 's/old/new/' {path} && cat {path} && stat -c %a {path}", "sed")
    assert sed == "new\n444\n", sed
    assert _listing(user_env, "/tmp/ro") == ["locked.txt"]


def test_a_writable_file_is_written_in_a_directory_that_takes_no_sibling(
    user_env: DockerManager,
) -> None:
    """A plain user's file in a directory that user may not write is written in place.

    The shell's ``>`` writes such a file and both editors do; Write
    answered "Permission denied" for the staging file it could not make
    beside it.  A file that directory does not hold yet is still refused,
    and nothing staged elsewhere is left behind.
    """
    path = "/tmp/closed/file.txt"
    user_env.Bash(f"mkdir /tmp/closed && printf 'old\\n' > {path} && chmod 555 /tmp/closed", "seed")
    tools = DockerTools(user_env.Bash)
    for content in ("small\n", "x" * 100_000):
        assert "Successfully wrote" in tools.Write(path, content)
        assert user_env.read_bytes(path) == content.encode()
    for content in ("new\n", "y" * 100_000):
        refused = tools.Write("/tmp/closed/other.txt", content)
        assert refused.startswith("Error") and "Permission denied" in refused, refused
    assert _listing(user_env, "/tmp/closed") == ["file.txt"]
    assert _listing(user_env, "/tmp") == ["closed"]


@pytest.fixture
def no_inode_env() -> Iterator[DockerManager]:
    """A manager attached to a container whose ``/full`` holds one file and takes no other."""
    yield from _attached(tmpfs={"/full": "size=65536,nr_inodes=2"})


def test_a_write_that_cannot_stage_beside_a_writable_directorys_file_leaves_it_whole(
    no_inode_env: DockerManager,
) -> None:
    """A directory its user may write that takes no new file all the same (no inode left).

    Only a directory that refuses the user gets its file written in
    place.  Here the Write fails, as it did before that case existed:
    written in place, a payload the disk does not hold cut the file.
    """
    path = "/full/file.txt"
    no_inode_env.Bash(f"printf 'old\\n' > {path}", "seed")
    tools = DockerTools(no_inode_env.Bash)
    for content in ("new\n", "x" * 200_000):
        result = tools.Write(path, content)
        assert result.startswith("Error") and "No space left on device" in result, result
        assert no_inode_env.read_bytes(path) == b"old\n"
    assert _listing(no_inode_env, "/full") == ["file.txt"]
    assert _listing(no_inode_env, "/tmp") == []


def test_write_through_a_symlink_updates_its_target_and_keeps_the_link(env: DockerManager) -> None:
    """A Write to a symlink lands in the file it points to, as the ``cat > link`` it stands for.

    The link stays a link (its mode is never the target's) and a dangling
    link makes its missing target, as the shell redirection would.
    """
    tools = DockerTools(env.Bash)
    tools.Write("/tmp/links/target.txt", "old\n")
    env.Bash(
        "cd /tmp/links && chmod 640 target.txt && ln -s target.txt link.txt"
        " && ln -s /tmp/links/missing.txt dangling.txt",
        "ln",
    )
    assert "Successfully wrote" in tools.Write("/tmp/links/link.txt", "new\n")
    checked = env.Bash(
        "cd /tmp/links && readlink link.txt && stat -c %a target.txt && cat target.txt && ls",
        "check",
    )
    assert checked.split("\n")[:6] == [
        "target.txt", "640", "new", "dangling.txt", "link.txt", "target.txt",
    ], checked
    assert "Successfully wrote" in tools.Write("/tmp/links/dangling.txt", "made\n")
    checked = env.Bash("cd /tmp/links && readlink dangling.txt && cat missing.txt", "check")
    assert checked.split("\n")[:2] == ["/tmp/links/missing.txt", "made"], checked


def test_manager_closed_between_pieces_raises_and_publishes_nothing(
    env: DockerManager,
) -> None:
    """A store cut short by the container going away raises instead of publishing."""
    tools = DockerTools(env.Bash)
    errors: list[BaseException] = []

    def write_big() -> None:
        try:
            tools.Write("/tmp/cut/target.txt", "D" * 80_000)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writer = threading.Thread(target=write_big)
    with Pause(DockerManager.Bash.__code__, _is_append_piece) as pause:
        writer.start()
        assert pause.paused.wait(60)
        env.close()
        pause.release()
    writer.join(60)
    assert not writer.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], KISSError), errors
