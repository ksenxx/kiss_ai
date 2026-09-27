# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``scripts/sync-memory.sh`` and ``merge_memory_pages.py``.

``./rsorcar`` (step 4c) syncs the agent's persistent memory -- the flat
directory of Markdown pages under ``~/.kiss/memories`` -- with the machine it
deploys to, in both directions, so the agent on either side remembers what the
agent on the other learned.

These tests run the real script against a sandbox "remote": a fake ``ssh`` on
``PATH`` executes the remote half locally with ``HOME`` pointed at another
directory, and a fake ``scp`` copies into that directory.  The pages are real
memory pages (written and read back through :class:`MemoryDir`) and the vector
index is the real one, so what is checked is what a deploy does.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
from kiss.core.memoryfield.pages import MemoryDir
from kiss.tests.conftest import posix_only

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "sync-memory.sh"
MERGE = ROOT / "src" / "kiss" / "scripts" / "merge_memory_pages.py"

pytestmark = posix_only("drives the bash sync-memory.sh with a bash ssh stand-in")

# ssh accepts its options before the destination; a caller may add ``-p``.
FAKE_SSH = """#!/bin/bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-p|-i|-l|-F|-c) shift 2 ;;
        -*) shift ;;
        *) break ;;
    esac
done
shift                       # drop the user@host argument
export HOME="$REMOTE_HOME"
exec bash -c "$*"
"""

# scp -q SRC user@host:.kiss/  ->  a copy under the remote's HOME.
FAKE_SCP = """#!/bin/bash
while [[ "${1:-}" == -* ]]; do shift; done
exec cp "$1" "$REMOTE_HOME/${2#*:}"
"""

# An ssh whose connection drops: exit status 255, like the real client.
DEAD_SSH = "#!/bin/bash\necho 'ssh: connect to host fakehost: Connection refused' >&2\nexit 255\n"


def page_text(body: str, updated: str, title: str = "t") -> str:
    """A page as MemoryDir writes it, with a chosen ``updated`` stamp."""
    return (
        f"---\ntitle: '{title}'\nuuid: 00000000-0000-0000-0000-000000000000\n"
        f"summary: s\ncreated: '2026-09-01T00:00:00Z'\nupdated: '{updated}'\n---\n{body}\n"
    )


class Sandbox:
    """Two fake machines and the stubs that connect them."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.local_home = tmp / "home"
        self.remote_home = tmp / "rhome"
        self.local_mem = self.local_home / ".kiss" / "memories"
        self.remote_mem = self.remote_home / ".kiss" / "memories"
        self.bindir = tmp / "bin"
        for directory in (self.local_home / ".kiss", self.remote_home / ".kiss", self.bindir):
            directory.mkdir(parents=True)
        self.install("ssh", FAKE_SSH)
        self.install("scp", FAKE_SCP)

    def install(self, name: str, text: str) -> None:
        """Put an executable stub named *name* on the sandbox PATH."""
        path = self.bindir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def run(self) -> subprocess.CompletedProcess[str]:
        """Run the real sync-memory.sh against the sandbox remote."""
        env = dict(os.environ)
        env.update(
            {
                "HOME": str(self.local_home),
                "REMOTE_HOME": str(self.remote_home),
                "PATH": f"{self.bindir}:{env['PATH']}",
                # The sandbox's memory, not the developer's own.
                "KISS_HOME": str(self.local_home / ".kiss"),
            }
        )
        return subprocess.run(
            ["bash", str(SCRIPT), "me@fakehost"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    """A fresh pair of machines per test."""
    return Sandbox(tmp_path)


def epoch(stamp: str) -> int:
    """Seconds since the epoch of an ``updated`` stamp."""
    return int(datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp())


def write(directory: Path, name: str, body: str, updated: str) -> None:
    """Create a page file directly, as another agent would have left it.

    The file's modification time is the stamp, as it is (to the second) for a
    page the agent writes.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(page_text(body, updated))
    os.utime(path, (epoch(updated), epoch(updated)))


def tombstone(directory: Path, name: str, deleted: str) -> Path:
    """Leave the tombstone ``MemoryDir.delete`` leaves, with a chosen deletion time."""
    (directory / ".tombstones").mkdir(parents=True, exist_ok=True)
    path = directory / ".tombstones" / name
    path.write_text(deleted + "\n")
    return path


def test_pages_from_both_machines_end_up_on_both(box: Sandbox) -> None:
    """A page that exists on one machine exists on both afterwards, unchanged."""
    MemoryDir(box.local_mem).write("laptop-lesson", "learned on the laptop", summary="a")
    MemoryDir(box.remote_mem).write("server-lesson", "learned on the server", summary="b")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert MemoryDir(box.local_mem).page_names() == ["laptop-lesson", "server-lesson"]
    assert MemoryDir(box.remote_mem).page_names() == ["laptop-lesson", "server-lesson"]
    assert MemoryDir(box.local_mem).read("server-lesson").body.strip() == "learned on the server"
    assert MemoryDir(box.remote_mem).read("laptop-lesson").body.strip() == "learned on the laptop"
    assert "This machine's memory: added 1 updated 0 kept 0 conflicts 0 removed 0" in result.stdout
    assert "me@fakehost's memory: added 1 updated 0 kept 1 conflicts 0 removed 0" in result.stdout
    assert "hold the same pages" in result.stdout


def test_the_newer_copy_of_a_page_wins_in_either_direction(box: Sandbox) -> None:
    """Of a page both have, the copy with the later ``updated`` replaces the other."""
    write(box.local_mem, "newer-here", "laptop v2", "2026-09-20T10:00:00Z")
    write(box.remote_mem, "newer-here", "server v1", "2026-09-19T10:00:00Z")
    write(box.local_mem, "newer-there", "laptop v1", "2026-09-19T10:00:00Z")
    write(box.remote_mem, "newer-there", "server v2", "2026-09-20T10:00:00Z")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    for mem in (box.local_mem, box.remote_mem):
        assert MemoryDir(mem).read("newer-here").body.strip() == "laptop v2"
        assert MemoryDir(mem).read("newer-there").body.strip() == "server v2"
    assert "This machine's memory: added 0 updated 1 kept 1 conflicts 0 removed 0" in result.stdout
    assert "me@fakehost's memory: added 0 updated 1 kept 1 conflicts 0 removed 0" in result.stdout


def test_a_page_changed_on_both_in_the_same_second_is_a_named_conflict(box: Sandbox) -> None:
    """Same stamp, different content: each machine keeps its own and the page is named."""
    write(box.local_mem, "clash", "laptop view", "2026-09-20T10:00:00Z")
    write(box.remote_mem, "clash", "server view", "2026-09-20T10:00:00Z")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert MemoryDir(box.local_mem).read("clash").body.strip() == "laptop view"
    assert MemoryDir(box.remote_mem).read("clash").body.strip() == "server view"
    assert "both machines changed (or one deleted) clash in the same second" in result.stdout
    assert "1 page(s) differ between the two machines" in result.stdout
    assert "hold the same pages" not in result.stdout


def test_domain_memories_travel_with_the_general_memory(box: Sandbox) -> None:
    """Pages of a nested domain memory (``<memory>/kiss/``) sync like general pages."""
    MemoryDir(box.local_mem / "kiss").write("laptop-repo-fact", "from the laptop", summary="a")
    MemoryDir(box.remote_mem / "kiss").write("server-repo-fact", "from the server", summary="b")
    MemoryDir(box.local_mem).write("general-fact", "general", summary="c")
    write(box.local_mem / "kiss", "clash", "laptop view", "2026-09-20T10:00:00Z")
    write(box.remote_mem / "kiss", "clash", "server view", "2026-09-20T10:00:00Z")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    for mem in (box.local_mem, box.remote_mem):
        assert MemoryDir(mem / "kiss").page_names() == [
            "clash", "laptop-repo-fact", "server-repo-fact",
        ]
        assert MemoryDir(mem).page_names() == ["general-fact"]
    assert MemoryDir(box.remote_mem / "kiss").read("clash").body.strip() == "server view"
    assert "This machine's memory: added 1 updated 0 kept 0 conflicts 1 removed 0" in result.stdout
    assert "me@fakehost's memory: added 2 updated 0 kept 1 conflicts 1 removed 0" in result.stdout
    assert "both machines changed (or one deleted) kiss/clash in the same second" in result.stdout


def test_a_second_sync_changes_nothing(box: Sandbox) -> None:
    """Once in sync, a sync copies nothing and rewrites no file."""
    MemoryDir(box.local_mem).write("a", "aa")
    MemoryDir(box.remote_mem).write("b", "bb")
    assert box.run().returncode == 0
    before = {p.name: p.stat().st_mtime_ns for p in box.remote_mem.iterdir()}
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "This machine's memory: added 0 updated 0 kept 2 conflicts 0 removed 0" in result.stdout
    assert "me@fakehost's memory: added 0 updated 0 kept 2 conflicts 0 removed 0" in result.stdout
    assert {p.name: p.stat().st_mtime_ns for p in box.remote_mem.iterdir()} == before


def test_the_index_does_not_travel_and_rebuilds_from_the_synced_pages(box: Sandbox) -> None:
    """The ``*.sqlite3`` cache stays where it is; the receiving index picks up the new page."""
    local = MemoryDir(box.local_mem)
    local.write("only-here", "a fact about carbon fibre woks")
    local_index = VectorIndex(local, embed=hashed_embedding)
    local_index.sync()
    remote = MemoryDir(box.remote_mem)
    remote.write("only-there", "a fact about the cron agent")
    remote_index = VectorIndex(remote, embed=hashed_embedding)
    remote_index.sync()
    remote_index_bytes = remote_index.path.read_bytes()
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    # No index file arrived beside the pages, and neither index was rewritten.
    assert sorted(p.name for p in box.remote_mem.iterdir() if p.suffix == ".sqlite3") == [
        remote_index.path.name
    ]
    assert remote_index.path.read_bytes() == remote_index_bytes
    assert "incoming" not in {p.name for p in box.local_mem.iterdir()}
    # The next agent to open either memory embeds what arrived.
    assert remote_index.sync().added == 1
    assert local_index.sync().added == 1
    assert [hit.name for hit in remote_index.search("carbon fibre woks", k=1)] == ["only-here"]


def test_a_remote_without_a_memory_gets_this_machines(box: Sandbox) -> None:
    """First deploy: nothing to bring back, everything here goes there."""
    MemoryDir(box.local_mem).write("a", "aa")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No memory on me@fakehost yet" in result.stdout
    assert MemoryDir(box.remote_mem).page_names() == ["a"]


def test_a_machine_without_a_memory_receives_the_remotes(box: Sandbox) -> None:
    """A fresh laptop gets the server's pages, and has nothing to send."""
    MemoryDir(box.remote_mem).write("b", "bb")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert MemoryDir(box.local_mem).page_names() == ["b"]
    # Pass 2 now sends the page that just arrived, and the remote keeps its own.
    assert "me@fakehost's memory: added 0 updated 0 kept 1 conflicts 0 removed 0" in result.stdout


def test_neither_machine_having_a_memory_is_not_an_error(box: Sandbox) -> None:
    """Two machines that never wrote a page have nothing to sync."""
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No memory on me@fakehost yet" in result.stdout
    assert "No memory on this machine" in result.stdout
    assert not box.local_mem.exists() and not box.remote_mem.exists()


def test_the_configured_memory_dir_is_honoured_on_both_sides(box: Sandbox) -> None:
    """``memory_dir`` in config.json names the directory on each machine."""
    local_dir = box.tmp / "elsewhere-local"
    remote_dir = box.tmp / "elsewhere-remote"
    (box.local_home / ".kiss" / "config.json").write_text(
        json.dumps({"memory_dir": str(local_dir)})
    )
    (box.remote_home / ".kiss" / "config.json").write_text(
        json.dumps({"memory_dir": str(remote_dir)})
    )
    MemoryDir(local_dir).write("a", "aa")
    MemoryDir(remote_dir).write("b", "bb")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert MemoryDir(local_dir).page_names() == ["a", "b"]
    assert MemoryDir(remote_dir).page_names() == ["a", "b"]
    assert not box.local_mem.exists() and not box.remote_mem.exists()


def test_an_unreachable_remote_is_not_reported_as_synced(box: Sandbox) -> None:
    """A dropped connection fails the run and leaves this machine's pages alone."""
    MemoryDir(box.local_mem).write("a", "aa")
    box.install("ssh", DEAD_SSH)
    result = box.run()
    assert result.returncode == 1
    assert "Could not ask me@fakehost where it keeps its memory" in result.stderr
    assert MemoryDir(box.local_mem).page_names() == ["a"]


def test_a_connection_lost_after_the_first_probe_is_reported(box: Sandbox) -> None:
    """ssh dying once the memory directory is known makes the run incomplete, not silent."""
    MemoryDir(box.local_mem).write("a", "aa")
    MemoryDir(box.remote_mem).write("b", "bb")
    # The first ssh call (where is your memory?) succeeds; every later one dies.
    box.install(
        "ssh",
        FAKE_SSH.replace(
            'export HOME="$REMOTE_HOME"',
            'if [ -e "$REMOTE_HOME/probed" ]; then exit 255; fi; touch "$REMOTE_HOME/probed"\n'
            'export HOME="$REMOTE_HOME"',
        ),
    )
    result = box.run()
    assert result.returncode == 1
    assert "Could not look for me@fakehost's memory (ssh exited 255)" in result.stdout
    assert "not in sync yet" in result.stderr
    assert MemoryDir(box.local_mem).page_names() == ["a"]
    assert MemoryDir(box.remote_mem).page_names() == ["b"]


def test_a_transfer_that_breaks_leaves_both_memories_as_they_were(box: Sandbox) -> None:
    """A tar stream cut short is an incomplete pass; the merge never sees half a page."""
    MemoryDir(box.local_mem).write("a", "aa")
    MemoryDir(box.remote_mem).write("b", "bb")
    box.install(
        "ssh",
        FAKE_SSH.replace(
            'exec bash -c "$*"',
            'case "$*" in *"tar -cf -"*) exit 1 ;; *"tar -xf -"*) cat >/dev/null; exit 1 ;; esac\n'
            'exec bash -c "$*"',
        ),
    )
    result = box.run()
    assert result.returncode == 1
    assert "Could not fetch the memory pages from me@fakehost" in result.stdout
    assert "Could not merge this machine's pages into me@fakehost:" in result.stdout
    assert MemoryDir(box.local_mem).page_names() == ["a"]
    assert MemoryDir(box.remote_mem).page_names() == ["b"]
    assert not (box.remote_home / ".kiss" / "memories.incoming").exists()


def test_the_helper_that_cannot_be_copied_stops_the_push(box: Sandbox) -> None:
    """Without merge_memory_pages.py on the remote, nothing is sent there."""
    MemoryDir(box.local_mem).write("a", "aa")
    box.install("scp", "#!/bin/bash\nexit 1\n")
    result = box.run()
    assert result.returncode == 1
    assert "Could not copy merge_memory_pages.py to me@fakehost" in result.stdout
    assert not box.remote_mem.exists()


def test_a_remote_answering_nonsense_for_its_home_directory_stops_the_run(
    box: Sandbox,
) -> None:
    """A one-line or relative answer to the opening probe is not a place to stage pages."""
    box.install("ssh", "#!/bin/bash\necho relative/path\n")
    result = box.run()
    assert result.returncode == 1
    assert "Could not tell where me@fakehost keeps its memory (answer: 'relative/path')" in (
        result.stderr
    )


def test_the_target_is_required(box: Sandbox) -> None:
    """No user@host is a usage error."""
    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "Usage:" in result.stderr


def test_paths_with_spaces_and_apostrophes_survive_the_remote_command_line(
    box: Sandbox,
) -> None:
    """A memory_dir holding a space and an apostrophe is quoted for the remote shell."""
    local_dir = box.tmp / "the laptop's memory"
    remote_dir = box.tmp / "the server's memory"
    (box.local_home / ".kiss" / "config.json").write_text(
        json.dumps({"memory_dir": str(local_dir)})
    )
    (box.remote_home / ".kiss" / "config.json").write_text(
        json.dumps({"memory_dir": str(remote_dir)})
    )
    MemoryDir(local_dir).write("a", "aa")
    MemoryDir(remote_dir).write("b", "bb")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert MemoryDir(local_dir).page_names() == ["a", "b"]
    assert MemoryDir(remote_dir).page_names() == ["a", "b"]


def test_a_memory_kept_where_the_sync_stages_its_pages_is_refused_untouched(
    box: Sandbox,
) -> None:
    """The scratch directory is removed wholesale, so a memory there is refused, not deleted."""
    remote_dir = box.remote_home / ".kiss" / "memories.incoming"
    (box.remote_home / ".kiss" / "config.json").write_text(
        json.dumps({"memory_dir": str(remote_dir)})
    )
    MemoryDir(remote_dir).write("precious", "must survive")
    MemoryDir(box.local_mem).write("a", "aa")
    result = box.run()
    assert result.returncode == 1
    assert "which this sync uses as its scratch directory" in result.stderr
    assert MemoryDir(remote_dir).page_names() == ["precious"]
    assert MemoryDir(box.local_mem).page_names() == ["a"]
    # The same for a memory below it.
    (box.remote_home / ".kiss" / "config.json").write_text(
        json.dumps({"memory_dir": str(remote_dir / "deeper")})
    )
    MemoryDir(remote_dir / "deeper").write("deep", "must survive too")
    assert box.run().returncode == 1
    assert MemoryDir(remote_dir / "deeper").page_names() == ["deep"]
    # And for the same directory under other spellings: a ``.`` component,
    # and a symlink to it.
    (box.remote_home / ".kiss" / "link-to-incoming").symlink_to(remote_dir)
    for alias in (
        str(box.remote_home / ".kiss" / "." / "memories.incoming"),
        str(box.remote_home / ".kiss" / "link-to-incoming"),
    ):
        (box.remote_home / ".kiss" / "config.json").write_text(json.dumps({"memory_dir": alias}))
        result = box.run()
        assert result.returncode == 1, alias
        assert "which this sync uses as its scratch directory" in result.stderr, alias
        assert MemoryDir(remote_dir).page_names() == ["precious"], alias


def test_a_relative_memory_dir_on_the_remote_is_refused_with_advice(box: Sandbox) -> None:
    """The sync cannot know what a relative directory is relative to, and says so."""
    (box.remote_home / ".kiss" / "config.json").write_text(json.dumps({"memory_dir": "custom"}))
    result = box.run()
    assert result.returncode == 1
    assert "keeps its memory in 'custom', which is not an absolute path" in result.stderr


def test_a_thousand_conflicts_are_all_named(box: Sandbox) -> None:
    """A conflict report longer than a pipe buffer is read whole, not cut short by SIGPIPE."""
    for i in range(1000):
        write(box.local_mem, f"page-{i:04d}", "laptop", "2026-09-20T10:00:00Z")
        write(box.remote_mem, f"page-{i:04d}", "server", "2026-09-20T10:00:00Z")
    result = box.run()
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("in the same second; each keeps its own copy.") == 1000
    assert "1000 page(s) differ between the two machines" in result.stdout


def test_a_page_deleted_on_one_machine_is_deleted_on_the_other(box: Sandbox) -> None:
    """``memory_delete`` on either machine reaches the other instead of being undone by it."""
    for mem in (box.local_mem, box.remote_mem):
        write(mem, "wrong-here", "a wrong lesson", "2026-09-20T10:00:00Z")
        write(mem, "wrong-there", "another wrong lesson", "2026-09-20T10:00:00Z")
        write(mem, "still-right", "a good lesson", "2026-09-20T10:00:00Z")
    MemoryDir(box.local_mem).delete("wrong-here")
    MemoryDir(box.remote_mem).delete("wrong-there")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert MemoryDir(box.local_mem).page_names() == ["still-right"]
    assert MemoryDir(box.remote_mem).page_names() == ["still-right"]
    # The tombstones are on both machines now, so a third one learns of the deletions too.
    for mem in (box.local_mem, box.remote_mem):
        assert sorted(p.name for p in (mem / ".tombstones").iterdir()) == [
            "wrong-here", "wrong-there",
        ]
    # ``kept 2``: the unchanged page, and the deleted one held out by this machine's tombstone.
    assert "This machine's memory: added 0 updated 0 kept 2 conflicts 0 removed 1" in result.stdout
    # Pass 2 carries the local deletion out there; the remote's own is done already.
    assert "me@fakehost's memory: added 0 updated 0 kept 1 conflicts 0 removed 1" in result.stdout
    assert "hold the same pages" in result.stdout
    # The next sync has nothing left to do, and the deleted pages stay deleted.
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "This machine's memory: added 0 updated 0 kept 1 conflicts 0 removed 0" in result.stdout
    assert MemoryDir(box.local_mem).page_names() == ["still-right"]
    assert MemoryDir(box.remote_mem).page_names() == ["still-right"]


def test_a_deletion_travels_on_to_a_machine_that_never_had_the_page(box: Sandbox) -> None:
    """A tombstone is copied even where there is no page to remove, so it reaches every machine."""
    tombstone(box.remote_mem, "long-gone", "2026-09-20T10:00:00Z")
    MemoryDir(box.remote_mem).write("b", "bb")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert (box.local_mem / ".tombstones" / "long-gone").read_text() == "2026-09-20T10:00:00Z\n"
    assert MemoryDir(box.local_mem).page_names() == ["b"]


def test_a_page_written_again_after_its_deletion_comes_back_everywhere(box: Sandbox) -> None:
    """A page newer than the deletion of its namesake wins, and the tombstone is dropped."""
    tombstone(box.remote_mem, "lesson", "2026-09-19T10:00:00Z")
    write(box.local_mem, "lesson", "learned again", "2026-09-20T10:00:00Z")
    tombstone(box.local_mem, "other", "2026-09-19T10:00:00Z")
    write(box.remote_mem, "other", "written again there", "2026-09-20T10:00:00Z")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    for mem in (box.local_mem, box.remote_mem):
        assert MemoryDir(mem).page_names() == ["lesson", "other"]
        assert MemoryDir(mem).read("lesson").body.strip() == "learned again"
        assert MemoryDir(mem).read("other").body.strip() == "written again there"
        assert list((mem / ".tombstones").iterdir()) == []
    assert "This machine's memory: added 1 updated 0 kept 0 conflicts 0 removed 0" in result.stdout
    assert "me@fakehost's memory: added 1 updated 0 kept 1 conflicts 0 removed 0" in result.stdout


def test_a_deletion_and_a_change_in_the_same_second_keep_the_page(box: Sandbox) -> None:
    """Too close to call goes to the page: a deletion is the destructive choice."""
    write(box.local_mem, "close-call", "changed", "2026-09-20T10:00:00Z")
    tombstone(box.remote_mem, "close-call", "2026-09-20T10:00:00Z")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert MemoryDir(box.local_mem).page_names() == ["close-call"]
    assert MemoryDir(box.remote_mem).page_names() == []
    assert (box.remote_mem / ".tombstones" / "close-call").exists()
    assert "both machines changed (or one deleted) close-call in the same second" in result.stdout
    assert "This machine's memory: added 0 updated 0 kept 0 conflicts 1 removed 0" in result.stdout
    assert "me@fakehost's memory: added 0 updated 0 kept 0 conflicts 1 removed 0" in result.stdout


def test_a_page_edited_by_hand_travels(box: Sandbox) -> None:
    """Same ``updated`` on both sides, one file modified later: that copy is the edit."""
    stamp = "2026-09-20T10:00:00Z"
    write(box.local_mem, "edited-here", "agent text, then fixed by hand", stamp)
    write(box.remote_mem, "edited-here", "agent text", stamp)
    os.utime(box.local_mem / "edited-here.md", (epoch(stamp) + 3600, epoch(stamp) + 3600))
    write(box.local_mem, "edited-there", "agent text", stamp)
    write(box.remote_mem, "edited-there", "agent text, then fixed by hand there", stamp)
    os.utime(box.remote_mem / "edited-there.md", (epoch(stamp) + 3600, epoch(stamp) + 3600))
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    for mem in (box.local_mem, box.remote_mem):
        assert MemoryDir(mem).read("edited-here").body.strip() == "agent text, then fixed by hand"
        assert (
            MemoryDir(mem).read("edited-there").body.strip()
            == "agent text, then fixed by hand there"
        )
    assert "This machine's memory: added 0 updated 1 kept 1 conflicts 0 removed 0" in result.stdout
    assert "me@fakehost's memory: added 0 updated 1 kept 1 conflicts 0 removed 0" in result.stdout
    assert "hold the same pages" in result.stdout


def test_a_clock_running_ahead_cannot_make_an_older_edit_win(box: Sandbox) -> None:
    """Two changes closer together than the measured clock difference are a conflict.

    The remote's clock is two minutes ahead: its python3 is a shim that
    moves ``time.time`` before running the probe.  Without the measurement,
    the server's copy of ``close`` -- stamped 30 s later by that fast clock
    though written earlier -- would silently overwrite the laptop's.
    """
    remote_bin = box.remote_home / "bin"
    remote_bin.mkdir()
    shim = remote_bin / "python3"
    shim.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = -c ]; then\n'
        f'    exec {sys.executable} -c "import time\n'
        "time.time = (lambda real: lambda: real() + 120)(time.time)\n"
        '$2" "${@:3}"\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    box.install(
        "ssh",
        FAKE_SSH.replace(
            'export HOME="$REMOTE_HOME"',
            'export HOME="$REMOTE_HOME" PATH="$REMOTE_HOME/bin:$PATH"',
        ),
    )
    write(box.local_mem, "close", "laptop, really the later edit", "2026-09-20T10:00:00Z")
    write(box.remote_mem, "close", "server, stamped by a fast clock", "2026-09-20T10:00:30Z")
    # Exactly the clock difference apart: written at the same real moment.
    write(box.local_mem, "boundary", "laptop", "2026-09-20T10:00:00Z")
    write(box.remote_mem, "boundary", "server", "2026-09-20T10:02:00Z")
    write(box.local_mem, "far", "laptop, an hour earlier", "2026-09-20T09:00:00Z")
    write(box.remote_mem, "far", "server, an hour later", "2026-09-20T10:00:00Z")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    # The clock difference is measured as the largest it may be, given the
    # probe's round trip (well under a second here), and rounded up.
    match = re.search(r"The clock on me@fakehost is about (\d+)s off", result.stdout)
    assert match and 120 <= int(match.group(1)) <= 125, result.stdout
    skew = match.group(1)
    assert MemoryDir(box.local_mem).read("close").body.strip() == "laptop, really the later edit"
    assert MemoryDir(box.remote_mem).read("close").body.strip() == "server, stamped by a fast clock"
    assert MemoryDir(box.local_mem).read("boundary").body.strip() == "laptop"
    assert MemoryDir(box.remote_mem).read("boundary").body.strip() == "server"
    for name in ("close", "boundary"):
        assert f"both machines changed (or one deleted) {name} within {skew}s of each other" in (
            result.stdout
        )
    for mem in (box.local_mem, box.remote_mem):
        assert MemoryDir(mem).read("far").body.strip() == "server, an hour later"
    assert "This machine's memory: added 0 updated 1 kept 0 conflicts 2 removed 0" in result.stdout
    assert "2 page(s) differ between the two machines" in result.stdout


def test_a_relative_memory_dir_here_is_refused_with_advice(box: Sandbox) -> None:
    """This machine's memory_dir gets the same check as the remote's."""
    (box.local_home / ".kiss" / "config.json").write_text(json.dumps({"memory_dir": "custom"}))
    MemoryDir(box.remote_mem).write("b", "bb")
    result = box.run()
    assert result.returncode == 1
    assert "This machine keeps its memory in 'custom', which is not an absolute path" in (
        result.stderr
    )
    assert MemoryDir(box.remote_mem).page_names() == ["b"]


# --- merge_memory_pages.py on its own ---------------------------------------


def merge(source: Path, dest: Path, *options: str) -> subprocess.CompletedProcess[str]:
    """Run the merge script the way the sync runs it."""
    return subprocess.run(
        [sys.executable, str(MERGE), *options, str(source), str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_merge_skips_everything_that_is_not_a_page(tmp_path: Path) -> None:
    """Index files, debris, symlinks, sub-directories and bad names stay behind."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    write(src, "good-page", "kept", "2026-09-20T10:00:00Z")
    (src / "hashed-bow-v1.sqlite3").write_bytes(b"index")
    (src / ".DS_Store").write_text("debris")
    (src / "backup.md~").write_text("debris")
    (src / "Bad_Name.md").write_text("not a page name")
    (src / "notes.txt").write_text("not markdown")
    (src / "page.sync-conflict-20260920.md").write_text("a sync tool's leftover")
    (src / "sub.md").mkdir()
    (src / "sub.md" / "nested.md").write_text("in a sub-directory")
    (src / "link.md").symlink_to(src / "good-page.md")
    result = merge(src, dst)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 1 updated 0 kept 0 conflicts 0 removed 0\n"
    assert sorted(p.name for p in dst.iterdir()) == ["good-page.md"]


def test_merge_recurses_one_level_into_domain_memories_only(tmp_path: Path) -> None:
    """``src/kiss/*.md`` lands in ``dst/kiss/``; deeper levels and symlinked dirs stay behind."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(src / "kiss", "repo-fact", "about the repo", "2026-09-20T10:00:00Z")
    write(src / "kiss" / "deeper", "too-deep", "two levels down", "2026-09-20T10:00:00Z")
    write(src / "other", "elsewhere", "another domain", "2026-09-20T10:00:00Z")
    (src / "linked").symlink_to(src / "other")
    write(dst / "kiss", "repo-fact", "older view", "2026-09-19T10:00:00Z")
    result = merge(src, dst)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 1 updated 1 kept 0 conflicts 0 removed 0\n"
    assert sorted(p.name for p in dst.iterdir()) == ["kiss", "other"]
    assert (dst / "kiss" / "repo-fact.md").read_bytes() == (
        (src / "kiss" / "repo-fact.md").read_bytes()
    )
    assert not (dst / "kiss" / "deeper").exists()
    assert (dst / "other" / "elsewhere.md").exists()


def test_merge_falls_back_to_the_file_time_without_a_parsable_stamp(tmp_path: Path) -> None:
    """Pages without frontmatter, with a bad stamp or an unterminated block use mtime."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    cases = {
        "no-frontmatter": ("plain body\n", "plain body, older\n"),
        "bad-stamp": (page_text("new", "yesterday"), page_text("old", "not a date")),
        "no-stamp": ("---\ntitle: x\n---\nnew\n", "---\ntitle: x\n---\nold\n"),
        # The source's block never closes, so its ancient stamp is body text,
        # not frontmatter (as kiss.core.memoryfield.pages reads it), and its
        # newer mtime decides against the destination's real, older stamp.
        "unterminated": (
            "---\nupdated: '2001-01-01T00:00:00Z'\nno end\n",
            page_text("old", "2020-01-01T00:00:00Z"),
        ),
    }
    for name, (newer, older) in cases.items():
        (src / f"{name}.md").write_text(newer)
        (dst / f"{name}.md").write_text(older)
        os.utime(src / f"{name}.md", (2_000_000_000, 2_000_000_000))
        os.utime(dst / f"{name}.md", (1_000_000_000, 1_000_000_000))
    result = merge(src, dst)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 0 updated 4 kept 0 conflicts 0 removed 0\n"
    for name, (newer, _) in cases.items():
        assert (dst / f"{name}.md").read_text() == newer
        # copy2 keeps the source's mtime, which is what a later comparison reads.
        assert (dst / f"{name}.md").stat().st_mtime == 2_000_000_000


def test_merge_compares_bytes_so_line_endings_count(tmp_path: Path) -> None:
    """An LF page and its CRLF twin with one stamp differ in bytes: a conflict, not 'kept'."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    text = page_text("same words", "2026-09-20T10:00:00Z")
    (src / "endings.md").write_bytes(text.replace("\n", "\r\n").encode())
    (dst / "endings.md").write_bytes(text.encode())
    for path in (src / "endings.md", dst / "endings.md"):
        os.utime(path, (1_000_000_000, 1_000_000_000))
    result = merge(src, dst)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 0 updated 0 kept 0 conflicts 1 removed 0\nendings\n"
    assert (dst / "endings.md").read_bytes() == text.encode()


@pytest.mark.skipif(
    sys.platform != "darwin", reason="needs a file the owner cannot replace: chflags uchg"
)
def test_merge_leaves_no_scratch_file_when_a_page_cannot_be_replaced(tmp_path: Path) -> None:
    """A rename that fails after the copy takes the ``*.md.incoming`` scratch file away again.

    The only way to make the rename fail once the copy into the same directory
    succeeded, without a test double, is a file the owner may not replace: the
    BSD immutable flag.  Linux needs root for the equivalent (``chattr +i``).
    """
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    write(src, "locked", "new", "2026-09-20T10:00:00Z")
    write(dst, "locked", "old", "2020-01-01T00:00:00Z")
    # ``os.chflags`` exists on BSD/macOS only: looked up by name so the type
    # checkers pass on every platform without a platform-dependent ignore.
    chflags = getattr(os, "chflags")
    chflags(dst / "locked.md", stat.UF_IMMUTABLE)
    try:
        result = merge(src, dst)
    finally:
        chflags(dst / "locked.md", 0)
    assert result.returncode != 0
    assert "locked.md" in result.stderr
    assert sorted(p.name for p in dst.iterdir()) == ["locked.md"]
    assert MemoryDir(dst).read("locked").body.strip() == "old"


def test_merge_of_a_missing_source_creates_nothing(tmp_path: Path) -> None:
    """A source that does not exist holds no pages; the destination is not even created."""
    result = merge(tmp_path / "absent", tmp_path / "dst")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 0 updated 0 kept 0 conflicts 0 removed 0\n"
    assert not (tmp_path / "dst").exists()


def test_merge_usage_error(tmp_path: Path) -> None:
    """The wrong number of arguments is refused with the usage line."""
    result = subprocess.run(
        [sys.executable, str(MERGE), str(tmp_path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "Usage: merge_memory_pages.py [--tolerance SECONDS] SOURCE_DIR DEST_DIR" in result.stderr
    # A tolerance that is not a number, or is negative, is a usage error too.
    for bad in ("soon", "-1"):
        result = subprocess.run(
            [sys.executable, str(MERGE), "--tolerance", bad, str(tmp_path), str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2, bad
        assert "Usage:" in result.stderr


def test_merge_tolerance_turns_close_calls_into_conflicts(tmp_path: Path) -> None:
    """Stamps closer together than the tolerance are not ordered; farther apart they are."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(src, "close-newer", "src", "2026-09-20T10:01:00Z")
    write(dst, "close-newer", "dst", "2026-09-20T10:00:00Z")
    write(src, "close-older", "src", "2026-09-20T10:00:00Z")
    write(dst, "close-older", "dst", "2026-09-20T10:01:00Z")
    write(src, "far-newer", "src", "2026-09-20T10:03:00Z")
    write(dst, "far-newer", "dst", "2026-09-20T10:00:00Z")
    # Same stamp, hand-edited 60 s later on the source: the file times are
    # just as uncertain, so this is a conflict too at this tolerance.
    write(src, "hand-edit", "src", "2026-09-20T10:00:00Z")
    write(dst, "hand-edit", "dst", "2026-09-20T10:00:00Z")
    later = epoch("2026-09-20T10:01:00Z")
    os.utime(src / "hand-edit.md", (later, later))
    result = merge(src, dst, "--tolerance", "120")
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "added 0 updated 1 kept 0 conflicts 3 removed 0\nclose-newer\nclose-older\nhand-edit\n"
    )
    assert MemoryDir(dst).read("far-newer").body.strip() == "src"
    for name in ("close-newer", "close-older", "hand-edit"):
        assert MemoryDir(dst).read(name).body.strip() == "dst"
    # Without a tolerance the same pages are ordered by their stamps and file times.
    result = merge(src, dst)
    assert result.stdout == "added 0 updated 2 kept 2 conflicts 0 removed 0\n"
    assert MemoryDir(dst).read("close-newer").body.strip() == "src"
    assert MemoryDir(dst).read("hand-edit").body.strip() == "src"
    assert MemoryDir(dst).read("close-older").body.strip() == "dst"


def test_merge_honours_tombstones_on_either_side(tmp_path: Path) -> None:
    """Every combination of a page on one side and a tombstone on the other."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    # Source tombstone newer than the destination's page: page removed, tombstone copied.
    write(dst, "removed", "old", "2026-09-19T10:00:00Z")
    tombstone(src, "removed", "2026-09-20T10:00:00Z")
    # Source tombstone older than the destination's page: page stays, tombstone not copied.
    write(dst, "rewritten", "new", "2026-09-20T10:00:00Z")
    tombstone(src, "rewritten", "2026-09-19T10:00:00Z")
    # Destination tombstone newer than the source's page: page held out.
    write(src, "held-out", "old", "2026-09-19T10:00:00Z")
    tombstone(dst, "held-out", "2026-09-20T10:00:00Z")
    # Destination tombstone older than the source's page: page added, tombstone dropped.
    write(src, "revived", "new", "2026-09-20T10:00:00Z")
    tombstone(dst, "revived", "2026-09-19T10:00:00Z")
    # Tombstones on both sides: the newer deletion time is kept.
    tombstone(src, "twice-newer", "2026-09-20T10:00:00Z")
    tombstone(dst, "twice-newer", "2026-09-19T10:00:00Z")
    tombstone(src, "twice-older", "2026-09-19T10:00:00Z")
    tombstone(dst, "twice-older", "2026-09-20T10:00:00Z")
    # Not tombstones: a bad name, a symlink, a sub-directory.
    (src / ".tombstones" / "Bad Name").write_text("2026-09-20T10:00:00Z\n")
    (src / ".tombstones" / "linked").symlink_to(src / ".tombstones" / "removed")
    (src / ".tombstones" / "nested").mkdir()
    # A tombstone without a readable stamp counts its file time.
    tombstone(src, "unstamped", "deleted at some point")
    deleted = epoch("2026-09-20T10:00:00Z")
    os.utime(src / ".tombstones" / "unstamped", (deleted, deleted))
    write(dst, "unstamped", "old", "2026-09-19T10:00:00Z")
    result = merge(src, dst)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 1 updated 0 kept 1 conflicts 0 removed 2\n"
    assert MemoryDir(dst).page_names() == ["revived", "rewritten"]
    assert sorted(p.name for p in (dst / ".tombstones").iterdir()) == [
        "held-out", "removed", "twice-newer", "twice-older", "unstamped",
    ]
    assert (dst / ".tombstones" / "twice-newer").read_text() == "2026-09-20T10:00:00Z\n"
    assert (dst / ".tombstones" / "twice-older").read_text() == "2026-09-20T10:00:00Z\n"
    assert (dst / ".tombstones" / "removed").read_text() == "2026-09-20T10:00:00Z\n"
    # A symlinked .tombstones directory is not one.
    other = tmp_path / "other"
    write(other, "page", "p", "2026-09-20T10:00:00Z")
    (other / ".tombstones").symlink_to(src / ".tombstones")
    target = tmp_path / "target"
    write(target, "removed", "still here", "2026-09-19T10:00:00Z")
    assert merge(other, target).stdout == "added 1 updated 0 kept 0 conflicts 0 removed 0\n"
    assert MemoryDir(target).page_names() == ["page", "removed"]


def test_memory_dir_delete_leaves_a_tombstone_the_merge_understands(tmp_path: Path) -> None:
    """The real ``MemoryDir.delete`` and ``write`` leave and clear tombstones the merge reads."""
    memory = MemoryDir(tmp_path / "mem")
    memory.write("lesson", "first version")
    memory.delete("lesson")
    stone = tmp_path / "mem" / ".tombstones" / "lesson"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\n", stone.read_text())
    assert memory.page_names() == []
    # The tombstone is newer than an old copy elsewhere, which the merge then removes.
    other = tmp_path / "other"
    write(other, "lesson", "stale copy", "2026-09-01T00:00:00Z")
    assert merge(tmp_path / "mem", other).stdout == (
        "added 0 updated 0 kept 0 conflicts 0 removed 1\n"
    )
    assert MemoryDir(other).page_names() == []
    # Writing the page again clears the tombstone.
    memory.write("lesson", "second version")
    assert not stone.exists()
    assert memory.page_names() == ["lesson"]


def test_merge_reconciles_a_page_beside_its_own_tombstone(tmp_path: Path) -> None:
    """Both present means an interrupted write: the page stays, the stale tombstone goes.

    A source page beside its own tombstone is a page, not a deletion.
    """
    src, dst = tmp_path / "src", tmp_path / "dst"
    # Destination: page plus stale tombstone; the source's tombstone is older than both.
    write(dst, "interrupted", "written again", "2026-09-20T10:00:00Z")
    tombstone(dst, "interrupted", "2026-09-19T10:00:00Z")
    tombstone(src, "interrupted", "2026-09-18T10:00:00Z")
    # Destination: page plus stale tombstone, and the source deleted the page
    # after it was written: the page is what the destination showed, and it goes.
    write(dst, "then-deleted", "old", "2026-09-18T10:00:00Z")
    tombstone(dst, "then-deleted", "2026-09-20T10:00:00Z")
    tombstone(src, "then-deleted", "2026-09-19T10:00:00Z")
    # Source: page beside its own tombstone; the destination has the page, older.
    write(src, "alive", "new", "2026-09-20T10:00:00Z")
    tombstone(src, "alive", "2026-09-21T10:00:00Z")
    write(dst, "alive", "old", "2026-09-19T10:00:00Z")
    result = merge(src, dst)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 0 updated 1 kept 0 conflicts 0 removed 1\n"
    assert MemoryDir(dst).page_names() == ["alive", "interrupted"]
    assert MemoryDir(dst).read("alive").body.strip() == "new"
    assert MemoryDir(dst).read("interrupted").body.strip() == "written again"
    assert sorted(p.name for p in (dst / ".tombstones").iterdir()) == ["then-deleted"]
    assert (dst / ".tombstones" / "then-deleted").read_text() == "2026-09-19T10:00:00Z\n"
    # The reverse direction brings the source in line, and drops its stale tombstone too.
    assert merge(dst, src).stdout == "added 1 updated 0 kept 1 conflicts 0 removed 0\n"
    assert MemoryDir(src).page_names() == ["alive", "interrupted"]
    assert sorted(p.name for p in (src / ".tombstones").iterdir()) == ["then-deleted"]


def test_merge_writes_no_tombstone_through_a_symlinked_directory(tmp_path: Path) -> None:
    """A ``.tombstones`` that is a symlink holds no tombstones and receives none."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "gone").write_text("unrelated file\n")
    dst.mkdir()
    (dst / ".tombstones").symlink_to(elsewhere)
    write(dst, "gone", "page", "2026-09-19T10:00:00Z")
    tombstone(src, "gone", "2026-09-20T10:00:00Z")
    tombstone(src, "never-here", "2026-09-20T10:00:00Z")
    result = merge(src, dst)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 0 updated 0 kept 0 conflicts 0 removed 1\n"
    assert MemoryDir(dst).page_names() == []
    assert (elsewhere / "gone").read_text() == "unrelated file\n"
    assert sorted(p.name for p in elsewhere.iterdir()) == ["gone"]
