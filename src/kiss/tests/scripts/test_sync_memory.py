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
import stat
import subprocess
import sys
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


def write(directory: Path, name: str, body: str, updated: str) -> None:
    """Create a page file directly, as another agent would have left it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(page_text(body, updated))


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
    assert "This machine's memory: added 1 updated 0 kept 0 conflicts 0" in result.stdout
    assert "me@fakehost's memory: added 1 updated 0 kept 1 conflicts 0" in result.stdout
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
    assert "This machine's memory: added 0 updated 1 kept 1 conflicts 0" in result.stdout
    assert "me@fakehost's memory: added 0 updated 1 kept 1 conflicts 0" in result.stdout


def test_a_page_changed_on_both_in_the_same_second_is_a_named_conflict(box: Sandbox) -> None:
    """Same stamp, different content: each machine keeps its own and the page is named."""
    write(box.local_mem, "clash", "laptop view", "2026-09-20T10:00:00Z")
    write(box.remote_mem, "clash", "server view", "2026-09-20T10:00:00Z")
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert MemoryDir(box.local_mem).read("clash").body.strip() == "laptop view"
    assert MemoryDir(box.remote_mem).read("clash").body.strip() == "server view"
    assert "both machines changed clash in the same second" in result.stdout
    assert "1 page(s) differ between the two machines" in result.stdout
    assert "hold the same pages" not in result.stdout


def test_a_second_sync_changes_nothing(box: Sandbox) -> None:
    """Once in sync, a sync copies nothing and rewrites no file."""
    MemoryDir(box.local_mem).write("a", "aa")
    MemoryDir(box.remote_mem).write("b", "bb")
    assert box.run().returncode == 0
    before = {p.name: p.stat().st_mtime_ns for p in box.remote_mem.iterdir()}
    result = box.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "This machine's memory: added 0 updated 0 kept 2 conflicts 0" in result.stdout
    assert "me@fakehost's memory: added 0 updated 0 kept 2 conflicts 0" in result.stdout
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
    assert "me@fakehost's memory: added 0 updated 0 kept 1 conflicts 0" in result.stdout


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


# --- merge_memory_pages.py on its own ---------------------------------------


def merge(source: Path, dest: Path) -> subprocess.CompletedProcess[str]:
    """Run the merge script the way the sync runs it."""
    return subprocess.run(
        [sys.executable, str(MERGE), str(source), str(dest)],
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
    assert result.stdout == "added 1 updated 0 kept 0 conflicts 0\n"
    assert sorted(p.name for p in dst.iterdir()) == ["good-page.md"]


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
    assert result.stdout == "added 0 updated 4 kept 0 conflicts 0\n"
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
    result = merge(src, dst)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 0 updated 0 kept 0 conflicts 1\nendings\n"
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
    os.chflags(dst / "locked.md", stat.UF_IMMUTABLE)
    try:
        result = merge(src, dst)
    finally:
        os.chflags(dst / "locked.md", 0)
    assert result.returncode != 0
    assert "locked.md" in result.stderr
    assert sorted(p.name for p in dst.iterdir()) == ["locked.md"]
    assert MemoryDir(dst).read("locked").body.strip() == "old"


def test_merge_of_a_missing_source_creates_nothing(tmp_path: Path) -> None:
    """A source that does not exist holds no pages; the destination is not even created."""
    result = merge(tmp_path / "absent", tmp_path / "dst")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "added 0 updated 0 kept 0 conflicts 0\n"
    assert not (tmp_path / "dst").exists()


def test_merge_usage_error(tmp_path: Path) -> None:
    """The wrong number of arguments is refused with the usage line."""
    result = subprocess.run(
        [sys.executable, str(MERGE), str(tmp_path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "Usage: merge_memory_pages.py SOURCE_DIR DEST_DIR" in result.stderr
