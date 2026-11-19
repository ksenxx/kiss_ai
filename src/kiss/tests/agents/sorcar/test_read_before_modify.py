"""End-to-end tests for the tool-level read-before-modify rule of ``UsefulTools``.

``Edit`` and ``Write`` refuse an existing file that this tool instance has
not shown the model through ``Read`` or written itself.  The rule used to be
a sentence in ``SYSTEM.md`` only; these tests pin the enforced behavior.
"""

import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.kiss_tmp import kiss_tmp_dir


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A directory with one text file and one PNG-like binary file."""
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)))
    return tmp_path


def test_edit_refuses_unread_file(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    out = tools.Edit(str(repo / "a.py"), "x = 1", "x = 3")
    assert out.startswith("Error:") and "had not been read" in out and "NOT applied" in out
    assert "    1 | x = 1" in out
    assert (repo / "a.py").read_text() == "x = 1\ny = 2\n"


def test_read_then_edit_succeeds(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    assert "x = 1" in tools.Read(str(repo / "a.py"))
    out = tools.Edit(str(repo / "a.py"), "x = 1", "x = 3")
    assert out.startswith("Successfully replaced 1 occurrence(s) in")
    assert (repo / "a.py").read_text() == "x = 3\ny = 2\n"


def test_windowed_read_counts_as_read(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    assert tools.Read(str(repo / "a.py"), max_lines=1, start_line=2) == "y = 2\n"
    assert tools.Edit(str(repo / "a.py"), "y = 2", "y = 4").startswith("Successfully")


def test_write_refuses_unread_existing_file_but_creates_new_files(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    out = tools.Write(str(repo / "a.py"), "z = 0\n")
    assert out.startswith("Error:") and "has not been read" in out and "overwriting" in out
    assert (repo / "a.py").read_text() == "x = 1\ny = 2\n"
    assert tools.Write(str(repo / "new.py"), "n = 1\n").startswith("Successfully wrote")
    assert (repo / "new.py").read_text() == "n = 1\n"


def test_read_then_write_overwrites(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    tools.Read(str(repo / "a.py"))
    assert tools.Write(str(repo / "a.py"), "z = 0\n").startswith("Successfully wrote")
    assert (repo / "a.py").read_text() == "z = 0\n"


def test_file_written_by_this_instance_can_be_edited_without_read(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    tools.Write(str(repo / "new.py"), "n = 1\n")
    assert tools.Edit(str(repo / "new.py"), "n = 1", "n = 2").startswith("Successfully")
    assert tools.Write(str(repo / "new.py"), "n = 3\n").startswith("Successfully wrote")


def test_binary_read_marks_file_as_read(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    out = tools.Read(str(repo / "pic.png"))
    assert "as image/png" in out
    assert tools.Write(str(repo / "pic.png"), "not a png\n").startswith("Successfully wrote")


def test_relative_and_absolute_paths_denote_the_same_file(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    tools.Read("a.py")
    assert tools.Edit(str(repo / "a.py"), "x = 1", "x = 5").startswith("Successfully")
    assert tools.Edit("./a.py", "y = 2", "y = 6").startswith("Successfully")
    assert (repo / "a.py").read_text() == "x = 5\ny = 6\n"


def test_read_set_is_per_instance(repo: Path) -> None:
    first = UsefulTools(work_dir=str(repo))
    second = UsefulTools(work_dir=str(repo))
    first.Read(str(repo / "a.py"))
    assert second.Edit(str(repo / "a.py"), "x = 1", "x = 9").startswith("Error:")
    assert first.Edit(str(repo / "a.py"), "x = 1", "x = 9").startswith("Successfully")


def test_failed_read_does_not_mark_file(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    assert tools.Read(str(repo / "missing.py")).startswith("Error: File not found")
    (repo / "missing.py").write_text("m = 1\n", encoding="utf-8")
    assert tools.Edit(str(repo / "missing.py"), "m = 1", "m = 2").startswith("Error:")
    # A Read past the end shows nothing of the file either.
    assert "past EOF (file has 2 lines)" in tools.Read(str(repo / "a.py"), start_line=99)
    assert tools.Write(str(repo / "a.py"), "gone\n") == (
        f"Error: {repo / 'a.py'} has not been read in this session. "
        "Call Read on it before overwriting it."
    )
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 2\n"


def test_unsupported_binary_read_does_not_mark_file(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    (repo / "blob.bin").write_bytes(b"\xff\xfe\x00\x01" * 16)
    assert tools.Read(str(repo / "blob.bin")).startswith("Error: Cannot read binary file")
    assert tools.Write(str(repo / "blob.bin"), "text\n").startswith("Error:")
    assert (repo / "blob.bin").read_bytes() == b"\xff\xfe\x00\x01" * 16


def test_oversized_binary_read_does_not_mark_file(repo: Path) -> None:
    from kiss.agents.sorcar import useful_tools

    tools = UsefulTools(work_dir=str(repo))
    big = repo / "big.png"
    big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (useful_tools._MAX_BINARY_READ_BYTES + 1))
    assert "too large" in tools.Read(str(big))
    assert tools.Write(str(big), "text\n").startswith("Error:")


def test_write_overwrites_unread_scratch_file_in_the_agents_temp_dir(repo: Path) -> None:
    """Scratch files, those in the agents' temp directory, may be overwritten
    without a Read.

    All eleven read-before-write refusals in the 2026-09-22 audit were on
    scratch notes (``ideas.md``, a cron run's own report file), each a
    wasted step; so were three of the seven read-before-edit refusals of the
    2026-11-03 audit (``<id>/information.md``, edited by a later session),
    so ``Edit`` of a scratch file goes through unread too, its result showing
    the region changed.  Source files keep the guard, and so do the files of
    a ``tmp`` directory of the work dir, which is a directory like any other.
    """
    tools = UsefulTools(work_dir=str(repo))
    scratch = Path(tempfile.mkdtemp(dir=kiss_tmp_dir())) / "notes.md"
    scratch.write_text("old\n", encoding="utf-8")
    assert tools.Write(str(scratch), "new\n").startswith("Successfully wrote")
    assert scratch.read_text() == "new\n"
    other = UsefulTools(work_dir=str(repo))
    out = other.Edit(str(scratch), "new", "newer")
    assert out.startswith("Successfully replaced 1 occurrence(s)") and "1 | newer" in out
    assert scratch.read_text() == "newer\n"
    assert other.Write(str(repo / "a.py"), "x = 9\n").startswith("Error:")
    out = other.Edit(str(repo / "a.py"), "x = 1", "x = 2")
    assert out.startswith("Error:") and "had not been read" in out
    in_tree = repo / "work" / "tmp" / "state.json"
    in_tree.parent.mkdir(parents=True)
    in_tree.write_text("{}", encoding="utf-8")
    out = other.Write("work/tmp/state.json", "[]")
    assert out.startswith("Error:") and "has not been read" in out
    out = other.Edit(str(in_tree), "{}", "[]")
    assert out.startswith("Error:") and "had not been read" in out
    assert in_tree.read_text() == "{}"


def test_work_dir_in_the_agents_temp_dir_is_not_scratch(tmp_path: Path) -> None:
    """A checkout a task made in its scratch dir and works in, or a file outside
    the agents' temp directory, keeps the read-before-overwrite guard; the
    task's scratch files next to that checkout, and the cron work directory
    under ``$KISS_HOME``, are scratch."""
    task_dir = Path(tempfile.mkdtemp(dir=kiss_tmp_dir()))
    checkout = task_dir / "clone"
    checkout.mkdir()
    source = checkout / "core.py"
    source.write_text("x = 1\n", encoding="utf-8")
    tools = UsefulTools(work_dir=str(checkout))
    out = tools.Write(str(source), "x = 2\n")
    assert out.startswith("Error:") and "has not been read" in out
    assert source.read_text() == "x = 1\n"
    note = task_dir / "PROGRESS.md"
    note.write_text("old", encoding="utf-8")
    assert tools.Write(str(note), "new").startswith("Successfully wrote")
    outside = tmp_path / "elsewhere" / "tmp" / "note.md"
    outside.parent.mkdir(parents=True)
    outside.write_text("old", encoding="utf-8")
    assert tools.Write(str(outside), "new").startswith("Error:")
    from kiss.core.config import kiss_home

    cron_note = kiss_home() / "cron" / "work" / "tmp" / "last-report.md"
    cron_note.parent.mkdir(parents=True, exist_ok=True)
    cron_note.write_text("old", encoding="utf-8")
    try:
        assert tools.Write(str(cron_note), "new").startswith("Successfully wrote")
        assert cron_note.read_text() == "new"
    finally:
        cron_note.unlink()


def test_cron_work_dir_is_scratch_through_a_symlinked_kiss_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``$KISS_HOME`` given as a symlink still marks its cron work files scratch:
    the comparison root is resolved like the file path is."""
    real_home = tmp_path / "realhome"
    (real_home / "cron" / "work").mkdir(parents=True)
    link = tmp_path / "home-link"
    link.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("KISS_HOME", str(link))
    note = link / "cron" / "work" / "report.md"
    note.write_text("old", encoding="utf-8")
    tools = UsefulTools(work_dir=None)
    assert tools.Write(str(note), "new").startswith("Successfully wrote")
    assert (real_home / "cron" / "work" / "report.md").read_text() == "new"
