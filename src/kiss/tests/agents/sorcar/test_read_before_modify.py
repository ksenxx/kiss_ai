"""End-to-end tests for the tool-level read-before-modify rule of ``UsefulTools``.

``Edit`` and ``Write`` refuse an existing file that this tool instance has
not shown the model through ``Read`` or written itself.  The rule used to be
a sentence in ``SYSTEM.md`` only; these tests pin the enforced behavior.
"""

from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A directory with one text file and one PNG-like binary file."""
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)))
    return tmp_path


def test_edit_refuses_unread_file(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    out = tools.Edit(str(repo / "a.py"), "x = 1", "x = 3")
    assert out.startswith("Error:") and "has not been read" in out and "editing" in out
    assert (repo / "a.py").read_text() == "x = 1\ny = 2\n"


def test_read_then_edit_succeeds(repo: Path) -> None:
    tools = UsefulTools(work_dir=str(repo))
    assert "x = 1" in tools.Read(str(repo / "a.py"))
    out = tools.Edit(str(repo / "a.py"), "x = 1", "x = 3")
    assert out.startswith("Successfully replaced 1")
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
