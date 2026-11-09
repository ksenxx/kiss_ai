# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Read tool's dedupe and outline mode (WP2a/2b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.config import DEFAULT_CONFIG


def _step_42() -> int:
    """An agent's step counter, as ``UsefulTools.current_step`` takes it."""
    return 42


def _code_file(path: Path, functions: int, filler: int) -> None:
    lines = []
    for i in range(functions):
        lines.append(f"def fn_{i}():")
        lines.extend(f"    x = {j}" for j in range(filler))
    lines.append("class Last:")
    lines.append("    pass")
    path.write_text("\n".join(lines) + "\n")


class TestDedupe:
    def test_repeat_read_of_unchanged_window_is_a_note(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        f = tmp_path / "a.py"
        f.write_text("one\ntwo\nthree\n")
        assert tools.Read("a.py") == "one\ntwo\nthree\n"
        note = tools.Read("a.py")
        assert note == (
            "Unchanged since your earlier Read at step 1 (lines 1-3 of a.py): the content "
            "is already in your context above. Pass force=True to re-read it."
        )
        # A different window is new content; repeating it is deduped too.
        assert tools.Read("a.py", start_line=2, max_lines=1) == (
            "two\n\n[truncated: 1 more lines; Read(file_path, start_line=3, max_lines=N) continues]"
        )
        assert tools.Read("a.py", start_line=2, max_lines=1).startswith(
            "Unchanged since your earlier Read at step 3 (lines 2-2 of a.py)"
        )
        # The key is the window's text: the same text at another range of another file
        # (its continuation note aside) is the content of the first Read.
        (tmp_path / "b.py").write_text("zero\none\ntwo\nthree\nfour\n")
        assert tools.Read("b.py", start_line=2, max_lines=3).startswith(
            "Unchanged since your earlier Read at step 1 (lines 1-3 of a.py)"
        )
        assert tools.Read("b.py", start_line=5, max_lines=1) == "four\n"
        # force re-sends; a changed file re-sends and re-arms the dedupe.
        assert tools.Read("a.py", force=True) == "one\ntwo\nthree\n"
        f.write_text("one\ntwo\nfour\n")
        assert tools.Read("a.py") == "one\ntwo\nfour\n"
        assert tools.Read("a.py").startswith("Unchanged since your earlier Read at step 8 ")
        # The agent's step counter, when it sets one, is what the note cites.
        tools.current_step = _step_42
        (tmp_path / "c.py").write_text("c\n")
        assert tools.Read("c.py") == "c\n"
        assert tools.Read("c.py").startswith("Unchanged since your earlier Read at step 42 ")
        # The agent forgets shown reads when they leave the model's context
        # (its context_reset_hook is this method): every window is new again.
        tools.forget_reads()
        assert tools.Read("a.py") == "one\ntwo\nfour\n"
        assert tools.Read("b.py", start_line=2, max_lines=3).startswith("one\ntwo\nthree\n")

    def test_dedupe_disabled_by_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(DEFAULT_CONFIG, "read_dedupe", False)
        tools = UsefulTools(work_dir=str(tmp_path))
        (tmp_path / "a.txt").write_text("hello\n")
        assert tools.Read("a.txt") == "hello\n"
        assert tools.Read("a.txt") == "hello\n"

    def test_errors_and_empty_files_are_not_deduped(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        (tmp_path / "e.txt").write_text("")
        assert tools.Read("e.txt") == "(file is empty)"
        assert tools.Read("e.txt") == "(file is empty)"
        assert tools.Read("missing.txt").startswith("Error: File not found")
        assert tools.Read("missing.txt").startswith("Error: File not found")


class TestOutline:
    def test_long_code_file_returns_outline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(DEFAULT_CONFIG, "read_outline_lines", 100)
        tools = UsefulTools(work_dir=str(tmp_path))
        _code_file(tmp_path / "big.py", functions=10, filler=20)
        out = tools.Read("big.py")
        assert out.startswith("big.py: 212 lines,")
        assert "outline of its 11 definitions/headings" in out
        assert "\n1: def fn_0():\n" in out and "\n211: class Last:" in out
        assert "Read(file_path, start_line=N, max_lines=M)" in out
        assert "x = 0" not in out
        # A range read still returns content, and the outline is not deduped
        # against it (different windows).
        assert tools.Read("big.py", start_line=2, max_lines=2) == (
            "    x = 0\n    x = 1\n\n[truncated: 209 more lines; "
            "Read(file_path, start_line=4, max_lines=N) continues]"
        )
        # Explicitly asking for a bigger whole-file window is the same
        # request, and a repeated outline is deduped like any other window.
        assert tools.Read("big.py", max_lines=5000).startswith(
            "Unchanged since your earlier Read at step 1 (outline of big.py)"
        )
        assert tools.Read("big.py", force=True).startswith("big.py: 212 lines,")

    def test_outline_caps_entries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(DEFAULT_CONFIG, "read_outline_lines", 100)
        tools = UsefulTools(work_dir=str(tmp_path))
        _code_file(tmp_path / "huge.py", functions=450, filler=1)
        out = tools.Read("huge.py")
        assert "... 51 more entries" in out
        assert "\n400: " not in out  # only the first 400 entries are listed

    def test_markdown_outline_uses_headings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(DEFAULT_CONFIG, "read_outline_lines", 10)
        tools = UsefulTools(work_dir=str(tmp_path))
        body = "".join(f"# Section {i}\n\ntext # not a heading\ndef x():\n" for i in range(6))
        (tmp_path / "doc.md").write_text(body)
        out = tools.Read("doc.md")
        assert "outline of its 6 definitions/headings" in out
        assert "def x()" not in out and "# Section 5" in out

    def test_file_without_symbols_falls_back_to_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(DEFAULT_CONFIG, "read_outline_lines", 10)
        tools = UsefulTools(work_dir=str(tmp_path))
        (tmp_path / "data.csv").write_text("".join(f"{i},{i}\n" for i in range(50)))
        out = tools.Read("data.csv", max_lines=2000)
        assert out.startswith("0,0\n1,1\n") and "outline" not in out

    def test_outline_disabled_by_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(DEFAULT_CONFIG, "read_outline_lines", 0)
        tools = UsefulTools(work_dir=str(tmp_path))
        _code_file(tmp_path / "big.py", functions=10, filler=20)
        assert tools.Read("big.py").startswith("def fn_0():\n")
