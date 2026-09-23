# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the lenient JSON decoding of ``parse_tasks_json``.

In one day (2026-09-21) 21 ``run_commands_parallel`` calls from reviewer
sub-agents were rejected with "commands must be a JSON array of
strings" because the model wrote grep's ``\\|`` / ``\\(`` inside JSON
strings (an invalid JSON escape) or a heredoc with raw newlines.  Each
rejection cost a full model step.  The parser now accepts both and the
error for the remaining cases explains the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.fanout_guard import parse_tasks_json
from kiss.agents.sorcar.useful_tools import UsefulTools


def test_lone_backslash_escapes_are_repaired() -> None:
    """``\\|`` and ``\\(`` inside a JSON string decode to the shell text."""
    raw = r'["grep -n \"a\|b\" f.py", "grep -n \"def run\(\" g.py"]'
    assert parse_tasks_json(raw, "commands") == [
        r'grep -n "a\|b" f.py',
        r'grep -n "def run\(" g.py',
    ]


def test_valid_pairs_stay_intact_next_to_lone_backslashes() -> None:
    """An already-doubled backslash is not doubled again by the repair."""
    raw = r'["grep \\. x", "grep \| y"]'
    assert parse_tasks_json(raw, "commands") == [r"grep \. x", r"grep \| y"]


def test_raw_newlines_inside_strings_are_accepted() -> None:
    """A heredoc typed with real newlines is one command."""
    raw = '["python - <<EOF\nprint(1)\nEOF"]'
    assert parse_tasks_json(raw, "commands") == ["python - <<EOF\nprint(1)\nEOF"]


def test_strict_json_is_unchanged() -> None:
    """Correctly escaped input decodes exactly as before."""
    assert parse_tasks_json(r'["a\\|b", "c\nd"]') == ["a\\|b", "c\nd"]


def test_unrepairable_input_names_the_backslash_rule() -> None:
    """An unescaped quote still fails, and the error says how to escape."""
    with pytest.raises(ValueError) as info:
        parse_tasks_json(r'["rg -n \"x|y\\(" src"]', "commands")
    message = str(info.value)
    assert message.startswith("commands must be a JSON array of strings")
    assert "backslash must be doubled" in message


def test_shell_substitution_hint_is_kept() -> None:
    """The ``$(cat file)`` case keeps its dedicated hint."""
    with pytest.raises(ValueError, match="Shell substitutions are not expanded"):
        parse_tasks_json("$(cat tmp/tasks.json)")


def test_run_commands_parallel_runs_grep_with_shell_escapes(tmp_path: Path) -> None:
    """The real tool runs a ``grep \\|`` command written the way models write it."""
    (tmp_path / "f.txt").write_text("alpha\nbeta\ngamma\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    raw = r'["grep -n \"alpha\|gamma\" f.txt"]'
    report = tools.run_commands_parallel(commands=raw, max_workers=1, timeout_seconds=30)
    assert "1:alpha" in report and "3:gamma" in report
    assert "Error" not in report
