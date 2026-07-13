# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``sorcar --version`` (GitHub issue #41).

``sorcar --version`` (and its short spelling ``-V``) must print
``sorcar <version>`` — sourced from :data:`kiss.__version__` — to
stdout and exit immediately with status 0, without requiring a task,
a model, API keys, or any other argument.  Before the fix the flag
fell through to normal argument parsing and errored.

These tests drive the real CLI entry point
(:func:`kiss.agents.sorcar.worktree_sorcar_agent.main`) with a
monkeypatched ``sys.argv`` — no mocks of application code.
"""

from __future__ import annotations

import sys

import pytest

from kiss import __version__
from kiss.agents.sorcar import worktree_sorcar_agent


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_version_and_exits_zero(
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``sorcar --version`` / ``sorcar -V`` prints the version and exits 0."""
    monkeypatch.setattr(sys, "argv", ["sorcar", flag])
    with pytest.raises(SystemExit) as excinfo:
        worktree_sorcar_agent.main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"sorcar {__version__}\n"
    assert captured.err == ""


def test_version_flag_wins_over_other_args(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--version`` exits before any task/model handling even with -t."""
    monkeypatch.setattr(
        sys, "argv", ["sorcar", "--version", "-t", "some task", "--worktree"],
    )
    with pytest.raises(SystemExit) as excinfo:
        worktree_sorcar_agent.main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"sorcar {__version__}\n"


def test_version_string_matches_kiss_version_module() -> None:
    """``kiss.__version__`` re-exports ``kiss._version.__version__``."""
    from kiss._version import __version__ as raw_version

    assert __version__ == raw_version
    assert __version__  # non-empty


def test_version_abbreviation_rejected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--vers`` is rejected because the parser sets allow_abbrev=False."""
    monkeypatch.setattr(sys, "argv", ["sorcar", "--vers"])
    with pytest.raises(SystemExit) as excinfo:
        worktree_sorcar_agent.main()
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "sorcar" in captured.err
