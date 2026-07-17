# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8 — ``~`` paths create a literal ``./~/`` directory.

``_absolutize`` treats ``~/notes.txt`` as a plain relative path and
anchors it under ``work_dir``, so ``Write("~/notes.txt", ...)`` silently
creates a directory literally named ``~`` inside the work dir (a classic
footgun: a later ``rm -rf ~`` cleanup attempt is catastrophic), and
Read/Edit on ``~``-prefixed paths never see the user's real home files.
Bash, by contrast, expands ``~`` through the shell — an inconsistency
between sibling tool code paths.
"""

from pathlib import Path

import pytest

from kiss.core.useful_tools import UsefulTools


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``$HOME`` at a throwaway directory for the test process."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_write_tilde_path_targets_home_not_literal_dir(
    fake_home: Path, tmp_path: Path
) -> None:
    work_dir = tmp_path / "wd"
    work_dir.mkdir()
    tools = UsefulTools(work_dir=str(work_dir))

    out = tools.Write("~/notes.txt", "HELLO\n")

    assert "Successfully wrote" in out, out
    # Must land in the (fake) home directory...
    assert (fake_home / "notes.txt").read_text() == "HELLO\n"
    # ...and must NOT create a literal '~' directory in the work dir.
    assert not (work_dir / "~").exists()


def test_read_tilde_path_reads_home_file(fake_home: Path, tmp_path: Path) -> None:
    (fake_home / "cfg.txt").write_text("CFG\n")
    tools = UsefulTools(work_dir=str(tmp_path))

    assert tools.Read("~/cfg.txt") == "CFG\n"


def test_edit_tilde_path_edits_home_file(fake_home: Path, tmp_path: Path) -> None:
    (fake_home / "cfg.txt").write_text("OLD\n")
    tools = UsefulTools(work_dir=str(tmp_path))

    out = tools.Edit("~/cfg.txt", "OLD", "NEW")

    assert "Successfully replaced" in out, out
    assert (fake_home / "cfg.txt").read_text() == "NEW\n"
    assert not (tmp_path / "~").exists()


def test_plain_relative_paths_still_anchor_under_work_dir(
    fake_home: Path, tmp_path: Path
) -> None:
    """Regression guard: non-tilde relative paths keep work_dir anchoring."""
    work_dir = tmp_path / "wd"
    work_dir.mkdir()
    tools = UsefulTools(work_dir=str(work_dir))

    out = tools.Write("plain.txt", "P\n")

    assert "Successfully wrote" in out, out
    assert (work_dir / "plain.txt").read_text() == "P\n"
