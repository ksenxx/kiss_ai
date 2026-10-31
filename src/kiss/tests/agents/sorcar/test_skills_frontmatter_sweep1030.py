# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-30 (S11): a SKILL.md whose frontmatter YAML cannot be built must not break tools.

``yaml.safe_load`` resolves a plain scalar that looks like a date into a
``datetime``: an impossible one (``2025-02-30``, an hour of 25) raises
``ValueError``, as does an integer longer than Python's 4300-digit
conversion limit, and a deeply nested block raises ``RecursionError``.
None of these is a ``yaml.YAMLError``, so ``parse_frontmatter`` let them
escape through ``discover_skills`` into ``make_skill_tool``, which every
Sorcar task calls while it assembles its tools: one user-authored skill
on the discovery path made every task fail at start.  The frontmatter of
such a file now counts as empty metadata, like any other malformed block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.skills import make_skill_tool, parse_frontmatter

_UNBUILDABLE_FRONTMATTER = [
    "updated: 2025-02-30",
    "started: 2024-01-01 25:00:00",
    "count: " + "1" * 5000,
    "nested: " + "[" * 3000 + "]" * 3000,
]
_IDS = ["impossible-date", "impossible-hour", "over-long-int", "deep-nesting"]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project directory with user-level skill homes isolated under *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claudehome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    work_dir = tmp_path / "project"
    good = work_dir / ".kiss" / "skills" / "good"
    good.mkdir(parents=True)
    (good / "SKILL.md").write_text("---\ndescription: A good skill\n---\nGood body.\n")
    return work_dir


def _write_bad_skill(work_dir: Path, frontmatter: str) -> None:
    bad = work_dir / ".claude" / "skills" / "bad"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nBad skill body.\n")


@pytest.mark.parametrize("frontmatter", _UNBUILDABLE_FRONTMATTER, ids=_IDS)
def test_unbuildable_frontmatter_is_empty_metadata(tmp_path: Path, frontmatter: str) -> None:
    """``parse_frontmatter`` returns empty metadata and the body instead of raising."""
    path = tmp_path / "SKILL.md"
    path.write_text(f"---\n{frontmatter}\n---\nThe body.\n")
    assert parse_frontmatter(path) == ({}, "The body.\n")


@pytest.mark.parametrize("frontmatter", _UNBUILDABLE_FRONTMATTER, ids=_IDS)
def test_one_bad_skill_does_not_break_the_skill_tool(project: Path, frontmatter: str) -> None:
    """The tool is built, lists both skills, and loads the bad one by its body."""
    _write_bad_skill(project, frontmatter)
    tool = make_skill_tool(str(project))
    assert tool is not None
    assert "<name>good</name>" in (tool.__doc__ or "")
    assert "<name>bad</name>" in (tool.__doc__ or "")
    assert "Bad skill body." in tool("bad")
    assert "Good body." in tool("good")
