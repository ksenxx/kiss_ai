# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 8 (SORCAR-EXT): multi-line skill descriptions leak newlines.

``Skill.description`` is documented as a "One-line summary", and the
two extraction paths in ``_parse_skill_file`` are meant to agree — yet
they didn't.  The body-fallback path collapses whitespace with
``" ".join(paragraph.split())``, but the frontmatter path only called
``.strip()``.  A YAML block scalar (``description: |``) or folded
multi-line description — perfectly valid frontmatter — therefore kept
its inner newlines, which:

* broke the aligned one-entry-per-line ``/skills`` listing
  (``format_skill_listing``), and
* put raw newlines inside the ``<description>...</description>`` line
  of the ``skill`` tool's catalog docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.skills import discover_skills, format_skill_listing


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every user-level skill location into *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    (tmp_path / "project").mkdir()
    return tmp_path


def test_block_scalar_description_is_one_line(isolated_homes: Path) -> None:
    """A YAML block-scalar description must collapse to a single line."""
    project = isolated_homes / "project"
    skill_dir = project / ".kiss" / "skills" / "multi"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: multi\n"
        "description: |\n"
        "  Reviews pull requests carefully\n"
        "  and suggests focused improvements.\n"
        "---\n"
        "Full instructions here.\n",
        encoding="utf-8",
    )
    skills = discover_skills(str(project))
    skill = skills["multi"]
    assert "\n" not in skill.description, (
        f"description kept newlines: {skill.description!r}"
    )
    assert skill.description == (
        "Reviews pull requests carefully and suggests focused improvements."
    )
    # The /skills listing stays one entry per line.
    listing = format_skill_listing({"multi": skill})
    assert len(listing.splitlines()) == 1


def test_frontmatter_and_body_fallback_paths_agree(
    isolated_homes: Path,
) -> None:
    """Both description sources collapse whitespace identically."""
    project = isolated_homes / "project"
    root = project / ".kiss" / "skills"
    (root / "front").mkdir(parents=True)
    (root / "front" / "SKILL.md").write_text(
        "---\ndescription: 'a   b\n\n  c'\n---\nBody.\n", encoding="utf-8",
    )
    (root / "fall").mkdir(parents=True)
    (root / "fall" / "SKILL.md").write_text(
        "---\nname: fall\n---\na   b\n  c\n\nRest of body.\n",
        encoding="utf-8",
    )
    skills = discover_skills(str(project))
    assert skills["front"].description == "a b c"
    assert skills["fall"].description == "a b c"
