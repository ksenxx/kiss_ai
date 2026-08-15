# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 8 (SORCAR-EXT): UTF-8 BOM breaks frontmatter parsing.

Windows editors (Notepad, some VS Code configs, PowerShell
``Out-File``) prepend a UTF-8 byte-order mark to Markdown files.
``skills._parse_skill_file`` reads with ``encoding="utf-8"``, which
keeps the BOM as ``\\ufeff`` at the start of the text — so the
``\\A---`` frontmatter regex no longer matched.  The declared
``description`` was silently dropped and the YAML block was mistaken
for the description via the first-paragraph fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.skills import discover_skills


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every user-level config location into *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    (tmp_path / "project").mkdir()
    return tmp_path


def test_bom_skill_file_frontmatter_is_parsed(isolated_homes: Path) -> None:
    """A BOM-prefixed SKILL.md keeps its frontmatter description."""
    project = isolated_homes / "project"
    skill_dir = project / ".kiss" / "skills" / "bomskill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: bomskill\ndescription: Handles BOM files\n---\n"
        "Body instructions.\n",
        encoding="utf-8-sig",
    )
    skills = discover_skills(str(project))
    assert skills["bomskill"].description == "Handles BOM files"
