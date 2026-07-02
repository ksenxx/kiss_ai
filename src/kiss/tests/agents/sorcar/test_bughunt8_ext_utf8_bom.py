# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 8 (SORCAR-EXT): UTF-8 BOM breaks frontmatter parsing.

Windows editors (Notepad, some VS Code configs, PowerShell
``Out-File``) prepend a UTF-8 byte-order mark to Markdown files.  Both
``custom_commands._parse_command_file`` and ``skills._parse_skill_file``
read with ``encoding="utf-8"``, which keeps the BOM as ``\\ufeff`` at
the start of the text — so the ``\\A---`` frontmatter regex no longer
matched.  The declared ``description``/``argument-hint`` were silently
dropped and the raw ``---`` YAML block leaked into the command's prompt
template (for skills, the YAML block was mistaken for the description
via the first-paragraph fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.custom_commands import discover_commands
from kiss.agents.sorcar.skills import discover_skills


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every user-level config location into *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    (tmp_path / "project").mkdir()
    return tmp_path


def test_bom_command_file_frontmatter_is_parsed(isolated_homes: Path) -> None:
    """A BOM-prefixed command file keeps its description and hint."""
    project = isolated_homes / "project"
    cmd_dir = project / ".kiss" / "commands"
    cmd_dir.mkdir(parents=True)
    # encoding="utf-8-sig" writes a real UTF-8 BOM, as Notepad does.
    (cmd_dir / "deploy.md").write_text(
        "---\ndescription: Deploy the app\nargument-hint: '[env]'\n---\n"
        "Deploy to $1.\n",
        encoding="utf-8-sig",
    )
    commands = discover_commands(str(project))
    cmd = commands["deploy"]
    assert cmd.description == "Deploy the app"
    assert cmd.argument_hint == "[env]"
    assert cmd.template == "Deploy to $1."
    assert "---" not in cmd.template


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
