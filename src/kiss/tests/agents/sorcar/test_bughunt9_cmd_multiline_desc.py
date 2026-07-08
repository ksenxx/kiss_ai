# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 9 (SORCAR-EXT): custom-command frontmatter parity with skills.

Skills got the multi-line-description fix in bughunt 8: a YAML block
scalar (``description: |``) collapses to one line so the aligned
``/skills`` listing and the catalog stay one entry per line.  The
exact same bug lived on in ``custom_commands._parse_command_file``:
``description`` and ``argument-hint`` were stored verbatim, so a
block-scalar or folded frontmatter value kept its inner newlines and
broke the aligned one-entry-per-line ``/commands`` listing (and the
``/help`` output that embeds the same strings).

``format_command_listing`` was also inconsistent with
``format_skill_listing``: skills truncate descriptions longer than 100
characters (``desc[:97] + "..."``) while commands printed them at any
length, wrecking the alignment the listing exists to provide.

These are real end-to-end tests: command files are written to a real
project directory and parsed from the real filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.custom_commands import (
    discover_commands,
    format_command_listing,
)


@pytest.fixture
def kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KISS_HOME (and CLAUDE_CONFIG_DIR) at isolated directories."""
    home = tmp_path / ".kisshome"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claudehome"))
    return home


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_block_scalar_description_is_one_line(
    kiss_home: Path, tmp_path: Path,
) -> None:
    """Block-scalar description/argument-hint collapse to single lines."""
    project = tmp_path / "proj"
    _write(
        project / ".kiss" / "commands" / "multi.md",
        "---\n"
        "description: |\n"
        "  Reviews pull requests carefully\n"
        "  and suggests focused improvements.\n"
        "argument-hint: |\n"
        "  [pr-number]\n"
        "  [focus]\n"
        "---\n"
        "Review PR $1.\n",
    )
    commands = discover_commands(str(project))
    cmd = commands["multi"]
    assert "\n" not in cmd.description, (
        f"description kept newlines: {cmd.description!r}"
    )
    assert cmd.description == (
        "Reviews pull requests carefully and suggests focused improvements."
    )
    assert "\n" not in cmd.argument_hint, (
        f"argument-hint kept newlines: {cmd.argument_hint!r}"
    )
    assert cmd.argument_hint == "[pr-number] [focus]"
    # The /commands listing stays one entry per line.
    listing = format_command_listing({"multi": cmd})
    assert len(listing.splitlines()) == 1


def test_listing_truncates_long_description(
    kiss_home: Path, tmp_path: Path,
) -> None:
    """Descriptions over 100 chars are capped exactly like skills."""
    project = tmp_path / "proj"
    long_desc = "verylongword " * 20  # ~260 chars once collapsed
    _write(
        project / ".kiss" / "commands" / "long.md",
        f"---\ndescription: {long_desc.strip()}\n---\nDo the long thing.\n",
    )
    commands = discover_commands(str(project))
    listing = format_command_listing(commands)
    (line,) = listing.splitlines()
    _, _, shown_desc = line.strip().partition("(project) ")
    assert shown_desc.endswith("...")
    assert len(shown_desc) == 100  # 97 chars + "..." — format_skill_listing parity
