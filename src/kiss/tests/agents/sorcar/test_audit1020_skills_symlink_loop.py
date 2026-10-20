# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""R4 review, issue 3: a looping symlink among a skill's resources must not block activation.

``_walk_resources`` guarded ``os.scandir`` against ``OSError`` but not
the per-entry ``is_file()``, which follows symlinks: a ``loop -> loop``
entry raised ``OSError`` (``ELOOP``) out of ``load_skill_content`` and a
valid skill could not be activated.  The bad entry is now skipped and
the valid resources are still listed.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar.skills import Skill, load_skill_content


def _skill(tmp_path: Path) -> Skill:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: skill\ndescription: d\n---\nbody\n")
    (skill / "a.txt").write_text("x", encoding="utf-8")
    return Skill("skill", "d", str(skill / "SKILL.md"), "user")


def _listed(content: str) -> list[str]:
    return [line[6:-7] for line in content.splitlines() if line.startswith("<file>")]


def test_a_looping_symlink_is_skipped_and_the_valid_resources_are_listed(tmp_path: Path) -> None:
    """``loop -> loop`` beside ``a.txt``: the skill loads and lists ``a.txt`` only."""
    skill = _skill(tmp_path)
    (tmp_path / "skill" / "loop").symlink_to("loop")
    content = load_skill_content(skill)
    assert content.startswith("<skill_content"), content
    assert "body" in content
    assert _listed(content) == ["a.txt"]


def test_a_looping_symlink_in_a_subdirectory_does_not_hide_later_entries(tmp_path: Path) -> None:
    """The loop sits in ``sub/`` before ``sub/z.txt`` in name order; both ``z.txt`` files list."""
    skill = _skill(tmp_path)
    sub = tmp_path / "skill" / "sub"
    sub.mkdir()
    (sub / "b-loop").symlink_to("b-loop")
    (sub / "z.txt").write_text("x", encoding="utf-8")
    (tmp_path / "skill" / "z.txt").write_text("x", encoding="utf-8")
    content = load_skill_content(skill)
    assert _listed(content) == ["a.txt", "sub/z.txt", "z.txt"]
