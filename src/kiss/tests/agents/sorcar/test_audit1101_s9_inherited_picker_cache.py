# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Editing the base SEA of a model-picker SEA must update the picker verdict.

``_registers_as_model`` caches each SEA's ``register_as_model()``
verdict per file revision, and a SEA may inherit that method from a
custom base (``class Child(sea_class(parent))``).  The cache was keyed
on the child's revision alone, so an edit of the parent alone (the
child untouched) kept the old verdict for good: ``model_sea`` still
offered a child whose fresh load no longer registers, and the reverse
edit left a newly registering child out of the picker.  The verdict is
now stamped with every file of the class's inheritance chain.

Real files, the real loader and registry; nothing is mocked.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.core.config import kiss_home


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Isolate the registry state and the session-wide ``SEAS.md`` per test."""
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()
    (kiss_home() / "SEAS.md").unlink(missing_ok=True)


def _write_parent(parent: Path, registers: bool) -> None:
    parent.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
        "class Parent(BaseSea):\n"
        "    def register_as_model(self):\n"
        f"        return {registers}\n",
        encoding="utf-8",
    )


def test_parent_edit_flips_the_inherited_picker_verdict(tmp_path: Path) -> None:
    """Only the parent changes; the child's verdict follows it in both directions."""
    folder = tmp_path / "seas"
    parent = tmp_path / "parent_sea.py"
    name = "derivedpicker"
    child = folder / name / f"{name}_sea.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        "from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        f"class Child(sea_class({str(parent)!r})):\n"
        "    def description(self):\n"
        "        return 'Inherits the model registration of its base.'\n",
        encoding="utf-8",
    )
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")

    _write_parent(parent, True)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child
    stamp = child.stat()

    _write_parent(parent, False)
    sea_commands.refresh_registry()
    assert sea_commands.load_sea(child).register_as_model() is False
    assert (child.stat().st_mtime_ns, child.stat().st_size) == (stamp.st_mtime_ns, stamp.st_size)
    assert sea_commands.model_sea(name) is None, "the picker still offers the child"
    assert name not in sea_commands.model_seas()

    _write_parent(parent, True)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child, "the picker lost the re-enabled child"
    assert sea_commands.model_seas()[name] == child

    # A parent mid-edit does not load: the child is no picker entry
    # while it is broken, and its repair (the child still untouched)
    # brings the child back without a daemon restart.
    parent.write_text("def register_as_model(:\n", encoding="utf-8")
    assert sea_commands.model_sea(name) is None
    assert sea_commands.model_sea(name) is None, "a broken load is cached until a file changes"
    _write_parent(parent, True)
    assert sea_commands.model_sea(name) == child, "the repaired parent was not re-read"


@pytest.mark.parametrize("parent_state", ["broken", "missing"])
def test_a_parent_broken_at_the_first_load_is_re_read_once_repaired(
    tmp_path: Path, parent_state: str
) -> None:
    """The very first load fails in the parent; its repair alone must bring the child in."""
    folder = tmp_path / "seas"
    parent = tmp_path / "parent_sea.py"
    name = "firstbroken"
    child = folder / name / f"{name}_sea.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        "from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        f"class Child(sea_class({str(parent)!r})):\n"
        "    pass\n",
        encoding="utf-8",
    )
    if parent_state == "broken":
        parent.write_text("class Parent(:\n", encoding="utf-8")
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert name in sea_commands.list_commands()
    assert sea_commands.model_sea(name) is None
    assert name not in sea_commands.model_seas()
    stamp = child.stat()

    # Only the parent changes: it now loads and registers.
    _write_parent(parent, True)
    sea_commands.refresh_registry()
    assert sea_commands.load_sea(child).register_as_model() is True
    assert (child.stat().st_mtime_ns, child.stat().st_size) == (stamp.st_mtime_ns, stamp.st_size)
    assert sea_commands.model_sea(name) == child, "the repaired parent was not re-read"
    assert sea_commands.model_seas()[name] == child
