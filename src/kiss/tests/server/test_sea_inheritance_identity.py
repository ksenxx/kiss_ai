# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Class identity, anchoring and side effects along a SEA inheritance chain.

* A class is the same class whether its file was imported normally or
  executed by ``load_sea``: it contributes once to a chain even when the
  picker base is the file-loaded copy and the script derives from the
  imported one.
* A relative ``work_dir`` is kept as written by the fold (the launcher
  anchors it at the calling task's directory, whichever file set it),
  whichever picker or subclass sits above or below it in the chain.
* A SEA that only inherits ``register_as_model()`` is a model-picker
  entry.
* ``check_sea`` never runs ``on_picked_as_model`` (picking has side
  effects) but still rejects one of the wrong shape.
* A ``system_prompt`` method's return is the run's system prompt as
  given, appended to or rewritten alike, and is not forwarded to
  sub-agents: they inherit the caller-supplied base prompt and suffix.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.agent_dispatch import RunOptions, inherit_from_parent
from kiss.agents.sorcar.sea_commands import (
    SeaError,
    base_settings,
    base_system_prompt,
    check_sea,
    load_sea,
)
from kiss.agents.sorcar.sea_settings import anchored_work_dir
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.config import kiss_home

PICKER_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea

class PickerSea(BaseSea):
    def register_as_model(self):
        return True
    def settings(self, settings):
        return settings | {'work_dir': 'assets'}
    def system_prompt(self, system_prompt):
        return system_prompt + '\\n\\nPICKER PROTOCOL'
"""


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[Path]:
    folder = tmp_path / "seas"
    folder.mkdir()
    home = kiss_home()
    home.mkdir(parents=True, exist_ok=True)
    seas_md = home / "SEAS.md"
    previous = seas_md.read_text(encoding="utf-8") if seas_md.is_file() else None
    seas_md.write_text(f"{folder}\n", encoding="utf-8")
    sea_commands._reset_for_tests()
    yield folder
    if previous is None:
        seas_md.unlink(missing_ok=True)
    else:
        seas_md.write_text(previous, encoding="utf-8")
    sea_commands._reset_for_tests()


def test_an_imported_class_and_its_file_loaded_copy_are_one_class(
    registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    picker = _write(registry / "picker" / "picker_sea.py", PICKER_SEA)
    monkeypatch.syspath_prepend(str(registry / "picker"))
    import picker_sea  # type: ignore[import-not-found]  # noqa: PLC0415

    class Child(picker_sea.PickerSea):
        def system_prompt(self, system_prompt: str) -> str:
            return system_prompt + "\n\nCHILD"

    child = Child()
    child.path = registry / "child" / "child_sea.py"
    # The picker layer is the file-loaded copy, the script's base the
    # imported module's class: distinct class objects, one source file.
    layers: list[BaseSea] = [load_sea(picker), child]
    assert type(layers[0]) is not picker_sea.PickerSea
    assert base_system_prompt(layers, "BASE") == "BASE\n\nPICKER PROTOCOL\n\nCHILD"
    assert base_system_prompt([child], "BASE") == "BASE\n\nPICKER PROTOCOL\n\nCHILD"


def test_relative_work_dir_is_kept_as_written_whichever_file_set_it(registry: Path) -> None:
    picker = _write(registry / "picker" / "picker_sea.py", PICKER_SEA)
    plain = _write(registry / "plain" / "plain_sea.py", """
from kiss.agents.seas.base.base_sea import BaseSea

class PlainSea(BaseSea):
    def prompt(self, task):
        return task
""")
    inheriting = _write(registry / "inheriting" / "inheriting_sea.py", """
from kiss.agents.sorcar.sea_commands import sea_class

class InheritingSea(sea_class('picker')):
    pass
""")
    overriding = _write(registry / "overriding" / "overriding_sea.py", """
from kiss.agents.sorcar.sea_commands import sea_class

class OverridingSea(sea_class('picker')):
    def settings(self, settings):
        return settings | {'work_dir': 'out'}
""")
    sea_commands.refresh_registry()
    # The fold never anchors a relative path: the launcher resolves it
    # against the calling task's directory, whichever file set it.
    assert base_settings([load_sea(picker)])["work_dir"] == "assets"
    assert base_settings([load_sea(picker), load_sea(plain)])["work_dir"] == "assets"
    assert base_settings([load_sea(inheriting)])["work_dir"] == "assets"
    assert base_settings([load_sea(overriding)])["work_dir"] == "out"
    # A native absolute base: Path renders the result with the OS
    # separator, so a POSIX literal would not round-trip on Windows.
    assert anchored_work_dir("out", str(registry)) == str(registry / "out")


def test_inherited_model_registration_is_discovered(registry: Path) -> None:
    _write(registry / "picker" / "picker_sea.py", PICKER_SEA)
    derived = _write(registry / "derived" / "derived_sea.py", """
from kiss.agents.sorcar.sea_commands import sea_class

class DerivedSea(sea_class('picker')):
    pass
""")
    plain = _write(registry / "plain" / "plain_sea.py", """
from kiss.agents.seas.base.base_sea import BaseSea

class PlainSea(BaseSea):
    pass
""")
    sea_commands.refresh_registry()
    assert sea_commands._registers_as_model(derived) is True
    assert sea_commands._registers_as_model(plain) is False
    assert {"picker", "derived"} <= set(sea_commands.model_seas())


def test_check_sea_does_not_pick_the_model_but_checks_the_hook_shape(tmp_path: Path) -> None:
    tally = tmp_path / "picked.txt"
    sea = _write(tmp_path / "router" / "router_sea.py", f"""
from pathlib import Path
from kiss.agents.seas.base.base_sea import BaseSea

class RouterSea(BaseSea):
    def description(self):
        return 'router'
    def register_as_model(self):
        return True
    def on_picked_as_model(self, work_dir):
        Path({str(tally)!r}).write_text(work_dir)
        return 'picked'
""")
    assert check_sea(sea).description == "router"
    assert not tally.exists(), "check_sea must not run on_picked_as_model"
    sea.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class RouterSea(BaseSea):
    def description(self):
        return 'router'
    def on_picked_as_model(self):
        return 'picked'
""", encoding="utf-8")
    with pytest.raises(SeaError, match="on_picked_as_model.*must accept the work directory"):
        check_sea(sea)


def test_system_prompt_hook_result_is_the_run_prompt_and_is_not_forwarded(
    tmp_path: Path,
) -> None:
    """The hook's return is used as given; children inherit only the caller's base and suffix."""
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original_run = parent_class.run
    composed: list[str] = []

    def stub_run(self_agent: Any, **kwargs: Any) -> str:
        composed.append(str(kwargs.get("system_prompt")))
        return "success: true\nis_continue: false\nsummary: ok\n"

    def rewrite(system_prompt: str) -> str:
        return system_prompt.replace("SUFFIX", "REWRITTEN")

    def append(system_prompt: str) -> str:
        return system_prompt + "APPENDED"

    def replace(system_prompt: str) -> str:
        return "ONLY THIS"

    parent_class.run = stub_run
    try:
        for hook, expected in ((rewrite, "BASEREWRITTEN"), (append, "BASESUFFIXAPPENDED"),
                               (replace, "ONLY THIS")):
            agent = SorcarAgent("parent")
            agent.run(
                prompt_template="t", work_dir=str(tmp_path), base_system_prompt="BASE",
                system_prompt="SUFFIX", system_prompt_hook=hook, web_tools=False,
            )
            # (The daemon appends its own operational notes after the hook.)
            assert composed[-1].startswith(expected), (hook.__name__, composed[-1])
            if hook is not append:
                assert "SUFFIX" not in composed[-1]
            # Appended, rewritten or replaced: the caller's own base prompt and
            # suffix are what a sub-agent inherits, as with ``prompt()``.
            options = inherit_from_parent(agent, "m", None, RunOptions()).options
            assert options.system_prompt == "BASE", hook.__name__
            assert options.add_to_system_prompt == "SUFFIX", hook.__name__
    finally:
        parent_class.run = original_run


def test_base_sea_itself_is_never_a_layer() -> None:
    """A SEA that overrides nothing contributes no method (the defaults are identities)."""
    class EmptySea(BaseSea):
        pass

    assert sea_commands.defines([EmptySea()], "system_prompt") is False
    assert base_system_prompt([EmptySea()], "BASE") == "BASE"
