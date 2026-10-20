# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-19 (S6): the SEA registry, loader and linter fixes.

End-to-end through the public entry points (``refresh_registry`` /
``list_commands`` / ``model_seas``, ``lint_all``, ``evaluate_sea``,
``import kiss.agents.sorcar``):

- a ``hidden`` key inside ``model_config`` is no settings key, so the
  registry lists the command and the linter reports nothing (S6-5);
- a Stop landing in a SEA method (``KeyboardInterrupt`` while the
  thread's stop event is set, or the per-tool ``ToolCallInterrupted``)
  propagates as itself instead of a ``SeaError``, so the model picker
  does not memoise a stopped ``register_as_model`` as ``False`` (S6-12);
- ``sea lint <folder>`` checks ``<folder>/<folder>_sea.py`` under the
  folder convention and falls back to the first ``*_sea.py`` (S6-8);
- the package ``__init__`` imports no agent (S6-1).
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands, sea_lint, sea_settings
from kiss.core.config import kiss_home
from kiss.core.stop_signal import set_thread_stop_event
from kiss.core.tool_interrupt import ToolCallInterrupted


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Drop the registry state before and after each test, and the test ``SEAS.md``."""
    sea_commands._reset_for_tests()
    set_thread_stop_event(None)
    yield
    sea_commands._reset_for_tests()
    set_thread_stop_event(None)
    (kiss_home() / "SEAS.md").unlink(missing_ok=True)


def _register(folder: Path) -> None:
    """Point ``$KISS_HOME/SEAS.md`` at *folder* and rescan."""
    home = kiss_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()


def _write_sea(folder: Path, name: str, source: str) -> Path:
    """Write ``<folder>/<name>/<name>_sea.py`` with *source* and return it."""
    path = folder / name / f"{name}_sea.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


_HIDDEN_IN_MODEL_CONFIG = '''\
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "passes a hidden flag to the model, not to the registry"

    def settings(self, settings):
        return settings | {"model_config": {"hidden": True, "nested": {"hidden": True}}}
'''


def test_hidden_inside_model_config_is_no_settings_key(tmp_path: Path) -> None:
    """The registry lists the command and the linter has no ``hidden-not-literal`` finding."""
    folder = tmp_path / "seas"
    sea = _write_sea(folder, "shown", _HIDDEN_IN_MODEL_CONFIG)
    assert sea_settings.declares_hidden(sea) is False
    assert sea_settings.parsed_sea(sea).literals.get("hidden") is None
    _register(folder)
    assert "shown" in sea_commands.list_commands()
    assert sea_lint.lint_all([sea]) == []


_STOPPABLE_MODEL = '''\
from pathlib import Path

from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "a model-picker entry whose registration is stopped once"

    def register_as_model(self):
        if not Path({marker!r}).exists():
            raise KeyboardInterrupt  # the task's Stop, landing inside the method
        return True
'''


def test_stop_in_register_as_model_is_not_memoised_as_a_verdict(tmp_path: Path) -> None:
    """A Stop propagates as itself and leaves no ``False`` under the file's stamp."""
    marker = tmp_path / "registered"
    folder = tmp_path / "seas"
    sea = _write_sea(folder, "picker", _STOPPABLE_MODEL.format(marker=str(marker)))
    _register(folder)
    stop = threading.Event()
    stop.set()
    set_thread_stop_event(stop)
    with pytest.raises(KeyboardInterrupt):
        sea_commands.model_seas()
    set_thread_stop_event(None)
    # The same revision, asked again once the method can answer: the
    # stopped call must not have been recorded as "not a model".
    marker.touch()
    assert sea_commands.model_seas().get("picker") == sea


def test_interrupt_without_a_stop_is_the_methods_own_failure(tmp_path: Path) -> None:
    """A ``KeyboardInterrupt`` while no Stop is pending is a broken SEA: logged, not a model."""
    folder = tmp_path / "seas"
    _write_sea(folder, "picker", _STOPPABLE_MODEL.format(marker=str(tmp_path / "never")))
    _register(folder)
    assert "picker" not in sea_commands.model_seas()


_TOOL_INTERRUPTED = '''\
from kiss.agents.seas.base.base_sea import BaseSea
from kiss.core.tool_interrupt import ToolCallInterrupted


class Sea(BaseSea):
    def description(self):
        return "its tools hook is interrupted"

    def tools(self, tools):
        raise ToolCallInterrupted("stop")
'''


def test_per_tool_stop_in_a_hook_propagates_unchanged(tmp_path: Path) -> None:
    """``ToolCallInterrupted`` from a hook reaches the caller as itself, not as ``SeaError``."""
    sea = _write_sea(tmp_path / "seas", "cut", _TOOL_INTERRUPTED)
    run = sea_commands.evaluate_sea(sea_commands.sea_layers(sea), "x")
    with pytest.raises(ToolCallInterrupted):
        run.tools_hook([])


_PLAIN = '''\
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "plain"
'''


def test_lint_of_a_folder_checks_its_conventional_script(tmp_path: Path) -> None:
    """``<folder>/<folder>_sea.py`` wins over an alphabetically earlier ``*_sea.py``."""
    folder = tmp_path / "zed"
    _write_sea(tmp_path, "zed", _PLAIN)
    (folder / "aaa_sea.py").write_text("def (\n", encoding="utf-8")
    assert sea_lint.lint_all([folder]) == []
    # Without the conventional script, the first ``*_sea.py`` is checked.
    other = tmp_path / "other"
    other.mkdir()
    (other / "aaa_sea.py").write_text("def (\n", encoding="utf-8")
    findings = sea_lint.lint_all([other])
    assert [f.path for f in findings] == [other / "aaa_sea.py"]
    # A folder with no script at all is reported under its own path.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert [f.path for f in sea_lint.lint_all([empty])] == [empty]


def test_package_import_loads_no_agent() -> None:
    """``import kiss.agents.sorcar`` costs nothing: the agent modules stay unloaded."""
    code = (
        "import sys, kiss.agents.sorcar\n"
        "loaded = [m for m in sys.modules if m.startswith('kiss.agents.sorcar.')]\n"
        "assert loaded == [], loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
