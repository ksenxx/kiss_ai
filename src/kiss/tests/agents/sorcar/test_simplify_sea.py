# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the SEA loader and registry fixes of the simplification pass.

* :func:`kiss.agents.sorcar.sea_settings.execute_python_file` serialises
  the ``sys.modules`` swap, so a broken revision executed in one thread
  cannot unregister the module a good revision registered in another;
* :func:`kiss.agents.sorcar.sea_commands.model_sea` reads the registry
  snapshot only (a miss on a real model name no longer walks every SEA
  folder), while the slash-command path keeps its miss rescan;
* ``check_sea`` returns one :class:`~kiss.agents.sorcar.sea_commands.SeaCheck`
  that ``sea_check`` and ``lint_sea`` read instead of re-running the
  SEA's ``settings()`` and ``tools()``;
* the one parse cache behind ``declares_hidden`` / ``declares_channel``
  / ``declared_literal`` / ``declared_bases`` sees an edit, and the
  model-picker detector follows an aliased base import.

Every test isolates :mod:`kiss.agents.sorcar.sea_commands` like
``test_sea_commands.py`` does.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from kiss.agents.seas.base.base_sea import WORKER_DEFAULTS
from kiss.agents.sorcar import sea_commands, sea_lint, sea_settings
from kiss.agents.sorcar.sea_settings import SeaError, execute_python_file, wire_field
from kiss.core.config import kiss_home


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Drop the registry state before and after each test, and the test ``SEAS.md``."""
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()
    (kiss_home() / "SEAS.md").unlink(missing_ok=True)


def _write_seas_md(folder: Path) -> None:
    """Point ``$KISS_HOME/SEAS.md`` at *folder*."""
    home = kiss_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")


def _write_sea(folder: Path, name: str, source: str) -> Path:
    """Write ``<folder>/<name>/<name>_sea.py`` with *source* and return it."""
    path = folder / name / f"{name}_sea.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


_BROKEN_REVISION = '''\
"""A revision that raises at import, once the test opens the gate."""

import time
from pathlib import Path

GATE = Path({gate!r})
GATE.with_name("started").touch()  # proves THIS revision is the one executing
deadline = time.monotonic() + 10
while not GATE.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
raise RuntimeError("broken revision")
'''

_GOOD_REVISION = 'MARK = "good"\n'


def _wait_for(condition: Callable[[], bool], what: str) -> None:
    """Poll *condition* for up to ten seconds."""
    deadline = time.monotonic() + 10
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def test_concurrent_executions_of_one_file_keep_the_good_module(tmp_path: Path) -> None:
    """A broken revision failing in one thread leaves the good revision registered.

    Thread 1 executes a revision that blocks at import (so its module is
    registered under the file's name) and then raises; meanwhile the file
    is rewritten and thread 2 executes the good revision.  Without the
    loader lock, thread 2 registers and returns, then thread 1's failure
    path ("no previous entry") pops thread 2's module: a ``@dataclass``
    or ``get_type_hints`` of the good revision would later fail with
    ``KeyError`` in ``sys.modules``.  With the lock, thread 2 waits for
    thread 1 to finish, and its module stays registered.
    """
    sea = tmp_path / "flaky_sea.py"
    gate = tmp_path / "gate"
    sea.write_text(_BROKEN_REVISION.format(gate=str(gate)), encoding="utf-8")
    errors: list[SeaError] = []
    namespaces: list[dict[str, object]] = []

    def _run_broken() -> None:
        try:
            execute_python_file(str(sea))
        except SeaError as exc:
            errors.append(exc)

    def _run_good() -> None:
        namespaces.append(execute_python_file(str(sea)))

    broken = threading.Thread(target=_run_broken)
    broken.start()
    # The module is registered before its source is read, so wait for
    # the broken revision's own marker: only then is the rewrite below
    # certain to be read by the second thread, not the first.
    _wait_for(gate.with_name("started").exists, "the broken revision to start executing")
    sea.write_text(_GOOD_REVISION, encoding="utf-8")
    good = threading.Thread(target=_run_good)
    good.start()
    time.sleep(0.3)  # unlocked: the good revision has returned by now
    gate.touch()
    broken.join(timeout=10)
    good.join(timeout=10)
    assert not broken.is_alive() and not good.is_alive()
    assert len(errors) == 1 and "broken revision" in str(errors[0])
    assert len(namespaces) == 1 and namespaces[0]["MARK"] == "good"
    name = str(namespaces[0]["__name__"])
    module = sys.modules.get(name)
    assert module is not None, "the good revision's module was unregistered"
    assert module.__dict__ is namespaces[0]


def test_execute_python_file_rejects_non_paths_and_non_py_files(tmp_path: Path) -> None:
    """The loader's diagnostics name the SEA field and the offending path."""
    with pytest.raises(SeaError, match="SEA field must be a path string, got int"):
        execute_python_file(7)
    text = tmp_path / "notes.txt"
    text.write_text("x", encoding="utf-8")
    with pytest.raises(SeaError, match=r"is not an existing Python \(\.py\) file"):
        execute_python_file(str(text))
    with pytest.raises(SeaError, match="failed to import: ZeroDivisionError"):
        boom = tmp_path / "boom_sea.py"
        boom.write_text("1 / 0\n", encoding="utf-8")
        execute_python_file(str(boom))


_ROUTER = """\
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "a router"

    def register_as_model(self):
        return True
"""

_PLAIN = """\
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "plain"
"""


def test_model_sea_reads_the_snapshot_and_get_command_rescans(tmp_path: Path) -> None:
    """A ``model_sea`` miss does not walk the folders; a ``get_command`` miss does.

    The task runner asks ``model_sea`` for every run whose model is a
    real model name, so a miss there must cost a dict lookup, not a
    rescan of every SEA folder.  The slash-command path keeps the miss
    rescan: a ``/late`` typed right after the file was created works.
    """
    folder = tmp_path / "seas"
    early = _write_sea(folder, "early", _ROUTER)
    _write_seas_md(folder)
    sea_commands.refresh_registry()
    refreshes: list[list[str]] = []
    sea_commands.subscribe(refreshes.append)
    assert sea_commands.model_sea("early") == early
    late = _write_sea(folder, "late", _ROUTER)
    assert sea_commands.model_sea("late") is None
    assert sea_commands.model_sea("gpt-6-astra") is None
    assert sea_commands.model_sea("") is None
    assert refreshes == [] and "late" not in sea_commands.list_commands()
    assert sea_commands.get_command("late") == late
    assert len(refreshes) == 1 and "late" in sea_commands.list_commands()
    assert sea_commands.model_sea("late") == late


def test_model_sea_populates_a_cold_registry() -> None:
    """Right after a reset, the bundled routers are found without an explicit refresh."""
    assert sea_commands.model_sea("autorouter") is not None
    assert sea_commands.model_sea("bestrouter") is not None
    assert sea_commands.model_sea("sh") is None


_COUNTING_SEA = """\
from pathlib import Path

from kiss.agents.seas.base.base_sea import BaseSea

LOG = Path({log!r})


def _count(method):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(method + "\\n")


class Sea(BaseSea):
    def description(self):
        return "counts how often its methods run"

    def settings(self, settings):
        _count("settings")
        return settings | {{"use_worktree": False}}

    def tools(self, tools):
        _count("tools")
        return tools
"""


def _counts(log: Path) -> dict[str, int]:
    """Return how often each method of the counting SEA ran since the log was cleared."""
    lines = log.read_text(encoding="utf-8").split() if log.exists() else []
    return {name: lines.count(name) for name in ("settings", "tools")}


def test_check_report_and_lint_run_settings_and_tools_once_per_evaluation(
    tmp_path: Path,
) -> None:
    """``sea_check`` and ``lint_sea`` read the check's result instead of re-running the SEA.

    ``check_sea`` evaluates the SEA the way a run does: ``settings()``
    once when the layers load (malformed settings fail at load time)
    and once when the run command is applied, ``tools()`` once for the
    tools hook.  The report and the linter add nothing to that (the
    linter's ``own_settings`` is one more ``settings()`` of the script
    alone).
    """
    log = tmp_path / "calls.log"
    sea = _write_sea(tmp_path / "seas", "count", _COUNTING_SEA.format(log=str(log)))
    check = sea_commands.check_sea(sea)
    assert isinstance(check, sea_commands.SeaCheck)
    assert check.seas[-1].path == sea and check.cmd["prompt"] == sea_commands.CHECK_SAMPLE_TASK
    assert check.description == "counts how often its methods run"
    assert check.settings["use_worktree"] is False and check.tools == []
    assert _counts(log) == {"settings": 2, "tools": 1}

    log.unlink()
    report = sea_commands.sea_check("count", sea)
    assert "settings: " in report and "tools added: none" in report
    assert "methods defined: tools" in report
    assert _counts(log) == {"settings": 2, "tools": 1}

    log.unlink()
    findings = sea_lint.lint_sea(sea)
    assert [f.code for f in findings] == []
    assert _counts(log) == {"settings": 3, "tools": 1}


_FLIPPING_HIDDEN = """\
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"hidden": True}
"""

_FLIPPING_CHANNEL = """\
from kiss.agents.seas.base.base_sea import ChannelSea


class Sea(ChannelSea):
    def settings(self, settings):
        return settings | {"timeout": 5}
"""


def test_parse_cache_sees_an_edit_in_every_reader(tmp_path: Path) -> None:
    """``declares_hidden``, ``declares_channel`` and ``declared_bases`` follow a rewrite."""
    sea = _write_sea(tmp_path / "seas", "flip", _FLIPPING_HIDDEN)
    assert sea_settings.declares_hidden(sea) is True
    assert sea_settings.declares_channel(sea) is False
    assert sea_settings.declared_bases(sea) == {"BaseSea"}
    assert sea_settings.declared_literal(sea, "timeout") is None
    time.sleep(0.01)
    sea.write_text(_FLIPPING_CHANNEL, encoding="utf-8")
    assert sea_settings.declares_hidden(sea) is False
    assert sea_settings.declares_channel(sea) is True
    assert sea_settings.declared_bases(sea) == {"ChannelSea"}
    assert sea_settings.declared_literal(sea, "timeout") == 5
    sea.write_text("def (\n", encoding="utf-8")
    assert sea_settings.declares_hidden(sea) is False
    assert sea_settings.declared_bases(sea) == set()
    assert sea_settings.declared_literal(sea, "timeout") is None
    # A broken revision is memoised too: the watcher must not re-read
    # it on every poll (the entry carries the revision's stamp).
    broken_stat = sea.stat()
    assert sea_settings._PARSE_CACHE[sea] == (
        (broken_stat.st_mtime_ns, broken_stat.st_size, broken_stat.st_ino), set(), {},
    )
    assert sea_settings.declares_hidden(tmp_path / "missing_sea.py") is False
    assert (tmp_path / "missing_sea.py") not in sea_settings._PARSE_CACHE
    sea.write_text(
        "from kiss.agents.sorcar.sea_commands import sea_class\n"
        "from kiss.agents.seas import base as b\n\n\n"
        "class Sea(sea_class('picker'), b.BaseSea):\n    pass\n",
        encoding="utf-8",
    )
    assert sea_settings.declared_bases(sea) == {"sea_class('picker')", "BaseSea"}


_ALIASED_ROUTER = """\
from kiss.agents.seas.bestrouter.bestrouter_sea import BestrouterSea as Router


class Sea(Router):
    pass
"""

_IMPORT_MARKER = """\
from pathlib import Path

from kiss.agents.seas.base.base_sea import BaseSea

Path({marker!r}).write_text("imported", encoding="utf-8")


class Sea(BaseSea):
    def description(self):
        return "never a model"
"""


def test_model_picker_detector_follows_an_aliased_base_and_skips_plain_seas(
    tmp_path: Path,
) -> None:
    """A SEA deriving from an aliased router registers; a plain SEA is not even imported."""
    folder = tmp_path / "seas"
    aliased = _write_sea(folder, "aliased", _ALIASED_ROUTER)
    marker = tmp_path / "imported"
    _write_sea(folder, "plain", _IMPORT_MARKER.format(marker=str(marker)))
    _write_seas_md(folder)
    sea_commands.refresh_registry()
    seas = sea_commands.model_seas()
    assert seas["aliased"] == aliased and "plain" not in seas
    assert not marker.exists(), "a SEA deriving from BaseSea without register_as_model ran"


def test_precedence_example_names_the_sh_lock() -> None:
    """The docs example reads ``/sh``'s declared lock and the refusal it produces."""
    from kiss.agents.sorcar.sea_docs import precedence_example

    text = precedence_example()
    assert "`/sh` declares" in text and '"locked"' in text
    assert "tool_profile" in text


def test_wire_field_keeps_its_aliases_and_camel_case() -> None:
    """The three legacy aliases and the camelCase rule are unchanged."""
    assert wire_field("add_to_prompt") == "appendToPrompt"
    assert wire_field("add_to_system_prompt") == "appendToSystemPrompt"
    assert wire_field("auto_classify") == "classifyTasks"
    assert wire_field("use_web_tools") == "useWebTools"
    assert wire_field("model") == "model"
    assert WORKER_DEFAULTS["use_worktree"] is False


_REJECTED_SEA = """\
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def prompt(self, task):
        return 7
"""


def test_apply_sea_evaluates_then_applies_the_run(tmp_path: Path) -> None:
    """``apply_sea`` is ``evaluate_sea`` + ``apply_run``: no SEAs leave the command alone.

    A method returning the wrong type is rejected before anything is
    written; a loaded chain is written exactly as ``apply_run`` writes
    the evaluated :class:`SeaRun`.
    """
    from kiss.agents.sorcar.sea_apply import apply_run, apply_sea, load_layers

    cmd: dict[str, object] = {"prompt": "do it", "parentTaskId": "t1"}
    assert apply_sea(cmd, []) == set() and cmd == {"prompt": "do it", "parentTaskId": "t1"}
    rejected = _write_sea(tmp_path / "seas", "rejected", _REJECTED_SEA)
    cmd["seaPath"] = str(rejected)
    with pytest.raises(SeaError, match="prompt.*must return a string, got int"):
        apply_sea(cmd)
    assert "prompt" in cmd and cmd["prompt"] == "do it"
    counting = _write_sea(tmp_path / "seas", "count", _COUNTING_SEA.format(log=str(tmp_path / "l")))
    seas = load_layers({"seaPath": str(counting)})
    run = sea_commands.evaluate_sea(seas, "do it", "t1")
    direct: dict[str, object] = {"seaPath": str(counting), "prompt": "do it"}
    via_apply = dict(direct)
    assert apply_run(direct, seas, run) == apply_sea(via_apply, seas) == {"useWorktree"}
    assert direct["useWorktree"] is False and via_apply["useWorktree"] is False


def test_sea_lint_fix_without_paths_rewrites_nothing_in_a_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sea lint --fix`` with no path targets the bundled scripts (and finds nothing to fix)."""
    from kiss.agents.sorcar.sea_cli import main
    from kiss.core import config

    monkeypatch.setattr(config.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "test-key")
    assert main(["lint", "--fix"]) == 0


def test_lint_all_registered_covers_a_user_command_once(tmp_path: Path) -> None:
    """``--registered`` lints a user SEA under its command name, walking the registry once."""
    folder = tmp_path / "seas"
    sea = _write_sea(
        folder,
        "silent",
        _PLAIN.replace(
            '    def description(self):\n        return "plain"\n',
            "    pass\n",
        ),
    )
    _write_seas_md(folder)
    sea_commands.refresh_registry()
    commands = sea_lint.registered_commands()
    assert commands[sea] == "silent"
    assert sea not in sea_lint.default_targets(False, commands)
    assert sea in sea_lint.default_targets(True, commands)
    findings = [f for f in sea_lint.lint_all(None, registered=True) if f.path == sea]
    assert [f.code for f in findings] == ["no-description"]
    assert "/silent has no description()" in findings[0].message
