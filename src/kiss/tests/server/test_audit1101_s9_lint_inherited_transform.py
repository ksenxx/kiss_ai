# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Lint must accept a ``settings`` transform that reads what its base class lays.

A run folds the ``settings`` methods base first, so a SEA's own method
receives its base's dict and may read it (``not
settings["use_web_tools"]``).  ``check_sea`` accepts such a script.
The linter then probed the same method over ``{}`` for its
``redundant-key`` comparison, inside the handler that reports a
failed load, so the ``KeyError`` of that synthetic probe was reported
as a fatal ``broken`` finding on a script the daemon runs fine.  The
probe is now separate: when it cannot evaluate the method, the
redundancy comparison is skipped and the runtime-backed checks go on.

Real script files, the real loader; nothing is mocked.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kiss.agents.sorcar.sea_commands import check_sea
from kiss.agents.sorcar.sea_lint import lint_all, lint_sea

TRANSFORM = """
from kiss.agents.seas.base.base_sea import WorkerSea

class TransformSea(WorkerSea):
    def description(self):
        return "Enable the browser relative to the worker default."

    def settings(self, settings):
        return settings | {"use_web_tools": not settings["use_web_tools"]}
"""


def _write(folder: Path, name: str, body: str) -> Path:
    path = folder / name / f"{name}_sea.py"
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_a_transform_reading_inherited_settings_is_not_broken(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """What the daemon loads cleanly lints cleanly and quietly; the other checks still run."""
    path = _write(tmp_path, "transform", TRANSFORM)
    assert check_sea(path).settings["use_web_tools"] is True
    with caplog.at_level(logging.WARNING, logger="kiss.sea_commands"):
        assert lint_sea(path) == []
    # The probe over ``{}`` is the linter's, not a SEA failure: no warning traceback.
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []
    # The runtime-backed checks after the probe still run: a model no
    # catalogue knows is reported on the same script.
    model = _write(tmp_path, "unknownmodel", TRANSFORM.replace(
        '"use_web_tools": not settings["use_web_tools"]',
        '"use_web_tools": not settings["use_web_tools"], "model": "no-such-model-xyz"',
    ))
    assert [f.code for f in lint_all([path, model])] == ["unknown-model"]
