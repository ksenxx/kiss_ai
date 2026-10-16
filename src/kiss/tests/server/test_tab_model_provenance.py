"""The tab's remembered model is a persisted value of the tab, not a daemon default.

A chat tab that picked a model keeps it in the daemon's ``_tab_models``
table, and a run command of the tab that carries no ``model`` gets that
model (``_run_task``) before the daemon-wide default would apply.  The
run's provenance record must say so: ``model`` is then neither in
``defaulted`` nor, when a SEA pins it, missing from ``pinned``.

Harness: a real ``AgentServer`` in a private ``KISS_HOME`` and its
``_apply_sea`` step on a ``run`` command, exactly as ``_run_task`` calls
it; no daemon thread is started.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.sea_apply import RUN_CONFIG_FIELD
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

_TAB_MODEL = "openai/gpt-4o"
_SEA_MODEL = "anthropic/claude-sonnet-4"

_PLAIN_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "pins nothing"
"""

_MODEL_SEA = f"""
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "pins the model"

    def settings(self, settings):
        return settings | {{"model": {_SEA_MODEL!r}}}
"""


@pytest.fixture
def server() -> Iterator[AgentServer]:
    """A server in a throwaway home whose daemon default differs from the tab's model."""
    home = IsolatedKissHome("kiss-tab-model-provenance-")
    try:
        srv = AgentServer(printer=JsonPrinter())
        srv._default_model = "daemon/default-model"
        yield srv
    finally:
        home.cleanup()


def _run(tmp_path: Path, sea_body: str) -> dict[str, Any]:
    sea = tmp_path / "pin_sea.py"
    sea.write_text(sea_body, encoding="utf-8")
    return {
        "type": "run", "tabId": "t", "prompt": "hello",
        "workDir": str(tmp_path), "seaPath": str(sea),
    }


def test_remembered_tab_model_is_the_tabs_value_not_a_default(
    server: AgentServer, tmp_path: Path,
) -> None:
    """A tab that picked a model runs on it, and the record does not call it defaulted."""
    server._tab_models["t"] = _TAB_MODEL
    cmd = _run(tmp_path, _PLAIN_SEA)
    server._apply_sea(cmd)
    record = cmd[RUN_CONFIG_FIELD]
    assert cmd["model"] == _TAB_MODEL
    assert "model" not in record["defaulted"]
    assert record["pinned"] == {}


def test_sea_model_beats_the_tabs_and_the_record_shows_what_it_replaced(
    server: AgentServer, tmp_path: Path,
) -> None:
    """The SEA's model wins over the tab's remembered one; ``pinned`` names both."""
    server._tab_models["t"] = _TAB_MODEL
    cmd = _run(tmp_path, _MODEL_SEA)
    server._apply_sea(cmd)
    record = cmd[RUN_CONFIG_FIELD]
    assert cmd["model"] == _SEA_MODEL
    assert record["pinned"]["model"] == [_TAB_MODEL, _SEA_MODEL]
    assert "model" not in record["defaulted"]


def test_a_tab_without_a_remembered_model_is_defaulted(
    server: AgentServer, tmp_path: Path,
) -> None:
    """With nothing remembered the command keeps no model and the record says defaulted."""
    server._tab_models.pop("t", None)
    cmd = _run(tmp_path, _PLAIN_SEA)
    server._apply_sea(cmd)
    record = cmd[RUN_CONFIG_FIELD]
    assert not cmd.get("model")
    assert "model" in record["defaulted"]
    assert server._tab_model("t") == "daemon/default-model"
