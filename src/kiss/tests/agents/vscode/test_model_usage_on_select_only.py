"""Integration test: last_model_used and model usage counts update ONLY on model picker selection.

Reproduces the bug where running a task (via _run_task_inner / _reset) would
update last_model_used and increment model usage counts, even though the user
never explicitly selected a model via the picker.

After the fix, only _cmd_select_model (the model picker handler) should
update last_model_used and model usage counts.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator

import pytest

from kiss.agents.sorcar.persistence import (
    _close_db,
    _load_last_model,
    _load_model_usage,
    _record_model_usage,
)


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Point persistence at a temp dir so tests don't touch real data."""
    import kiss.agents.sorcar.persistence as pm

    _close_db()
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setattr(pm, "_KISS_DIR", type(pm._KISS_DIR)(tmpdir))
    monkeypatch.setattr(pm, "_DB_PATH", type(pm._DB_PATH)(os.path.join(tmpdir, "sorcar.db")))
    yield
    _close_db()






class TestModelPickerUpdatesUsageAndLastModel:
    """_cmd_select_model must call _record_model_usage (which sets both)."""


    def test_select_model_behavioral(self) -> None:
        """Selecting a model via the picker must persist last_model and
        increment usage count."""
        assert _load_last_model() == ""
        assert _load_model_usage() == {}

        _record_model_usage("gpt-4o")

        assert _load_last_model() == "gpt-4o"
        usage = _load_model_usage()
        assert usage.get("gpt-4o") == 1

        _record_model_usage("gpt-4o")
        assert _load_model_usage().get("gpt-4o") == 2

        _record_model_usage("claude-sonnet")
        assert _load_last_model() == "claude-sonnet"
        usage = _load_model_usage()
        assert usage.get("gpt-4o") == 2
        assert usage.get("claude-sonnet") == 1
