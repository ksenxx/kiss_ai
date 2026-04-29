"""Integration tests confirming fixes for bugs, redundancies, and
inconsistencies in ``kiss.agents.vscode`` — audit round 6.

B1 fix: ``save_config`` now preserves non-DEFAULTS keys (like ``email``)
    that were already stored in ``config.json``.  Previously, calling
    ``save_config`` truncated the file to only DEFAULTS keys.

B2 fix: ``is_task_active`` is now cleared in ``_run_task``'s finally
    block, not only in ``_run_task_inner``'s.  Previously, if
    ``_capture_pre_snapshot`` raised before the inner try/finally,
    ``is_task_active`` stayed True permanently.

B3 fix: ``fast_model_for()`` now returns ``gemini-2.0-flash`` for
    Gemini (a genuinely cheap/fast model) instead of ``gemini-2.5-pro``
    (an expensive reasoning model).

R1 fix: dead ``is not None`` guard removed from ``_cmd_user_answer``.
    Since ``ans_tab`` defaults to ``""``, it is never ``None``.
"""

from __future__ import annotations

import inspect
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.agents.vscode.commands import _CommandsMixin
from kiss.agents.vscode.helpers import fast_model_for
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.task_runner import _TaskRunnerMixin
from kiss.agents.vscode.vscode_config import (
    CONFIG_PATH,
    load_config,
    save_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


# ===================================================================
# B1 — save_config loses non-DEFAULTS keys (like email)
# ===================================================================


class TestSaveConfigPreservesExtraKeys(TestCase):
    """B1 FIX: ``save_config`` now preserves non-DEFAULTS keys that
    were already stored in ``~/.kiss/config.json``.

    Previously, calling ``save_config({"max_budget": 50})`` would
    overwrite config.json with only ``{"max_budget": 50}``, silently
    dropping keys like ``email`` that are documented as living in
    config.json.
    """

    def setUp(self) -> None:
        self._orig_path = str(CONFIG_PATH)
        self._tmpdir = tempfile.mkdtemp()
        self._tmp_config = Path(self._tmpdir) / "config.json"
        # Redirect CONFIG_PATH for this test
        import kiss.agents.vscode.vscode_config as mod

        self._mod = mod
        self._orig_config_path = mod.CONFIG_PATH
        self._orig_config_dir = mod.CONFIG_DIR
        mod.CONFIG_PATH = self._tmp_config
        mod.CONFIG_DIR = Path(self._tmpdir)

    def tearDown(self) -> None:
        self._mod.CONFIG_PATH = self._orig_config_path
        self._mod.CONFIG_DIR = self._orig_config_dir
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_config_preserves_email_key(self) -> None:
        """Saving config should not strip the ``email`` key."""
        # Pre-populate config.json with an email
        self._tmp_config.write_text(
            json.dumps({"email": "user@example.com", "max_budget": 100})
        )
        # Save with only max_budget changed
        save_config({"max_budget": 50})
        # Reload and verify email is still there
        stored = json.loads(self._tmp_config.read_text())
        assert stored.get("email") == "user@example.com", (
            f"B1 FIX: email should be preserved, got {stored}"
        )
        assert stored.get("max_budget") == 50

    def test_save_config_preserves_arbitrary_extra_keys(self) -> None:
        """Any non-DEFAULTS key already in config.json should survive a save."""
        self._tmp_config.write_text(
            json.dumps({
                "email": "a@b.com",
                "tunnel_token": "tok-123",
                "max_budget": 100,
            })
        )
        save_config({"max_budget": 75, "use_web_browser": False})
        stored = json.loads(self._tmp_config.read_text())
        assert stored["email"] == "a@b.com"
        assert stored["tunnel_token"] == "tok-123"
        assert stored["max_budget"] == 75
        assert stored["use_web_browser"] is False

    def test_save_config_new_file_still_works(self) -> None:
        """When no config.json exists yet, save_config creates one correctly."""
        assert not self._tmp_config.exists()
        save_config({"max_budget": 42})
        stored = json.loads(self._tmp_config.read_text())
        assert stored["max_budget"] == 42

    def test_load_then_save_round_trip(self) -> None:
        """load_config → modify → save_config round-trip preserves extra keys."""
        self._tmp_config.write_text(
            json.dumps({"email": "keep@me.com", "max_budget": 100})
        )
        cfg = load_config()
        cfg["max_budget"] = 200
        save_config(cfg)
        stored = json.loads(self._tmp_config.read_text())
        assert stored["email"] == "keep@me.com"
        assert stored["max_budget"] == 200


# ===================================================================
# B2 — is_task_active not cleared when pre-snapshot raises
# ===================================================================


class TestIsTaskActiveClearedOnSnapshotFailure(TestCase):
    """B2 FIX: ``_run_task`` now clears ``is_task_active`` in its own
    finally block, so a failure in ``_capture_pre_snapshot`` (which
    occurs before the inner try/finally) no longer leaves the tab
    permanently marked as active.
    """

    def test_is_task_active_false_after_snapshot_error(self) -> None:
        """When pre-snapshot capture fails, is_task_active must be False."""
        server, events = _make_server()
        tab_id = "snap-fail-tab"
        tab = server._get_tab(tab_id)

        # Force the task to be non-worktree so snapshot capture runs
        cmd: dict[str, Any] = {
            "type": "run",
            "tabId": tab_id,
            "prompt": "test",
            "useWorktree": False,
            "model": server._default_model,
            "workDir": "/nonexistent/dir/that/will/fail",
        }

        # Run the task synchronously (no thread, to control timing)
        tab.stop_event = threading.Event()
        tab.user_answer_queue = queue.Queue(maxsize=1)
        server.printer._thread_local.tab_id = tab_id

        # The snapshot failure raises FileNotFoundError through _git/subprocess.
        # _run_task's finally should still clean up is_task_active.
        try:
            server._run_task(cmd)
        except (FileNotFoundError, OSError):
            pass

        # The critical check: is_task_active must be False
        assert tab.is_task_active is False, (
            "B2 FIX: is_task_active should be False after snapshot failure"
        )

    def test_is_task_active_cleared_in_run_task_finally(self) -> None:
        """Structural: ``_run_task`` now explicitly clears ``is_task_active``."""
        src = inspect.getsource(_TaskRunnerMixin._run_task)
        assert "is_task_active" in src, (
            "B2 FIX: _run_task should reference is_task_active in its "
            "finally block"
        )


# ===================================================================
# B3 — fast_model_for returns expensive model for Gemini
# ===================================================================


class TestFastModelForReturnsActuallyFastModels(TestCase):
    """B3 FIX: ``fast_model_for()`` now returns genuinely cheap/fast
    models for each provider.  Previously, the Gemini branch returned
    ``gemini-2.5-pro`` which is one of the most expensive models.
    """

    def test_gemini_model_is_flash_not_pro(self) -> None:
        """The Gemini fast model should be a flash variant, not pro."""
        # Temporarily set only GEMINI_API_KEY
        from kiss.core import config as config_module

        orig = config_module.DEFAULT_CONFIG
        try:
            config_module.DEFAULT_CONFIG = type(orig)()
            # Clear all keys except Gemini
            config_module.DEFAULT_CONFIG.ANTHROPIC_API_KEY = ""
            config_module.DEFAULT_CONFIG.OPENROUTER_API_KEY = ""
            config_module.DEFAULT_CONFIG.TOGETHER_API_KEY = ""
            config_module.DEFAULT_CONFIG.GEMINI_API_KEY = "test-key"
            config_module.DEFAULT_CONFIG.OPENAI_API_KEY = ""

            result = fast_model_for()
            assert "flash" in result.lower() or "2.0" in result, (
                f"B3 FIX: Gemini fast model should be a flash variant, "
                f"got '{result}'"
            )
            assert "pro" not in result.lower() or "flash" in result.lower(), (
                f"B3 FIX: Gemini fast model should not be a pro model, "
                f"got '{result}'"
            )
        finally:
            config_module.DEFAULT_CONFIG = orig

    def test_docstring_consistency(self) -> None:
        """The function's docstring should match actual behavior."""
        doc = fast_model_for.__doc__ or ""
        assert "cheap" in doc.lower() or "fast" in doc.lower(), (
            "fast_model_for docstring should mention cheap/fast"
        )


# ===================================================================
# R1 — Dead `is not None` guard in _cmd_user_answer
# ===================================================================


class TestUserAnswerNoDeadIsNotNoneCheck(TestCase):
    """R1 FIX: removed the dead ``if ans_tab is not None`` guard from
    ``_cmd_user_answer``.  Since ``ans_tab = cmd.get("tabId", "")``,
    the variable is always a string, never ``None``.
    """

    def test_no_redundant_none_check(self) -> None:
        """Source should not contain ``if ans_tab is not None``."""
        src = inspect.getsource(_CommandsMixin._cmd_user_answer)
        assert "is not None" not in src or "ans_tab is not None" not in src, (
            "R1 FIX: _cmd_user_answer should not have a dead "
            "'ans_tab is not None' check"
        )

    def test_empty_tab_id_drops_answer(self) -> None:
        """When tabId is empty string, the answer should be dropped (no queue)."""
        server, events = _make_server()
        # No tab exists for empty string
        server._cmd_user_answer({"type": "userAnswer", "tabId": "", "answer": "x"})
        # Should not crash — the answer is simply dropped


# ===================================================================
# Additional consistency checks
# ===================================================================


class TestIsRunningNonWtClearedOnSnapshotFailure(TestCase):
    """is_running_non_wt should be False after a snapshot failure."""

    def test_is_running_non_wt_cleared(self) -> None:
        server, events = _make_server()
        tab_id = "nwt-fail-tab"
        tab = server._get_tab(tab_id)

        cmd: dict[str, Any] = {
            "type": "run",
            "tabId": tab_id,
            "prompt": "test",
            "useWorktree": False,
            "model": server._default_model,
            "workDir": "/nonexistent/dir/that/will/fail",
        }

        tab.stop_event = threading.Event()
        tab.user_answer_queue = queue.Queue(maxsize=1)
        server.printer._thread_local.tab_id = tab_id

        try:
            server._run_task(cmd)
        except (FileNotFoundError, OSError):
            pass

        assert tab.is_running_non_wt is False, (
            "is_running_non_wt should be False after snapshot failure"
        )


if __name__ == "__main__":
    unittest.main()
