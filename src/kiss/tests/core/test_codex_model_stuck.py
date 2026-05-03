"""Integration test: codex/gpt-5.5 must not get stuck in an infinite retry loop.

Root cause: when a ``codex/*`` model fails on the first model call (e.g.
codex CLI not found, model not supported), ``RelentlessAgent.perform_task``
entered an infinite-like retry loop because:

1.  ``KISSAgent.step_count`` is 1 (incremented before the model call), not 0.
2.  The early-exit condition ``executor.step_count == 0`` never fires.
3.  The summarizer also uses the broken model and fails.
4.  ``is_continue: True`` is returned, causing the next sub-session.
5.  Budget stays at 0 (no tokens consumed), so budget checks never fire.
6.  The loop repeats for ``max_sub_sessions`` (default 10000) iterations.

Fix: change ``step_count == 0`` to ``step_count <= 1`` so that first-step
failures are treated as non-recoverable.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml

from kiss.core.models import codex_model as codex_module


class TestCodexModelStuckBug:
    """Codex model failure must not spin forever in the RelentlessAgent loop."""

    def test_codex_model_returns_quickly_when_cli_missing(self) -> None:
        """When the codex CLI is not installed, the agent must fail promptly
        instead of retrying 10000 times in the RelentlessAgent loop."""
        from kiss.core.relentless_agent import RelentlessAgent

        saved_path = os.environ.get("PATH", "")
        saved_candidates = codex_module._UI_CANDIDATE_PATHS
        try:
            # Ensure codex CLI is not found
            os.environ["PATH"] = ""
            codex_module._UI_CANDIDATE_PATHS = ()

            agent = RelentlessAgent("codex-stuck-test")
            agent._reset(
                model_name="codex/gpt-5.5",
                max_sub_sessions=100,
                max_steps=5,
                max_budget=10.0,
                work_dir=tempfile.mkdtemp(),
                docker_image=None,
            )
            agent.system_prompt = "You are a helpful assistant."
            agent.task_description = "Say hello"

            def noop() -> str:
                """A no-op tool."""
                return "ok"

            result = agent.perform_task([noop])
            parsed = yaml.safe_load(result)
            assert isinstance(parsed, dict)
            assert parsed["success"] is False
            # The key assertion: is_continue must be False (not True),
            # meaning the agent stopped rather than spinning forever.
            assert parsed.get("is_continue", False) is False, (
                "Agent should NOT set is_continue=True when the model fails "
                "on the very first call — that causes infinite retries."
            )
        finally:
            os.environ["PATH"] = saved_path
            codex_module._UI_CANDIDATE_PATHS = saved_candidates

    def test_codex_model_does_not_loop_in_vscode_server(self) -> None:
        """End-to-end: submitting a task with codex/gpt-5.5 when the CLI is
        missing must broadcast a result event promptly, not hang."""
        import kiss.agents.sorcar.persistence as th
        from kiss.agents.vscode.server import VSCodeServer

        saved_path = os.environ.get("PATH", "")
        saved_candidates = codex_module._UI_CANDIDATE_PATHS
        old_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        tmpdir = tempfile.mkdtemp()
        try:
            os.environ["PATH"] = ""
            codex_module._UI_CANDIDATE_PATHS = ()
            # Force codex/gpt-5.5 into the available set by faking the CLI
            fake_codex = Path(tmpdir) / "bin" / "codex"
            fake_codex.parent.mkdir(parents=True, exist_ok=True)
            fake_codex.write_text("#!/bin/sh\nexit 1\n")
            fake_codex.chmod(0o755)
            os.environ["PATH"] = str(fake_codex.parent)

            kiss_dir = Path(tmpdir) / ".kiss"
            kiss_dir.mkdir(parents=True, exist_ok=True)
            th._KISS_DIR = kiss_dir
            th._DB_PATH = kiss_dir / "sorcar.db"
            th._db_conn = None

            os.environ.setdefault("KISS_WORKDIR", tmpdir)
            server = VSCodeServer()

            events: list[dict[str, Any]] = []
            lock = threading.Lock()
            orig_broadcast = server.printer.broadcast

            def capture(e: dict[str, Any]) -> None:
                with lock:
                    events.append(dict(e))
                orig_broadcast(e)

            server.printer.broadcast = capture  # type: ignore[assignment]

            tab_id = "codex-stuck-test"
            server._handle_command({
                "type": "run",
                "prompt": "say hello",
                "model": "codex/gpt-5.5",
                "workDir": tmpdir,
                "tabId": tab_id,
            })

            tab = server._get_tab(tab_id)
            t = tab.task_thread
            assert t is not None
            # Must complete within 30 seconds — before the fix this would
            # spin for thousands of iterations and effectively hang.
            t.join(timeout=30)
            assert not t.is_alive(), (
                "Task thread is still alive after 30s — the agent is stuck "
                "in the RelentlessAgent retry loop."
            )

            with lock:
                status_false = [
                    e for e in events
                    if e.get("type") == "status" and e.get("running") is False
                ]
            assert len(status_false) >= 1, (
                "Expected status:running:False event. Events: "
                + str([e.get("type") for e in events])
            )
        finally:
            os.environ["PATH"] = saved_path
            codex_module._UI_CANDIDATE_PATHS = saved_candidates
            if th._db_conn is not None:
                th._db_conn.close()
                th._db_conn = None
            (th._DB_PATH, th._db_conn, th._KISS_DIR) = old_db
            shutil.rmtree(tmpdir, ignore_errors=True)
