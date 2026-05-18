"""Integration test: PWD reported by the agent must be consistent across modes.

Repro for the bug where, with worktree mode enabled, the agent answers
``PWD`` with the internal ``.kiss-worktrees/<slug>`` directory path
instead of the user-facing repo root.  Without worktree mode the agent
correctly reports the repo root.
"""

from __future__ import annotations

import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path

from kiss.core.relentless_agent import (
    IMPORTANT_INSTRUCTIONS,
    RelentlessAgent,
    _user_visible_work_dir,
)


def _make_finish_response() -> dict:
    """Build a fake OpenAI tool-call response invoking ``finish``."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps(
                                    {
                                        "success": True,
                                        "is_continue": False,
                                        "summary": "done",
                                    }
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
        },
    }


class _CapturingServer:
    """Fake OpenAI-compatible server that captures the request system prompt."""

    def __init__(self) -> None:
        self.captured_system_prompt: str = ""
        captured = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    req = json.loads(body)
                except Exception:
                    req = {}
                for msg in req.get("messages", []):
                    if msg.get("role") == "system":
                        captured.captured_system_prompt = str(msg.get("content", ""))
                        break
                resp_body = json.dumps(_make_finish_response()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()


class TestUserVisibleWorkDir(unittest.TestCase):
    """Unit tests for the path-stripping helper."""

    def test_non_worktree_path_unchanged(self) -> None:
        """Paths outside .kiss-worktrees are returned unchanged."""
        self.assertEqual(_user_visible_work_dir("/Users/x/repo"), "/Users/x/repo")

    def test_worktree_root_stripped(self) -> None:
        """``<repo>/.kiss-worktrees/<slug>`` becomes ``<repo>``."""
        wt = "/Users/x/repo/.kiss-worktrees/kiss_wt-abc-123"
        self.assertEqual(_user_visible_work_dir(wt), "/Users/x/repo")

    def test_worktree_subdir_stripped(self) -> None:
        """``<repo>/.kiss-worktrees/<slug>/<sub>`` becomes ``<repo>/<sub>``."""
        wt = "/Users/x/repo/.kiss-worktrees/kiss_wt-abc-123/src/pkg"
        self.assertEqual(_user_visible_work_dir(wt), "/Users/x/repo/src/pkg")

    def test_dangling_worktrees_marker(self) -> None:
        """A path ending in ``.kiss-worktrees`` with no slug is returned unchanged."""
        self.assertEqual(
            _user_visible_work_dir("/Users/x/repo/.kiss-worktrees"),
            "/Users/x/repo/.kiss-worktrees",
        )


class TestImportantInstructionsRendering(unittest.TestCase):
    """Direct rendering of IMPORTANT_INSTRUCTIONS via the helper."""

    def test_worktree_dir_not_leaked(self) -> None:
        """The worktree segment must not appear in IMPORTANT_INSTRUCTIONS."""
        wt = "/Users/x/repo/.kiss-worktrees/kiss_wt-abc-123"
        rendered = IMPORTANT_INSTRUCTIONS.format(
            step_threshold="98",
            work_dir=_user_visible_work_dir(wt),
            current_pid="1",
        )
        self.assertIn("/Users/x/repo", rendered)
        self.assertNotIn(".kiss-worktrees", rendered)
        self.assertNotIn("kiss_wt-abc-123", rendered)


class TestPwdConsistencyEndToEnd(unittest.TestCase):
    """End-to-end: drive ``RelentlessAgent.run()`` and inspect the system prompt."""

    def _run_with_work_dir(self, work_dir: str) -> str:
        server = _CapturingServer()
        try:
            agent = RelentlessAgent("PwdConsistency")
            agent.run(
                model_name="test-model",
                prompt_template="What is PWD?",
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=2,
                work_dir=work_dir,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{server.port}/v1",
                    "api_key": "sk-test",
                },
            )
        finally:
            server.stop()
        return server.captured_system_prompt

    def test_pwd_without_worktree_mode_reports_repo_root(self) -> None:
        """Baseline: non-worktree run reports the repo root as Work dir."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "myrepo"
            repo.mkdir()
            sp = self._run_with_work_dir(str(repo))
        self.assertIn(str(repo.resolve()), sp)
        self.assertNotIn(".kiss-worktrees", sp)

    def test_pwd_with_worktree_mode_reports_repo_root(self) -> None:
        """Bug repro / fix: worktree-mode run must NOT leak the worktree path.

        When the work_dir is inside ``<repo>/.kiss-worktrees/<slug>/``, the
        agent's system prompt must report the repo root (``<repo>``) as
        the Work dir — matching the non-worktree behavior — so that asking
        ``what is PWD?`` gives the same answer in both modes.
        """
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "myrepo"
            wt_dir = repo / ".kiss-worktrees" / "kiss_wt-abc123-1234567890"
            wt_dir.mkdir(parents=True)
            sp = self._run_with_work_dir(str(wt_dir))
        # User-visible repo root must appear in the prompt.
        self.assertIn(str(repo.resolve()), sp)
        # Internal worktree segment must NOT appear.
        self.assertNotIn(".kiss-worktrees", sp)
        self.assertNotIn("kiss_wt-abc123-1234567890", sp)


if __name__ == "__main__":
    unittest.main()
