# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: malformed attachment kills the task thread (BUG-D).

``_run_task_inner`` decoded every attachment with
``base64.b64decode(att.get("data", ""))`` BEFORE its big try block.
``_run_task`` wraps the inner call in try/finally with **no except**,
so a malformed attachment (invalid base64, or a non-dict list entry)
raised ``binascii.Error`` / ``AttributeError`` straight out of the
task thread: the user saw the spinner stop (the finally's
``status running: false``) with no result or error event and nothing
persisted.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.server.server as _server_module
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.server import VSCodeServer


class TestBadAttachment(unittest.TestCase):
    """A malformed attachment must not raise out of ``_run_task``."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt3-att-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        self._orig_followup = _server_module.generate_followup_text

        def fake_followup(task: str, result: str, model: str) -> str:
            return ""

        _server_module.generate_followup_text = fake_followup  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run
        self.run_calls: list[dict[str, Any]] = []
        calls = self.run_calls

        def stub_run(self_agent: object, **kwargs: object) -> str:
            calls.append(dict(kwargs))
            return "success: true\nsummary: ok\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        _server_module.generate_followup_text = self._orig_followup
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_malformed_attachments_do_not_kill_task_thread(self) -> None:
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()
        # Synchronous call on this thread: pre-fix the binascii.Error /
        # AttributeError escapes _run_task and fails this test exactly
        # the way it kills the real daemon's task thread.
        self.server._run_task({
            "type": "run",
            "prompt": "bughunt3 bad attachment task",
            "tabId": "att-tab",
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
            "attachments": [
                {"data": "%%%not-b64%%%", "mimeType": "image/png"},
                "not-a-dict",
                {"data": "aGk=", "mimeType": "text/plain"},
            ],
        })
        # The lifecycle guarantee must hold: a final running=False status.
        finals = [
            e for e in self.events
            if e.get("type") == "status" and e.get("running") is False
        ]
        assert finals, "missing final status running=False broadcast"
        # When the agent actually ran (models available in the test
        # env), the one valid attachment must have been decoded and
        # the two malformed ones skipped.
        if self.run_calls:
            atts = self.run_calls[0].get("attachments")
            assert atts is not None
            atts_list = cast("list[Any]", atts)
            assert len(atts_list) == 1, (
                f"expected only the valid attachment, got {atts_list!r}"
            )
            assert atts_list[0].data == b"hi"

    def test_threading_excepthook_not_triggered(self) -> None:
        work_dir = str(Path(self.tmpdir) / "plain2")
        Path(work_dir).mkdir()
        hook_errors: list[object] = []
        orig_hook = threading.excepthook
        threading.excepthook = lambda args: hook_errors.append(args)  # type: ignore[assignment]
        try:
            thread = threading.Thread(
                target=self.server._run_task,
                args=({
                    "type": "run",
                    "prompt": "bughunt3 bad attachment threaded",
                    "tabId": "att-tab-2",
                    "workDir": work_dir,
                    "useWorktree": False,
                    "autoCommit": False,
                    "model": "",
                    "attachments": [{"data": "%%%not-b64%%%", "mimeType": "x"}],
                },),
                daemon=True,
            )
            thread.start()
            thread.join(timeout=60)
        finally:
            threading.excepthook = orig_hook
        assert not hook_errors, (
            f"BUG: task thread died with unhandled exception: {hook_errors!r}"
        )


if __name__ == "__main__":
    unittest.main()
