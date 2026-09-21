# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: codex/gpt-5.5 generate() must enforce timeout on stream reading.

Root cause: in ``CodexModel.generate()``, ``_parse_stream_events(proc.stdout)``
blocks indefinitely reading stdout from the codex CLI subprocess.  The
``proc.wait(timeout=timeout)`` comes AFTER the blocking stream read, so
the timeout is never enforced while the codex agent is thinking/working.

For a simple task like "hi", the codex agent (gpt-5.5) may decide to do
extensive autonomous work (many command executions, reasoning), causing
the stream to block for a very long time — effectively hanging forever
from the user's perspective.

Fix: enforce the timeout on the entire subprocess execution (including
stream reading), not just ``proc.wait()``.  Use a background thread for
stream reading with ``thread.join(timeout=...)``, and kill the process
if the timeout elapses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.core.models.codex_model import CodexModel
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

# The stand-ins are Python, not bash, so they also start on Windows
# (``install_cli`` wraps them in a ``codex.cmd`` shim there).

# Simulate a codex agent that starts a turn and then goes silent while
# keeping stdout open.  The timeout bounds *silence*: every stdout line
# pushes the deadline out (``_CLIProcess.lines``), so a stand-in that
# kept emitting events would legitimately never time out.
_FAKE_CODEX_HANGS = """
    import json
    import sys
    import time

    sys.stdin.read()
    print(json.dumps({"type": "thread.started", "thread_id": "test-thread"}), flush=True)
    print(json.dumps({"type": "turn.started"}), flush=True)
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_reasoning", "text": "..."}}),
          flush=True)
    time.sleep(60)
"""

_FAKE_CODEX_QUICK = """
    import json
    import sys

    sys.stdin.read()
    print(json.dumps({"type": "thread.started", "thread_id": "test-thread"}), flush=True)
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "Hello!"}}),
          flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 10, "cached_input_tokens": 0,
                                "output_tokens": 5}}),
          flush=True)
"""


class TestCodexStreamTimeout:
    """CodexModel.generate() must time out when the codex agent hangs."""

    @pytest.mark.slow
    def test_generate_times_out_when_codex_agent_hangs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A codex CLI that goes silent mid-turn must be killed by the timeout.

        We create a fake ``codex`` script that starts a turn and then
        stalls with stdout open (a codex agent stuck thinking or waiting
        on a command).  ``generate()`` must raise ``TimeoutError`` within
        a few seconds, not hang indefinitely.
        A stall is the retryable class — ``KISSAgent._run_agentic_loop``
        re-raises every ``KISSError`` but retries anything else — so the
        agent re-asks the model instead of aborting the whole task.
        """
        install_cli(tmp_path, monkeypatch, "codex", _FAKE_CODEX_HANGS)
        m = CodexModel("codex/gpt-5.5", model_config={"timeout": 3})
        m.initialize("hi")

        with pytest.raises(TimeoutError, match="timed out"):
            m.generate()

    def test_generate_succeeds_within_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A codex CLI that finishes quickly must still work normally."""
        install_cli(tmp_path, monkeypatch, "codex", _FAKE_CODEX_QUICK)
        m = CodexModel("codex/gpt-5.5", model_config={"timeout": 10})
        m.initialize("hi")

        content, response = m.generate()
        assert content == "Hello!"
        assert response["thread_id"] == "test-thread"
