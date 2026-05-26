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

import os
import textwrap
from pathlib import Path

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models import codex_model as codex_module
from kiss.core.models.codex_model import CodexModel


class TestCodexStreamTimeout:
    """CodexModel.generate() must time out when the codex agent hangs."""

    @pytest.mark.slow
    def test_generate_times_out_when_codex_agent_hangs(self, tmp_path: Path) -> None:
        """A codex CLI that emits events forever must be killed by the timeout.

        We create a fake ``codex`` script that writes JSONL events to
        stdout in an infinite loop (simulating a codex agent that keeps
        thinking/running commands forever).  ``generate()`` must raise
        ``KISSError`` within a few seconds, not hang indefinitely.
        """
        # Create a fake codex script that emits events forever
        fake_codex = tmp_path / "codex"
        fake_codex.write_text(textwrap.dedent("""\
            #!/bin/bash
            # Simulate a codex agent that thinks forever
            cat /dev/stdin > /dev/null  # consume stdin
            echo '{"type":"thread.started","thread_id":"test-thread"}'
            echo '{"type":"turn.started"}'
            while true; do
                echo '{"type":"item.completed","item":{"type":"agent_reasoning","text":"..."}}'
                sleep 0.1
            done
        """))
        fake_codex.chmod(0o755)

        saved_path = os.environ.get("PATH", "")
        saved_candidates = codex_module._UI_CANDIDATE_PATHS
        try:
            os.environ["PATH"] = str(tmp_path) + ":" + saved_path
            codex_module._UI_CANDIDATE_PATHS = ()

            m = CodexModel("codex/gpt-5.5", model_config={"timeout": 3})
            m.initialize("hi")

            with pytest.raises(KISSError, match="timed out"):
                m.generate()
        finally:
            os.environ["PATH"] = saved_path
            codex_module._UI_CANDIDATE_PATHS = saved_candidates

    def test_generate_succeeds_within_timeout(self, tmp_path: Path) -> None:
        """A codex CLI that finishes quickly must still work normally."""
        fake_codex = tmp_path / "codex"
        fake_codex.write_text(textwrap.dedent("""\
            #!/bin/bash
            cat /dev/stdin > /dev/null
            echo '{"type":"thread.started","thread_id":"test-thread"}'
            echo '{"type":"item.completed","item":{"type":"agent_message","text":"Hello!"}}'
            USAGE='{"input_tokens":10,"cached_input_tokens":0,"output_tokens":5}'
            echo "{\"type\":\"turn.completed\",\"usage\":$USAGE}"
        """))
        fake_codex.chmod(0o755)

        saved_path = os.environ.get("PATH", "")
        saved_candidates = codex_module._UI_CANDIDATE_PATHS
        try:
            os.environ["PATH"] = str(tmp_path) + ":" + saved_path
            codex_module._UI_CANDIDATE_PATHS = ()

            m = CodexModel("codex/gpt-5.5", model_config={"timeout": 10})
            m.initialize("hi")

            content, response = m.generate()
            assert content == "Hello!"
            assert response["thread_id"] == "test-thread"
        finally:
            os.environ["PATH"] = saved_path
            codex_module._UI_CANDIDATE_PATHS = saved_candidates
