# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt regression: ClaudeCodeModel must enforce its timeout on the stream read.

``ClaudeCodeModel.generate`` historically applied the configured
``timeout`` only to ``proc.wait()`` *after* the stream-parsing loop had
already finished.  The blocking read of the CLI's stdout itself had no
deadline, so a hung or stalled ``claude`` CLI froze the agent forever
regardless of the configured timeout.  ``CodexModel`` runs the stream
parsing in a reader thread joined with the timeout; this test pins the
same behavior for ``ClaudeCodeModel``.

The test uses a real subprocess: a fake ``claude`` executable on PATH
that emits one valid stream-json event and then stalls far longer than
the configured timeout.
"""

import time
from pathlib import Path

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel

_FAKE_CLAUDE = """#!/bin/bash
# Ignore the prompt on stdin; emit one event, then stall well past the
# configured timeout while keeping stdout open.
/bin/cat > /dev/null
echo '{"type":"assistant","message":{"id":"m1","content":[{"type":"text","text":"partial"}]}}'
exec /bin/sleep 30
"""


def _install_fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = tmp_path / "claude"
    cli.write_text(_FAKE_CLAUDE)
    cli.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))


def test_generate_times_out_on_stalled_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled claude CLI must abort with KISSError within the timeout."""
    _install_fake_claude(tmp_path, monkeypatch)
    model = ClaudeCodeModel("cc/opus", model_config={"timeout": 2})
    model.initialize("hello")
    start = time.monotonic()
    with pytest.raises(KISSError, match="timed out"):
        model.generate()
    elapsed = time.monotonic() - start
    # Old behavior blocked for the fake CLI's full 30s sleep; the fix
    # must abort right after the 2s timeout (5s reader-join grace max).
    assert elapsed < 15, f"generate() blocked for {elapsed:.1f}s despite timeout=2"


def test_generate_with_tools_times_out_on_stalled_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agentic path (stop_on_tool_calls) must honor the timeout too."""
    _install_fake_claude(tmp_path, monkeypatch)
    model = ClaudeCodeModel("cc/opus", model_config={"timeout": 2})
    model.initialize("hello")

    def finish(result: str) -> str:
        """Finish the task.

        Args:
            result: The final result.
        """
        return result

    start = time.monotonic()
    with pytest.raises(KISSError, match="timed out"):
        model.generate_and_process_with_tools({"finish": finish})
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"tool call path blocked for {elapsed:.1f}s despite timeout=2"
