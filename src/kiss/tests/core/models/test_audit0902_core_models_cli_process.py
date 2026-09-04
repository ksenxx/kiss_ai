# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for two CLI-supervisor defects found in the 2026-09-02 audit.

Every test drives a REAL stand-in ``claude`` / ``codex`` executable (a
Python script installed on ``PATH``) through the adapter's own locator,
real pipes and real threads — nothing is mocked.

* **Race / hang** — ``_CLIProcess`` guarded the stdout/stderr reader
  threads against a grandchild that inherits a pipe, but not the stdin
  *writer* thread.  A child that spawns a grandchild holding the prompt
  pipe open and then exits without reading the prompt left the writer
  parked inside ``write(2)`` (the prompt outgrows the 64 KiB pipe
  buffer) holding the ``BufferedWriter`` lock: ``send_prompt`` waited
  for the whole turn deadline although the child's reply was already
  complete, and ``close()`` then blocked in ``proc.stdin.close()`` for
  as long as the grandchild lived.  The fix round (review #4) made the
  writer cancellable, so after ``generate()`` returns no ``cli-stdin-write``
  thread is alive and the prompt pipe's descriptor is closed, however
  long the grandchild lives; the tests assert exactly that and are
  synchronised on the child's own events (the grandchild pid it records
  before replying) rather than on narrow wall-clock margins.
* **Inconsistency** — ``ClaudeCodeModel.generate`` accepted exit status
  ``-15`` (SIGTERM) as success while ``CodexModel`` did not, so a Claude
  CLI killed mid-turn returned its truncated text as a finished answer.
"""

import contextlib
import os
import signal
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models import claude_code_model as cc_module
from kiss.core.models import codex_model as cx_module
from kiss.core.models.claude_code_model import ClaudeCodeModel, _find_claude_cli
from kiss.core.models.codex_model import CodexModel, _find_codex_cli
from kiss.core.models.model import _CLIProcess

# Well past the 64 KiB pipe buffer, so an unread prompt blocks the writer.
_BIG_PROMPT = "x" * (300 * 1024)
# How long the grandchild keeps the prompt pipe open.  The turn deadline
# below is shorter, so a supervisor that waits on the writer times the
# turn out instead of returning the child's finished reply; both are far
# longer than a healthy turn, so the elapsed-time bound below has a wide
# margin on a loaded host.  The grandchild is killed by the test.
_GRANDCHILD_LIFETIME = 120
_TURN_TIMEOUT = 30
_HEALTHY_TURN_SECONDS = 10
# Where the stand-in records its grandchild's pid before replying.
_GRANDCHILD_PID_FILE = "GRANDCHILD_PID_FILE"


def _install_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, body: str
) -> None:
    """Install an executable stand-in CLI and let the adapter discover it."""
    script = tmp_path / name
    script.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(body).replace(
            _GRANDCHILD_PID_FILE, repr(str(tmp_path / "grandchild.pid"))
        )
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    if name == "claude":
        monkeypatch.setattr(cc_module, "_find_claude_cli", _find_claude_cli)
    else:
        monkeypatch.setattr(cx_module, "_find_codex_cli", _find_codex_cli)


_SPAWN_GRANDCHILD = f"""
    import subprocess
    import sys

    # The grandchild inherits OUR stdin -- the adapter's prompt pipe -- and
    # outlives us, but not stdout/stderr, so the adapter sees EOF on both.
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep({_GRANDCHILD_LIFETIME})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with open({_GRANDCHILD_PID_FILE}, "w") as f:
        f.write(str(grandchild.pid))
"""


def _grandchild_pid(tmp_path: Path) -> int:
    """The pid the stand-in recorded before replying."""
    pid_file = tmp_path / "grandchild.pid"
    assert pid_file.exists(), "the stand-in never recorded its grandchild"
    return int(pid_file.read_text())


def _kill_grandchild(tmp_path: Path) -> None:
    """Kill the recorded grandchild, if it is still around."""
    if (tmp_path / "grandchild.pid").exists():
        with contextlib.suppress(ProcessLookupError):
            os.kill(_grandchild_pid(tmp_path), signal.SIGKILL)


def _writer_threads() -> list[threading.Thread]:
    """Every live ``cli-stdin-write`` thread in this process."""
    return [t for t in threading.enumerate() if t.name == "cli-stdin-write"]


def _open_fds() -> int:
    """Number of descriptors this process holds (Linux ``/proc``; else 0)."""
    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else 0

_CLAUDE_GRANDCHILD_HOLDS_STDIN = _SPAWN_GRANDCHILD + """
    import json

    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1",
                                  "content": [{"type": "text", "text": "DONE"}]}}),
          flush=True)
    print(json.dumps({"type": "result", "result": "DONE",
                      "usage": {"input_tokens": 10, "output_tokens": 5}}),
          flush=True)
"""

_CODEX_GRANDCHILD_HOLDS_STDIN = _SPAWN_GRANDCHILD + """
    import json

    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "DONE"}}),
          flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 10, "cached_input_tokens": 0,
                                "output_tokens": 5}}), flush=True)
"""

_CLAUDE_SIGTERMS_ITSELF = """
    import json
    import os
    import signal
    import sys

    sys.stdin.read()
    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1",
                                  "content": [{"type": "text", "text": "partial"}]}}),
          flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
"""

_CODEX_SIGTERMS_ITSELF = """
    import json
    import os
    import signal
    import sys

    sys.stdin.read()
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "partial"}}),
          flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
"""


class TestWriterThreadDoesNotHoldTheTurnHostage:
    """A finished reply is returned even when nobody ever reads the prompt,
    and the supervisor leaves neither a writer thread nor a descriptor
    behind while the grandchild keeps the pipe open."""

    @staticmethod
    def _assert_turn_completes_and_cleans_up(
        tmp_path: Path, model: ClaudeCodeModel | CodexModel
    ) -> None:
        """Run one turn; assert the reply, the bound, and the post-conditions."""
        assert not _writer_threads()
        fds_before = _open_fds()

        started = time.monotonic()
        content, _response = model.generate()
        elapsed = time.monotonic() - started

        assert content == "DONE"
        assert elapsed < _HEALTHY_TURN_SECONDS, (
            f"generate() took {elapsed:.1f}s: the supervisor waited on the "
            "stdin writer instead of returning the child's finished reply"
        )
        # Explicit child event: the grandchild was recorded before the reply
        # and is still alive, so only cancellation can have freed the writer.
        os.kill(_grandchild_pid(tmp_path), 0)
        assert not _writer_threads(), (
            "generate() returned with the stdin writer still parked in write(2)"
        )
        assert _open_fds() == fds_before, "the turn leaked a descriptor"

    def test_claude_reply_is_returned_while_a_grandchild_holds_stdin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_GRANDCHILD_HOLDS_STDIN)
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": _TURN_TIMEOUT})
        model.initialize(_BIG_PROMPT)
        try:
            self._assert_turn_completes_and_cleans_up(tmp_path, model)
        finally:
            _kill_grandchild(tmp_path)

    def test_codex_reply_is_returned_while_a_grandchild_holds_stdin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_cli(tmp_path, monkeypatch, "codex", _CODEX_GRANDCHILD_HOLDS_STDIN)
        model = CodexModel("codex/default", model_config={"timeout": _TURN_TIMEOUT})
        model.initialize(_BIG_PROMPT)
        try:
            self._assert_turn_completes_and_cleans_up(tmp_path, model)
        finally:
            _kill_grandchild(tmp_path)


class TestCloseBeforeAnyPromptWasSent:
    """A supervisor torn down before ``send_prompt`` still releases every pipe."""

    def test_close_without_a_writer_closes_all_three_pipes(self) -> None:
        proc = _CLIProcess([sys.executable, "-c", "pass"], "probe", 5.0)
        assert proc.wait_for_exit() == 0
        proc.close()
        assert proc._proc.stdin is not None and proc._proc.stdin.closed
        assert proc._proc.stdout is not None and proc._proc.stdout.closed
        assert proc._proc.stderr is not None and proc._proc.stderr.closed


class TestSigtermIsAFailureOnBothAdapters:
    """A CLI killed by SIGTERM mid-turn must not pass its partial text off as done."""

    def test_claude_killed_by_sigterm_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_SIGTERMS_ITSELF)
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": 20})
        model.initialize("hello")

        with pytest.raises(KISSError, match=r"exit -15"):
            model.generate()
        assert all(m["role"] != "assistant" for m in model.conversation)

    def test_codex_killed_by_sigterm_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_cli(tmp_path, monkeypatch, "codex", _CODEX_SIGTERMS_ITSELF)
        model = CodexModel("codex/default", model_config={"timeout": 20})
        model.initialize("hello")

        with pytest.raises(KISSError, match=r"exit -15"):
            model.generate()
        assert all(m["role"] != "assistant" for m in model.conversation)
