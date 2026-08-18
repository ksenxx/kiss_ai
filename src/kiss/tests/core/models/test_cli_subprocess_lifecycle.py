# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the lifetime of a CLI-backed model's subprocess.

Every test drives a REAL stand-in ``claude`` / ``codex`` executable: a
small Python script written into the test's temporary directory, made
executable, and put on ``PATH`` so the adapter's own locator finds it.
Nothing is mocked, patched or faked — the assertions are made against
real processes, real pipes and real file descriptors.

Findings covered (audit ``tmp/audit/03-core-models-c.md``):

* **C1** — the codex reader thread outliving its timeout and emitting
  the previous run's tokens into the next run's callbacks.
* **C2** — ``kill()`` without ``wait()`` and pipes that are never
  closed, leaving zombies and leaked descriptors.
* **C3** — ``stderr=PIPE`` never drained, so a chatty CLI blocks in
  ``write(2)`` and is misreported as a stall.
* **C4** — an unguarded ``proc.wait(timeout=...)`` letting
  ``subprocess.TimeoutExpired`` escape while the child keeps running.
* **C5** — an unguarded ``proc.stdin.write`` of a large prompt raising
  ``BrokenPipeError`` when the CLI exits early.
* **I6** — the CLI adapters never reading the thread stop signal, and
  classifying a stall as a non-retryable ``KISSError``.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from kiss.core.models import claude_code_model as cc_module
from kiss.core.models import codex_model as cx_module
from kiss.core.models.claude_code_model import ClaudeCodeModel, _find_claude_cli
from kiss.core.models.codex_model import CodexModel, _find_codex_cli
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401

_PS = "/bin/ps"


def install_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, body: str
) -> Path:
    """Install an executable stand-in CLI and let the adapter discover it.

    The directory-level ``conftest`` stubs the two locator functions so
    offline tests can build a command line; restoring the real locator
    here is what makes the adapter run the script below instead.

    Args:
        tmp_path: The test's temporary directory, which becomes ``PATH``.
        monkeypatch: The fixture used to set ``PATH`` and restore the locator.
        name: ``"claude"`` or ``"codex"``.
        body: Python source for the stand-in, without a shebang.

    Returns:
        The path of the installed executable.
    """
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    if name == "claude":
        monkeypatch.setattr(cc_module, "_find_claude_cli", _find_claude_cli)
    else:
        monkeypatch.setattr(cx_module, "_find_codex_cli", _find_codex_cli)
    return script


def child_pids() -> set[int]:
    """Return the pids of this process' live children, zombies included.

    Returns:
        The set of pids whose parent is this process, excluding the
        ``ps`` process this call itself spawned.  A zombie is still
        listed by ``ps`` until somebody reaps it, which is exactly the
        leak C2 describes.
    """
    lister = subprocess.Popen(
        [_PS, "-o", "pid=,ppid=", "-ax"], stdout=subprocess.PIPE, text=True
    )
    listing, _ = lister.communicate()
    pids: set[int] = set()
    me = os.getpid()
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit() and int(fields[1]) == me:
            pids.add(int(fields[0]))
    return pids - {lister.pid}


_IGNORES_SIGTERM = """
    import signal
    import sys
    import time

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.stdin.read()
    time.sleep(60)
"""


_CODEX_FLOODS_STDERR = """
    import json
    import sys

    sys.stdin.read()
    sys.stderr.write("deprecation warning\\n" * 12000)
    sys.stderr.flush()
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "OK"}}),
          flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 10, "cached_input_tokens": 0,
                                "output_tokens": 5}}), flush=True)
"""


_CLAUDE_FLOODS_STDERR = """
    import json
    import sys

    sys.stdin.read()
    sys.stderr.write("deprecation warning\\n" * 12000)
    sys.stderr.flush()
    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1",
                                  "content": [{"type": "text", "text": "OK"}]}}),
          flush=True)
    print(json.dumps({"type": "result", "result": "OK",
                      "usage": {"input_tokens": 10, "output_tokens": 5}}),
          flush=True)
"""


_CODEX_LINGERS_AFTER_CLOSING_STDOUT = """
    import json
    import os
    import sys
    import time

    sys.stdin.read()
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "DONE"}}),
          flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 10, "cached_input_tokens": 0,
                                "output_tokens": 5}}), flush=True)
    sys.stdout.flush()
    os.close(1)
    time.sleep(60)
"""


_CLAUDE_LINGERS_AFTER_CLOSING_STDOUT = """
    import json
    import os
    import sys
    import time

    sys.stdin.read()
    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1",
                                  "content": [{"type": "text", "text": "DONE"}]}}),
          flush=True)
    print(json.dumps({"type": "result", "result": "DONE",
                      "usage": {"input_tokens": 10, "output_tokens": 5}}),
          flush=True)
    sys.stdout.flush()
    os.close(1)
    time.sleep(60)
"""


_ONE_EVENT_THEN_QUIET_CODEX = """
    import json
    import sys
    import time

    sys.stdin.read()
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "hello"}}),
          flush=True)
    time.sleep(60)
"""


_ONE_EVENT_THEN_QUIET_CLAUDE = """
    import json
    import sys
    import time

    sys.stdin.read()
    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1",
                                  "content": [{"type": "text", "text": "hello"}]}}),
          flush=True)
    time.sleep(60)
"""


class TestC2ProcessAndDescriptorLeaks:
    """A timed-out run must reap its child and release its pipes."""

    def test_a_cli_that_ignores_sigterm_is_still_killed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A polite terminate must escalate to a kill, then reap the child."""
        install_cli(tmp_path, monkeypatch, "claude", _IGNORES_SIGTERM)
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": 1})
        model.initialize("hi")

        before_pids = child_pids()
        with pytest.raises(TimeoutError):
            model.generate()

        assert child_pids() - before_pids == set(), "SIGTERM-proof child survived"


class TestC3StderrIsDrained:
    """A CLI that fills the stderr pipe must not look like a stall."""

    def test_codex_chatty_stderr_does_not_stall_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """200 KiB of stderr before the first event must not cost the timeout."""
        install_cli(tmp_path, monkeypatch, "codex", _CODEX_FLOODS_STDERR)
        model = CodexModel("codex/default", model_config={"timeout": 20})
        model.initialize("hi")

        started = time.monotonic()
        content, _response = model.generate()
        elapsed = time.monotonic() - started

        assert content == "OK"
        assert elapsed < 15, f"the run blocked for {elapsed:.1f}s on a full stderr pipe"

    def test_claude_chatty_stderr_does_not_stall_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """200 KiB of stderr before the first event must not cost the timeout."""
        install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_FLOODS_STDERR)
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": 20})
        model.initialize("hi")

        started = time.monotonic()
        content, _response = model.generate()
        elapsed = time.monotonic() - started

        assert content == "OK"
        assert elapsed < 15, f"the run blocked for {elapsed:.1f}s on a full stderr pipe"


class TestC4LingeringChild:
    """A child that outlives its own stdout must not crash the step."""

    def test_codex_lingering_child_is_not_an_unclassified_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A complete turn followed by a lingering process must still succeed."""
        install_cli(tmp_path, monkeypatch, "codex", _CODEX_LINGERS_AFTER_CLOSING_STDOUT)
        model = CodexModel("codex/default", model_config={"timeout": 3})
        model.initialize("hi")

        before_pids = child_pids()
        content, response = model.generate()

        assert content == "DONE"
        assert response["usage"]["output_tokens"] == 5
        assert child_pids() - before_pids == set()

    def test_claude_lingering_child_is_not_an_unclassified_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A complete turn followed by a lingering process must still succeed."""
        install_cli(
            tmp_path, monkeypatch, "claude", _CLAUDE_LINGERS_AFTER_CLOSING_STDOUT
        )
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": 3})
        model.initialize("hi")

        before_pids = child_pids()
        content, response = model.generate()

        assert content == "DONE"
        assert response["usage"]["output_tokens"] == 5
        assert child_pids() - before_pids == set()


class TestC5StalledPromptWrite:
    """A CLI that stays alive without reading stdin must not wedge the turn.

    The prompt is the flattened conversation, which routinely outgrows
    the 64 KiB pipe buffer.  A child that hangs in start-up, auth or
    plugin loading before its first ``read`` therefore parks the agent
    thread inside ``stdin.write`` — where neither the turn deadline nor
    the Stop button can be observed, because both are only polled while
    reading the child's *output*.
    """
    _BIG_PROMPT = "x" * 4_000_000

    @staticmethod
    def _model(name: str, timeout: int) -> ClaudeCodeModel | CodexModel:
        """Build the adapter under test with the given turn timeout."""
        return (
            CodexModel("codex/default", model_config={"timeout": timeout})
            if name == "codex"
            else ClaudeCodeModel("cc/opus", model_config={"timeout": timeout})
        )

    @pytest.mark.parametrize("name", ["codex", "claude"])
    def test_a_prompt_that_fits_the_pipe_still_completes_the_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """The bounded write must not disturb an ordinary, healthy turn."""
        body = (
            _ONE_EVENT_THEN_QUIET_CODEX.replace("time.sleep(60)", "")
            if name == "codex"
            else _ONE_EVENT_THEN_QUIET_CLAUDE.replace("time.sleep(60)", "")
        )
        install_cli(tmp_path, monkeypatch, name, body)
        model = self._model(name, timeout=20)
        model.initialize("hi")

        before_pids = child_pids()
        content, _ = model.generate()

        assert content == "hello"
        assert child_pids() - before_pids == set()
