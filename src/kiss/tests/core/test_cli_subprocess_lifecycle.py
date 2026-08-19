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
import textwrap
import threading
import time
from pathlib import Path

import pytest

from kiss.core import stop_signal
from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.test_cli_subprocess_lifecycle import (  # noqa: F401
    _ONE_EVENT_THEN_QUIET_CLAUDE,
    _ONE_EVENT_THEN_QUIET_CODEX,
    _PS,
    child_pids,
    install_cli,
)


def open_fd_count() -> int:
    """Return the number of descriptors this process currently holds."""
    return len(os.listdir("/dev/fd"))


def generate_on_a_worker_thread(
    model: "ClaudeCodeModel | CodexModel", stop_event: threading.Event | None = None
) -> tuple[threading.Thread, list[BaseException]]:
    """Run ``generate()`` on a daemon worker thread, capturing what it raises.

    A daemon thread keeps a regression *visible* rather than fatal: when
    the call under test blocks forever the test still fails on its
    ``join(timeout=...)`` and the session continues, instead of wedging
    pytest until somebody kills it.

    Args:
        model: The adapter whose turn is driven.
        stop_event: Optional Stop event bound to the worker thread, so
            the test can press Stop from the outside.

    Returns:
        The started thread and the list that receives its exception.
    """
    raised: list[BaseException] = []
    running = threading.Event()

    def _worker() -> None:
        if stop_event is not None:
            stop_signal.set_thread_stop_event(stop_event)
        running.set()
        try:
            model.generate()
        except BaseException as exc:  # noqa: BLE001 – the test classifies it
            raised.append(exc)
        finally:
            stop_signal.set_thread_stop_event(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    running.wait(timeout=5)
    return thread, raised


_SLEEPS_FOREVER = """
    import sys
    import time

    sys.stdin.read()
    time.sleep(60)
"""


_CODEX_LATE_WRITER = """
    import json
    import sys
    import time

    time.sleep(7)
    sys.stdout.write(json.dumps({"type": "item.completed",
                                 "item": {"type": "agent_message",
                                          "text": "LATE"}}) + "\\n")
    sys.stdout.flush()
"""


_CODEX_LEAKS_A_GRANDCHILD = """
    import json
    import subprocess
    import sys
    import time

    sys.stdin.read()
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "FIRST"}}),
          flush=True)
    # The grandchild inherits stdout, so killing us does NOT close the
    # pipe: whoever is reading it stays blocked until the grandchild
    # writes its line and exits.
    subprocess.Popen([sys.executable, sys.argv[0] + ".late.py"])
    time.sleep(60)
"""


_EXITS_WITHOUT_READING_STDIN = """
    import os
    import sys

    sys.stderr.write("unknown --model value\\n")
    sys.stderr.flush()
    os._exit(2)
"""


_ALIVE_BUT_NEVER_READS_STDIN = """
    import time

    # Never touches stdin: the pipe buffer fills and stays full for the
    # whole minute, so whoever is writing the prompt blocks in write(2).
    time.sleep(60)
"""


class TestC1LateReaderOutput:
    """A timed-out run must never emit into the next run's callbacks."""

    def test_codex_late_output_never_reaches_the_next_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Output that arrives after the timeout must be dropped, not printed.

        The stand-in CLI leaves a grandchild holding the stdout pipe, so
        the adapter's reader is still parked in ``read(2)`` when the
        timeout fires and the child is killed.  Whatever that reader
        eventually parses belongs to a run that already failed; the
        model instance has meanwhile been rebound to the next task's
        printer callbacks (``KISSAgent._reset``).
        """
        (tmp_path / "codex.late.py").write_text(textwrap.dedent(_CODEX_LATE_WRITER))
        install_cli(tmp_path, monkeypatch, "codex", _CODEX_LEAKS_A_GRANDCHILD)

        first_run: list[str] = []
        model = CodexModel(
            "codex/default",
            model_config={"timeout": 1},
            token_callback=first_run.append,
        )
        model.initialize("hi")

        started = time.monotonic()
        with pytest.raises((TimeoutError, KISSError)):
            model.generate()

        next_run: list[str] = []
        model.token_callback = next_run.append
        time.sleep(max(0.0, started + 9.0 - time.monotonic()))

        assert next_run == [], (
            f"the abandoned reader emitted {next_run} into the next run"
        )


class TestC2ProcessAndDescriptorLeaks:
    """A timed-out run must reap its child and release its pipes."""

    @pytest.mark.parametrize("name", ["codex", "claude"])
    def test_timed_out_run_leaves_no_child_and_no_leaked_descriptors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """Twenty timed-out calls must not accumulate zombies or descriptors."""
        install_cli(tmp_path, monkeypatch, name, _SLEEPS_FOREVER)
        model: ClaudeCodeModel | CodexModel = (
            CodexModel("codex/default", model_config={"timeout": 0.2})
            if name == "codex"
            else ClaudeCodeModel("cc/opus", model_config={"timeout": 0.2})
        )
        model.initialize("hi")

        before_pids = child_pids()
        before_fds = open_fd_count()
        for _ in range(20):
            with pytest.raises((TimeoutError, KISSError)):
                model.generate()

        leaked_pids = child_pids() - before_pids
        leaked_fds = open_fd_count() - before_fds
        assert leaked_pids == set(), f"unreaped children: {sorted(leaked_pids)}"
        assert leaked_fds < 5, f"leaked {leaked_fds} descriptors over 20 calls"


class TestC5EarlyExitDuringPromptWrite:
    """A CLI that rejects its arguments must be reported by exit status."""

    @pytest.mark.parametrize("name", ["codex", "claude"])
    def test_early_exit_during_large_prompt_write_reports_exit_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """A megabyte prompt to a dead CLI must raise KISSError, not OSError."""
        install_cli(tmp_path, monkeypatch, name, _EXITS_WITHOUT_READING_STDIN)
        model: ClaudeCodeModel | CodexModel = (
            CodexModel("codex/default", model_config={"timeout": 10})
            if name == "codex"
            else ClaudeCodeModel("cc/opus", model_config={"timeout": 10})
        )
        model.initialize("x" * 1_500_000)

        with pytest.raises(KISSError, match="exit 2"):
            model.generate()


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
    def test_a_stalled_prompt_write_ends_at_the_turn_deadline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """The configured timeout must fire even before the child reads."""
        install_cli(tmp_path, monkeypatch, name, _ALIVE_BUT_NEVER_READS_STDIN)
        model = self._model(name, timeout=2)
        model.initialize(self._BIG_PROMPT)

        before_pids = child_pids()
        started = time.monotonic()
        thread, raised = generate_on_a_worker_thread(model)
        thread.join(timeout=25)
        elapsed = time.monotonic() - started

        assert not thread.is_alive(), "stdin.write ignored the turn deadline"
        assert raised and isinstance(raised[0], TimeoutError), raised
        assert not isinstance(raised[0], KISSError), (
            "a stalled prompt write must stay retryable"
        )
        assert elapsed < 20, f"the deadline fired {elapsed:.1f}s late"
        assert child_pids() - before_pids == set(), "the stalled CLI survived"

    @pytest.mark.parametrize("name", ["codex", "claude"])
    def test_stop_interrupts_a_stalled_prompt_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """Stop must unwind a prompt write the child is not draining."""
        install_cli(tmp_path, monkeypatch, name, _ALIVE_BUT_NEVER_READS_STDIN)
        model = self._model(name, timeout=120)
        model.initialize(self._BIG_PROMPT)

        before_pids = child_pids()
        stop_event = threading.Event()
        thread, raised = generate_on_a_worker_thread(model, stop_event)
        time.sleep(1.0)
        stop_event.set()
        thread.join(timeout=15)

        assert not thread.is_alive(), "Stop could not reach the blocked writer"
        assert raised and isinstance(raised[0], KeyboardInterrupt), raised
        assert child_pids() - before_pids == set(), "the CLI child survived Stop"


class TestI6StopAndStallClassification:
    """The CLI adapters must honour Stop and classify a stall as retryable."""

    @pytest.mark.parametrize("name", ["codex", "claude"])
    def test_stop_aborts_a_quiet_cli_at_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """Pressing Stop during a quiet CLI turn must unwind within seconds."""
        body = (
            _ONE_EVENT_THEN_QUIET_CODEX
            if name == "codex"
            else _ONE_EVENT_THEN_QUIET_CLAUDE
        )
        install_cli(tmp_path, monkeypatch, name, body)
        model: ClaudeCodeModel | CodexModel = (
            CodexModel("codex/default", model_config={"timeout": 20})
            if name == "codex"
            else ClaudeCodeModel("cc/opus", model_config={"timeout": 20})
        )
        model.initialize("hi")

        before_pids = child_pids()
        stop_event = threading.Event()
        thread, raised = generate_on_a_worker_thread(model, stop_event)
        time.sleep(1.0)
        stop_event.set()
        thread.join(timeout=6)

        assert not thread.is_alive(), "Stop did not interrupt the CLI turn"
        assert raised and isinstance(raised[0], KeyboardInterrupt), raised
        assert child_pids() - before_pids == set(), "the CLI child survived Stop"

    @pytest.mark.parametrize("name", ["codex", "claude"])
    def test_a_stall_is_a_retryable_timeout_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """A stalled CLI must raise ``TimeoutError`` — the retryable class.

        ``KISSAgent._run_agentic_loop`` re-raises every ``KISSError``
        immediately and only retries other exceptions, so reporting a
        transient stall as ``KISSError`` aborts the whole task where the
        equivalent Anthropic condition would simply be retried.
        """
        body = (
            _ONE_EVENT_THEN_QUIET_CODEX
            if name == "codex"
            else _ONE_EVENT_THEN_QUIET_CLAUDE
        )
        install_cli(tmp_path, monkeypatch, name, body)
        model: ClaudeCodeModel | CodexModel = (
            CodexModel("codex/default", model_config={"timeout": 1})
            if name == "codex"
            else ClaudeCodeModel("cc/opus", model_config={"timeout": 1})
        )
        model.initialize("hi")

        with pytest.raises(TimeoutError) as excinfo:
            model.generate()

        assert "timed out" in str(excinfo.value)
        assert not isinstance(excinfo.value, KISSError), (
            "a stall must not be reported as a non-retryable KISSError"
        )
