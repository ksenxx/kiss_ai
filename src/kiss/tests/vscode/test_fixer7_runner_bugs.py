# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for fixer-7 findings (round 6).

Covers:

- F1: ``_cmd_run`` wedged a tab forever when the ``clear`` broadcast or
  ``thread.start()`` raised after ``tab.task_thread`` was assigned.
- F3: ``_resolve_user_answer_queue`` lacked the task-ownership filter
  its two task_runner siblings have, letting a stale viewer answer be
  hijacked by an unrelated running task.
- F4: the ``bash_stream`` inline flush broadcast stale text after a
  concurrent ``reset()`` without the generation re-check that
  ``_flush_bash`` performs.
- F10: a junk ``KISS_VOICE_MIC_BLOCK_SIZE`` env value crashed the voice
  listener instead of falling back to the default block size.
- F11: a wake-word match suppressed during cooldown never reset the
  recognizer, so a phantom WAKE fired from stale audio after the
  cooldown expired.
- F16: a ``KeyboardInterrupt`` (Stop during pre-task setup) escaped
  ``_run_task``'s ``except Exception`` so the task vanished silently
  with no ``result`` event.

No mocks/patches/fakes: real :class:`JsonPrinter` subclasses (the same
technique the existing vscode regression tests use) capture or steer
broadcasts, real :class:`VSCodeServer` / :class:`_RunningAgentState` /
:class:`WorktreeSorcarAgent` objects are exercised, and the wake-word
test streams real synthesized speech through the real Vosk recognizer.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* instead of writing it to stdout."""
        self.events.append(event)


def _pop_tabs(*tab_ids: str) -> None:
    """Remove test-created entries from the global tab registry."""
    for tab_id in tab_ids:
        _RunningAgentState.running_agent_states.pop(tab_id, None)


# ---------------------------------------------------------------------------
# F1 — _cmd_run: tab wedged forever when the clear broadcast raises
# ---------------------------------------------------------------------------


class _FailingClearPrinter(_CapturePrinter):
    """Transport whose ``clear`` broadcast always fails (subclass error)."""

    def broadcast(self, event: dict[str, Any]) -> None:
        """Raise for ``clear`` events; record everything else."""
        if event.get("type") == "clear":
            raise RuntimeError("transport send failed")
        super().broadcast(event)


def test_f1_failed_clear_broadcast_does_not_wedge_tab() -> None:
    printer = _FailingClearPrinter()
    server = VSCodeServer(printer=printer)
    tab_id = "f1-tab"
    cmd = {"type": "run", "tabId": tab_id, "prompt": "do something"}
    try:
        with pytest.raises(RuntimeError):
            server._cmd_run(dict(cmd))
        tab = _RunningAgentState.running_agent_states[tab_id]
        # The in-flight markers armed before the failed broadcast must
        # be rolled back — the worker thread never ran, so nothing else
        # will ever reset them.
        assert tab.task_thread is None
        assert tab.stop_event is None
        assert tab.user_answer_queue is None
        # Behavioral proof the tab is NOT wedged: a second run takes
        # the fresh-task path again (and fails at the same broadcast)
        # instead of the busy path.  Pre-fix, the busy path silently
        # queued the prompt as a follow-up no agent would ever drain
        # and echoed it back as a ``prompt`` event.
        with pytest.raises(RuntimeError):
            server._cmd_run(dict(cmd))
        assert tab.pending_user_messages == []
        assert all(e.get("type") != "prompt" for e in printer.events)
    finally:
        _pop_tabs(tab_id)


# ---------------------------------------------------------------------------
# F3 — _resolve_user_answer_queue: cross-task user-answer hijack
# ---------------------------------------------------------------------------


def test_f3_stale_viewer_answer_not_hijacked_by_unrelated_task() -> None:
    printer = _CapturePrinter()
    server = VSCodeServer(printer=printer)
    viewer, owner = "f3-viewer", "f3-owner"
    try:
        # ``viewer`` and ``owner`` co-subscribed to task 100, which has
        # FINISHED (cleanup_task intentionally preserves subscriber
        # sets).  ``owner`` is now running a brand-new UNRELATED task
        # 200 with a live answer queue owned by that task.
        printer.subscribe_tab(100, viewer)
        printer.subscribe_tab(100, owner)
        owner_state = _RunningAgentState(owner, "test-model")
        _RunningAgentState.running_agent_states[owner] = owner_state
        owner_state.user_answer_queue = queue.Queue(maxsize=1)
        owner_state.is_task_active = True
        owner_state.agent = WorktreeSorcarAgent("Sorcar VS Code")
        owner_state.agent._last_task_id = "200"
        printer.subscribe_tab(200, owner)

        # A stale answer submitted from the viewer's still-open modal
        # (for finished task 100) must NOT be delivered into task 200's
        # live queue.
        server._cmd_user_answer({"tabId": viewer, "answer": "stale"})
        assert owner_state.user_answer_queue.empty()
        assert all(e.get("type") != "askUserDone" for e in printer.events)

        # Correct routing is preserved: when the owner's live agent IS
        # running the shared task, the viewer's answer reaches it.
        owner_state.agent._last_task_id = "100"
        server._cmd_user_answer({"tabId": viewer, "answer": "real answer"})
        assert owner_state.user_answer_queue.get_nowait() == "real answer"
    finally:
        _pop_tabs(viewer, owner)


# ---------------------------------------------------------------------------
# F4 — bash_stream inline flush: stale broadcast after concurrent reset()
# ---------------------------------------------------------------------------


class _ResetRacePrinter(JsonPrinter):
    """Printer that lets a concurrent ``reset()`` race the inline flush.

    ``broadcast`` signals a waiting driver thread the moment a
    ``system_output`` emission begins, then gives that thread a window
    to complete ``reset()`` before recording whether the reset finished
    first.  With the generation re-check in place the broadcast happens
    while ``_bash_lock`` is held, so the racing ``reset()`` cannot
    complete first; without it, the reset slips into the window and the
    stale text is emitted into the new turn.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.in_system_output = threading.Event()
        self.reset_done = threading.Event()
        self.stale_emission = False

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*; for system_output, yield to the resetter."""
        if event.get("type") == "system_output":
            self.in_system_output.set()
            if self.reset_done.wait(timeout=0.4):
                self.stale_emission = True
        self.events.append(event)


def test_f4_inline_bash_flush_not_emitted_after_reset() -> None:
    printer = _ResetRacePrinter()
    task_key = "f4-task"

    def do_reset() -> None:
        printer._thread_local.task_id = task_key
        printer.in_system_output.wait(timeout=2.0)
        printer.reset()
        printer.reset_done.set()

    resetter = threading.Thread(target=do_reset, daemon=True)
    resetter.start()
    printer._thread_local.task_id = task_key
    # ``last_flush`` starts at 0.0, so the first chunk takes the inline
    # (immediate) flush path rather than arming the 0.1 s timer.
    printer.print("previous-turn output", type="bash_stream")
    resetter.join(timeout=5.0)
    assert not resetter.is_alive()
    # The chunk itself must have been flushed (the reset raced it, it
    # did not precede it) ...
    assert any(
        e.get("type") == "system_output"
        and "previous-turn output" in e.get("text", "")
        for e in printer.events
    )
    # ... but the emission must not have happened AFTER the new turn's
    # reset() completed — that would leak the previous turn's stale
    # bash output into the new turn's transcript.
    assert not printer.stale_emission


# ---------------------------------------------------------------------------
# F10 — voice_wake: junk KISS_VOICE_MIC_BLOCK_SIZE must not crash
# ---------------------------------------------------------------------------


def test_f10_mic_block_size_falls_back_on_junk_env() -> None:
    from kiss.agents.vscode.voice_wake import BLOCK_SIZE, mic_block_size

    saved = os.environ.pop("KISS_VOICE_MIC_BLOCK_SIZE", None)
    try:
        assert mic_block_size() == BLOCK_SIZE
        for junk in ("abc", "", "4000.5", "-1", "0", " "):
            os.environ["KISS_VOICE_MIC_BLOCK_SIZE"] = junk
            assert mic_block_size() == BLOCK_SIZE, junk
        os.environ["KISS_VOICE_MIC_BLOCK_SIZE"] = "2048"
        assert mic_block_size() == 2048
    finally:
        if saved is None:
            os.environ.pop("KISS_VOICE_MIC_BLOCK_SIZE", None)
        else:
            os.environ["KISS_VOICE_MIC_BLOCK_SIZE"] = saved


# ---------------------------------------------------------------------------
# F11 — voice_wake: no phantom WAKE after a cooldown-suppressed match
# ---------------------------------------------------------------------------

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))


def _have_vosk() -> bool:
    try:
        import vosk  # noqa: F401
    except Exception:
        return False
    return True


def _say_sorcar_pcm(directory: Path) -> bytes:
    """Synthesize one spoken "Sorcar" as raw 16kHz mono s16le PCM."""
    aiff = directory / "sorcar.aiff"
    wav = directory / "sorcar.wav"
    subprocess.run(["say", "Sorcar", "-o", str(aiff)], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    with wave.open(str(wav), "rb") as wf:
        return wf.readframes(wf.getnframes())


@pytest.mark.skipif(
    not HAVE_MAC_TTS, reason="requires macOS `say` and `afconvert`",
)
@pytest.mark.skipif(not _have_vosk(), reason="requires vosk")
def test_f11_cooldown_suppressed_wake_resets_cleanly(tmp_path: Path) -> None:
    """The suppressed-match path (now calling ``Reset()``) end to end.

    A second "Sorcar" spoken inside the cooldown must (a) be
    suppressed, (b) leave no stale recognizer state that fires a
    phantom WAKE out of the following silence once the cooldown
    expires, and (c) not deafen the detector — a third "Sorcar" spoken
    well after the cooldown must still fire.

    Note: with the alias grammar in this environment Vosk emits empty
    *partial* results (matches arrive only as endpointed FINAL
    results), so the exact pre-fix phantom (a leftover partial
    re-matching every silent block) is not synthetically reproducible
    here; this test locks in the observable contract around the fixed
    code path instead.
    """
    from kiss.agents.vscode.voice_wake import (
        BLOCK_SIZE,
        COOLDOWN_SECONDS,
        DEFAULT_MODELS_DIR,
        SAMPLE_RATE,
        WakeDetector,
        ensure_model,
    )

    model_dir = ensure_model(DEFAULT_MODELS_DIR)
    detector = WakeDetector(model_dir)
    wakes: list[float] = []
    silence_block = b"\x00" * (BLOCK_SIZE * 2)

    def feed_pcm(pcm: bytes) -> None:
        step = BLOCK_SIZE * 2  # frames -> bytes (s16le)
        for i in range(0, len(pcm), step):
            if detector.feed(pcm[i : i + step]):
                wakes.append(detector._audio_seconds)

    def feed_silence(seconds: float) -> None:
        for _ in range(int(seconds * SAMPLE_RATE / BLOCK_SIZE)):
            if detector.feed(silence_block):
                wakes.append(detector._audio_seconds)

    def feed_silence_until_wake(max_seconds: float) -> None:
        before = len(wakes)
        for _ in range(int(max_seconds * SAMPLE_RATE / BLOCK_SIZE)):
            if detector.feed(silence_block):
                wakes.append(detector._audio_seconds)
                return
        assert len(wakes) > before, "expected a wake within the window"

    pcm = _say_sorcar_pcm(tmp_path)
    utterance_seconds = len(pcm) / 2 / SAMPLE_RATE
    if utterance_seconds > COOLDOWN_SECONDS - 1.0:
        pytest.skip("TTS utterance too long to land inside the cooldown")

    # First "Sorcar": genuine wake (fires during the trailing silence
    # once Vosk endpoints the utterance).
    feed_pcm(pcm)
    feed_silence_until_wake(2.0)
    wake1 = wakes[0]

    # Second "Sorcar" immediately after: its match lands well inside
    # the cooldown and must be suppressed — and the suppression (which
    # now resets the recognizer) must not leave state that fires a
    # phantom WAKE out of the following 4 s of pure silence after the
    # cooldown expires.
    feed_pcm(pcm)
    feed_silence(4.0)
    if len(wakes) > 1 and wakes[1] - wake1 >= COOLDOWN_SECONDS - 0.05:
        pytest.skip(
            "second utterance endpointed after the cooldown expired; "
            "cannot arrange suppression with this TTS timing"
        )
    assert wakes == [wake1], f"unexpected wake(s) after suppression: {wakes}"

    # Third "Sorcar" well past the cooldown: the added ``Reset()`` on
    # the suppressed path must not deafen the detector.
    feed_pcm(pcm)
    feed_silence_until_wake(2.0)
    assert len(wakes) == 2
    assert wakes[1] - wake1 >= COOLDOWN_SECONDS


# ---------------------------------------------------------------------------
# F16 — _run_task: Stop (KeyboardInterrupt) during setup vanished silently
# ---------------------------------------------------------------------------


class _StopDuringSetupPrinter(_CapturePrinter):
    """Injects a real ``KeyboardInterrupt`` into the task setup path.

    Raises on the first ``status running:true`` broadcast — i.e. inside
    ``_run_task``'s ``try`` before ``_run_task_inner`` runs, the same
    place a Stop-injected ``KeyboardInterrupt`` lands during
    ``_capture_pre_snapshot`` of a large repo.
    """

    def __init__(self) -> None:
        super().__init__()
        self._raised = False

    def broadcast(self, event: dict[str, Any]) -> None:
        """Raise KeyboardInterrupt once on the first status broadcast."""
        if (
            not self._raised
            and event.get("type") == "status"
            and event.get("running") is True
        ):
            self._raised = True
            raise KeyboardInterrupt("stop clicked during setup")
        super().broadcast(event)


def test_f16_keyboard_interrupt_during_setup_broadcasts_result(
    tmp_path: Path,
) -> None:
    printer = _StopDuringSetupPrinter()
    server = VSCodeServer(printer=printer)
    tab_id = "f16-tab"
    try:
        # Pre-fix this re-raised out of ``_run_task`` (only
        # ``Exception`` was caught): the spinner stopped but no result
        # was ever broadcast — the task silently vanished.
        server._run_task({
            "tabId": tab_id,
            "prompt": "x",
            "workDir": str(tmp_path),
        })
        results = [e for e in printer.events if e.get("type") == "result"]
        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["tabId"] == tab_id
        assert "stopped" in results[0]["text"].lower()
        # The finally still ends the spinner.
        assert any(
            e.get("type") == "status" and e.get("running") is False
            for e in printer.events
        )
    finally:
        _pop_tabs(tab_id)
