# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Claude-CLI-style anchored input box for the ``sorcar`` command line.

While a task runs in the terminal the user can keep typing follow-up
instructions into a bordered input box pinned to the bottom of the
screen.  Submitted lines are queued exactly the way the VS Code
frontend's ``appendUserMessage`` command queues them — appended to the
owning :class:`~kiss.agents.sorcar.running_agent_state._RunningAgentState`'s
``pending_user_messages`` list under ``_registry_lock`` so the live
agent's pre-step hook
(:meth:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent._drain_pending_user_messages`)
injects them into the model conversation before the next model step.

Implementation notes (researched against Claude Code / opencode TUIs):

* The bottom box is anchored using the DEC "Set Top and Bottom Margins"
  (DECSTBM) scroll region — ``ESC[1;{rows-BOX_H}r``.  All agent output
  scrolls inside the region while the box rows below it stay put.
* The terminal is put in a semi-raw mode (``ICANON``/``ECHO`` off,
  ``ISIG`` kept so ``Ctrl+C`` still interrupts) and keys are read in the
  main thread via :func:`select.select` so the loop can also notice the
  worker finishing.
* Agent output is intercepted by swapping :data:`sys.stdout` for a
  lock-guarded wrapper *before* the agent builds its
  :class:`~kiss.core.print_to_console.ConsolePrinter`, so console writes
  and box redraws never interleave on the wire.

Everything degrades gracefully: on Windows, when stdin/stdout is not a
TTY, or when the terminal is too small, :func:`run_with_steering` simply
runs the agent normally with no box.
"""

from __future__ import annotations

import os
import queue
import select
import shutil
import sys
import threading
from typing import TYPE_CHECKING, Any, cast

from kiss.agents.sorcar.cli_panel import (
    CYAN,
    DIM,
    RESET,
    STEER_TITLE,
    panel_body,
    panel_bottom,
    panel_cols,
    panel_top,
)
from kiss.agents.sorcar.persistence import _allocate_chat_id
from kiss.agents.sorcar.running_agent_state import _RunningAgentState

if TYPE_CHECKING:
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

try:  # POSIX-only terminal control; absent on Windows.
    import termios

    _HAVE_TERMIOS = True
except ImportError:  # pragma: no cover - exercised only on Windows
    termios = None  # type: ignore[assignment]
    _HAVE_TERMIOS = False

# Height (rows) reserved at the bottom of the screen for the input box.
_BOX_H = 3
# Minimum terminal height for which the anchored box is worthwhile.
_MIN_ROWS = _BOX_H + 3
_ESC = "\x1b"


def supports_steering() -> bool:
    """Return whether an anchored input box can be rendered.

    Requires a POSIX terminal (``termios`` available) with both stdin
    and stdout attached to a TTY.

    Returns:
        ``True`` when the interactive box should be used.
    """
    if not _HAVE_TERMIOS:
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def _term_size() -> tuple[int, int]:
    """Return ``(rows, cols)`` for the controlling terminal.

    Falls back to ``(24, 80)`` when the size cannot be determined.

    Returns:
        A ``(rows, cols)`` tuple, both guaranteed ``>= 1``.
    """
    size = shutil.get_terminal_size(fallback=(80, 24))
    return max(size.lines, 1), max(size.columns, 1)


class _StdoutProxy:
    """A ``sys.stdout`` replacement that serialises writes with box redraws.

    All attribute access other than :meth:`write`/:meth:`flush` is
    delegated to the real stream so Rich still detects the TTY, colour
    support and terminal width correctly.

    Attributes:
        _stream: The original stdout stream.
        _lock: Shared re-entrant lock guarding terminal writes.
    """

    def __init__(self, stream: Any, lock: threading.RLock) -> None:
        self._stream = stream
        self._lock = lock

    def write(self, text: str) -> int:
        """Write *text* to the underlying stream under the shared lock.

        Args:
            text: The string to write.

        Returns:
            The number of characters written.
        """
        with self._lock:
            n = self._stream.write(text)
            self._stream.flush()
            return int(n)

    def flush(self) -> None:
        """Flush the underlying stream under the shared lock."""
        with self._lock:
            self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        # Delegated for isatty(), fileno(), encoding, etc.
        return getattr(self._stream, name)


class _InputBox:
    """Renders and edits the anchored bottom input line.

    The box owns the raw-terminal lifecycle (scroll region, cursor
    visibility, ``termios`` mode) and the in-progress edit buffer.  It
    does *not* know anything about agents — submitted lines are handed
    back to the owner via the callback passed to :meth:`feed`.

    Attributes:
        lock: Shared lock (also used by :class:`_StdoutProxy`).
        buf: The current edit buffer.
        title: Text shown in the top border.
        status: Right-aligned text shown in the bottom border.
    """

    def __init__(self, lock: threading.RLock, out: Any) -> None:
        self.lock = lock
        self.buf = ""
        self.title = STEER_TITLE
        self.status = ""
        self._out = out
        self._fd = -1
        self._old_term: Any = None
        self._active = False

    def start(self) -> None:
        """Enter raw mode, reserve the scroll region and draw the box."""
        assert termios is not None
        rows, _ = _term_size()
        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        new = termios.tcgetattr(self._fd)
        # Disable canonical mode + echo; keep ISIG so Ctrl+C interrupts.
        new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSANOW, new)
        with self.lock:
            out = self._out
            # Push existing content up so the box does not overwrite it.
            out.write("\n" * _BOX_H)
            # Scroll region = everything above the box.
            out.write(f"{_ESC}[1;{rows - _BOX_H}r")
            out.write(f"{_ESC}[?25l")  # hide the real cursor
            # Park the output cursor on the last region row.
            out.write(f"{_ESC}[{rows - _BOX_H};1H")
            out.flush()
            self._active = True
            self._draw_locked()

    def stop(self) -> None:
        """Reset the scroll region, restore the cursor and terminal mode."""
        if not self._active:
            return
        assert termios is not None
        rows, _ = _term_size()
        top_row = rows - _BOX_H + 1
        with self.lock:
            out = self._out
            out.write(f"{_ESC}[r")  # reset scroll region to full screen
            # Erase the box's rows so the steering panel does not linger
            # once the task ends.  Otherwise the idle REPL prompt would
            # be drawn *below* the stale steering box, leaving two input
            # panels stacked on screen at once.
            for row in range(top_row, rows + 1):
                out.write(f"{_ESC}[{row};1H{_ESC}[2K")
            out.write(f"{_ESC}[?25h")  # show cursor
            # Park the cursor on the box's old first row so following
            # output (the returning idle prompt) flows from there.
            out.write(f"{_ESC}[{top_row};1H")
            out.flush()
            self._active = False
        if self._old_term is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
            self._old_term = None

    def redraw(self) -> None:
        """Redraw the box, preserving the output cursor position."""
        with self.lock:
            if self._active:
                self._draw_locked()

    def _draw_locked(self) -> None:
        rows, _ = _term_size()
        cols = panel_cols()
        top_row = rows - _BOX_H + 1
        out = self._out

        top = panel_top(self.title, cols)
        bottom = panel_bottom(self.status, cols)
        body, is_placeholder = panel_body(self.buf, cols)
        mid_color = DIM if is_placeholder else ""
        mid = "│ " + mid_color + body + RESET + " │"

        # Save output cursor, paint the three box rows, restore it.
        out.write(_ESC + "7")
        out.write(f"{_ESC}[{top_row};1H{_ESC}[2K{CYAN}{top}{RESET}")
        out.write(f"{_ESC}[{top_row + 1};1H{_ESC}[2K{CYAN}│{RESET}{mid[1:-1]}{CYAN}│{RESET}")
        out.write(f"{_ESC}[{top_row + 2};1H{_ESC}[2K{CYAN}{bottom}{RESET}")
        out.write(_ESC + "8")
        out.flush()

    def feed(self, data: bytes, on_submit: Any, on_abort: Any) -> None:
        """Process a chunk of raw keyboard input.

        Args:
            data: Raw bytes read from stdin.
            on_submit: Callable invoked with each completed line (string).
            on_abort: Callable invoked when Ctrl+C is pressed.
        """
        text = data.decode("utf-8", "ignore")
        i = 0
        changed = False
        while i < len(text):
            ch = text[i]
            if ch == "\x1b":
                # Swallow CSI escape sequences (arrow keys, etc.).
                if i + 1 < len(text) and text[i + 1] == "[":
                    j = i + 2
                    while j < len(text) and not ("@" <= text[j] <= "~"):
                        j += 1
                    i = j + 1
                    continue
                i += 1
                continue
            if ch in ("\r", "\n"):
                line = self.buf
                self.buf = ""
                changed = True
                on_submit(line)
            elif ch in ("\x7f", "\x08"):
                if self.buf:
                    self.buf = self.buf[:-1]
                    changed = True
            elif ch == "\x15":  # Ctrl+U clears the line
                if self.buf:
                    self.buf = ""
                    changed = True
            elif ch == "\x03":  # Ctrl+C
                on_abort()
                return
            elif ch >= " ":
                self.buf += ch
                changed = True
            i += 1
        if changed:
            self.redraw()


class SteeringSession:
    """Runs an agent task while accepting queued follow-up instructions.

    Attributes:
        agent: The live agent instance.
        state: The agent's registry entry (holds ``pending_user_messages``).
        lock: Shared terminal lock.
    """

    def __init__(
        self,
        agent: SorcarAgent,
        state: _RunningAgentState,
        chat_id: str,
    ) -> None:
        self.agent = agent
        self.state = state
        self.chat_id = chat_id
        self.lock = threading.RLock()
        # Capture the real stdout now (before :meth:`run` swaps in the
        # proxy) so box rendering writes straight to the terminal.
        self._real_stdout = sys.stdout
        self.box = _InputBox(self.lock, self._real_stdout)
        self._done = threading.Event()
        self._aborted = threading.Event()
        self._result = ""
        self._error: BaseException | None = None
        self._queued_count = 0
        # ask_user_question coordination.
        self._answer_q: queue.Queue[str] = queue.Queue(maxsize=1)
        self._question_pending = threading.Event()

    def ask_user_question(self, question: str) -> str:
        """Collect an answer to *question* through the bottom input box.

        Runs in the worker thread; blocks until the user submits a line.

        Args:
            question: The question text to display above the box.

        Returns:
            The user's typed answer (possibly empty).
        """
        with self.lock:
            sys.stdout.write(f"\n\x1b[33m? {question}\x1b[0m\n")
            sys.stdout.flush()
        prev_title = self.box.title
        self.box.title = " answer the question above, then Enter "
        self.box.redraw()
        self._question_pending.set()
        try:
            return self._answer_q.get()
        finally:
            self._question_pending.clear()
            self.box.title = prev_title
            self.box.redraw()

    def _on_submit(self, line: str) -> None:
        if self._question_pending.is_set():
            try:
                self._answer_q.put_nowait(line)
            except queue.Full:  # pragma: no cover - drained by waiter
                pass
            return
        text = line.strip()
        if not text:
            return
        with _RunningAgentState._registry_lock:
            self.state.pending_user_messages.append(text)
        self._queued_count += 1
        self.box.status = f" queued: {self._queued_count} "
        with self.lock:
            sys.stdout.write(f"\x1b[2m▸ queued: {text}\x1b[0m\n")
            sys.stdout.flush()

    def _on_abort(self) -> None:
        self._aborted.set()
        self._done.set()

    def _worker(self, run_kwargs: dict[str, Any]) -> None:
        try:
            self._result = self.agent.run(**run_kwargs)
        except BaseException as exc:  # noqa: BLE001 - surfaced to caller
            self._error = exc
        finally:
            self._done.set()

    def run(self, run_kwargs: dict[str, Any]) -> str:
        """Run the task with the box active and return the YAML result.

        Args:
            run_kwargs: Keyword arguments forwarded to ``agent.run``.

        Returns:
            The agent's YAML result string.

        Raises:
            KeyboardInterrupt: If the user aborts with Ctrl+C.
        """
        real_stdout = self._real_stdout
        proxy = _StdoutProxy(real_stdout, self.lock)
        sys.stdout = cast(Any, proxy)
        self.box.start()
        worker = threading.Thread(
            target=self._worker, args=(run_kwargs,), daemon=True
        )
        worker.start()
        try:
            self._loop()
        finally:
            self.box.stop()
            sys.stdout = real_stdout
        if self._aborted.is_set():
            raise KeyboardInterrupt
        if self._error is not None:
            raise self._error
        return self._result

    def _loop(self) -> None:
        fd = sys.stdin.fileno()
        while not self._done.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
            except (InterruptedError, OSError):
                continue
            except KeyboardInterrupt:
                self._on_abort()
                return
            if not ready:
                continue
            try:
                data = os.read(fd, 4096)
            except (InterruptedError, OSError):
                continue
            if not data:
                continue
            self.box.feed(data, self._on_submit, self._on_abort)


def run_with_steering(
    agent: SorcarAgent, run_kwargs: dict[str, Any]
) -> str:
    """Run *agent* with a Claude-CLI-style steering input box when possible.

    When the terminal supports it, registers a transient
    :class:`_RunningAgentState` so the agent's pre-step hook can drain
    instructions the user queues in the box, then runs the task with the
    box pinned to the bottom of the screen.  Falls back to a plain
    ``agent.run`` otherwise.

    Args:
        agent: The agent to run.
        run_kwargs: Keyword arguments for ``agent.run``.

    Returns:
        The agent's YAML result string.
    """
    rows, _ = _term_size()
    if not supports_steering() or rows < _MIN_ROWS:
        return str(agent.run(**run_kwargs))

    chat_id = getattr(agent, "_chat_id", "") or _allocate_chat_id()
    agent._chat_id = chat_id  # type: ignore[attr-defined]
    # The pre-step drain hook keys off ``_tab_id``; align it with the
    # chat id so a single registry entry serves both the worktree
    # agent's own registration and the drain lookup.
    agent._tab_id = chat_id  # type: ignore[attr-defined]

    state = _RunningAgentState(
        chat_id,
        getattr(agent, "model_name", "") or "",
        agent=cast(Any, agent),
    )
    state.chat_id = chat_id
    state.is_task_active = True
    _RunningAgentState.register(chat_id, state)

    session = SteeringSession(agent, state, chat_id)
    kwargs = dict(run_kwargs)
    kwargs["ask_user_question_callback"] = session.ask_user_question
    try:
        return session.run(kwargs)
    finally:
        if _RunningAgentState.running_agent_states.get(chat_id) is state:
            state.is_task_active = False
            _RunningAgentState.unregister(chat_id)
