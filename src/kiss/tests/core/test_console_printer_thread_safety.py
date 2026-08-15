# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end thread-safety tests for :class:`ConsolePrinter` (audit F1).

``SorcarAgent.run_tasks_parallel`` forwards the parent's printer object
*verbatim* to every parallel sub-agent thread, and ``_LiveUsageMonitor``
prints from a third daemon thread.  ``ConsolePrinter`` therefore has the
same fan-out ``JsonPrinter`` does, and must protect its streaming state
the same way.

These tests use a real ``ConsolePrinter``, real ``threading.Thread``s and
a real in-memory stream — no mocks, patches, fakes or test doubles.
"""

from __future__ import annotations

import io
import sys
import threading
from collections.abc import Iterator

import pytest

from kiss.core.print_to_console import ConsolePrinter

# Enough repetitions that the interleaving happens on every platform
# without making the suite slow.
_CYCLES = 60
_TAGS = ("A", "B", "C", "D")

# ANSI produced by rich for the ``dim cyan italic`` thinking style.
_THINKING_ANSI = "\x1b[2;3;36m"


class TtyStringIO(io.StringIO):
    """An in-memory stream that reports itself as a terminal.

    ``rich`` only emits ANSI style sequences when ``file.isatty()`` is
    true.  The thinking-block tests assert on the *style* each token was
    rendered with, so they need a stream rich treats as a terminal.  This
    is a real ``io.StringIO`` — writes and reads behave exactly as they
    do for the plain class.
    """

    def isatty(self) -> bool:
        """Return ``True`` so rich renders styles as ANSI escapes."""
        return True


@pytest.fixture
def fast_thread_switching() -> Iterator[None]:
    """Shrink the interpreter's thread switch interval for this test.

    The ``_mid_line`` window between "write the text" and "record that
    the cursor is mid-line" is a couple of bytecodes wide.  At the
    default 5 ms switch interval the scheduler almost never preempts
    there, so the corruption is invisible.  Lowering the interval makes
    the *real* race reproducible without changing any production code.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _bash_tool_cycle(printer: ConsolePrinter, tag: str, cycles: int) -> None:
    """Emit the bash-stream / tool-call / tool-result sequence *cycles* times."""
    for _ in range(cycles):
        printer.print(f"{tag}-out\n", type="bash_stream")
        printer.print("Bash", type="tool_call", tool_input={"command": tag})
        printer.print(
            f"{tag}-done",
            type="tool_result",
            tool_name="Bash",
            tool_input={"command": tag},
        )


def _run_threads(targets: list[threading.Thread]) -> None:
    for thread in targets:
        thread.start()
    for thread in targets:
        thread.join(timeout=60)
    for thread in targets:
        assert not thread.is_alive(), "worker thread did not finish"


def test_single_threaded_bash_cycle_counts_are_the_contract() -> None:
    """Pin the per-cycle output shape the concurrent test asserts against.

    One cycle opens a ``RESULT`` rule for the streamed bash output, closes
    it on the following ``tool_call``, then opens and closes a second
    ``RESULT`` rule around the tool result body.
    """
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)

    _bash_tool_cycle(printer, "A", _CYCLES)

    text = buf.getvalue()
    assert text.count("RESULT") == 2 * _CYCLES
    assert text.count("A-done") == _CYCLES
    assert text.count("A-out") == _CYCLES


def test_parallel_sub_agents_do_not_corrupt_bash_block_state(
    fast_thread_switching: None,
) -> None:
    """Four sub-agent threads sharing one printer keep their own bash state.

    With ``_bash_streamed`` as a single shared attribute, one thread's
    ``tool_call`` clears the flag another thread had just set, so the
    victim's ``tool_result`` silently drops its body (or emits a
    duplicate ``RESULT`` rule).  Every thread must produce exactly the
    single-threaded output shape.
    """
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)

    _run_threads([
        threading.Thread(target=_bash_tool_cycle, args=(printer, tag, _CYCLES))
        for tag in _TAGS
    ])

    text = buf.getvalue()
    for tag in _TAGS:
        assert text.count(f"{tag}-out") == _CYCLES, f"{tag} lost streamed output"
        assert text.count(f"{tag}-done") == _CYCLES, f"{tag} lost a result body"
    assert text.count("RESULT") == 2 * _CYCLES * len(_TAGS)


def test_thinking_block_style_is_not_shared_between_threads() -> None:
    """One sub-agent leaving a thinking block must not restyle another's tokens.

    ``_current_block_type`` is a single string shared by every thread, so
    when thread B ends its (independent) thinking block, thread A's
    remaining thinking tokens are streamed in the plain style and B's
    plain tokens can pick up A's ``dim cyan italic``.  Tokens must stay
    attributed to the channel their own thread is in.
    """
    buf = TtyStringIO()
    printer = ConsolePrinter(file=buf)
    a_is_thinking = threading.Event()
    b_left_thinking = threading.Event()
    tokens = 20

    def thinking_agent() -> None:
        printer.thinking_callback(True)
        a_is_thinking.set()
        assert b_left_thinking.wait(timeout=30)
        for _ in range(tokens):
            printer.token_callback("a")

    def plain_agent() -> None:
        assert a_is_thinking.wait(timeout=30)
        printer.thinking_callback(False)
        for _ in range(tokens):
            printer.token_callback("b")
        b_left_thinking.set()

    _run_threads([
        threading.Thread(target=thinking_agent),
        threading.Thread(target=plain_agent),
    ])

    text = buf.getvalue()
    assert text.count(f"{_THINKING_ANSI}a") == tokens, "thinking tokens lost their style"
    assert f"{_THINKING_ANSI}b" not in text, "plain tokens were styled as thinking"


def test_reset_does_not_clear_another_threads_block_state() -> None:
    """One agent starting a new turn must not restyle another's tokens.

    ``reset()`` is called per turn.  With the block type shared, a
    sub-agent starting its next turn wiped the thinking state of every
    other sub-agent mid-block; only the cursor position, which really is
    shared, may be reset globally.
    """
    buf = TtyStringIO()
    printer = ConsolePrinter(file=buf)
    a_is_thinking = threading.Event()
    b_has_reset = threading.Event()
    tokens = 10

    def thinking_agent() -> None:
        printer.thinking_callback(True)
        a_is_thinking.set()
        assert b_has_reset.wait(timeout=30)
        for _ in range(tokens):
            printer.token_callback("a")

    def resetting_agent() -> None:
        assert a_is_thinking.wait(timeout=30)
        printer.reset()
        assert printer._bash_streamed is False
        assert printer._current_block_type == ""
        assert printer._mid_line is False
        b_has_reset.set()

    _run_threads([
        threading.Thread(target=thinking_agent),
        threading.Thread(target=resetting_agent),
    ])

    assert buf.getvalue().count(f"{_THINKING_ANSI}a") == tokens


def test_raw_writes_never_land_inside_another_threads_panel(
    fast_thread_switching: None,
) -> None:
    """A panel rendered by one thread is never split by another's raw write.

    ``bash_stream`` and ``tool_result`` bodies bypass rich and write
    straight to the output stream, and ``_flush_newline`` decides whether
    a newline is needed from the shared ``_mid_line`` flag.  Without a
    lock, a raw write can land between the flag being consulted and the
    panel being rendered, producing a line that is half bash output and
    half panel border.
    """
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)
    cycles = 400

    def panel_agent() -> None:
        for _ in range(cycles):
            printer.print("Bash", type="tool_call", tool_input={"command": "panel"})

    def raw_agent() -> None:
        for _ in range(cycles):
            printer.print("rawtext", type="bash_stream")

    _run_threads([
        threading.Thread(target=panel_agent),
        threading.Thread(target=raw_agent),
    ])

    mixed = [
        line
        for line in buf.getvalue().splitlines()
        if "rawtext" in line and ("╭" in line or "╰" in line or "│" in line)
    ]
    assert not mixed, f"raw output merged into a panel line: {mixed[:3]}"


def test_usage_offsets_are_not_shared_between_sub_agents() -> None:
    """Each sub-agent's result panel must carry its OWN accumulated totals.

    ``RelentlessAgent`` snapshots the totals it has already spent into
    ``tokens_offset`` / ``budget_offset`` / ``steps_offset`` at every
    sub-session start, and ``ConsolePrinter`` adds them back when it
    renders the result panel.  With one shared copy per printer object,
    whichever parallel sub-agent wrote last decides what every sibling's
    panel reports — a total that belongs to a different task.
    """
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)
    first_snapshot_taken = threading.Event()
    second_snapshot_taken = threading.Event()

    def early_agent() -> None:
        printer.tokens_offset = 100
        printer.budget_offset = 1.0
        printer.steps_offset = 10
        first_snapshot_taken.set()
        assert second_snapshot_taken.wait(timeout=30)
        printer.print(
            "early done",
            type="result",
            total_tokens=10,
            cost="$0.1000",
            step_count=1,
        )

    def late_agent() -> None:
        assert first_snapshot_taken.wait(timeout=30)
        printer.tokens_offset = 200
        printer.budget_offset = 2.0
        printer.steps_offset = 20
        second_snapshot_taken.set()

    _run_threads([
        threading.Thread(target=early_agent),
        threading.Thread(target=late_agent),
    ])

    text = buf.getvalue()
    assert "tokens=110" in text, f"the sibling's token offset won:\n{text}"
    assert "cost=$1.1000" in text, f"the sibling's budget offset won:\n{text}"
    assert "steps=11" in text, f"the sibling's step offset won:\n{text}"


def test_usage_offsets_start_at_zero_in_every_thread() -> None:
    """A sub-agent that never snapshots must report only its own usage.

    Per-thread state must not mean per-thread *garbage*: a thread that
    has written no offset sees the same zeroes a fresh printer has.
    """
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)
    printer.tokens_offset = 500
    printer.budget_offset = 5.0
    printer.steps_offset = 50
    seen: list[tuple[int, float, int]] = []

    def fresh_agent() -> None:
        seen.append(
            (printer.tokens_offset, printer.budget_offset, printer.steps_offset)
        )
        printer.print(
            "fresh done", type="result", total_tokens=7, cost="$0.2000", step_count=2
        )

    _run_threads([threading.Thread(target=fresh_agent)])

    assert seen == [(0, 0.0, 0)]
    text = buf.getvalue()
    assert "tokens=7" in text
    assert "cost=$0.2000" in text
    assert "steps=2" in text


def test_usage_offsets_survive_a_reset_within_the_same_thread() -> None:
    """The offsets belong to the sub-session, not to a single print call.

    ``reset()`` runs once per turn while a sub-session keeps spending,
    so clearing the snapshot there would make the panel under-report
    everything the earlier turns of that session cost.
    """
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)
    printer.tokens_offset = 100
    printer.budget_offset = 1.0
    printer.steps_offset = 10

    printer.reset()
    printer.print(
        "same thread", type="result", total_tokens=10, cost="$0.1000", step_count=1
    )

    text = buf.getvalue()
    assert "tokens=110" in text
    assert "cost=$1.1000" in text
    assert "steps=11" in text


def test_message_result_panel_uses_this_threads_token_offset() -> None:
    """The Claude-SDK message path reads the same per-thread snapshot."""
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)
    ready = threading.Event()
    rendered = threading.Event()

    class _ResultMessage:
        """A real Claude-SDK-shaped result message (duck-typed by the printer)."""

        def __init__(self, result: str) -> None:
            self.result = result

    def early_agent() -> None:
        printer.tokens_offset = 100
        printer.budget_offset = 1.0
        ready.set()
        assert rendered.wait(timeout=30)
        printer.print(
            _ResultMessage("early message"),
            type="message",
            budget_used=0.1,
            total_tokens_used=10,
        )

    def late_agent() -> None:
        assert ready.wait(timeout=30)
        printer.tokens_offset = 900
        printer.budget_offset = 9.0
        rendered.set()

    _run_threads([
        threading.Thread(target=early_agent),
        threading.Thread(target=late_agent),
    ])

    text = buf.getvalue()
    assert "tokens=110" in text, f"the sibling's token offset won:\n{text}"
    assert "cost=$1.1000" in text, f"the sibling's budget offset won:\n{text}"
