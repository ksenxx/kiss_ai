# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt (kiss/ui/cli): end-to-end regression tests for real defects.

Each test class reproduces one defect found in ``src/kiss/ui/cli/``
through the REAL code paths (no mocks / patches / fakes):

1. ``_InputBox._open_completion_menu`` invoked ``completer_fn``
   unguarded: a completer that raises crashed the whole REPL stdin
   pump on Tab, while the typing path (``_refresh_typing_menu``)
   explicitly catches completer exceptions ("completer must not break
   editing").  The Tab paths must be equally resilient.

2. The anchored REPL's plain-lines history file corrupted multi-line
   entries: ``_save_history_lines`` wrote the raw ``\\n`` bytes and
   ``_load_history_lines`` split per physical line, so one multi-line
   instruction (Shift+Enter) reloaded as several bogus fragments.

3. ``_EventDispatcher`` missed the codebase's own null-coercion
   convention (review #36 hardening) in two spots: an ``askUser``
   event with ``question: null`` queued the literal string ``"None"``,
   and a ``result`` / ``usage_info`` event with null numeric totals
   raised ``TypeError`` inside the printer — in the live client the
   loop thread swallows the exception, so the user's final Result
   panel silently vanished.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

from kiss.core.print_to_console import ConsolePrinter
from kiss.ui.cli.cli_client import _EventDispatcher
from kiss.ui.cli.cli_repl import _load_history_lines, _save_history_lines
from kiss.ui.cli.cli_steering import _InputBox


class _SinkOut(io.StringIO):
    """Real writable text stream for the box's terminal output."""

    def flush(self) -> None:  # StringIO.flush is a no-op already
        return


def _make_box() -> _InputBox:
    """Build a real ``_InputBox`` (no PTY / termios needed for feed)."""
    return _InputBox(threading.RLock(), _SinkOut())


def _feed(box: _InputBox, data: bytes) -> list[str]:
    """Feed raw bytes into the box and return the submitted lines."""
    submitted: list[str] = []
    box.feed(data, submitted.append, lambda: None, on_eof=lambda: None)
    return submitted


class TestTabWithRaisingCompleter:
    """Defect 1: Tab crashed the REPL when the completer raised."""

    def test_tab_does_not_propagate_completer_exception(self) -> None:
        box = _make_box()

        def raising_completer(buf: str) -> list[str]:
            raise RuntimeError("completer boom")

        box.completer_fn = raising_completer
        # The typing path is documented as resilient and must stay so.
        _feed(box, b"a")
        assert box.buf == "a"
        # Tab must be exactly as resilient: no exception, menu closed,
        # buffer untouched.
        _feed(box, b"\t")
        assert box.buf == "a"
        assert not box._menu_open

    def test_single_item_requery_tab_survives_raising_completer(self) -> None:
        """The Tab-on-single-item re-query path must be guarded too."""
        box = _make_box()
        # A healthy completer opens a single-item preview menu while
        # typing (the real "complete while typing" path).
        box.completer_fn = lambda buf: [buf + "x"]
        _feed(box, b"a")
        assert box._menu_open
        assert box._menu_items == ["ax"]
        # The backend starts failing before the user presses Tab
        # (e.g. the history DB went away).  Tab re-queries the
        # completer before accepting — it must not crash the pump.
        def raising_completer(buf: str) -> list[str]:
            raise RuntimeError("completer boom")

        box.completer_fn = raising_completer
        _feed(box, b"\t")
        assert box.buf == "a"
        assert not box._menu_open

    def test_healthy_completer_still_opens_menu_on_tab(self) -> None:
        """The fix must not break normal Tab completion."""
        box = _make_box()
        box.completer_fn = lambda buf: [buf + "x", buf + "y"]
        _feed(box, b"a")
        # Close the typing-preview menu, then re-open via Tab.
        box._reset_completion_state()
        _feed(box, b"\t")
        assert box._menu_open
        assert box._menu_repls == ["ax", "ay"]


class TestAnchoredHistoryMultilineRoundTrip:
    """Defect 2: multi-line history entries fragmented on reload."""

    def test_multiline_entry_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "hist"
        entries = [
            "single line",
            "multi\nline entry",
            "ends with backslash \\",
            "literal \\n stays two chars",
        ]
        _save_history_lines(path, entries)
        assert _load_history_lines(path) == entries

    def test_repeated_save_load_cycles_are_stable(self, tmp_path: Path) -> None:
        path = tmp_path / "hist"
        entries = ["a\nb", "c\\d", "plain"]
        for _ in range(3):
            _save_history_lines(path, entries)
            entries = _load_history_lines(path)
        assert entries == ["a\nb", "c\\d", "plain"]

    def test_legacy_plain_file_loads_verbatim(self, tmp_path: Path) -> None:
        """Pre-fix files (no header) must NOT be unescape-mangled."""
        path = tmp_path / "hist"
        path.write_text(
            "_HiStOrY_V2_\nC:\\new\\path\nplain line\n\n", encoding="utf-8",
        )
        assert _load_history_lines(path) == ["C:\\new\\path", "plain line"]

    def test_empty_history_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "hist"
        _save_history_lines(path, [])
        assert _load_history_lines(path) == []

    def test_cap_keeps_last_1000_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "hist"
        entries = [f"task {i}" for i in range(1005)]
        _save_history_lines(path, entries)
        loaded = _load_history_lines(path)
        assert len(loaded) == 1000
        assert loaded[0] == "task 5"
        assert loaded[-1] == "task 1004"

    def test_missing_file_yields_empty_history(self, tmp_path: Path) -> None:
        assert _load_history_lines(tmp_path / "nope") == []


class TestDispatcherNullCoercion:
    """Defect 3: null-coercion gaps in ``_EventDispatcher``."""

    def _dispatcher(self) -> _EventDispatcher:
        return _EventDispatcher(ConsolePrinter(), tab_id="tab1")

    def test_askuser_null_question_becomes_empty_string(self) -> None:
        d = self._dispatcher()
        d.dispatch({"type": "askUser", "question": None, "tabId": "tab1"})
        assert d.ask_user_q.get_nowait() == ""

    def test_askuser_missing_question_becomes_empty_string(self) -> None:
        d = self._dispatcher()
        d.dispatch({"type": "askUser", "tabId": "tab1"})
        assert d.ask_user_q.get_nowait() == ""

    def test_askuser_real_question_passes_through(self) -> None:
        d = self._dispatcher()
        d.dispatch(
            {"type": "askUser", "question": "Deploy?", "tabId": "tab1"},
        )
        assert d.ask_user_q.get_nowait() == "Deploy?"

    def test_result_with_null_totals_still_renders(self, capsys) -> None:
        d = self._dispatcher()
        # Pre-fix this raised TypeError (None + int) inside the
        # printer; in the live client the loop thread swallowed it and
        # the user's Result panel silently vanished.
        d.dispatch({
            "type": "result",
            "text": "bughunt-cli-result-body",
            "total_tokens": None,
            "cost": None,
            "step_count": None,
            "tabId": "tab1",
        })
        out = capsys.readouterr().out
        assert "bughunt-cli-result-body" in out

    def test_usage_info_with_null_totals_does_not_raise(self) -> None:
        d = self._dispatcher()
        d.dispatch({
            "type": "usage_info",
            "text": "usage",
            "total_tokens": None,
            "cost": None,
            "total_steps": None,
            "tabId": "tab1",
        })

    def test_result_with_real_totals_unchanged(self, capsys) -> None:
        d = self._dispatcher()
        d.dispatch({
            "type": "result",
            "text": "normal-result",
            "total_tokens": 12,
            "cost": "$0.10",
            "step_count": 3,
            "tabId": "tab1",
        })
        out = capsys.readouterr().out
        assert "normal-result" in out
