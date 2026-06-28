# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``/resume --limit N`` command-line flag.

The ``/resume`` slash command used to hard-code the recent-chats
listing to the latest 20 sessions.  Users can now override the limit
with ``--limit N`` (or ``--limit=N``), e.g. ``/resume --limit 5`` to
see only the five most recent chats.

These tests exercise the argument parser, the ``_print_recent_chats``
helper with custom limits, the slash-command handler in the in-process
REPL (:mod:`cli_repl`), and the client-mode slash handler
(:mod:`cli_client`) using the real SQLite-backed persistence layer —
no mocks, patches, or test doubles.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.cli_helpers import (
    DEFAULT_RECENT_CHATS_LIMIT,
    _parse_resume_arg,
    _print_recent_chats,
)
from kiss.agents.sorcar.persistence import _add_task


class TestParseResumeArg:
    """``_parse_resume_arg`` handles every documented argument shape."""

    def test_empty_arg_defaults_to_listing_with_default_limit(self) -> None:
        chat_id, limit = _parse_resume_arg("")
        assert chat_id == ""
        assert limit == DEFAULT_RECENT_CHATS_LIMIT

    def test_bare_chat_id_returned_with_default_limit(self) -> None:
        chat_id, limit = _parse_resume_arg("deadbeef" * 4)
        assert chat_id == "deadbeef" * 4
        assert limit == DEFAULT_RECENT_CHATS_LIMIT

    def test_limit_flag_with_space_separator(self) -> None:
        chat_id, limit = _parse_resume_arg("--limit 5")
        assert chat_id == ""
        assert limit == 5

    def test_limit_flag_with_equals_separator(self) -> None:
        chat_id, limit = _parse_resume_arg("--limit=42")
        assert chat_id == ""
        assert limit == 42

    def test_chat_id_with_limit_is_accepted(self) -> None:
        chat_id, limit = _parse_resume_arg("abc --limit 3")
        assert chat_id == "abc"
        assert limit == 3

    def test_limit_before_chat_id_is_accepted(self) -> None:
        chat_id, limit = _parse_resume_arg("--limit 7 abc")
        assert chat_id == "abc"
        assert limit == 7

    def test_missing_limit_value_raises(self) -> None:
        with pytest.raises(ValueError, match="--limit"):
            _parse_resume_arg("--limit")

    def test_non_integer_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _parse_resume_arg("--limit abc")

    def test_zero_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _parse_resume_arg("--limit 0")

    def test_negative_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _parse_resume_arg("--limit -3")

    def test_two_bare_words_rejected(self) -> None:
        with pytest.raises(ValueError, match="unexpected"):
            _parse_resume_arg("abc def")


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved: tuple[Path, sqlite3.Connection | None, Path] = (
            th._DB_PATH, th._db_conn, th._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _set_timestamp(self, task_id: str, ts: float) -> None:
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET timestamp = ? WHERE id = ?",
                (ts, task_id),
            )
            db.commit()

    def _create_chats(self, n: int) -> list[str]:
        """Create *n* distinct chats with strictly increasing timestamps."""
        base = time.time()
        chat_ids: list[str] = []
        for i in range(n):
            tid, cid = _add_task(f"task in chat {i}")
            self._set_timestamp(tid, base - (n - i) * 1.0)
            chat_ids.append(cid)
        return chat_ids


class TestPrintRecentChatsLimitParam(_TempDbTestBase):
    """``_print_recent_chats`` respects the ``limit`` keyword argument."""

    def test_explicit_limit_caps_listing_below_default(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        chat_ids = self._create_chats(15)

        _print_recent_chats(limit=5)
        out = capsys.readouterr().out

        printed = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed) == 5
        # Only the 5 most recently created chats are shown.
        assert set(printed) == set(chat_ids[-5:])

    def test_explicit_limit_above_count_prints_all(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        chat_ids = self._create_chats(3)

        _print_recent_chats(limit=100)
        out = capsys.readouterr().out

        printed = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed) == 3
        assert set(printed) == set(chat_ids)

    def test_default_limit_still_caps_at_20(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        chat_ids = self._create_chats(25)

        _print_recent_chats()
        out = capsys.readouterr().out

        printed = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed) == 20
        assert set(printed) == set(chat_ids[-20:])


def _make_recording_agent() -> Any:
    """Build a :class:`ChatSorcarAgent`-shaped recorder.

    Subclassing the real class is required because ``_handle_resume``
    enforces an ``isinstance(agent, ChatSorcarAgent)`` check; this
    subclass overrides ``__init__`` to skip the heavy real
    initialisation while keeping the type identity intact, and
    records every ``resume_chat_by_id`` call for assertions.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    class _Agent(ChatSorcarAgent):  # type: ignore[misc]
        def __init__(self) -> None:
            self.resumed: list[str] = []

        def resume_chat_by_id(self, chat_id: str) -> None:  # type: ignore[override]
            self.resumed.append(chat_id)

    return _Agent()


class TestHandleResumeRespectsLimit(_TempDbTestBase):
    """``cli_repl._handle_resume`` parses ``--limit`` and uses it."""

    def test_no_arg_uses_default_limit(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kiss.agents.sorcar.cli_repl import _handle_resume

        chat_ids = self._create_chats(22)
        agent = _make_recording_agent()

        _handle_resume(agent, "")
        out = capsys.readouterr().out

        printed = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed) == DEFAULT_RECENT_CHATS_LIMIT == 20
        assert set(printed) == set(chat_ids[-20:])
        assert agent.resumed == []

    def test_limit_flag_overrides_default(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kiss.agents.sorcar.cli_repl import _handle_resume

        chat_ids = self._create_chats(22)
        agent = _make_recording_agent()

        _handle_resume(agent, "--limit 4")
        out = capsys.readouterr().out

        printed = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed) == 4
        assert set(printed) == set(chat_ids[-4:])
        assert agent.resumed == []

    def test_limit_equals_form_overrides_default(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kiss.agents.sorcar.cli_repl import _handle_resume

        chat_ids = self._create_chats(6)
        agent = _make_recording_agent()

        _handle_resume(agent, "--limit=2")
        out = capsys.readouterr().out

        printed = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed) == 2
        assert set(printed) == set(chat_ids[-2:])

    def test_chat_id_still_resumes_and_ignores_limit(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kiss.agents.sorcar.cli_repl import _handle_resume

        agent = _make_recording_agent()

        _handle_resume(agent, "abc123 --limit 5")
        out = capsys.readouterr().out

        # The listing branch must not have run.
        assert "Chat ID:" not in out
        assert "Resumed chat abc123." in out
        assert agent.resumed == ["abc123"]

    def test_invalid_limit_prints_error_and_does_not_resume(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kiss.agents.sorcar.cli_repl import _handle_resume

        agent = _make_recording_agent()

        _handle_resume(agent, "--limit notanint")
        out = capsys.readouterr().out

        assert "Invalid /resume argument" in out
        assert agent.resumed == []


class _RecordingClient:
    """Stub client capturing ``send`` payloads for cli_client handler."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

        class _Dispatcher:
            def __init__(self) -> None:
                self.chat_id = ""

        self.dispatcher = _Dispatcher()

    def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class TestClientHandleResumeRespectsLimit(_TempDbTestBase):
    """``cli_client._handle_client_slash`` honours ``--limit``."""

    def test_no_arg_lists_default_limit(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kiss.agents.sorcar.cli_client import _handle_client_slash

        chat_ids = self._create_chats(25)
        client = _RecordingClient()

        keep_repl = _handle_client_slash(client, "/resume")  # type: ignore[arg-type]
        out = capsys.readouterr().out

        assert keep_repl is False
        assert client.sent == []
        printed = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed) == DEFAULT_RECENT_CHATS_LIMIT == 20
        assert set(printed) == set(chat_ids[-20:])

    def test_limit_flag_caps_client_listing(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kiss.agents.sorcar.cli_client import _handle_client_slash

        chat_ids = self._create_chats(10)
        client = _RecordingClient()

        _handle_client_slash(client, "/resume --limit 3")  # type: ignore[arg-type]
        out = capsys.readouterr().out

        printed = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed) == 3
        assert set(printed) == set(chat_ids[-3:])
        assert client.sent == []

    def test_chat_id_sends_resume_session_and_ignores_limit(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kiss.agents.sorcar.cli_client import _handle_client_slash

        client = _RecordingClient()

        _handle_client_slash(client, "/resume abcdef --limit 5")  # type: ignore[arg-type]
        out = capsys.readouterr().out

        assert client.sent == [{"type": "resumeSession", "chatId": "abcdef"}]
        assert client.dispatcher.chat_id == "abcdef"
        assert "Resumed chat abcdef." in out
        # No listing branch should have run.
        assert "Chat ID:" not in out

    def test_invalid_limit_prints_error_and_does_not_send(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kiss.agents.sorcar.cli_client import _handle_client_slash

        client = _RecordingClient()

        keep_repl = _handle_client_slash(client, "/resume --limit nope")  # type: ignore[arg-type]
        out = capsys.readouterr().out

        assert keep_repl is False
        assert client.sent == []
        assert "Invalid /resume argument" in out
