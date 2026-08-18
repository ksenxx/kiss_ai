# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Hermes-gateway behaviours of ChannelRunner.

Covers persistent state (including schema validation of malformed
files), the tick lock and its serialization with the pairing admin,
the circuit breaker, the delivery ledger with redelivery, DM pairing
(including the closed allow set and the ``--approve`` /
``--list-pending`` CLI flow), per-channel cursor persistence,
per-channel model overrides and omission-sentinel CLI defaults, the
typing-indicator hook, thread-continuation selection,
continuation-failure cursor retention with follow-up retry, and
``updated_at``-based thread pruning with rotation — all with real
in-test backend classes and real state files (no mocks or patches).
Paths that would launch a daemon task (and hence a real LLM) are
exercised through the factored helpers instead.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import (
    ChannelConfig,
    ChannelRunner,
    ToolMethodBackend,
    _handle_pairing_admin,
    channel_state_lock,
    default_channel_state,
    derive_state_path,
    load_channel_state,
    resolve_channel_overrides,
    sanitize_state_component,
    save_channel_state,
)
from kiss.agents.third_party_agents._channel_cli import (
    _build_arg_parser,
    _build_run_kwargs,
)
from kiss.agents.third_party_agents.telegram_agent import TelegramAgent
from kiss.agents.third_party_agents.telegram_agent import main as telegram_main
from kiss.core import config as config_module
from kiss.core.models.model_info import get_default_model


class RecordingBackend(ToolMethodBackend):
    """In-memory channel backend that records every interaction.

    A real class defined for the tests (not a mock of KISS code): it
    implements the backend protocol with deterministic in-memory data.
    """

    def __init__(
        self,
        messages: list[dict[str, Any]] | None = None,
        connect_ok: bool = True,
        send_fail_times: int = 0,
        cursor: str = "",
    ) -> None:
        self.messages = messages or []
        self.connect_ok = connect_ok
        self.send_fail_times = send_fail_times
        self.cursor = cursor
        self.sent: list[tuple[str, str, str]] = []
        self.typing: list[tuple[str, str]] = []
        self.connect_calls = 0
        self.poll_calls = 0
        self.polled_oldest: list[str] = []
        self._connection_info = "recording backend"

    def connect(self) -> bool:
        """Record the connect attempt and return the configured result."""
        self.connect_calls += 1
        return self.connect_ok

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 50
    ) -> tuple[list[dict[str, Any]], str]:
        """Return the configured message list and cursor, recording *oldest*."""
        self.poll_calls += 1
        self.polled_oldest.append(oldest)
        return list(self.messages), self.cursor

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Record the send, raising for the first *send_fail_times* calls."""
        if self.send_fail_times > 0:
            self.send_fail_times -= 1
            raise ConnectionError("simulated send failure")
        self.sent.append((channel_id, text, thread_ts))

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """A message is from the bot when its ``bot`` flag is set."""
        return bool(msg.get("bot"))

    def send_typing(self, channel_id: str, thread_ts: str = "") -> None:
        """Record the typing indicator call."""
        self.typing.append((channel_id, thread_ts))


class LedgerCheckingBackend(RecordingBackend):
    """Backend whose send asserts the ledger was persisted before sending."""

    def __init__(self, state_path: Path) -> None:
        super().__init__()
        self._state_path = state_path
        self.ledger_sizes_at_send: list[int] = []

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Record how many ledger entries were on disk at send time."""
        on_disk = json.loads(self._state_path.read_text(encoding="utf-8"))
        self.ledger_sizes_at_send.append(len(on_disk.get("ledger", [])))
        super().send_message(channel_id, text, thread_ts)


class ThreadBackend(RecordingBackend):
    """Recording backend that also supports thread polling."""

    def __init__(
        self,
        thread_replies: dict[str, list[dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.thread_replies = thread_replies or {}
        self.thread_polls: list[str] = []

    def poll_thread_messages(
        self, channel_id: str, thread_ts: str, oldest: str, limit: int = 100
    ) -> tuple[list[dict[str, Any]], str]:
        """Return the configured replies for *thread_ts*."""
        self.thread_polls.append(thread_ts)
        return list(self.thread_replies.get(thread_ts, [])), ""


class RaisingTypingBackend(RecordingBackend):
    """Backend whose typing indicator always fails."""

    def send_typing(self, channel_id: str, thread_ts: str = "") -> None:
        """Simulate a platform error in the typing indicator."""
        raise ConnectionError("typing not supported right now")


class LaunchOutcomeRunner(ChannelRunner):
    """Runner whose task launch is a real in-test implementation.

    Mirrors the ``_launch_task`` contract without a daemon: a
    configured error raises before any thread state is stored, and a
    success stores the thread state and returns a YAML task result —
    exactly what the real launcher does.
    """

    def __init__(
        self, *args: Any, launch_error: Exception | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.launch_error = launch_error
        self.launched_prompts: list[str] = []

    def _launch_task(
        self, channel_id: str, thread_ts: str, prompt: str, last_reply_ts: str
    ) -> str:
        """Record the launch; raise the configured error or succeed."""
        self.launched_prompts.append(prompt)
        if self.launch_error is not None:
            raise self.launch_error
        self._store_thread_state(thread_ts, "chat-cont", last_reply_ts)
        return "success: true\nsummary: continuation done\n"


def _make_runner(
    backend: Any,
    state_path: Path | None,
    **kwargs: Any,
) -> ChannelRunner:
    """Build a ChannelRunner wired to *backend* with test defaults."""
    return ChannelRunner(
        backend=backend,
        channel_name="",
        agent_name="Hermes Test Agent",
        state_path=state_path,
        **kwargs,
    )


class TestChannelState:
    """Persistent-state schema, atomicity, and permissions."""

    def test_save_is_atomic_and_0600(self, tmp_path: Path) -> None:
        """save_channel_state leaves a 0600 file and no temp leftovers."""
        path = tmp_path / "state" / "channel_state_default_gen.json"
        state = default_channel_state()
        state["failures"] = 3
        save_channel_state(path, state)
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600
        assert load_channel_state(path)["failures"] == 3

    def test_load_missing_returns_default_schema(self, tmp_path: Path) -> None:
        """A missing file yields the full default schema."""
        state = load_channel_state(tmp_path / "nope.json")
        assert state == default_channel_state()

    def test_load_corrupt_returns_default_schema(self, tmp_path: Path) -> None:
        """A corrupt file yields the default schema instead of raising."""
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_channel_state(path) == default_channel_state()

    def test_load_merges_missing_keys(self, tmp_path: Path) -> None:
        """Partial state files are merged over the default schema."""
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"failures": 2}), encoding="utf-8")
        state = load_channel_state(path)
        assert state["failures"] == 2
        assert state["threads"] == {}
        assert state["ledger"] == []

    def test_sanitize_state_component(self) -> None:
        """Non-alphanumeric runs collapse to underscores."""
        assert sanitize_state_component("my channel#1!") == "my_channel_1_"
        assert sanitize_state_component("") == "default"

    def test_derive_state_path_uses_module_config(self) -> None:
        """Agents with a module-level _config store state next to config.json."""
        digest = hashlib.sha256(b"ws 1\x00gen#eral").hexdigest()[:10]
        path = derive_state_path(TelegramAgent, "Telegram", "ws 1", "gen#eral")
        assert path.name == f"channel_state_ws_1_gen_eral_{digest}.json"
        assert path.parent.name == "telegram"

    def test_derive_state_path_fallback_without_config(self) -> None:
        """Classes without a module _config fall back to channel_state dir."""
        digest = hashlib.sha256(b"default\x00general").hexdigest()[:10]
        path = derive_state_path(RecordingBackend, "My Agent", "default", "general")
        assert path.parent.name == "My_Agent"
        assert path.parent.parent.name == "channel_state"
        assert path.name == f"channel_state_default_general_{digest}.json"

    def test_derive_state_path_no_collision_after_sanitization(self) -> None:
        """Raw pairs whose sanitized forms coincide get distinct paths."""
        colliding_pairs = [(("a-b", "c/d"), ("a_b", "c?d")), (("w s", "ch#1"), ("w_s", "ch_1"))]
        for (ws1, ch1), (ws2, ch2) in colliding_pairs:
            path1 = derive_state_path(TelegramAgent, "Telegram", ws1, ch1)
            path2 = derive_state_path(TelegramAgent, "Telegram", ws2, ch2)
            assert path1 != path2

    def test_derive_state_path_long_names_truncated(self) -> None:
        """Very long workspace/channel names yield a bounded file name."""
        path = derive_state_path(TelegramAgent, "Telegram", "w" * 100, "c" * 100)
        assert len(path.name) < 100
        assert path.name.startswith("channel_state_" + "w" * 24 + "_" + "c" * 24 + "_")

    def test_save_survives_stale_fixed_name_tmp(self, tmp_path: Path) -> None:
        """A leftover artifact named <state>.tmp cannot clobber a save.

        The save helper uses a unique mkstemp name per writer, so even a
        directory squatting on the old fixed ``.tmp`` name is harmless
        and concurrent savers cannot overwrite each other's temp file.
        """
        path = tmp_path / "state.json"
        path.with_suffix(".tmp").mkdir(parents=True)
        state = default_channel_state()
        state["failures"] = 7
        save_channel_state(path, state)
        assert load_channel_state(path)["failures"] == 7
        leftovers = [
            p for p in tmp_path.iterdir() if p.is_file() and p.name.endswith(".tmp")
        ]
        assert leftovers == []


class TestTickLock:
    """The non-blocking per-channel tick lock."""

    def test_second_tick_returns_zero_while_lock_held(self, tmp_path: Path) -> None:
        """run_once returns 0 without connecting when the lock is held."""
        state_path = tmp_path / "channel_state_default_gen.json"
        lock_path = state_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        backend = RecordingBackend()
        runner = _make_runner(backend, state_path)
        with lock_path.open("a+", encoding="utf-8") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert runner.run_once() == 0
        assert backend.connect_calls == 0

    def test_lock_released_after_tick(self, tmp_path: Path) -> None:
        """A finished tick releases the lock so the next tick runs."""
        state_path = tmp_path / "channel_state_default_gen.json"
        backend = RecordingBackend()
        runner = _make_runner(backend, state_path)
        assert runner.run_once() == 0
        assert runner.run_once() == 0
        assert backend.connect_calls == 2


class TestCircuitBreaker:
    """Failure counting, pausing, short-circuiting, and reset."""

    def test_failures_increment_and_pause_after_five(self, tmp_path: Path) -> None:
        """Five consecutive transport failures pause the channel."""
        state_path = tmp_path / "channel_state_default_gen.json"
        backend = RecordingBackend(connect_ok=False)
        runner = _make_runner(backend, state_path)
        for expected in range(1, 6):
            with pytest.raises(RuntimeError, match="Failed to connect"):
                runner.run_once()
            assert load_channel_state(state_path)["failures"] == expected
        assert load_channel_state(state_path)["paused_until"] > time.time()

    def test_paused_channel_short_circuits(self, tmp_path: Path) -> None:
        """A paused channel skips the tick entirely and returns 0."""
        state_path = tmp_path / "channel_state_default_gen.json"
        state = default_channel_state()
        state["paused_until"] = time.time() + 100
        save_channel_state(state_path, state)
        backend = RecordingBackend()
        runner = _make_runner(backend, state_path)
        assert runner.run_once() == 0
        assert backend.connect_calls == 0

    def test_successful_tick_resets_failures(self, tmp_path: Path) -> None:
        """A successful tick resets the consecutive-failure counter."""
        state_path = tmp_path / "channel_state_default_gen.json"
        state = default_channel_state()
        state["failures"] = 3
        save_channel_state(state_path, state)
        backend = RecordingBackend()
        runner = _make_runner(backend, state_path)
        assert runner.run_once() == 0
        assert load_channel_state(state_path)["failures"] == 0

    def test_poll_failure_counts_as_transport_failure(self, tmp_path: Path) -> None:
        """A poll_messages exception increments the failure counter."""

        class PollFailBackend(RecordingBackend):
            """Backend whose poll always raises."""

            def poll_messages(
                self, channel_id: str, oldest: str, limit: int = 50
            ) -> tuple[list[dict[str, Any]], str]:
                """Simulate a transport failure during polling."""
                raise ConnectionError("poll failed")

        state_path = tmp_path / "channel_state_default_gen.json"
        runner = _make_runner(PollFailBackend(), state_path)
        with pytest.raises(ConnectionError):
            runner.run_once()
        assert load_channel_state(state_path)["failures"] == 1


class TestDeliveryLedger:
    """At-least-once reply delivery through the persistent ledger."""

    def test_ledger_recorded_before_send_and_removed_on_success(
        self, tmp_path: Path
    ) -> None:
        """The reply is in the on-disk ledger at send time, gone after."""
        state_path = tmp_path / "channel_state_default_gen.json"
        backend = LedgerCheckingBackend(state_path)
        runner = _make_runner(backend, state_path)
        runner._state = load_channel_state(state_path)
        runner._send_reply("C1", "hello there", "1.0")
        assert backend.ledger_sizes_at_send == [1]
        assert backend.sent == [("C1", "hello there", "1.0")]
        runner._save_state()
        assert load_channel_state(state_path)["ledger"] == []

    def test_failed_send_stays_in_ledger(self, tmp_path: Path) -> None:
        """A reply whose sends all fail stays persisted in the ledger."""
        state_path = tmp_path / "channel_state_default_gen.json"
        backend = RecordingBackend(send_fail_times=2)
        runner = _make_runner(backend, state_path)
        runner._state = load_channel_state(state_path)
        runner._send_reply("C1", "lost reply", "2.0")
        ledger = load_channel_state(state_path)["ledger"]
        assert len(ledger) == 1
        assert ledger[0]["channel_id"] == "C1"
        assert ledger[0]["thread_ts"] == "2.0"
        assert ledger[0]["text"] == "lost reply"
        assert ledger[0]["created"] > 0

    def test_next_tick_redelivers_with_recovered_prefix(self, tmp_path: Path) -> None:
        """The next tick redelivers pending replies with the prefix."""
        state_path = tmp_path / "channel_state_default_gen.json"
        failing = RecordingBackend(send_fail_times=2)
        runner = _make_runner(failing, state_path)
        runner._state = load_channel_state(state_path)
        runner._send_reply("C1", "lost reply", "2.0")

        working = RecordingBackend()
        second = _make_runner(working, state_path)
        assert second.run_once() == 0
        assert working.sent == [("C1", "(recovered reply) lost reply", "2.0")]
        assert load_channel_state(state_path)["ledger"] == []

    def test_redelivery_failure_keeps_entry(self, tmp_path: Path) -> None:
        """A redelivery that fails again keeps the entry for later."""
        state_path = tmp_path / "channel_state_default_gen.json"
        state = default_channel_state()
        state["ledger"] = [
            {"channel_id": "C1", "thread_ts": "3.0", "text": "hi", "created": 1.0}
        ]
        save_channel_state(state_path, state)
        backend = RecordingBackend(send_fail_times=2)
        runner = _make_runner(backend, state_path)
        assert runner.run_once() == 0
        assert load_channel_state(state_path)["ledger"] == state["ledger"]

    def test_stateless_send_reply_unchanged(self, tmp_path: Path) -> None:
        """Without a state path _send_reply behaves exactly as before."""
        backend = RecordingBackend()
        runner = _make_runner(backend, None)
        runner._send_reply("C1", "plain", "4.0")
        assert backend.sent == [("C1", "plain", "4.0")]


class TestDmPairing:
    """DM pairing codes, no-spam behaviour, and the approval flow."""

    def _pairing_runner(
        self, backend: RecordingBackend, state_path: Path
    ) -> ChannelRunner:
        return _make_runner(
            backend,
            state_path,
            allow_users=["alice"],
            dm_pairing=True,
            cli_name="kiss-faketest",
        )

    def test_unknown_sender_gets_pairing_code_once(self, tmp_path: Path) -> None:
        """An unapproved sender gets one code; repeats are silent."""
        state_path = tmp_path / "channel_state_default_gen.json"
        msg = {"user": "bob", "text": "hi", "ts": "10.0"}
        backend = RecordingBackend(messages=[msg])
        runner = self._pairing_runner(backend, state_path)
        assert runner.run_once() == 0
        assert len(backend.sent) == 1
        channel_id, text, thread_ts = backend.sent[0]
        assert thread_ts == "10.0"
        assert "not authorized" in text
        assert "kiss-faketest --approve " in text
        state = load_channel_state(state_path)
        code = state["pending_pairing"]["bob"]["code"]
        assert len(code) == 8
        assert text.endswith(f"--approve {code}")

        # Second tick: bob is still pending, no new pairing reply.
        assert runner.run_once() == 0
        assert len(backend.sent) == 1
        assert load_channel_state(state_path)["pending_pairing"]["bob"]["code"] == code

    def test_pairing_disabled_skips_silently(self, tmp_path: Path) -> None:
        """Without --pairing, unapproved senders are skipped silently."""
        state_path = tmp_path / "channel_state_default_gen.json"
        msg = {"user": "bob", "text": "hi", "ts": "10.0"}
        backend = RecordingBackend(messages=[msg])
        runner = _make_runner(backend, state_path, allow_users=["alice"])
        assert runner.run_once() == 0
        assert backend.sent == []
        assert load_channel_state(state_path)["pending_pairing"] == {}

    def test_effective_allow_unions_approved_users(self, tmp_path: Path) -> None:
        """allow_users is extended by the state's approved_users."""
        backend = RecordingBackend()
        runner = _make_runner(backend, tmp_path / "s.json", allow_users=["alice"])
        runner._state = default_channel_state()
        runner._state["approved_users"] = ["bob"]
        assert runner._effective_allow() == {"alice", "bob"}

    def test_no_allow_list_means_everyone_allowed(self, tmp_path: Path) -> None:
        """Without allow_users (and no pairing) the allow set is None."""
        runner = _make_runner(RecordingBackend(), tmp_path / "s.json")
        runner._state = default_channel_state()
        runner._state["approved_users"] = ["bob"]
        assert runner._effective_allow() is None

    def test_pairing_without_allow_list_is_closed(self, tmp_path: Path) -> None:
        """With pairing on and no allow list, only approved users pass."""
        runner = _make_runner(RecordingBackend(), tmp_path / "s.json", dm_pairing=True)
        runner._state = default_channel_state()
        assert runner._effective_allow() == set()
        runner._state["approved_users"] = ["bob"]
        assert runner._effective_allow() == {"bob"}

    def test_pairing_without_allow_list_sends_code(self, tmp_path: Path) -> None:
        """An unknown sender gets the pairing flow, not a free pass."""
        state_path = tmp_path / "channel_state_default_gen.json"
        msg = {"user": "bob", "text": "hi", "ts": "10.0"}
        backend = RecordingBackend(messages=[msg])
        runner = _make_runner(
            backend, state_path, dm_pairing=True, cli_name="kiss-faketest"
        )
        assert runner.run_once() == 0
        assert len(backend.sent) == 1
        assert "not authorized" in backend.sent[0][1]
        assert "bob" in load_channel_state(state_path)["pending_pairing"]

    def test_pairing_reply_names_cli(self, tmp_path: Path) -> None:
        """The pairing reply names the channel CLI command."""
        runner = _make_runner(
            RecordingBackend(), tmp_path / "s.json", cli_name="kiss-x"
        )
        assert "kiss-x --approve c0dec0de" in runner._pairing_reply_text("c0dec0de")

    def test_pairing_reply_is_complete_shell_command(self, tmp_path: Path) -> None:
        """The suggested command includes --channel and --workspace."""
        runner = ChannelRunner(
            backend=RecordingBackend(),
            channel_name="mychan",
            agent_name="Hermes Test Agent",
            state_path=tmp_path / "s.json",
            workspace="ws2",
            dm_pairing=True,
            cli_name="kiss-telegram",
        )
        text = runner._pairing_reply_text("ab12cd34")
        assert "kiss-telegram --channel mychan --workspace ws2 --approve ab12cd34" in text

    def test_pairing_reply_default_workspace_omits_flag(self, tmp_path: Path) -> None:
        """The default workspace is not spelled out in the command."""
        runner = ChannelRunner(
            backend=RecordingBackend(),
            channel_name="mychan",
            agent_name="Hermes Test Agent",
            state_path=tmp_path / "s.json",
            dm_pairing=True,
            cli_name="kiss-telegram",
        )
        text = runner._pairing_reply_text("ab12cd34")
        assert "kiss-telegram --channel mychan --approve ab12cd34" in text
        assert "--workspace" not in text

    def test_pairing_reply_without_cli_name(self, tmp_path: Path) -> None:
        """Without a CLI name a generic hint is used."""
        runner = _make_runner(RecordingBackend(), tmp_path / "s.json")
        assert "the channel CLI --approve" in runner._pairing_reply_text("c0dec0de")


class TestPairingAdminCli:
    """channel_main --approve / --list-pending driven through a real main()."""

    def _seed_state(self, channel: str, pending: dict[str, Any]) -> Path:
        state_path = derive_state_path(TelegramAgent, "Telegram", "default", channel)
        state = default_channel_state()
        state["pending_pairing"] = pending
        save_channel_state(state_path, state)
        return state_path

    def _run_main(self, argv: list[str]) -> None:
        original = sys.argv
        sys.argv = argv
        try:
            telegram_main()
        finally:
            sys.argv = original

    def test_approve_moves_user_to_approved(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--approve CODE approves the matching pending user."""
        state_path = self._seed_state(
            "pair-approve", {"bob": {"code": "abcd1234", "ts": 1.0}}
        )
        self._run_main(
            ["kiss-telegram", "--channel", "pair-approve", "--approve", "abcd1234"]
        )
        assert "Approved user: bob" in capsys.readouterr().out
        state = load_channel_state(state_path)
        assert state["approved_users"] == ["bob"]
        assert state["pending_pairing"] == {}

    def test_approve_unknown_code_exits_nonzero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--approve with an unknown code prints an error and exits 1."""
        self._seed_state("pair-unknown", {"bob": {"code": "abcd1234", "ts": 1.0}})
        with pytest.raises(SystemExit) as exc_info:
            self._run_main(
                ["kiss-telegram", "--channel", "pair-unknown", "--approve", "ffff0000"]
            )
        assert exc_info.value.code == 1
        assert "no pending pairing request" in capsys.readouterr().out

    def test_approve_requires_channel(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--approve without --channel exits nonzero with an error."""
        with pytest.raises(SystemExit) as exc_info:
            self._run_main(["kiss-telegram", "--approve", "abcd1234"])
        assert exc_info.value.code == 1
        assert "require --channel" in capsys.readouterr().out

    def test_approve_is_serialized_with_tick_lock(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--approve blocks while a tick holds the channel state lock.

        The admin thread must not complete its read-modify-write while
        the lock is held (otherwise a running tick's stale in-memory
        save could overwrite the approval); once the lock is released
        the approval proceeds and is persisted.
        """
        state_path = self._seed_state(
            "pair-lockrace", {"bob": {"code": "abcd1234", "ts": 1.0}}
        )
        admin = threading.Thread(
            target=_handle_pairing_admin,
            args=(TelegramAgent, "Telegram", "default", "pair-lockrace", "abcd1234", False),
        )
        with channel_state_lock(state_path, blocking=False) as lock_fp:
            assert lock_fp is not None
            admin.start()
            deadline = time.time() + 0.5
            while time.time() < deadline:
                assert admin.is_alive(), "admin approve ran while the tick lock was held"
                time.sleep(0.05)
            # Simulate the running tick's save happening under the lock.
            state = load_channel_state(state_path)
            state["failures"] = 1
            save_channel_state(state_path, state)
        admin.join(timeout=10)
        assert not admin.is_alive()
        final = load_channel_state(state_path)
        assert final["approved_users"] == ["bob"]
        assert final["pending_pairing"] == {}
        assert final["failures"] == 1
        assert "Approved user: bob" in capsys.readouterr().out

    def test_list_pending_prints_pairs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--list-pending prints user_id:code pairs and exits cleanly."""
        self._seed_state(
            "pair-list",
            {
                "bob": {"code": "abcd1234", "ts": 1.0},
                "carol": {"code": "beef5678", "ts": 2.0},
            },
        )
        self._run_main(["kiss-telegram", "--channel", "pair-list", "--list-pending"])
        out = capsys.readouterr().out
        assert "bob:abcd1234" in out
        assert "carol:beef5678" in out


class TestModelOverrides:
    """Per-channel model/budget overrides from a real config.json."""

    def test_overrides_applied_when_flags_omitted(self, tmp_path: Path) -> None:
        """channel_model_name/channel_max_budget apply for omitted flags."""
        config = ChannelConfig(tmp_path / "chan", ("token",))
        config.save(
            {
                "token": "x",
                "channel_model_name": "override-model",
                "channel_max_budget": "2.5",
            }
        )
        cfg = config.load()
        assert cfg is not None
        model, budget = resolve_channel_overrides(cfg, None, None, "default-model", 5.0)
        assert model == "override-model"
        assert budget == 2.5

    def test_explicit_cli_values_win(self) -> None:
        """User-passed -m / -b values are never overridden."""
        cfg = {"channel_model_name": "override-model", "channel_max_budget": "2.5"}
        model, budget = resolve_channel_overrides(
            cfg, "user-model", 9.0, "default-model", 5.0
        )
        assert model == "user-model"
        assert budget == 9.0

    def test_explicit_values_equal_to_defaults_win(self) -> None:
        """Explicitly passing the default value blocks config overrides."""
        cfg = {"channel_model_name": "override-model", "channel_max_budget": "2.5"}
        model, budget = resolve_channel_overrides(
            cfg, "default-model", 5.0, "default-model", 5.0
        )
        assert model == "default-model"
        assert budget == 5.0

    def test_omitted_flags_without_config_use_defaults(self) -> None:
        """Omitted flags with no config resolve to the given defaults."""
        assert resolve_channel_overrides(None, None, None, "dm", 5.0) == ("dm", 5.0)
        assert resolve_channel_overrides({}, None, None, "dm", 5.0) == ("dm", 5.0)

    def test_bad_budget_values_ignored(self) -> None:
        """Unparseable or non-positive budget overrides fall back to default."""
        for bad in ("abc", "nan", "inf", "-1", "0"):
            cfg = {"channel_max_budget": bad}
            _, budget = resolve_channel_overrides(cfg, "m", None, "m", 5.0)
            assert budget == 5.0

    def test_none_config_is_tolerated(self) -> None:
        """A missing adapter config leaves explicit CLI values untouched."""
        assert resolve_channel_overrides(None, "m", 5.0, "dm", 9.0) == ("m", 5.0)

    def test_empty_override_resolves_default_model(self) -> None:
        """An empty channel_model_name resolves to the default model."""
        model, _ = resolve_channel_overrides({}, None, 5.0, "default-model", 5.0)
        assert model == "default-model"


class TestCliOmissionSentinels:
    """-m/-b parse to None when omitted; interactive mode resolves them."""

    def test_parser_defaults_are_none(self) -> None:
        """Omitted -m/-b parse as None so omission is detectable."""
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.model_name is None
        assert args.max_budget is None

    def test_explicit_values_parse_verbatim(self) -> None:
        """Explicit -m/-b values (even equal to defaults) are kept."""
        default_budget = config_module.DEFAULT_CONFIG.max_budget
        parser = _build_arg_parser()
        args = parser.parse_args(["-m", get_default_model(), "-b", str(default_budget)])
        assert args.model_name == get_default_model()
        assert args.max_budget == default_budget

    def test_build_run_kwargs_resolves_omitted_flags(self, tmp_path: Path) -> None:
        """Interactive mode resolves None to the real defaults."""
        parser = _build_arg_parser()
        args = parser.parse_args(["-t", "noop", "-w", str(tmp_path)])
        kwargs = _build_run_kwargs(args)
        assert kwargs["model_name"] == get_default_model()
        assert kwargs["max_budget"] == config_module.DEFAULT_CONFIG.max_budget

    def test_build_run_kwargs_keeps_explicit_flags(self, tmp_path: Path) -> None:
        """Interactive mode passes explicit -m/-b through unchanged."""
        parser = _build_arg_parser()
        args = parser.parse_args(
            ["-t", "noop", "-w", str(tmp_path), "-m", "some-model", "-b", "1.25"]
        )
        kwargs = _build_run_kwargs(args)
        assert kwargs["model_name"] == "some-model"
        assert kwargs["max_budget"] == 1.25


class TestTypingIndicator:
    """The best-effort typing-indicator hook."""

    def test_default_send_typing_is_noop_and_not_a_tool(self) -> None:
        """ToolMethodBackend.send_typing exists, no-ops, and is not a tool."""
        backend = ToolMethodBackend()
        backend.send_typing("C1")
        backend.send_typing("C1", "1.0")
        names = [m.__name__ for m in backend.get_tool_methods()]
        assert "send_typing" not in names

    def test_runner_records_typing_call(self, tmp_path: Path) -> None:
        """_notify_typing forwards to the backend's send_typing."""
        backend = RecordingBackend()
        runner = _make_runner(backend, tmp_path / "s.json")
        runner._notify_typing("C1", "1.0")
        assert backend.typing == [("C1", "1.0")]

    def test_typing_failure_is_swallowed(self, tmp_path: Path) -> None:
        """A raising send_typing never breaks the runner."""
        runner = _make_runner(RaisingTypingBackend(), tmp_path / "s.json")
        runner._notify_typing("C1", "1.0")

    def test_backend_without_send_typing_is_tolerated(self, tmp_path: Path) -> None:
        """Backends predating the hook (no send_typing) are fine."""

        class BareBackend:
            """Minimal backend without any typing support."""

        runner = _make_runner(BareBackend(), tmp_path / "s.json")
        runner._notify_typing("C1", "1.0")


class TestThreadContinuations:
    """Thread-continuation selection, pruning, and state continuity."""

    def test_select_thread_followups(self, tmp_path: Path) -> None:
        """Only new, non-bot, allowed, non-empty replies are selected."""
        backend = ThreadBackend()
        runner = _make_runner(backend, tmp_path / "s.json", allow_users=["alice"])
        runner._state = default_channel_state()
        replies: list[dict[str, Any]] = [
            {"user": "alice", "text": "parent", "ts": "100.0"},
            {"user": "alice", "text": "handled question", "ts": "101.0"},
            {"user": "", "bot": True, "text": "bot answer", "ts": "102.0"},
            {"user": "mallory", "text": "not allowed", "ts": "103.0"},
            {"user": "alice", "text": "  ", "ts": "104.0"},
            {"user": "alice", "text": "follow-up A", "ts": "105.0"},
            {"user": "alice", "text": "follow-up B", "ts": "106.0"},
            {"user": "alice", "text": "", "ts": ""},
        ]
        selected = runner._select_thread_followups(replies, "100.0", "101.0")
        assert [m["text"] for m in selected] == ["follow-up A", "follow-up B"]

    def test_followup_during_task_selected_despite_bot_answer(
        self, tmp_path: Path
    ) -> None:
        """A follow-up that arrived mid-task is selected after the bot posts.

        Deduplication is only on last_reply_ts: follow-up B (ts 102)
        arrived while the previous task ran, so it is older than the
        bot's answer (ts 103) but must still be selected — the old
        behaviour excluded it forever.
        """
        runner = _make_runner(ThreadBackend(), tmp_path / "s.json")
        runner._state = default_channel_state()
        replies: list[dict[str, Any]] = [
            {"user": "alice", "text": "question", "ts": "101.0"},
            {"user": "alice", "text": "follow-up B", "ts": "102.0"},
            {"user": "", "bot": True, "text": "answer", "ts": "103.0"},
        ]
        selected = runner._select_thread_followups(replies, "100.0", "101.0")
        assert [m["text"] for m in selected] == ["follow-up B"]

    def test_followups_older_than_last_reply_ts_excluded(self, tmp_path: Path) -> None:
        """Messages at or before last_reply_ts are not selected."""
        runner = _make_runner(ThreadBackend(), tmp_path / "s.json")
        runner._state = default_channel_state()
        replies = [{"user": "alice", "text": "seen already", "ts": "105.0"}]
        assert runner._select_thread_followups(replies, "100.0", "105.0") == []

    def test_threads_capped_at_twenty_per_tick(self, tmp_path: Path) -> None:
        """At most 20 threads are polled per tick, most recently updated first."""
        runner = _make_runner(ThreadBackend(), tmp_path / "s.json")
        runner._state = default_channel_state()
        now = time.time()
        for i in range(25):
            runner._state["threads"][f"thread-{i}"] = {
                "chat_id": f"chat{i}",
                "last_reply_ts": f"{now - i:.4f}",
                "updated_at": now - i,
            }
        selected = runner._threads_to_process()
        assert len(selected) == 20
        assert selected[0] == "thread-0"

    def test_thread_rotation_prevents_starvation(self, tmp_path: Path) -> None:
        """Successive ticks rotate through >20 threads so none starves."""
        runner = _make_runner(ThreadBackend(), tmp_path / "s.json")
        runner._state = default_channel_state()
        now = time.time()
        for i in range(25):
            runner._state["threads"][f"thread-{i}"] = {
                "chat_id": f"chat{i}",
                "last_reply_ts": f"{now - i:.4f}",
                "updated_at": now - i,
            }
        first = runner._threads_to_process()
        assert runner._state["thread_rotation"] == 20
        second = runner._threads_to_process()
        assert set(first) | set(second) == {f"thread-{i}" for i in range(25)}
        assert {f"thread-{i}" for i in range(20, 25)} <= set(second)

    def test_stale_threads_pruned_by_updated_at(self, tmp_path: Path) -> None:
        """Entries with updated_at older than 7 days are pruned.

        The thread keys here are platform message IDs (huge snowflakes
        and ISO strings) that are meaningless as Unix times — only the
        updated_at stamp may decide staleness.
        """
        state_path = tmp_path / "s.json"
        runner = _make_runner(ThreadBackend(), state_path)
        runner._state = default_channel_state()
        now = time.time()
        runner._state["threads"]["1266434234230768927"] = {
            "chat_id": "c1",
            "last_reply_ts": "1266434234230768927",
            "updated_at": now - 60,
        }
        runner._state["threads"]["2026-01-01T00:00:00Z"] = {
            "chat_id": "c2",
            "last_reply_ts": "2026-01-01T00:00:00Z",
            "updated_at": now - 8 * 24 * 3600,
        }
        runner._prune_stale_threads()
        assert runner._threads_to_process() == ["1266434234230768927"]
        assert "2026-01-01T00:00:00Z" not in load_channel_state(state_path)["threads"]

    def test_pruning_runs_without_thread_poll_support(self, tmp_path: Path) -> None:
        """Stale entries are pruned on ticks of poll-less backends too."""
        state_path = tmp_path / "s.json"
        now = time.time()
        state = default_channel_state()
        state["threads"]["fresh"] = {
            "chat_id": "c1",
            "last_reply_ts": "1.0",
            "updated_at": now - 60,
        }
        state["threads"]["stale"] = {
            "chat_id": "c2",
            "last_reply_ts": "2.0",
            "updated_at": now - 8 * 24 * 3600,
        }
        save_channel_state(state_path, state)
        backend = RecordingBackend()  # no poll_thread_messages
        runner = _make_runner(backend, state_path)
        assert runner.run_once() == 0
        threads = load_channel_state(state_path)["threads"]
        assert "fresh" in threads
        assert "stale" not in threads

    def test_thread_state_read_write_helpers(self, tmp_path: Path) -> None:
        """_store_thread_state persists what _stored_chat_id reads back."""
        state_path = tmp_path / "s.json"
        runner = _make_runner(ThreadBackend(), state_path)
        runner._state = load_channel_state(state_path)
        assert runner._stored_chat_id("100.0") == ""
        before = time.time()
        runner._store_thread_state("100.0", "chat-42", "105.0")
        assert runner._stored_chat_id("100.0") == "chat-42"
        entry = load_channel_state(state_path)["threads"]["100.0"]
        assert entry["chat_id"] == "chat-42"
        assert entry["last_reply_ts"] == "105.0"
        assert entry["updated_at"] >= before

    def test_tick_with_no_new_followups_runs_clean(self, tmp_path: Path) -> None:
        """A tick over known threads with no follow-ups launches nothing."""
        state_path = tmp_path / "s.json"
        now = time.time()
        thread_ts = f"{now - 60:.4f}"
        state = default_channel_state()
        state["threads"][thread_ts] = {
            "chat_id": "c1",
            "last_reply_ts": thread_ts,
            "updated_at": now - 60,
        }
        save_channel_state(state_path, state)
        backend = ThreadBackend(
            thread_replies={
                thread_ts: [
                    {"user": "", "bot": True, "text": "answer", "ts": f"{now - 50:.4f}"}
                ]
            }
        )
        runner = _make_runner(backend, state_path)
        assert runner.run_once() == 0
        assert backend.thread_polls == [thread_ts]
        assert backend.sent == []

    def test_thread_poll_failure_skips_thread(self, tmp_path: Path) -> None:
        """A failing poll_thread_messages skips the thread gracefully."""

        class FailingThreadBackend(ThreadBackend):
            """Thread backend whose thread polling raises."""

            def poll_thread_messages(
                self, channel_id: str, thread_ts: str, oldest: str, limit: int = 100
            ) -> tuple[list[dict[str, Any]], str]:
                """Simulate a transport failure during thread polling."""
                raise ConnectionError("thread poll failed")

        state_path = tmp_path / "s.json"
        now = time.time()
        thread_ts = f"{now - 60:.4f}"
        state = default_channel_state()
        state["threads"][thread_ts] = {
            "chat_id": "c1",
            "last_reply_ts": thread_ts,
            "updated_at": now - 60,
        }
        save_channel_state(state_path, state)
        runner = _make_runner(FailingThreadBackend(), state_path)
        assert runner.run_once() == 0
        assert load_channel_state(state_path)["failures"] == 0

    def test_bot_posted_after_detects_direct_reply(self, tmp_path: Path) -> None:
        """A bot message newer than the snapshot suppresses the summary."""
        backend = ThreadBackend(
            thread_replies={
                "100.0": [
                    {"user": "", "bot": True, "text": "old answer", "ts": "102.0"},
                    {"user": "", "bot": True, "text": "direct reply", "ts": "105.0"},
                ]
            }
        )
        runner = _make_runner(backend, tmp_path / "s.json")
        assert runner._bot_posted_after("C1", "100.0", 102.0) is True

    def test_bot_posted_after_ignores_snapshot_and_older(self, tmp_path: Path) -> None:
        """Bot messages at or before the snapshot do not suppress."""
        backend = ThreadBackend(
            thread_replies={
                "100.0": [{"user": "", "bot": True, "text": "old answer", "ts": "102.0"}]
            }
        )
        runner = _make_runner(backend, tmp_path / "s.json")
        assert runner._bot_posted_after("C1", "100.0", 102.0) is False

    def test_bot_posted_after_without_thread_polling(self, tmp_path: Path) -> None:
        """Backends without poll_thread_messages never suppress."""
        runner = _make_runner(RecordingBackend(), tmp_path / "s.json")
        assert runner._bot_posted_after("C1", "100.0", 0.0) is False

    def test_bot_posted_after_poll_failure_is_safe(self, tmp_path: Path) -> None:
        """A failing re-poll falls back to sending the summary."""

        class RepollFailBackend(ThreadBackend):
            """Thread backend whose thread polling raises."""

            def poll_thread_messages(
                self, channel_id: str, thread_ts: str, oldest: str, limit: int = 100
            ) -> tuple[list[dict[str, Any]], str]:
                """Simulate a transport failure during the re-poll."""
                raise ConnectionError("re-poll failed")

        runner = _make_runner(RepollFailBackend(), tmp_path / "s.json")
        assert runner._bot_posted_after("C1", "100.0", 0.0) is False


class TestCursorPersistence:
    """Per-channel poll-cursor persistence across ticks."""

    def test_first_tick_polls_zero_and_persists_cursor(self, tmp_path: Path) -> None:
        """The first tick polls with '0' and stores the returned cursor."""
        state_path = tmp_path / "s.json"
        backend = RecordingBackend(cursor="42")
        runner = _make_runner(backend, state_path)
        assert runner.run_once() == 0
        assert backend.polled_oldest == ["0"]
        assert load_channel_state(state_path)["cursor"] == "42"

    def test_next_tick_polls_from_persisted_cursor(self, tmp_path: Path) -> None:
        """A later tick passes the persisted cursor as oldest."""
        state_path = tmp_path / "s.json"
        first = RecordingBackend(cursor="42")
        assert _make_runner(first, state_path).run_once() == 0
        second = RecordingBackend(cursor="43")
        assert _make_runner(second, state_path).run_once() == 0
        assert second.polled_oldest == ["42"]
        assert load_channel_state(state_path)["cursor"] == "43"

    def test_cursor_not_advanced_when_tick_fails_midway(self, tmp_path: Path) -> None:
        """A tick failing while handling a message keeps the old cursor."""

        class MidTickFailBackend(RecordingBackend):
            """Backend whose message handling fails after a successful poll."""

            def strip_bot_mention(self, text: str) -> str:
                """Simulate a failure while processing a polled message."""
                raise ConnectionError("mid-tick failure")

        state_path = tmp_path / "s.json"
        msg = {"user": "alice", "text": "hi", "ts": "1.0"}
        backend = MidTickFailBackend(messages=[msg], cursor="42")
        runner = _make_runner(backend, state_path)
        with pytest.raises(ConnectionError):
            runner.run_once()
        state = load_channel_state(state_path)
        assert state["cursor"] == "0"
        assert state["failures"] == 1

    def test_empty_returned_cursor_keeps_previous(self, tmp_path: Path) -> None:
        """An empty new cursor leaves the persisted cursor unchanged."""
        state_path = tmp_path / "s.json"
        state = default_channel_state()
        state["cursor"] = "42"
        save_channel_state(state_path, state)
        backend = RecordingBackend(cursor="")
        assert _make_runner(backend, state_path).run_once() == 0
        assert backend.polled_oldest == ["42"]
        assert load_channel_state(state_path)["cursor"] == "42"

    def test_stateless_runner_always_polls_zero(self, tmp_path: Path) -> None:
        """Without a state path the legacy oldest='0' behaviour remains."""
        backend = RecordingBackend(cursor="42")
        runner = _make_runner(backend, None)
        assert runner.run_once() == 0
        assert runner.run_once() == 0
        assert backend.polled_oldest == ["0", "0"]


class TestContinuationFailureCursor:
    """Cursor commit and retry semantics for failed thread continuations."""

    @staticmethod
    def _seed_state(state_path: Path, thread_ts: str, last_reply_ts: str) -> None:
        """Persist a state with cursor '42' and one known thread."""
        state = default_channel_state()
        state["cursor"] = "42"
        state["threads"][thread_ts] = {
            "chat_id": "c1",
            "last_reply_ts": last_reply_ts,
            "updated_at": time.time(),
        }
        save_channel_state(state_path, state)

    @staticmethod
    def _make_launch_runner(
        backend: Any, state_path: Path, launch_error: Exception | None = None
    ) -> LaunchOutcomeRunner:
        """Build a LaunchOutcomeRunner wired to *backend* with test defaults."""
        return LaunchOutcomeRunner(
            backend=backend,
            channel_name="",
            agent_name="Hermes Test Agent",
            state_path=state_path,
            launch_error=launch_error,
        )

    def test_failed_launch_keeps_cursor_and_last_reply_ts(self, tmp_path: Path) -> None:
        """A raising continuation launch keeps the old cursor and marker."""
        state_path = tmp_path / "s.json"
        self._seed_state(state_path, "100.0", "101.0")
        backend = ThreadBackend(
            cursor="99",
            thread_replies={
                "100.0": [{"user": "alice", "text": "follow up", "ts": "105.0"}]
            },
        )
        runner = self._make_launch_runner(
            backend, state_path, launch_error=ConnectionError("daemon down")
        )
        assert runner.run_once() == 0
        assert runner.launched_prompts, "the continuation launch must be attempted"
        state = load_channel_state(state_path)
        assert state["cursor"] == "42", "a failed continuation must keep the old cursor"
        assert state["threads"]["100.0"]["last_reply_ts"] == "101.0", (
            "last_reply_ts must not advance on a failed launch"
        )
        assert state["failures"] == 0, "launch failures are not transport failures"
        assert any("Error processing your message" in text for _, text, _ in backend.sent)

    def test_retried_followup_selected_despite_error_reply(self, tmp_path: Path) -> None:
        """The next tick retries the follow-up even with the bot error reply posted."""
        state_path = tmp_path / "s.json"
        self._seed_state(state_path, "100.0", "101.0")
        failing = ThreadBackend(
            cursor="99",
            thread_replies={
                "100.0": [{"user": "alice", "text": "follow up", "ts": "105.0"}]
            },
        )
        assert (
            self._make_launch_runner(
                failing, state_path, launch_error=ConnectionError("daemon down")
            ).run_once()
            == 0
        )
        retry_backend = ThreadBackend(
            cursor="99",
            thread_replies={
                "100.0": [
                    {"user": "alice", "text": "follow up", "ts": "105.0"},
                    {
                        "user": "",
                        "bot": True,
                        "text": "Error processing your message: daemon down",
                        "ts": "106.0",
                    },
                ]
            },
        )
        runner = self._make_launch_runner(retry_backend, state_path)
        assert runner.run_once() == 1
        assert any("follow up" in p for p in runner.launched_prompts), (
            "the bot error reply must not suppress the retried follow-up"
        )
        state = load_channel_state(state_path)
        assert state["cursor"] == "99"
        assert state["threads"]["100.0"]["last_reply_ts"] == "105.0"

    def test_successful_continuation_advances_cursor(self, tmp_path: Path) -> None:
        """A fully successful continuation tick commits the new cursor."""
        state_path = tmp_path / "s.json"
        self._seed_state(state_path, "100.0", "101.0")
        backend = ThreadBackend(
            cursor="99",
            thread_replies={
                "100.0": [{"user": "alice", "text": "follow up", "ts": "105.0"}]
            },
        )
        runner = self._make_launch_runner(backend, state_path)
        assert runner.run_once() == 1
        state = load_channel_state(state_path)
        assert state["cursor"] == "99"
        assert state["threads"]["100.0"]["last_reply_ts"] == "105.0"
        assert any("continuation done" in text for _, text, _ in backend.sent)

    def test_thread_poll_failure_keeps_cursor_without_breaker(
        self, tmp_path: Path
    ) -> None:
        """A thread-poll failure keeps the cursor and skips the breaker."""

        class FailingThreadBackend(ThreadBackend):
            """Thread backend whose thread polling raises."""

            def poll_thread_messages(
                self, channel_id: str, thread_ts: str, oldest: str, limit: int = 100
            ) -> tuple[list[dict[str, Any]], str]:
                """Simulate a transport failure during thread polling."""
                raise ConnectionError("thread poll failed")

        state_path = tmp_path / "s.json"
        self._seed_state(state_path, "100.0", "101.0")
        runner = self._make_launch_runner(FailingThreadBackend(cursor="99"), state_path)
        assert runner.run_once() == 0
        state = load_channel_state(state_path)
        assert state["cursor"] == "42", "a failed thread poll must keep the old cursor"
        assert state["failures"] == 0, "thread-poll failures are not transport failures"


class TestStateSchemaValidation:
    """JSON-valid but schema-invalid state files are normalized safely."""

    @pytest.mark.parametrize(
        ("raw", "field", "expected"),
        [
            ('{"paused_until": "not-a-number"}', "paused_until", 0.0),
            ('{"paused_until": NaN}', "paused_until", 0.0),
            ('{"paused_until": Infinity}', "paused_until", 0.0),
            ('{"paused_until": true}', "paused_until", 0.0),
            ('{"threads": ["a", "b"]}', "threads", {}),
            ('{"threads": {"t1": "not-a-dict"}}', "threads", {}),
            ('{"failures": "three"}', "failures", 0),
            ('{"failures": -2}', "failures", 0),
            ('{"failures": true}', "failures", 0),
            ('{"ledger": "oops"}', "ledger", []),
            ('{"ledger": [{"channel_id": 1}, "x", 5]}', "ledger", []),
            ('{"approved_users": "bob"}', "approved_users", []),
            ('{"approved_users": ["bob", 5, null]}', "approved_users", ["bob"]),
            ('{"pending_pairing": ["x"]}', "pending_pairing", {}),
            ('{"pending_pairing": {"bob": {"code": 5}}}', "pending_pairing", {}),
            ('{"pending_pairing": {"bob": "code"}}', "pending_pairing", {}),
            ('{"cursor": 5}', "cursor", "0"),
            ('{"cursor": ""}', "cursor", "0"),
            ('{"thread_rotation": -3}', "thread_rotation", 0),
            ('{"thread_rotation": "x"}', "thread_rotation", 0),
        ],
    )
    def test_malformed_field_replaced_with_default(
        self, tmp_path: Path, raw: str, field: str, expected: Any
    ) -> None:
        """Each malformed canonical field falls back to its default."""
        path = tmp_path / "state.json"
        path.write_text(raw, encoding="utf-8")
        assert load_channel_state(path)[field] == expected

    def test_malformed_thread_entry_fields_normalized(self, tmp_path: Path) -> None:
        """Wrong-typed fields inside a thread entry become safe defaults."""
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {"threads": {"t1": {"chat_id": 5, "last_reply_ts": [1], "updated_at": "x"}}}
            ),
            encoding="utf-8",
        )
        assert load_channel_state(path)["threads"]["t1"] == {
            "chat_id": "",
            "last_reply_ts": "",
            "updated_at": 0.0,
        }

    def test_pairing_ts_normalized_to_float(self, tmp_path: Path) -> None:
        """A valid pending code with a bad ts keeps the code, zeroes ts."""
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps({"pending_pairing": {"bob": {"code": "abcd1234", "ts": "bad"}}}),
            encoding="utf-8",
        )
        assert load_channel_state(path)["pending_pairing"]["bob"] == {
            "code": "abcd1234",
            "ts": 0.0,
        }

    def test_valid_ledger_entry_preserved(self, tmp_path: Path) -> None:
        """Well-formed ledger entries survive normalization intact."""
        path = tmp_path / "state.json"
        entry = {"channel_id": "C1", "thread_ts": "1.0", "text": "hi", "created": 2.0}
        path.write_text(json.dumps({"ledger": [entry]}), encoding="utf-8")
        assert load_channel_state(path)["ledger"] == [entry]

    def test_run_once_survives_schema_invalid_state(self, tmp_path: Path) -> None:
        """A tick over a schema-invalid state file neither crashes nor corrupts."""
        state_path = tmp_path / "state.json"
        state_path.write_text(
            '{"paused_until": "not-a-number", "threads": [], "failures": "x", '
            '"ledger": "oops", "cursor": 7}',
            encoding="utf-8",
        )
        backend = RecordingBackend()
        runner = _make_runner(backend, state_path)
        assert runner.run_once() == 0
        assert backend.polled_oldest == ["0"]
        assert load_channel_state(state_path) == default_channel_state()


class TestBackwardCompatibility:
    """Stateless construction and behaviour keep working unchanged."""

    def test_runner_without_state_path(self) -> None:
        """The legacy constructor signature still works."""
        runner = ChannelRunner(
            backend=RecordingBackend(),
            channel_name="general",
            agent_name="legacy",
        )
        assert runner._state_path is None
        assert runner._state is None

    def test_stateless_run_once_processes_nothing_quietly(self) -> None:
        """A stateless tick with no messages returns 0 and disconnects."""
        backend = RecordingBackend()
        runner = _make_runner(backend, None)
        assert runner.run_once() == 0
        assert backend.connect_calls == 1
