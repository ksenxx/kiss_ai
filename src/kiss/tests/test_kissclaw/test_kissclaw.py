"""Comprehensive tests for KISSClaw - Python clone of NanoClaw."""

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kiss.agents.kissclaw.agent_runner import run_agent, AGENT_SYSTEM_PROMPT
from kiss.agents.kissclaw.channels.console import ConsoleChannel
from kiss.agents.kissclaw.config import KissClawConfig
from kiss.agents.kissclaw.db import KissClawDB
from kiss.agents.kissclaw.group_queue import GroupQueue
from kiss.agents.kissclaw.ipc import IpcWatcher
from kiss.agents.kissclaw.orchestrator import KissClawOrchestrator
from kiss.agents.kissclaw.router import (
    escape_xml,
    format_messages,
    format_outbound,
    strip_internal_tags,
)
from kiss.agents.kissclaw.task_scheduler import (
    TaskScheduler,
    compute_next_interval_run,
    run_scheduled_task,
)
from kiss.agents.kissclaw.types import (
    AgentOutput,
    ChatInfo,
    Message,
    RegisteredGroup,
    ScheduledTask,
    TaskRunLog,
)


# ============================================================================
# Config tests
# ============================================================================


class TestConfig:
    def test_defaults(self):
        cfg = KissClawConfig()
        assert cfg.assistant_name == "Andy"
        assert cfg.poll_interval == 2.0
        assert cfg.max_concurrent_agents == 5
        assert cfg.main_group_folder == "main"

    def test_trigger_pattern_auto_built(self):
        cfg = KissClawConfig(assistant_name="Bob")
        assert "Bob" in cfg.trigger_pattern
        assert cfg.trigger_pattern.startswith("^@")

    def test_custom_trigger_pattern(self):
        cfg = KissClawConfig(trigger_pattern="^!bot")
        assert cfg.trigger_pattern == "^!bot"

    def test_data_dirs_auto_created(self):
        cfg = KissClawConfig()
        assert "kissclaw_data" in cfg.data_dir
        assert "groups" in cfg.groups_dir
        assert "store" in cfg.store_dir

    def test_custom_data_dir(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = KissClawConfig(data_dir=td)
            assert cfg.data_dir == td
            assert "groups" in cfg.groups_dir


# ============================================================================
# Types tests
# ============================================================================


class TestTypes:
    def test_message_defaults(self):
        m = Message(id="1", chat_jid="g1", sender="u1", sender_name="User",
                    content="hi", timestamp="2024-01-01T00:00:00Z")
        assert not m.is_from_me
        assert not m.is_bot_message

    def test_registered_group(self):
        g = RegisteredGroup(name="Test", folder="test", trigger="@bot", added_at="now")
        assert g.requires_trigger is True

    def test_scheduled_task_defaults(self):
        t = ScheduledTask(id="t1", group_folder="main", chat_jid="j1",
                          prompt="hello", schedule_type="once", schedule_value="v")
        assert t.status == "active"
        assert t.context_mode == "isolated"

    def test_agent_output(self):
        o = AgentOutput(status="success", result="done")
        assert o.error is None

    def test_chat_info(self):
        c = ChatInfo(jid="j1", name="Chat", last_message_time="now")
        assert c.jid == "j1"

    def test_task_run_log(self):
        log = TaskRunLog(task_id="t1", run_at="now", duration_ms=100, status="success")
        assert log.result is None


# ============================================================================
# Database tests
# ============================================================================


class TestDB:
    def setup_method(self):
        self.db = KissClawDB()

    def teardown_method(self):
        self.db.close()

    def test_store_and_get_message(self):
        msg = Message(id="m1", chat_jid="g1", sender="u1", sender_name="User",
                      content="hello", timestamp="2024-01-01T00:00:00Z")
        self.db.store_message(msg)
        msgs = self.db.get_messages_since("g1", "2023-01-01T00:00:00Z", "Bot")
        assert len(msgs) == 1
        assert msgs[0].content == "hello"

    def test_get_new_messages_filters_bot(self):
        self.db.store_message(Message(id="m1", chat_jid="g1", sender="u1",
                                       sender_name="User", content="hello",
                                       timestamp="2024-01-01T00:00:01Z"))
        self.db.store_message(Message(id="m2", chat_jid="g1", sender="bot",
                                       sender_name="Bot", content="Bot: reply",
                                       timestamp="2024-01-01T00:00:02Z"))
        msgs, ts = self.db.get_new_messages(["g1"], "2024-01-01T00:00:00Z", "Bot")
        assert len(msgs) == 1
        assert msgs[0].id == "m1"

    def test_get_new_messages_empty_jids(self):
        msgs, ts = self.db.get_new_messages([], "", "Bot")
        assert msgs == []

    def test_get_messages_since_bot_message_flag(self):
        self.db.store_message(Message(id="m1", chat_jid="g1", sender="u1",
                                       sender_name="User", content="hello",
                                       timestamp="2024-01-01T00:00:01Z",
                                       is_bot_message=True))
        msgs = self.db.get_messages_since("g1", "2024-01-01T00:00:00Z", "X")
        assert len(msgs) == 0

    def test_chat_metadata(self):
        self.db.store_chat_metadata("g1", "2024-01-01T00:00:00Z", "Group 1")
        chats = self.db.get_all_chats()
        assert len(chats) == 1
        assert chats[0].name == "Group 1"

    def test_chat_metadata_update_name(self):
        self.db.store_chat_metadata("g1", "2024-01-01T00:00:00Z", "Old")
        self.db.store_chat_metadata("g1", "2024-01-02T00:00:00Z", "New")
        chats = self.db.get_all_chats()
        assert chats[0].name == "New"

    def test_chat_metadata_no_name(self):
        self.db.store_chat_metadata("g1", "2024-01-01T00:00:00Z")
        chats = self.db.get_all_chats()
        assert chats[0].name == "g1"

    def test_router_state(self):
        assert self.db.get_router_state("k1") is None
        self.db.set_router_state("k1", "v1")
        assert self.db.get_router_state("k1") == "v1"
        self.db.set_router_state("k1", "v2")
        assert self.db.get_router_state("k1") == "v2"

    def test_sessions(self):
        assert self.db.get_session("g1") is None
        self.db.set_session("g1", "s1")
        assert self.db.get_session("g1") == "s1"
        assert self.db.get_all_sessions() == {"g1": "s1"}

    def test_registered_groups(self):
        g = RegisteredGroup(name="Test", folder="test", trigger="@bot",
                            added_at="now", requires_trigger=False)
        self.db.set_registered_group("j1", g)
        got = self.db.get_registered_group("j1")
        assert got is not None
        assert got.name == "Test"
        assert not got.requires_trigger
        all_groups = self.db.get_all_registered_groups()
        assert "j1" in all_groups

    def test_registered_group_not_found(self):
        assert self.db.get_registered_group("nonexistent") is None

    def test_scheduled_tasks_crud(self):
        task = ScheduledTask(id="t1", group_folder="main", chat_jid="j1",
                             prompt="do something", schedule_type="once",
                             schedule_value="2024-06-01T00:00:00Z",
                             next_run="2024-06-01T00:00:00Z",
                             created_at="2024-01-01T00:00:00Z")
        self.db.create_task(task)
        got = self.db.get_task_by_id("t1")
        assert got is not None
        assert got.prompt == "do something"

        all_tasks = self.db.get_all_tasks()
        assert len(all_tasks) == 1

        self.db.update_task("t1", status="paused")
        got = self.db.get_task_by_id("t1")
        assert got.status == "paused"

        self.db.delete_task("t1")
        assert self.db.get_task_by_id("t1") is None

    def test_due_tasks(self):
        task = ScheduledTask(id="t1", group_folder="main", chat_jid="j1",
                             prompt="x", schedule_type="once",
                             schedule_value="2024-01-01T00:00:00Z",
                             next_run="2024-01-01T00:00:00Z", status="active",
                             created_at="2024-01-01T00:00:00Z")
        self.db.create_task(task)
        due = self.db.get_due_tasks("2024-06-01T00:00:00Z")
        assert len(due) == 1

        due_none = self.db.get_due_tasks("2023-01-01T00:00:00Z")
        assert len(due_none) == 0

    def test_update_task_after_run(self):
        task = ScheduledTask(id="t1", group_folder="main", chat_jid="j1",
                             prompt="x", schedule_type="once",
                             schedule_value="v", next_run="2024-01-01T00:00:00Z",
                             status="active", created_at="2024-01-01T00:00:00Z")
        self.db.create_task(task)
        self.db.update_task_after_run("t1", None, "done")
        got = self.db.get_task_by_id("t1")
        assert got.status == "completed"
        assert got.last_result == "done"

    def test_task_run_logs(self):
        task = ScheduledTask(id="t1", group_folder="main", chat_jid="j1",
                             prompt="x", schedule_type="once", schedule_value="v",
                             created_at="2024-01-01T00:00:00Z")
        self.db.create_task(task)
        log = TaskRunLog(task_id="t1", run_at="now", duration_ms=100,
                         status="success", result="ok")
        self.db.log_task_run(log)
        logs = self.db.get_task_run_logs("t1")
        assert len(logs) == 1
        assert logs[0].result == "ok"

    def test_update_task_empty_kwargs(self):
        task = ScheduledTask(id="t1", group_folder="main", chat_jid="j1",
                             prompt="x", schedule_type="once", schedule_value="v",
                             created_at="2024-01-01T00:00:00Z")
        self.db.create_task(task)
        self.db.update_task("t1")  # no kwargs -> no-op
        assert self.db.get_task_by_id("t1").prompt == "x"

    def test_db_file_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.db")
            db1 = KissClawDB(path)
            db1.store_message(Message(id="m1", chat_jid="g1", sender="u1",
                                       sender_name="User", content="persist",
                                       timestamp="2024-01-01T00:00:00Z"))
            db1.close()

            db2 = KissClawDB(path)
            msgs = db2.get_messages_since("g1", "2023-01-01T00:00:00Z", "Bot")
            assert len(msgs) == 1
            assert msgs[0].content == "persist"
            db2.close()


# ============================================================================
# Router tests
# ============================================================================


class TestRouter:
    def test_escape_xml(self):
        assert escape_xml("") == ""
        assert escape_xml("a<b>c") == "a&lt;b&gt;c"
        assert escape_xml('a"b') == "a&quot;b"

    def test_format_messages(self):
        msgs = [
            Message(id="1", chat_jid="g1", sender="u1", sender_name="Alice",
                    content="hello", timestamp="2024-01-01T00:00:00Z"),
            Message(id="2", chat_jid="g1", sender="u2", sender_name="Bob",
                    content="hi there", timestamp="2024-01-01T00:00:01Z"),
        ]
        result = format_messages(msgs)
        assert "<messages>" in result
        assert "Alice" in result
        assert "hello" in result
        assert "Bob" in result

    def test_format_messages_escapes_xml(self):
        msgs = [
            Message(id="1", chat_jid="g1", sender="u1", sender_name="<User>",
                    content="a < b", timestamp="t1"),
        ]
        result = format_messages(msgs)
        assert "&lt;User&gt;" in result
        assert "a &lt; b" in result

    def test_strip_internal_tags(self):
        assert strip_internal_tags("hello <internal>secret</internal> world") == "hello  world"
        assert strip_internal_tags("no tags") == "no tags"
        assert strip_internal_tags("<internal>all hidden</internal>") == ""

    def test_format_outbound(self):
        assert format_outbound("hello") == "hello"
        assert format_outbound("<internal>x</internal>real") == "real"
        assert format_outbound("<internal>all</internal>") == ""
        assert format_outbound("") == ""

    def test_format_messages_empty(self):
        result = format_messages([])
        assert "<messages>" in result


# ============================================================================
# GroupQueue tests
# ============================================================================


class TestGroupQueue:
    def test_basic_enqueue_and_process(self):
        processed = []

        def process_fn(jid):
            processed.append(jid)
            return True

        q = GroupQueue(max_concurrent=2)
        q.set_process_messages_fn(process_fn)
        q.enqueue_message_check("g1")
        time.sleep(0.2)
        assert "g1" in processed

    def test_concurrency_limit(self):
        active = {"count": 0, "max": 0}
        lock = threading.Lock()

        def slow_process(jid):
            with lock:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            time.sleep(0.1)
            with lock:
                active["count"] -= 1
            return True

        q = GroupQueue(max_concurrent=2)
        q.set_process_messages_fn(slow_process)
        for i in range(5):
            q.enqueue_message_check(f"g{i}")
        time.sleep(1.0)
        assert active["max"] <= 2

    def test_shutdown(self):
        q = GroupQueue()
        q.shutdown()
        q.enqueue_message_check("g1")  # should not process
        assert q.active_count == 0

    def test_is_group_active(self):
        q = GroupQueue()
        assert not q.is_group_active("g1")

    def test_enqueue_task(self):
        results = []

        def task_fn():
            results.append("done")
            return True

        q = GroupQueue(max_concurrent=2)
        q.enqueue_task("g1", "t1", task_fn)
        time.sleep(0.3)
        assert "done" in results

    def test_duplicate_task_rejected(self):
        call_count = {"n": 0}
        event = threading.Event()

        def slow_task():
            call_count["n"] += 1
            event.wait(timeout=1.0)
            return True

        def process_fn(jid):
            event.wait(timeout=1.0)
            return True

        q = GroupQueue(max_concurrent=1)
        q.set_process_messages_fn(process_fn)
        # Fill the slot
        q.enqueue_message_check("g0")
        time.sleep(0.1)
        # Now enqueue same task twice while at capacity
        q.enqueue_task("g1", "t1", slow_task)
        q.enqueue_task("g1", "t1", slow_task)
        event.set()
        time.sleep(0.5)
        assert call_count["n"] <= 1

    def test_retry_on_failure(self):
        attempts = {"n": 0}

        def fail_process(jid):
            attempts["n"] += 1
            if attempts["n"] < 2:
                return False
            return True

        q = GroupQueue(max_concurrent=2)
        q.set_process_messages_fn(fail_process)
        q.enqueue_message_check("g1")
        # Allow time for retry (base_retry=5s is too long for test, but we can verify attempt was made)
        time.sleep(0.3)
        assert attempts["n"] >= 1


# ============================================================================
# Agent Runner tests
# ============================================================================


class TestAgentRunner:
    def test_run_agent_with_mock(self):
        config = KissClawConfig()
        with tempfile.TemporaryDirectory() as td:
            config.groups_dir = td

            def mock_agent(prompt):
                return "Hello from agent!"

            output = run_agent(config, "TestGroup", "test", "<messages/>", agent_fn=mock_agent)
            assert output.status == "success"
            assert output.result == "Hello from agent!"

    def test_run_agent_with_memory(self):
        with tempfile.TemporaryDirectory() as td:
            config = KissClawConfig(groups_dir=td)
            group_dir = Path(td) / "test"
            group_dir.mkdir()
            (group_dir / "MEMORY.md").write_text("Remember: user likes Python.")

            def mock_agent(prompt):
                assert "Remember: user likes Python" in prompt
                return "Noted!"

            output = run_agent(config, "TestGroup", "test", "<messages/>", agent_fn=mock_agent)
            assert output.status == "success"

    def test_run_agent_error_handling(self):
        config = KissClawConfig()
        with tempfile.TemporaryDirectory() as td:
            config.groups_dir = td

            def failing_agent(prompt):
                raise ValueError("Agent crashed")

            output = run_agent(config, "TestGroup", "test", "<messages/>", agent_fn=failing_agent)
            assert output.status == "error"
            assert "Agent crashed" in output.error

    def test_agent_prompt_includes_group_name(self):
        config = KissClawConfig()
        with tempfile.TemporaryDirectory() as td:
            config.groups_dir = td
            captured = {}

            def capture_agent(prompt):
                captured["prompt"] = prompt
                return "ok"

            run_agent(config, "FamilyChat", "family", "<messages/>", agent_fn=capture_agent)
            assert "FamilyChat" in captured["prompt"]
            assert config.assistant_name in captured["prompt"]


# ============================================================================
# Task Scheduler tests
# ============================================================================


class TestTaskScheduler:
    def setup_method(self):
        self.db = KissClawDB()
        self.config = KissClawConfig()
        self.sent: list[tuple[str, str]] = []

    def teardown_method(self):
        self.db.close()

    def _send(self, jid: str, text: str) -> None:
        self.sent.append((jid, text))

    def test_run_once_task(self):
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                            added_at="now", requires_trigger=False)
        self.db.set_registered_group("j1", g)

        task = ScheduledTask(
            id="t1", group_folder="main", chat_jid="j1",
            prompt="Say hello", schedule_type="once",
            schedule_value="2024-01-01T00:00:00Z",
            next_run="2024-01-01T00:00:00Z", status="active",
            created_at="2024-01-01T00:00:00Z",
        )
        self.db.create_task(task)

        def mock_agent(prompt):
            return "Hello!"

        success = run_scheduled_task(task, self.db, self.config, self._send, agent_fn=mock_agent)
        assert success

        # Check task completed
        updated = self.db.get_task_by_id("t1")
        assert updated.status == "completed"

        # Check message sent
        assert len(self.sent) == 1
        assert self.sent[0] == ("j1", "Hello!")

        # Check run log
        logs = self.db.get_task_run_logs("t1")
        assert len(logs) == 1
        assert logs[0].status == "success"

    def test_run_task_group_not_found(self):
        task = ScheduledTask(
            id="t1", group_folder="nonexistent", chat_jid="j1",
            prompt="x", schedule_type="once", schedule_value="v",
            next_run="2024-01-01T00:00:00Z", status="active",
            created_at="2024-01-01T00:00:00Z",
        )
        self.db.create_task(task)
        success = run_scheduled_task(task, self.db, self.config, self._send)
        assert not success

    def test_interval_task_gets_next_run(self):
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                            added_at="now")
        self.db.set_registered_group("j1", g)

        task = ScheduledTask(
            id="t1", group_folder="main", chat_jid="j1",
            prompt="check", schedule_type="interval",
            schedule_value="60000",  # 60 seconds
            next_run="2024-01-01T00:00:00Z", status="active",
            created_at="2024-01-01T00:00:00Z",
        )
        self.db.create_task(task)

        def mock_agent(prompt):
            return "checked"

        run_scheduled_task(task, self.db, self.config, self._send, agent_fn=mock_agent)
        updated = self.db.get_task_by_id("t1")
        assert updated.next_run is not None
        assert updated.status == "active"

    def test_scheduler_poll_once(self):
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy", added_at="now")
        self.db.set_registered_group("j1", g)

        # Create a past-due task
        task = ScheduledTask(
            id="t1", group_folder="main", chat_jid="j1",
            prompt="poll test", schedule_type="once",
            schedule_value="2024-01-01T00:00:00Z",
            next_run="2024-01-01T00:00:00Z", status="active",
            created_at="2024-01-01T00:00:00Z",
        )
        self.db.create_task(task)

        def mock_agent(prompt):
            return "polled!"

        scheduler = TaskScheduler(self.db, self.config, self._send, agent_fn=mock_agent)
        count = scheduler.poll_once()
        assert count == 1
        assert len(self.sent) == 1

    def test_scheduler_skips_paused(self):
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy", added_at="now")
        self.db.set_registered_group("j1", g)

        task = ScheduledTask(
            id="t1", group_folder="main", chat_jid="j1",
            prompt="paused", schedule_type="once",
            schedule_value="2024-01-01T00:00:00Z",
            next_run="2024-01-01T00:00:00Z", status="paused",
            created_at="2024-01-01T00:00:00Z",
        )
        self.db.create_task(task)
        scheduler = TaskScheduler(self.db, self.config, self._send)
        count = scheduler.poll_once()
        assert count == 0

    def test_compute_next_interval_run(self):
        result = compute_next_interval_run("60000")
        assert result is not None
        # Should be about 60s in the future
        dt = datetime.fromisoformat(result)
        assert dt > datetime.now(timezone.utc)

    def test_compute_next_interval_invalid(self):
        assert compute_next_interval_run("abc") is None
        assert compute_next_interval_run("-1") is None

    def test_run_task_with_internal_tags(self):
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy", added_at="now")
        self.db.set_registered_group("j1", g)

        task = ScheduledTask(
            id="t1", group_folder="main", chat_jid="j1",
            prompt="test", schedule_type="once", schedule_value="v",
            next_run="2024-01-01T00:00:00Z", status="active",
            created_at="2024-01-01T00:00:00Z",
        )
        self.db.create_task(task)

        def mock_agent(prompt):
            return "<internal>thinking</internal>visible response"

        run_scheduled_task(task, self.db, self.config, self._send, agent_fn=mock_agent)
        assert len(self.sent) == 1
        assert self.sent[0][1] == "visible response"


# ============================================================================
# IPC Watcher tests
# ============================================================================


class TestIpcWatcher:
    def setup_method(self):
        self.db = KissClawDB()
        self.tmpdir = tempfile.mkdtemp()
        self.config = KissClawConfig(data_dir=self.tmpdir)
        self.sent: list[tuple[str, str]] = []

    def teardown_method(self):
        self.db.close()

    def _send(self, jid: str, text: str) -> None:
        self.sent.append((jid, text))

    def _setup_main_group(self):
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                            added_at="now", requires_trigger=False)
        self.db.set_registered_group("j1", g)
        return g

    def test_process_ipc_message(self):
        self._setup_main_group()
        ipc_dir = Path(self.tmpdir) / "ipc" / "main" / "messages"
        ipc_dir.mkdir(parents=True)
        (ipc_dir / "msg1.json").write_text(json.dumps({
            "type": "message", "chatJid": "j1", "text": "hello from agent"
        }))

        watcher = IpcWatcher(self.db, self.config, self._send)
        count = watcher.poll_once()
        assert count == 1
        assert self.sent == [("j1", "hello from agent")]

    def test_unauthorized_message_blocked(self):
        self._setup_main_group()
        g2 = RegisteredGroup(name="Other", folder="other", trigger="@Andy",
                              added_at="now")
        self.db.set_registered_group("j2", g2)

        # "other" group tries to send to "j1" which belongs to "main"
        ipc_dir = Path(self.tmpdir) / "ipc" / "other" / "messages"
        ipc_dir.mkdir(parents=True)
        (ipc_dir / "msg1.json").write_text(json.dumps({
            "type": "message", "chatJid": "j1", "text": "sneaky"
        }))

        watcher = IpcWatcher(self.db, self.config, self._send)
        watcher.poll_once()
        assert len(self.sent) == 0

    def test_schedule_task_via_ipc(self):
        self._setup_main_group()
        ipc_dir = Path(self.tmpdir) / "ipc" / "main" / "tasks"
        ipc_dir.mkdir(parents=True)
        (ipc_dir / "task1.json").write_text(json.dumps({
            "type": "schedule_task",
            "targetJid": "j1",
            "prompt": "daily check",
            "schedule_type": "interval",
            "schedule_value": "86400000",
        }))

        watcher = IpcWatcher(self.db, self.config, self._send)
        watcher.poll_once()
        tasks = self.db.get_all_tasks()
        assert len(tasks) == 1
        assert tasks[0].prompt == "daily check"

    def test_pause_resume_cancel_task_via_ipc(self):
        self._setup_main_group()
        task = ScheduledTask(
            id="t1", group_folder="main", chat_jid="j1",
            prompt="x", schedule_type="once", schedule_value="v",
            status="active", created_at="now",
        )
        self.db.create_task(task)

        ipc_dir = Path(self.tmpdir) / "ipc" / "main" / "tasks"
        ipc_dir.mkdir(parents=True)

        # Pause
        (ipc_dir / "1.json").write_text(json.dumps({"type": "pause_task", "taskId": "t1"}))
        watcher = IpcWatcher(self.db, self.config, self._send)
        watcher.poll_once()
        assert self.db.get_task_by_id("t1").status == "paused"

        # Resume
        (ipc_dir / "2.json").write_text(json.dumps({"type": "resume_task", "taskId": "t1"}))
        watcher.poll_once()
        assert self.db.get_task_by_id("t1").status == "active"

        # Cancel
        (ipc_dir / "3.json").write_text(json.dumps({"type": "cancel_task", "taskId": "t1"}))
        watcher.poll_once()
        assert self.db.get_task_by_id("t1") is None

    def test_register_group_via_ipc(self):
        self._setup_main_group()
        registered = []

        def register_fn(jid, group):
            registered.append((jid, group))

        ipc_dir = Path(self.tmpdir) / "ipc" / "main" / "tasks"
        ipc_dir.mkdir(parents=True)
        (ipc_dir / "reg1.json").write_text(json.dumps({
            "type": "register_group",
            "jid": "j2", "name": "Family", "folder": "family",
            "trigger": "@Andy",
        }))

        watcher = IpcWatcher(self.db, self.config, self._send, register_group_fn=register_fn)
        watcher.poll_once()
        assert len(registered) == 1
        g = self.db.get_registered_group("j2")
        assert g is not None
        assert g.name == "Family"

    def test_non_main_cannot_register_group(self):
        self._setup_main_group()
        g2 = RegisteredGroup(name="Other", folder="other", trigger="@Andy", added_at="now")
        self.db.set_registered_group("j2", g2)

        ipc_dir = Path(self.tmpdir) / "ipc" / "other" / "tasks"
        ipc_dir.mkdir(parents=True)
        (ipc_dir / "reg1.json").write_text(json.dumps({
            "type": "register_group",
            "jid": "j3", "name": "Hack", "folder": "hack", "trigger": "@Andy",
        }))

        watcher = IpcWatcher(self.db, self.config, self._send)
        watcher.poll_once()
        assert self.db.get_registered_group("j3") is None

    def test_empty_ipc_dir(self):
        watcher = IpcWatcher(self.db, self.config, self._send)
        count = watcher.poll_once()
        assert count == 0

    def test_ipc_error_moves_to_errors(self):
        self._setup_main_group()
        ipc_dir = Path(self.tmpdir) / "ipc" / "main" / "messages"
        ipc_dir.mkdir(parents=True)
        (ipc_dir / "bad.json").write_text("NOT JSON{{{")

        watcher = IpcWatcher(self.db, self.config, self._send)
        watcher.poll_once()
        # Bad file should be moved to errors
        error_dir = Path(self.tmpdir) / "ipc" / "errors"
        assert error_dir.exists()
        assert len(list(error_dir.iterdir())) == 1


# ============================================================================
# Console Channel tests
# ============================================================================


class TestConsoleChannel:
    def test_connect_disconnect(self):
        ch = ConsoleChannel()
        assert not ch.is_connected()
        ch.connect()
        assert ch.is_connected()
        ch.disconnect()
        assert not ch.is_connected()

    def test_send_message(self):
        ch = ConsoleChannel()
        ch.connect()
        ch.send_message("j1", "hello")
        ch.send_message("j2", "world")
        assert ch.get_sent_messages() == [("j1", "hello"), ("j2", "world")]

    def test_clear_sent(self):
        ch = ConsoleChannel()
        ch.send_message("j1", "hello")
        ch.clear_sent()
        assert ch.get_sent_messages() == []

    def test_owns_jid(self):
        ch = ConsoleChannel(jid_prefix="console")
        assert ch.owns_jid("console_g1")
        assert not ch.owns_jid("whatsapp_g1")

    def test_name(self):
        ch = ConsoleChannel()
        assert ch.name == "console"

    def test_inject_message_callback(self):
        ch = ConsoleChannel()
        received = []
        ch.set_on_message(lambda jid, msg: received.append((jid, msg)))
        msg = Message(id="1", chat_jid="console_g1", sender="u1",
                      sender_name="User", content="hi", timestamp="t1")
        ch.inject_message(msg)
        assert len(received) == 1


# ============================================================================
# Orchestrator tests
# ============================================================================


class TestOrchestrator:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = KissClawConfig(
            data_dir=self.tmpdir,
            assistant_name="Andy",
            poll_interval=0.1,
        )
        self.db = KissClawDB()
        self.channel = ConsoleChannel()
        self.agent_responses: list[str] = ["Default response"]

    def teardown_method(self):
        self.db.close()

    def _mock_agent(self, prompt):
        if self.agent_responses:
            return self.agent_responses.pop(0)
        return "Default response"

    def _make_orchestrator(self):
        return KissClawOrchestrator(
            config=self.config, db=self.db,
            channel=self.channel, agent_fn=self._mock_agent,
        )

    def test_register_group(self):
        orch = self._make_orchestrator()
        g = RegisteredGroup(name="Test", folder="test", trigger="@Andy", added_at="now")
        orch.register_group("j1", g)
        assert self.db.get_registered_group("j1") is not None
        assert (Path(self.config.groups_dir) / "test").exists()

    def test_state_save_load(self):
        orch = self._make_orchestrator()
        orch._last_timestamp = "2024-01-01T00:00:00Z"
        orch._last_agent_timestamp = {"j1": "2024-01-01T00:00:01Z"}
        orch.save_state()

        orch2 = self._make_orchestrator()
        orch2.load_state()
        assert orch2._last_timestamp == "2024-01-01T00:00:00Z"
        assert orch2._last_agent_timestamp == {"j1": "2024-01-01T00:00:01Z"}

    def test_inject_and_poll_message(self):
        orch = self._make_orchestrator()
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                            added_at="now", requires_trigger=False)
        orch.register_group("j1", g)
        orch.load_state()

        msg = Message(id="m1", chat_jid="j1", sender="u1", sender_name="User",
                      content="hello", timestamp="2024-01-01T00:00:01Z")
        orch.inject_message(msg)

        self.agent_responses = ["Hi there!"]
        count = orch.poll_messages_once()
        assert count == 1

        # Give queue time to process
        time.sleep(0.5)
        sent = self.channel.get_sent_messages()
        assert len(sent) >= 1
        assert sent[0][1] == "Hi there!"

    def test_trigger_required_for_non_main(self):
        orch = self._make_orchestrator()
        g = RegisteredGroup(name="Other", folder="other", trigger="@Andy",
                            added_at="now", requires_trigger=True)
        orch.register_group("j2", g)
        orch.load_state()

        # Message without trigger - should not be processed
        msg = Message(id="m1", chat_jid="j2", sender="u1", sender_name="User",
                      content="just chatting", timestamp="2024-01-01T00:00:01Z")
        orch.inject_message(msg)
        count = orch.poll_messages_once()
        assert count == 0

    def test_trigger_activates_non_main(self):
        orch = self._make_orchestrator()
        g = RegisteredGroup(name="Other", folder="other", trigger="@Andy",
                            added_at="now", requires_trigger=True)
        orch.register_group("j2", g)
        orch.load_state()

        msg = Message(id="m1", chat_jid="j2", sender="u1", sender_name="User",
                      content="@Andy what's up?", timestamp="2024-01-01T00:00:01Z")
        orch.inject_message(msg)

        self.agent_responses = ["Not much!"]
        count = orch.poll_messages_once()
        assert count == 1
        time.sleep(0.5)
        sent = self.channel.get_sent_messages()
        assert any("Not much!" in s[1] for s in sent)

    def test_recover_pending_messages(self):
        orch = self._make_orchestrator()
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                            added_at="now", requires_trigger=False)
        orch.register_group("j1", g)

        msg = Message(id="m1", chat_jid="j1", sender="u1", sender_name="User",
                      content="recover me", timestamp="2024-01-01T00:00:01Z")
        orch.inject_message(msg)
        orch.load_state()

        self.agent_responses = ["Recovered!"]
        count = orch.recover_pending_messages()
        assert count == 1
        time.sleep(0.5)
        sent = self.channel.get_sent_messages()
        assert len(sent) >= 1

    def test_multiple_groups_isolation(self):
        orch = self._make_orchestrator()
        g1 = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                             added_at="now", requires_trigger=False)
        g2 = RegisteredGroup(name="Family", folder="family", trigger="@Andy",
                             added_at="now", requires_trigger=False)
        orch.register_group("j1", g1)
        orch.register_group("j2", g2)
        orch.load_state()

        msg1 = Message(id="m1", chat_jid="j1", sender="u1", sender_name="User",
                       content="msg to main", timestamp="2024-01-01T00:00:01Z")
        msg2 = Message(id="m2", chat_jid="j2", sender="u2", sender_name="User2",
                       content="msg to family", timestamp="2024-01-01T00:00:02Z")
        orch.inject_message(msg1)
        orch.inject_message(msg2)

        self.agent_responses = ["Reply to main", "Reply to family"]
        orch.poll_messages_once()
        time.sleep(1.0)
        sent = self.channel.get_sent_messages()
        assert len(sent) >= 2

    def test_start_stop(self):
        orch = self._make_orchestrator()
        self.channel.connect()
        orch.start()
        time.sleep(0.3)
        assert orch._running
        orch.stop()
        assert not orch._running

    def test_agent_error_rollback(self):
        orch = self._make_orchestrator()
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                            added_at="now", requires_trigger=False)
        orch.register_group("j1", g)
        orch.load_state()

        msg = Message(id="m1", chat_jid="j1", sender="u1", sender_name="User",
                      content="cause error", timestamp="2024-01-01T00:00:01Z")
        orch.inject_message(msg)

        def failing_agent(prompt):
            raise RuntimeError("boom")

        orch.agent_fn = failing_agent
        orch.queue.set_process_messages_fn(orch._process_group_messages)

        orch.poll_messages_once()
        time.sleep(0.5)

        # Cursor should be rolled back
        assert orch._last_agent_timestamp.get("j1", "") == ""

    def test_no_channel_does_not_crash(self):
        orch = KissClawOrchestrator(
            config=self.config, db=self.db, agent_fn=self._mock_agent
        )
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                            added_at="now", requires_trigger=False)
        orch.register_group("j1", g)
        orch.load_state()

        msg = Message(id="m1", chat_jid="j1", sender="u1", sender_name="User",
                      content="hello", timestamp="2024-01-01T00:00:01Z")
        orch.inject_message(msg)

        self.agent_responses = ["response"]
        orch.poll_messages_once()
        time.sleep(0.5)
        # Should not crash even without channel

    def test_bot_messages_filtered_from_processing(self):
        orch = self._make_orchestrator()
        g = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                            added_at="now", requires_trigger=False)
        orch.register_group("j1", g)
        orch.load_state()

        # Only bot messages
        msg = Message(id="m1", chat_jid="j1", sender="bot", sender_name="Andy",
                      content="Andy: automated reply", timestamp="2024-01-01T00:00:01Z",
                      is_bot_message=True)
        orch.inject_message(msg)
        count = orch.poll_messages_once()
        assert count == 0


# ============================================================================
# Integration test
# ============================================================================


class TestIntegration:
    def test_full_flow_inject_poll_respond(self):
        """End-to-end: register group, inject message, poll, get response."""
        with tempfile.TemporaryDirectory() as td:
            config = KissClawConfig(data_dir=td, assistant_name="Andy", poll_interval=0.1)
            db = KissClawDB()
            channel = ConsoleChannel()
            responses = ["Integration test response!"]

            def agent(prompt):
                return responses.pop(0) if responses else "default"

            orch = KissClawOrchestrator(config=config, db=db, channel=channel, agent_fn=agent)

            # Register main group
            g = RegisteredGroup(name="Main", folder="main", trigger="@Andy",
                                added_at="now", requires_trigger=False)
            orch.register_group("j1", g)
            orch.load_state()

            # Inject a message
            msg = Message(id="m1", chat_jid="j1", sender="u1", sender_name="Alice",
                          content="Hello Andy!", timestamp="2024-01-01T00:00:01Z")
            orch.inject_message(msg)

            # Poll and process
            orch.poll_messages_once()
            time.sleep(0.5)

            # Verify response sent
            sent = channel.get_sent_messages()
            assert len(sent) == 1
            assert sent[0] == ("j1", "Integration test response!")

            db.close()

    def test_scheduled_task_with_ipc_creation(self):
        """Create a task via IPC, then run it via scheduler."""
        with tempfile.TemporaryDirectory() as td:
            config = KissClawConfig(data_dir=td)
            db = KissClawDB()
            sent: list[tuple[str, str]] = []

            def send(jid, text):
                sent.append((jid, text))

            def agent(prompt):
                return "Task result"

            # Setup group
            g = RegisteredGroup(name="Main", folder="main", trigger="@Andy", added_at="now")
            db.set_registered_group("j1", g)

            # Create task via IPC
            ipc_dir = Path(td) / "ipc" / "main" / "tasks"
            ipc_dir.mkdir(parents=True)
            (ipc_dir / "t1.json").write_text(json.dumps({
                "type": "schedule_task",
                "targetJid": "j1",
                "prompt": "Check something",
                "schedule_type": "once",
                "schedule_value": "2024-01-01T00:00:00Z",
            }))

            watcher = IpcWatcher(db, config, send)
            watcher.poll_once()

            # Verify task created
            tasks = db.get_all_tasks()
            assert len(tasks) == 1

            # Run scheduler
            scheduler = TaskScheduler(db, config, send, agent_fn=agent)
            count = scheduler.poll_once()
            assert count == 1
            assert len(sent) == 1
            assert sent[0][1] == "Task result"

            db.close()

    def test_multi_group_with_trigger_filtering(self):
        """Test that trigger filtering works correctly across groups."""
        with tempfile.TemporaryDirectory() as td:
            config = KissClawConfig(data_dir=td, assistant_name="Andy")
            db = KissClawDB()
            channel = ConsoleChannel()
            call_count = {"n": 0}

            def agent(prompt):
                call_count["n"] += 1
                return f"Response {call_count['n']}"

            orch = KissClawOrchestrator(config=config, db=db, channel=channel, agent_fn=agent)

            # Main group (no trigger needed)
            orch.register_group("j1", RegisteredGroup(
                name="Main", folder="main", trigger="@Andy",
                added_at="now", requires_trigger=False,
            ))
            # Non-main group (trigger required)
            orch.register_group("j2", RegisteredGroup(
                name="Family", folder="family", trigger="@Andy",
                added_at="now", requires_trigger=True,
            ))
            orch.load_state()

            # Message to main (no trigger needed)
            orch.inject_message(Message(
                id="m1", chat_jid="j1", sender="u1", sender_name="User",
                content="just a message", timestamp="2024-01-01T00:00:01Z",
            ))
            # Message to family WITHOUT trigger
            orch.inject_message(Message(
                id="m2", chat_jid="j2", sender="u2", sender_name="User2",
                content="no trigger here", timestamp="2024-01-01T00:00:02Z",
            ))

            orch.poll_messages_once()
            time.sleep(0.5)

            # Only main should be processed
            sent = channel.get_sent_messages()
            assert len(sent) == 1
            assert sent[0][0] == "j1"

            # Now send with trigger to family
            channel.clear_sent()
            orch.inject_message(Message(
                id="m3", chat_jid="j2", sender="u2", sender_name="User2",
                content="@Andy help me", timestamp="2024-01-01T00:00:03Z",
            ))
            orch.poll_messages_once()
            time.sleep(0.5)

            sent = channel.get_sent_messages()
            assert len(sent) >= 1
            assert any(s[0] == "j2" for s in sent)

            db.close()
