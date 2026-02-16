"""Main orchestrator for KISSClaw - coordinates channels, DB, queue, scheduler, IPC."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from kiss.agents.kissclaw.agent_runner import run_agent
from kiss.agents.kissclaw.channels.base import Channel
from kiss.agents.kissclaw.config import KissClawConfig
from kiss.agents.kissclaw.db import KissClawDB
from kiss.agents.kissclaw.group_queue import GroupQueue
from kiss.agents.kissclaw.ipc import IpcWatcher
from kiss.agents.kissclaw.router import format_messages, format_outbound
from kiss.agents.kissclaw.task_scheduler import TaskScheduler
from kiss.agents.kissclaw.types import Message, RegisteredGroup

logger = logging.getLogger(__name__)


class KissClawOrchestrator:
    """Main orchestrator: state, message loop, agent invocation.

    Mirrors NanoClaw's index.ts orchestrator pattern.
    """

    def __init__(
        self,
        config: KissClawConfig | None = None,
        db: KissClawDB | None = None,
        channel: Channel | None = None,
        agent_fn: object | None = None,
    ) -> None:
        self.config = config or KissClawConfig()
        self.db = db or KissClawDB()
        self.channel = channel
        self.agent_fn = agent_fn

        self.queue = GroupQueue(max_concurrent=self.config.max_concurrent_agents)
        self.queue.set_process_messages_fn(self._process_group_messages)

        self.scheduler = TaskScheduler(
            db=self.db,
            config=self.config,
            send_message_fn=self._send_message,
            agent_fn=self.agent_fn,
        )

        self.ipc_watcher = IpcWatcher(
            db=self.db,
            config=self.config,
            send_message_fn=self._send_message,
            register_group_fn=self._register_group,
        )

        # State
        self._last_timestamp: str = ""
        self._last_agent_timestamp: dict[str, str] = {}
        self._sessions: dict[str, str] = {}
        self._running = False
        self._message_loop_thread: threading.Thread | None = None

    # --- State management ---
    def load_state(self) -> None:
        self._last_timestamp = self.db.get_router_state("last_timestamp") or ""
        agent_ts_json = self.db.get_router_state("last_agent_timestamp")
        if agent_ts_json:
            try:
                self._last_agent_timestamp = json.loads(agent_ts_json)
            except (json.JSONDecodeError, TypeError):
                self._last_agent_timestamp = {}
        self._sessions = self.db.get_all_sessions()

    def save_state(self) -> None:
        self.db.set_router_state("last_timestamp", self._last_timestamp)
        self.db.set_router_state("last_agent_timestamp", json.dumps(self._last_agent_timestamp))

    # --- Group management ---
    def _register_group(self, jid: str, group: RegisteredGroup) -> None:
        self.db.set_registered_group(jid, group)
        group_dir = Path(self.config.groups_dir) / group.folder
        group_dir.mkdir(parents=True, exist_ok=True)

    def register_group(self, jid: str, group: RegisteredGroup) -> None:
        self._register_group(jid, group)

    # --- Message sending ---
    def _send_message(self, jid: str, text: str) -> None:
        if self.channel:
            self.channel.send_message(jid, text)

    # --- Process group messages (called by GroupQueue) ---
    def _process_group_messages(self, chat_jid: str) -> bool:
        registered = self.db.get_all_registered_groups()
        group = registered.get(chat_jid)
        if not group:
            return True

        is_main = group.folder == self.config.main_group_folder
        since_ts = self._last_agent_timestamp.get(chat_jid, "")
        missed = self.db.get_messages_since(chat_jid, since_ts, self.config.assistant_name)

        if not missed:
            return True

        # Check trigger for non-main groups
        if not is_main and group.requires_trigger:
            trigger_re = re.compile(self.config.trigger_pattern, re.IGNORECASE)
            has_trigger = any(trigger_re.search(m.content.strip()) for m in missed)
            if not has_trigger:
                return True

        formatted = format_messages(missed)

        # Update cursor
        prev_cursor = self._last_agent_timestamp.get(chat_jid, "")
        self._last_agent_timestamp[chat_jid] = missed[-1].timestamp
        self.save_state()

        output = run_agent(
            self.config, group.name, group.folder, formatted, agent_fn=self.agent_fn
        )

        if output.status == "error":
            # Rollback cursor
            self._last_agent_timestamp[chat_jid] = prev_cursor
            self.save_state()
            return False

        if output.result:
            text = format_outbound(output.result)
            if text:
                self._send_message(chat_jid, text)

        return True

    # --- Message loop ---
    def poll_messages_once(self) -> int:
        """Poll for new messages and enqueue processing. Returns count of new message groups."""
        registered = self.db.get_all_registered_groups()
        jids = list(registered.keys())
        messages, new_ts = self.db.get_new_messages(jids, self._last_timestamp, self.config.assistant_name)

        if not messages:
            return 0

        self._last_timestamp = new_ts
        self.save_state()

        # Deduplicate by group
        by_group: dict[str, list[Message]] = {}
        for msg in messages:
            by_group.setdefault(msg.chat_jid, []).append(msg)

        enqueued = 0
        for chat_jid, group_msgs in by_group.items():
            group = registered.get(chat_jid)
            if not group:
                continue

            is_main = group.folder == self.config.main_group_folder
            needs_trigger = not is_main and group.requires_trigger

            if needs_trigger:
                trigger_re = re.compile(self.config.trigger_pattern, re.IGNORECASE)
                has_trigger = any(trigger_re.search(m.content.strip()) for m in group_msgs)
                if not has_trigger:
                    continue

            self.queue.enqueue_message_check(chat_jid)
            enqueued += 1

        return enqueued

    def _message_loop(self) -> None:
        while self._running:
            try:
                self.poll_messages_once()
            except Exception:
                logger.exception("Error in message loop")
            time.sleep(self.config.poll_interval)

    # --- Recovery ---
    def recover_pending_messages(self) -> int:
        """Check for unprocessed messages in registered groups."""
        registered = self.db.get_all_registered_groups()
        count = 0
        for chat_jid, group in registered.items():
            since_ts = self._last_agent_timestamp.get(chat_jid, "")
            pending = self.db.get_messages_since(chat_jid, since_ts, self.config.assistant_name)
            if pending:
                self.queue.enqueue_message_check(chat_jid)
                count += 1
        return count

    # --- Lifecycle ---
    def start(self) -> None:
        """Start all subsystems."""
        self.load_state()

        # Ensure directories exist
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.groups_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.store_dir).mkdir(parents=True, exist_ok=True)

        if self.channel:
            self.channel.connect()

        self.scheduler.start()
        self.ipc_watcher.start()
        self.recover_pending_messages()

        self._running = True
        self._message_loop_thread = threading.Thread(target=self._message_loop, daemon=True)
        self._message_loop_thread.start()

        logger.info("KISSClaw running (trigger: @%s)", self.config.assistant_name)

    def stop(self) -> None:
        """Stop all subsystems."""
        self._running = False
        self.scheduler.stop()
        self.ipc_watcher.stop()
        self.queue.shutdown()
        if self.channel:
            self.channel.disconnect()

    def inject_message(self, msg: Message) -> None:
        """Inject a message into the DB (for testing or programmatic use)."""
        self.db.store_message(msg)
        self.db.store_chat_metadata(msg.chat_jid, msg.timestamp)
