"""IPC file watcher for KISSClaw - processes message and task files from agents."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from kiss.agents.kissclaw.config import KissClawConfig
from kiss.agents.kissclaw.db import KissClawDB
from kiss.agents.kissclaw.task_scheduler import compute_next_cron_run, compute_next_interval_run
from kiss.agents.kissclaw.types import RegisteredGroup, ScheduledTask

logger = logging.getLogger(__name__)


class IpcWatcher:
    """Watches IPC directories for messages and task commands from agents."""

    def __init__(
        self,
        db: KissClawDB,
        config: KissClawConfig,
        send_message_fn: Callable[[str, str], None],
        register_group_fn: Callable[[str, RegisteredGroup], None] | None = None,
    ) -> None:
        self.db = db
        self.config = config
        self.send_message_fn = send_message_fn
        self.register_group_fn = register_group_fn
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def poll_once(self) -> int:
        """Process IPC files once. Returns total files processed."""
        ipc_base = Path(self.config.data_dir) / "ipc"
        if not ipc_base.exists():
            return 0

        registered = self.db.get_all_registered_groups()
        count = 0

        for group_dir in ipc_base.iterdir():
            if not group_dir.is_dir() or group_dir.name == "errors":
                continue
            source_group = group_dir.name
            is_main = source_group == self.config.main_group_folder

            # Process messages
            messages_dir = group_dir / "messages"
            if messages_dir.exists():
                for f in sorted(messages_dir.glob("*.json")):
                    try:
                        data = json.loads(f.read_text())
                        if (
                            data.get("type") == "message"
                            and data.get("chatJid")
                            and data.get("text")
                        ):
                            target_jid = data["chatJid"]
                            target_group = registered.get(target_jid)
                            if is_main or (target_group and target_group.folder == source_group):
                                self.send_message_fn(target_jid, data["text"])
                                count += 1
                            else:
                                logger.warning(
                                    "Blocked unauthorized IPC message from %s to %s",
                                    source_group, target_jid,
                                )
                        f.unlink()
                    except Exception:
                        logger.exception("Error processing IPC message %s", f)
                        error_dir = ipc_base / "errors"
                        error_dir.mkdir(exist_ok=True)
                        f.rename(error_dir / f"{source_group}-{f.name}")

            # Process tasks
            tasks_dir = group_dir / "tasks"
            if tasks_dir.exists():
                for f in sorted(tasks_dir.glob("*.json")):
                    try:
                        data = json.loads(f.read_text())
                        self._process_task_ipc(data, source_group, is_main)
                        count += 1
                        f.unlink()
                    except Exception:
                        logger.exception("Error processing IPC task %s", f)
                        error_dir = ipc_base / "errors"
                        error_dir.mkdir(exist_ok=True)
                        f.rename(error_dir / f"{source_group}-{f.name}")

        return count

    def _process_task_ipc(self, data: dict, source_group: str, is_main: bool) -> None:
        action = data.get("type", "")
        registered = self.db.get_all_registered_groups()

        if action == "schedule_task":
            target_jid = data.get("targetJid", "")
            target_group = registered.get(target_jid)
            if not target_group:
                return
            if not is_main and target_group.folder != source_group:
                logger.warning("Blocked unauthorized schedule_task from %s", source_group)
                return

            schedule_type = data.get("schedule_type", "")
            schedule_value = data.get("schedule_value", "")
            prompt = data.get("prompt", "")
            if not all([prompt, schedule_type, schedule_value]):
                return

            next_run: str | None = None
            if schedule_type == "cron":
                next_run = compute_next_cron_run(schedule_value)
            elif schedule_type == "interval":
                next_run = compute_next_interval_run(schedule_value)
            elif schedule_type == "once":
                try:
                    next_run = datetime.fromisoformat(schedule_value).isoformat()
                except ValueError:
                    return

            task_id = f"task-{int(time.time() * 1000)}"
            context_mode = data.get("context_mode", "isolated")
            if context_mode not in ("group", "isolated"):
                context_mode = "isolated"

            self.db.create_task(ScheduledTask(
                id=task_id,
                group_folder=target_group.folder,
                chat_jid=target_jid,
                prompt=prompt,
                schedule_type=schedule_type,
                schedule_value=schedule_value,
                context_mode=context_mode,
                next_run=next_run,
                status="active",
                created_at=datetime.now(UTC).isoformat(),
            ))

        elif action == "pause_task":
            task = self.db.get_task_by_id(data.get("taskId", ""))
            if task and (is_main or task.group_folder == source_group):
                self.db.update_task(task.id, status="paused")

        elif action == "resume_task":
            task = self.db.get_task_by_id(data.get("taskId", ""))
            if task and (is_main or task.group_folder == source_group):
                self.db.update_task(task.id, status="active")

        elif action == "cancel_task":
            task = self.db.get_task_by_id(data.get("taskId", ""))
            if task and (is_main or task.group_folder == source_group):
                self.db.delete_task(task.id)

        elif action == "register_group":
            if not is_main:
                return
            jid = data.get("jid", "")
            name = data.get("name", "")
            folder = data.get("folder", "")
            trigger = data.get("trigger", "")
            if all([jid, name, folder, trigger]):
                group = RegisteredGroup(
                    name=name, folder=folder, trigger=trigger,
                    added_at=datetime.now(UTC).isoformat(),
                    requires_trigger=data.get("requiresTrigger", True),
                )
                self.db.set_registered_group(jid, group)
                if self.register_group_fn:
                    self.register_group_fn(jid, group)

    def _loop(self) -> None:
        while self._running:
            try:
                self.poll_once()
            except Exception:
                logger.exception("Error in IPC watcher loop")
            time.sleep(self.config.ipc_poll_interval)
