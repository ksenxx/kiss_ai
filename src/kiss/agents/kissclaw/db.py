"""SQLite database for KISSClaw - messages, groups, sessions, tasks, state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from kiss.agents.kissclaw.types import (
    ChatInfo,
    Message,
    RegisteredGroup,
    ScheduledTask,
    TaskRunLog,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    jid TEXT PRIMARY KEY,
    name TEXT,
    last_message_time TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT,
    chat_jid TEXT,
    sender TEXT,
    sender_name TEXT,
    content TEXT,
    timestamp TEXT,
    is_from_me INTEGER,
    is_bot_message INTEGER DEFAULT 0,
    PRIMARY KEY (id, chat_jid)
);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,
    group_folder TEXT NOT NULL,
    chat_jid TEXT NOT NULL,
    prompt TEXT NOT NULL,
    schedule_type TEXT NOT NULL,
    schedule_value TEXT NOT NULL,
    context_mode TEXT DEFAULT 'isolated',
    next_run TEXT,
    last_run TEXT,
    last_result TEXT,
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON scheduled_tasks(next_run);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON scheduled_tasks(status);
CREATE TABLE IF NOT EXISTS task_run_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    run_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    result TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS router_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    group_folder TEXT PRIMARY KEY,
    session_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS registered_groups (
    jid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    folder TEXT NOT NULL UNIQUE,
    trigger_pattern TEXT NOT NULL,
    added_at TEXT NOT NULL,
    requires_trigger INTEGER DEFAULT 1
);
"""


class KissClawDB:
    """SQLite database for KISSClaw state."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- Chat metadata ---
    def store_chat_metadata(self, jid: str, timestamp: str, name: str | None = None) -> None:
        if name:
            self.conn.execute(
                "INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?) "
                "ON CONFLICT(jid) DO UPDATE SET name=excluded.name, "
                "last_message_time=MAX(last_message_time, excluded.last_message_time)",
                (jid, name, timestamp),
            )
        else:
            self.conn.execute(
                "INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?) "
                "ON CONFLICT(jid) DO UPDATE SET "
                "last_message_time=MAX(last_message_time, excluded.last_message_time)",
                (jid, jid, timestamp),
            )
        self.conn.commit()

    def get_all_chats(self) -> list[ChatInfo]:
        rows = self.conn.execute(
            "SELECT jid, name, last_message_time FROM chats ORDER BY last_message_time DESC"
        ).fetchall()
        return [ChatInfo(jid=r["jid"], name=r["name"], last_message_time=r["last_message_time"]) for r in rows]

    # --- Messages ---
    def store_message(self, msg: Message) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO messages (id, chat_jid, sender, sender_name, content, "
            "timestamp, is_from_me, is_bot_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (msg.id, msg.chat_jid, msg.sender, msg.sender_name, msg.content,
             msg.timestamp, int(msg.is_from_me), int(msg.is_bot_message)),
        )
        self.conn.commit()

    def get_new_messages(
        self, jids: list[str], last_timestamp: str, bot_prefix: str
    ) -> tuple[list[Message], str]:
        if not jids:
            return [], last_timestamp
        placeholders = ",".join("?" for _ in jids)
        rows = self.conn.execute(
            f"SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me "
            f"FROM messages WHERE timestamp > ? AND chat_jid IN ({placeholders}) "
            f"AND is_bot_message = 0 AND content NOT LIKE ? ORDER BY timestamp",
            [last_timestamp, *jids, f"{bot_prefix}:%"],
        ).fetchall()
        messages = [
            Message(
                id=r["id"], chat_jid=r["chat_jid"], sender=r["sender"],
                sender_name=r["sender_name"], content=r["content"],
                timestamp=r["timestamp"], is_from_me=bool(r["is_from_me"]),
            )
            for r in rows
        ]
        new_ts = last_timestamp
        for m in messages:
            if m.timestamp > new_ts:
                new_ts = m.timestamp
        return messages, new_ts

    def get_messages_since(self, chat_jid: str, since_ts: str, bot_prefix: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me "
            "FROM messages WHERE chat_jid = ? AND timestamp > ? "
            "AND is_bot_message = 0 AND content NOT LIKE ? ORDER BY timestamp",
            (chat_jid, since_ts, f"{bot_prefix}:%"),
        ).fetchall()
        return [
            Message(
                id=r["id"], chat_jid=r["chat_jid"], sender=r["sender"],
                sender_name=r["sender_name"], content=r["content"],
                timestamp=r["timestamp"], is_from_me=bool(r["is_from_me"]),
            )
            for r in rows
        ]

    # --- Router state ---
    def get_router_state(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM router_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_router_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO router_state (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    # --- Sessions ---
    def get_session(self, group_folder: str) -> str | None:
        row = self.conn.execute(
            "SELECT session_id FROM sessions WHERE group_folder = ?", (group_folder,)
        ).fetchone()
        return row["session_id"] if row else None

    def set_session(self, group_folder: str, session_id: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions (group_folder, session_id) VALUES (?, ?)",
            (group_folder, session_id),
        )
        self.conn.commit()

    def get_all_sessions(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT group_folder, session_id FROM sessions").fetchall()
        return {r["group_folder"]: r["session_id"] for r in rows}

    # --- Registered groups ---
    def set_registered_group(self, jid: str, group: RegisteredGroup) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO registered_groups "
            "(jid, name, folder, trigger_pattern, added_at, requires_trigger) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (jid, group.name, group.folder, group.trigger, group.added_at,
             1 if group.requires_trigger else 0),
        )
        self.conn.commit()

    def get_registered_group(self, jid: str) -> RegisteredGroup | None:
        row = self.conn.execute(
            "SELECT * FROM registered_groups WHERE jid = ?", (jid,)
        ).fetchone()
        if not row:
            return None
        return RegisteredGroup(
            name=row["name"], folder=row["folder"], trigger=row["trigger_pattern"],
            added_at=row["added_at"],
            requires_trigger=row["requires_trigger"] == 1,
        )

    def get_all_registered_groups(self) -> dict[str, RegisteredGroup]:
        rows = self.conn.execute("SELECT * FROM registered_groups").fetchall()
        return {
            r["jid"]: RegisteredGroup(
                name=r["name"], folder=r["folder"], trigger=r["trigger_pattern"],
                added_at=r["added_at"],
                requires_trigger=r["requires_trigger"] == 1,
            )
            for r in rows
        }

    # --- Scheduled tasks ---
    def create_task(self, task: ScheduledTask) -> None:
        self.conn.execute(
            "INSERT INTO scheduled_tasks (id, group_folder, chat_jid, prompt, "
            "schedule_type, schedule_value, context_mode, next_run, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task.id, task.group_folder, task.chat_jid, task.prompt,
             task.schedule_type, task.schedule_value, task.context_mode,
             task.next_run, task.status, task.created_at),
        )
        self.conn.commit()

    def get_task_by_id(self, task_id: str) -> ScheduledTask | None:
        row = self.conn.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return None
        return ScheduledTask(
            id=row["id"], group_folder=row["group_folder"], chat_jid=row["chat_jid"],
            prompt=row["prompt"], schedule_type=row["schedule_type"],
            schedule_value=row["schedule_value"], context_mode=row["context_mode"] or "isolated",
            next_run=row["next_run"], last_run=row["last_run"],
            last_result=row["last_result"], status=row["status"],
            created_at=row["created_at"],
        )

    def get_all_tasks(self) -> list[ScheduledTask]:
        rows = self.conn.execute(
            "SELECT * FROM scheduled_tasks ORDER BY created_at DESC"
        ).fetchall()
        return [
            ScheduledTask(
                id=r["id"], group_folder=r["group_folder"], chat_jid=r["chat_jid"],
                prompt=r["prompt"], schedule_type=r["schedule_type"],
                schedule_value=r["schedule_value"], context_mode=r["context_mode"] or "isolated",
                next_run=r["next_run"], last_run=r["last_run"],
                last_result=r["last_result"], status=r["status"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_due_tasks(self, now_iso: str) -> list[ScheduledTask]:
        rows = self.conn.execute(
            "SELECT * FROM scheduled_tasks WHERE status = 'active' "
            "AND next_run IS NOT NULL AND next_run <= ? ORDER BY next_run",
            (now_iso,),
        ).fetchall()
        return [
            ScheduledTask(
                id=r["id"], group_folder=r["group_folder"], chat_jid=r["chat_jid"],
                prompt=r["prompt"], schedule_type=r["schedule_type"],
                schedule_value=r["schedule_value"], context_mode=r["context_mode"] or "isolated",
                next_run=r["next_run"], last_run=r["last_run"],
                last_result=r["last_result"], status=r["status"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def update_task(self, task_id: str, **kwargs: str | None) -> None:
        fields = []
        values: list[str | None] = []
        for k, v in kwargs.items():
            fields.append(f"{k} = ?")
            values.append(v)
        if not fields:
            return
        values.append(task_id)
        self.conn.execute(
            f"UPDATE scheduled_tasks SET {', '.join(fields)} WHERE id = ?", values
        )
        self.conn.commit()

    def update_task_after_run(self, task_id: str, next_run: str | None, last_result: str) -> None:
        now_iso = _now_iso()
        self.conn.execute(
            "UPDATE scheduled_tasks SET next_run=?, last_run=?, last_result=?, "
            "status = CASE WHEN ? IS NULL THEN 'completed' ELSE status END WHERE id=?",
            (next_run, now_iso, last_result, next_run, task_id),
        )
        self.conn.commit()

    def delete_task(self, task_id: str) -> None:
        self.conn.execute("DELETE FROM task_run_logs WHERE task_id = ?", (task_id,))
        self.conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        self.conn.commit()

    def log_task_run(self, log: TaskRunLog) -> None:
        self.conn.execute(
            "INSERT INTO task_run_logs (task_id, run_at, duration_ms, status, result, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (log.task_id, log.run_at, log.duration_ms, log.status, log.result, log.error),
        )
        self.conn.commit()

    def get_task_run_logs(self, task_id: str) -> list[TaskRunLog]:
        rows = self.conn.execute(
            "SELECT * FROM task_run_logs WHERE task_id = ? ORDER BY run_at DESC", (task_id,)
        ).fetchall()
        return [
            TaskRunLog(
                task_id=r["task_id"], run_at=r["run_at"], duration_ms=r["duration_ms"],
                status=r["status"], result=r["result"], error=r["error"],
            )
            for r in rows
        ]


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
