"""Data types for KISSClaw, mirroring NanoClaw's types.ts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    id: str
    chat_jid: str
    sender: str
    sender_name: str
    content: str
    timestamp: str
    is_from_me: bool = False
    is_bot_message: bool = False


@dataclass
class RegisteredGroup:
    name: str
    folder: str
    trigger: str
    added_at: str
    requires_trigger: bool = True


@dataclass
class ScheduledTask:
    id: str
    group_folder: str
    chat_jid: str
    prompt: str
    schedule_type: str  # 'cron', 'interval', 'once'
    schedule_value: str
    context_mode: str = "isolated"  # 'group' or 'isolated'
    next_run: str | None = None
    last_run: str | None = None
    last_result: str | None = None
    status: str = "active"  # 'active', 'paused', 'completed'
    created_at: str = ""


@dataclass
class TaskRunLog:
    task_id: str
    run_at: str
    duration_ms: int
    status: str  # 'success' or 'error'
    result: str | None = None
    error: str | None = None


@dataclass
class AgentOutput:
    status: str  # 'success' or 'error'
    result: str | None = None
    error: str | None = None


@dataclass
class ChatInfo:
    jid: str
    name: str
    last_message_time: str
