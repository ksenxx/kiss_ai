"""Scheduled task execution for KISSClaw."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from kiss.agents.kissclaw.config import KissClawConfig
from kiss.agents.kissclaw.db import KissClawDB
from kiss.agents.kissclaw.types import ScheduledTask, TaskRunLog

logger = logging.getLogger(__name__)


def compute_next_cron_run(cron_expr: str) -> str | None:
    """Compute the next run time for a cron expression.

    Uses croniter if available, otherwise returns None.
    """
    try:
        from croniter import croniter  # type: ignore[import-untyped]
        now = datetime.now(UTC)
        cron = croniter(cron_expr, now)
        return str(cron.get_next(datetime).isoformat())
    except ImportError:
        logger.warning("croniter not installed, cron scheduling unavailable")
        return None
    except Exception:
        logger.warning("Invalid cron expression: %s", cron_expr)
        return None


def compute_next_interval_run(interval_ms_str: str) -> str | None:
    try:
        ms = int(interval_ms_str)
        if ms <= 0:
            return None
        next_time = datetime.now(UTC).timestamp() + ms / 1000.0
        return datetime.fromtimestamp(next_time, tz=UTC).isoformat()
    except (ValueError, TypeError):
        return None


def run_scheduled_task(
    task: ScheduledTask,
    db: KissClawDB,
    config: KissClawConfig,
    send_message_fn: Callable[[str, str], None],
    agent_fn: object | None = None,
) -> bool:
    """Execute a scheduled task."""
    from kiss.agents.kissclaw.agent_runner import run_agent
    from kiss.agents.kissclaw.router import format_outbound

    start_ms = int(time.time() * 1000)
    logger.info("Running scheduled task %s for group %s", task.id, task.group_folder)

    # Build a synthetic message from the task prompt
    groups = db.get_all_registered_groups()
    group = None
    for g in groups.values():
        if g.folder == task.group_folder:
            group = g
            break

    if not group:
        error = f"Group not found: {task.group_folder}"
        logger.error(error)
        db.log_task_run(TaskRunLog(
            task_id=task.id,
            run_at=datetime.now(UTC).isoformat(),
            duration_ms=int(time.time() * 1000) - start_ms,
            status="error", error=error,
        ))
        return False

    # Format the task prompt as a message
    from kiss.agents.kissclaw.router import format_messages
    from kiss.agents.kissclaw.types import Message
    synthetic_msg = Message(
        id=f"task-{task.id}",
        chat_jid=task.chat_jid,
        sender="scheduler",
        sender_name="Scheduler",
        content=task.prompt,
        timestamp=datetime.now(UTC).isoformat(),
    )
    formatted = format_messages([synthetic_msg])

    output = run_agent(config, group.name, group.folder, formatted, agent_fn=agent_fn)

    duration_ms = int(time.time() * 1000) - start_ms
    result_text = output.result
    error = output.error  # type: ignore[assignment]

    if result_text:
        text = format_outbound(result_text)
        if text:
            send_message_fn(task.chat_jid, text)

    db.log_task_run(TaskRunLog(
        task_id=task.id,
        run_at=datetime.now(UTC).isoformat(),
        duration_ms=duration_ms,
        status="error" if error else "success",
        result=result_text,
        error=error,
    ))

    # Compute next run
    next_run: str | None = None
    if task.schedule_type == "cron":
        next_run = compute_next_cron_run(task.schedule_value)
    elif task.schedule_type == "interval":
        next_run = compute_next_interval_run(task.schedule_value)
    # 'once' tasks -> next_run stays None -> status becomes 'completed'

    summary = error or (result_text[:200] if result_text else "Completed")
    db.update_task_after_run(task.id, next_run, summary)
    return error is None


class TaskScheduler:
    """Polls for due tasks and executes them."""

    def __init__(
        self,
        db: KissClawDB,
        config: KissClawConfig,
        send_message_fn: Callable[[str, str], None],
        agent_fn: object | None = None,
    ) -> None:
        self.db = db
        self.config = config
        self.send_message_fn = send_message_fn
        self.agent_fn = agent_fn
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
        """Check for and run due tasks. Returns number of tasks run."""
        now_iso = datetime.now(UTC).isoformat()
        due_tasks = self.db.get_due_tasks(now_iso)
        count = 0
        for task in due_tasks:
            current = self.db.get_task_by_id(task.id)
            if not current or current.status != "active":
                continue
            run_scheduled_task(task, self.db, self.config, self.send_message_fn, self.agent_fn)
            count += 1
        return count

    def _loop(self) -> None:
        while self._running:
            try:
                self.poll_once()
            except Exception:
                logger.exception("Error in scheduler loop")
            time.sleep(self.config.scheduler_poll_interval)
