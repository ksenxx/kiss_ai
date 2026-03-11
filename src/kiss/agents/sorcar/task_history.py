"""Task history, proposals, and model usage persistence.

Task history is stored in JSONL format (one JSON object per line) for
efficiency.  Chat events for each task are stored in separate files under
``~/.kiss/chat_events/`` and loaded on demand, keeping memory usage low
even with thousands of tasks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import socket
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_KISS_DIR = Path.home() / ".kiss"
HISTORY_FILE = _KISS_DIR / "task_history.jsonl"
_CHAT_EVENTS_DIR = _KISS_DIR / "chat_events"

MODEL_USAGE_FILE = _KISS_DIR / "model_usage.json"
MAX_HISTORY = 2000


def _ensure_kiss_dir() -> None:
    _KISS_DIR.mkdir(parents=True, exist_ok=True)


_HistoryEntry = dict[str, object]

SAMPLE_TASKS: list[_HistoryEntry] = [
    {"task": "run 'uv run check' and fix"},
    {
        "task": (
            "plan a trip to Yosemite over the weekend based on"
            " warnings and hotel availability, create an html"
            " report, and show it to me."
        ),
    },
    {
        "task": (
            "find the cheapest afternoon non-stop flight from"
            " SFO to NYC around April 15, create an html"
            " report, and show it to me."
        ),
    },
    {
        "task": (
            "implement and validate results from the research"
            " paper https://arxiv.org/pdf/2505.10961 using relentless_coding_agent and kiss_agent"
        ),
    },
    {
        "task": (
            "can you use src/kiss/scripts/redundancy_analyzer.py"
            " to get rid of redundant test methods?  Make sure"
            " that you don't decrease the overall branch coverage"
            " after removing the redundant test methods."
        ),
    },
    {
        "task": (
            "can you write integration tests (possibly"
            " running 'uv run sorcar') with no mocks or test"
            " doubles to achieve 100% branch coverage of the"
            " project files? Please check the branch coverage"
            " first for the existing tests with the coverage"
            " tool.  Then try to reach uncovered branches by"
            " crafting integration tests without any mocks, test"
            " doubles. You MUST repeat the task until you get"
            " 100% branch coverage or you cannot increase branch"
            " coverage after 10 tries."
        ),
    },
    {
        "task": (
            "find redundancy, duplication, AI slop, lack of"
            " elegant abstractions, and inconsistencies in the"
            " code of the project, and fix them. Make sure that"
            " you test every change by writing and running"
            " integration tests with no mocks or test doubles to"
            " achieve 100% branch coverage. Do not change any"
            " functionality or UI. Make that existing tests pass."
        ),
    },
    {
        "task": (
            "can you please work hard and carefully to precisly"
            " detect all actual race conditions in"
            " the project? You can add"
            " random delays within 0.1 seconds before racing"
            " events to reliably trigger a race condition to"
            " confirm a race condition."
        ),
    },
]


def _task_events_path(task: str) -> Path:
    """Return the file path for storing a task's chat events.

    Args:
        task: The task description string.

    Returns:
        Path to the chat events JSON file.
    """
    h = hashlib.sha256(task.encode()).hexdigest()[:24]
    return _CHAT_EVENTS_DIR / f"{h}.json"


_history_cache: list[_HistoryEntry] | None = None
_history_lock = threading.Lock()


def _migrate_old_format() -> None:
    """Migrate from old task_history.json to JSONL + separate chat events."""
    old_file = HISTORY_FILE.parent / "task_history.json"
    if not old_file.exists():
        return
    try:
        data = json.loads(old_file.read_text())
        if not isinstance(data, list):
            old_file.unlink(missing_ok=True)
            return
        _CHAT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        lines: list[str] = []
        for item in data[:MAX_HISTORY]:
            task = item.get("task", "")
            if not task or task in seen:
                continue
            seen.add(task)
            has_events = bool(item.get("chat_events"))
            if has_events:
                _task_events_path(task).write_text(json.dumps(item["chat_events"]))
            lines.append(json.dumps({"task": task, "has_events": has_events}))
        _ensure_kiss_dir()
        HISTORY_FILE.write_text("\n".join(lines) + "\n" if lines else "")
        old_file.unlink(missing_ok=True)
    except (json.JSONDecodeError, OSError):
        logger.debug("Exception caught", exc_info=True)


def _load_history_unlocked() -> list[_HistoryEntry]:
    """Load task history from cache or disk. Must be called with _history_lock held.

    Returns:
        List of history entries with 'task' and 'has_events' keys.
        Chat events are NOT included — use _load_task_chat_events() instead.
    """
    global _history_cache
    if _history_cache is not None:
        return _history_cache
    _migrate_old_format()
    if HISTORY_FILE.exists():
        try:
            seen: set[str] = set()
            result: list[_HistoryEntry] = []
            for line in HISTORY_FILE.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                task_str = item["task"]
                if task_str not in seen:
                    seen.add(task_str)
                    result.append(
                        {"task": task_str, "has_events": bool(item.get("has_events"))}
                    )
            if result:
                _history_cache = result[:MAX_HISTORY]
                return _history_cache
        except (json.JSONDecodeError, OSError):
            logger.debug("Exception caught", exc_info=True)
    _save_history_unlocked(
        [{"task": t["task"], "has_events": False} for t in SAMPLE_TASKS]
    )
    return _history_cache  # type: ignore[return-value]


def _load_history() -> list[_HistoryEntry]:
    """Load task history from cache or disk. Thread-safe.

    Returns:
        List of history entries with 'task' and 'has_events' keys.
    """
    with _history_lock:
        return _load_history_unlocked()


def _save_history_unlocked(entries: list[_HistoryEntry]) -> None:
    """Save history to cache and disk as JSONL. Must be called with _history_lock held."""
    global _history_cache
    _history_cache = entries[:MAX_HISTORY]
    try:
        _ensure_kiss_dir()
        lines = [
            json.dumps({"task": e["task"], "has_events": bool(e.get("has_events"))})
            for e in _history_cache
        ]
        HISTORY_FILE.write_text("\n".join(lines) + "\n" if lines else "")
    except OSError:
        logger.debug("Exception caught", exc_info=True)


def _save_history(entries: list[_HistoryEntry]) -> None:
    """Save history to cache and disk. Thread-safe."""
    with _history_lock:
        _save_history_unlocked(entries)


def _load_task_chat_events(task: str) -> list[dict[str, object]]:
    """Load chat events for a specific task from its dedicated file.

    Args:
        task: The task description string.

    Returns:
        List of chat event dicts, or empty list if none stored.
    """
    path = _task_events_path(task)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            logger.debug("Exception caught", exc_info=True)
    return []


def _set_latest_chat_events(
    events: list[dict[str, object]], task: str | None = None
) -> None:
    """Save chat events for a task to a separate file.

    Args:
        events: The chat events to store.
        task: If given, find the history entry by task name.
              Otherwise update history[0].
    """
    with _history_lock:
        if not _history_cache:
            return
        target_task: str
        if task:
            for entry in _history_cache:
                if entry["task"] == task:
                    entry["has_events"] = bool(events)
                    target_task = str(entry["task"])
                    break
            else:
                return
        else:
            _history_cache[0]["has_events"] = bool(events)
            _history_cache[0].pop("result", None)
            target_task = str(_history_cache[0]["task"])
        _save_history_unlocked(_history_cache)
    # Write events to separate file outside the lock
    if events:
        try:
            _CHAT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
            _task_events_path(target_task).write_text(json.dumps(events))
        except OSError:
            logger.debug("Exception caught", exc_info=True)
    else:
        try:
            _task_events_path(target_task).unlink(missing_ok=True)
        except OSError:
            logger.debug("Exception caught", exc_info=True)



def _load_json_dict(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            logger.debug("Exception caught", exc_info=True)
    return {}


def _int_values(raw: dict) -> dict[str, int]:
    return {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _load_usage(path: Path) -> dict[str, int]:
    return _int_values(_load_json_dict(path))


def _load_model_usage() -> dict[str, int]:
    return _load_usage(MODEL_USAGE_FILE)


def _load_last_model() -> str:
    last = _load_json_dict(MODEL_USAGE_FILE).get("_last")
    return last if isinstance(last, str) else ""


def _increment_usage(file_path: Path, key: str, extra: dict[str, object] | None = None) -> None:
    """Increment a usage counter in a JSON file and optionally set extra keys.

    Args:
        file_path: Path to the JSON usage file.
        key: The key whose integer count to increment.
        extra: Optional extra key-value pairs to merge into the file.
    """
    usage = _load_json_dict(file_path)
    usage[key] = int(usage.get(key, 0)) + 1
    if extra:
        usage.update(extra)
    try:
        _ensure_kiss_dir()
        file_path.write_text(json.dumps(usage))
    except OSError:
        logger.debug("Exception caught", exc_info=True)


def _record_model_usage(model: str) -> None:
    _increment_usage(MODEL_USAGE_FILE, model, extra={"_last": model})


FILE_USAGE_FILE = _KISS_DIR / "file_usage.json"


def _load_file_usage() -> dict[str, int]:
    return _load_usage(FILE_USAGE_FILE)


def _record_file_usage(path: str) -> None:
    """Increment the access count for a file path."""
    _increment_usage(FILE_USAGE_FILE, path)


def _add_task(task: str) -> None:
    """Add a task to history, deduplicating. Thread-safe."""
    with _history_lock:
        _load_history_unlocked()
        assert _history_cache is not None
        history = [e for e in _history_cache if e["task"] != task]
        history.insert(0, {"task": task, "has_events": False})
        _save_history_unlocked(history[:MAX_HISTORY])


def _get_task_history_md_path() -> Path:
    from kiss.core import config as config_module

    return Path(config_module.DEFAULT_CONFIG.agent.artifact_dir).parent / "TASK_HISTORY.md"


def _init_task_history_md() -> Path:
    path = _get_task_history_md_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Task History\n\n")
    return path


def _append_task_to_md(task: str, result: str) -> None:
    path = _get_task_history_md_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Task History\n\n")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"## [{timestamp}] {task}\n\n### Result\n\n{result}\n\n---\n\n"
    with path.open("a") as f:
        f.write(entry)


def _cleanup_stale_cs_dirs(max_age_hours: int = 24) -> int:
    """Remove stale code-server data directories.

    Scans ``~/.kiss/cs-*`` directories and removes those that are older
    than ``max_age_hours`` and have no active process on their port.

    Args:
        max_age_hours: Maximum age in hours before a directory is eligible
            for cleanup.

    Returns:
        Number of directories removed.
    """
    threshold = time.time() - max_age_hours * 3600
    removed = 0
    for d in sorted(_KISS_DIR.glob("cs-*")):
        if not d.is_dir():
            continue
        try:
            if d.stat().st_mtime > threshold:
                continue
            # Check if a process is still listening on the port
            port_file = d / "cs-port"
            if port_file.exists():
                try:
                    port = int(port_file.read_text().strip())
                    with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                        continue  # Still in use
                except (ConnectionRefusedError, OSError, ValueError):
                    pass
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
        except OSError:
            logger.debug("Exception caught", exc_info=True)
    return removed
