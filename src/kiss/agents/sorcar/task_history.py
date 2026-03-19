"""Task history, proposals, and model usage persistence.

Task history is stored in JSONL format (one JSON object per line) for
efficiency.  Chat events for each task are stored in separate files under
``~/.kiss/chat_events/`` and loaded on demand, keeping memory usage low
even with millions of tasks.

The file stores entries in chronological order (oldest first, newest
last).  New entries are appended to the end of the file, avoiding
full-file rewrites.  A small in-memory cache (``_RECENT_CACHE_SIZE``
entries, most-recent-first) avoids file I/O for the common case.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger(__name__)


def _log_exc() -> None:
    logger.debug("Exception caught", exc_info=True)

_KISS_DIR = Path.home() / ".kiss"
HISTORY_FILE = _KISS_DIR / "task_history.jsonl"
_CHAT_EVENTS_DIR = _KISS_DIR / "chat_events"

MODEL_USAGE_FILE = _KISS_DIR / "model_usage.json"
MAX_HISTORY = 1_000_000

# Only keep this many entries in memory for fast access
_RECENT_CACHE_SIZE = 500


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
            " to get rid of redundant test methods?  Make sure"
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
            " tool.  Then try to reach uncovered branches by"
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


def _new_events_filename() -> str:
    """Generate a unique non-existent filename for storing chat events.

    Returns:
        Filename string (e.g. ``evt_abcdef1234567890.json``).
    """
    while True:  # pragma: no branch – UUID collision is astronomically unlikely
        name = f"evt_{uuid.uuid4().hex[:16]}.json"
        if not (_CHAT_EVENTS_DIR / name).exists():  # pragma: no branch
            return name


def _task_events_path(task: str) -> Path:
    """Return the file path for a task's chat events by looking up the history.

    Searches the in-memory cache for the task's ``events_file`` field.
    Returns a non-existent path if the task is not found.

    Args:
        task: The task description string.

    Returns:
        Path to the chat events JSON file.
    """
    with _HISTORY_LOCK:
        if _history_cache is None:
            _refresh_cache()
        assert _history_cache is not None
        for entry in _history_cache:
            if entry["task"] == task:
                filename = str(entry.get("events_file", ""))
                if filename:
                    return _CHAT_EVENTS_DIR / filename
    return _CHAT_EVENTS_DIR / "nonexistent.json"


# In-memory cache of the most recent entries (most-recent-first).
_history_cache: list[_HistoryEntry] | None = None
# Single cross-process + cross-thread lock for both the file and the in-memory cache.
_HISTORY_LOCK = FileLock(HISTORY_FILE.with_suffix(".lock"))


def _migrate_old_format() -> None:
    """Migrate from old task_history.json to chronological JSONL.

    The old JSON array stored entries most-recent-first.  We reverse
    so the JSONL file is chronological (oldest first, newest last).
    """
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
        # Reverse: old format was most-recent-first, new is chronological
        for item in reversed(data[:MAX_HISTORY]):
            task = item.get("task", "")
            if not task or task in seen:
                continue
            seen.add(task)
            has_events = bool(item.get("chat_events"))
            events_file = _new_events_filename()
            if has_events:
                (_CHAT_EVENTS_DIR / events_file).write_text(
                    json.dumps(item["chat_events"])
                )
            lines.append(
                json.dumps({
                    "task": task,
                    "has_events": has_events,
                    "result": "",
                    "events_file": events_file,
                })
            )
        _ensure_kiss_dir()
        HISTORY_FILE.write_text(
            "\n".join(lines) + "\n" if lines else ""
        )
        old_file.unlink(missing_ok=True)
    except (json.JSONDecodeError, OSError):
        _log_exc()


def _parse_line(line: str) -> _HistoryEntry | None:
    """Parse a JSONL line into a history entry or None on failure."""
    line = line.strip()
    if not line:
        return None
    try:
        item = json.loads(line)
        task = item["task"]
        return {
            "task": task,
            "has_events": bool(item.get("has_events")),
            "result": item.get("result", ""),
            "events_file": item.get("events_file", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return None


def _iter_lines_reverse(path: Path) -> Iterator[str]:
    """Yield non-empty lines from *path* in reverse order (last line first).

    Reads the file in chunks from the end so only a small amount of data
    is in memory at any time.

    Args:
        path: Path to the file to read.

    Yields:
        Stripped line strings, starting from the last line.
    """
    with path.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        if pos == 0:
            return
        remaining = b""
        chunk_size = 8192
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size) + remaining
            parts = chunk.split(b"\n")
            remaining = parts[0]
            for part in reversed(parts[1:]):
                stripped = part.strip()
                if stripped:
                    yield stripped.decode("utf-8")
        if remaining.strip():  # pragma: no branch – files normally end with newline
            yield remaining.strip().decode("utf-8")


def _read_recent_entries(
    limit: int,
) -> list[_HistoryEntry]:
    """Read the most recent *limit* unique entries, most-recent-first.

    Reads the file from the end so only a small window is in memory.

    Args:
        limit: Maximum unique entries to return.

    Returns:
        List of entries, most-recent-first.
    """
    if not HISTORY_FILE.exists():
        return []
    seen: set[str] = set()
    result: list[_HistoryEntry] = []
    try:
        for raw_line in _iter_lines_reverse(HISTORY_FILE):
            entry = _parse_line(raw_line)
            if entry is None:
                continue
            task_str = str(entry["task"])
            if task_str in seen:
                continue
            seen.add(task_str)
            result.append(entry)
            if len(result) >= limit:
                break
    except OSError:  # pragma: no cover
        _log_exc()
    return result


def _read_file_entries(
    limit: int = 0,
) -> list[_HistoryEntry]:
    """Read all entries forward, dedup keeping last occurrence, most-recent-first.

    Args:
        limit: Maximum unique entries to return.  0 means all
            (up to MAX_HISTORY).

    Returns:
        List of unique entries, most-recent-first.
    """
    if not HISTORY_FILE.exists():
        return []
    cap = limit if limit > 0 else MAX_HISTORY
    entries: dict[str, _HistoryEntry] = {}
    try:
        with HISTORY_FILE.open() as f:
            for raw_line in f:
                entry = _parse_line(raw_line)
                if entry is None:
                    continue
                key = str(entry["task"])
                # Remove then re-insert so the key moves to the end
                entries.pop(key, None)
                entries[key] = entry
    except OSError:  # pragma: no cover
        _log_exc()
    # Most-recent-first: reverse chronological order
    all_entries = list(reversed(entries.values()))
    return all_entries[:cap]


def _seed_sample_tasks() -> None:
    """Write SAMPLE_TASKS to history file on first run.  Must hold _HISTORY_LOCK."""
    _ensure_kiss_dir()
    _CHAT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a") as f:
        for sample in SAMPLE_TASKS:
            entry: _HistoryEntry = {
                "task": sample["task"],
                "has_events": False,
                "result": "",
                "events_file": _new_events_filename(),
            }
            f.write(json.dumps(entry) + "\n")


def _refresh_cache() -> list[_HistoryEntry]:
    """Reload the recent cache from disk.  Must hold _HISTORY_LOCK."""
    global _history_cache
    _migrate_old_format()
    entries = _read_recent_entries(_RECENT_CACHE_SIZE)
    if not entries:
        # First run after installation — persist sample tasks to disk
        _seed_sample_tasks()
        entries = _read_recent_entries(_RECENT_CACHE_SIZE)
    _history_cache = entries
    return _history_cache


def _load_history(limit: int = 0) -> list[_HistoryEntry]:
    """Load task history entries (most-recent-first). Thread-safe.

    Args:
        limit: Maximum number of entries to return.
            0 returns all entries (up to MAX_HISTORY).
            If limit <= _RECENT_CACHE_SIZE, served from the in-memory
            cache without file I/O.

    Returns:
        List of history entries with 'task' and 'has_events' keys.
    """
    with _HISTORY_LOCK:
        if _history_cache is None:
            _refresh_cache()
        assert _history_cache is not None
        if limit <= 0:
            # Caller wants all — full forward read, dedup, reverse
            entries = _read_file_entries(limit=0)
            return entries if entries else list(_history_cache)
        if limit <= len(_history_cache):
            return _history_cache[:limit]
        # Need more than cache has — read from tail
        entries = _read_recent_entries(limit)
        return entries if entries else list(_history_cache)



def _search_history(
    query: str, limit: int = 50
) -> list[_HistoryEntry]:
    """Search history entries by substring match. Thread-safe.

    Reads from the end of the file so the most recent matches are
    returned first, without loading all entries into memory.

    Args:
        query: Case-insensitive substring to match against task text.
        limit: Maximum number of matching entries to return.

    Returns:
        List of matching entries, most-recent-first.
    """
    if not query:
        return _load_history(limit=limit)
    q_lower = query.lower()
    with _HISTORY_LOCK:
        if _history_cache is None:
            _refresh_cache()
    if not HISTORY_FILE.exists():
        return []
    seen: set[str] = set()
    results: list[_HistoryEntry] = []
    try:
        for raw_line in _iter_lines_reverse(HISTORY_FILE):
            entry = _parse_line(raw_line)
            if entry is None:
                continue
            task_str = str(entry["task"])
            if task_str in seen:
                continue
            seen.add(task_str)
            if q_lower in task_str.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
    except OSError:  # pragma: no cover
        _log_exc()
    return results


def _get_history_entry(idx: int) -> _HistoryEntry | None:
    """Get a single history entry by its index (0 = most recent). Thread-safe.

    Args:
        idx: Zero-based index into the history list.

    Returns:
        The entry, or None if index is out of range.
    """
    with _HISTORY_LOCK:
        if _history_cache is None:
            _refresh_cache()
        assert _history_cache is not None
        if 0 <= idx < len(_history_cache):
            return _history_cache[idx]
    # Beyond cache — read from tail of file
    if not HISTORY_FILE.exists():
        return None
    seen: set[str] = set()
    count = 0
    try:
        for raw_line in _iter_lines_reverse(HISTORY_FILE):
            entry = _parse_line(raw_line)
            if entry is None:
                continue
            task_str = str(entry["task"])
            if task_str in seen:
                continue
            seen.add(task_str)
            if count == idx:
                return entry
            count += 1
    except OSError:  # pragma: no cover
        _log_exc()
    return None



def _load_task_chat_events(
    task: str,
) -> list[dict[str, object]]:
    """Load chat events for a specific task from its dedicated file.

    Looks up the ``events_file`` from the task history entry via
    :func:`_task_events_path`.  Returns an empty list if the task is
    not found in the history or the file does not exist.

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
            _log_exc()
    return []


def _find_cache_entry(task: str | None) -> _HistoryEntry | None:
    """Find a cache entry by task name or return the most recent entry.

    Must be called with ``_HISTORY_LOCK`` held and cache initialised.

    Args:
        task: Task name to search for, or ``None`` for the most recent.

    Returns:
        The matching entry, or ``None`` if not found / cache empty.
    """
    if not _history_cache:
        return None
    if task:
        for entry in _history_cache:
            if entry["task"] == task:
                return entry
        return None
    return _history_cache[0]


def _append_entry_to_file(
    task: str, has_events: bool, result: str, events_file: str,
) -> None:
    """Append an entry to the JSONL history file.  Must hold ``_HISTORY_LOCK``."""
    _ensure_kiss_dir()
    entry_dict: _HistoryEntry = {
        "task": task,
        "has_events": has_events,
        "result": result,
        "events_file": events_file,
    }
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(entry_dict) + "\n")


def _set_latest_chat_events(
    events: list[dict[str, object]],
    task: str | None = None,
    result: str = "",
) -> None:
    """Save chat events for a task to a separate file.

    Appends an updated entry to the history file (instead of
    rewriting) so the dedup logic picks up the new has_events flag.

    Args:
        events: The chat events to store.
        task: If given, find the history entry by task name.
              Otherwise update history[0].
        result: The task result text to store in the history entry.
    """
    with _HISTORY_LOCK:
        if _history_cache is None:
            _refresh_cache()
        entry = _find_cache_entry(task)
        if entry is None:
            return
        entry["has_events"] = bool(events)
        if task and result:
            entry["result"] = result
        elif not task:
            entry.pop("result", None)
        events_file = str(entry.get("events_file", ""))
        _append_entry_to_file(
            str(entry["task"]), bool(events), result, events_file,
        )
    # Write events to separate file outside the lock using atomic replace
    if not events_file:
        return
    path = _CHAT_EVENTS_DIR / events_file
    if events:
        try:
            _CHAT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(path, events)
        except OSError:  # pragma: no cover
            _log_exc()
    else:
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            _log_exc()



def _load_json_dict(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            _log_exc()
    return {}


def _int_values(raw: dict) -> dict[str, int]:
    return {
        str(k): int(v)
        for k, v in raw.items()
        if isinstance(v, (int, float))
    }


def _load_usage(path: Path) -> dict[str, int]:
    return _int_values(_load_json_dict(path))


def _load_model_usage() -> dict[str, int]:
    return _load_usage(MODEL_USAGE_FILE)


def _load_last_model() -> str:
    last = _load_json_dict(MODEL_USAGE_FILE).get("_last")
    return last if isinstance(last, str) else ""


def _atomic_write_json(path: Path, data: object) -> None:
    """Write *data* as JSON to *path* atomically (write-tmp-then-replace)."""
    tmp = path.with_suffix(".tmp")
    try:
        _ensure_kiss_dir()
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
    except OSError:  # pragma: no cover
        _log_exc()
        tmp.unlink(missing_ok=True)


def _update_json_locked(path: Path, fn: Callable[[dict[str, object]], None]) -> None:
    """Read-modify-write a JSON dict file under a cross-process file lock.

    Args:
        path: Path to the JSON file.
        fn: Mutator that receives the loaded dict and modifies it in-place.
    """
    try:
        _ensure_kiss_dir()
        with FileLock(path.with_suffix(".lock")):
            data = _load_json_dict(path)
            fn(data)
            _atomic_write_json(path, data)
    except OSError:  # pragma: no cover
        _log_exc()


def _increment_usage(
    file_path: Path,
    key: str,
    extra: dict[str, object] | None = None,
) -> None:
    """Increment a usage counter in a JSON file and optionally set extra keys.

    Args:
        file_path: Path to the JSON usage file.
        key: The key whose integer count to increment.
        extra: Optional extra key-value pairs to merge into the file.
    """
    def _update(data: dict[str, object]) -> None:
        data[key] = int(data.get(key, 0)) + 1  # type: ignore[call-overload]
        if extra:
            data.update(extra)

    _update_json_locked(file_path, _update)


def _save_last_model(model: str) -> None:
    """Persist the selected model name without incrementing usage count.

    Args:
        model: The model name to save as the last-selected model.
    """
    _update_json_locked(MODEL_USAGE_FILE, lambda d: d.update({"_last": model}))


def _record_model_usage(model: str) -> None:
    _increment_usage(
        MODEL_USAGE_FILE, model, extra={"_last": model}
    )


FILE_USAGE_FILE = _KISS_DIR / "file_usage.json"
_MAX_FILE_USAGE_ENTRIES = 1000


def _load_file_usage() -> dict[str, int]:
    return _load_usage(FILE_USAGE_FILE)


def _record_file_usage(path: str) -> None:
    """Increment the access count for a file path.

    Moves the key to the end of the JSON dict so that insertion order
    reflects recency (most recently used file is last).  Keeps at most
    ``_MAX_FILE_USAGE_ENTRIES`` entries, evicting the least recently
    used (earliest in insertion order) when the limit is exceeded.
    """
    def _update(data: dict[str, object]) -> None:
        old_val = data.pop(path, 0)
        data[path] = int(old_val) + 1  # type: ignore[call-overload]
        # Evict least recently used entries (front of dict) if over limit.
        excess = len(data) - _MAX_FILE_USAGE_ENTRIES
        if excess > 0:
            for key in list(data)[:excess]:
                del data[key]

    _update_json_locked(FILE_USAGE_FILE, _update)


def _add_task(task: str) -> None:
    """Append a task to the history file. Thread-safe.

    Appends one line to the end of the file.  Duplicate entries are
    handled during reads (the last occurrence wins).

    Args:
        task: The task description string.
    """
    entry: _HistoryEntry = {
        "task": task,
        "has_events": False,
        "result": "",
        "events_file": _new_events_filename(),
    }

    with _HISTORY_LOCK:
        if _history_cache is None:
            _refresh_cache()
        _ensure_kiss_dir()
        with HISTORY_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        # Update in-memory cache (most-recent-first)
        assert _history_cache is not None
        _history_cache[:] = [
            e for e in _history_cache if e["task"] != task
        ]
        _history_cache.insert(0, entry)
        if len(_history_cache) > _RECENT_CACHE_SIZE:
            _history_cache[:] = (
                _history_cache[:_RECENT_CACHE_SIZE]
            )



def _cleanup_stale_cs_dirs(max_age_hours: int = 24) -> int:
    """Remove the code-server data directory if stale.

    Checks ``~/.kiss/cs-data`` and removes it if older than
    ``max_age_hours`` and no process is listening on its port.
    Also removes any legacy ``~/.kiss/cs-*`` per-workdir directories
    and ``cs-port-*`` files.

    Args:
        max_age_hours: Maximum age in hours before the directory is
            eligible for cleanup.

    Returns:
        Number of directories removed.
    """
    threshold = time.time() - max_age_hours * 3600
    removed = 0
    # Clean up legacy per-workdir cs-* directories and cs-port-* files
    for p in sorted(_KISS_DIR.glob("cs-port-*")):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                _log_exc()
    for d in sorted(_KISS_DIR.glob("cs-*")):
        if not d.is_dir() or d.name in ("cs-extensions", "cs-data"):
            continue
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
    # Check the single cs-data directory
    d = _KISS_DIR / "cs-data"
    if not d.is_dir():
        return removed
    try:
        if d.stat().st_mtime > threshold:
            return removed
        _pf = d / "cs-port"
        if _pf.exists():
            try:
                port = int(_pf.read_text().strip())
                with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                    return removed  # still in use
            except (ConnectionRefusedError, OSError, ValueError):
                pass
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
    except OSError:  # pragma: no cover
        _log_exc()
    return removed
