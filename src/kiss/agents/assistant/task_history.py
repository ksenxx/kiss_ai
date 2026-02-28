"""Task history, proposals, and model usage persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_KISS_DIR = Path.home() / ".kiss"
HISTORY_FILE = _KISS_DIR / "task_history.json"
PROPOSALS_FILE = _KISS_DIR / "proposed_tasks.json"
MODEL_USAGE_FILE = _KISS_DIR / "model_usage.json"
MAX_HISTORY = 1000

SAMPLE_TASKS = [
    {"task": "run 'uv run check' and fix", "result": ""},
    {
        "task": (
            "plan a trip to Yosemite over the weekend based on"
            " warnings and hotel availability"
        ),
        "result": "",
    },
    {
        "task": (
            "find the cheapest afternoon non-stop flight"
            " from SFO to NYC around March 15"
        ),
        "result": "",
    },
    {
        "task": (
            "run <<command>> in the background, monitor output,"
            " fix errors, and optimize the code iteratively. "
        ),
        "result": "",
    },
    {
        "task": (
            "implement and validate results from the research"
            " paper https://arxiv.org/pdf/2505.10961 using relentless_coding_agent and kiss_agent"
        ),
        "result": "",
    },
    {
        "task": (
            "develop an automated evaluation framework for"
            " agent performance against benchmarks"
        ),
        "result": "",
    },
    {
        "task": (
            "launch a browser, research technical innovations,"
            " and compile a document incrementally"
        ),
        "result": "",
    },
    {
        "task": (
            "read all *.md files, check consistency with"
            " the code, and fix any inconsistencies"
        ),
        "result": "",
    },
    {
        "task": (
            "remove duplicate or redundant tests while"
            " ensuring coverage doesn't decrease"
        ),
        "result": "",
    },
]


_history_cache: list[dict[str, str]] | None = None


def _load_history() -> list[dict[str, str]]:
    global _history_cache
    if _history_cache is not None:
        return _history_cache
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            if isinstance(data, list) and data:
                seen: set[str] = set()
                result: list[dict[str, str]] = []
                for t in data[:MAX_HISTORY]:
                    task_str = t["task"]
                    if task_str not in seen:
                        seen.add(task_str)
                        result.append(t)
                _history_cache = result
                return result
        except (json.JSONDecodeError, OSError):
            pass
    _save_history(SAMPLE_TASKS)
    return _history_cache


def _save_history(entries: list[dict[str, str]]) -> None:
    global _history_cache
    _history_cache = entries[:MAX_HISTORY]
    try:
        _KISS_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(_history_cache, indent=2))
    except OSError:
        pass


def _set_latest_result(result: str) -> None:
    history = _load_history()
    if history:
        history[0]["result"] = result
        _save_history(history)


def _load_proposals() -> list[str]:
    if PROPOSALS_FILE.exists():
        try:
            data = json.loads(PROPOSALS_FILE.read_text())
            if isinstance(data, list):
                return [str(t) for t in data if isinstance(t, str) and t.strip()][:5]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_proposals(proposals: list[str]) -> None:
    try:
        _KISS_DIR.mkdir(parents=True, exist_ok=True)
        PROPOSALS_FILE.write_text(json.dumps(proposals))
    except OSError:
        pass


def _read_model_usage_file() -> dict:
    if MODEL_USAGE_FILE.exists():
        try:
            data = json.loads(MODEL_USAGE_FILE.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _load_model_usage() -> dict[str, int]:
    raw = _read_model_usage_file()
    return {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _load_last_model() -> str:
    last = _read_model_usage_file().get("_last")
    return last if isinstance(last, str) else ""


def _record_model_usage(model: str) -> None:
    usage = _read_model_usage_file()
    usage[model] = int(usage.get(model, 0)) + 1
    usage["_last"] = model
    try:
        _KISS_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_USAGE_FILE.write_text(json.dumps(usage))
    except OSError:
        pass


def _add_task(task: str) -> None:
    history = [e for e in _load_history() if e["task"] != task]
    history.insert(0, {"task": task, "result": ""})
    _save_history(history[:MAX_HISTORY])


def _get_task_history_md_path() -> Path:
    from kiss.core import config as config_module
    return Path(config_module.DEFAULT_CONFIG.agent.artifact_dir).parent / "TASK_HISTORY.md"


def _init_task_history_md() -> Path:
    path = _get_task_history_md_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Task History\n\n")
    return path


def _append_task_to_md(task: str, result: str) -> None:
    path = _init_task_history_md()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"## [{timestamp}] {task}\n\n### Result\n\n{result}\n\n---\n\n"
    with path.open("a") as f:
        f.write(entry)
