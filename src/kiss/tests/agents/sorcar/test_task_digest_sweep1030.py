# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of two misleading task digests.

* The rsi7d agent's ``run_overview`` told the model to page the transcript
  with ``task_transcript(...)`` and ``task_step(...)``, tools the rsi7d
  agent does not have (it has ``run_transcript`` and ``run_entry``).
* The ``/ask`` context labelled whatever progress log was newest in the
  task's work dir "written by the task", although work dirs are shared:
  a log last modified before the task started or after it ended belongs
  to another task.  The log shown is now the one the task's own ``Write``
  / ``Edit`` calls wrote (``test_task_digest_progress_log_wave3.py``).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from kiss.agents.seas.rsi7d import rsi7d_sea
from kiss.agents.sorcar import task_digest
from kiss.agents.sorcar.persistence import _add_task, _append_chat_event, _flush_chat_events


def _persist(events: list[dict[str, Any]], extra: dict[str, object]) -> str:
    """Persist a task row with *extra* columns and *events*; return its id."""
    task_id, _chat = _add_task("Refactor the parser", extra={"model": "test-model", **extra})
    for event in events:
        _append_chat_event(event, task_id=task_id)
    _flush_chat_events(task_id)
    return task_id


def _write(path: Path, text: str, mtime: float) -> None:
    """Write *text* to *path* (parents created) and set its modification time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _wrote(path: Path, text: str) -> list[dict[str, Any]]:
    """Events of the task's successful ``Write`` of *text* to *path*."""
    return [
        {"type": "tool_call", "name": "Write", "path": str(path), "content": text},
        {"type": "tool_result", "content": f"Successfully wrote {len(text)} characters"},
    ]


def test_rsi7d_overview_names_only_tools_the_rsi7d_agent_has() -> None:
    events: list[dict[str, Any]] = [{"type": "prompt", "text": "p"}]
    for i in range(15):
        events.append({"type": "tool_call", "name": "summary", "description": f"summary {i}"})
    task_id = _persist(events, {"startTs": int(time.time() * 1000)})

    text = rsi7d_sea.run_overview(task_id)

    assert "... 3 earlier summaries omitted (see run_transcript) ..." in text
    named = set(re.findall(r"(\w+)\(task_id\b", text)) | set(re.findall(r"\(see (\w+)\)", text))
    assert named == {"run_transcript", "run_entry"}
    assert named <= {tool.__name__ for tool in rsi7d_sea.Rsi7dSea().tools([])}


def test_context_shows_only_a_progress_log_written_during_the_tasks_run(tmp_path: Path) -> None:
    now = time.time()
    start, end = now - 3600, now - 1800
    _write(tmp_path / "tmp" / "PROGRESS.md", "# own log\nparser split\n", start + 60)
    # Newer, but rewritten by another task after this one ended.
    _write(tmp_path / "PROGRESS.md", "# another task's log\n", now)
    finished = _persist(
        [{"type": "prompt", "text": "p"},
         *_wrote(tmp_path / "tmp" / "PROGRESS.md", "# own log\nparser split\n")],
        {"work_dir": str(tmp_path), "startTs": int(start * 1000), "endTs": int(end * 1000)},
    )

    text = task_digest.context(finished)

    log = text.partition("== Progress log written by the task (newest entries) ==\n")[2]
    assert log.startswith("# own log\nparser split\n\n== Transcript")
    assert "another task's log" not in text


def test_context_omits_a_log_last_written_before_the_task_started(tmp_path: Path) -> None:
    now = time.time()
    _write(tmp_path / "tmp" / "PROGRESS.md", "# previous task's log\n", now - 600)
    running = _persist(
        [{"type": "prompt", "text": "p"}],
        {"work_dir": str(tmp_path), "startTs": int((now - 60) * 1000)},
    )

    text = task_digest.context(running)

    assert "Status: running" in text
    assert "Progress log" not in text and "previous task's log" not in text
    # Once the running task writes the log, it is shown.
    _write(tmp_path / "tmp" / "PROGRESS.md", "# fresh log\n", time.time())
    for event in _wrote(tmp_path / "tmp" / "PROGRESS.md", "# fresh log\n"):
        _append_chat_event(event, task_id=running)
    _flush_chat_events(running)
    assert "== Progress log written by the task (newest entries) ==\n# fresh log" in (
        task_digest.context(running)
    )
