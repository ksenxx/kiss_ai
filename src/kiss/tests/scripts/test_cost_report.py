# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.scripts.cost_report`` on a real SQLite database."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from kiss.scripts import cost_report

_SCHEMA = (
    "CREATE TABLE task_history (id TEXT PRIMARY KEY, timestamp REAL NOT NULL, "
    "task TEXT NOT NULL, result TEXT DEFAULT '', model TEXT DEFAULT '', "
    "tokens INTEGER DEFAULT 0, cost REAL DEFAULT 0.0, steps INTEGER DEFAULT 0, "
    "parent_task_id TEXT DEFAULT '');"
    "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, "
    "seq INTEGER NOT NULL, event_json TEXT NOT NULL, timestamp REAL NOT NULL);"
)


def _usage(
    step: int, context: int, cost: float, cache_read: int | None = None,
    model: str = "claude-test",
) -> dict:
    event = {
        "type": "usage_info",
        "text": (
            f"Steps: {step}/100, Context: {context:,}/500,000 tokens, "
            f"Budget: ${cost:.4f}/$10.00, "
        ),
        "cost": f"${cost:.4f}",
        "total_steps": step,
        "model": model,
    }
    if cache_read is not None:
        event["cache_read"] = cache_read
    return event


def _seed(db: Path, now: float) -> None:
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    tasks = [
        # id, parent, task, model, cost, tokens, steps
        ("root", "", "Implement the feature", "claude-fable-5", 10.0, 1000, 20),
        ("rev", "root", "Review the diff for bugs", "gpt-5.6-sol", 6.0, 500, 5),
        ("rev2", "rev", "Summarize files", "gpt-5.6-sol", 2.0, 100, 2),
        ("wrap", "root", "Run this exact bash command: uv run pytest -q", "claude-fable-5",
         0.3, 50, 3),
        ("old", "", "Ancient task", "claude-fable-5", 99.0, 9, 9),
        ("orphan", "old", "Late child of an old task", "m", 1.0, 10, 1),
        ("cycle_a", "cycle_b", "a", "m", 1.0, 1, 1),
        ("cycle_b", "cycle_a", "b", "m", 1.0, 1, 1),
    ]
    for tid, parent, task, model, cost, tokens, steps in tasks:
        ts = now - 30 * 86400 if tid == "old" else now - 60
        conn.execute(
            "INSERT INTO task_history VALUES (?,?,?,?,?,?,?,?,?)",
            (tid, ts, task, "", model, tokens, cost, steps, parent),
        )
    root_events = [
        {"type": "prompt", "text": "task"},
        _usage(1, 20_000, 0.10),
        {"type": "tool_call", "name": "Read", "path": "./SORCAR.md"},
        _usage(2, 60_000, 0.30, cache_read=0),
        {"type": "tool_call", "name": "Read", "path": "/repo/a.py"},
        _usage(3, 150_000, 0.70, cache_read=1000),
        {"type": "tool_call", "name": "Read", "path": "/repo/a.py"},
        _usage(4, 250_000, 1.20, cache_read=2000),
        {"type": "tool_call", "name": "Read", "file_path": "/home/u/.kiss/SORCAR.md"},
        _usage(5, 420_000, 2.00),
        {"type": "prompt", "text": "task <h3>Previous Session 1</h3>"},
        _usage(6, 30_000, 2.10),
        {"type": "usage_info", "text": "Steps: 7/100"},
        "not json",
        {"type": "prompt", "text": "another restart at low context"},
        json.dumps([1, 2]),
        # A fan-out: the live monitor's usage_info (no step advance) and the
        # step right after run_parallel carry the children's spend.
        {"type": "tool_call", "name": "run_parallel", "tasks": "[...]"},
        _usage(6, 30_000, 4.00),
        _usage(7, 40_000, 5.00),
        {"type": "tool_call", "name": "Bash", "command": "ls"},
        _usage(8, 45_000, 5.10, cache_read=5, model="gpt-x"),
    ]
    sub_events = [_usage(1, 12_000, 0.05), _usage(2, 13_000, 0.10)]
    for tid, events in (("root", root_events), ("rev", sub_events), ("wrap", sub_events)):
        for seq, event in enumerate(events):
            raw = event if isinstance(event, str) else json.dumps(event)
            conn.execute(
                "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?,?,?,?)",
                (tid, seq, raw, now),
            )
    conn.commit()
    conn.close()


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db = tmp_path / "sorcar.db"
    _seed(db, time.time())
    return db


def test_kpis_from_seeded_db(seeded_db: Path) -> None:
    rows = cost_report.load_tasks(str(seeded_db), time.time() - 86400)
    kpis = cost_report.compute_kpis(rows)
    # "old" is outside the window but loaded (not counted) as the orphan's ancestor.
    assert rows["old"].in_window is False and rows["orphan"].in_window is True
    assert kpis["tasks"] == 7 and kpis["top_level_tasks"] == 1 and kpis["subagents"] == 6
    # Parent rows fold their children's spend: root ($10, includes rev/rev2/
    # wrap) + orphan ($1, its parent is outside the window); the cycle rows
    # each have an in-window parent and are not counted twice.
    assert kpis["cost_usd"] == pytest.approx(11.0)
    assert kpis["sorcar_md_reads"] == 1  # the ~/.kiss one is not counted
    assert kpis["reads"] == 4 and kpis["repeat_reads"] == 1
    assert kpis["repeat_read_ratio"] == 0.25
    # Root steps 1-6 and 8 are its own; step 7 (after run_parallel) and the
    # repeated step-6 monitor event are not; the two children add 2 each.
    assert kpis["steps_by_context_bucket"] == {
        "<50k": 1 + 1 + 1 + 4, "50-100k": 1, "100-200k": 1, "200-300k": 1, ">300k": 1,
    }
    cost_buckets = kpis["cost_by_context_bucket"]
    assert cost_buckets[">300k"] == pytest.approx(0.80)
    assert cost_buckets["200-300k"] == pytest.approx(0.50)
    assert cost_buckets["<50k"] == pytest.approx(0.10 + 0.10 + 0.10 + 2 * 0.10)
    assert kpis["steps_over_200k_ratio"] == pytest.approx(2 / 11, abs=1e-4)
    assert kpis["subagent_step1_context_avg"] == 12_000
    # Two restarts: one at 420k/500k (a hand-off), one at 30k (not).
    assert kpis["session_restarts"] == 2 and kpis["context_handoffs"] == 1
    # Topmost reviewer row rev ($6, includes rev2) of the $10 root tree.
    assert kpis["reviewer_trees"] == 1
    assert kpis["reviewer_share_max"] == pytest.approx(0.6)
    assert kpis["reviewer_trees_over_half"] == 1
    assert kpis["reviewer_nesting_max"] == 1
    assert kpis["shell_wrapper_subagents"] == 1
    assert kpis["shell_wrapper_cost_usd"] == pytest.approx(0.3)
    # cache_read on Claude steps 2, 3, 4 of root (hits on 3 and 4); the
    # gpt-x step 8 is not an Anthropic step and is ignored.
    assert kpis["cache_field_steps"] == 3
    assert kpis["cache_hit_ratio"] == pytest.approx(2 / 3, abs=1e-4)


def test_empty_window(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    conn.close()
    kpis = cost_report.compute_kpis(cost_report.load_tasks(str(db), 0))
    assert kpis["tasks"] == 0 and kpis["repeat_read_ratio"] == 0.0
    assert kpis["cache_hit_ratio"] is None and kpis["reviewer_nesting_max"] == 0
    assert "tasks" in cost_report.format_report(kpis, 24)


def test_main_text_and_json(seeded_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cost_report.main(["--db", str(seeded_db), "--hours", "48"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("Cost report for the last 48 h")
    assert "steps_by_context_bucket" in text and "<50k=7" in text
    assert cost_report.main(["--db", str(seeded_db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tasks"] == 7


def test_main_missing_db(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cost_report.main(["--db", str(tmp_path / "nope.db")])
