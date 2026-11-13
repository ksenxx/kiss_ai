# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.scripts.cost_report`` on a real SQLite database."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.core.config import kiss_home
from kiss.scripts import cost_report
from kiss.tests.branded_package import make_branded_package

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
        ("orphan2", "old", "Another late child of the old task", "m", 9.0, 10, 1),
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
        {"type": "tool_call", "name": "Read", "path": "./AGENTS.md"},
        {"type": "tool_call", "name": "Read", "path": "./SORCAR.md"},  # its pre-2026.10.2 name
        _usage(2, 60_000, 0.30, cache_read=0),
        {"type": "tool_call", "name": "Read", "path": "/repo/a.py"},
        _usage(3, 150_000, 0.70, cache_read=1000),
        # The same window: Read's defaults, as a range is persisted (strings in "extras").
        {"type": "tool_call", "name": "Read", "path": "/repo/a.py",
         "extras": {"start_line": "1", "max_lines": "2000"}},
        _usage(4, 250_000, 1.20, cache_read=2000),
        {"type": "tool_call", "name": "Read", "file_path": "/home/u/.kiss/AGENTS.md"},
        # Not a repository's either: this home's file (also spelled with ``~``, see
        # seeded_db), a stock home's with Windows separators, a longer name.  A
        # repository's: one on Windows, a clone inside a home (install.sh's
        # ~/.kiss/kiss_ai) and, to stock KISS, the two only Seamless Loop knows as its home's.
        *({"type": "tool_call", "name": "Read", "path": path} for path in (
            str(kiss_home() / "AGENTS.md"), f"~/{kiss_home().name}/AGENTS.md",
            "C:\\Users\\u\\.kiss\\SORCAR.md", "/repo/NOT_AGENTS.md", "C:\\repo\\AGENTS.md",
            "/home/u/.kiss/kiss_ai/AGENTS.md", "~/.s10s/AGENTS.md", "/home/u/.s10s/AGENTS.md",
        )),
        # Three windows of one file, the second read twice: one repeat.
        *({"type": "tool_call", "name": "Read", "path": "/repo/big.py",
           "extras": {"start_line": start, "max_lines": "400"}}
          for start in ("1", "401", "401", "801")),
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
        # A streamed Bash output: 9,000 chars of system_output and a 3,000
        # char persisted result make one oversized 12,000 char output.
        {"type": "tool_call", "name": "Bash", "command": "ls"},
        {"type": "system_output", "text": "x" * 5_000},
        {"type": "system_output", "text": "x" * 4_000},
        {"type": "tool_result", "tool_name": "Bash", "content": "x" * 3_000},
        _usage(8, 45_000, 5.10, cache_read=5, model="gpt-x"),
        {"type": "tool_call", "name": "Edit", "path": "/repo/a.py"},
        {"type": "tool_result", "tool_name": "Edit", "is_error": True,
         "content": "REFUSED: the Edit call did NOT run and had no effect. Read it first."},
        # Pre-2026-10-24 guard wordings, counted by their category.
        {"type": "tool_call", "name": "Bash", "command": "git status"},
        {"type": "tool_result", "tool_name": "Bash", "is_error": True, "content": (
            "Error: command references the parent-repo path '/r', which is outside"
        )},
        {"type": "tool_call", "name": "Edit", "path": "/repo/b.py"},
        {"type": "tool_result", "tool_name": "Edit", "is_error": True,
         "content": "Error: /repo/b.py has not been read in this session. Call Read first."},
        {"type": "tool_call", "name": "Bash", "command": "ls"},
        {"type": "tool_result", "tool_name": "Bash", "is_error": True, "content": (
            "Step 10 is a multiple of 10: call summary(description=...) first, recapping"
        )},
        {"type": "tool_call", "name": "Bash", "command": "sed -i"},
        {"type": "tool_result", "tool_name": "Bash", "is_error": True, "content": (
            "REFUSED: the `sed -i` call did NOT run: it writes a multi-line substitution"
        )},
        {"type": "tool_call", "name": "Bash", "command": "python3 - <<EOF"},
        {"type": "tool_result", "tool_name": "Bash", "is_error": True, "content": (
            "REFUSED: the python heredoc did NOT run: it rewrites a file with .replace("
        )},
        # The current wording: the guard's text behind REFUSED_PREFIX.
        {"type": "tool_call", "name": "Write", "path": "/repo/c.py"},
        {"type": "tool_result", "tool_name": "Write", "is_error": True, "content": (
            "REFUSED: the Write call did NOT run and had no effect. Error: /repo/c.py had "
            "not been read in this session, so the edit was NOT applied."
        )},
        # A successful Read of the guard's source quotes the wording: not a refusal.
        {"type": "tool_call", "name": "Read", "path": "/repo/useful_tools.py"},
        {"type": "tool_result", "tool_name": "Read", "is_error": False, "content": (
            '1109: f"Error: {file_path} has not been read in this session. "\n'
            "1110: Step 10 is a multiple of 10: call summary(description=...) first"
        )},
        {"type": "tool_call", "name": "summary", "description": "- did things"},
        {"type": "tool_result", "tool_name": "summary", "content": "ok"},
        {"type": "tool_call", "name": "summary", "description": "Steps 1-10 (framework recap",
         "extras": {"source": "framework"}},
        {"type": "tool_result", "tool_name": "summary", "content": "ok"},
        {"type": "llm_call", "model": "claude", "step": 5, "output_tokens": 900,
         "cost": 0.5},
        {"type": "llm_call", "model": "claude", "step": 6, "output_tokens": 300,
         "routine": True, "effort": "low", "cost": 0.25},
        {"type": "llm_call", "model": "claude", "step": 7, "output_tokens": 500,
         "routine": True, "effort": "", "cost": "$0.25"},
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
def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # ``~`` is the directory that holds the session's KISS_HOME, so the seed can spell it.
    monkeypatch.setenv("HOME", str(kiss_home().parent))
    monkeypatch.setenv("USERPROFILE", str(kiss_home().parent))
    db = tmp_path / "history.db"
    _seed(db, time.time())
    return db


def test_kpis_from_seeded_db(seeded_db: Path) -> None:
    rows = cost_report.load_tasks(str(seeded_db), time.time() - 86400)
    kpis = cost_report.compute_kpis(rows)
    # "old" is outside the window but loaded (not counted) as the orphan's ancestor.
    assert rows["old"].in_window is False and rows["orphan"].in_window is True
    assert kpis["tasks"] == 8 and kpis["top_level_tasks"] == 1 and kpis["subagents"] == 7
    # Parent rows fold their children's spend: root ($10, includes rev/rev2/
    # wrap) + orphan and orphan2 ($1 + $9, their parent is outside the
    # window); the cycle rows each have an in-window parent and are not
    # counted twice.
    assert kpis["cost_usd"] == pytest.approx(20.0)
    # ./AGENTS.md, ./SORCAR.md, the Windows one, the clone's and the two .s10s ones,
    # no home's; a.py and the second window of big.py are read twice.
    assert (kpis["agents_md_reads"], kpis["reads"], kpis["repeat_reads"]) == (6, 18, 2)
    assert kpis["repeat_read_ratio"] == pytest.approx(2 / 18, abs=1e-4)
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
    # Seven refusals (legacy and prefixed wordings of each guard plus the
    # generic prefix; the Read quoting the wordings is not one), one model
    # summary, one framework recap; eleven tool results of which the
    # streamed 12k Bash output is oversized and the p99.
    assert kpis["refused_tool_calls"] == 7
    assert kpis["refused_by_reason"] == {
        "summary_guard": 1, "unread_file": 2, "worktree_path": 1, "edit_bypass": 2, "other": 1,
    }
    assert kpis["summaries_by_model"] == 1 and kpis["summaries_by_framework"] == 1
    assert kpis["tool_outputs"] == 11 and kpis["tool_outputs_oversized"] == 1
    assert kpis["tool_output_p99_chars"] == 12_000
    # Three model calls, two routine (one at the lowered effort): 400
    # output tokens a routine call against 900 for the other.
    assert kpis["llm_calls"] == 3 and kpis["routine_calls"] == 2
    assert kpis["routine_calls_lowered"] == 1
    assert kpis["routine_output_tokens_mean"] == 400
    assert kpis["other_output_tokens_mean"] == 900
    assert kpis["shell_wrapper_subagents"] == 1
    assert kpis["shell_wrapper_cost_usd"] == pytest.approx(0.3)
    # cache_read on Claude steps 2, 3, 4 of root (hits on 3 and 4); the
    # gpt-x step 8 is not an Anthropic step and is ignored.
    assert kpis["cache_field_steps"] == 3
    assert kpis["cache_hit_ratio"] == pytest.approx(2 / 3, abs=1e-4)
    # Two trees of $10 each: the root's, and the old root's two in-window
    # children taken together.  Cost per call is the window's own llm_call
    # spend ($1 over three calls), not the tree-folded total; rates are per
    # 1k model calls.
    assert kpis["tree_cost_median_usd"] == 10.0 and kpis["tree_cost_p90_usd"] == 10.0
    assert kpis["cost_per_llm_call_usd"] == pytest.approx(1 / 3, abs=1e-4)
    assert kpis["refused_per_1k_calls"] == pytest.approx(7000 / 3, abs=0.01)
    assert kpis["summaries_by_model_per_1k_calls"] == pytest.approx(1000 / 3, abs=0.01)
    assert kpis["oversized_per_1k_calls"] == pytest.approx(1000 / 3, abs=0.01)
    assert kpis["tool_output_chars_total"] == sum(
        size for r in rows.values() for size in r.output_sizes
    ) > 12_000


def test_brand_home_instructions_are_not_repository_reads(seeded_db: Path, tmp_path: Path) -> None:
    """A Seamless Loop install knows ``.s10s`` as its home: its two files are the user's."""
    proc = subprocess.run(
        [sys.executable, "-m", "kiss.scripts.cost_report", "--db", str(seeded_db), "--json"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": str(make_branded_package(tmp_path / "brand"))},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["agents_md_reads"] == 4


def test_helper_of_an_earlier_reviewer_is_reviewer_spend(seeded_db: Path) -> None:
    """A reviewer that started before the window owns its in-window helper's spend, once."""
    now = time.time()
    conn = sqlite3.connect(seeded_db)
    conn.executemany(
        "INSERT INTO task_history (id, parent_task_id, task, cost, timestamp) VALUES (?,?,?,?,?)",
        [("oldrev", "old", "Review the old diff", 50.0, now - 30 * 86400),
         ("late", "oldrev", "Summarize files", 30.0, now - 60),
         ("late2", "late", "Count lines", 20.0, now - 50)],
    )
    conn.commit()
    conn.close()
    kpis = cost_report.compute_kpis(cost_report.load_tasks(str(seeded_db), now - 86400))
    # Of the old tree's in-window $1 + $9 + $30, late's $30 (which folds late2's $20).
    assert kpis["reviewer_trees"] == 2 and kpis["reviewer_share_max"] == pytest.approx(0.75)


def test_empty_window(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    conn.close()
    kpis = cost_report.compute_kpis(cost_report.load_tasks(str(db), 0))
    assert kpis["tasks"] == 0 and kpis["repeat_read_ratio"] == 0.0
    assert kpis["cache_hit_ratio"] is None and kpis["reviewer_nesting_max"] == 0
    assert kpis["tree_cost_median_usd"] == 0 and kpis["cost_per_llm_call_usd"] == 0.0
    assert kpis["refused_per_1k_calls"] == 0.0
    assert "tasks" in cost_report.format_report(kpis, "the last 24 h")


def test_a_window_holding_more_tasks_than_one_statement_binds(tmp_path: Path) -> None:
    """Every task of the window has its events scanned and its ancestors loaded, however many.

    One statement binds 32,766 values on a stock SQLite (250,000 on Debian's); the ids of
    the window's tasks, and then those of their parents outside it, were bound all at once,
    so a longer window died of ``too many SQL variables``.  Every task here has a parent of
    its own.
    """
    db = tmp_path / "long.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    tasks = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) + 1
    conn.execute(
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?)"
        " INSERT INTO task_history (id, timestamp, task, parent_task_id)"
        " SELECT 't' || i, 1.0, '', 'p' || i FROM n",
        (tasks,),
    )
    # The parents still in the history started before the window, one of them under a root.
    conn.executemany(
        "INSERT INTO task_history (id, timestamp, task, parent_task_id) VALUES (?, 0.0, '', ?)",
        [("p1", "root"), (f"p{tasks}", ""), ("root", "")],
    )
    conn.executemany(
        "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, 1, ?, 1.0)",
        [(task, json.dumps({"type": "llm_call"})) for task in ("t1", f"t{tasks}")],
    )
    conn.commit()
    conn.close()
    rows = cost_report.load_tasks(str(db), 0.5)
    assert len(rows) == tasks + 3
    assert [rows[task].llm_calls for task in ("t1", "t2", f"t{tasks}")] == [1, 0, 1]
    assert [rows[task].in_window for task in ("t1", "p1", f"p{tasks}", "root")] == [
        True, False, False, False,
    ]


def test_a_count_field_that_is_not_a_number_contributes_nothing(tmp_path: Path) -> None:
    """``output_tokens`` and ``total_steps`` of a foreign event count as zero, not as a crash.

    So does a ``cost`` no total can hold: NaN, an infinity, an integer beyond a float.
    """
    db = tmp_path / "foreign.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    conn.execute("INSERT INTO task_history (id, timestamp, task) VALUES ('t', 1.0, 'x')")
    events = [
        {"type": "llm_call", "output_tokens": "n/a", "cost": 0.5},
        {"type": "llm_call", "output_tokens": [7]},
        {**_usage(1, 1000, 0.5), "total_steps": {"n": 1}},
        *({"type": "llm_call", "cost": cost} for cost in (float("nan"), float("inf"), 10**400)),
    ]
    conn.executemany(
        "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES ('t', ?, ?, 1.0)",
        [(seq, json.dumps(event)) for seq, event in enumerate(events)],
    )
    conn.commit()
    conn.close()
    row = cost_report.load_tasks(str(db), 0)["t"]
    assert (row.llm_calls, row.llm_output_tokens, row.llm_cost) == (5, 0, 0.5)
    assert cost_report.compute_kpis({"t": row})["cost_per_llm_call_usd"] == 0.1
    assert (row.first_context, row.bucket_steps) == (1000, {})


def _charge_until(db: Path, stop: threading.Event) -> None:
    """Until *stop*: record a $1 ``llm_call`` of task ``t0`` and add it to its row, atomically."""
    conn = sqlite3.connect(db)
    event = json.dumps({"type": "llm_call", "cost": 1.0})
    while not stop.is_set():
        with conn:
            conn.execute("UPDATE task_history SET cost = cost + 1 WHERE id = 't0'")
            conn.execute(
                "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES ('t0', 1, ?, 1.0)",
                (event,),
            )
    conn.close()


def test_a_report_of_a_database_in_use_reads_one_snapshot(tmp_path: Path) -> None:
    """A task's totals and its events are those of one moment while the daemon writes.

    The rows and the events were read in separate statements, each of its own snapshot: a
    commit between them gave a task the cost of one moment and the calls of a later one.
    """
    db = tmp_path / "live.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")  # the daemon's mode: readers do not block it
    conn.executescript(_SCHEMA)
    conn.executemany(
        "INSERT INTO task_history (id, timestamp, task) VALUES (?, 1.0, 'x')",
        [(f"t{number}",) for number in range(200)],
    )
    conn.commit()
    conn.close()
    stop = threading.Event()
    writer = threading.Thread(target=_charge_until, args=(db, stop))
    writer.start()
    try:
        loads = [cost_report.load_tasks(str(db), 0)["t0"] for _ in range(10)]
    finally:
        stop.set()
        writer.join()
    assert [row.cost - row.llm_cost for row in loads] == [0.0] * 10
    assert loads[-1].llm_calls > loads[0].llm_calls  # the writer did commit in between


def test_absolute_window(seeded_db: Path) -> None:
    now = time.time()
    # [now-120, now-30) holds the seeded rows started at now-60; the old
    # row (30 days ago) stays an uncounted ancestor; a window ending
    # before them is empty.
    rows = cost_report.load_tasks(str(seeded_db), now - 120, now - 30)
    assert cost_report.compute_kpis(rows)["tasks"] == 8
    # An explicit zero end is a bound, not "no bound".
    assert cost_report.load_tasks(str(seeded_db), -1, 0) == {}
    assert cost_report.compute_kpis(cost_report.load_tasks(str(seeded_db), 0, now - 120))[
        "tasks"
    ] == 1  # only "old"


def test_main_text_and_json(seeded_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cost_report.main(["--db", str(seeded_db), "--hours", "48"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("Cost report for the last 48 h")
    assert "steps_by_context_bucket" in text and "<50k=7" in text
    assert cost_report.main(["--db", str(seeded_db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tasks"] == 8


def test_main_since_until_and_baseline(
    seeded_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 3600))
    assert cost_report.main(["--db", str(seeded_db), "--since", since, "--json"]) == 0
    baseline = json.loads(capsys.readouterr().out)
    assert baseline["tasks"] == 8
    baseline["refused_tool_calls"] = 4
    baseline["cost_usd"] = 0
    baseline["cache_hit_ratio"] = None
    baseline["only_in_baseline"] = 1
    baseline["steps_by_context_bucket"]["<50k"] = 14
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    # An epoch --since with an ISO --until that ends before the seeded rows.
    until = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 120))
    assert cost_report.main(
        ["--db", str(seeded_db), "--since", "0", "--until", until, "--baseline", str(path)]
    ) == 0
    text = capsys.readouterr().out
    assert text.startswith("Cost report for 1970-01-01T00:00Z to ")
    assert " against the baseline" in text.splitlines()[0]
    assert cost_report.main(
        ["--db", str(seeded_db), "--since", since, "--baseline", str(path), "--json"]
    ) == 0
    table = json.loads(capsys.readouterr().out)
    assert table["refused_tool_calls"] == {"baseline": 4, "current": 7, "delta": 3, "pct": 75.0}
    assert table["steps_by_context_bucket.<50k"]["pct"] == -50.0
    # A zero baseline has a delta but no percentage; a None or a missing
    # side has neither; a key only in the baseline is still listed.
    assert table["cost_usd"]["delta"] == 20.0 and table["cost_usd"]["pct"] is None
    assert table["cache_hit_ratio"]["delta"] is None
    assert table["only_in_baseline"] == {"baseline": 1, "current": None, "delta": None, "pct": None}
    assert cost_report.main(
        ["--db", str(seeded_db), "--since", since, "--baseline", str(path)]
    ) == 0
    text = capsys.readouterr().out
    assert "refused_tool_calls" in text and "+75.0%" in text
    # --until alone bounds the --hours window: nothing started in the 48 h
    # before two minutes ago.
    assert cost_report.main(["--db", str(seeded_db), "--hours", "48", "--until", until]) == 0
    text = capsys.readouterr().out
    assert text.startswith("Cost report for the 48 h before ") and "tasks" in text
    assert text.splitlines()[1].split()[-1] == "0"
    with pytest.raises(ValueError):
        cost_report.main(["--db", str(seeded_db), "--since", "not-a-date"])


def test_a_path_that_reads_as_uri_syntax_names_the_database(
    seeded_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """URI punctuation in a legal host path never changes which database is opened."""
    name = "kiss#home %41" if os.name == "nt" else "kiss#home ?mode=rw %41"
    home = seeded_db.parent / name
    home.mkdir()
    moved = seeded_db.rename(home / "history.db")
    assert cost_report.main(["--db", str(moved), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tasks"] == 8


def test_main_missing_db(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cost_report.main(["--db", str(tmp_path / "nope.db")])
