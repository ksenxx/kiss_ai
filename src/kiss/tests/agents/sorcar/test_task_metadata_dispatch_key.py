# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``infer_subagent_seas`` stays linear in the length of a dispatched task text.

The path-stripping key it compares the parent's ``run_agent`` task with
the child's task text was a backtracking regex, quadratic on a long
whitespace-free token (a pasted blob): 40 000 characters took 8 s, and
the backfill runs it once per call and once per child.
"""

from __future__ import annotations

import json
import sqlite3
import time

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.task_metadata import infer_subagent_seas


def _db_with_dispatch(task_text: str, child_text: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    th._init_tables(conn)
    conn.execute(
        "INSERT INTO task_history (id, timestamp, task) VALUES ('p', 1.0, 'parent')"
    )
    conn.execute(
        "INSERT INTO task_history (id, timestamp, task, parent_task_id) "
        "VALUES ('c', 2.0, ?, 'p')",
        (child_text,),
    )
    event = {
        "type": "tool_call",
        "name": "run_agent",
        "extras": {"agent": "/x/write_paper/write_paper_sea.py", "task": task_text},
    }
    conn.execute(
        "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES ('p', 1, ?, 1.0)",
        (json.dumps(event),),
    )
    conn.commit()
    return conn


def test_infer_subagent_seas_is_linear_on_a_long_slash_free_token() -> None:
    blob = "A" * 100_000
    conn = _db_with_dispatch(f"summarize {blob} now", f"[channel preamble] summarize {blob} now")
    started = time.monotonic()
    assert infer_subagent_seas(conn, "p") == {"c": "write_paper_sea"}
    assert time.monotonic() - started < 5.0


def test_infer_subagent_seas_ignores_path_tokens_and_extra_whitespace() -> None:
    # The parent dispatched with the repo's absolute path; the child's task
    # text names the worktree's and spaces it differently.
    conn = _db_with_dispatch(
        "review /home/u/repo/a.tex  and C:\\docs\\b.tex\tcarefully",
        "review /wt/repo/a.tex and D:\\wt\\b.tex carefully",
    )
    assert infer_subagent_seas(conn, "p") == {"c": "write_paper_sea"}
    # A child whose text differs in a non-path word does not match.
    conn = _db_with_dispatch("review /home/u/repo/a.tex carefully", "review /wt/a.tex quickly")
    assert infer_subagent_seas(conn, "p") == {}
