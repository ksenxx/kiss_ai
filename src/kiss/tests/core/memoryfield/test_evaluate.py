"""End-to-end tests for the recall evaluation against a synthetic Sorcar task database.

Deliberately uncovered: the ``if not query`` branch in ``llm_probes`` (a
probe model returning an empty completion). It cannot be reached without a
fake model and is a one-line skip-with-warning.
"""

import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from kiss.core.memoryfield.evaluate import (
    HAND_PROBES,
    HASHED_EMBEDDING_MODEL_CODE,
    KeywordIndex,
    PastTask,
    Probe,
    build_memory_from_tasks,
    duplicate_family_count,
    family_rank_of,
    format_results_table,
    hand_probes,
    html_to_text,
    llm_probes,
    load_past_tasks,
    main,
    rank_of,
    reciprocal_rank_fusion,
    run_evaluation,
    score_ranks,
    task_families,
    task_page_body,
)
from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
from kiss.core.memoryfield.pages import MAX_PAGE_BYTES, MemoryDir

live_api = pytest.mark.live_api
requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; probe generation needs it",
)
requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; real-embedding CLI test needs it",
)

TASKS = [
    # (id, task, result)
    (
        "bf690e27" + "0" * 24,
        "what is the simplest memory system for an AI agent?",
        "<h3>Files plus a SQLite vector index</h3><p>Use Markdown pages.</p>",
    ),
    (
        "f69f2770" + "0" * 24,
        "how do I complete drive cycle on BMW 2017 330i?",
        "<p>Drive at <strong>steady</strong> highway speed for 15 minutes, then idle.</p>"
        "<ul><li>cold start</li></ul>",
    ),
    (
        "e16312e8" + "0" * 24,
        "Compare SENSARTE with Carote, GreenPan, and Caraway on safety",
        "<table><tr><td>GreenPan</td><td>ceramic</td></tr></table>",
    ),
    ("aaaaaaaa" + "0" * 24, "run all tests", "<p>All 4200 tests passed.</p>"),
    ("bbbbbbbb" + "0" * 24, "run all tests", "<p>Two failures in the vscode suite.</p>"),
]


def make_db(path: Path, extra_rows: list[tuple[Any, ...]] | None = None) -> Path:
    """Create a minimal ``sorcar.db`` with the real ``task_history`` columns."""
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TABLE task_history (id TEXT PRIMARY KEY, timestamp REAL NOT NULL,"
            " task TEXT NOT NULL,"
            " result TEXT DEFAULT '', parent_task_id TEXT DEFAULT '')"
        )
        now = time.time()
        for i, (task_id, task, result) in enumerate(TASKS):
            conn.execute(
                "INSERT INTO task_history VALUES (?, ?, ?, ?, '')", (task_id, now - i, task, result)
            )
        for row in extra_rows or []:
            conn.execute("INSERT INTO task_history VALUES (?, ?, ?, ?, ?)", row)
    return path


def test_html_to_text() -> None:
    text = html_to_text(
        "<h3>Title</h3><p>One <em>two</em>   \n</p><ul><li>a</li><li>b</li></ul>plain"
    )
    assert text == "Title\n\nOne two\n\na\n\nb\n\nplain"
    assert html_to_text("no tags") == "no tags"
    assert html_to_text("") == ""


def test_load_past_tasks_filters_failed_and_child_rows(tmp_path: Path) -> None:
    long_result = "<p>" + "x" * 400 + "</p>"
    db = make_db(
        tmp_path / "sorcar.db",
        extra_rows=[
            ("child" + "0" * 27, time.time() + 10, "child task", long_result, "parent-id"),
            ("fail1" + "0" * 27, time.time() + 11, "t", "Task failed: boom" + long_result, ""),
            ("fail2" + "0" * 27, time.time() + 12, "t", "Task stopped by user", ""),
            ("short" + "0" * 27, time.time() + 13, "t", "<p>tiny</p>", ""),
            ("interrupted" + "0" * 21, time.time() + 14, "t", "Task interrupted" + long_result, ""),
        ],
    )
    tasks = load_past_tasks(db, limit=10, min_result_chars=20)
    assert [t.task_id[:8] for t in tasks] == [
        "bf690e27",
        "f69f2770",
        "e16312e8",
        "aaaaaaaa",
        "bbbbbbbb",
    ]
    assert load_past_tasks(db, limit=2, min_result_chars=20)[1].task_id.startswith("f69f2770")
    assert load_past_tasks(db, limit=10, min_result_chars=500) == []
    # NULL parent_task_id is also a top-level task (Sorcar's canonical predicate).
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            "INSERT INTO task_history VALUES (?, ?, ?, ?, NULL)",
            ("nullpar" + "0" * 25, time.time() + 20, "null parent task", long_result),
        )
    assert load_past_tasks(db, limit=1, min_result_chars=20)[0].task == "null parent task"

    first = tasks[0]
    assert first.page_name == "what-is-the-simplest-memory-system-for-an-ai-bf690e27"
    assert first.title == "what is the simplest memory system for an AI agent?"
    blank = PastTask(task_id="deadbeef" + "0" * 24, timestamp=0.0, task="   \n", result="r")
    assert blank.page_name == "task-deadbeef" and blank.title == blank.task_id


def test_load_past_tasks_handles_special_characters_in_path(tmp_path: Path) -> None:
    db = make_db(tmp_path / "history?copy #1.db")
    assert len(load_past_tasks(db, limit=10, min_result_chars=10)) == 5


def test_task_page_body_and_truncation() -> None:
    task = PastTask(
        task_id="abc", timestamp=1_757_700_000.0, task="do X", result="<p>done <b>well</b></p>"
    )
    body = task_page_body(task)
    assert body.startswith("# Task (2025-09-12)\n\ndo X\n\n# Result\n\ndone well\n")
    huge = PastTask(task_id="abc", timestamp=0.0, task="t", result="é" * 20_000)
    truncated = task_page_body(huge)
    assert truncated.endswith("[truncated]\n")
    assert len(truncated.encode("utf-8")) <= MAX_PAGE_BYTES


def test_build_memory_keyword_index_and_fusion(tmp_path: Path) -> None:
    db = make_db(tmp_path / "sorcar.db")
    tasks = load_past_tasks(db, limit=10, min_result_chars=10)
    memory = MemoryDir(tmp_path / "memory")
    names = build_memory_from_tasks(memory, tasks)
    assert sorted(names) == memory.page_names()
    page = memory.read(names[0])
    assert page.frontmatter["source"].startswith("sorcar.db task_history bf690e27")
    assert page.title == tasks[0].title
    # Reconciliation: existing pages are kept byte-for-byte, extras are pruned, missing ones added.
    memory.write("stray", "not part of the corpus")
    assert build_memory_from_tasks(memory, tasks[:2]) == names[:2]
    assert memory.page_names() == sorted(names[:2])
    assert memory.read(names[0]).raw == page.raw
    assert build_memory_from_tasks(memory, tasks) == names

    with closing(KeywordIndex(memory)) as keyword:
        assert keyword.search("drive cycle for the BMW?", k=3)[0].startswith(
            "how-do-i-complete-drive-cycle"
        )
        assert keyword.search("!!! ???", k=3) == []
        assert keyword.search("", k=3) == []
        assert len(keyword.search("tests", k=5)) == 2

    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]], k=2)
    assert fused == ["a", "b"]
    assert reciprocal_rank_fusion([["a"], ["b"]], k=5) == ["a", "b"]
    assert reciprocal_rank_fusion([], k=3) == []


def test_probe_helpers_and_metrics() -> None:
    tasks = [PastTask(task_id=t[0], timestamp=0.0, task=t[1], result=t[2]) for t in TASKS]
    probes = hand_probes(tasks)
    known = {t.task_id[:8] for t in tasks}
    assert len(probes) == sum(1 for _, prefix in HAND_PROBES if prefix in known) == 3
    assert probes[0].source == "hand" and probes[0].gold.endswith("bf690e27")

    assert rank_of("b", ["a", "b"]) == 2 and rank_of("z", ["a"]) is None
    families = task_families(tasks)
    assert families[tasks[3].page_name] == families[tasks[4].page_name] == "run all tests"
    assert (
        PastTask(task_id="x", timestamp=0.0, task="  Run\nALL   tests ", result="").family
        == "run all tests"
    )
    assert family_rank_of(tasks[3].page_name, ["other", tasks[4].page_name], families) == 2
    assert family_rank_of(tasks[3].page_name, ["other"], families) is None
    assert (
        family_rank_of("unknown", ["x", "unknown"], families) == 2
    )  # unknown pages are their own family
    assert duplicate_family_count(families) == 2
    assert duplicate_family_count({}) == 0

    result = score_ranks("r", [1, None, 3, 2], [10.0, 20.0, 30.0, 40.0])
    assert result.recall == {1: 0.25, 3: 0.75, 5: 0.75}
    assert result.mrr == pytest.approx((1 + 1 / 3 + 1 / 2) / 4)
    assert result.family_recall == result.recall and result.family_mrr == result.mrr
    assert result.mean_latency_ms == 25.0
    empty = score_ranks("e", [], [])
    assert (
        empty.recall == {1: 0.0, 3: 0.0, 5: 0.0}
        and empty.mrr == 0.0
        and empty.mean_latency_ms == 0.0
    )
    table = format_results_table({"r": result}, 4)
    assert table.splitlines()[2].startswith(
        "| r | 0.25 | 0.75 | 0.75 | 0.46 | 0.25 | 0.75 | 0.75 | 0.46 |"
    )
    assert table.splitlines()[2].endswith("| 25 | 4 |")


def test_run_evaluation_offline(tmp_path: Path) -> None:
    db = make_db(tmp_path / "sorcar.db")
    tasks = load_past_tasks(db, limit=10, min_result_chars=10)
    memory = MemoryDir(tmp_path / "memory")
    build_memory_from_tasks(memory, tasks)
    vector = VectorIndex(memory, embed=hashed_embedding, model_code="v")
    hashed = VectorIndex(memory, embed=hashed_embedding, model_code=HASHED_EMBEDDING_MODEL_CODE)
    vector.sync()
    hashed.sync()
    probes = hand_probes(tasks) + [
        Probe(query="did the whole test suite pass", gold=tasks[3].page_name, source="llm")
    ]
    with closing(KeywordIndex(memory)) as keyword:
        results, rows = run_evaluation(
            memory, probes, vector, hashed, keyword, families=task_families(tasks)
        )
        strict, _ = run_evaluation(memory, probes, vector, hashed, keyword)
    assert set(results) == {"vector", "hashed", "bm25", "hybrid"}
    assert len(rows) == 4 and rows[0]["gold_title"] == tasks[0].title
    # The drive-cycle probe shares "drive cycle" with exactly one page.
    drive = rows[1]
    assert drive["rank_vector"] == 1 and drive["rank_bm25"] == 1 and drive["rank_hybrid"] == 1
    # The repeated "run all tests" pages are a family: strict rank may miss, family rank must hit.
    tests_row = rows[3]
    assert tests_row["family_rank_bm25"] == 1
    assert results["bm25"].family_recall[1] >= results["bm25"].recall[1]
    assert (tmp_path / "memory" / "v.sqlite3").exists()
    # Without a family map every page is its own family.
    assert strict["bm25"].family_recall == strict["bm25"].recall


def test_main_cli_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = make_db(tmp_path / "sorcar.db")
    out = tmp_path / "eval" / "results.json"
    args = [
        "--db",
        str(db),
        "--memory-dir",
        str(tmp_path / "memory"),
        "--out",
        str(out),
        "--limit",
        "10",
        "--llm-probes",
        "0",
        "--embedding-model",
        HASHED_EMBEDDING_MODEL_CODE,
        "--min-result-chars",
        "10",
    ]
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "Corpus: 5 pages" in printed and "2 pages belong to repeated task families" in printed
    assert "Probes from source 'hand'" in printed
    data = json.loads(out.read_text())
    assert data["corpus_pages"] == 5 and data["pages_in_duplicate_families"] == 2
    assert set(data["results"]) == {"vector", "hashed", "bm25", "hybrid"}
    assert len(data["probes"]) == 3
    assert (tmp_path / "memory" / f"{HASHED_EMBEDDING_MODEL_CODE}.sqlite3").exists()

    # Second run reuses the pages; --rebuild rewrites them and clears the indexes.
    assert main(args) == 0
    assert "Corpus: 5 pages" in capsys.readouterr().out
    assert main([*args, "--rebuild"]) == 0
    assert json.loads(out.read_text())["corpus_pages"] == 5
    # A smaller --limit prunes the corpus exactly instead of leaving stale pages behind.
    smaller = [a if a != "10" else "2" for a in args]
    assert main(smaller) == 0
    assert json.loads(out.read_text())["corpus_pages"] == 2
    assert len(list((tmp_path / "memory").glob("*.md"))) == 2
    assert main(args) == 0
    assert json.loads(out.read_text())["corpus_pages"] == 5


@live_api
@requires_openai
def test_main_cli_with_real_embedding_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(tmp_path / "sorcar.db")
    out = tmp_path / "eval" / "results.json"
    assert (
        main(
            [
                "--db",
                str(db),
                "--memory-dir",
                str(tmp_path / "memory"),
                "--out",
                str(out),
                "--limit",
                "10",
                "--llm-probes",
                "0",
                "--min-result-chars",
                "10",
            ]
        )
        == 0
    )
    assert (tmp_path / "memory" / "text-embedding-3-small.sqlite3").exists()
    data = json.loads(out.read_text())
    assert data["embedding_model"] == "text-embedding-3-small"
    # The drive-cycle probe is unambiguous: real embeddings must rank it first.
    drive = next(p for p in data["probes"] if p["gold"].startswith("how-do-i-complete-drive-cycle"))
    assert drive["rank_vector"] == 1
    assert "Corpus: 5 pages" in capsys.readouterr().out


@live_api
@requires_anthropic
def test_llm_probes_and_probe_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = make_db(tmp_path / "sorcar.db")
    tasks = load_past_tasks(db, limit=10, min_result_chars=10)
    memory = MemoryDir(tmp_path / "memory")
    build_memory_from_tasks(memory, tasks)
    probes = llm_probes(memory, [tasks[1].page_name], "claude-haiku-4-5")
    assert len(probes) == 1 and probes[0].source == "llm" and probes[0].gold == tasks[1].page_name
    assert 3 <= len(probes[0].query.split()) <= 40

    out = tmp_path / "eval" / "results.json"
    args = [
        "--db",
        str(db),
        "--memory-dir",
        str(tmp_path / "memory"),
        "--out",
        str(out),
        "--limit",
        "10",
        "--llm-probes",
        "1",
        "--embedding-model",
        HASHED_EMBEDDING_MODEL_CODE,
        "--probe-model",
        "claude-haiku-4-5",
        "--min-result-chars",
        "10",
    ]
    assert main(args) == 0
    cache = out.with_name("llm_probes.json")
    cached = json.loads(cache.read_text())
    assert len(cached["probes"]) == 1 and cached["meta"]["count"] == 1
    assert "Probes from source 'llm'" in capsys.readouterr().out
    # A second run with the same request reads the cached probes instead of calling the model.
    cached["probes"][0]["query"] = "cached question?"
    cache.write_text(json.dumps(cached))
    assert main(args) == 0
    assert json.loads(out.read_text())["probes"][-1]["query"] == "cached question?"
    # A different probe count invalidates the cache and regenerates.
    two_probes = [
        a if a != "1" else "2" for a in args
    ]  # only "--llm-probes 1" carries the value "1"
    assert main(two_probes) == 0
    regenerated = json.loads(cache.read_text())
    assert regenerated["meta"]["count"] == 2 and len(regenerated["probes"]) == 2
    assert all(p["query"] != "cached question?" for p in regenerated["probes"])
