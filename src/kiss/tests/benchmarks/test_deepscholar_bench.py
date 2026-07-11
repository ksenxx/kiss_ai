# Author: Koushik Sen (ksen@berkeley.edu)

"""Focused tests for the DeepScholar-Bench Sorcar harness."""

from __future__ import annotations

import subprocess
from pathlib import Path

import kiss.benchmarks.deepscholar_bench.run as ds_run
from kiss.benchmarks.deepscholar_bench.prompts import build_task


def test_prompt_requires_numeric_arxiv_markdown_links() -> None:
    """The prompt emits the citation shape required by citation evaluators."""
    prompt = build_task("/tmp/result.md", "An abstract.", "2025-01-02")

    assert "/tmp/result.md" in prompt
    assert "An abstract." in prompt
    assert "2025-01-02" in prompt
    assert "[1](https://arxiv.org/abs/1706.03762)" in prompt
    assert "Do NOT put author names or titles inside the square brackets" in prompt


def test_load_queries_preserves_dataset_row_ids_and_slices(tmp_path: Path) -> None:
    """Output folder IDs continue to address the corresponding dataset rows."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "papers_with_related_works.csv").write_text(
        "arxiv_id,abstract,published_date\n"
        '2501.00001v1,"First abstract",2025-01-01T12:00:00+00:00\n'
        '2501.00002v1,"Second abstract",2025-01-02T12:00:00+00:00\n'
        '2501.00003v1,"Third abstract",2025-01-03T12:00:00+00:00\n'
    )

    queries = ds_run.load_queries(tmp_path, n_tasks=1, start_idx=1)

    assert len(queries) == 1
    assert queries[0]["idx"] == 1
    assert queries[0]["arxiv_id"] == "2501.00002v1"
    assert queries[0]["abstract"] == "Second abstract"
    assert "2025-01-02T12:00:00+00:00" in queries[0]["query"]


def test_run_sorcar_one_reuses_nonempty_output(tmp_path: Path) -> None:
    """Interrupted runs resume without spending money on completed tasks."""
    output = tmp_path / "7" / "output.md"
    output.parent.mkdir()
    output.write_text("existing result")

    result = ds_run.run_sorcar_one(
        {
            "idx": 7,
            "arxiv_id": "2501.00007v1",
            "abstract": "Abstract",
            "cutoff_date": "2025-01-07",
        },
        tmp_path,
        "claude-opus-4-7",
        timeout=10,
        max_budget=1.25,
        overwrite_existing=False,
    )

    assert result["status"] == "reused_existing"
    assert output.read_text() == "existing result"


def test_run_eval_passes_only_valid_numeric_task_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Evaluator never tries to parse summary files or incomplete tasks."""
    results = tmp_path / "results"
    for task_id in ("10", "2"):
        task = results / task_id
        task.mkdir(parents=True)
        (task / "output.md").write_text("[1](https://arxiv.org/abs/1706.03762)")
    empty_task = results / "3"
    empty_task.mkdir()
    (empty_task / "output.md").write_text("")
    (results / "summary.json").write_text("[]")

    seen: list[str] = []

    def fake_run(
        cmd: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        seen.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ds_run, "_run", fake_run)
    repo = tmp_path / "repo"
    output_csv = ds_run.run_eval(
        repo=repo,
        python=Path("/fake/python"),
        results_dir=results,
        eval_dir=tmp_path / "evaluation",
        judge_model="gpt-4o",
        evals=["organization"],
    )

    file_id_pos = seen.index("--file-id")
    output_pos = seen.index("--output-folder")
    assert seen[file_id_pos + 1 : output_pos] == ["2", "10"]
    assert "summary.json" not in seen
    assert "3" not in seen[file_id_pos + 1 : output_pos]
    assert output_csv == tmp_path / "evaluation" / "results.csv"
