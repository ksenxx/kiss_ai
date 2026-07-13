# Author: Koushik Sen (ksen@berkeley.edu)

"""Run the DeepScholar-Bench benchmark with KISS Sorcar as the SUT.

DeepScholar-Bench (https://github.com/guestrin-lab/deepscholar) is a
live benchmark for generative research synthesis.  This module drives
end-to-end evaluation of Sorcar on that benchmark:

1. Clones (or reuses) the ``guestrin-lab/deepscholar`` repository into a
   local scratch workdir.
2. Materialises a dedicated Python 3.10 virtualenv (via ``uv``) and
   installs the DeepScholar-Bench ``requirements.txt`` into it — this
   isolates the evaluator's dependencies (LOTUS, etc.) from Sorcar's
   own environment.
3. Loads (or generates from ``papers_with_related_works.csv``) the
   query set and, for each query, invokes ``sorcar -t <prompt>`` in a
   fresh subdirectory to write a single ``.md`` file with markdown-style
   ``[number](https://arxiv.org/abs/<id>)`` citations.
4. Runs ``python -m eval.main --modes openai_deepresearch`` inside the
   cloned repo so all Sorcar outputs are scored with all seven shipped
   metrics against the benchmark's ground-truth files.
5. Parses ``outputs/evaluation/results.csv`` and prints a summary table.

Usage
-----

.. code-block:: bash

    uv run python -m kiss.benchmarks.deepscholar_bench.run \\
        --model claude-opus-4-7 \\
        --n-concurrent 4 \\
        --n-tasks 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kiss.benchmarks.deepscholar_bench.prompts import build_task

DEEPSCHOLAR_REPO_URL = "https://github.com/guestrin-lab/deepscholar.git"
# Pin to the exact commit the code in this module was written against
# so upstream changes cannot silently break the harness.  Update as
# DeepScholar-Bench evolves and re-verify metrics.
DEEPSCHOLAR_PIN = "c95413b3b2f3255b461b90d0ce650f685ae2d1ff"

# Query template borrowed verbatim from
# ``deepscholar/deepscholar_base/main.py::load_queries`` so we produce
# the same queries.csv the upstream reference pipeline would.
QUERY_TEMPLATE = (
    "Your task is to write a Related Works section for an academic paper "
    "given the paper's abstract. Your response should provide the Related "
    "Works section and references. Only include references from arXiv that "
    "are published before {cutoff_date}. Mention them in a separate, numbered "
    "reference list at the end and use the reference numbers to provide "
    "in-line citations in the Related Works section for all claims referring "
    "to a source (e.g., description of source [3]. Further details "
    "[6][7][8][9][10].) Each in-line citation must consist of a single "
    "reference number within a pair of brackets. Do not use any other "
    "citation format. Do not exceed 600 words for the related works "
    "section. Here is the paper abstract: {abstract}"
)

# We ask the DeepScholar-Bench evaluator to use the
# ``openai_deepresearch`` parser because it needs only a single
# markdown file per task with inline ``[number](https://arxiv.org/abs/ID)``
# citations — the exact shape we prompt Sorcar for.  See
# ``deepscholar/eval/parsers/openai_deepresearch.py``.
EVAL_PARSER_MODE = "openai_deepresearch"

# Evaluations that do NOT require the pkl ``metadata.pickle`` file used
# only by the deepscholar_base pipeline.
DEFAULT_EVALS = (
    "organization",
    "nugget_coverage",
    "reference_coverage",
    "cite_p",
    "claim_coverage",
    "coverage_relevance_rate",
    "document_importance",
)


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` and stream output; optionally capture stdout/stderr.

    Args:
        cmd: Command tokens.
        cwd: Working directory.
        env: Environment overlay (merged onto ``os.environ``).
        check: If True, raise on non-zero exit.
        capture: If True, capture output (returned in ``.stdout``/``.stderr``);
            otherwise stream to this process's stdout/stderr.

    Returns:
        Completed process handle.
    """
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=merged,
        check=check,
        text=True,
        capture_output=capture,
    )


def ensure_repo(workdir: Path) -> Path:
    """Clone (or update) the DeepScholar-Bench repo under ``workdir``.

    Args:
        workdir: Root directory that will hold the checkout at
            ``workdir/deepscholar``.

    Returns:
        Path to the cloned repository.
    """
    repo = workdir / "deepscholar"
    if not repo.exists():
        workdir.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", DEEPSCHOLAR_REPO_URL, str(repo)])
    _run(["git", "fetch", "--all", "--tags"], cwd=repo)
    _run(["git", "checkout", DEEPSCHOLAR_PIN], cwd=repo)
    ensure_openai_parser_wired(repo)
    return repo


def ensure_openai_parser_wired(repo: Path) -> None:
    """Apply the pinned checkout's missing OpenAI parser-factory wiring.

    Commit :data:`DEEPSCHOLAR_PIN` ships ``openai_deepresearch.py`` and lists
    its enum value, but omits the parser import/factory branch. It also discards
    the parser's normalized numeric-citation text and shifts already-linked
    numeric citations by one. Those defects respectively crash evaluation and
    break the claim/citation metrics. Keep the minimal, auditable compatibility
    patch next to this harness and apply it idempotently.

    Args:
        repo: Pinned DeepScholar-Bench checkout.
    """
    patch = Path(__file__).with_name("openai_parser.patch")
    reverse_check = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if reverse_check.returncode == 0:
        return  # Already applied by a previous/resumed invocation.

    apply_check = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if apply_check.returncode != 0:
        raise RuntimeError(
            "DeepScholar OpenAI-parser compatibility patch does not apply: "
            f"{apply_check.stderr.strip()}"
        )
    _run(["git", "apply", str(patch)], cwd=repo)


def ensure_eval_venv(repo: Path) -> Path:
    """Create a Python 3.10 venv (via uv) and install DS-Bench deps.

    The DeepScholar-Bench evaluator depends on LOTUS and a heavy
    scientific stack that we do not want in Sorcar's own env.  We
    materialise a dedicated ``.venv`` inside the cloned repo.

    Args:
        repo: Path to the cloned DeepScholar-Bench repository.

    Returns:
        Path to the ``.venv/bin/python`` executable to use for
        ``python -m eval.main`` invocations.
    """
    venv = repo / ".venv"
    python = venv / "bin" / "python"
    if not python.exists():
        _run(["uv", "venv", "--python", "3.10", str(venv)], cwd=repo)
    # Always re-run pip install (idempotent) to catch any missing
    # transitive deps after a partial previous install.
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(repo / "requirements.txt"),
        ],
        # requirements.txt contains ``-e ./eval/nuggetizer``; resolve that
        # local path against the pinned benchmark checkout, not the caller's
        # working directory.
        cwd=repo,
    )
    # NLTK does not ship tokenizer data with the Python package.  Both
    # citation-level evaluators call ``sent_tokenize`` and NLTK 3.9.1 needs
    # ``punkt`` plus ``punkt_tab``; install them before an expensive eval run
    # rather than failing after generation has completed.
    _run(
        [
            str(python),
            "-c",
            (
                "import nltk; "
                "assert nltk.download('punkt', quiet=True); "
                "assert nltk.download('punkt_tab', quiet=True)"
            ),
        ]
    )
    return python


def load_queries(
    repo: Path,
    n_tasks: int | None,
    start_idx: int,
) -> list[dict[str, Any]]:
    """Load (or generate) queries from the DeepScholar-Bench dataset.

    The upstream ``queries.csv`` is generated on the fly from
    ``papers_with_related_works.csv`` using the shipped
    :data:`QUERY_TEMPLATE`.  We reproduce that generation here so the
    harness works on a fresh checkout without invoking the
    ``data_pipeline`` scraper.

    Args:
        repo: Path to the cloned DeepScholar-Bench repository.
        n_tasks: If not None, limit to this many queries.
        start_idx: Skip the first ``start_idx`` queries.

    Returns:
        List of query dicts with keys ``idx``, ``arxiv_id``,
        ``abstract``, ``cutoff_date``, ``query``.
    """
    import pandas as pd  # type: ignore[import-untyped]

    papers_csv = repo / "dataset" / "papers_with_related_works.csv"
    df = pd.read_csv(papers_csv)

    queries: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        cutoff = str(row.get("published_date", "")).split(" ")[0]
        abstract = str(row.get("abstract", "")).strip()
        arxiv_id = str(row.get("arxiv_id", f"unknown_{idx}"))
        queries.append(
            {
                "idx": int(idx),  # type: ignore[arg-type]
                "arxiv_id": arxiv_id,
                "abstract": abstract,
                "cutoff_date": cutoff,
                "query": QUERY_TEMPLATE.format(cutoff_date=cutoff, abstract=abstract),
            }
        )

    end = len(queries) if n_tasks is None else min(len(queries), start_idx + n_tasks)
    return queries[start_idx:end]


def run_sorcar_one(
    query: dict[str, Any],
    results_dir: Path,
    model: str,
    timeout: int,
    max_budget: float,
    overwrite_existing: bool,
) -> dict[str, Any]:
    """Run a single Sorcar invocation for one DeepScholar-Bench query.

    Sorcar is told (via the prompt in
    :mod:`kiss.benchmarks.deepscholar_bench.prompts`) to write its
    output to ``<results_dir>/<idx>/output.md``.  The idx-scoped
    subdirectory is also used as the Sorcar ``-w`` work directory so
    each task runs in isolation.

    Args:
        query: Query dict from :func:`load_queries`.
        results_dir: Root output directory (one subdir per query).
        model: Model name to pass to ``sorcar -m``.
        timeout: Per-task timeout in seconds.
        max_budget: Maximum Sorcar spend in USD for this query.
        overwrite_existing: Regenerate even when a non-empty ``output.md``
            already exists.

    Returns:
        Result dict: ``idx``, ``arxiv_id``, ``status``, ``elapsed``,
        ``output_md`` (path), and ``error`` when applicable.
    """
    idx = query["idx"]
    task_dir = results_dir / str(idx)
    task_dir.mkdir(parents=True, exist_ok=True)
    output_md = task_dir / "output.md"
    if not overwrite_existing and output_md.exists() and output_md.stat().st_size > 0:
        return {
            "idx": idx,
            "arxiv_id": query["arxiv_id"],
            "status": "reused_existing",
            "elapsed": 0.0,
            "output_md": str(output_md),
        }
    # Regeneration must never accidentally score yesterday's artefact if the
    # subprocess fails before writing a replacement.
    if output_md.exists():
        output_md.unlink()

    task_prompt = build_task(
        output_md_path=str(output_md),
        abstract=query["abstract"],
        cutoff_date=query["cutoff_date"],
    )

    # Note: DO NOT pass ``--no-web``.  DeepScholar-Bench requires Sorcar
    # to search arXiv for real paper IDs, and ``--no-web`` disables the
    # web tools entirely (see ``cli_helpers.py::web_tools``).
    cmd = [
        "sorcar",
        "-t",
        task_prompt,
        "-w",
        str(task_dir),
        "-m",
        model,
        "-b",
        str(max_budget),
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        stdout_log = task_dir / "sorcar_stdout.log"
        stderr_log = task_dir / "sorcar_stderr.log"
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")
        # Prefer the produced file regardless of sorcar's exit code —
        # a non-zero return code often reflects a cosmetic issue
        # (e.g., budget warning) rather than a missing artefact.
        if output_md.exists() and output_md.stat().st_size > 0:
            return {
                "idx": idx,
                "arxiv_id": query["arxiv_id"],
                "status": "success" if proc.returncode == 0 else "success_nonzero_rc",
                "return_code": proc.returncode,
                "elapsed": elapsed,
                "output_md": str(output_md),
            }
        # Fall back: dump stdout so the parser still has something to
        # score rather than skipping the task entirely.
        output_md.write_text(proc.stdout or "", encoding="utf-8")
        return {
            "idx": idx,
            "arxiv_id": query["arxiv_id"],
            "status": "no_output_file_wrote_stdout",
            "return_code": proc.returncode,
            "elapsed": elapsed,
            "output_md": str(output_md),
        }
    except subprocess.TimeoutExpired:
        return {
            "idx": idx,
            "arxiv_id": query["arxiv_id"],
            "status": "timeout",
            "elapsed": time.time() - start,
            "output_md": None,
        }
    except Exception as exc:  # pragma: no cover — defensive
        return {
            "idx": idx,
            "arxiv_id": query["arxiv_id"],
            "status": "error",
            "error": repr(exc),
            "elapsed": time.time() - start,
            "output_md": None,
        }


def generate_all(
    queries: list[dict[str, Any]],
    results_dir: Path,
    model: str,
    n_concurrent: int,
    per_task_timeout: int,
    max_budget: float,
    overwrite_existing: bool,
) -> list[dict[str, Any]]:
    """Run Sorcar for every query (in a thread pool) and collect results.

    Args:
        queries: Query list from :func:`load_queries`.
        results_dir: Root output directory.
        model: Sorcar model flag.
        n_concurrent: Number of parallel Sorcar invocations.
        per_task_timeout: Per-task timeout in seconds.
        max_budget: Maximum Sorcar spend in USD per query.
        overwrite_existing: Regenerate non-empty existing outputs.

    Returns:
        Per-query result dicts in the same order as ``queries``.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * len(queries)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        fut_to_i = {
            ex.submit(
                run_sorcar_one,
                q,
                results_dir,
                model,
                per_task_timeout,
                max_budget,
                overwrite_existing,
            ): i
            for i, q in enumerate(queries)
        }
        for fut in concurrent.futures.as_completed(fut_to_i):
            i = fut_to_i[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # pragma: no cover
                results[i] = {
                    "idx": queries[i]["idx"],
                    "status": "worker_exception",
                    "error": repr(exc),
                }
            r = results[i]
            assert r is not None
            print(
                f"[{i + 1}/{len(queries)}] idx={r.get('idx')} "
                f"status={r.get('status')} elapsed={r.get('elapsed', 0):.1f}s",
                flush=True,
            )
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return [r for r in results if r is not None]


def run_eval(
    repo: Path,
    python: Path,
    results_dir: Path,
    eval_dir: Path,
    judge_model: str,
    evals: list[str],
) -> Path:
    """Invoke DeepScholar-Bench's ``eval.main`` on the generated outputs.

    Args:
        repo: Cloned repo path.
        python: Path to the evaluator venv Python.
        results_dir: Directory containing per-query subfolders (each
            with an ``output.md`` file).
        eval_dir: Where evaluator writes CSVs.
        judge_model: LLM used as the LLM-judge (default ``gpt-4o``).
        evals: Metrics to run (subset of ``DEFAULT_EVALS`` or ``["all"]``).

    Returns:
        Path to the aggregated ``results.csv`` produced by the evaluator.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Restrict evaluation to numeric task directories containing an artifact.
    # ``results_dir`` also contains summary.json, which the upstream evaluator
    # otherwise attempts (and fails) to interpret as a paper ID.
    file_ids = sorted(
        (
            child.name
            for child in results_dir.iterdir()
            if child.is_dir()
            and child.name.isdigit()
            and (child / "output.md").is_file()
            and (child / "output.md").stat().st_size > 0
        ),
        key=int,
    )
    if not file_ids:
        raise RuntimeError(f"No non-empty generated outputs found under {results_dir}")

    cmd = [
        str(python),
        "-m",
        "eval.main",
        "--modes",
        EVAL_PARSER_MODE,
        "--evals",
        *evals,
        "--input-folder",
        str(results_dir),
        "--file-id",
        *file_ids,
        "--output-folder",
        str(eval_dir),
        "--dataset-path",
        str(repo / "dataset" / "papers_with_related_works.csv"),
        "--important-citations-path",
        str(repo / "dataset" / "important_citations.csv"),
        "--nugget-groundtruth-dir-path",
        str(repo / "dataset" / "gt_nuggets_outputs"),
        "--model-name",
        judge_model,
    ]
    _run(cmd, cwd=repo)
    return eval_dir / "results.csv"


def print_summary(results_csv: Path, sorcar_summary: Path) -> None:
    """Pretty-print evaluator + generator summaries.

    Args:
        results_csv: Path to the aggregated eval CSV.
        sorcar_summary: Path to the generator summary JSON produced by
            :func:`generate_all`.
    """
    import pandas as pd  # type: ignore[import-untyped]

    print("\n" + "=" * 70)
    print("DeepScholar-Bench — KISS Sorcar Results")
    print("=" * 70)

    if sorcar_summary.exists():
        raw = json.loads(sorcar_summary.read_text(encoding="utf-8"))
        n = len(raw)
        by_status: dict[str, int] = {}
        for r in raw:
            by_status[r.get("status", "unknown")] = by_status.get(r.get("status", "unknown"), 0) + 1
        print("\nGeneration (Sorcar) summary:")
        print(f"  Total queries : {n}")
        for k, v in sorted(by_status.items()):
            print(f"  {k:<32} {v}")

    if not results_csv.exists():
        print(f"\nWARNING: No eval results.csv at {results_csv}")
        return

    df = pd.read_csv(results_csv)
    print("\nEvaluator results (raw):")
    print(df.to_string(index=False))

    # Extract per-metric mean for the openai_deepresearch mode row.
    row = df[df["baseline_name"] == EVAL_PARSER_MODE]
    if not row.empty:
        row = row.iloc[0]
        print("\nHeadline metrics (mean over graded tasks):")
        for col in row.index:
            if col == "baseline_name":
                continue
            val = row[col]
            print(f"  {col:<32} {val}")


def main() -> None:
    """CLI entry point for the DeepScholar-Bench Sorcar harness."""
    parser = argparse.ArgumentParser(
        description="Evaluate KISS Sorcar on DeepScholar-Bench.",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Model to pass to sorcar -m for report generation.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="LLM used as the LLM-judge by DeepScholar-Bench evaluators.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd() / "deepscholar_bench_workdir",
        help="Scratch dir for the deepscholar checkout, venv and outputs.",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=4,
        help="Number of concurrent Sorcar invocations.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="Cap the number of queries evaluated (default: all).",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Skip the first START_IDX queries.",
    )
    parser.add_argument(
        "--per-task-timeout",
        type=int,
        default=1200,
        help="Per-query Sorcar timeout in seconds (default 1200).",
    )
    parser.add_argument(
        "--sorcar-budget",
        type=float,
        default=1.25,
        help="Maximum Sorcar spend in USD per query (default: 1.25).",
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        default=list(DEFAULT_EVALS),
        help="Evaluation metrics to run (or 'all').",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip the Sorcar generation step; run eval on existing outputs.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the evaluation step; only run Sorcar generation.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate selected tasks even when output.md already exists.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove results_dir before running generation.",
    )
    args = parser.parse_args()
    if args.n_concurrent < 1:
        parser.error("--n-concurrent must be at least 1")
    if args.n_tasks is not None and args.n_tasks < 1:
        parser.error("--n-tasks must be at least 1")
    if args.start_idx < 0:
        parser.error("--start-idx must be non-negative")
    if args.per_task_timeout < 1:
        parser.error("--per-task-timeout must be at least 1 second")
    if args.sorcar_budget <= 0:
        parser.error("--sorcar-budget must be positive")

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    repo = ensure_repo(workdir)

    results_dir = workdir / "outputs" / "results"
    eval_dir = workdir / "outputs" / "evaluation"

    if args.clean and results_dir.exists() and not args.skip_generation:
        shutil.rmtree(results_dir)

    if not args.skip_generation:
        queries = load_queries(repo, args.n_tasks, args.start_idx)
        print(
            f"Running Sorcar on {len(queries)} queries "
            f"(start_idx={args.start_idx}, model={args.model}, "
            f"n_concurrent={args.n_concurrent})",
            flush=True,
        )
        # Ensure sorcar CLI is on PATH before we spend budget on the venv.
        if shutil.which("sorcar") is None:
            print(
                "ERROR: `sorcar` CLI not found on PATH. Install kiss-agent-framework "
                "(e.g. `uv tool install --python 3.13 .`) before running this harness.",
                file=sys.stderr,
            )
            sys.exit(1)
        generate_all(
            queries=queries,
            results_dir=results_dir,
            model=args.model,
            n_concurrent=args.n_concurrent,
            per_task_timeout=args.per_task_timeout,
            max_budget=args.sorcar_budget,
            overwrite_existing=args.overwrite_existing,
        )

    results_csv = eval_dir / "results.csv"
    if not args.skip_eval:
        python = ensure_eval_venv(repo)
        results_csv = run_eval(
            repo=repo,
            python=python,
            results_dir=results_dir,
            eval_dir=eval_dir,
            judge_model=args.judge_model,
            evals=args.evals,
        )

    print_summary(results_csv, results_dir / "summary.json")


if __name__ == "__main__":
    main()
