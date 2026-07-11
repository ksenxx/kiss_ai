# DeepScholar-Bench for KISS Sorcar

Evaluates KISS Sorcar on
[DeepScholar-Bench](https://github.com/guestrin-lab/deepscholar), the
UC Berkeley / Guestrin-Lab live benchmark for generative research
synthesis (paper: [arXiv:2508.20033](https://arxiv.org/abs/2508.20033)).

The harness

1. Clones `guestrin-lab/deepscholar` at a pinned commit into a scratch
   working directory.
1. Creates a Python 3.10 venv (via `uv`) and installs the benchmark's
   evaluator dependencies (LOTUS + friends).
1. For every query in `dataset/papers_with_related_works.csv`, launches
   `sorcar -t <prompt> -w <task-dir> -m <model>` in parallel. Sorcar
   produces a single `output.md` per task with inline
   `[number](https://arxiv.org/abs/<id>)` citations — the format the
   `openai_deepresearch` parser understands.
1. Runs `python -m eval.main --modes openai_deepresearch --evals all`
   inside the cloned repo.
1. Prints an aggregated summary from `results.csv`.

## Quick start

```bash
uv run python -m kiss.benchmarks.deepscholar_bench.run \
    --model claude-opus-4-7 \
    --n-concurrent 4 \
    --n-tasks 10
```

`--n-tasks` caps the number of queries so you can smoke-test cheaply.
Drop the flag to run the full benchmark (63 queries at the pinned
commit).

## Required environment variables

- `ANTHROPIC_API_KEY` (or the provider key your `--model` uses) —
  consumed by Sorcar for report generation.
- `OPENAI_API_KEY` — required by DeepScholar-Bench's LLM-judge
  evaluators (default judge model `gpt-4o`).
- `TAVILY_API_KEY` — required by some upstream evaluators for web
  lookups; strongly recommended.

## CLI reference

| Flag | Default | Meaning |
| ------------------------ | ------------------- | ------- |
| `--model` | `claude-opus-4-7` | Model passed to `sorcar -m` for generation. |
| `--judge-model` | `gpt-4o` | Model used as the LLM-judge by DS-Bench evaluators. |
| `--workdir` | `./deepscholar_bench_workdir` | Scratch dir for the clone, venv and outputs. |
| `--n-concurrent` | `4` | Number of concurrent Sorcar invocations. |
| `--n-tasks` | all | Cap the number of queries evaluated. |
| `--start-idx` | `0` | Skip the first N queries. |
| `--per-task-timeout` | `1200` (s) | Per-query Sorcar timeout. |
| `--sorcar-budget` | `$1.25` | Maximum Sorcar spend per query. |
| `--evals` | see `DEFAULT_EVALS` | Metrics to run. |
| `--skip-generation` | off | Reuse existing outputs, only run eval. |
| `--skip-eval` | off | Only run Sorcar; skip scoring. |
| `--overwrite-existing` | off | Regenerate selected existing outputs. |
| `--clean` | off | Delete `outputs/results/` before running. |

Non-empty existing `output.md` files are reused by default, so an interrupted
full run can be resumed. Use `--overwrite-existing` for selected-task
regeneration or `--clean` for a completely fresh run.

## Outputs

```
<workdir>/
├── deepscholar/                       # pinned checkout
├── deepscholar/.venv/                 # Python 3.10 evaluator venv
├── outputs/
│   ├── results/<idx>/output.md        # Sorcar-generated related work
│   ├── results/<idx>/sorcar_stdout.log
│   ├── results/<idx>/sorcar_stderr.log
│   ├── results/summary.json           # per-task generation status
│   └── evaluation/results.csv         # aggregated metrics
```

## Metrics

DeepScholar-Bench reports (see `deepscholar/eval/evaluator/`):

- `organization` — structural coherence of the generated related-work.
- `nugget_coverage` — fraction of ground-truth "nuggets" covered.
- `reference_coverage` — fraction of important prior citations included.
- `cite_p` — citation precision (each claim traceable to a cited paper).
- `claim_coverage`, `coverage_relevance_rate`, `document_importance` —
  additional retrieval-quality signals.

The authoritative numbers are those printed in the final summary
table from `outputs/evaluation/results.csv`.

## References

- Paper: [DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis](https://arxiv.org/abs/2508.20033)
- Leaderboard: <https://guestrin-lab.github.io/deepscholar-leaderboard/leaderboard/deepscholar_bench_leaderboard.html>
- Repo: <https://github.com/guestrin-lab/deepscholar>
