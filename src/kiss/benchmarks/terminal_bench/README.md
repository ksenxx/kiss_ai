# Terminal-Bench 2.0 Benchmark for KISS Sorcar

Runs KISS Sorcar on [Terminal-Bench 2.0](https://www.tbench.ai/) using the
[Harbor](https://github.com/harbor-framework/harbor) framework.

`harbor` is already declared as a project dependency, so `uv sync` (or any
`uv run ...` invocation inside this repo) will install it. Docker must also
be installed and running locally.

## Quick Run

```bash
uv run python -m kiss.benchmarks.terminal_bench.run \
    --model anthropic/claude-opus-4-6 --n-concurrent 8
```

This will:

1. Check that Docker Hub credentials are configured (warns if not — without
   `docker login`, Docker Hub limits anonymous pulls to 100 per 6 hours).
1. Pre-pull every unique Docker image referenced by the dataset's `task.toml`
   files so each image is fetched exactly once (with 3 retries). Skip this
   step with `--skip-pre-pull`.
1. Invoke the `harbor run` CLI with
   `--agent-import-path kiss.benchmarks.terminal_bench.agent:SorcarHarborAgent`
   and the chosen model / concurrency / trial count.

## Leaderboard Submission (5 trials per task)

The leaderboard requires `-k 5` (5 attempts per task) to compute confidence
intervals:

```bash
uv run python -m kiss.benchmarks.terminal_bench.run \
    --model anthropic/claude-opus-4-6 --n-concurrent 8 -k 5
```

Or using the harbor CLI directly:

```bash
uv run harbor run \
    --dataset terminal-bench@2.0 \
    --agent-import-path kiss.benchmarks.terminal_bench.agent:SorcarHarborAgent \
    --model anthropic/claude-opus-4-6 \
    --n-concurrent 8 \
    -k 5
```

## CLI Flags

| Flag | Default | Meaning |
| ----------------------- | ---------------------------- | ------------------------------------------------------------------ |
| `--model` | `anthropic/claude-opus-4-6` | Model name in harbor's `provider/model` format. |
| `--dataset` | `terminal-bench@2.0` | Harbor dataset specifier (`name@version`). |
| `--n-concurrent` | `8` | Number of concurrent task containers. |
| `-k`, `--trials` | `1` | Attempts per task. Use `5` for leaderboard submission. |
| `--skip-pre-pull` | off | Skip the Docker image pre-pull step (not recommended). |
| `--score-results PATH` | — | Print a graded summary table from a harbor results JSON and exit. |

## Scoring an Existing Results File

After a run, harbor writes a results JSON under its output directory. To
render a binary + partial score summary locally:

```bash
uv run python -m kiss.benchmarks.terminal_bench.run \
    --score-results path/to/harbor-results.json
```

The partial-score column reads `metadata.tests_passed` / `tests_total` /
`partial_score` if the agent (or a downstream tool) populated them, and
shows `-` otherwise. The authoritative leaderboard score is the binary
pass/fail computed by harbor's own verifier.

## Skipped Tasks

`SorcarHarborAgent` hard-skips 9 Terminal-Bench 2.0 tasks that have failed
0/6 across multiple internal evaluation runs (CompCert build, Windows 3.11
GUI install, YouTube/video OCR, GPT-2 code-golf, fastText training,
Caffe-CIFAR-10, Doom-for-MIPS, MTEB leaderboard, OCaml GC). Skipped tasks
leave the container untouched; harbor's verifier still runs and records
them as failed (binary score `0`), so they count against the total.
See the `_SKIP_PHRASES` tuple in
[agent.py](./agent.py) for the exact list.

## References

- [Terminal-Bench Leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0)
- [Terminal-Bench GitHub](https://github.com/harbor-framework/terminal-bench)
- [Terminal-Bench 2.0 GitHub](https://github.com/laude-institute/terminal-bench-2)
- [Harbor Framework](https://github.com/harbor-framework/harbor)
