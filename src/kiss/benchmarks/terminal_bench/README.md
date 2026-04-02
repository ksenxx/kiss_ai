# Terminal-Bench 2.0 Benchmark for KISS Sorcar

Runs KISS Sorcar on [Terminal-Bench 2.0](https://www.tbench.ai/) using the
[Harbor](https://github.com/harbor-framework/harbor) framework.

## Setup

```bash
# Install harbor (the official Terminal-Bench 2.0 harness)
uv pip install harbor
```

## Quick Run

```bash
python -m kiss.benchmarks.terminal_bench.run \
    --model claude-opus-4-6 --n-concurrent 8
```

## Leaderboard Submission (5 trials per task)

The leaderboard requires `-k 5` (5 attempts per task) to compute confidence
intervals:

```bash
python -m kiss.benchmarks.terminal_bench.run \
    --model claude-opus-4-6 --n-concurrent 8 -k 5
```

Or using the harbor CLI directly:

```bash
uv run harbor run \
    --dataset terminal-bench@2.0 \
    --agent-import-path kiss.benchmarks.terminal_bench.agent:SorcarHarborAgent \
    --model claude-opus-4-6 \
    --n-concurrent 8
```

## References

- [Terminal-Bench Leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0)
- [Terminal-Bench GitHub](https://github.com/harbor-framework/terminal-bench)
- [Terminal-Bench 2.0 GitHub](https://github.com/laude-institute/terminal-bench-2)
- [Harbor Framework](https://github.com/harbor-framework/harbor)
