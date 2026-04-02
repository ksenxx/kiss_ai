# SWE-bench Pro Benchmark for KISS Sorcar

Runs KISS Sorcar on [SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os)
(731 public tasks from real GitHub issues) and evaluates the generated patches.

## Setup

```bash
# Clone the vendor evaluation code
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git vendor
uv pip install -r vendor/requirements.txt
uv pip install datasets
```

## Quick Test (5 instances)

```bash
python -m kiss.benchmarks.swebench_pro.run \
    --model claude-opus-4-6 --budget 0.50 --max-instances 5
```

## Full Run

```bash
python -m kiss.benchmarks.swebench_pro.run \
    --model claude-opus-4-6 --budget 2.00 --workers 8
```

## Evaluate

```bash
python -m kiss.benchmarks.swebench_pro.eval \
    --patch-path ../results/swebench_pro_patches.json --num-workers 8
```

## References

- [SWE-bench Pro Leaderboard](https://scale.com/leaderboard/swe_bench_pro_public)
- [SWE-bench Pro GitHub](https://github.com/scaleapi/SWE-bench_Pro-os)
- [SWE-bench Pro HuggingFace Dataset](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)
