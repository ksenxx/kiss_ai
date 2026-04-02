# Author: Koushik Sen (ksen@berkeley.edu)

"""Run sorcar on SWE-bench Pro instances and collect patches.

Usage:
    python -m kiss.benchmarks.swebench_pro.run \
        --model claude-opus-4-6 \
        --budget 2.00 \
        --max-instances 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from kiss.benchmarks.swebench_pro.adapter import make_sorcar_task

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_instance(instance: dict, model: str, budget: float) -> dict:
    """Run sorcar on a single SWE-bench Pro instance inside its Docker container.

    Steps:
    1. docker run the instance's image (jefzda/sweap-images:<dockerhub_tag>)
    2. Inside the container, invoke sorcar with the task prompt
    3. Capture the generated patch (git diff from /app)
    4. Return {"instance_id": ..., "model_patch": ..., "prefix": model}

    Note: The patch key is "model_patch" (also accepted: "patch") to match
    the format expected by swe_bench_pro_eval.py.

    Args:
        instance: A SWE-bench Pro dataset row.
        model: LLM model name (e.g. "claude-opus-4-6").
        budget: Max USD budget per instance.

    Returns:
        A dict with instance_id, model_patch, and prefix fields.
    """
    task = make_sorcar_task(instance)
    instance_id = instance["instance_id"]
    dockerhub_tag = instance["dockerhub_tag"]
    image = f"jefzda/sweap-images:{dockerhub_tag}"

    # Build sorcar command to run inside the container
    escaped_task = task.replace('"', '\\"')
    sorcar_cmd = (
        f'sorcar -t "{escaped_task}" -w /app --headless true -n '
        f"--model {model} --budget {budget}"
    )

    # Run inside Docker, capture output
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        f"ANTHROPIC_API_KEY={os.environ.get('ANTHROPIC_API_KEY', '')}",
        "-e",
        f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', '')}",
        "-e",
        f"GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY', '')}",
        image,
        "bash",
        "-c",
        f"pip install -q kiss-sorcar && {sorcar_cmd} && cd /app && git diff",
        f"pip install -q kiss-agent-framework && {sorcar_cmd} && cd /app && git diff",
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        patch = result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        print(f"  ERROR on {instance_id}: {e}", file=sys.stderr)
        patch = ""

    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "prefix": model,
    }


def run_all(
    model: str,
    budget: float,
    max_instances: int | None = None,
    workers: int = 1,
) -> None:
    """Iterate over all SWE-bench Pro public instances, generate patches.

    Args:
        model: LLM model name (e.g. "claude-opus-4-6").
        budget: Max USD budget per instance.
        max_instances: Cap for quick testing (None = all 731).
        workers: Number of parallel workers (currently sequential).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' package required. Install with: uv pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    results = []

    for i, instance in enumerate(dataset):
        if max_instances is not None and i >= max_instances:
            break
        print(f"[{i + 1}] Running {instance['instance_id']}...")
        result = run_instance(instance, model, budget)
        results.append(result)
        status = "✅ patch" if result["model_patch"] else "❌ empty"
        print(f"  {status}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "swebench_pro_patches.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} patches to {output_path}")


def main() -> None:
    """CLI entry point for SWE-bench Pro benchmark runner."""
    parser = argparse.ArgumentParser(description="Run sorcar on SWE-bench Pro")
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="LLM model name",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=2.0,
        help="Max USD budget per instance",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Max instances to run (None = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    args = parser.parse_args()
    run_all(args.model, args.budget, args.max_instances, args.workers)


if __name__ == "__main__":
    main()
