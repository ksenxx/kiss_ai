# Author: Koushik Sen (ksen@berkeley.edu)

"""Run WebArena with the sorcar agent.

Usage:
    python -m kiss.benchmarks.webarena.run \
        --config-dir path/to/webarena/config_files \
        --model claude-opus-4-6

WebArena requires its web application servers to be running locally.
See README.md for setup instructions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kiss.benchmarks.webarena.agent import SorcarWebArenaAgent

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_webarena(
    config_dir: Path,
    model: str = "claude-opus-4-6",
    max_tasks: int | None = None,
    timeout: int = 600,
) -> None:
    """Run sorcar on WebArena task configs and save results.

    Args:
        config_dir: Directory containing WebArena JSON task configs.
        model: LLM model name (e.g. "claude-opus-4-6").
        max_tasks: Cap for quick testing (None = all configs).
        timeout: Max seconds per task.
    """
    config_files = sorted(config_dir.glob("*.json"))
    if not config_files:
        print(f"ERROR: No JSON configs found in {config_dir}", file=sys.stderr)
        sys.exit(1)

    if max_tasks is not None:
        config_files = config_files[:max_tasks]

    agent = SorcarWebArenaAgent(model=model, timeout=timeout)
    results = []

    for i, config_file in enumerate(config_files):
        print(f"[{i + 1}/{len(config_files)}] {config_file.name}")
        result = agent.run_task(config_file)
        results.append(result)

        score = result["score"]
        if score == 1.0:
            status = "✅ pass"
        elif score == 0.0:
            status = "❌ fail"
        else:
            status = "⚠️  n/a (non-string eval)"
        print(f"  {status}  answer: {result['answer'][:80]}")

    scoreable = [r for r in results if r["score"] >= 0.0]
    if scoreable:
        accuracy = sum(r["score"] for r in scoreable) / len(scoreable)
        print(f"\nAccuracy: {accuracy:.1%} ({len(scoreable)} scoreable tasks)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "webarena_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {output_path}")


def main() -> None:
    """CLI entry point for WebArena benchmark runner."""
    parser = argparse.ArgumentParser(description="Run WebArena with sorcar agent")
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory containing WebArena JSON task config files",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="LLM model name",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Max tasks to run (None = all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Max seconds per task",
    )
    args = parser.parse_args()
    run_webarena(args.config_dir, args.model, args.max_tasks, args.timeout)


if __name__ == "__main__":
    main()
