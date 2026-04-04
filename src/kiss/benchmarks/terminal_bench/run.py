# Author: Koushik Sen (ksen@berkeley.edu)

"""Run Terminal-Bench 2.0 with the sorcar harbor agent.

Usage:
    python -m kiss.benchmarks.terminal_bench.run \
        --model anthropic/claude-opus-4-6 \
        --n-concurrent 8

This uses harbor's --agent-import-path flag to load our custom
SorcarHarborAgent without needing to modify the harbor source.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

AGENT_IMPORT_PATH = "kiss.benchmarks.terminal_bench.agent:SorcarHarborAgent"


def is_docker_hub_authenticated() -> bool:
    """Check whether Docker Hub credentials are configured.

    Reads ~/.docker/config.json to find the credential store, then
    queries it via ``docker-credential-<store> list``.  Returns True
    if any credential is stored for ``https://index.docker.io/``.
    Falls back to checking the ``auths`` dict when no credential
    store is configured.
    """
    config_path = Path.home() / ".docker" / "config.json"
    if not config_path.exists():  # pragma: no branch
        return False

    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    creds_store = config.get("credsStore")
    if creds_store:  # pragma: no branch
        try:
            result = subprocess.run(
                [f"docker-credential-{creds_store}", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:  # pragma: no branch
                creds = json.loads(result.stdout)
                return any(
                    "index.docker.io" in url for url in creds
                )
        except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
            pass

    auths = config.get("auths", {})
    return any("index.docker.io" in url for url in auths)


async def _resolve_docker_images(dataset: str) -> list[str]:
    """Resolve unique Docker image names from a harbor dataset.

    Uses harbor's ``DatasetConfig`` to download/cache task definitions,
    then reads each task's ``task.toml`` to extract ``docker_image``.

    Args:
        dataset: Harbor dataset specifier (e.g. "terminal-bench@2.0").

    Returns:
        Sorted list of unique Docker image names.
    """
    try:
        from harbor.models.job.config import DatasetConfig
        from harbor.models.task.config import TaskConfig as TaskTomlConfig
    except ImportError:
        return []

    parts = dataset.split("@", 1)
    ds = DatasetConfig(
        name=parts[0],
        version=parts[1] if len(parts) > 1 else None,
    )
    task_configs = await ds.get_task_configs()

    images: set[str] = set()
    for tc in task_configs:
        task_toml = tc.get_local_path() / "task.toml"
        if task_toml.exists():  # pragma: no branch
            try:
                cfg = TaskTomlConfig.model_validate_toml(task_toml.read_text())
                if cfg.environment.docker_image:  # pragma: no branch
                    images.add(cfg.environment.docker_image)
            except Exception:
                continue
    return sorted(images)


def pre_pull_images(dataset: str) -> None:
    """Pre-pull all Docker images needed by a harbor dataset.

    Resolves the dataset's task definitions, extracts unique Docker
    image names, and pulls each one sequentially.  Because Docker
    caches pulled images locally, subsequent ``docker compose up``
    calls by harbor will not trigger additional pulls, avoiding
    Docker Hub rate limits.

    Args:
        dataset: Harbor dataset specifier (e.g. "terminal-bench@2.0").
    """
    images = asyncio.run(_resolve_docker_images(dataset))
    if not images:  # pragma: no branch
        print("No Docker images to pre-pull.")
        return

    print(f"Pre-pulling {len(images)} Docker images...")
    failed: list[str] = []
    for i, image in enumerate(images, 1):  # pragma: no branch
        print(f"  [{i}/{len(images)}] {image}")
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:  # pragma: no branch
            failed.append(image)
            print(f"    WARN: pull failed: {result.stdout.strip()}")

    if failed:  # pragma: no branch
        print(
            f"WARNING: Failed to pull {len(failed)}/{len(images)} images.",
            file=sys.stderr,
        )
    else:
        print(f"All {len(images)} images pulled successfully.")


def run_terminal_bench(
    model: str = "anthropic/claude-opus-4-6",
    dataset: str = "terminal-bench@2.0",
    n_concurrent: int = 8,
    trials: int = 1,
    skip_pre_pull: bool = False,
) -> None:
    """Run Terminal-Bench 2.0 using the harbor CLI with the sorcar agent.

    Before invoking harbor, checks that Docker Hub credentials are
    configured (to avoid unauthenticated pull rate limits) and pre-pulls
    all task Docker images so each unique image is fetched exactly once.

    Args:
        model: Model name in harbor format (provider/model).
        dataset: Harbor dataset specifier (e.g. "terminal-bench@2.0").
        n_concurrent: Number of concurrent task containers.
        trials: Number of attempts per task (-k flag). Use 5 for leaderboard.
        skip_pre_pull: If True, skip the image pre-pull step.
    """
    if not is_docker_hub_authenticated():  # pragma: no branch
        print(
            "WARNING: Not authenticated to Docker Hub.\n"
            "  Without authentication, Docker Hub limits pulls to 100 per 6 hours.\n"
            "  Run 'docker login' to avoid rate limits.\n",
            file=sys.stderr,
        )

    if not skip_pre_pull:  # pragma: no branch
        pre_pull_images(dataset)

    cmd = [
        "harbor",
        "run",
        "--dataset",
        dataset,
        "--agent-import-path",
        AGENT_IMPORT_PATH,
        "--model",
        model,
        "--n-concurrent",
        str(n_concurrent),
        "-k",
        str(trials),
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "ERROR: 'harbor' CLI not found. Install with: uv pip install harbor",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: harbor exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


def score_results(results_path: Path) -> None:
    """Print a graded summary table from a harbor results JSON file.

    Reads harbor's output JSON (list of task result dicts) and prints
    binary score, partial score (fraction of tests passed), and a
    summary line. Tasks with no partial score data (skipped or missing
    metadata) show "-" in the partial column.

    Args:
        results_path: Path to harbor results JSON file.
    """
    try:
        results: list[dict[str, Any]] = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: Could not read {results_path}: {e}", file=sys.stderr)
        sys.exit(1)

    binary_total = 0
    partial_sum = 0.0
    partial_count = 0

    print(f"\n{'Task':<45} {'Binary':>7} {'Partial':>8} {'Tests':>10}")
    print("-" * 73)

    for r in results:
        task_id = r.get("task_id") or r.get("name") or "unknown"
        binary = r.get("score", 0)
        meta = r.get("metadata") or {}
        passed = meta.get("tests_passed")
        total = meta.get("tests_total")
        partial = meta.get("partial_score")

        binary_total += int(bool(binary))

        if partial is not None:
            partial_sum += partial
            partial_count += 1
            partial_str = f"{partial:.1%}"
            tests_str = f"{passed}/{total}"
        else:
            partial_str = "-"
            tests_str = "-"

        binary_str = "pass" if binary else "fail"
        print(f"{str(task_id):<45} {binary_str:>7} {partial_str:>8} {tests_str:>10}")

    print("-" * 73)
    n = len(results)
    binary_pct = binary_total / n if n else 0.0
    partial_avg = partial_sum / partial_count if partial_count else 0.0
    print(f"{'TOTAL':<45} {binary_pct:.1%} ({binary_total}/{n})")
    if partial_count:
        print(
            f"{'PARTIAL AVG (scoreable tasks)':<45} {partial_avg:.1%}"
            f" ({partial_count} tasks)"
        )


def main() -> None:
    """CLI entry point for Terminal-Bench 2.0 runner."""
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench 2.0 with sorcar agent"
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-opus-4-6",
        help="Model name in provider/model format",
    )
    parser.add_argument(
        "--dataset",
        default="terminal-bench@2.0",
        help="Harbor dataset specifier",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=8,
        help="Number of concurrent containers",
    )
    parser.add_argument(
        "-k",
        "--trials",
        type=int,
        default=1,
        help="Number of attempts per task (use 5 for leaderboard submission)",
    )
    parser.add_argument(
        "--skip-pre-pull",
        action="store_true",
        help="Skip pre-pulling Docker images (not recommended)",
    )
    parser.add_argument(
        "--score-results",
        type=Path,
        default=None,
        metavar="RESULTS_JSON",
        help="Print graded summary from a harbor results JSON file and exit",
    )
    args = parser.parse_args()
    if args.score_results:
        score_results(args.score_results)
        return
    run_terminal_bench(
        args.model,
        args.dataset,
        args.n_concurrent,
        args.trials,
        args.skip_pre_pull,
    )


if __name__ == "__main__":
    main()
