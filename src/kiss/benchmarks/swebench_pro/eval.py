# Author: Koushik Sen (ksen@berkeley.edu)

"""Thin wrapper around the official SWE-bench Pro evaluation script.

Usage:
    python -m kiss.benchmarks.swebench_pro.eval \
        --patch-path results/swebench_pro_patches.json \
        --num-workers 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parent / "vendor"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_eval(
    patch_path: str,
    num_workers: int = 8,
    use_local_docker: bool = True,
    block_network: bool = False,
    docker_platform: str | None = None,
    redo: bool = False,
) -> None:
    """Run the official SWE-bench Pro evaluation script.

    Args:
        patch_path: Path to the patches JSON file.
        num_workers: Number of parallel Docker workers.
        use_local_docker: Use local Docker instead of Modal.
        block_network: Block network access inside eval containers.
        docker_platform: Docker platform (e.g. "linux/amd64" for Apple Silicon).
        redo: Re-evaluate even if output exists.
    """
    eval_script = VENDOR_DIR / "swe_bench_pro_eval.py"
    raw_sample = VENDOR_DIR / "swe_bench_pro_full.csv"
    scripts_dir = VENDOR_DIR / "run_scripts"
    output_dir = RESULTS_DIR / "swebench_pro_eval"

    if not eval_script.exists():
        print(
            f"ERROR: Vendor directory not set up. Run:\n"
            f"  cd {VENDOR_DIR.parent}\n"
            f"  git clone https://github.com/scaleapi/SWE-bench_Pro-os.git vendor",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(eval_script),
        f"--raw_sample_path={raw_sample}",
        f"--patch_path={patch_path}",
        f"--output_dir={output_dir}",
        f"--scripts_dir={scripts_dir}",
        "--dockerhub_username=jefzda",
        f"--num_workers={num_workers}",
    ]
    if use_local_docker:
        cmd.append("--use_local_docker")
    if block_network:
        cmd.append("--block_network")
    if docker_platform:
        cmd.append(f"--docker_platform={docker_platform}")
    if redo:
        cmd.append("--redo")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nResults saved to {output_dir}")


def main() -> None:
    """CLI entry point for SWE-bench Pro evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate sorcar patches on SWE-bench Pro"
    )
    parser.add_argument(
        "--patch-path",
        default=str(RESULTS_DIR / "swebench_pro_patches.json"),
        help="Path to patches JSON",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--use-local-docker",
        action="store_true",
        default=True,
    )
    parser.add_argument("--block-network", action="store_true", default=False)
    parser.add_argument(
        "--docker-platform",
        default=None,
        help="e.g. linux/amd64 for Apple Silicon",
    )
    parser.add_argument("--redo", action="store_true", default=False)
    args = parser.parse_args()
    run_eval(
        args.patch_path,
        args.num_workers,
        args.use_local_docker,
        args.block_network,
        args.docker_platform,
        args.redo,
    )


if __name__ == "__main__":
    main()
