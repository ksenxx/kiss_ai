# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Launch a Terminal-Bench job on Harbor with KISS Sorcar as the agent.

Builds the ``harbor run`` command with the benchmark's settings (the dataset,
the KISS agent, the task resource limits from each ``task.toml`` enforced by
Docker) and runs it in the foreground.  The benchmark daemon must be running
(:mod:`kiss.benchmarks.terminal_bench.daemon`).

Example (from the repo root)::

    uv run python -m kiss.benchmarks.terminal_bench.run --model claude-opus-5-5 \\
        -k 5 -n 40 --jobs-dir /data/jobs --job-name opus-5-5 \\
        -x gpu-task-a -x gpu-task-b

``--dataset`` names the Harbor dataset as ``org/name[@version]``; the default
is the latest release of ``terminal-bench/terminal-bench``, so pin a version
for a run that has to be reproducible.  ``-x`` excludes tasks (the GPU tasks
on a host without GPUs); the summary still counts them as failed attempts of
the full dataset.  ``--agent oracle`` runs the reference solutions instead,
the sandbox check Terminal-Bench asks for before an agent run.

The GPU tasks run on Modal (``modal token new`` first) with Harbor's
``modal`` environment; the agent's tools reach the sandbox through the exec
bridge (:mod:`kiss.benchmarks.terminal_bench.exec_bridge_server`)::

    uv run python -m kiss.benchmarks.terminal_bench.run --model claude-opus-5-5 \\
        --env modal -i gpu-task-a -i gpu-task-b \\
        -k 5 -n 3 --jobs-dir /data/jobs --job-name opus-5-5-gpu

``--effort max`` sends the model's highest reasoning effort with every request
(the leaderboard entries run at ``xhigh`` or ``max``); ``--reviewer`` names the
model of the finish gate's read-only reviews (``same`` for the trial's model,
``none`` to skip them) and ``--blind-verifier`` the model that writes the gate's
independent check from the task statement alone (same values).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from kiss.benchmarks.terminal_bench.daemon import DEFAULT_HOME, ENDPOINT_FILE

DATASET = "terminal-bench/terminal-bench"  # no ``@version``: Harbor's latest release
KISS_AGENT = "kiss.benchmarks.terminal_bench.harbor_agent:SeaHarborAgent"


def harbor_version() -> str:
    """The installed Harbor version (the ``modal`` extra is pinned to it)."""
    return importlib.metadata.version("harbor")


def harbor_command(args: argparse.Namespace) -> list[str]:
    """The ``harbor run`` argv for *args*.

    Args:
        args: Parsed CLI arguments (see :func:`main`).

    Returns:
        The command, starting with ``uv run harbor run`` (the project pins Harbor's
        version); ``--env modal`` adds ``--with harbor[modal]==<installed version>``
        so the ``modal`` SDK is on the path of the run.
    """
    cmd = ["uv", "run"]
    if args.env == "modal":  # Harbor's Modal environment imports the ``modal`` SDK
        cmd += ["--with", f"harbor[modal]=={harbor_version()}"]
    cmd += [
        "harbor", "run",
        "-d", args.dataset, "-a", args.agent,
        "-k", str(args.attempts), "-n", str(args.concurrency),
        "-o", str(args.jobs_dir), "--job-name", args.job_name,
        "--cpus", "limit", "--memory", "limit", "-y",
    ]
    if args.agent == KISS_AGENT:
        cmd += ["-m", args.model, "--ak", f"budget_usd={args.budget_usd}",
                "--ak", f"home={args.home}", "--ak", f"reviewer={args.reviewer}",
                "--ak", f"blind_verifier={args.blind_verifier}"]
        if args.effort:
            cmd += ["--ak", f"reasoning_effort={args.effort}"]
    if args.env:
        cmd += ["-e", args.env]
    org = args.dataset.split("/")[0]
    for task in args.exclude:  # Harbor matches the full ``org/name``
        cmd += ["-x", task if "/" in task else f"{org}/{task}"]
    for task in args.include:
        cmd += ["-i", task if "/" in task else f"{org}/{task}"]
    return cmd


def build_parser() -> argparse.ArgumentParser:
    """The launcher's command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="claude-opus-5-5", help="KISS model name")
    parser.add_argument("--agent", default=KISS_AGENT,
                        help="Harbor agent (default: the KISS agent; 'oracle' for the check)")
    parser.add_argument("-k", "--attempts", type=int, default=5, help="attempts per task")
    parser.add_argument("-n", "--concurrency", type=int, default=40, help="concurrent trials")
    parser.add_argument("--budget-usd", type=float, default=100.0, help="USD cap per trial")
    parser.add_argument("--home", default=str(DEFAULT_HOME), help="benchmark daemon's KISS_HOME")
    parser.add_argument("--dataset", default=DATASET,
                        help="Harbor dataset, org/name[@version] (default: the latest "
                             "release of %(default)s)")
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    parser.add_argument("--job-name", required=True)
    parser.add_argument("-x", "--exclude", action="append", default=[],
                        help="task to skip (bare name or org/name)")
    parser.add_argument("-i", "--include", action="append", default=[],
                        help="task to run (bare name or org/name); default: every task")
    parser.add_argument("--env", default="",
                        help="Harbor environment type (e.g. modal for the GPU tasks); "
                             "default: local Docker")
    parser.add_argument("--effort", default="max",
                        help="reasoning effort of every model request (Claude: low..max; "
                             "'' for the level of the model's catalog entry)")
    parser.add_argument("--reviewer", default="same",
                        help="model of the finish gate's read-only reviews: 'same' (the "
                             "trial's model), a model name, or 'none'")
    parser.add_argument("--blind-verifier", default="same",
                        help="model that writes the gate's independent check from the task "
                             "statement alone: 'same', a model name, or 'none'")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the Harbor job and return its exit status.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).
    """
    args = build_parser().parse_args(argv)
    endpoint = Path(args.home).expanduser() / ENDPOINT_FILE
    if args.agent == KISS_AGENT and not endpoint.is_file():
        print(f"benchmark daemon not running: {endpoint} is missing "
              f"(start it with: uv run python -m kiss.benchmarks.terminal_bench.daemon "
              f"--home {args.home})", file=sys.stderr)
        return 2
    cmd = harbor_command(args)
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})


if __name__ == "__main__":
    sys.exit(main())
