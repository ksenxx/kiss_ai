#!/usr/bin/env python
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Run a SEA in this process: no kiss-web daemon, no UI.

    uv run python scripts/run_sea.py <sea> "<task>" [--work-dir DIR] [-m MODEL] [-b USD]
                                     [--no-trajectories] [--json]

``<sea>`` is a SEA name (``sh``, ``weather``: resolved through the same
registry the daemon uses, i.e. the bundled ``seas/`` folder, the folders
listed in ``$KISS_HOME/SEAS.md`` and ``third_party_agents/``) or the path
of a SEA file (``<name>/<name>_sea.py``).  The command-line form of
:func:`kiss.agents.sorcar.headless.run_sea`, which loads the SEA exactly
as the daemon does and runs it with the SEA's settings, prompt and hooks.
Needs only an API key in the environment (``$KISS_HOME/api_keys.env`` is
loaded).  Prints the agent's YAML result (with ``--json``, the whole
outcome: result, error, counters and the trajectory files the run
wrote); exits 0 on success, 1 otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("sea", help="SEA name (e.g. sh) or path of a SEA file (<n>/<n>_sea.py)")
    parser.add_argument("task", help="the task text the SEA's prompt(task) receives")
    parser.add_argument("--work-dir", default="", help="working directory (default: cwd)")
    parser.add_argument("-m", "--model", default="", help="model (SEA 'model' setting wins)")
    parser.add_argument("-b", "--max-budget", type=float, default=None, help="USD budget")
    parser.add_argument(
        "--no-trajectories", action="store_true", help="write no trajectory files"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the whole outcome as JSON, not the result"
    )
    args = parser.parse_args()

    from kiss.agents.sorcar.headless import run_sea
    from kiss.agents.sorcar.sea_settings import SeaError

    try:
        outcome = run_sea(
            args.sea,
            args.task,
            model=args.model,
            work_dir=args.work_dir,
            max_budget=args.max_budget,
            save_trajectories=not args.no_trajectories,
            verbose=sys.stdout.isatty() and not args.json,
        )
    except SeaError as exc:
        sys.exit(str(exc))
    if args.json:
        payload = dataclasses.asdict(outcome)
        payload["trajectory_paths"] = [str(path) for path in outcome.trajectory_paths]
        print(json.dumps(payload, indent=2))
    elif not sys.stdout.isatty():
        print(outcome.result or outcome.error)
    elif outcome.error:
        print(outcome.error, file=sys.stderr)
    return 0 if outcome.success else 1


if __name__ == "__main__":
    sys.exit(main())
