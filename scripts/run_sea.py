#!/usr/bin/env python
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Run a SEA in this process: no kiss-web daemon, no UI.

    uv run python scripts/run_sea.py <sea> "<task>" [--work-dir DIR] [-m MODEL] [-b USD]

``<sea>`` is a SEA name (``sh``, ``weather``: resolved through the same
registry the daemon uses, i.e. the bundled ``seas/`` folder, the folders
listed in ``$KISS_HOME/SEAS.md`` and ``third_party_agents/``) or the path
of a SEA file (``<name>/<name>_sea.py``).  Loads it exactly as the daemon
does (``sea_layers`` + ``evaluate_sea``) and runs ``SorcarAgent.run`` with
the SEA's settings, prompt and hooks.
Needs only an API key in the environment (``$KISS_HOME/api_keys.env`` is
loaded).  Prints the agent's YAML result; exits 0 on success, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def resolve_sea(sea: str) -> Path:
    """Return the SEA script for *sea*, a registered SEA name or a file path.

    A path wins when it names an existing file; otherwise the name is
    looked up in the SEA registry.  Raises ``SystemExit`` with the list
    of known names when neither matches.
    """
    from kiss.agents.sorcar.sea_commands import get_command, list_commands

    path = Path(sea)
    if path.is_file():
        return path.resolve()
    found = get_command(sea)
    if found is None:
        sys.exit(f"unknown SEA {sea!r}; known names: {', '.join(list_commands())}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("sea", help="SEA name (e.g. sh) or path of a SEA file (<n>/<n>_sea.py)")
    parser.add_argument("task", help="the task text the SEA's prompt(task) receives")
    parser.add_argument("--work-dir", default="", help="working directory (default: cwd)")
    parser.add_argument("-m", "--model", default="", help="model (SEA 'model' setting wins)")
    parser.add_argument("-b", "--max-budget", type=float, default=None, help="USD budget")
    args = parser.parse_args()

    from kiss.core.vscode_config import load_api_keys_readonly

    load_api_keys_readonly()

    from kiss.agents.sorcar.sea_commands import evaluate_sea, sea_layers
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    sea_path = resolve_sea(args.sea)
    run = evaluate_sea(sea_layers(sea_path), args.task)
    s = run.settings
    work_dir = s.get("work_dir") or args.work_dir or os.getcwd()
    result = SorcarAgent(sea_path.stem).run(
        model_name=s.get("model") or args.model or None,
        prompt_template=run.prompt,
        work_dir=work_dir,
        max_budget=s.get("max_budget", args.max_budget),
        tool_profile=s.get("tool_profile", ""),
        web_tools=bool(s.get("use_web_tools", False)),
        use_memory=s.get("use_memory"),
        model_config=s.get("model_config"),
        docker_image=s.get("docker_image") or None,
        system_prompt_hook=run.system_prompt_hook,
        tools_hook=run.tools_hook,
        llm_call_hook=run.llm_call_hook,
        tool_call_hook=run.tool_call_hook,
        verbose=sys.stdout.isatty(),
    )
    if not sys.stdout.isatty():
        print(result)
    try:
        return 0 if yaml.safe_load(result).get("success") else 1
    except Exception:
        # A ``finish`` called with wrong arguments ends the run with the
        # tool error text, not a YAML mapping.
        return 1


if __name__ == "__main__":
    sys.exit(main())
