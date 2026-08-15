#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
"""CLI to list, enable, and disable KISS Sorcar MCP connectors.

Connectors are described in ``catalog.json`` next to this file; enabling
one merges its ``mcpServers`` entry into ``~/.kiss/mcp.json`` (or the
project's ``.kiss/mcp.json`` with ``--scope project``) through Sorcar's
own config writer, so locking and atomic writes are shared with the
agent.  Secrets never enter the config: credential-based connectors read
environment variables that Sorcar's stdio launcher inherits from your
shell.

Usage:
    uv run python connectors/enable.py list
    uv run python connectors/enable.py enable github [--scope user|project]
    uv run python connectors/enable.py disable github
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kiss.agents.sorcar.mcp_servers import (  # noqa: E402
    MCPServerConfig,
    load_mcp_servers,
    remove_mcp_server,
    save_mcp_server,
)

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"


def load_catalog() -> dict[str, dict]:
    """Return the connector catalog keyed by connector name."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def missing_prereqs(entry: dict) -> list[str]:
    """Return the executables required by *entry* that are not on PATH."""
    return [p for p in entry.get("prereqs", []) if shutil.which(p) is None]


def missing_env(entry: dict) -> list[str]:
    """Return the required environment variables that are not set."""
    return [v for v in entry.get("required_env", []) if not os.environ.get(v)]


def expand_home(value: str) -> str:
    """Expand a leading ``~`` in *value* to the user's home directory."""
    return os.path.expanduser(value) if value.startswith("~") else value


def config_for(name: str, entry: dict) -> MCPServerConfig:
    """Build the :class:`MCPServerConfig` for catalog *entry*.

    ``~`` in the command and arguments is expanded so the entry written
    to ``mcp.json`` contains absolute paths.
    """
    raw = entry["config"]
    return MCPServerConfig(
        name=name,
        transport=raw.get("type", "stdio"),
        command=expand_home(raw.get("command", "")),
        args=tuple(expand_home(a) for a in raw.get("args", [])),
        env=tuple(raw.get("env", {}).items()),
        url=raw.get("url", ""),
        headers=tuple(raw.get("headers", {}).items()),
    )


def clone_if_needed(entry: dict) -> str | None:
    """Clone the connector's repository if the catalog asks for one.

    Returns:
        An error message on failure, else ``None``.
    """
    clone = entry.get("clone")
    if not clone:
        return None
    dest = Path(expand_home(clone["dest"]))
    if dest.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone["repo"], str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"git clone failed: {result.stderr.strip()}"
    print(f"cloned {clone['repo']} -> {dest}")
    return None


def cmd_list(work_dir: str) -> int:
    """Print every catalog connector with its live/enabled status."""
    catalog = load_catalog()
    enabled = load_mcp_servers(work_dir)
    width = max(len(n) for n in catalog)
    for name, entry in catalog.items():
        state = "ENABLED " if name in enabled else "disabled"
        blockers = []
        prereqs = missing_prereqs(entry)
        env = missing_env(entry)
        if prereqs:
            blockers.append("missing tools: " + ", ".join(prereqs))
        if env:
            blockers.append("missing env: " + ", ".join(env))
        suffix = f"  [{'; '.join(blockers)}]" if blockers else ""
        print(f"{name:<{width}}  {state}  auth={entry['auth']:<8}"
              f" {entry['description']}{suffix}")
    return 0


def cmd_enable(name: str, scope: str, force: bool, work_dir: str) -> int:
    """Enable connector *name* by writing it into the MCP config."""
    catalog = load_catalog()
    if name not in catalog:
        print(f"unknown connector: {name}; run `list` to see the catalog")
        return 1
    entry = catalog[name]
    problems = missing_prereqs(entry)
    if problems:
        print("missing executables: " + ", ".join(problems))
        if name == "whatsapp" and "go" in problems:
            print("install Go first: brew install go")
        if not force:
            print("fix the above or re-run with --force")
            return 1
    env = missing_env(entry)
    if env:
        print("missing environment variables: " + ", ".join(env))
        for step in entry.get("setup", []):
            print(f"  setup: {step}")
        if not force:
            print("export them in your shell profile (they are inherited by"
                  " the server at launch and never written to mcp.json),"
                  " or re-run with --force")
            return 1
    error = clone_if_needed(entry)
    if error:
        print(error)
        return 1
    path = save_mcp_server(config_for(name, entry), scope, work_dir)
    print(f"enabled {name} in {path}")
    for step in entry.get("setup", []):
        print(f"  next: {step}")
    if entry.get("notes"):
        print(f"  note: {entry['notes']}")
    return 0


def cmd_disable(name: str, work_dir: str) -> int:
    """Disable connector *name* by removing it from every config file."""
    removed = remove_mcp_server(name, work_dir)
    if removed:
        for path in removed:
            print(f"removed {name} from {path}")
        return 0
    print(f"{name} was not enabled")
    return 1


def main() -> int:
    """Parse arguments and dispatch to the subcommand handlers."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show all connectors and their status")
    enable = sub.add_parser("enable", help="enable a connector")
    enable.add_argument("name")
    enable.add_argument("--scope", choices=("user", "project"), default="user")
    enable.add_argument("--force", action="store_true",
                        help="enable even with missing prereqs/env")
    disable = sub.add_parser("disable", help="disable a connector")
    disable.add_argument("name")
    args = parser.parse_args()
    work_dir = os.getcwd()
    if args.cmd == "list":
        return cmd_list(work_dir)
    if args.cmd == "enable":
        return cmd_enable(args.name, args.scope, args.force, work_dir)
    return cmd_disable(args.name, work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
