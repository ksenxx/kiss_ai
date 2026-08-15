#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
"""End-to-end verification of every configured MCP connector.

Connects to each server visible from the current directory through
Sorcar's own MCP client (the same ``MCPManager`` the agent uses), lists
its tools, and reports per-server health.  Exits non-zero if any
configured server is unreachable, so this doubles as a smoke test.

Usage:
    uv run python connectors/verify.py [server ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kiss.agents.sorcar.mcp_servers import (  # noqa: E402
    MCPManager,
    load_mcp_servers,
)


def main() -> int:
    """Connect to every configured server and print tool inventories."""
    work_dir = os.getcwd()
    servers = load_mcp_servers(work_dir)
    only = set(sys.argv[1:])
    if only:
        servers = {n: c for n, c in servers.items() if n in only}
    if not servers:
        print("no MCP servers configured (see connectors/README.md)")
        return 1
    manager = MCPManager.instance()
    failures = 0
    for name, config in sorted(servers.items()):
        conn = manager.connect(config)
        if conn.session is None:
            failures += 1
            print(f"FAIL {name}: {conn.error or 'unknown error'}")
            continue
        tool_names = [t.name for t in conn.tools]
        print(f"OK   {name} ({len(tool_names)} tools): "
              + ", ".join(tool_names[:12])
              + (" ..." if len(tool_names) > 12 else ""))
    print(f"\n{len(servers) - failures}/{len(servers)} servers healthy")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
