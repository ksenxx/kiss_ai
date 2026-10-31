"""``connectors/enable.py enable`` reports an unusable MCP config instead of crashing.

``save_mcp_server`` raises ``ValueError`` (with a "fix or remove it" message
written for the user) when ``$KISS_HOME/mcp.json`` is not an MCP config, and
``OSError`` when it cannot be written.  ``cmd_enable`` let both escape as a
traceback, unlike every other failure of the CLI, which prints a line and
exits with status 1.  The CLI runs as a real subprocess on a temp KISS home.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import kiss

REPO_ROOT = Path(kiss.__file__).resolve().parents[2]
ENABLE_SCRIPT = REPO_ROOT / "connectors" / "enable.py"


def _enable(kiss_home: Path, work_dir: Path, name: str) -> subprocess.CompletedProcess[str]:
    """Run ``enable.py enable <name>`` with ``$KISS_HOME`` = *kiss_home* in *work_dir*."""
    return subprocess.run(
        [sys.executable, str(ENABLE_SCRIPT), "enable", name],
        cwd=work_dir,
        env={**os.environ, "KISS_HOME": str(kiss_home)},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_enable_into_a_malformed_config_prints_the_problem_and_exits_1(tmp_path: Path) -> None:
    """A syntax error in ``mcp.json`` is reported in one line and the file is untouched."""
    config = tmp_path / "mcp.json"
    config.write_text('{"mcpServers": {"mine": ', encoding="utf-8")
    proc = _enable(tmp_path, tmp_path, "deepwiki")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    assert f"could not enable deepwiki: {config} is not valid JSON" in proc.stdout
    assert "fix or remove it" in proc.stdout
    assert config.read_text(encoding="utf-8") == '{"mcpServers": {"mine": '


def test_enable_into_a_good_config_adds_the_server(tmp_path: Path) -> None:
    """The success path still writes the server and keeps the other entries."""
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"mine": {"command": "x"}}}), encoding="utf-8")
    proc = _enable(tmp_path, tmp_path, "deepwiki")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"enabled deepwiki in {config}" in proc.stdout
    servers = json.loads(config.read_text(encoding="utf-8"))["mcpServers"]
    assert set(servers) == {"mine", "deepwiki"}
