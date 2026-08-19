# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent-script contract tests extracted from
``kiss.tests.agents.third_party_agents.test_agent_dispatch``.

Moved here because their full dependency closure touches only
kiss.agents.sorcar (the cron agent module) and kiss.server (the
daemon's ``apply_agent_overrides`` agent-file loader) — unlike the
rest of the dispatch suite, they never create a ``run_agent`` tool or
resolve a channel, so the third-party package is not involved.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from kiss.agents.sorcar import cron_agent
from kiss.server.agent_file import AgentFileError, apply_agent_overrides


def test_cron_agent_module_is_a_valid_agent_script() -> None:
    # The contract the cron dispatch relies on: passing the cron
    # module as ``extension_agent_path`` makes it its own tools file (its
    # ``get_tools()`` returns the cron_job tool) and moves the session
    # to ~/.kiss/cron/work with no git lifecycle.
    cmd = {"agentPath": cron_agent.__file__, "toolsFile": ""}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {"toolsFile", "workDir", "useWorktree", "autoCommit"}
    assert cmd["toolsFile"] == cron_agent.__file__
    assert cmd["workDir"] == cron_agent.get_work_dir()
    assert cmd["useWorktree"] is False
    assert cmd["autoCommit"] is False

def test_agent_script_get_tools_list_normalizes_to_own_path(
    tmp_path: Path,
) -> None:
    script = tmp_path / "self_tools_agent.py"
    script.write_text(textwrap.dedent("""
        def _hello() -> str:
            \"\"\"Say hello.

            Returns:
                A greeting.
            \"\"\"
            return "hello"

        def get_tools() -> list:
            return [_hello]
    """))
    cmd = {"agentPath": str(script), "toolsFile": ""}
    assert apply_agent_overrides(cmd) == {"toolsFile"}
    assert cmd["toolsFile"] == str(script)

def test_agent_script_get_tools_wrong_type_still_rejected(
    tmp_path: Path,
) -> None:
    script = tmp_path / "bad_tools_agent.py"
    script.write_text("def get_tools():\n    return 42\n")
    cmd = {"agentPath": str(script), "toolsFile": ""}
    with pytest.raises(AgentFileError, match="get_tools"):
        apply_agent_overrides(cmd)
