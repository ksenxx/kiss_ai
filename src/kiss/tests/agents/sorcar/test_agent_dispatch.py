# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sorcar-layer test split out of the agent-dispatch suite.

Every other agent-dispatch test creates or uses the ``run_agent`` tool,
and ``make_run_agent_tool`` unconditionally expands its docstring with
``available_channels()`` — a real directory scan of
``src/kiss/agents/third_party_agents`` — so those tests are forced into
``tests/agents/third_party_agents`` by the third-party placement rule.
This test's closure is pure ``kiss.agents.sorcar.mcp_servers``.
"""

from __future__ import annotations


def test_dispatch_tools_reserved_against_mcp_collisions() -> None:
    from kiss.agents.sorcar.mcp_servers import _RESERVED_TOOL_NAMES

    assert {"run_agent", "cron_job"} <= _RESERVED_TOOL_NAMES
