# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests targeting uncovered sorcar branches. No mocks or test doubles."""

from __future__ import annotations

import pytest


class TestRelentlessAgentDockerBash:
    def test_docker_bash_raises_without_manager(self) -> None:
        from kiss.agents.sorcar.relentless_agent import RelentlessAgent
        from kiss.core.kiss_error import KISSError

        agent = RelentlessAgent("test")
        agent._reset(
            model_name="gemini-3-flash-preview",
            max_sub_sessions=1,
            max_steps=3,
            max_budget=0.01,
            work_dir=None,
            docker_image=None,
        )
        with pytest.raises(KISSError, match="Docker manager not initialized"):
            agent._docker_bash("echo hi", "test")
