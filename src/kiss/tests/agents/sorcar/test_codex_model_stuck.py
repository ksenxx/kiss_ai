# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: codex/gpt-5.5 must not get stuck in an infinite retry loop.

Root cause: when a ``codex/*`` model fails on the first model call (e.g.
codex CLI not found, model not supported), ``RelentlessAgent.perform_task``
entered an infinite-like retry loop because:

1.  ``KISSAgent.step_count`` is 1 (incremented before the model call), not 0.
2.  The early-exit condition ``executor.step_count == 0`` never fires.
3.  The summarizer also uses the broken model and fails.
4.  ``is_continue: True`` is returned, causing the next sub-session.
5.  Budget stays at 0 (no tokens consumed), so budget checks never fire.
6.  The loop repeats for ``max_sub_sessions`` (default 10000) iterations.

Fix: change ``step_count == 0`` to ``step_count <= 1`` so that first-step
failures are treated as non-recoverable.
"""

from __future__ import annotations

import os
import tempfile

import yaml

from kiss.core.models import codex_model as codex_module


class TestCodexModelStuckBug:
    """Codex model failure must not spin forever in the RelentlessAgent loop."""

    def test_codex_model_returns_quickly_when_cli_missing(self) -> None:
        """When the codex CLI is not installed, the agent must fail promptly
        instead of retrying 10000 times in the RelentlessAgent loop."""
        from kiss.agents.sorcar.relentless_agent import RelentlessAgent

        saved_path = os.environ.get("PATH", "")
        saved_candidates = codex_module._UI_CANDIDATE_PATHS
        try:
            os.environ["PATH"] = ""
            codex_module._UI_CANDIDATE_PATHS = ()

            agent = RelentlessAgent("codex-stuck-test")
            agent._reset(
                model_name="codex/gpt-5.5",
                max_sub_sessions=100,
                max_steps=5,
                max_budget=10.0,
                work_dir=tempfile.mkdtemp(),
                docker_image=None,
            )
            agent.system_prompt = "You are a helpful assistant."
            agent.task_description = "Say hello"

            def noop() -> str:
                """A no-op tool."""
                return "ok"

            result = agent.perform_task([noop])
            parsed = yaml.safe_load(result)
            assert isinstance(parsed, dict)
            assert parsed["success"] is False
            assert parsed.get("is_continue", False) is False, (
                "Agent should NOT set is_continue=True when the model fails "
                "on the very first call — that causes infinite retries."
            )
        finally:
            os.environ["PATH"] = saved_path
            codex_module._UI_CANDIDATE_PATHS = saved_candidates
