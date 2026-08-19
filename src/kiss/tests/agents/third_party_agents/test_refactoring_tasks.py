# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for refactoring tasks: ChannelConfig, _find_tool_call_ids,
_build_openai_tools_schema, _resolve_openai_tools_schema,
_build_text_based_tools_prompt, _parse_text_based_tool_calls,
and related helpers.

The ``_ArtifactDirProxy`` tests moved to
``tests/core/test_refactoring_tasks.py``: they depend only on
``kiss.core.config``.

No mocks, patches, fakes, or any form of test doubles.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.third_party_agents._channel_agent_utils import ChannelConfig


class TestChannelConfig:
    """Integration tests for ChannelConfig: save, load, clear, missing keys, permissions."""

    def test_save_load_clear(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ("token",))
        cfg.save({"token": "abc123", "extra": "val"})
        loaded = cfg.load()
        assert loaded == {"token": "abc123", "extra": "val"}
        cfg.clear()
        assert cfg.load() is None
        assert not cfg.path.exists()
