# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``ChannelConfig`` save/load/clear round trip on a real temp directory.

The model-helper tests (``_find_tool_call_ids`` & co.) live in
``tests/core/models/test_refactoring_tasks.py`` and the
``_ArtifactDirProxy`` tests in ``tests/core/test_refactoring_tasks.py``.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.third_party_agents._channel_agent_utils import ChannelConfig


class TestChannelConfig:
    """ChannelConfig writes, reads back and removes its JSON file."""

    def test_save_load_clear(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ("token",))
        cfg.save({"token": "abc123", "extra": "val"})
        loaded = cfg.load()
        assert loaded == {"token": "abc123", "extra": "val"}
        cfg.clear()
        assert cfg.load() is None
        assert not cfg.path.exists()
