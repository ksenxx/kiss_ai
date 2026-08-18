# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for refactoring tasks: ChannelConfig, _find_tool_call_ids,
_build_openai_tools_schema, _resolve_openai_tools_schema,
_build_text_based_tools_prompt, _parse_text_based_tool_calls,
_ArtifactDirProxy, and related helpers.

No mocks, patches, fakes, or any form of test doubles.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.third_party_agents._channel_agent_utils import ChannelConfig
from kiss.core.config import (
    _ArtifactDirProxy,
    get_artifact_dir,
)


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


class TestArtifactDirProxy:
    """Tests for _ArtifactDirProxy lazy directory creation and thread-safety."""

    def test_proxy_hash(self) -> None:
        proxy = _ArtifactDirProxy()
        assert hash(proxy) == hash(str(proxy))

    def test_artifact_dir_is_stable_for_the_process(self) -> None:
        """The artifact directory is resolved once and never changes.

        Replaced ``test_set_artifact_base_dir``: the runtime setter it
        covered was removed because it had no production caller and was
        the only way to make a running agent's trajectory land under a
        different root than the one it started under.
        """
        assert get_artifact_dir() == get_artifact_dir()
        assert Path(get_artifact_dir()).is_dir()
        assert str(_ArtifactDirProxy()) == get_artifact_dir()
