"""Integration tests that reproduce bugs listed in bugs.md.

Each test demonstrates the buggy behavior. All tests should FAIL
until the corresponding bug is fixed. No mocks, patches, fakes,
or test doubles are used.
"""

import inspect


class TestC4ThoughtSignaturesNotCleared:
    def test_reset_conversation_clears_thought_signatures(self) -> None:
        """reset_conversation() should clear _thought_signatures.

        The bug: only initialize() clears it, so stale signatures
        accumulate across sub-sessions.
        """
        from kiss.core.models.gemini_model import GeminiModel

        model = GeminiModel.__new__(GeminiModel)
        model.conversation = []
        model.usage_info_for_messages = ""
        model._thought_signatures = {"stale-key": b"stale-value"}

        model.reset_conversation()

        assert model._thought_signatures == {}, (
            f"reset_conversation() should clear _thought_signatures, "
            f"but it still contains: {model._thought_signatures}"
        )








class TestI2FindChannelReturnsName:
    def test_find_channel_does_actual_lookup(self) -> None:
        """find_channel should look up channel by name, not echo it back.

        The bug: it returns the name as-is, which is a string like
        'general', not a Discord snowflake ID.
        """
        from kiss.agents.third_party_agents.discord_agent import DiscordChannelBackend

        backend = DiscordChannelBackend()
        result = backend.find_channel("general")
        assert result != "general" or result is None, (
            f"find_channel('general') returned '{result}' — the name echoed "
            f"back as a channel ID. Should do actual channel lookup or return None."
        )








class TestI6DocstringReferencesNonExistentParams:
    def test_open_docstring_does_not_reference_args(self) -> None:
        """open() takes no parameters, so its docstring shouldn't list Args."""
        from kiss.docker.docker_manager import DockerManager

        doc = inspect.getdoc(DockerManager.open) or ""
        sig = inspect.signature(DockerManager.open)
        params = [p for p in sig.parameters if p != "self"]

        assert "image_name" not in doc or params, (
            "open() docstring references 'image_name' parameter but "
            f"open() takes no arguments (params={params})"
        )






class TestI10ArtifactDirProxyMissingEqHash:
    def test_artifact_dir_proxy_supports_equality(self) -> None:
        """_ArtifactDirProxy should support == comparison with strings."""
        from kiss.core.config import _ArtifactDirProxy

        proxy = _ArtifactDirProxy()
        path_str = str(proxy)

        assert proxy == path_str, (
            f"_ArtifactDirProxy.__eq__ not implemented: "
            f"proxy == '{path_str}' returned False. "
            f"String comparisons with the proxy silently fail."
        )



