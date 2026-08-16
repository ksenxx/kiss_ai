# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests that reproduce bugs listed in bugs.md.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.test_bugs_reproduction``; the non-core tests remain there.
"""




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
