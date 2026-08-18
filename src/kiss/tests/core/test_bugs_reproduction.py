# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests that reproduce bugs listed in bugs.md.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.agents.third_party_agents.test_bugs_reproduction``; the non-core tests remain there.
"""



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
