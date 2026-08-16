# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests that reproduce bugs listed in bugs.md.

Each test demonstrates the buggy behavior. All tests should FAIL
until the corresponding bug is fixed. No mocks, patches, fakes,
or test doubles are used.
"""

import inspect


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
        from kiss.agents.sorcar.docker_manager import DockerManager

        doc = inspect.getdoc(DockerManager.open) or ""
        sig = inspect.signature(DockerManager.open)
        params = [p for p in sig.parameters if p != "self"]

        assert "image_name" not in doc or params, (
            "open() docstring references 'image_name' parameter but "
            f"open() takes no arguments (params={params})"
        )
