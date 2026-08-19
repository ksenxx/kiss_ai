# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests that reproduce bugs listed in bugs.md.

Each test demonstrates the buggy behavior. All tests should FAIL
until the corresponding bug is fixed. No mocks, patches, fakes,
or test doubles are used.

The bug I6 test, which depends only on ``kiss.agents.sorcar``, lives
in ``kiss.tests.agents.sorcar.test_bugs_reproduction``.
"""


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
