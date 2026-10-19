# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Live integration test: cc/* models drive KISS tools through the claude CLI.

``ClaudeCodeModel`` runs the CLI with ``--tools ""``: it is a plain model, so
KISS-level text-based tool calling must work, i.e. the model prints the
``tool_calls`` JSON as plain text instead of attempting (and failing) a
native invocation of the framework tool.

Requires the ``claude`` CLI to be installed and authenticated.
"""

import shutil

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel

requires_claude = pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude CLI not installed",
)


@requires_claude
@pytest.mark.live_cli
@pytest.mark.usefixtures("claude_live_billing")
class TestCCAgenticLive:
    """Live runs against the real claude CLI."""

    @pytest.mark.slow
    def test_kiss_tool_calls_emitted_as_text_not_native(self) -> None:
        """KISS framework tools must be called via plain-text tool_calls JSON.

        Regression test: with native tools enabled, the model used to try
        a native invocation of the framework tool, fail with "no such
        tool", and apologize — returning zero parsed tool calls.
        """

        def get_weather(city: str) -> str:
            """Get the current weather for a city.

            Args:
                city: The city name.

            Returns:
                Weather description.
            """
            return f"Sunny in {city}"

        m = ClaudeCodeModel("cc/haiku")
        m.initialize("What is the weather in Paris? Use the get_weather tool.")
        calls, _content, _response = m.generate_and_process_with_tools(
            {"get_weather": get_weather}
        )

        weather_calls = [c for c in calls if c["name"] == "get_weather"]
        assert weather_calls, f"expected a get_weather tool call, got {calls}"
        assert weather_calls[0]["arguments"].get("city", "").lower().startswith("paris")
