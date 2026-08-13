# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Live integration tests: cc/* models run the claude CLI agentically.

Mirrors how codex/* models run ``codex exec`` in agentic mode.  Verifies
against the real ``claude`` CLI that:

1. The CLI's native tools (Bash, ...) execute for the model — the run
   spans several assistant messages and the tool activity streams as
   thinking blocks.
2. KISS-level text-based tool calling still works: the model prints the
   ``tool_calls`` JSON as plain text instead of attempting (and failing)
   a native invocation of the framework tool.

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
class TestCCAgenticLive:
    """Live agentic runs against the real claude CLI."""

    @pytest.mark.slow
    def test_native_tool_execution_streams_as_thinking(
        self, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """The CLI executes its own Bash tool; activity streams as thinking."""
        import os

        thinking: list[str] = []
        in_thinking = [False]

        def token_cb(text: str) -> None:
            if in_thinking[0]:
                thinking.append(text)

        def thinking_cb(is_start: bool) -> None:
            in_thinking[0] = is_start

        note = tmp_path / "note.txt"  # type: ignore[operator]
        note.write_text("kiss-agentic-probe\n")
        cwd = os.getcwd()
        os.chdir(tmp_path)  # type: ignore[arg-type]
        try:
            m = ClaudeCodeModel(
                "cc/haiku", token_callback=token_cb, thinking_callback=thinking_cb
            )
            m.initialize(
                "Use your Bash tool to run 'cat note.txt' and report the "
                "exact contents."
            )
            content, response = m.generate()
        finally:
            os.chdir(cwd)

        assert "kiss-agentic-probe" in content
        joined = "".join(thinking)
        assert "cat note.txt" in joined, f"tool use not in thinking: {joined!r}"
        counts = m.extract_input_output_token_counts_from_response(response)
        assert counts[1] > 0, f"no output tokens reported: {counts}"

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
