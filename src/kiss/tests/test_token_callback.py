"""Integration tests for the async token_callback streaming feature.

These tests use REAL API calls -- no mocks.
"""

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import TokenCallback
from kiss.tests.conftest import (
    requires_gemini_api_key,
)


def _make_collector() -> tuple[TokenCallback, list[str]]:
    tokens: list[str] = []

    async def _callback(token: str) -> None:
        tokens.append(token)

    return _callback, tokens


@requires_gemini_api_key
class TestToolOutputStreaming:
    @pytest.mark.timeout(120)
    def test_tool_error_output_streamed(self):
        callback, tokens = _make_collector()
        agent = KISSAgent("test-tool-error-stream")

        def failing_tool(x: str) -> str:
            """A tool that always fails.

            Args:
                x: Any input string.

            Returns:
                Never returns successfully.
            """
            raise ValueError("intentional test failure")

        try:
            agent.run(
                model_name="gemini-2.0-flash",
                prompt_template="Call the failing_tool with x='test'.",
                tools=[failing_tool],
                is_agentic=True,
                max_steps=3,
                token_callback=callback,
            )
        except Exception:
            pass
        assert "intentional test failure" in "".join(tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
