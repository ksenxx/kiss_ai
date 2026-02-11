"""Integration tests for printer-based streaming in KISSAgent.

These tests use REAL API calls -- no mocks.
"""

from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.printer import Printer
from kiss.tests.conftest import (
    requires_gemini_api_key,
)


class CollectorPrinter(Printer):
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.prints: list[tuple[str, dict]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        self.prints.append((type, kwargs))
        return ""

    async def token_callback(self, token: str) -> None:
        self.tokens.append(token)

    def reset(self) -> None:
        self.tokens.clear()
        self.prints.clear()


@requires_gemini_api_key
class TestToolOutputStreaming:
    @pytest.mark.timeout(120)
    def test_tool_error_output_streamed(self):
        printer = CollectorPrinter()
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
                printer=printer,
            )
        except Exception:
            pass
        tool_result_prints = [
            p for p in printer.prints if p[0] == "tool_result"
        ]
        assert len(tool_result_prints) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
