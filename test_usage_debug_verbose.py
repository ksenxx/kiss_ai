"""Debug script with verbose output to diagnose the SDK API key issue."""

import asyncio
import os
import sys
from claude_agent_sdk import ClaudeAgentOptions, query

async def test_with_debug():
    """Test with debug output enabled."""

    async def simple_prompt():
        yield {"type": "user", "message": {"role": "user", "content": "What is 2+2?"}}

    # Enable debug output by passing stderr handler
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are a helpful assistant.",
        allowed_tools=[],
        max_budget_usd=1.0,
        env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
        debug_stderr=sys.stderr,  # Enable debug output
    )

    print("Starting query with debug enabled...", file=sys.stderr)
    print(f"API key length: {len(os.environ.get('ANTHROPIC_API_KEY', ''))}", file=sys.stderr)

    try:
        async for message in query(prompt=simple_prompt(), options=options):
            msg_type = type(message).__name__
            print(f"Message: {msg_type}", file=sys.stderr)

            if msg_type == "ResultMessage":
                print(f"Result: {message.result}")
                print(f"Is error: {message.is_error}")
                if hasattr(message, 'usage') and message.usage:
                    print(f"Usage: {message.usage}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(test_with_debug())
