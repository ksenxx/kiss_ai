"""Debug script to test Claude Agent SDK usage tracking."""

import asyncio
from claude_agent_sdk import ClaudeAgentOptions, query

async def test_usage():
    """Test if usage information is returned by the SDK."""

    async def simple_prompt():
        yield {"type": "user", "message": {"role": "user", "content": "What is 2+2? Just answer with the number."}}

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are a helpful assistant.",
        allowed_tools=[],  # No tools for simplicity
        max_budget_usd=1.0,
    )

    print("Starting query...")
    message_count = 0

    async for message in query(prompt=simple_prompt(), options=options):
        message_count += 1
        msg_type = type(message).__name__
        print(f"\nMessage {message_count}: {msg_type}")
        print(f"  Attributes: {[attr for attr in dir(message) if not attr.startswith('_')]}")

        # Check for usage attribute
        if hasattr(message, 'usage'):
            print(f"  usage: {message.usage}")
        else:
            print(f"  usage: NOT FOUND")

        # Check for total_cost_usd
        if hasattr(message, 'total_cost_usd'):
            print(f"  total_cost_usd: {message.total_cost_usd}")

        # If it's a ResultMessage, print details
        if msg_type == "ResultMessage":
            print(f"  result: {message.result}")
            print(f"  is_error: {message.is_error}")
            print(f"  num_turns: {message.num_turns}")

if __name__ == "__main__":
    asyncio.run(test_usage())
