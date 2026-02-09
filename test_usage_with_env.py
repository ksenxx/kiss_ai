"""Test Claude Agent SDK with explicit API key in env."""

import asyncio
import os
from claude_agent_sdk import ClaudeAgentOptions, query

async def test_with_env():
    """Test if passing API key via env parameter works."""

    async def simple_prompt():
        yield {"type": "user", "message": {"role": "user", "content": "What is 2+2? Just answer with the number."}}

    # Pass API key explicitly in env parameter
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are a helpful assistant.",
        allowed_tools=[],
        max_budget_usd=1.0,
        env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
    )

    print("Starting query with explicit API key in env...")
    message_count = 0

    try:
        async for message in query(prompt=simple_prompt(), options=options):
            message_count += 1
            msg_type = type(message).__name__
            print(f"\nMessage {message_count}: {msg_type}")

            if hasattr(message, 'usage'):
                usage = message.usage
                if usage:
                    print(f"  usage: {usage}")
                    if isinstance(usage, dict):
                        print(f"    input_tokens: {usage.get('input_tokens', 0)}")
                        print(f"    output_tokens: {usage.get('output_tokens', 0)}")

            if hasattr(message, 'total_cost_usd'):
                print(f"  total_cost_usd: {message.total_cost_usd}")

            if msg_type == "ResultMessage":
                print(f"  result: {message.result}")
                print(f"  is_error: {message.is_error}")

        print(f"\nTotal messages received: {message_count}")

    except Exception as e:
        print(f"\nError: {e}")
        print(f"Messages received before error: {message_count}")

if __name__ == "__main__":
    asyncio.run(test_with_env())
