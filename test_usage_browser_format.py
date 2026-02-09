"""Test that usage info is properly formatted for browser display."""

from unittest.mock import Mock, patch, call
from kiss.agents.coding_agents.print_to_browser import BrowserPrinter


def test_usage_info_formatting():
    """Test that markdown usage info is converted to structured format."""
    print(f"\n{'='*80}")
    print("TESTING USAGE INFO BROWSER FORMATTING")
    print(f"{'='*80}")

    printer = BrowserPrinter()

    # Mock the broadcast method to capture events
    broadcast_calls = []

    def mock_broadcast(event):
        broadcast_calls.append(event)

    printer._broadcast = mock_broadcast

    # Create markdown usage info (same format as _get_usage_info_string)
    usage_markdown = """#### Usage Information
- Token usage: 1500/200000
- Agent budget: $0.0105 / $10.00
- Global budget: $0.0221 / $200.00
- Step: 10/10
"""

    # Print the usage info
    printer.print_usage_info(usage_markdown)

    # Check that broadcast was called
    assert len(broadcast_calls) == 1, "Should broadcast exactly once"

    event = broadcast_calls[0]
    print(f"\nBroadcast event:")
    print(f"  Type: {event.get('type')}")
    print(f"  Items: {event.get('items')}")

    # Verify the event structure
    assert event["type"] == "usage_info", "Event type should be 'usage_info'"
    assert "items" in event, "Event should contain 'items' key"
    assert isinstance(event["items"], list), "Items should be a list"

    # Verify the items content
    items = event["items"]
    expected_items = [
        "Token usage: 1500/200000",
        "Agent budget: $0.0105 / $10.00",
        "Global budget: $0.0221 / $200.00",
        "Step: 10/10"
    ]

    print(f"\nExpected items: {len(expected_items)}")
    print(f"Actual items: {len(items)}")

    for i, (expected, actual) in enumerate(zip(expected_items, items), 1):
        print(f"{i}. Expected: {expected}")
        print(f"   Actual:   {actual}")
        assert actual == expected, f"Item {i} mismatch"

    assert len(items) == len(expected_items), \
        f"Expected {len(expected_items)} items, got {len(items)}"

    print(f"\n✅ SUCCESS: Usage info properly formatted for browser")
    return True


def test_usage_info_with_different_values():
    """Test formatting with different usage values."""
    print(f"\n{'='*80}")
    print("TESTING DIFFERENT USAGE VALUES")
    print(f"{'='*80}")

    printer = BrowserPrinter()
    broadcast_calls = []
    printer._broadcast = lambda e: broadcast_calls.append(e)

    test_cases = [
        {
            "input": """#### Usage Information
- Token usage: 0/100000
- Agent budget: $0.0000 / $1.00
- Global budget: $0.0000 / $200.00
- Step: 1/5
""",
            "expected_count": 4,
            "description": "First step"
        },
        {
            "input": """#### Usage Information
- Token usage: 99999/100000
- Agent budget: $9.9999 / $10.00
- Global budget: $199.9999 / $200.00
- Step: 999/1000
""",
            "expected_count": 4,
            "description": "Near limits"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        broadcast_calls.clear()
        printer.print_usage_info(case["input"])

        event = broadcast_calls[0]
        items = event.get("items", [])

        print(f"\nTest case {i}: {case['description']}")
        print(f"  Items count: {len(items)}")
        print(f"  Expected: {case['expected_count']}")

        assert len(items) == case["expected_count"], \
            f"Test case {i}: Expected {case['expected_count']} items, got {len(items)}"

        for j, item in enumerate(items, 1):
            print(f"  {j}. {item}")

        print(f"  ✅ Passed")

    print(f"\n✅ All test cases passed!")
    return True


def test_html_rendering_format():
    """Test that the format is suitable for HTML rendering."""
    print(f"\n{'='*80}")
    print("TESTING HTML RENDERING FORMAT")
    print(f"{'='*80}")

    printer = BrowserPrinter()
    broadcast_calls = []
    printer._broadcast = lambda e: broadcast_calls.append(e)

    usage_info = """#### Usage Information
- Token usage: 500/100000
- Agent budget: $0.0050 / $10.00
- Global budget: $0.0050 / $200.00
- Step: 5/10
"""

    printer.print_usage_info(usage_info)

    event = broadcast_calls[0]
    items = event["items"]

    # Simulate HTML rendering (as done in JavaScript)
    html_output = '<strong>Usage Information</strong><br>' + \
                  '<br>'.join(f'• {item}' for item in items)

    print(f"\nSimulated HTML output:")
    print(html_output)

    # Verify no markdown remains
    for item in items:
        assert not item.startswith('- '), "Items should not start with markdown bullets"
        assert not item.startswith('#'), "Items should not contain markdown headers"

    print(f"\n✅ Format is clean and ready for HTML rendering")
    return True


if __name__ == "__main__":
    try:
        test_usage_info_formatting()
        print()
        test_usage_info_with_different_values()
        print()
        test_html_rendering_format()

        print(f"\n{'='*80}")
        print("ALL TESTS PASSED ✅")
        print(f"{'='*80}")
        print("\nSummary:")
        print("- Markdown converted to structured items")
        print("- Items properly extracted without markdown syntax")
        print("- Format suitable for HTML rendering")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
