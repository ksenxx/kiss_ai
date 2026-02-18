"""Integration test: navigate to a real website, fill a form, and show results."""

import re

from kiss.core.web_use_tool import WebUseTool


def find_id(tree: str, pattern: str) -> int:
    match = re.search(pattern, tree)
    assert match, f"Pattern not found: {pattern}\nTree:\n{tree}"
    return int(match.group(1))


def main() -> None:
    web = WebUseTool(browser_type="chromium", headless=True)

    try:
        print("=" * 70)
        print("Step 1: Navigate to httpbin.org/forms/post")
        print("=" * 70)
        tree = web.go_to_url("https://httpbin.org/forms/post")
        print(tree)

        print("\n" + "=" * 70)
        print("Step 2: Fill in 'Customer name'")
        print("=" * 70)
        tree = web.type_text(find_id(tree, r"\[(\d+)\].*textbox.*[Cc]ustomer"), "Alice Smith")
        print(tree)

        print("\n" + "=" * 70)
        print("Step 3: Fill in 'Telephone'")
        print("=" * 70)
        tree = web.type_text(find_id(tree, r"\[(\d+)\].*textbox.*[Tt]el"), "555-1234")
        print(tree)

        print("\n" + "=" * 70)
        print("Step 4: Fill in 'E-mail address'")
        print("=" * 70)
        tree = web.type_text(find_id(tree, r"\[(\d+)\].*textbox.*[Ee].?mail"), "alice@example.com")
        print(tree)

        print("\n" + "=" * 70)
        print("Step 5: Select pizza size 'Large' (radio button)")
        print("=" * 70)
        tree = web.click(find_id(tree, r"\[(\d+)\].*radio.*[Ll]arge"))
        print(tree)

        print("\n" + "=" * 70)
        print("Step 6: Check topping 'Bacon'")
        print("=" * 70)
        tree = web.click(find_id(tree, r"\[(\d+)\].*checkbox.*[Bb]acon"))
        print(tree)

        print("\n" + "=" * 70)
        print("Step 7: Check topping 'Mushroom'")
        print("=" * 70)
        tree = web.click(find_id(tree, r"\[(\d+)\].*checkbox.*[Mm]ushroom"))
        print(tree)

        print("\n" + "=" * 70)
        print("Step 8: Fill in delivery instructions")
        print("=" * 70)
        tree = web.type_text(
            find_id(tree, r"\[(\d+)\].*textbox.*[Dd]elivery"),
            "Ring the doorbell twice please",
        )
        print(tree)

        print("\n" + "=" * 70)
        print("Step 9: Submit the form")
        print("=" * 70)
        tree = web.click(find_id(tree, r"\[(\d+)\].*button.*[Ss]ubmit"))
        print(tree)

        print("\n" + "=" * 70)
        print("Step 10: Screenshot of submission result")
        print("=" * 70)
        print(web.screenshot("form_submission_result.png"))

        print("\n" + "=" * 70)
        print("DONE - Form filled and submitted successfully!")
        print("=" * 70)

    finally:
        web.close()


if __name__ == "__main__":
    main()
