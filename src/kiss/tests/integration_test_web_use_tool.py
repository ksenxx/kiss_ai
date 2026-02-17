"""Integration test: navigate to a real website, fill a form, and show results."""

import re

from kiss.core.web_use_tool import WebUseTool


def find_id(dom: str, pattern: str) -> int:
    match = re.search(pattern, dom)
    assert match, f"Pattern not found: {pattern}\nDOM:\n{dom}"
    return int(match.group(1))


def main() -> None:
    web = WebUseTool(browser_type="chromium", headless=True)

    try:
        print("=" * 70)
        print("Step 1: Navigate to httpbin.org/forms/post")
        print("=" * 70)
        dom = web.go_to_url("https://httpbin.org/forms/post")
        print(dom)

        print("\n" + "=" * 70)
        print("Step 2: Fill in 'Customer name'")
        print("=" * 70)
        dom = web.type_text(find_id(dom, r'\[(\d+)\].*name="custname"'), "Alice Smith")
        print(dom)

        print("\n" + "=" * 70)
        print("Step 3: Fill in 'Telephone'")
        print("=" * 70)
        dom = web.type_text(find_id(dom, r'\[(\d+)\].*type="tel"'), "555-1234")
        print(dom)

        print("\n" + "=" * 70)
        print("Step 4: Fill in 'E-mail address'")
        print("=" * 70)
        dom = web.type_text(find_id(dom, r'\[(\d+)\].*type="email"'), "alice@example.com")
        print(dom)

        print("\n" + "=" * 70)
        print("Step 5: Select pizza size 'Large' (radio button)")
        print("=" * 70)
        dom = web.click(find_id(dom, r'\[(\d+)\].*value="large"'))
        print(dom)

        print("\n" + "=" * 70)
        print("Step 6: Check topping 'Bacon'")
        print("=" * 70)
        dom = web.click(find_id(dom, r'\[(\d+)\].*value="bacon"'))
        print(dom)

        print("\n" + "=" * 70)
        print("Step 7: Check topping 'Mushroom'")
        print("=" * 70)
        dom = web.click(find_id(dom, r'\[(\d+)\].*value="mushroom"'))
        print(dom)

        print("\n" + "=" * 70)
        print("Step 8: Fill in delivery instructions")
        print("=" * 70)
        dom = web.type_text(
            find_id(dom, r"\[(\d+)\] <textarea"), "Ring the doorbell twice please"
        )
        print(dom)

        print("\n" + "=" * 70)
        print("Step 9: Submit the form")
        print("=" * 70)
        dom = web.click(find_id(dom, r"\[(\d+)\] <button"))
        print(dom)

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
