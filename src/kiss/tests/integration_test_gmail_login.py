"""Integration test: log in to Gmail using the new zero-JS-injection WebUseTool."""

import re
import time

from kiss.agents.assistant.web_use_tool import KISS_PROFILE_DIR, WebUseTool


def find_id(tree: str, pattern: str) -> int:
    match = re.search(pattern, tree)
    assert match, f"Pattern not found: {pattern}\nTree:\n{tree}"
    return int(match.group(1))


def main() -> None:
    web = WebUseTool(
        browser_type="chromium",
        headless=False,
        user_data_dir=KISS_PROFILE_DIR,
    )

    try:
        print("=" * 70)
        print("Step 1: Navigate to Gmail")
        print("=" * 70)
        tree = web.go_to_url("https://mail.google.com")
        print(tree)
        print()

        if "Choose an account" in tree:
            print("=" * 70)
            print("Step 2: Select account from account chooser")
            print("=" * 70)
            acct_id = find_id(tree, r"\[(\d+)\].*link.*kissagent1")
            tree = web.click(acct_id)
            print(tree)
            time.sleep(2)
            tree = web.get_page_content()
            print(tree)
            print()
        elif "textbox" in tree and ("Email" in tree or "phone" in tree):
            print("=" * 70)
            print("Step 2: Enter email address")
            print("=" * 70)
            email_id = find_id(tree, r"\[(\d+)\].*textbox")
            tree = web.type_text(
                email_id, "kissagent1@gmail.com", press_enter=True
            )
            print(tree)
            time.sleep(3)
            tree = web.get_page_content()
            print(tree)
            print()

        if "textbox" in tree and ("assword" in tree or "Enter your password" in tree):
            print("=" * 70)
            print("Step 3: Enter password")
            print("=" * 70)
            pw_id = find_id(tree, r"\[(\d+)\].*textbox")
            tree = web.type_text(
                pw_id, "Fot AI Assistant.", press_enter=True
            )
            print(tree)
            time.sleep(5)
            tree = web.get_page_content()
            print(tree)
            print()

        print("=" * 70)
        print("Step 4: Current page state")
        print("=" * 70)
        tree = web.get_page_content()
        print(tree)

        print("\n" + "=" * 70)
        print("Step 5: Screenshot")
        print("=" * 70)
        print(web.screenshot("gmail_login_result.png"))

        print("\n" + "=" * 70)
        url = web._page.url if web._page else "unknown"
        if "mail.google.com" in url and "inbox" in url.lower():
            print("SUCCESS - Logged into Gmail inbox!")
        elif "mail.google.com/mail" in url:
            print("SUCCESS - Reached Gmail!")
        else:
            print(f"Current URL: {url}")
            print("Login may have hit CAPTCHA or other verification.")
        print("=" * 70)

    finally:
        web.close()


if __name__ == "__main__":
    main()
