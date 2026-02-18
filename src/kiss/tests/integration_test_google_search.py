"""Integration test: search Google with a visible browser and summarize results."""

import re
from pathlib import Path
from urllib.parse import quote_plus

from kiss.core.web_use_tool import WebUseTool

BROWSER_PROFILE = str(Path.home() / ".kiss" / "browser_profile")


def main() -> None:
    web = WebUseTool(
        browser_type="chromium",
        headless=False,
        user_data_dir=BROWSER_PROFILE,
    )
    query = "Python programming language"

    try:
        print("=" * 70)
        print("Step 1: Navigate directly to Google search results")
        print("=" * 70)
        url = f"https://www.google.com/search?q={quote_plus(query)}"
        tree = web.go_to_url(url)
        print(tree[:4000])
        print("... (truncated)\n")

        if "/sorry/" in tree:
            import time

            print("Google CAPTCHA detected.")
            print("Waiting 30s -- solve the CAPTCHA in the browser window...")
            time.sleep(30)
            tree = web.get_page_content()
            print(tree[:4000])
            print("... (truncated)\n")

        print("=" * 70)
        print("Step 2: Screenshot of search results")
        print("=" * 70)
        print(web.screenshot("google_search_results.png"))

        print("\n" + "=" * 70)
        print("Step 3: Extract links from accessibility tree")
        print("=" * 70)

        links = re.findall(r"\[(\d+)\] link \"([^\"]+)\"", tree)

        skip_patterns = {
            "google.com", "gstatic.com", "googleapis.com",
            "youtube.com", "accounts.google", "support.google",
            "maps.google", "play.google", "Sign in", "More options",
            "Settings", "Tools", "Images", "Videos", "News", "Shopping",
            "All", "Maps", "Books", "Flights", "Finance",
        }

        results: list[tuple[str, str]] = []
        for elem_id, title in links:
            if any(p in title for p in skip_patterns):
                continue
            if len(title) < 5:
                continue
            results.append((elem_id, title))

        print(f"\nFound {len(results)} result links:\n")
        for i, (elem_id, title) in enumerate(results[:10], 1):
            print(f"  {i}. [{elem_id}] {title}\n")

        if results:
            print("=" * 70)
            print(f"Step 4: Click first result: '{results[0][1]}'")
            print("=" * 70)
            first_id = int(results[0][0])
            tree = web.click(first_id)
            print(tree[:3000])
            print("... (truncated)\n")
            print(web.screenshot("first_result_page.png"))

        print("\n" + "=" * 70)
        print("DONE - Google search completed successfully!")
        print("=" * 70)

    finally:
        web.close()


if __name__ == "__main__":
    main()
