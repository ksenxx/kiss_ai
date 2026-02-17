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
        dom = web.go_to_url(url)
        print(dom[:4000])
        print("... (truncated)\n")

        if "/sorry/" in dom:
            import time

            print("Google CAPTCHA detected.")
            print("Waiting 30s -- solve the CAPTCHA in the browser window...")
            time.sleep(30)
            dom = web.get_page_content()
            print(dom[:4000])
            print("... (truncated)\n")

        print("=" * 70)
        print("Step 2: Screenshot of search results")
        print("=" * 70)
        print(web.screenshot("google_search_results.png"))

        print("\n" + "=" * 70)
        print("Step 3: Extract and summarize non-sponsored links")
        print("=" * 70)

        links = re.findall(
            r'\[(\d+)\] <a href="(https?://[^"]+)"[^>]*>([^<]{5,})', dom
        )

        skip_domains = {
            "google.com", "google.", "gstatic.com", "googleapis.com",
            "youtube.com", "accounts.google", "support.google",
            "maps.google", "play.google",
        }

        seen: set[str] = set()
        results: list[tuple[str, str, str]] = []

        for elem_id, link_url, title in links:
            title = title.strip()
            if any(d in link_url for d in skip_domains):
                continue
            if link_url in seen:
                continue
            seen.add(link_url)
            results.append((elem_id, link_url, title))

        print(f"\nFound {len(results)} non-sponsored result links:\n")
        for i, (elem_id, link_url, title) in enumerate(results[:10], 1):
            print(f"  {i}. [{elem_id}] {title}")
            print(f"     {link_url}\n")

        if results:
            print("=" * 70)
            print(f"Step 4: Click first result: '{results[0][2]}'")
            print("=" * 70)
            first_id = int(results[0][0])
            dom = web.click(first_id)
            print(dom[:3000])
            print("... (truncated)\n")
            print(web.screenshot("first_result_page.png"))

        print("\n" + "=" * 70)
        print("DONE - Google search completed successfully!")
        print("=" * 70)

    finally:
        web.close()


if __name__ == "__main__":
    main()
