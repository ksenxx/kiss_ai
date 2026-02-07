"""Test suite for web search functionality in useful_tools.py.

These tests make actual network calls to search engines and external websites.
Note: Tests may be skipped if the search provider returns a CAPTCHA or blocks requests.
"""

import unittest

import pytest

from kiss.core.useful_tools import fetch_url, search_web


def _search_is_blocked(result: str) -> bool:
    return "No search results found" in result or "Failed to perform search" in result


class TestFetchPageContent(unittest.TestCase):
    def test_fetch_page_content_success(self) -> None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        result = fetch_url("https://example.com", headers)
        self.assertIn("Example Domain", result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_fetch_page_content_invalid_url(self) -> None:
        result = fetch_url("https://this-domain-does-not-exist-12345.com", {"User-Agent": "Test"})
        self.assertIn("Failed to fetch content", result)

    def test_fetch_page_content_truncation(self) -> None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        result = fetch_url("https://example.com", headers, max_characters=50)
        self.assertLessEqual(len(result), 70)
        self.assertIn("[truncated]", result)


@pytest.mark.timeout(180)
class TestSearchWeb(unittest.TestCase):
    def test_search_web_returns_results(self) -> None:
        result = search_web("Python programming language", max_results=2)
        if _search_is_blocked(result):
            self.skipTest("Search provider returned CAPTCHA or blocked request")
        self.assertIn("Title:", result)
        self.assertIn("URL:", result)
        self.assertIn("Content:", result)

    def test_search_web_max_results_respected(self) -> None:
        result = search_web("machine learning", max_results=1)
        if _search_is_blocked(result):
            self.skipTest("Search provider returned CAPTCHA or blocked request")
        self.assertEqual(result.count("Title:"), 1)

    def test_search_web_fetches_page_content(self) -> None:
        result = search_web("Wikipedia", max_results=1)
        if "No search results" not in result:
            self.assertIn("Content:", result)
            content_start = result.find("Content:") + len("Content:")
            content = result[content_start:].strip()
            self.assertGreater(len(content), 50)

    def test_search_web_with_special_characters(self) -> None:
        result = search_web("C++ programming language", max_results=1)
        if _search_is_blocked(result):
            self.skipTest("Search provider returned CAPTCHA or blocked request")
        self.assertIn("Title:", result)
        self.assertIn("URL:", result)

    def test_search_web_captcha_query_returns_no_results(self) -> None:
        self.assertEqual(search_web("captcha", max_results=0), "No search results found.")


@pytest.mark.timeout(180)
class TestSearchWebIntegration(unittest.TestCase):
    def test_search_returns_valid_urls(self) -> None:
        result = search_web("neural network tutorial", max_results=2)
        if _search_is_blocked(result):
            self.skipTest("Search provider returned CAPTCHA or blocked request")
        url_lines = [line for line in result.split("\n") if line.startswith("URL:")]
        self.assertGreater(len(url_lines), 0)
        for url_line in url_lines:
            url = url_line.replace("URL:", "").strip()
            self.assertTrue(url.startswith("http://") or url.startswith("https://"))

    def test_search_with_unicode_query(self) -> None:
        result = search_web("プログラミング", max_results=1)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
