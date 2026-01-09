# Author: Koushik Sen (ksen@berkeley.edu)

"""Test suite for web search functionality in utils.py.

These tests make actual network calls to search engines and external websites.
Uses Playwright headless browser to render JavaScript-heavy pages.

Note: These tests may be skipped if the search provider (Startpage) returns
a CAPTCHA or blocks automated requests. This is expected behavior for
web scraping tests that depend on external services.
"""

import unittest

import pytest

from kiss.core.utils import _fetch_page_content, search_web


def _search_is_blocked(result: str) -> bool:
    """Check if search result indicates CAPTCHA or blocking."""
    return "No search results found" in result or "Failed to perform search" in result


class TestFetchPageContent(unittest.TestCase):
    """Tests for _fetch_page_content helper function."""

    def test_fetch_page_content_success(self) -> None:
        """Test successful page content fetching from a real website."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        result = _fetch_page_content("https://example.com", headers)

        # example.com has minimal content
        self.assertIn("Example Domain", result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_fetch_page_content_invalid_url(self) -> None:
        """Test handling of invalid/unreachable URL."""
        headers = {"User-Agent": "Test Agent"}
        result = _fetch_page_content("https://this-domain-does-not-exist-12345.com", headers)

        self.assertIn("Failed to fetch content", result)

    def test_fetch_page_content_truncation(self) -> None:
        """Test content truncation for pages."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        # Fetch with very small max length to test truncation
        result = _fetch_page_content("https://example.com", headers, max_content_length=50)

        self.assertLessEqual(len(result), 70)  # 50 + "... [truncated]"
        self.assertIn("[truncated]", result)


@pytest.mark.timeout(180)
class TestSearchWeb(unittest.TestCase):
    """Tests for search_web function with real network calls using Playwright."""

    def test_search_web_returns_results(self) -> None:
        """Test that search_web returns actual search results."""
        result = search_web("Python programming language", max_results=2)

        if _search_is_blocked(result):
            self.skipTest("Search provider returned CAPTCHA or blocked request")

        # Should return results with titles, URLs, and content
        self.assertIn("Title:", result)
        self.assertIn("URL:", result)
        self.assertIn("Content:", result)
        self.assertIn("http", result)

    def test_search_web_max_results_respected(self) -> None:
        """Test that max_results parameter limits the number of results."""
        result = search_web("machine learning", max_results=1)

        if _search_is_blocked(result):
            self.skipTest("Search provider returned CAPTCHA or blocked request")

        # Count the number of "Title:" occurrences
        title_count = result.count("Title:")
        self.assertEqual(title_count, 1)

    def test_search_web_fetches_page_content(self) -> None:
        """Test that search_web fetches actual page content."""
        result = search_web("Wikipedia", max_results=1)

        # Should have fetched some content (unless rate limited)
        if "No search results" not in result:
            self.assertIn("Content:", result)
            # Content should be more than just a placeholder
            content_start = result.find("Content:") + len("Content:")
            content = result[content_start:].strip()
            self.assertGreater(len(content), 50)

    def test_search_web_with_special_characters(self) -> None:
        """Test that search handles special characters in query."""
        result = search_web("C++ programming language", max_results=1)

        if _search_is_blocked(result):
            self.skipTest("Search provider returned CAPTCHA or blocked request")

        self.assertIn("Title:", result)
        self.assertIn("URL:", result)


@pytest.mark.timeout(180)
class TestSearchWebIntegration(unittest.TestCase):
    """Integration tests for search_web with various queries."""

    def test_search_technical_query(self) -> None:
        """Test search with a technical query."""
        result = search_web("BeautifulSoup Python documentation", max_results=2)

        if _search_is_blocked(result):
            self.skipTest("Search provider returned CAPTCHA or blocked request")

        self.assertIn("Title:", result)
        self.assertIn("URL:", result)
        self.assertIn("Content:", result)

    def test_search_returns_valid_urls(self) -> None:
        """Test that search returns valid HTTP URLs."""
        result = search_web("neural network tutorial", max_results=2)

        if _search_is_blocked(result):
            self.skipTest("Search provider returned CAPTCHA or blocked request")

        # Extract URLs from result
        lines = result.split("\n")
        url_lines = [line for line in lines if line.startswith("URL:")]

        self.assertGreater(len(url_lines), 0)
        for url_line in url_lines:
            url = url_line.replace("URL:", "").strip()
            self.assertTrue(url.startswith("http://") or url.startswith("https://"))

    def test_search_with_unicode_query(self) -> None:
        """Test that unicode queries are handled."""
        result = search_web("プログラミング", max_results=1)  # "programming" in Japanese

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
