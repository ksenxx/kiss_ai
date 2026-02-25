"""Test suite for web search functionality in useful_tools.py."""

import unittest

from kiss.agents.assistant.useful_tools import search_web


class TestSearchWeb(unittest.TestCase):
    def test_search_web_captcha_query_returns_no_results(self) -> None:
        self.assertEqual(search_web("captcha", max_results=0), "No search results found.")


if __name__ == "__main__":
    unittest.main()
