# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ntfy.sh duplicate-URL suppression.

The web server posts the active Cloudflare tunnel URL to ntfy.sh so
remote subscribers can rediscover the URL after a restart.  When a
watchdog restart or named-tunnel re-registration produces the *same*
public hostname, re-publishing the URL would needlessly wake every
subscriber.  These tests run a real local HTTP server that emulates
ntfy.sh's poll + publish endpoints and verify that
:func:`_post_url_to_message_board` consults the topic's most recent
cached message and skips POSTs when the URL is unchanged.
"""

from __future__ import annotations

import unittest

from kiss.server.web_server import (
    _fetch_last_ntfy_message,
    _post_url_to_message_board,
)
from kiss.tests.agents.vscode._ntfy_emulator import NtfyServerContext


class TestNtfyDeduplication(unittest.TestCase):
    """End-to-end verification of duplicate-URL suppression."""

    def setUp(self) -> None:
        self.ntfy = NtfyServerContext()

    def tearDown(self) -> None:
        self.ntfy.stop()

    def test_fetch_returns_none_for_empty_topic(self) -> None:
        """An empty topic has no cached messages."""
        result = _fetch_last_ntfy_message(
            "empty-topic", base_url=self.ntfy.base_url,
        )
        self.assertIsNone(result)

    def test_first_post_succeeds_when_topic_empty(self) -> None:
        """No prior post means the URL must be published."""
        url = "https://red-fox-1234.trycloudflare.com"
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        topic, body, _headers = self.ntfy.posts[0]
        self.assertTrue(topic.startswith("kiss-"))
        self.assertEqual(body, url)

    def test_duplicate_post_is_skipped(self) -> None:
        """Reposting the same URL must not hit ntfy.sh."""
        url = "https://red-fox-1234.trycloudflare.com"
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        topic = self.ntfy.posts[0][0]
        latest = _fetch_last_ntfy_message(topic, base_url=self.ntfy.base_url)
        self.assertEqual(latest, url)

    def test_different_url_is_posted(self) -> None:
        """A changed URL must be published even after a prior post."""
        first = "https://red-fox-1234.trycloudflare.com"
        second = "https://blue-bear-5678.trycloudflare.com"
        _post_url_to_message_board(first, base_url=self.ntfy.base_url)
        _post_url_to_message_board(second, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 2)
        self.assertEqual(self.ntfy.posts[0][1], first)
        self.assertEqual(self.ntfy.posts[1][1], second)
        topic = self.ntfy.posts[0][0]
        latest = _fetch_last_ntfy_message(topic, base_url=self.ntfy.base_url)
        self.assertEqual(latest, second)

    def test_localhost_url_never_posted(self) -> None:
        """``https://localhost...`` URLs are not meant for ntfy."""
        _post_url_to_message_board(
            "https://localhost:8787", base_url=self.ntfy.base_url,
        )
        self.assertEqual(len(self.ntfy.posts), 0)

    def test_empty_url_never_posted(self) -> None:
        """Empty URLs are silently ignored."""
        _post_url_to_message_board("", base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 0)

    def test_click_header_is_set_to_url(self) -> None:
        """The ``Click`` header makes the ntfy notification clickable.

        Without this header, tapping the message in the ntfy.sh web UI
        or the mobile app does nothing because the URL in the body is
        rendered as plain text.  Per
        https://docs.ntfy.sh/publish/#click-action, the ``Click``
        header is the only supported way to attach a navigation
        target to a message, so it must equal the URL we published.
        """
        url = "https://red-fox-1234.trycloudflare.com"
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        _topic, body, headers = self.ntfy.posts[0]
        self.assertEqual(body, url)
        click = next(
            (v for k, v in headers.items() if k.lower() == "click"),
            None,
        )
        self.assertEqual(click, url)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
