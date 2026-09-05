# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ntfy.sh duplicate-URL suppression.

The web server posts the active Cloudflare tunnel URL to ntfy.sh so
remote subscribers can rediscover the URL after a restart.  When a
watchdog restart or named-tunnel re-registration produces the *same*
public hostname, re-publishing the URL would needlessly wake every
subscriber — but only for a while: a same-URL message older than
:data:`_NTFY_REPOST_MAX_AGE` must be reposted so a daemon restart
bumps the URL back to the top of the subscriber's ntfy feed and
restarts ntfy.sh's 12h message-cache clock.  These
tests run a real local HTTP server that emulates ntfy.sh's poll +
publish endpoints and verify that :func:`_post_url_to_message_board`
consults the topic's most recent cached message and skips POSTs only
when the URL is unchanged *and* fresh.
"""

from __future__ import annotations

import time
import unittest

from kiss.server.web_server import (
    _NTFY_REPOST_MAX_AGE,
    _fetch_last_ntfy_message,
    _post_url_to_message_board,
)
from kiss.tests.server._ntfy_emulator import NtfyServerContext


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

    def test_fetch_returns_message_and_publish_time(self) -> None:
        """The poll result carries the message's epoch publish time."""
        url = "https://red-fox-1234.trycloudflare.com"
        before = time.time()
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        after = time.time()
        topic = self.ntfy.posts[0][0]
        latest = _fetch_last_ntfy_message(topic, base_url=self.ntfy.base_url)
        assert latest is not None
        self.assertEqual(latest[0], url)
        # The emulator truncates to whole seconds, hence the -1 margin.
        self.assertGreaterEqual(latest[1], int(before) - 1)
        self.assertLessEqual(latest[1], after)

    def test_fresh_duplicate_post_is_skipped(self) -> None:
        """Reposting the same URL within the max age must not hit ntfy.sh."""
        url = "https://red-fox-1234.trycloudflare.com"
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        topic = self.ntfy.posts[0][0]
        latest = _fetch_last_ntfy_message(topic, base_url=self.ntfy.base_url)
        assert latest is not None
        self.assertEqual(latest[0], url)

    def test_stale_same_url_is_reposted(self) -> None:
        """A same-URL message older than the max age is posted again.

        This is the daemon-restart case: the tunnel URL is unchanged
        (adopted cloudflared), but the ntfy message is hours old and
        buried in the subscriber's feed (and nearing ntfy.sh's 12h
        cache expiry), so the URL must be bumped back to the top.
        """
        url = "https://red-fox-1234.trycloudflare.com"
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        topic = self.ntfy.posts[0][0]
        body, _posted_at = self.ntfy.messages[topic][0]
        stale = time.time() - _NTFY_REPOST_MAX_AGE - 10
        self.ntfy.messages[topic][0] = (body, stale)
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 2)
        self.assertEqual(self.ntfy.posts[1][1], url)

    def test_same_url_without_time_field_is_reposted(self) -> None:
        """A cached message lacking a ``time`` field counts as stale.

        If the ntfy server ever omits the publish time, the message's
        age is unknown, so discoverability wins over dedup and the
        URL is posted again.
        """
        url = "https://red-fox-1234.trycloudflare.com"
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        topic = self.ntfy.posts[0][0]
        self.ntfy.messages[topic][0] = (url, None)
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 2)

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
        assert latest is not None
        self.assertEqual(latest[0], second)

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
