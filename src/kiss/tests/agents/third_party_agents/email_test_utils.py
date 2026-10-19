# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""RFC822 message builders shared by the Email channel agent tests."""

from __future__ import annotations


def raw_plain(**overrides: str) -> bytes:
    """Build a plain-text RFC822 message, allowing header overrides.

    Args:
        overrides: Header values replacing the defaults; an empty value drops the header.

    Returns:
        The raw message bytes.
    """
    headers = {
        "From": "Alice Example <alice@example.com>",
        "To": "bot@example.com",
        "Subject": "Hello there",
        "Date": "Mon, 01 Jan 2024 12:00:00 +0000",
        "Message-ID": "<abc123@example.com>",
        "Content-Type": 'text/plain; charset="utf-8"',
    }
    headers.update(overrides)
    lines = [f"{k}: {v}" for k, v in headers.items() if v]
    return ("\r\n".join(lines) + "\r\n\r\nHi bot, please help.\r\n").encode("utf-8")
