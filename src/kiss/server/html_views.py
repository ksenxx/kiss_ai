# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""HTML documents a chat page staged for its "Open in Browser" button.

A browser tab cannot be handed a file of the daemon's machine: a remote
page has no ``file://`` access to it and the page's origin serves no
files.  So the page uploads the document its preview shows
(``stageHtmlView``, pictures inlined) under a random token and opens
``/view/<token>/<name>`` in a new tab, which
:meth:`~kiss.server.web_server.RemoteAccessServer._html_view_response`
answers from the :class:`HtmlViews` of the daemon.

The documents are kept in memory only, at most :data:`MAX_DOCUMENTS` of
them and :data:`MAX_BYTES` in total, the oldest dropped first.  The
upload and the tab's request race, so a request that comes first waits
for the upload (:meth:`HtmlViews.get`).
"""

from __future__ import annotations

import asyncio
import re
from functools import partial

TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{16,64}")
"""What a document's token looks like: the page mints 16 random bytes in hex."""
MAX_DOCUMENTS = 8
"""The most documents kept."""
MAX_BYTES = 64 * 1024 * 1024
"""The most UTF-8 bytes the kept documents hold together (one frame carries no more)."""
WAIT_SECONDS = 10.0
"""How long a request waits for a document that is not staged yet."""
MAX_WAITERS = 8
"""The most requests that wait at a time; a further one is answered at once."""
RESPONSE_HEADERS = [
    (
        "Content-Security-Policy",
        "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox "
        "allow-forms allow-modals allow-downloads",
    ),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
]
"""The headers every answer carrying a staged document must have.

``sandbox`` without ``allow-same-origin`` gives the document an opaque
origin: its scripts run, but cannot reach the storage, the cookies or
the WebSocket of the chat page, which the same host serves.  That
isolation is why a document is only ever served under these headers.
"""


class HtmlViews:
    """The documents staged on one daemon, oldest first.

    Every method runs on the daemon's event loop.

    Attributes:
        waiting: The requests waiting for their document right now.
    """

    def __init__(self) -> None:
        self._documents: dict[str, bytes] = {}
        self._staged = asyncio.Condition()
        self.waiting = 0

    async def stage(self, token: str, html: str) -> None:
        """Keep *html* as the document of *token* and wake the requests waiting for it.

        A malformed token is ignored.  The oldest documents are dropped
        until the limits hold again.

        Args:
            token: The page's random name for the document.
            html: The document's text.
        """
        if not TOKEN_RE.fullmatch(token):
            return
        async with self._staged:
            # JSON may carry a lone surrogate, which has no UTF-8 form.
            self._documents[token] = html.encode("utf-8", errors="replace")
            while (
                len(self._documents) > MAX_DOCUMENTS
                or sum(map(len, self._documents.values())) > MAX_BYTES
            ):
                del self._documents[next(iter(self._documents))]
            self._staged.notify_all()

    async def get(self, token: str) -> bytes | None:
        """Return the document staged under *token*, waiting for one still on its way.

        A request for a well-formed token that is not staged waits up to
        :data:`WAIT_SECONDS` for it, unless :data:`MAX_WAITERS` requests
        are waiting already.

        Args:
            token: The token the request names.

        Returns:
            The document's UTF-8 bytes, or ``None`` when there is none.
        """
        async with self._staged:
            if (
                token not in self._documents
                and TOKEN_RE.fullmatch(token)
                and self.waiting < MAX_WAITERS
            ):
                self.waiting += 1
                try:
                    async with asyncio.timeout(WAIT_SECONDS):
                        await self._staged.wait_for(partial(self._documents.__contains__, token))
                except TimeoutError:
                    pass
                finally:
                    self.waiting -= 1
            return self._documents.get(token)
