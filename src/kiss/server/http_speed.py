# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Smaller and cacheable HTTP answers for the web app: gzip, and immutable media.

Every plain HTTP answer of the daemon passes through :func:`speed_up_response`,
the ``process_response`` hook of its websockets listeners
(``RemoteAccessServer._serve_wss``):

* A text body (HTML, JavaScript, CSS, JSON, SVG, WebAssembly) of at least
  :data:`MIN_BYTES` goes out gzip-compressed to a browser that accepts it
  (with ``Vary: Accept-Encoding`` either way).
  The web app's page and assets shrink to about 40 % (7.6 MB of assets become
  about 3 MB), which is what a browser far from the daemon waits for: behind
  a tunnel, or the s10s hub's relay from a phone.  Compressed bodies are kept
  by the sha256 of their bytes (:class:`GzipCache`), so a media file is
  compressed once, not on every page load.
* A ``/media/<file>?v=<hash>`` asset, whose URL changes whenever its bytes do
  (``web_server._media_url``), may be kept by the browser for a year.  The
  page and every other answer stay ``no-store``.  The service worker's
  precache then reuses what the page itself has just downloaded instead of
  fetching every asset a second time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gzip
import hashlib
import threading
from collections import OrderedDict
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

MIN_BYTES = 1024
"""Bodies smaller than this are sent as they are (gzip would save next to nothing)."""
CACHE_BYTES = 64 * 1024 * 1024
"""The most compressed bytes :class:`GzipCache` keeps."""
COMPRESSIBLE_TYPES = (
    "text/",
    "application/javascript",
    "application/json",
    "application/manifest+json",
    "image/svg+xml",
    "application/wasm",
)
IMMUTABLE = "public, max-age=31536000, immutable"
_NO_STORE_FIELDS = frozenset({"cache-control", "pragma", "expires"})


class GzipCache:
    """gzip that remembers its recent results, keyed by the sha256 of the input."""

    def __init__(self) -> None:
        self.size = 0
        self.entries: OrderedDict[bytes, bytes] = OrderedDict()
        self.lock = threading.Lock()

    def compress(self, body: bytes) -> bytes:
        """Return *body* gzip-compressed, from memory when the same bytes were compressed before."""
        key = hashlib.sha256(body).digest()
        with self.lock:
            packed = self.entries.get(key)
            if packed is not None:
                self.entries.move_to_end(key)
                return packed
        packed = gzip.compress(body, compresslevel=6, mtime=0)
        with self.lock:
            if key not in self.entries:
                self.entries[key] = packed
                self.size += len(packed)
                while self.size > CACHE_BYTES:
                    _, dropped = self.entries.popitem(last=False)
                    self.size -= len(dropped)
        return packed


_GZIP = GzipCache()


def accepts_gzip(accept_encoding: str) -> bool:
    """Whether an ``Accept-Encoding`` value admits gzip (``gzip;q=0`` refuses it)."""
    for part in accept_encoding.lower().split(","):
        coding, _, params = part.partition(";")
        if coding.strip() == "gzip":
            weight = params.replace(" ", "").removeprefix("q=")
            try:
                return float(weight or "1") > 0
            except ValueError:
                return True
    return False


def is_versioned_media(target: str) -> bool:
    """Whether a request target is a ``/media/`` asset named with its content hash (``?v=``)."""
    url = urlsplit(target)
    return url.path.startswith("/media/") and bool(parse_qs(url.query).get("v"))


async def speed_up_response(
    connection: ServerConnection, request: Request, response: Response
) -> Response:
    """The ``process_response`` hook: compress and cache-mark a ``200`` answer when that helps.

    Args:
        connection: The connection the answer goes out on (unused).
        request: The request being answered.
        response: The answer :meth:`~RemoteAccessServer._process_request`
            (or the WebSocket handshake) produced.

    Returns:
        The answer to send: *response* itself, or a copy with a gzip body
        and/or long-lived caching.
    """
    if response.status_code != 200:
        return response
    drop: set[str] = set()
    add: list[tuple[str, str]] = []
    body = response.body
    if is_versioned_media(request.path):
        drop |= _NO_STORE_FIELDS
        add.append(("Cache-Control", IMMUTABLE))
    content_type = ", ".join(response.headers.get_all("Content-Type"))
    if (
        len(body) >= MIN_BYTES
        and "Content-Encoding" not in response.headers
        and content_type.startswith(COMPRESSIBLE_TYPES)
    ):
        add.append(("Vary", "Accept-Encoding"))  # on both forms, so no cache mixes them up
        if accepts_gzip(", ".join(request.headers.get_all("Accept-Encoding"))):
            body = await asyncio.to_thread(_GZIP.compress, body)
            drop.add("content-length")
            add += [("Content-Length", str(len(body))), ("Content-Encoding", "gzip")]
    if not add:
        return response
    kept = [(k, v) for k, v in response.headers.raw_items() if k.lower() not in drop]
    return dataclasses.replace(response, headers=Headers(kept + add), body=body)
