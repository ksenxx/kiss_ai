# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Brave Search Agent — channel agent for the Brave Search REST API.

Provides web, news, image, and video search through the Brave Search
API (https://api.search.brave.com/res/v1) using a subscription token
sent as ``X-Subscription-Token`` on every call.  Stores config in
``~/.kiss/third_party_agents/brave_search/config.json``.

The Brave Search API has no inbound message stream, so this adapter is
outbound-only and the ``--channel`` poll mode is disabled (``main``
passes ``make_backend=None`` to ``channel_main``).

Usage::

    agent = BraveSearchAgent()
    agent.run(prompt_template="Find recent news about quantum computing")
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_DEFAULT_BASE_URL = "https://api.search.brave.com/res/v1"
_MAX_OUTPUT_CHARS = 8000

_BRAVE_SEARCH_DIR = Path.home() / ".kiss" / "third_party_agents" / "brave_search"
_config = ChannelConfig(_BRAVE_SEARCH_DIR, ("api_key",))


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp *value* into the inclusive range [*low*, *high*].

    Args:
        value: The candidate value.
        low: Lowest allowed value.
        high: Highest allowed value.

    Returns:
        *value* limited to the given range.
    """
    return max(low, min(high, value))


class BraveSearchChannelBackend(ToolMethodBackend):
    """Channel backend for the Brave Search REST API.

    Talks to the Brave Search API over HTTPS with a subscription
    token.  Outbound-only: there is no inbound message stream, so poll
    mode is disabled entirely.
    """

    def __init__(self) -> None:
        self._base_url: str = _DEFAULT_BASE_URL
        self._api_key: str = ""
        self._http: Any = requests
        self._muse: bool = False
        self._request_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the Brave Search config from disk.

        In Muse-auth mode (``KISS_MUSE_AUTH=1``) the real subscription
        token lives in the Muse vault as a header-kind credential
        (auto-enrolled from the legacy config on first connect); this
        process only holds a surrogate, sent as a bearer to the daemon,
        which swaps it into the real ``X-Subscription-Token`` header at
        the network boundary.

        Returns:
            True if a valid config with ``api_key`` was loaded.
        """
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import (
                MuseBoundarySession,
                mint_surrogate,
            )

            handle = mint_surrogate("brave_search")
            if handle is None:
                # Vault-first: the legacy config key is only read when
                # the vault has no enrollment yet (one-time migration).
                from kiss.agents.third_party_agents.muse_auth.client import bearer_surrogate

                cfg = _config.load()
                surrogate = bearer_surrogate(
                    "brave_search",
                    (cfg or {}).get("api_key", ""),
                    header="X-Subscription-Token",
                )
            else:
                surrogate = handle.token
            if not surrogate:
                self._connection_info = "No Brave Search credential in the Muse vault or config."
                return False
            self._api_key = surrogate
            self._http = MuseBoundarySession("brave_search")
            self._muse = True
            self._connection_info = "Brave Search API key configured (Muse-auth)."
            return True
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No Brave Search config found."
            return False
        self._api_key = cfg["api_key"]
        self._connection_info = "Brave Search API key configured."
        return True

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        """Issue an authenticated GET request to the Brave Search API.

        Args:
            path: API path such as ``/web/search``.
            params: Query parameters (must include ``q``).

        Returns:
            The parsed JSON response body.

        Raises:
            RuntimeError: On an HTTP error status.
            requests.RequestException: On a transport failure.
        """
        url = self._base_url.rstrip("/") + path
        headers = {"Accept": "application/json"}
        if self._muse:
            # The surrogate travels as a bearer; the daemon swaps it
            # into the real X-Subscription-Token at the boundary.
            headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            headers["X-Subscription-Token"] = self._api_key
        with self._request_lock:
            resp = self._http.get(url, headers=headers, params=params, timeout=_TIMEOUT)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def brave_web_search(
        self,
        query: str,
        count: int = 10,
        offset: int = 0,
        country: str = "",
        search_lang: str = "",
        freshness: str = "",
    ) -> str:
        """Search the web with Brave Search.

        Args:
            query: The search query.  Supports operators such as
                ``site:example.com`` and quoted exact phrases.
            count: Maximum results to return (clamped to 1-20).
            offset: Zero-based results page (clamped to 0-9).
            country: Optional 2-letter country code (e.g. ``"US"``) to
                target results from a specific country.
            search_lang: Optional content language filter (e.g. ``"en"``).
            freshness: Optional recency filter: ``"pd"`` (day), ``"pw"``
                (week), ``"pm"`` (month), ``"py"`` (year), or a range
                like ``"2024-01-01to2024-06-30"``.

        Returns:
            JSON string ``{"ok": true, "results": [{title, url,
            description, age}, ...]}`` plus ``infobox``/``faq`` entries
            when present, or ``{"ok": false, "error": ...}``.
        """
        try:
            if not query.strip():
                return json.dumps({"ok": False, "error": "query cannot be empty"})
            params: dict[str, Any] = {"q": query, "count": _clamp(count, 1, 20)}
            if offset:
                params["offset"] = _clamp(offset, 0, 9)
            if country:
                params["country"] = country
            if search_lang:
                params["search_lang"] = search_lang
            if freshness:
                params["freshness"] = freshness
            data = self._get("/web/search", params)
            out: dict[str, Any] = {
                "ok": True,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "description": r.get("description", ""),
                        "age": r.get("age", ""),
                    }
                    for r in data.get("web", {}).get("results", [])
                ],
            }
            infobox = [
                {
                    "title": r.get("title", ""),
                    "description": r.get("long_desc") or r.get("description", ""),
                    "url": r.get("url", ""),
                }
                for r in data.get("infobox", {}).get("results", [])
            ]
            if infobox:
                out["infobox"] = infobox
            faq = [
                {
                    "question": r.get("question", ""),
                    "answer": r.get("answer", ""),
                    "url": r.get("url", ""),
                }
                for r in data.get("faq", {}).get("results", [])
            ]
            if faq:
                out["faq"] = faq
            return json.dumps(out)[:_MAX_OUTPUT_CHARS]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def brave_news_search(self, query: str, count: int = 10, freshness: str = "") -> str:
        """Search recent news articles with Brave Search.

        Args:
            query: The news search query.
            count: Maximum results to return (clamped to 1-50).
            freshness: Optional recency filter: ``"pd"`` (day), ``"pw"``
                (week), ``"pm"`` (month), ``"py"`` (year), or a range
                like ``"2024-01-01to2024-06-30"``.

        Returns:
            JSON string ``{"ok": true, "results": [{title, url,
            description, age, source}, ...]}`` or
            ``{"ok": false, "error": ...}``.
        """
        try:
            if not query.strip():
                return json.dumps({"ok": False, "error": "query cannot be empty"})
            params: dict[str, Any] = {"q": query, "count": _clamp(count, 1, 50)}
            if freshness:
                params["freshness"] = freshness
            data = self._get("/news/search", params)
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                    "age": r.get("age", ""),
                    "source": r.get("meta_url", {}).get("hostname", ""),
                }
                for r in data.get("results", [])
            ]
            return json.dumps({"ok": True, "results": results})[:_MAX_OUTPUT_CHARS]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def brave_image_search(self, query: str, count: int = 10) -> str:
        """Search images with Brave Search.

        Args:
            query: The image search query.
            count: Maximum results to return (clamped to 1-200).

        Returns:
            JSON string ``{"ok": true, "results": [{title, page_url,
            image_url, thumbnail, source}, ...]}`` or
            ``{"ok": false, "error": ...}``.
        """
        try:
            if not query.strip():
                return json.dumps({"ok": False, "error": "query cannot be empty"})
            params: dict[str, Any] = {"q": query, "count": _clamp(count, 1, 200)}
            data = self._get("/images/search", params)
            results = [
                {
                    "title": r.get("title", ""),
                    "page_url": r.get("url", ""),
                    "image_url": r.get("properties", {}).get("url", ""),
                    "thumbnail": r.get("thumbnail", {}).get("src", ""),
                    "source": r.get("source", ""),
                }
                for r in data.get("results", [])
            ]
            return json.dumps({"ok": True, "results": results})[:_MAX_OUTPUT_CHARS]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def brave_video_search(self, query: str, count: int = 10) -> str:
        """Search videos with Brave Search.

        Args:
            query: The video search query.
            count: Maximum results to return (clamped to 1-50).

        Returns:
            JSON string ``{"ok": true, "results": [{title, url,
            description, age, duration, creator}, ...]}`` or
            ``{"ok": false, "error": ...}``.
        """
        try:
            if not query.strip():
                return json.dumps({"ok": False, "error": "query cannot be empty"})
            params: dict[str, Any] = {"q": query, "count": _clamp(count, 1, 50)}
            data = self._get("/videos/search", params)
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                    "age": r.get("age", ""),
                    "duration": r.get("video", {}).get("duration", ""),
                    "creator": r.get("video", {}).get("creator", ""),
                }
                for r in data.get("results", [])
            ]
            return json.dumps({"ok": True, "results": results})[:_MAX_OUTPUT_CHARS]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class BraveSearchAgent(BaseChannelAgent):
    """Channel agent with Brave Search API tools."""

    channel_system_prompt = (
        "You can search the internet through the Brave Search API. Use "
        "brave_web_search for general web queries (facts, documentation, "
        "sites; supports country/search_lang targeting, freshness filters, "
        "and pagination via offset), brave_news_search for current events "
        "and recent headlines (supports freshness filters), "
        "brave_image_search to find images (returns page, image, and "
        "thumbnail URLs), and brave_video_search to find videos (returns "
        "duration and creator). Prefer brave_news_search over "
        "brave_web_search when recency matters most."
    )

    def __init__(self) -> None:
        super().__init__("Brave Search Agent")
        self._backend = BraveSearchChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            # Muse-auth mode: connect() wires a vault surrogate and the
            # boundary session; the real key never enters this process
            # once migrated.
            self._backend.connect()
            return
        cfg = _config.load()
        if cfg:
            self._backend._api_key = cfg["api_key"]

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._api_key)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_brave_search_auth() -> str:
            """Check if Brave Search is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._is_authenticated():
                return (
                    "Not configured for Brave Search. Use "
                    "authenticate_brave_search(api_key) to configure.\n"
                    "Get an API key at https://api-dashboard.search.brave.com/ "
                    "(create an account, pick a plan — a free tier exists — "
                    "and generate a subscription token)."
                )
            return json.dumps({"ok": True, "message": "Brave Search API key is configured."})

        def authenticate_brave_search(api_key: str) -> str:
            """Configure the Brave Search API subscription token.

            Args:
                api_key: Subscription token from
                    https://api-dashboard.search.brave.com/.

            Returns:
                Configuration result or error message.
            """
            if not api_key.strip():
                return "api_key cannot be empty."
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            try:
                _config.save({"api_key": api_key.strip()})
                if muse_auth_enabled():
                    # Re-enroll straight into the Muse vault: clear any
                    # existing entry first so a rotated key replaces
                    # the old one (connect() is vault-first and would
                    # otherwise keep minting the stale credential).
                    from kiss.agents.third_party_agents.muse_auth.client import (
                        clear_credentials,
                    )

                    clear_credentials("brave_search")
                    agent._backend.connect()
                else:
                    agent._backend._api_key = api_key.strip()
            except Exception as e:
                return json.dumps({"ok": False, "error": f"could not save config: {e}"})
            return json.dumps({"ok": True, "message": "Brave Search configured."})

        def clear_brave_search_auth() -> str:
            """Clear the stored Brave Search configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._api_key = ""
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("brave_search")
            return "Brave Search configuration cleared."

        return [check_brave_search_auth, authenticate_brave_search, clear_brave_search_auth]


def main() -> None:
    """Run the BraveSearchAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): the Brave Search API
    has no inbound message stream to poll.
    """
    channel_main(
        BraveSearchAgent,
        "kiss-brave",
        channel_name="Brave Search",
        make_backend=None,
    )


def tools() -> list:
    """Return the Brave Search channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return BraveSearchAgent()._get_tools()


if __name__ == "__main__":
    main()
