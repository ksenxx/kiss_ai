# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Firecrawl Agent — channel agent for the Firecrawl v2 REST API.

Provides web scraping, site mapping, web search, and crawl management
through the Firecrawl cloud API (https://api.firecrawl.dev) using an
API key (sent as ``Authorization: Bearer`` on every call).  Stores
config in ``~/.kiss/third_party_agents/firecrawl/config.json``; the
optional ``base_url`` config key points the agent at a self-hosted
Firecrawl instance instead of the cloud API.

Firecrawl has no inbound message stream, so this adapter is
outbound-only and the ``--channel`` poll mode is disabled (``main``
passes ``make_backend=None`` to ``channel_main``).

Usage::

    agent = FirecrawlAgent()
    agent.run(prompt_template="Scrape https://example.com and summarize it")
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 120
_DEFAULT_BASE_URL = "https://api.firecrawl.dev"
_SEARCH_SOURCES = ("web", "news", "images")
_RESULT_FIELDS = ("title", "url", "description", "snippet", "date", "imageUrl")


def _bad_segment(value: str, name: str) -> str | None:
    """Reject *value* if it cannot safely form a single URL path segment.

    Values containing a path separator or a ``..`` sequence could
    traverse out of the intended API endpoint, so they are refused up
    front (defense in depth on top of ``quote(value, safe="")``).

    Args:
        value: Caller-supplied identifier destined for a URL path segment.
        name: Parameter name used in the error message.

    Returns:
        An ``{"ok": false, "error": ...}`` JSON string if *value* is
        unsafe or empty, otherwise None.
    """
    if not value or "/" in value or "\\" in value or ".." in value:
        return json.dumps(
            {
                "ok": False,
                "error": f"invalid {name}: must be non-empty without path separators or '..'",
            }
        )
    return None


_FIRECRAWL_DIR = Path.home() / ".kiss" / "third_party_agents" / "firecrawl"
_config = ChannelConfig(_FIRECRAWL_DIR, ("api_key",))


class FirecrawlChannelBackend(ToolMethodBackend):
    """Channel backend for the Firecrawl v2 REST API.

    Talks to the Firecrawl cloud API (or a self-hosted instance) over
    HTTP with a Bearer API key.  Outbound-only: Firecrawl has no
    inbound message stream, so the channel poll mode is disabled.
    """

    def __init__(self) -> None:
        self._base_url: str = _DEFAULT_BASE_URL
        self._api_key: str = ""
        self._request_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the Firecrawl config from disk.

        Returns:
            True if a valid config with ``api_key`` was loaded.
        """
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No Firecrawl config found."
            return False
        self._api_key = cfg["api_key"]
        self._base_url = cfg.get("base_url") or _DEFAULT_BASE_URL
        self._connection_info = f"Firecrawl configured at {self._base_url}"
        return True

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict:
        """Issue an authenticated Firecrawl v2 REST request.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, or ``"DELETE"``).
            path: API path starting with ``/v2/``.
            payload: Optional JSON body for POST requests.

        Returns:
            The decoded JSON response object.

        Raises:
            RuntimeError: On an HTTP error status, a non-JSON response
                body, a non-object JSON response, or an API-level
                ``{"success": false}`` reply.
        """
        url = self._base_url.rstrip("/") + path
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with self._request_lock:
            resp = requests.request(method, url, headers=headers, json=payload, timeout=_TIMEOUT)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            result: Any = resp.json()
        except ValueError:
            raise RuntimeError(f"non-JSON response: {resp.text[:200]}") from None
        if not isinstance(result, dict):
            raise RuntimeError(f"unexpected response shape: {str(result)[:200]}")
        if result.get("success") is False:
            raise RuntimeError(f"Firecrawl API error: {str(result.get('error', result))[:500]}")
        return result

    def firecrawl_scrape(
        self, url: str, formats: str = "markdown", only_main_content: bool = True
    ) -> str:
        """Scrape a single URL and return its content.

        Args:
            url: The URL to scrape, e.g. ``"https://example.com"``.
            formats: Comma-separated output formats such as
                ``"markdown"``, ``"html"``, ``"links"``, or ``"summary"``
                (default ``"markdown"``).
            only_main_content: Return only the main page content,
                excluding headers, navs, and footers (default True).

        Returns:
            JSON string with ok status, the scraped content per
            requested format (truncated), and page metadata
            (title, status_code, source url).
        """
        try:
            if not url.strip():
                return json.dumps({"ok": False, "error": "url cannot be empty"})
            format_list = [f.strip() for f in formats.split(",") if f.strip()]
            if not format_list:
                return json.dumps({"ok": False, "error": "formats cannot be empty"})
            result = self._request(
                "POST",
                "/v2/scrape",
                {
                    "url": url.strip(),
                    "formats": format_list,
                    "onlyMainContent": only_main_content,
                },
            )
            data = result.get("data") or {}
            metadata = data.get("metadata") or {}
            condensed: dict[str, Any] = {
                "ok": True,
                "title": metadata.get("title", ""),
                "status_code": metadata.get("statusCode", 0),
                "url": metadata.get("sourceURL", url.strip()),
            }
            for field in ("markdown", "summary", "html", "rawHtml"):
                if data.get(field):
                    condensed[field] = str(data[field])[:8000]
            if data.get("links"):
                condensed["links"] = data["links"][:200]
            return json.dumps(condensed)[:16000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def firecrawl_map(self, url: str, search: str = "", limit: int = 100) -> str:
        """Discover the URLs of a website (sitemap plus crawl-based discovery).

        Args:
            url: The base URL to map, e.g. ``"https://example.com"``.
            search: Optional query to order results by relevance
                (e.g. ``"blog"`` returns URLs containing "blog" first).
            limit: Maximum number of links to return (default 100).

        Returns:
            JSON string with ok status and the discovered links, each
            with url, title, and description when available.
        """
        try:
            if not url.strip():
                return json.dumps({"ok": False, "error": "url cannot be empty"})
            if limit < 1:
                return json.dumps({"ok": False, "error": "limit must be a positive integer"})
            payload: dict[str, Any] = {"url": url.strip(), "limit": limit}
            if search.strip():
                payload["search"] = search.strip()
            result = self._request("POST", "/v2/map", payload)
            links = [
                {key: link[key] for key in ("url", "title", "description") if link.get(key)}
                for link in result.get("links", [])
                if isinstance(link, dict)
            ]
            return json.dumps({"ok": True, "count": len(links), "links": links})[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def firecrawl_search(self, query: str, limit: int = 5, sources: str = "web") -> str:
        """Search the web (and optionally news or images) via Firecrawl.

        Args:
            query: The search query; supports operators like
                ``site:``, ``intitle:``, and quoted phrases.
            limit: Maximum results per source (default 5).
            sources: Comma-separated sources among ``"web"``,
                ``"news"``, and ``"images"`` (default ``"web"``).

        Returns:
            JSON string with ok status and condensed results grouped
            by source (title, url, description/snippet, date).
        """
        try:
            if not query.strip():
                return json.dumps({"ok": False, "error": "query cannot be empty"})
            if limit < 1:
                return json.dumps({"ok": False, "error": "limit must be a positive integer"})
            source_list = [s.strip() for s in sources.split(",") if s.strip()]
            if not source_list:
                return json.dumps({"ok": False, "error": "sources cannot be empty"})
            for source in source_list:
                if source not in _SEARCH_SOURCES:
                    return json.dumps(
                        {
                            "ok": False,
                            "error": f"invalid source {source!r}: "
                            f"must be one of {', '.join(_SEARCH_SOURCES)}",
                        }
                    )
            result = self._request(
                "POST",
                "/v2/search",
                {
                    "query": query.strip(),
                    "limit": limit,
                    "sources": [{"type": s} for s in source_list],
                },
            )
            data = result.get("data") or {}
            condensed: dict[str, Any] = {"ok": True}
            for source in source_list:
                condensed[source] = [
                    {key: item[key] for key in _RESULT_FIELDS if item.get(key)}
                    for item in data.get(source, [])
                    if isinstance(item, dict)
                ]
            return json.dumps(condensed)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def firecrawl_start_crawl(self, url: str, limit: int = 10) -> str:
        """Start an asynchronous crawl of a website.

        The crawl runs server-side; use ``firecrawl_get_crawl_status``
        with the returned crawl id to retrieve progress and page data.

        Args:
            url: The base URL to start crawling from.
            limit: Maximum number of pages to crawl (default 10).

        Returns:
            JSON string with ok status and the crawl id.
        """
        try:
            if not url.strip():
                return json.dumps({"ok": False, "error": "url cannot be empty"})
            if limit < 1:
                return json.dumps({"ok": False, "error": "limit must be a positive integer"})
            result = self._request("POST", "/v2/crawl", {"url": url.strip(), "limit": limit})
            return json.dumps({"ok": True, "crawl_id": str(result.get("id", ""))})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def firecrawl_get_crawl_status(self, crawl_id: str) -> str:
        """Get the status and page data of a crawl job.

        Args:
            crawl_id: The crawl job id returned by
                ``firecrawl_start_crawl``.

        Returns:
            JSON string with ok status, the crawl status
            (``scraping``/``completed``/``failed``), completed/total
            page counts, credits used, condensed per-page data
            (url, title, truncated markdown), and — when the response
            data exceeds Firecrawl's 10MB page limit — a ``next`` URL
            indicating more page data exists.
        """
        try:
            err = _bad_segment(crawl_id, "crawl_id")
            if err:
                return err
            result = self._request("GET", f"/v2/crawl/{quote(crawl_id, safe='')}")
            pages = []
            for page in result.get("data", []):
                if not isinstance(page, dict):
                    continue
                metadata = page.get("metadata") or {}
                pages.append(
                    {
                        "url": metadata.get("sourceURL", ""),
                        "title": metadata.get("title", ""),
                        "markdown": str(page.get("markdown", ""))[:2000],
                    }
                )
            out: dict[str, Any] = {
                "ok": True,
                "status": result.get("status", ""),
                "completed": result.get("completed", 0),
                "total": result.get("total", 0),
                "credits_used": result.get("creditsUsed", 0),
                "pages": pages,
            }
            if result.get("next"):
                out["next"] = str(result["next"])
            return json.dumps(out)[:16000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def firecrawl_cancel_crawl(self, crawl_id: str) -> str:
        """Cancel a running crawl job.

        Args:
            crawl_id: The crawl job id returned by
                ``firecrawl_start_crawl``.

        Returns:
            JSON string with ok status and the cancellation status.
        """
        try:
            err = _bad_segment(crawl_id, "crawl_id")
            if err:
                return err
            result = self._request("DELETE", f"/v2/crawl/{quote(crawl_id, safe='')}")
            return json.dumps({"ok": True, "status": result.get("status", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class FirecrawlAgent(BaseChannelAgent):
    """Channel agent with Firecrawl v2 REST API tools."""

    channel_system_prompt = (
        "You have access to the Firecrawl web-data API. Use "
        "firecrawl_scrape to fetch a single page as markdown (or other "
        "formats), firecrawl_map to discover the URLs of a website, "
        "firecrawl_search to search the web/news/images, "
        "firecrawl_start_crawl to start an asynchronous multi-page "
        "crawl, firecrawl_get_crawl_status to poll a crawl and read "
        "its page data, and firecrawl_cancel_crawl to cancel a crawl. "
        "There is no inbound message stream on this channel."
    )

    def __init__(self) -> None:
        super().__init__("Firecrawl Agent")
        self._backend = FirecrawlChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._api_key = cfg["api_key"]
            self._backend._base_url = cfg.get("base_url") or _DEFAULT_BASE_URL

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._api_key)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_firecrawl_auth() -> str:
            """Check if Firecrawl is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._is_authenticated():
                return (
                    "Not configured for Firecrawl. Use authenticate_firecrawl() "
                    "to configure.\n"
                    "You need an API key from https://www.firecrawl.dev "
                    "(sign up, then copy the key from the dashboard). "
                    "Optionally pass base_url for a self-hosted instance."
                )
            return json.dumps({"ok": True, "base_url": agent._backend._base_url})

        def authenticate_firecrawl(api_key: str, base_url: str = "") -> str:
            """Configure the Firecrawl API key and optional base URL.

            Args:
                api_key: Firecrawl API key from
                    https://www.firecrawl.dev.
                base_url: Optional base URL of a self-hosted Firecrawl
                    instance; empty uses the cloud API
                    ``https://api.firecrawl.dev``.

            Returns:
                Configuration result or error message.
            """
            if not api_key.strip():
                return "api_key cannot be empty."
            cfg = {"api_key": api_key.strip()}
            if base_url.strip():
                cfg["base_url"] = base_url.strip()
            try:
                _config.save(cfg)
                agent._backend._api_key = api_key.strip()
                agent._backend._base_url = base_url.strip() or _DEFAULT_BASE_URL
            except Exception as e:
                return json.dumps({"ok": False, "error": f"could not save config: {e}"})
            return json.dumps({"ok": True, "message": "Firecrawl configured."})

        def clear_firecrawl_auth() -> str:
            """Clear the stored Firecrawl configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._api_key = ""
            agent._backend._base_url = _DEFAULT_BASE_URL
            return "Firecrawl configuration cleared."

        return [check_firecrawl_auth, authenticate_firecrawl, clear_firecrawl_auth]


def main() -> None:
    """Run the FirecrawlAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): Firecrawl has no
    inbound message stream to poll.
    """
    channel_main(
        FirecrawlAgent,
        "kiss-firecrawl",
        channel_name="Firecrawl",
        make_backend=None,
    )


def get_tools() -> list:
    """Return the Firecrawl channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return FirecrawlAgent()._get_tools()


if __name__ == "__main__":
    main()
