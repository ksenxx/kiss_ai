"""Exa AI-powered web search tool for LLM agents.

Provides structured web search capabilities using the Exa API
(https://exa.ai) as an alternative to browser-based search. Returns
clean, structured results optimised for agent pipelines.

Requires the ``exa-py`` package and an ``EXA_API_KEY`` environment variable.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

try:
    from exa_py import Exa
except ImportError:  # pragma: no cover
    Exa = None  # type: ignore[assignment,misc]

_VALID_CATEGORIES = frozenset({
    "company",
    "research paper",
    "news",
    "personal site",
    "financial report",
    "people",
})


def is_available() -> bool:
    """Return True if Exa search can be used (SDK installed and key set)."""
    return Exa is not None and bool(os.environ.get("EXA_API_KEY"))


class ExaSearchTool:
    """Web search tool powered by the Exa API.

    Exa is an AI-native search engine that returns clean, structured results.
    Unlike browser-based search, Exa returns results directly as structured
    data with optional full-text content, highlights, and summaries.

    The tool is available when the ``exa-py`` package is installed and the
    ``EXA_API_KEY`` environment variable is set.
    """

    def __init__(self) -> None:
        if Exa is None:
            raise ImportError(
                "exa-py is required for Exa search: pip install exa-py"
            )
        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EXA_API_KEY environment variable is not set. "
                "Get a key at https://dashboard.exa.ai/api-keys"
            )
        self._client: Any = Exa(api_key=api_key)
        self._client.headers["x-exa-integration"] = "kiss-ai"

    def _format_results(self, results: Any) -> str:
        """Format Exa search results into a readable string."""
        if not results.results:
            return "No results found."
        parts: list[str] = []
        for i, item in enumerate(results.results, 1):
            title = getattr(item, "title", "") or ""
            url = getattr(item, "url", "") or ""
            score = getattr(item, "score", None)
            published = getattr(item, "published_date", None)

            header = f"[{i}] {title}\n    URL: {url}"
            if score is not None:
                header += f"\n    Score: {score:.4f}"
            if published:
                header += f"\n    Published: {published}"

            highlights = getattr(item, "highlights", None)
            text = getattr(item, "text", None)
            summary = getattr(item, "summary", None)

            if summary:
                header += f"\n    Summary: {summary}"
            if highlights:
                header += "\n    Highlights:\n      " + "\n      ".join(highlights)
            elif text:
                preview = text[:500].strip()
                if len(text) > 500:
                    preview += "..."
                header += f"\n    Content: {preview}"
            parts.append(header)
        return "\n\n".join(parts)

    def exa_search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "auto",
        category: str = "",
        include_domains: str = "",
        exclude_domains: str = "",
        include_text: str = "",
        exclude_text: str = "",
        start_published_date: str = "",
        end_published_date: str = "",
        include_content: bool = True,
        content_mode: str = "highlights",
    ) -> str:
        """Search the web using Exa and return structured results.

        Use this tool when you need to find information on the web. It is
        faster and more reliable than navigating to a search engine in the
        browser. Returns clean, structured results with optional content.

        Args:
            query: The search query string.
            num_results: Number of results to return (1-100, default 10).
            search_type: Search algorithm. One of "auto" (default), "neural",
                "instant". "auto" intelligently combines search methods.
            category: Optional category filter. One of: "company",
                "research paper", "news", "personal site",
                "financial report", "people". Leave empty for no filter.
            include_domains: Comma-separated list of domains to restrict
                results to (e.g. "arxiv.org,nature.com"). Leave empty
                for no restriction.
            exclude_domains: Comma-separated list of domains to exclude.
            include_text: Text that must appear in results.
            exclude_text: Text to exclude from results.
            start_published_date: Filter results published after this date
                (ISO 8601 format, e.g. "2024-01-01").
            end_published_date: Filter results published before this date.
            include_content: Whether to retrieve page content (default True).
            content_mode: How to retrieve content. One of "highlights"
                (relevant snippets, default), "text" (full text),
                "summary" (AI summary).

        Returns:
            Formatted search results with title, URL, and content for each
            result. Returns an error message string if the search fails.
        """
        try:
            kwargs: dict[str, Any] = {
                "num_results": min(max(num_results, 1), 100),
                "type": search_type,
            }

            if category and category in _VALID_CATEGORIES:
                kwargs["category"] = category

            if include_domains:
                kwargs["include_domains"] = [
                    d.strip() for d in include_domains.split(",") if d.strip()
                ]
            if exclude_domains:
                kwargs["exclude_domains"] = [
                    d.strip() for d in exclude_domains.split(",") if d.strip()
                ]
            if include_text:
                kwargs["include_text"] = [include_text]
            if exclude_text:
                kwargs["exclude_text"] = [exclude_text]
            if start_published_date:
                kwargs["start_published_date"] = start_published_date
            if end_published_date:
                kwargs["end_published_date"] = end_published_date

            if include_content:
                contents: dict[str, Any] = {}
                if content_mode == "text":
                    contents["text"] = True
                elif content_mode == "summary":
                    contents["summary"] = True
                else:
                    contents["highlights"] = True
                kwargs["contents"] = contents

            results = self._client.search(query, **kwargs)
            return self._format_results(results)
        except Exception as e:
            logger.debug("Exa search error", exc_info=True)
            return f"Error performing Exa search: {e}"

    def exa_find_similar(
        self,
        url: str,
        num_results: int = 10,
        include_domains: str = "",
        exclude_domains: str = "",
        include_content: bool = True,
    ) -> str:
        """Find web pages similar to a given URL using Exa.

        Use this when you have a specific page and want to find related
        content, similar articles, or competing products/services.

        Args:
            url: The URL to find similar pages for.
            num_results: Number of results to return (1-100, default 10).
            include_domains: Comma-separated list of domains to restrict to.
            exclude_domains: Comma-separated list of domains to exclude.
            include_content: Whether to retrieve page content (default True).

        Returns:
            Formatted list of similar pages with title, URL, and highlights.
            Returns an error message string if the search fails.
        """
        try:
            kwargs: dict[str, Any] = {
                "num_results": min(max(num_results, 1), 100),
            }

            if include_domains:
                kwargs["include_domains"] = [
                    d.strip() for d in include_domains.split(",") if d.strip()
                ]
            if exclude_domains:
                kwargs["exclude_domains"] = [
                    d.strip() for d in exclude_domains.split(",") if d.strip()
                ]

            if include_content:
                kwargs["contents"] = {"highlights": True}

            results = self._client.find_similar(url, **kwargs)
            return self._format_results(results)
        except Exception as e:
            logger.debug("Exa find_similar error", exc_info=True)
            return f"Error finding similar pages: {e}"

    def exa_get_contents(
        self,
        urls: str,
        content_mode: str = "text",
    ) -> str:
        """Retrieve the contents of specific URLs using Exa.

        Use this when you already know the URLs and want to extract their
        content without performing a search first.

        Args:
            urls: Comma-separated list of URLs to retrieve content from.
            content_mode: How to retrieve content. One of "text" (full text,
                default), "highlights" (relevant snippets), "summary"
                (AI summary).

        Returns:
            Formatted content for each URL. Returns an error message string
            if retrieval fails.
        """
        try:
            url_list = [u.strip() for u in urls.split(",") if u.strip()]
            if not url_list:
                return "Error: No valid URLs provided."

            contents: dict[str, Any] = {}
            if content_mode == "highlights":
                contents["highlights"] = True
            elif content_mode == "summary":
                contents["summary"] = True
            else:
                contents["text"] = True

            results = self._client.get_contents(url_list, **contents)
            return self._format_results(results)
        except Exception as e:
            logger.debug("Exa get_contents error", exc_info=True)
            return f"Error retrieving contents: {e}"

    def get_tools(self) -> list[Callable[..., str]]:
        """Return callable search tools for registration with an agent.

        Returns:
            List of callables: exa_search, exa_find_similar, exa_get_contents.
        """
        return [
            self.exa_search,
            self.exa_find_similar,
            self.exa_get_contents,
        ]
