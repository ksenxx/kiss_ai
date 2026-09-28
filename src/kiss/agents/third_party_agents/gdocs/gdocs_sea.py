# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Google Docs Agent — channel agent for the Google Docs REST API.

Provides access to Google Docs; sign-in and every API call go through
Composio (see :mod:`._composio_google`), which holds the Google token.
Documents are read and edited through plain ``requests`` calls against
``https://docs.googleapis.com/v1`` and listed through the Drive v3
``files`` endpoint.

The Docs REST API has no inbound message stream, so this adapter is
outbound-only: ``poll_messages`` always returns no messages and poll
mode is disabled (``main`` passes ``make_backend=None``).

Usage::

    agent = GoogleDocsAgent()
    agent.run(prompt_template="Create a doc titled 'Notes' and add a summary")
"""

from __future__ import annotations

import json
import threading
from typing import Any
from urllib.parse import quote

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
)
from kiss.agents.third_party_agents._composio_google import (
    ComposioSession,
    connected_account_id,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    google_auth_prompt,
    make_google_auth_tools,
)

_SERVICE = "google_docs"
_TIMEOUT = 30


def description() -> str:
    """Return the one-sentence help text shown by ``/gdocs help``."""
    return (
        "Creates, reads, edits and lists Google Docs through the Docs and Drive REST APIs "
        "with sign-in handled by Composio (outbound only, no message polling); use it with "
        "`run_agent(agent=\"gdocs\", task=...)` or the `kiss-gdocs -t \"<task>\"` CLI."
    )


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
        unsafe, otherwise None.
    """
    if "/" in value or "\\" in value or ".." in value:
        return json.dumps(
            {"ok": False, "error": f"invalid {name}: must not contain path separators or '..'"}
        )
    return None


def _escape_drive_query(query: str) -> str:
    """Escape a value for embedding in a Drive ``q`` single-quoted string.

    Args:
        query: Raw user-supplied search text.

    Returns:
        The text with backslashes and single quotes backslash-escaped.
    """
    return query.replace("\\", "\\\\").replace("'", "\\'")


def _extract_content_text(content: list[Any]) -> list[str]:
    """Collect plain-text runs from a list of Docs structural elements.

    Walks paragraphs (``paragraph.elements[].textRun.content``) and
    tables (recursing into each ``tableCells[].content``); other
    structural elements (section breaks, tables of contents) are
    skipped.

    Args:
        content: The ``body.content`` list from a Docs ``Document``.

    Returns:
        The text fragments in document order.
    """
    parts: list[str] = []
    for element in content:
        if not isinstance(element, dict):
            continue
        paragraph = element.get("paragraph")
        if isinstance(paragraph, dict):
            for run in paragraph.get("elements", []):
                text = run.get("textRun", {}).get("content", "")
                if text:
                    parts.append(text)
        table = element.get("table")
        if isinstance(table, dict):
            for row in table.get("tableRows", []):
                for cell in row.get("tableCells", []):
                    parts.extend(_extract_content_text(cell.get("content", [])))
    return parts


def _extract_tabs_text(tabs: list[Any]) -> list[str]:
    """Collect plain-text runs from a Docs ``tabs`` array, depth-first.

    Walks each tab's ``documentTab.body.content`` (via
    :func:`_extract_content_text`) and then recurses into its
    ``childTabs``, so text from every tab in a multi-tab document is
    collected in tab order.

    Args:
        tabs: The ``tabs`` list from a Docs ``Document`` fetched with
            ``includeTabsContent=true``.

    Returns:
        The text fragments across all tabs in document order.
    """
    parts: list[str] = []
    for tab in tabs:
        if not isinstance(tab, dict):
            continue
        document_tab = tab.get("documentTab")
        if isinstance(document_tab, dict):
            parts.extend(
                _extract_content_text(document_tab.get("body", {}).get("content", []))
            )
        child_tabs = tab.get("childTabs")
        if isinstance(child_tabs, list):
            parts.extend(_extract_tabs_text(child_tabs))
    return parts


class GoogleDocsChannelBackend(ToolMethodBackend):
    """Channel backend for the Google Docs REST API.

    Talks to ``docs.googleapis.com`` (and Drive v3 for listing) through
    Composio's proxy, which signs each request with the user's Google
    token.  Outbound-only: there is no inbound message stream, so
    :meth:`poll_messages` always returns no messages.
    """

    def __init__(self) -> None:
        self._http: Any = ComposioSession(_SERVICE)
        self._base_url: str = "https://docs.googleapis.com/v1"
        self._drive_base_url: str = "https://www.googleapis.com/drive/v3"
        self._request_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Check that Google Docs is connected through Composio.

        Returns:
            True if a Composio connection exists.
        """
        if not connected_account_id(_SERVICE):
            self._connection_info = "Google Docs is not connected. Please authenticate first."
            return False
        self._connection_info = "Google Docs connected through Composio."
        return True

    def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue an authenticated Google API request.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, or ``"PUT"``).
            url: Full request URL.
            params: Optional query parameters.
            payload: Optional JSON body.

        Returns:
            ``{"ok": True, "result": <parsed JSON>}`` on success or
            ``{"ok": False, "error": ...}`` on an HTTP error status.
        """
        with self._request_lock:
            resp = self._http.request(
                method, url, params=params, json=payload,
                timeout=_TIMEOUT,
            )
        if resp.status_code >= 400:
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
        try:
            result: Any = resp.json()
        except ValueError:
            result = resp.text
        return {"ok": True, "result": result}

    def _batch_update(self, document_id: str, requests_list: list[Any]) -> dict[str, Any]:
        """POST a ``documents.batchUpdate`` request.

        Args:
            document_id: The document to update (already path-checked).
            requests_list: List of Docs ``Request`` objects.

        Returns:
            The ``_request`` result dict.
        """
        url = f"{self._base_url}/documents/{quote(document_id, safe='')}:batchUpdate"
        return self._request("POST", url, payload={"requests": requests_list})

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Return no messages: the Docs REST API has no inbound message stream.

        Args:
            channel_id: Ignored.
            oldest: Cursor, returned unchanged.
            limit: Ignored.

        Returns:
            ``([], oldest)``.
        """
        return [], oldest

    def gdocs_create_document(self, title: str) -> str:
        """Create a new blank Google Doc with the given title.

        Args:
            title: Title of the new document.

        Returns:
            JSON string with ok status, the new document_id, and title.
        """
        try:
            resp = self._request("POST", f"{self._base_url}/documents", payload={"title": title})
            if not resp["ok"]:
                return json.dumps(resp)
            doc = resp["result"]
            return json.dumps(
                {
                    "ok": True,
                    "document_id": doc.get("documentId", ""),
                    "title": doc.get("title", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdocs_read_document(self, document_id: str) -> str:
        """Read a Google Doc and return its plain text content.

        Fetches the document with ``includeTabsContent=true`` and walks
        every tab (including nested child tabs), concatenating the text
        runs of each tab body's paragraphs and tables.  Documents
        without a ``tabs`` array fall back to the legacy top-level body.

        Args:
            document_id: The ID of the document to read.

        Returns:
            JSON string with ok status, title, and the extracted text
            across all tabs (truncated to 8000 characters).
        """
        try:
            err = _bad_segment(document_id, "document_id")
            if err:
                return err
            url = f"{self._base_url}/documents/{quote(document_id, safe='')}"
            resp = self._request("GET", url, params={"includeTabsContent": "true"})
            if not resp["ok"]:
                return json.dumps(resp)
            doc = resp["result"]
            tabs = doc.get("tabs")
            if isinstance(tabs, list):
                text = "".join(_extract_tabs_text(tabs))
            else:
                text = "".join(_extract_content_text(doc.get("body", {}).get("content", [])))
            return json.dumps(
                {"ok": True, "title": doc.get("title", ""), "text": text[:8000]}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdocs_append_text(self, document_id: str, text: str) -> str:
        """Append text to the end of a Google Doc's body.

        Args:
            document_id: The ID of the document to append to.
            text: Text to insert at the end of the document body.

        Returns:
            JSON string with ok status.
        """
        try:
            err = _bad_segment(document_id, "document_id")
            if err:
                return err
            resp = self._batch_update(
                document_id,
                [{"insertText": {"endOfSegmentLocation": {"segmentId": ""}, "text": text}}],
            )
            if not resp["ok"]:
                return json.dumps(resp)
            return json.dumps(
                {"ok": True, "document_id": resp["result"].get("documentId", "")}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdocs_replace_text(
        self, document_id: str, find: str, replace: str, match_case: bool = True
    ) -> str:
        """Replace all occurrences of a string in a Google Doc.

        Args:
            document_id: The ID of the document to update.
            find: The text to search for.
            replace: The replacement text.
            match_case: Whether the search is case-sensitive. Default: True.

        Returns:
            JSON string with ok status and the number of occurrences changed.
        """
        try:
            err = _bad_segment(document_id, "document_id")
            if err:
                return err
            resp = self._batch_update(
                document_id,
                [
                    {
                        "replaceAllText": {
                            "replaceText": replace,
                            "containsText": {"text": find, "matchCase": match_case},
                        }
                    }
                ],
            )
            if not resp["ok"]:
                return json.dumps(resp)
            replies = resp["result"].get("replies", [])
            changed = 0
            if replies:
                changed = replies[0].get("replaceAllText", {}).get("occurrencesChanged", 0)
            return json.dumps({"ok": True, "occurrences_changed": changed})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdocs_insert_text(self, document_id: str, text: str, index: int) -> str:
        """Insert text at a specific index in a Google Doc's body.

        Args:
            document_id: The ID of the document to update.
            text: Text to insert.
            index: Zero-based body index to insert at (the body starts
                at index 1).

        Returns:
            JSON string with ok status.
        """
        try:
            err = _bad_segment(document_id, "document_id")
            if err:
                return err
            resp = self._batch_update(
                document_id,
                [{"insertText": {"location": {"index": index}, "text": text}}],
            )
            if not resp["ok"]:
                return json.dumps(resp)
            return json.dumps(
                {"ok": True, "document_id": resp["result"].get("documentId", "")}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdocs_batch_update(self, document_id: str, requests_json: str) -> str:
        """Apply a raw list of Docs API batchUpdate requests to a document.

        Use for operations without a dedicated tool (styling, tables,
        bullets, deleting ranges, ...). See the Docs API ``Request``
        reference for the available request objects.

        Args:
            document_id: The ID of the document to update.
            requests_json: JSON array string of Docs ``Request``
                objects, e.g.
                ``'[{"insertText": {"location": {"index": 1}, "text": "hi"}}]'``.

        Returns:
            JSON string with ok status and the batchUpdate replies.
        """
        try:
            err = _bad_segment(document_id, "document_id")
            if err:
                return err
            try:
                requests_list = json.loads(requests_json)
            except ValueError as e:
                return json.dumps({"ok": False, "error": f"requests_json is not valid JSON: {e}"})
            if not isinstance(requests_list, list):
                return json.dumps({"ok": False, "error": "requests_json must be a JSON array"})
            resp = self._batch_update(document_id, requests_list)
            if not resp["ok"]:
                return json.dumps(resp)
            return json.dumps(
                {"ok": True, "replies": resp["result"].get("replies", [])}
            )[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdocs_list_documents(self, query: str = "", max_results: int = 20) -> str:
        """List Google Docs in the user's Drive, optionally filtered by name.

        Args:
            query: Optional substring to match against document names.
            max_results: Maximum number of documents to return. Default: 20.

        Returns:
            JSON string with ok status and files (id, name,
            modifiedTime, webViewLink).
        """
        try:
            q = "mimeType='application/vnd.google-apps.document'"
            if query:
                q += f" and name contains '{_escape_drive_query(query)}'"
            resp = self._request(
                "GET",
                f"{self._drive_base_url}/files",
                params={
                    "q": q,
                    "pageSize": max_results,
                    "fields": "files(id,name,modifiedTime,webViewLink)",
                },
            )
            if not resp["ok"]:
                return json.dumps(resp)
            return json.dumps(
                {"ok": True, "files": resp["result"].get("files", [])}
            )[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class GoogleDocsAgent(BaseChannelAgent):
    """Channel agent with Google Docs REST API tools."""

    channel_system_prompt = google_auth_prompt(_SERVICE, "Google Docs")

    def __init__(self) -> None:
        super().__init__("Google Docs Agent")
        self._backend = GoogleDocsChannelBackend()

    def _is_authenticated(self) -> bool:
        """Return True if Google Docs is connected through Composio."""
        return bool(connected_account_id(_SERVICE))

    def _get_auth_tools(self) -> list:
        """Return the Composio sign-in tool set for Google Docs."""
        return make_google_auth_tools(self, _SERVICE, "Google Docs", self._backend.connect)


def main() -> None:
    """Run the GoogleDocsAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): the Docs REST API
    has no inbound message stream to poll.
    """
    channel_main(
        GoogleDocsAgent,
        "kiss-gdocs",
        channel_name="Google Docs",
        make_backend=None,
    )


def tools() -> list:
    """Return the Google Docs channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    Composio connection recorded under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return GoogleDocsAgent()._get_tools()


if __name__ == "__main__":
    main()
