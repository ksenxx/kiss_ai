# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Notion Agent — channel agent for the Notion REST API.

Provides access to a Notion workspace via the official REST API using
an internal-integration token (sent as ``Authorization: Bearer`` with
``Notion-Version: 2022-06-28`` on every call).  Stores config in
``~/.kiss/third_party_agents/notion/config.json``.

Notion's REST API has no inbound message stream, so this adapter is
outbound-only and the ``--channel`` poll mode is disabled (``main``
passes ``make_backend=None`` to ``channel_main``).

Usage::

    agent = NotionAgent()
    agent.run(prompt_template="Search my Notion for 'meeting notes'")
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from kiss.agents.third_party_agents._browser_handoff import portal_handoff
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_NOTION_VERSION = "2022-06-28"
_MAX_OUTPUT = 8000


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


def _parse_json_param(raw: str, name: str, expect: type) -> tuple[Any, str]:
    """Parse a JSON string tool parameter, enforcing its container type.

    Args:
        raw: The raw JSON string supplied by the caller.
        name: Parameter name used in error messages.
        expect: Required container type (``dict`` or ``list``).

    Returns:
        ``(value, "")`` on success, or ``(None, error_json)`` where
        *error_json* is an ``{"ok": false, ...}`` JSON string.
    """
    try:
        value = json.loads(raw)
    except ValueError:
        return None, json.dumps({"ok": False, "error": f"{name} is not valid JSON"})
    if not isinstance(value, expect):
        kind = "object" if expect is dict else "array"
        return None, json.dumps({"ok": False, "error": f"{name} must be a JSON {kind}"})
    return value, ""


def _plain_text(rich: Any) -> str:
    """Join the ``plain_text`` fields of a Notion rich-text array.

    Args:
        rich: A Notion rich-text list (or anything else, yielding "").

    Returns:
        The concatenated plain text.
    """
    if not isinstance(rich, list):
        return ""
    return "".join(str(item.get("plain_text", "")) for item in rich if isinstance(item, dict))


def _text_rich(text: str) -> dict[str, Any]:
    """Build a single Notion rich-text object from plain text.

    Args:
        text: The text content.

    Returns:
        A ``{"type": "text", "text": {"content": ...}}`` dict.
    """
    return {"type": "text", "text": {"content": text}}


def _page_title(page: dict[str, Any]) -> str:
    """Extract a page's title from its ``title``-type property.

    Args:
        page: A Notion page object.

    Returns:
        The title plain text, or "" when no title property exists.
    """
    for prop in (page.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return _plain_text(prop.get("title"))
    return ""


def _prop_value(prop: dict[str, Any]) -> Any:
    """Condense one Notion page property to a plain value.

    Args:
        prop: A Notion property value object.

    Returns:
        Plain text for title/rich_text, the ``name`` for select-like
        values, the scalar itself for numbers/strings/booleans/None,
        or ``"<type>"`` for anything more complex.
    """
    ptype = prop.get("type", "")
    value = prop.get(ptype)
    if ptype in ("title", "rich_text"):
        return _plain_text(value)
    if isinstance(value, dict) and "name" in value:
        return value["name"]
    if value is None or isinstance(value, (str, int, float)):
        return value
    return f"<{ptype}>"


def _condense_page(page: dict[str, Any]) -> dict[str, Any]:
    """Condense a Notion page object to its useful fields.

    Args:
        page: A Notion page object.

    Returns:
        Dict with id, title, url, archived flag, parent, and
        last_edited_time.
    """
    return {
        "id": page.get("id", ""),
        "title": _page_title(page),
        "url": page.get("url", ""),
        "archived": page.get("archived", False),
        "parent": page.get("parent", {}),
        "last_edited_time": page.get("last_edited_time", ""),
    }


def _condense_block(block: dict[str, Any]) -> dict[str, Any]:
    """Condense a Notion block to its id, type, plain text, and child flag.

    Args:
        block: A Notion block object.

    Returns:
        Dict with id, type, text, and has_children.
    """
    btype = block.get("type", "")
    value = block.get(btype) or {}
    text = _plain_text(value.get("rich_text")) if isinstance(value, dict) else ""
    return {
        "id": block.get("id", ""),
        "type": btype,
        "text": text,
        "has_children": block.get("has_children", False),
    }


def _condense_search_result(result: dict[str, Any]) -> dict[str, Any]:
    """Condense one search result (page or database) to id/title/url.

    Args:
        result: A Notion page or database object.

    Returns:
        Dict with object kind, id, title, and url.
    """
    obj = result.get("object", "")
    title = _page_title(result) if obj == "page" else _plain_text(result.get("title"))
    return {"object": obj, "id": result.get("id", ""), "title": title, "url": result.get("url", "")}


_NOTION_DIR = Path.home() / ".kiss" / "third_party_agents" / "notion"
_config = ChannelConfig(_NOTION_DIR, ("token",))


class NotionChannelBackend(ToolMethodBackend):
    """Channel backend for the Notion REST API.

    Talks to the Notion API over HTTPS with an internal-integration
    token.  Outbound-only: Notion has no inbound message stream over
    plain REST.
    """

    def __init__(self) -> None:
        self._base_url: str = "https://api.notion.com/v1"
        self._token: str = ""
        self._http: Any = requests
        self._request_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the Notion config from disk.

        In Muse-auth mode (the default) the real integration
        token lives in the Muse vault (auto-enrolled from the legacy
        config on first connect); this process only holds a surrogate
        and every API call is executed at the daemon boundary.

        Returns:
            True if a valid config with a ``token`` was loaded.
        """
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import (
                MuseBoundarySession,
                mint_surrogate,
            )

            handle = mint_surrogate("notion")
            if handle is None:
                # Vault-first: the legacy config token is only read when
                # the vault has no enrollment yet (one-time migration).
                from kiss.agents.third_party_agents.muse_auth.client import bearer_surrogate

                cfg = _config.load()
                surrogate = bearer_surrogate("notion", (cfg or {}).get("token", ""))
            else:
                surrogate = handle.token
            if not surrogate:
                self._connection_info = "No Notion credential in the Muse vault or config."
                return False
            self._token = surrogate
            self._http = MuseBoundarySession("notion")
            # The credential lives in the vault now; scrub any plaintext
            # copy left in the legacy config so the agent process never
            # holds the real token again.
            _config.scrub_secrets(("token",))
            self._connection_info = "Notion API configured (Muse-auth)."
            return True
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No Notion config found."
            return False
        self._token = cfg["token"]
        self._connection_info = "Notion API configured."
        return True

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Issue an authenticated Notion REST request.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, or ``"PATCH"``).
            path: API path relative to the base URL, e.g. ``"/search"``.
            payload: Optional JSON body.
            params: Optional query-string parameters.

        Returns:
            ``(data, "")`` with the parsed JSON response on success, or
            ``(None, error_json)`` where *error_json* is an
            ``{"ok": false, ...}`` JSON string.
        """
        url = self._base_url.rstrip("/") + path
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }
        with self._request_lock:
            resp = self._http.request(
                method, url, headers=headers, json=payload, params=params, timeout=_TIMEOUT
            )
        if resp.status_code >= 400:
            return None, json.dumps(
                {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
            )
        try:
            data: dict[str, Any] = resp.json()
        except ValueError:
            return None, json.dumps({"ok": False, "error": "non-JSON response from Notion API"})
        return data, ""

    def notion_search(
        self, query: str, filter_type: str = "", start_cursor: str = "", page_size: int = 20
    ) -> str:
        """Search Notion pages and databases shared with the integration by title.

        Args:
            query: Title text to search for; empty returns everything shared.
            filter_type: Optional ``"page"`` or ``"database"`` to restrict
                the object kind.
            start_cursor: Pagination cursor from a previous call.
            page_size: Maximum results to return (default 20).

        Returns:
            JSON string with ok status, condensed results (object, id,
            title, url), next_cursor, and has_more.
        """
        try:
            body: dict[str, Any] = {"query": query, "page_size": page_size}
            if filter_type:
                if filter_type not in ("page", "database"):
                    return json.dumps(
                        {"ok": False, "error": "filter_type must be 'page' or 'database'"}
                    )
                body["filter"] = {"property": "object", "value": filter_type}
            if start_cursor:
                body["start_cursor"] = start_cursor
            data, err = self._request("POST", "/search", body)
            if err:
                return err
            assert data is not None
            out = {
                "ok": True,
                "results": [_condense_search_result(r) for r in data.get("results", [])],
                "next_cursor": data.get("next_cursor"),
                "has_more": data.get("has_more", False),
            }
            return json.dumps(out)[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_get_page(self, page_id: str) -> str:
        """Get a Notion page's title, URL, parent, and condensed properties.

        Args:
            page_id: The page ID (UUID).

        Returns:
            JSON string with ok status and the condensed page (id,
            title, url, archived, parent, last_edited_time, properties
            condensed to plain values).
        """
        try:
            err = _bad_segment(page_id, "page_id")
            if err:
                return err
            data, req_err = self._request("GET", f"/pages/{quote(page_id, safe='')}")
            if req_err:
                return req_err
            assert data is not None
            page = _condense_page(data)
            page["properties"] = {
                name: _prop_value(prop) for name, prop in (data.get("properties") or {}).items()
            }
            return json.dumps({"ok": True, "page": page})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_get_block_children(
        self, block_id: str, start_cursor: str = "", page_size: int = 50
    ) -> str:
        """List a block's (or page's) child blocks with their plain text.

        Args:
            block_id: The block or page ID whose children to list.
            start_cursor: Pagination cursor from a previous call.
            page_size: Maximum blocks to return (default 50).

        Returns:
            JSON string with ok status, blocks (id, type, text,
            has_children), next_cursor, and has_more.
        """
        try:
            err = _bad_segment(block_id, "block_id")
            if err:
                return err
            params = {"page_size": str(page_size)}
            if start_cursor:
                params["start_cursor"] = start_cursor
            data, req_err = self._request(
                "GET", f"/blocks/{quote(block_id, safe='')}/children", params=params
            )
            if req_err:
                return req_err
            assert data is not None
            out = {
                "ok": True,
                "blocks": [_condense_block(b) for b in data.get("results", [])],
                "next_cursor": data.get("next_cursor"),
                "has_more": data.get("has_more", False),
            }
            return json.dumps(out)[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_append_paragraph(self, block_id: str, text: str) -> str:
        """Append a paragraph of plain text to a page or block.

        Args:
            block_id: The page or block ID to append to.
            text: The paragraph text.

        Returns:
            JSON string with ok status and the appended block ids.
        """
        try:
            err = _bad_segment(block_id, "block_id")
            if err:
                return err
            if not text.strip():
                return json.dumps({"ok": False, "error": "text cannot be empty"})
            children = [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [_text_rich(text)]},
                }
            ]
            return self._append_children(block_id, children)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_append_blocks(self, block_id: str, children_json: str) -> str:
        """Append caller-supplied Notion block objects to a page or block.

        Args:
            block_id: The page or block ID to append to.
            children_json: JSON array string of Notion block objects,
                e.g. ``'[{"object":"block","type":"paragraph",...}]'``.

        Returns:
            JSON string with ok status and the appended block ids.
        """
        try:
            err = _bad_segment(block_id, "block_id")
            if err:
                return err
            children, parse_err = _parse_json_param(children_json, "children_json", list)
            if parse_err:
                return parse_err
            return self._append_children(block_id, children)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def _append_children(self, block_id: str, children: list[dict[str, Any]]) -> str:
        """PATCH child blocks onto *block_id* and condense the response.

        Args:
            block_id: The (already validated) parent block or page ID.
            children: Notion block objects to append.

        Returns:
            JSON string with ok status and the appended block summaries.
        """
        data, err = self._request(
            "PATCH", f"/blocks/{quote(block_id, safe='')}/children", {"children": children}
        )
        if err:
            return err
        assert data is not None
        out = {"ok": True, "appended": [_condense_block(b) for b in data.get("results", [])]}
        return json.dumps(out)[:_MAX_OUTPUT]

    def notion_create_page(
        self, parent_id: str, title: str, parent_type: str = "page", properties_json: str = ""
    ) -> str:
        """Create a new Notion page under a page or database parent.

        Args:
            parent_id: The parent page or database ID.
            title: The new page's title (may be empty when
                *properties_json* supplies the title property).
            parent_type: ``"page"`` (default) or ``"database"``.
            properties_json: Optional JSON object string of extra Notion
                page properties, merged over the title property (needed
                for database parents with typed columns).

        Returns:
            JSON string with ok status and the created page's id, title,
            and url.
        """
        try:
            if parent_type not in ("page", "database"):
                return json.dumps(
                    {"ok": False, "error": "parent_type must be 'page' or 'database'"}
                )
            properties: dict[str, Any] = {}
            if title:
                properties["title"] = {"title": [_text_rich(title)]}
            if properties_json:
                extra, parse_err = _parse_json_param(properties_json, "properties_json", dict)
                if parse_err:
                    return parse_err
                properties.update(extra)
            parent = (
                {"page_id": parent_id} if parent_type == "page" else {"database_id": parent_id}
            )
            data, err = self._request(
                "POST", "/pages", {"parent": parent, "properties": properties}
            )
            if err:
                return err
            assert data is not None
            return json.dumps({"ok": True, "page": _condense_page(data)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_update_page(
        self, page_id: str, properties_json: str = "", archived: str = ""
    ) -> str:
        """Update a Notion page's properties and/or archive state.

        Args:
            page_id: The page ID to update.
            properties_json: Optional JSON object string of Notion page
                properties to set.
            archived: Tri-state string: ``""`` leaves the archive state
                unchanged, ``"true"`` moves the page to the trash,
                ``"false"`` restores it.

        Returns:
            JSON string with ok status and the updated page summary.
        """
        try:
            err = _bad_segment(page_id, "page_id")
            if err:
                return err
            body: dict[str, Any] = {}
            if properties_json:
                properties, parse_err = _parse_json_param(
                    properties_json, "properties_json", dict
                )
                if parse_err:
                    return parse_err
                body["properties"] = properties
            if archived == "true":
                body["archived"] = True
            elif archived == "false":
                body["archived"] = False
            elif archived:
                return json.dumps(
                    {"ok": False, "error": "archived must be '', 'true', or 'false'"}
                )
            if not body:
                return json.dumps(
                    {"ok": False, "error": "nothing to update: pass properties_json or archived"}
                )
            data, req_err = self._request(
                "PATCH", f"/pages/{quote(page_id, safe='')}", body
            )
            if req_err:
                return req_err
            assert data is not None
            return json.dumps({"ok": True, "page": _condense_page(data)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_get_database(self, database_id: str) -> str:
        """Get a Notion database's title, URL, and property schema.

        Args:
            database_id: The database ID (UUID).

        Returns:
            JSON string with ok status and the condensed database (id,
            title, url, and ``properties`` mapping column name to type).
        """
        try:
            err = _bad_segment(database_id, "database_id")
            if err:
                return err
            data, req_err = self._request("GET", f"/databases/{quote(database_id, safe='')}")
            if req_err:
                return req_err
            assert data is not None
            db = {
                "id": data.get("id", ""),
                "title": _plain_text(data.get("title")),
                "url": data.get("url", ""),
                "properties": {
                    name: prop.get("type", "")
                    for name, prop in (data.get("properties") or {}).items()
                },
            }
            return json.dumps({"ok": True, "database": db})[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_query_database(
        self,
        database_id: str,
        filter_json: str = "",
        sorts_json: str = "",
        start_cursor: str = "",
        page_size: int = 20,
    ) -> str:
        """Query a Notion database's pages with optional filter and sorts.

        Args:
            database_id: The database ID to query.
            filter_json: Optional JSON object string with a Notion filter,
                e.g. ``'{"property":"Status","select":{"equals":"Done"}}'``.
            sorts_json: Optional JSON array string of Notion sort objects.
            start_cursor: Pagination cursor from a previous call.
            page_size: Maximum pages to return (default 20).

        Returns:
            JSON string with ok status, condensed page results,
            next_cursor, and has_more.
        """
        try:
            err = _bad_segment(database_id, "database_id")
            if err:
                return err
            body: dict[str, Any] = {"page_size": page_size}
            if filter_json:
                filter_obj, parse_err = _parse_json_param(filter_json, "filter_json", dict)
                if parse_err:
                    return parse_err
                body["filter"] = filter_obj
            if sorts_json:
                sorts, parse_err = _parse_json_param(sorts_json, "sorts_json", list)
                if parse_err:
                    return parse_err
                body["sorts"] = sorts
            if start_cursor:
                body["start_cursor"] = start_cursor
            data, req_err = self._request(
                "POST", f"/databases/{quote(database_id, safe='')}/query", body
            )
            if req_err:
                return req_err
            assert data is not None
            out = {
                "ok": True,
                "results": [_condense_page(p) for p in data.get("results", [])],
                "next_cursor": data.get("next_cursor"),
                "has_more": data.get("has_more", False),
            }
            return json.dumps(out)[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_list_users(self, start_cursor: str = "") -> str:
        """List the users in the Notion workspace.

        Args:
            start_cursor: Pagination cursor from a previous call.

        Returns:
            JSON string with ok status, users (id, name, type),
            next_cursor, and has_more.
        """
        try:
            params: dict[str, str] = {}
            if start_cursor:
                params["start_cursor"] = start_cursor
            data, err = self._request("GET", "/users", params=params)
            if err:
                return err
            assert data is not None
            users = [
                {"id": u.get("id", ""), "name": u.get("name", ""), "type": u.get("type", "")}
                for u in data.get("results", [])
            ]
            out = {
                "ok": True,
                "users": users,
                "next_cursor": data.get("next_cursor"),
                "has_more": data.get("has_more", False),
            }
            return json.dumps(out)[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_create_comment(self, page_id: str, text: str) -> str:
        """Add a comment to a Notion page.

        Args:
            page_id: The page ID to comment on.
            text: The comment text.

        Returns:
            JSON string with ok status and the created comment's id and
            discussion_id.
        """
        try:
            if not text.strip():
                return json.dumps({"ok": False, "error": "text cannot be empty"})
            body = {"parent": {"page_id": page_id}, "rich_text": [_text_rich(text)]}
            data, err = self._request("POST", "/comments", body)
            if err:
                return err
            assert data is not None
            comment = {
                "id": data.get("id", ""),
                "discussion_id": data.get("discussion_id", ""),
            }
            return json.dumps({"ok": True, "comment": comment})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def notion_get_comments(self, block_id: str, start_cursor: str = "") -> str:
        """List the unresolved comments on a page or block.

        Args:
            block_id: The page or block ID whose comments to list.
            start_cursor: Pagination cursor from a previous call.

        Returns:
            JSON string with ok status, comments (id, text,
            created_time, created_by), next_cursor, and has_more.
        """
        try:
            err = _bad_segment(block_id, "block_id")
            if err:
                return err
            params = {"block_id": block_id}
            if start_cursor:
                params["start_cursor"] = start_cursor
            data, req_err = self._request("GET", "/comments", params=params)
            if req_err:
                return req_err
            assert data is not None
            comments = [
                {
                    "id": c.get("id", ""),
                    "text": _plain_text(c.get("rich_text")),
                    "created_time": c.get("created_time", ""),
                    "created_by": (c.get("created_by") or {}).get("id", ""),
                }
                for c in data.get("results", [])
            ]
            out = {
                "ok": True,
                "comments": comments,
                "next_cursor": data.get("next_cursor"),
                "has_more": data.get("has_more", False),
            }
            return json.dumps(out)[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class NotionAgent(BaseChannelAgent):
    """Channel agent with Notion REST API tools."""

    channel_system_prompt = (
        "You are operating a Notion workspace through its REST API. Use "
        "notion_search to find pages and databases by title, "
        "notion_get_page for a page's properties, "
        "notion_get_block_children to read page content, "
        "notion_append_paragraph or notion_append_blocks to add content, "
        "notion_create_page to create pages under a page or database "
        "parent, notion_update_page to change properties or archive, "
        "notion_get_database and notion_query_database for database "
        "schemas and rows, notion_list_users for workspace members, and "
        "notion_create_comment / notion_get_comments for page comments. "
        "Only pages and databases shared with the integration are "
        "visible."
    )

    def __init__(self) -> None:
        super().__init__("Notion Agent")
        self._backend = NotionChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            # Muse-auth mode: connect() wires a vault surrogate and the
            # boundary session; the real token never enters this process
            # once migrated.
            self._backend.connect()
            return
        cfg = _config.load()
        if cfg:
            self._backend._token = cfg["token"]

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._token)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_notion_auth() -> str:
            """Check if Notion is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._is_authenticated():
                return (
                    "Not configured for Notion. Use authenticate_notion() to "
                    "configure.\n"
                    "Create an internal integration at "
                    "https://www.notion.so/profile/integrations, copy its "
                    "secret token, and share the pages or databases you want "
                    "the agent to access with that integration."
                    + "\n"
                    + portal_handoff("https://www.notion.so/profile/integrations")
                )
            return json.dumps({"ok": True, "message": "Notion integration token is configured."})

        def authenticate_notion(token: str) -> str:
            """Configure the Notion internal-integration token.

            Args:
                token: The integration secret token (starts with ``ntn_``
                    or ``secret_``).

            Returns:
                Configuration result or error message.
            """
            if not token.strip():
                return "token cannot be empty."
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            try:
                _config.save({"token": token.strip()})
                if muse_auth_enabled():
                    # Re-enroll straight into the Muse vault: clear any
                    # existing entry first so a rotated token replaces
                    # the old one (connect() is vault-first and would
                    # otherwise keep minting the stale credential).
                    from kiss.agents.third_party_agents.muse_auth.client import (
                        clear_credentials,
                    )

                    clear_credentials("notion")
                    agent._backend.connect()
                else:
                    agent._backend._token = token.strip()
            except Exception as e:
                return json.dumps({"ok": False, "error": f"could not save config: {e}"})
            return json.dumps({"ok": True, "message": "Notion configured."})

        def clear_notion_auth() -> str:
            """Clear the stored Notion configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._token = ""
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("notion")
            return "Notion configuration cleared."

        return [check_notion_auth, authenticate_notion, clear_notion_auth]


def main() -> None:
    """Run the NotionAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): Notion's REST API has
    no inbound message stream to poll.
    """
    channel_main(
        NotionAgent,
        "kiss-notion",
        channel_name="Notion",
        make_backend=None,
    )


def tools() -> list:
    """Return the Notion channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return NotionAgent()._get_tools()


if __name__ == "__main__":
    main()
