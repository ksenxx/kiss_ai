# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Google Drive Agent — channel agent for the Google Drive REST API.

Provides access to Google Drive using plain REST calls against
``https://www.googleapis.com/drive/v3`` (endpoint paths per
https://developers.google.com/drive/api/reference/rest/v3).  Sign-in and
every API call go through Composio (see :mod:`._composio_google`), which
holds the user's Google token; multipart uploads travel as the proxy's
binary body.

The Drive REST API has no inbound message stream, so this adapter is
outbound-only: ``main`` passes ``make_backend=None`` to ``channel_main``
so the ``--channel`` poll mode is disabled.

Usage::

    agent = GoogleDriveAgent()
    agent.run(prompt_template="Find my spreadsheets modified this week")
"""

from __future__ import annotations

import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from kiss.agents.seas.base.base_sea import BaseSea
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

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_SERVICE = "google_drive"

_FILE_FIELDS = "id,name,mimeType,size,modifiedTime,webViewLink,parents"
_MULTIPART_UPLOAD_LIMIT = 5 * 1024 * 1024
_GOOGLE_APPS_PREFIX = "application/vnd.google-apps"
_FOLDER_MIME = "application/vnd.google-apps.folder"
_SPREADSHEET_MIME = "application/vnd.google-apps.spreadsheet"


class GdriveSea(BaseSea):
    """The ``/gdrive`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/gdrive help``."""
        return (
            "Lists, searches, reads, uploads, moves and shares files in the user's Google Drive "
            "through the Drive v3 REST API with sign-in and calls proxied by Composio; use it with "
            '`run_agent(agent="gdrive", task="Find my spreadsheets modified this week")` or the '
            "`kiss-gdrive -t '<task>'` CLI (outbound-only, no --channel poll mode)."
        )

    def tools(self, tools: list[Any]) -> list[Any]:
        """Return the Google Drive channel tools (the SEA ``tools`` method).

        Called by the kiss-web daemon when this module's path is passed as
        the API's ``extension_agent_path``: builds a fresh agent from the
        Composio connection recorded under ``~/.kiss`` and returns its
        authentication and backend tools.
        """
        return tools + GoogleDriveAgent()._get_tools()

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Run as a ``channel`` worker (``kiss.server.sorcar.run`` agent-script contract).

        No git lifecycle, nothing inherited from the calling task, the
        channel preamble in the system prompt (see
        :mod:`kiss.agents.sorcar.sea_settings`).
        """
        return settings | {"kind": "channel"}

    def system_prompt(self, system_prompt: str) -> str:
        """Return the channel guidance appended to the run's system prompt."""
        return system_prompt + "\n\n" + GoogleDriveAgent.channel_system_prompt


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


def _http_error(resp: requests.Response) -> str:
    """Format an HTTP error response as an ok:false JSON string.

    Args:
        resp: The error response (status >= 400).

    Returns:
        ``{"ok": false, "error": "HTTP <status>: <body>"}`` JSON string.
    """
    return json.dumps({"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"})


class GoogleDriveChannelBackend(ToolMethodBackend):
    """Channel backend for the Google Drive REST API.

    Talks to the Drive v3 API through Composio's proxy, which signs
    each request with the user's Google token.  Outbound-only: there is
    no inbound message stream over plain REST.
    """

    def __init__(self) -> None:
        self._http: Any = ComposioSession(_SERVICE)
        self._base_url: str = "https://www.googleapis.com/drive/v3"
        self._upload_base_url: str = "https://www.googleapis.com/upload/drive/v3"
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Check that Google Drive is connected through Composio.

        Returns:
            True if a Composio connection exists.
        """
        if not connected_account_id(_SERVICE):
            self._connection_info = "Google Drive is not connected."
            return False
        self._connection_info = "Google Drive connected through Composio."
        return True

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Issue an authenticated Drive API request expecting a JSON reply.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, ``"PATCH"``).
            path: API path relative to the base URL, starting with ``/``.
            params: Optional query parameters.
            payload: Optional JSON body.

        Returns:
            JSON string ``{"ok": true, "result": ...}`` on success or
            ``{"ok": false, "error": ...}`` on an HTTP error status.
        """
        url = self._base_url.rstrip("/") + path
        resp = self._http.request(
            method,
            url,
            params=params,
            json=payload,
            timeout=_TIMEOUT,
        )
        if resp.status_code >= 400:
            return _http_error(resp)
        try:
            result: Any = resp.json()
        except ValueError:
            result = resp.text
        return json.dumps({"ok": True, "result": result})

    def _fetch_file_bytes(
        self, file_id: str, export_mime_type: str
    ) -> tuple[bytes, dict[str, Any], str]:
        """Fetch a file's content bytes, exporting Google Workspace docs.

        Google Workspace files (mimeType ``application/vnd.google-apps.*``)
        have no raw bytes and must be exported; regular files are
        downloaded with ``alt=media``.

        Args:
            file_id: Drive file ID (already validated as path-safe).
            export_mime_type: Export MIME type override for Google
                Workspace docs. When empty, ``text/csv`` is used for
                spreadsheets and ``text/plain`` for everything else.

        Returns:
            ``(data, metadata, error)`` — content bytes and file
            metadata on success (*error* empty), or an
            ``{"ok": false, ...}`` JSON string in *error* on failure.
        """
        file_path = f"/files/{quote(file_id, safe='')}"
        meta_resp = self._http.get(
            self._base_url.rstrip("/") + file_path,
            params={"fields": "id,name,mimeType"},
            timeout=_TIMEOUT,
        )
        if meta_resp.status_code >= 400:
            return b"", {}, _http_error(meta_resp)
        metadata: dict[str, Any] = meta_resp.json()
        mime = str(metadata.get("mimeType", ""))
        if mime.startswith(_GOOGLE_APPS_PREFIX):
            export_mime = export_mime_type or (
                "text/csv" if mime == _SPREADSHEET_MIME else "text/plain"
            )
            content_resp = self._http.get(
                self._base_url.rstrip("/") + file_path + "/export",
                    params={"mimeType": export_mime},
                timeout=_TIMEOUT,
            )
        else:
            content_resp = self._http.get(
                self._base_url.rstrip("/") + file_path,
                    params={"alt": "media"},
                timeout=_TIMEOUT,
            )
        if content_resp.status_code >= 400:
            return b"", {}, _http_error(content_resp)
        return content_resp.content, metadata, ""

    def gdrive_search_files(
        self,
        query: str = "",
        max_results: int = 20,
        page_token: str = "",
        order_by: str = "",
    ) -> str:
        """Search for files and folders in Google Drive.

        Args:
            query: Drive query string (https://developers.google.com/drive/api/guides/ref-search-terms),
                e.g. ``"name contains 'report'"``,
                ``"mimeType = 'application/vnd.google-apps.folder'"``,
                ``"'FOLDER_ID' in parents"``. Empty lists all files.
            max_results: Maximum number of files to return (default 20).
            page_token: Page token from a previous response. Optional.
            order_by: Sort keys such as ``"modifiedTime desc"`` or
                ``"name"``. Optional.

        Returns:
            JSON string with ok status, the files (id, name, mimeType,
            size, modifiedTime, webViewLink, parents), and
            ``next_page_token`` when more results exist.
        """
        try:
            params = {
                "pageSize": str(max_results),
                "fields": f"files({_FILE_FIELDS}),nextPageToken",
            }
            if query:
                params["q"] = query
            if page_token:
                params["pageToken"] = page_token
            if order_by:
                params["orderBy"] = order_by
            parsed = json.loads(self._request("GET", "/files", params=params))
            if not parsed.get("ok"):
                return json.dumps(parsed)
            result = parsed["result"] or {}
            out: dict[str, Any] = {"ok": True, "files": result.get("files", [])}
            if result.get("nextPageToken"):
                out["next_page_token"] = result["nextPageToken"]
            return json.dumps(out, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdrive_get_file(self, file_id: str) -> str:
        """Get a file's metadata by ID.

        Args:
            file_id: Drive file ID.

        Returns:
            JSON string with ok status and the file metadata (id, name,
            mimeType, size, modifiedTime, webViewLink, parents).
        """
        try:
            err = _bad_segment(file_id, "file_id")
            if err:
                return err
            path = f"/files/{quote(file_id, safe='')}"
            parsed = json.loads(self._request("GET", path, params={"fields": _FILE_FIELDS}))
            if not parsed.get("ok"):
                return json.dumps(parsed)
            return json.dumps({"ok": True, "file": parsed["result"]}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdrive_read_file(self, file_id: str, export_mime_type: str = "") -> str:
        """Read a file's content as text.

        Google Workspace docs are exported (default ``text/plain``, or
        ``text/csv`` for spreadsheets); regular files are downloaded
        directly.

        Args:
            file_id: Drive file ID.
            export_mime_type: Export MIME type override for Google
                Workspace docs (e.g. ``"text/csv"``). Optional.

        Returns:
            JSON string with ok status, file name, MIME type, and the
            content text (truncated to 8000 characters).
        """
        try:
            err = _bad_segment(file_id, "file_id")
            if err:
                return err
            data, metadata, fetch_err = self._fetch_file_bytes(file_id, export_mime_type)
            if fetch_err:
                return fetch_err
            return json.dumps(
                {
                    "ok": True,
                    "name": metadata.get("name", ""),
                    "mime_type": metadata.get("mimeType", ""),
                    "content": data.decode("utf-8", errors="replace")[:8000],
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdrive_download_file(
        self, file_id: str, save_path: str, export_mime_type: str = ""
    ) -> str:
        """Download a file's content to a local path.

        Google Workspace docs are exported (default ``text/plain``, or
        ``text/csv`` for spreadsheets); regular files are downloaded
        directly.

        Args:
            file_id: Drive file ID.
            save_path: Local filesystem path to write the bytes to.
            export_mime_type: Export MIME type override for Google
                Workspace docs (e.g. ``"application/pdf"``). Optional.

        Returns:
            JSON string with ok status, the saved path, and byte size.
        """
        try:
            err = _bad_segment(file_id, "file_id")
            if err:
                return err
            data, _, fetch_err = self._fetch_file_bytes(file_id, export_mime_type)
            if fetch_err:
                return fetch_err
            target = Path(save_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return json.dumps({"ok": True, "path": str(target), "size": len(data)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdrive_upload_file(
        self, local_path: str, name: str = "", folder_id: str = "", mime_type: str = ""
    ) -> str:
        """Upload a local file to Google Drive (multipart upload).

        Google documents multipart upload only for files up to 5 MB, so
        larger local files are rejected with an error.

        Args:
            local_path: Path of the local file to upload (at most 5 MB).
            name: Name for the file in Drive; defaults to the local
                file name. Optional.
            folder_id: Destination folder ID; defaults to My Drive root.
                Optional.
            mime_type: MIME type; guessed from the file name when empty.
                Optional.

        Returns:
            JSON string with ok status and the created file's metadata.
        """
        try:
            if folder_id:
                err = _bad_segment(folder_id, "folder_id")
                if err:
                    return err
            source = Path(local_path)
            if not source.is_file():
                return json.dumps({"ok": False, "error": f"local file not found: {local_path}"})
            if source.stat().st_size > _MULTIPART_UPLOAD_LIMIT:
                return json.dumps(
                    {"ok": False, "error": "file exceeds the 5 MB multipart upload limit"}
                )
            file_name = name or source.name
            media_type = mime_type or (
                mimetypes.guess_type(file_name)[0] or "application/octet-stream"
            )
            metadata: dict[str, Any] = {"name": file_name}
            if folder_id:
                metadata["parents"] = [folder_id]
            # Drive requires an RFC 2387 multipart/related body (JSON
            # metadata part first, then the media part); the proxy carries
            # it as a binary body.
            boundary = uuid.uuid4().hex
            body = (
                (
                    f"--{boundary}\r\n"
                    "Content-Type: application/json; charset=UTF-8\r\n\r\n"
                    f"{json.dumps(metadata)}\r\n"
                    f"--{boundary}\r\n"
                    f"Content-Type: {media_type}\r\n\r\n"
                ).encode()
                + source.read_bytes()
                + f"\r\n--{boundary}--\r\n".encode()
            )
            resp = self._http.post(
                self._upload_base_url + "/files",
                headers={"Content-Type": f"multipart/related; boundary={boundary}"},
                params={"uploadType": "multipart", "fields": _FILE_FIELDS},
                data=body,
                timeout=_TIMEOUT,
            )
            if resp.status_code >= 400:
                return _http_error(resp)
            return json.dumps({"ok": True, "file": resp.json()}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdrive_create_folder(self, name: str, parent_id: str = "") -> str:
        """Create a folder in Google Drive.

        Args:
            name: Folder name.
            parent_id: Parent folder ID; defaults to My Drive root.
                Optional.

        Returns:
            JSON string with ok status and the created folder's metadata.
        """
        try:
            if parent_id:
                err = _bad_segment(parent_id, "parent_id")
                if err:
                    return err
            body: dict[str, Any] = {"name": name, "mimeType": _FOLDER_MIME}
            if parent_id:
                body["parents"] = [parent_id]
            parsed = json.loads(
                self._request("POST", "/files", params={"fields": _FILE_FIELDS}, payload=body)
            )
            if not parsed.get("ok"):
                return json.dumps(parsed)
            return json.dumps({"ok": True, "folder": parsed["result"]})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdrive_share_file(self, file_id: str, email: str, role: str = "reader") -> str:
        """Share a file or folder with a user.

        Args:
            file_id: Drive file or folder ID to share.
            email: Email address of the user to share with.
            role: Permission role: ``"reader"``, ``"commenter"``,
                ``"writer"``, or ``"organizer"``. Default: ``"reader"``.

        Returns:
            JSON string with ok status and the created permission.
        """
        try:
            err = _bad_segment(file_id, "file_id")
            if err:
                return err
            path = f"/files/{quote(file_id, safe='')}/permissions"
            body = {"type": "user", "role": role, "emailAddress": email}
            parsed = json.loads(self._request("POST", path, payload=body))
            if not parsed.get("ok"):
                return json.dumps(parsed)
            return json.dumps({"ok": True, "permission": parsed["result"]})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdrive_move_file(self, file_id: str, folder_id: str) -> str:
        """Move a file to another folder.

        Fetches the file's current parents, then re-parents it in one
        PATCH with ``addParents``/``removeParents``.

        Args:
            file_id: Drive file ID to move.
            folder_id: Destination folder ID.

        Returns:
            JSON string with ok status and the moved file's metadata.
        """
        try:
            err = _bad_segment(file_id, "file_id") or _bad_segment(folder_id, "folder_id")
            if err:
                return err
            path = f"/files/{quote(file_id, safe='')}"
            parsed = json.loads(self._request("GET", path, params={"fields": "parents"}))
            if not parsed.get("ok"):
                return json.dumps(parsed)
            parents = (parsed["result"] or {}).get("parents", [])
            params = {"addParents": folder_id, "fields": _FILE_FIELDS}
            if parents:
                params["removeParents"] = ",".join(parents)
            moved = json.loads(self._request("PATCH", path, params=params, payload={}))
            if not moved.get("ok"):
                return json.dumps(moved)
            return json.dumps({"ok": True, "file": moved["result"]})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gdrive_trash_file(self, file_id: str) -> str:
        """Move a file to the trash.

        Args:
            file_id: Drive file ID to trash.

        Returns:
            JSON string with ok status.
        """
        try:
            err = _bad_segment(file_id, "file_id")
            if err:
                return err
            path = f"/files/{quote(file_id, safe='')}"
            parsed = json.loads(self._request("PATCH", path, payload={"trashed": True}))
            if not parsed.get("ok"):
                return json.dumps(parsed)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class GoogleDriveAgent(BaseChannelAgent):
    """Channel agent with Google Drive REST API tools."""

    channel_system_prompt = google_auth_prompt(_SERVICE, "Google Drive")

    def __init__(self) -> None:
        super().__init__("Google Drive Agent")
        self._backend = GoogleDriveChannelBackend()

    def _is_authenticated(self) -> bool:
        """Return True if Google Drive is connected through Composio."""
        return bool(connected_account_id(_SERVICE))

    def _get_auth_tools(self) -> list:
        """Return the Composio sign-in tool set for Google Drive."""
        return make_google_auth_tools(self, _SERVICE, "Google Drive", self._backend.connect)


def main() -> None:
    """Run the GoogleDriveAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): the Drive REST API
    has no inbound message stream to poll.
    """
    channel_main(
        GoogleDriveAgent,
        "kiss-gdrive",
        channel_name="Google Drive",
        make_backend=None,
    )


if __name__ == "__main__":
    main()
