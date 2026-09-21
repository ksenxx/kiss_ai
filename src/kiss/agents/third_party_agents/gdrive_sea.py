# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Google Drive Agent — channel agent for the Google Drive REST API.

Provides authenticated access to Google Drive via OAuth2 using plain
REST calls against ``https://www.googleapis.com/drive/v3`` (endpoint
paths per https://developers.google.com/drive/api/reference/rest/v3)
and multipart uploads against
``https://www.googleapis.com/upload/drive/v3``.  Credentials are
handled by the shared Google Workspace OAuth helpers and persisted
under ``~/.kiss/third_party_agents/google_drive/token.json``.

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

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    fresh_access_token,
    google_api_session,
    load_google_credentials,
    make_google_auth_tools,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_SERVICE = "google_drive"
_SCOPES = ["https://www.googleapis.com/auth/drive"]

_FILE_FIELDS = "id,name,mimeType,size,modifiedTime,webViewLink,parents"
# Google documents multipart uploads only for files up to 5 MB
# (https://developers.google.com/drive/api/guides/manage-uploads).
_MULTIPART_UPLOAD_LIMIT = 5 * 1024 * 1024
_GOOGLE_APPS_PREFIX = "application/vnd.google-apps"
_FOLDER_MIME = "application/vnd.google-apps.folder"
_SPREADSHEET_MIME = "application/vnd.google-apps.spreadsheet"


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

    Talks to the Drive v3 API over HTTP with an OAuth2 bearer token.
    Outbound-only: there is no inbound message stream over plain REST.
    """

    def __init__(self) -> None:
        self._creds: Any = None
        self._http: Any = google_api_session(_SERVICE)
        self._token: str = ""
        self._base_url: str = "https://www.googleapis.com/drive/v3"
        self._upload_base_url: str = "https://www.googleapis.com/upload/drive/v3"
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load stored Google Drive OAuth2 credentials from disk.

        Returns:
            True if valid credentials were loaded.
        """
        self._creds = load_google_credentials(_SERVICE, _SCOPES)
        if self._creds is None:
            self._connection_info = "No Google Drive credentials found."
            return False
        self._connection_info = "Google Drive credentials loaded."
        return True

    def _headers(self) -> dict[str, str]:
        """Return the Authorization header for an API request.

        Returns:
            Header dict with the bearer token (the direct test override
            ``_token`` wins over the stored credentials).
        """
        return {"Authorization": f"Bearer {self._token or fresh_access_token(self._creds)}"}

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
            headers=self._headers(),
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
            headers=self._headers(),
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
                headers=self._headers(),
                params={"mimeType": export_mime},
                timeout=_TIMEOUT,
            )
        else:
            content_resp = self._http.get(
                self._base_url.rstrip("/") + file_path,
                headers=self._headers(),
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
            data = source.read_bytes()
            file_name = name or source.name
            media_type = mime_type or (
                mimetypes.guess_type(file_name)[0] or "application/octet-stream"
            )
            metadata: dict[str, Any] = {"name": file_name}
            if folder_id:
                metadata["parents"] = [folder_id]
            # Drive requires an RFC 2387 multipart/related body (JSON
            # metadata part first, then the media part), which requests'
            # files= (multipart/form-data) cannot produce — build it by hand.
            boundary = uuid.uuid4().hex
            body = (
                (
                    f"--{boundary}\r\n"
                    "Content-Type: application/json; charset=UTF-8\r\n\r\n"
                    f"{json.dumps(metadata)}\r\n"
                    f"--{boundary}\r\n"
                    f"Content-Type: {media_type}\r\n\r\n"
                ).encode()
                + data
                + f"\r\n--{boundary}--\r\n".encode()
            )
            headers = self._headers()
            headers["Content-Type"] = f"multipart/related; boundary={boundary}"
            resp = self._http.post(
                self._upload_base_url.rstrip("/") + "/files",
                headers=headers,
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

    channel_system_prompt = (
        "\n\n## Google Drive Authentication\n"
        "Always call check_google_drive_auth() first; if it returns ok, report "
        "that Google Drive credentials are configured and stop — never start "
        "an OAuth flow over valid credentials. If credentials.json is missing, call "
        "start_google_drive_browser_setup() to create an OAuth Desktop-app "
        "client in Google Cloud Console; if credentials.json exists, call "
        "authenticate_google_drive() directly.\n"
        "When authenticate_google_drive() returns status 'consent_required' "
        "with an auth_url, do NOT open the auth_url or any accounts.google.com "
        "page in your own browser, and never ask for or type the user's Google "
        "password or 2FA code: Google sign-in pages are often blocked in the "
        "built-in browser (net::ERR_FAILED), and the sign-in belongs to the "
        "user. Hand off consent instead:\n"
        "1. Call ask_user_question() with the full auth_url, asking the user "
        "to open it in their OWN browser, approve access, and paste back the "
        "complete redirect URL from the address bar (it looks like "
        "http://localhost:PORT/?state=...&code=... and shows a connection "
        "error page — that is expected).\n"
        "2. The loopback consent server runs on THIS machine: deliver the "
        "pasted URL to it with Bash: curl -s '<pasted redirect URL>' (quote "
        "the URL; it contains & characters).\n"
        "3. Call finish_google_drive_auth(); if it returns 'pending', wait 2 "
        "seconds and call it once more.\n"
        "If any browser navigation to a Google page fails, do not retry it or "
        "relaunch the browser — switch to this hand-off immediately. Finish "
        "by verifying with check_google_drive_auth()."
    )

    def __init__(self) -> None:
        super().__init__("Google Drive Agent")
        self._backend = GoogleDriveChannelBackend()
        self._backend._creds = load_google_credentials(_SERVICE, _SCOPES)

    def _is_authenticated(self) -> bool:
        """Return True if the backend has credentials or a direct token."""
        return bool(self._backend._creds is not None or self._backend._token)

    def _get_auth_tools(self) -> list:
        """Return the standard Google OAuth tool set for Drive."""
        backend = self._backend

        def on_credentials(creds: Any) -> None:
            """Wire new (or cleared) OAuth credentials into the backend.

            Args:
                creds: New credentials, or None after clearing.
            """
            backend._creds = creds

        return make_google_auth_tools(
            self, _SERVICE, "Google Drive", _SCOPES, on_credentials=on_credentials
        )


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


def tools() -> list:
    """Return the Google Drive channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return GoogleDriveAgent()._get_tools()


if __name__ == "__main__":
    main()
