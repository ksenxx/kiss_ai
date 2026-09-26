# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Overleaf Agent — channel agent for overleaf.com (and Overleaf Server Pro / CE).

Overleaf is not an OAuth provider and has no public REST API, and its
``POST /login`` is CAPTCHA-protected.  This agent therefore drives the
same web routes the Overleaf editor uses (defined in the open-source
``overleaf/overleaf`` repository) with the user's browser session
cookie ``overleaf_session2``: the user signs in in their own browser by
any method (password, Google, ORCID, IEEE, institutional SSO) and pastes
the cookie back.  Mutating requests carry the session's CSRF token in
``x-csrf-token``.  File-tree entity ids come from the editor's socket.io
connection (:mod:`._overleaf_realtime`).  Optionally a Git
authentication token enables the official Git bridge (a premium
feature).  Config lives in ``~/.kiss/third_party_agents/overleaf/config.json``.

Overleaf has no inbound message stream, so ``--channel`` poll mode is
disabled (``main`` passes ``make_backend=None`` to ``channel_main``).

Usage::

    agent = OverleafAgent()
    agent.run(prompt_template="List my Overleaf projects")
"""

from __future__ import annotations

import base64
import json
import os
import posixpath
import subprocess
from html.parser import HTMLParser
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
from kiss.agents.third_party_agents._overleaf_realtime import fetch_file_tree, flatten_tree

_TIMEOUT = 60
_MAX_OUTPUT = 8000
_LOG_TAIL = 4000
_DEFAULT_HOST = "https://www.overleaf.com"
_COOKIE_NAME = "overleaf_session2"
_PRIVILEGES = ("readOnly", "readAndWrite", "review")
_EXPIRED = (
    "Overleaf session expired or invalid; run authenticate_overleaf with a "
    "fresh overleaf_session2 cookie"
)
_NO_GIT_TOKEN = json.dumps(
    {
        "ok": False,
        "error": "git_token not configured: create a Git authentication token at "
        "https://www.overleaf.com/user/settings (Git integration section; premium "
        "feature) and pass it to authenticate_overleaf(session_cookie=..., git_token=...)",
    }
)
# action -> (HTTP method, path template) for overleaf_set_project_state.
_STATE_ACTIONS = {
    "archive": ("POST", "/Project/{}/archive"),
    "unarchive": ("DELETE", "/Project/{}/archive"),
    "trash": ("POST", "/project/{}/trash"),
    "untrash": ("DELETE", "/project/{}/trash"),
    "delete": ("DELETE", "/Project/{}"),
    "restore": ("POST", "/Project/{}/restore"),
}

_OVERLEAF_DIR = Path.home() / ".kiss" / "third_party_agents" / "overleaf"
_config = ChannelConfig(_OVERLEAF_DIR, ("session_cookie",))


def _error(message: str) -> str:
    """Return an ``{"ok": false, "error": message}`` JSON string."""
    return json.dumps({"ok": False, "error": message})


def _ok(data: Any) -> str:
    """Return an ``{"ok": true, "result": data}`` JSON string capped at 8000 chars."""
    return json.dumps({"ok": True, "result": data})[:_MAX_OUTPUT]


def _text_result(path: str, text: str) -> str:
    """Render a file read as JSON of at most 8000 characters.

    JSON escaping can expand a character up to six-fold, so when the
    whole text does not fit, the longest strictly shorter content prefix
    whose *serialized* result (with ``truncated: true``) fits is found by
    binary search (a prefix never needs more than 8000 characters).

    Args:
        path: The normalised project-relative path that was read.
        text: The decoded file content.

    Returns:
        ``{"ok": true, "path", "truncated", "content"}`` as a JSON string,
        or an error if even the empty-content result is too long.
    """
    out = {"ok": True, "path": path, "truncated": False, "content": text}
    if text and len(json.dumps(out, ensure_ascii=False)) > _MAX_OUTPUT:
        low, high = 0, min(len(text) - 1, _MAX_OUTPUT)
        while low < high:
            mid = (low + high + 1) // 2
            cut = {**out, "truncated": True, "content": text[:mid]}
            if len(json.dumps(cut, ensure_ascii=False)) <= _MAX_OUTPUT:
                low = mid
            else:
                high = mid - 1
        out.update(truncated=True, content=text[:low])
    result = json.dumps(out, ensure_ascii=False)
    if len(result) > _MAX_OUTPUT:
        return _error("the file path is too long to return within the 8000-character limit")
    return result


def _seg(value: str) -> str:
    """Percent-encode *value* as a single URL path segment."""
    return quote(value, safe="")


def _normalize_cookie(value: str) -> str:
    """Extract the ``overleaf_session2`` value from a pasted cookie.

    Accepts the bare value or ``overleaf_session2=<value>`` (optionally
    followed by ``; other=cookie``), with surrounding whitespace/quotes.

    Args:
        value: The text the user pasted.

    Returns:
        The bare cookie value ("" when nothing usable was pasted).
    """
    cookie = value.strip().strip("'\"").split(";")[0].strip()
    cookie = cookie.removeprefix(f"{_COOKIE_NAME}=")
    return cookie.strip().strip("'\"")


class _MetaParser(HTMLParser):
    """Collect ``<meta name=... content=...>`` pairs from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.metas: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record the content of every named meta tag."""
        values = dict(attrs)
        if tag == "meta" and values.get("name"):
            self.metas[str(values["name"])] = values.get("content") or ""


def _meta_content(html: str, name: str) -> str | None:
    """Return the (unescaped) ``content`` of ``<meta name="name">`` in *html*.

    Args:
        html: An HTML document.
        name: The meta tag name, e.g. ``"ol-csrfToken"``.

    Returns:
        The content attribute value, or None when the tag is absent.
    """
    parser = _MetaParser()
    parser.feed(html)
    return parser.metas.get(name)


def _condense_project(project: dict[str, Any], tag_names: list[str]) -> dict[str, Any]:
    """Condense a dashboard project entry to its useful fields.

    Args:
        project: One entry of the ``ol-prefetchedProjectsBlob`` list.
        tag_names: Names of the tags that contain the project.

    Returns:
        Dict with id, name, lastUpdated, accessLevel, owner, archived,
        trashed, source, and tags.
    """
    owner = project.get("owner") or {}
    owner_name = f"{owner.get('firstName', '')} {owner.get('lastName', '')}".strip()
    return {
        "id": project.get("id", ""),
        "name": project.get("name", ""),
        "lastUpdated": project.get("lastUpdated", ""),
        "accessLevel": project.get("accessLevel", ""),
        "owner": {"email": owner.get("email", ""), "name": owner_name},
        "archived": bool(project.get("archived")),
        "trashed": bool(project.get("trashed")),
        "source": project.get("source", ""),
        "tags": tag_names,
    }


class OverleafChannelBackend(ToolMethodBackend):
    """Channel backend for Overleaf's web routes and Git bridge.

    Authenticates with the user's ``overleaf_session2`` browser cookie;
    outbound-only (Overleaf has no inbound message stream).
    """

    def __init__(self) -> None:
        self._configure("")

    def _configure(
        self, session_cookie: str, git_token: str = "", host: str = _DEFAULT_HOST
    ) -> None:
        """Reset the HTTP session for a cookie, Git token, and host.

        Args:
            session_cookie: The (normalised) ``overleaf_session2`` value;
                "" leaves the backend unauthenticated.
            git_token: Optional Git authentication token.
            host: Overleaf base URL.
        """
        self._host = host.strip().rstrip("/") or _DEFAULT_HOST
        self._cookie = session_cookie
        self._git_token = git_token.strip()
        self._csrf = ""
        self._session = requests.Session()
        if session_cookie:
            # No domain restriction: the session only talks to self._host, and
            # the cookie jar never sends a domain-scoped cookie to "localhost".
            self._session.cookies.set(_COOKIE_NAME, session_cookie)

    def connect(self) -> bool:
        """Load the Overleaf config from disk.

        Returns:
            True if a config with a ``session_cookie`` was loaded.
        """
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No Overleaf config found."
            return False
        self._configure(
            cfg["session_cookie"], cfg.get("git_token", ""), cfg.get("host", _DEFAULT_HOST)
        )
        self._connection_info = f"Overleaf session configured for {self._host}."
        return True

    def _csrf_token(self) -> str:
        """Return the session's CSRF token, fetching it once.

        Tries ``GET /dev/csrf`` (plain text) first and falls back to the
        ``ol-csrfToken`` meta tag of the project dashboard.

        Returns:
            The token ("" when neither source yields one).
        """
        if not self._csrf:
            resp = self._session.get(
                f"{self._host}/dev/csrf",
                headers={"Accept": "text/plain"},
                allow_redirects=False,
                timeout=_TIMEOUT,
            )
            token = resp.text.strip() if resp.status_code == 200 else ""
            if not token:
                page = self._session.get(
                    f"{self._host}/project",
                    headers={"Accept": "text/html"},
                    allow_redirects=False,
                    timeout=_TIMEOUT,
                )
                token = _meta_content(page.text, "ol-csrfToken") or ""
            self._csrf = token
        return self._csrf

    def _send_once(
        self, method: str, path: str, headers: dict[str, str], body: dict[str, Any]
    ) -> requests.Response:
        """Send one request, adding the CSRF token to mutating methods.

        Args:
            method: HTTP method.
            path: Path relative to the host.
            headers: Base request headers.
            body: ``json`` / ``params`` / ``data`` / ``files`` arguments.

        Returns:
            The raw response (redirects are not followed).
        """
        if method not in ("GET", "HEAD"):
            headers = {**headers, "x-csrf-token": self._csrf_token()}
        return self._session.request(
            method,
            self._host + path,
            headers=headers,
            allow_redirects=False,
            timeout=_TIMEOUT,
            **body,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, Any] | None = None,
        accept: str = "application/json",
    ) -> tuple[requests.Response | None, str]:
        """Issue an authenticated request to the Overleaf web app.

        Mutating requests carry ``x-csrf-token``; a 403 on them refreshes
        the token and retries once.  Redirects are not followed so a
        bounce to ``/login`` is reported as an expired session.

        Args:
            method: HTTP method.
            path: Path relative to the host, e.g. ``"/user/personal_info"``.
            json: Optional JSON body.
            params: Optional query parameters.
            data: Optional form fields (multipart when *files* is given).
            files: Optional multipart files.
            accept: ``Accept`` header value.

        Returns:
            ``(response, "")`` on success, or ``(None, error_json)``.
        """
        if any(part in (".", "..") for part in path.split("/")):
            return None, _error(f"invalid path segment in {path!r}")
        headers = {"Accept": accept, "Referer": f"{self._host}/project"}
        body = {"json": json, "params": params, "data": data, "files": files}
        try:
            resp = self._send_once(method, path, headers, body)
            if resp.status_code == 403 and method not in ("GET", "HEAD"):
                self._csrf = ""
                resp = self._send_once(method, path, headers, body)
        except Exception as e:
            return None, _error(str(e))
        status = resp.status_code
        if status == 401 or (status in (302, 303) and "/login" in resp.headers.get("Location", "")):
            return None, _error(_EXPIRED)
        if status == 429:
            return None, _error("HTTP 429: Overleaf rate limit hit; wait a minute and retry")
        if status >= 300:
            return None, _error(f"HTTP {status}: {resp.text[:500]}")
        return resp, ""

    def _json_call(self, method: str, path: str, **kwargs: Any) -> tuple[Any, str]:
        """Issue a request and parse its JSON body (``{}`` when empty).

        Args:
            method: HTTP method.
            path: Path relative to the host.
            **kwargs: Extra :meth:`_request` keyword arguments.

        Returns:
            ``(data, "")`` on success, or ``(None, error_json)``.
        """
        resp, err = self._request(method, path, **kwargs)
        if resp is None:
            return None, err
        if not resp.content:
            return {}, ""
        try:
            return resp.json(), ""
        except ValueError:
            return None, _error("non-JSON response from Overleaf")

    def _call(self, method: str, path: str, **kwargs: Any) -> str:
        """Issue a JSON request and render the tool result string.

        Args:
            method: HTTP method.
            path: Path relative to the host.
            **kwargs: Extra :meth:`_request` keyword arguments.

        Returns:
            ``{"ok": true, "result": ...}`` or an error JSON string.
        """
        data, err = self._json_call(method, path, **kwargs)
        return err or _ok(data)

    def _ack(self, method: str, path: str, **kwargs: Any) -> str:
        """Issue a request whose success body carries no data (e.g. plain ``OK``).

        Args:
            method: HTTP method.
            path: Path relative to the host.
            **kwargs: Extra :meth:`_request` keyword arguments.

        Returns:
            ``{"ok": true, "status": <HTTP status>}`` or an error JSON string.
        """
        resp, err = self._request(method, path, **kwargs)
        if resp is None:
            return err
        return json.dumps({"ok": True, "status": resp.status_code})

    def _save(self, path: str, output_path: str, **kwargs: Any) -> str:
        """Download *path* and write the bytes to *output_path*.

        Args:
            path: Path relative to the host.
            output_path: Local destination file.
            **kwargs: Extra :meth:`_request` keyword arguments.

        Returns:
            JSON string with ok status, path, and byte count.
        """
        resp, err = self._request("GET", path, accept="*/*", **kwargs)
        if resp is None:
            return err
        try:
            Path(output_path).expanduser().write_bytes(resp.content)
        except OSError as e:
            return _error(f"could not write {output_path}: {e}")
        return json.dumps({"ok": True, "path": output_path, "bytes": len(resp.content)})

    def _tree(self, project_id: str) -> tuple[str, list[dict[str, str]]]:
        """Fetch a project's file tree over the real-time connection.

        Args:
            project_id: The project id.

        Returns:
            ``(root_folder_id, entries)`` as built by ``flatten_tree``.

        Raises:
            PermissionError: The session cookie was rejected (message
                tells the user to run authenticate_overleaf).
        """
        try:
            project = fetch_file_tree(self._session, self._host, project_id, _TIMEOUT)
        except PermissionError:
            raise PermissionError(_EXPIRED) from None
        root = project["rootFolder"][0]
        return root["_id"], flatten_tree(root)

    def _resolve(self, project_id: str, path: str) -> tuple[str, str]:
        """Resolve a project-relative path to ``(entity_type, entity_id)``.

        Args:
            project_id: The project id.
            path: ``/``-separated path such as ``chapters/intro.tex``.

        Returns:
            ``(type, id)`` with type ``"doc"``, ``"file"``, or ``"folder"``.

        Raises:
            LookupError: The path is empty or does not exist.
        """
        clean = path.strip().strip("/")
        for entry in self._tree(project_id)[1]:
            if clean and entry["path"] == clean:
                return entry["type"], entry["id"]
        raise LookupError(f"no such file or folder in project: {path!r}")

    def _ensure_folder(self, project_id: str, folder_path: str) -> str:
        """Return the id of *folder_path*, creating missing folders.

        Args:
            project_id: The project id.
            folder_path: ``/``-separated folder path ("" is the root).

        Returns:
            The folder id.

        Raises:
            RuntimeError: Overleaf refused to create a folder.
        """
        root_id, entries = self._tree(project_id)
        folders = {e["path"]: e["id"] for e in entries if e["type"] == "folder"}
        folder_id = root_id
        parts = [p for p in folder_path.split("/") if p]
        for i, name in enumerate(parts):
            current = "/".join(parts[: i + 1])
            if current not in folders:
                data, err = self._json_call(
                    "POST",
                    f"/project/{_seg(project_id)}/folder",
                    json={"name": name, "parent_folder_id": folder_id},
                )
                if err:
                    raise RuntimeError(json.loads(err)["error"])
                folders[current] = data["_id"]
            folder_id = folders[current]
        return folder_id

    def _entity_bytes(self, project_id: str, path: str) -> tuple[bytes | None, str]:
        """Download the content of a doc or binary file by path.

        Args:
            project_id: The project id.
            path: Project-relative path of a doc or file.

        Returns:
            ``(content, "")`` on success, or ``(None, error_json)``.
        """
        kind, entity_id = self._resolve(project_id, path)
        pid, eid = _seg(project_id), _seg(entity_id)
        if kind == "folder":
            return None, _error(f"{path!r} is a folder; use overleaf_list_files")
        url = (
            f"/Project/{pid}/doc/{eid}/download" if kind == "doc" else f"/Project/{pid}/file/{eid}"
        )
        resp, err = self._request("GET", url, accept="*/*")
        return (None, err) if resp is None else (resp.content, "")

    def _upload(self, project_id: str, folder_id: str, name: str, content: bytes, mime: str) -> str:
        """Upload *content* as *name* into a folder (replacing a same-named entity).

        Args:
            project_id: The project id.
            folder_id: Destination folder id.
            name: File name.
            content: File bytes.
            mime: MIME type of the upload.

        Returns:
            Tool result JSON string with the entity id and type.
        """
        return self._call(
            "POST",
            f"/Project/{_seg(project_id)}/upload",
            params={"folder_id": folder_id},
            data={"name": name, "relativePath": "null", "type": mime},
            files={"qqfile": (name, content, mime)},
        )

    def _compile(self, project_id: str, draft: bool, stop_on_first_error: bool) -> tuple[Any, str]:
        """Run a compile and return Overleaf's JSON response.

        Args:
            project_id: The project id.
            draft: Compile in draft mode.
            stop_on_first_error: Stop at the first LaTeX error.

        Returns:
            ``(data, "")`` with ``status`` and ``outputFiles``, or
            ``(None, error_json)``.
        """
        return self._json_call(
            "POST",
            f"/project/{_seg(project_id)}/compile",
            params={"enable_pdf_caching": "true"},
            json={
                "check": "silent",
                "draft": draft,
                "incrementalCompilesEnabled": True,
                "rootDoc_id": "",
                "stopOnFirstError": stop_on_first_error,
            },
        )

    # ----------------------------------------------------------------- account

    def overleaf_whoami(self) -> str:
        """Get the signed-in Overleaf user's profile and plan features.

        Returns:
            JSON string with ok status and the user (id, email,
            first_name, last_name, ...) with plan limits under
            ``features`` (collaborators, compileTimeout, gitBridge, ...).
        """
        info, err = self._json_call("GET", "/user/personal_info")
        if err:
            return err
        features, err = self._json_call("GET", "/user/features")
        if err:
            return err
        return _ok({**info, "features": features})

    # ---------------------------------------------------------------- projects

    def overleaf_list_projects(
        self, include_archived: bool = False, include_trashed: bool = False, tag: str = ""
    ) -> str:
        """List the user's Overleaf projects from the project dashboard.

        Args:
            include_archived: Also list archived projects.
            include_trashed: Also list trashed projects.
            tag: Only list projects in the tag with this name.

        Returns:
            JSON string with ok status, ``totalSize`` reported by
            Overleaf, and projects (id, name, lastUpdated, accessLevel,
            owner email/name, archived, trashed, source, tags).
        """
        resp, err = self._request("GET", "/project", accept="text/html")
        if resp is None:
            return err
        blob = _meta_content(resp.text, "ol-prefetchedProjectsBlob")
        if blob is None:
            return _error(_EXPIRED)
        projects = json.loads(blob).get("projects", [])
        tags = json.loads(_meta_content(resp.text, "ol-tags") or "[]")
        result = []
        for project in projects:
            names = [t["name"] for t in tags if project.get("id") in t.get("project_ids", [])]
            item = _condense_project(project, names)
            if (
                (include_archived or not item["archived"])
                and (include_trashed or not item["trashed"])
                and (not tag or tag in names)
            ):
                result.append(item)
        out = {"ok": True, "totalSize": json.loads(blob).get("totalSize"), "projects": result}
        return json.dumps(out)[:_MAX_OUTPUT]

    def overleaf_create_project(self, name: str, template: str = "none") -> str:
        """Create a new Overleaf project.

        Args:
            name: The project name.
            template: ``"none"`` for a blank project or ``"example"`` for
                Overleaf's example project.

        Returns:
            JSON string with ok status and ``project_id``.
        """
        return self._call("POST", "/project/new", json={"projectName": name, "template": template})

    def overleaf_upload_project_zip(self, name: str, zip_path: str) -> str:
        """Create a new project from a local zip archive.

        Args:
            name: The new project's name.
            zip_path: Local path of the ``.zip`` file.

        Returns:
            JSON string with ok status and the new ``project_id``.
        """
        try:
            content = Path(zip_path).expanduser().read_bytes()
        except OSError as e:
            return _error(f"could not read {zip_path}: {e}")
        return self._call(
            "POST",
            "/project/new/upload",
            data={"name": name},
            files={"qqfile": (f"{name}.zip", content, "application/zip")},
        )

    def overleaf_rename_project(self, project_id: str, new_name: str) -> str:
        """Rename a project.

        Args:
            project_id: The project id.
            new_name: The new project name.

        Returns:
            JSON string with ok status and the HTTP status code.
        """
        return self._ack(
            "POST", f"/project/{_seg(project_id)}/rename", json={"newProjectName": new_name}
        )

    def overleaf_clone_project(self, project_id: str, new_name: str) -> str:
        """Copy a project into a new project owned by the user.

        Args:
            project_id: The project to copy.
            new_name: Name of the copy.

        Returns:
            JSON string with ok status and the new project's details.
        """
        return self._call(
            "POST", f"/Project/{_seg(project_id)}/clone", json={"projectName": new_name}
        )

    def overleaf_set_project_state(self, project_id: str, action: str) -> str:
        """Archive, trash, restore, or permanently delete a project.

        Args:
            project_id: The project id.
            action: One of ``archive``, ``unarchive``, ``trash``,
                ``untrash``, ``restore`` (undelete), or ``delete``.
                ``delete`` PERMANENTLY deletes the project for everyone
                and needs explicit user confirmation.

        Returns:
            JSON string with ok status and the HTTP status code.
        """
        if action not in _STATE_ACTIONS:
            return _error(f"action must be one of {', '.join(_STATE_ACTIONS)}")
        method, template = _STATE_ACTIONS[action]
        return self._ack(method, template.format(_seg(project_id)))

    def overleaf_update_project_settings(
        self,
        project_id: str,
        compiler: str = "",
        root_doc_id: str = "",
        spell_check_language: str = "",
        image_name: str = "",
    ) -> str:
        """Change a project's compiler, main document, spell-check language, or TeX Live image.

        Args:
            project_id: The project id.
            compiler: ``pdflatex``, ``latex``, ``xelatex``, or ``lualatex``.
            root_doc_id: Id of the main ``.tex`` doc (see overleaf_list_files).
            spell_check_language: Language code such as ``en`` ("" keeps it).
            image_name: TeX Live image name (Server Pro / premium).

        Returns:
            JSON string with ok status.
        """
        values = {
            "compiler": compiler,
            "rootDocId": root_doc_id,
            "spellCheckLanguage": spell_check_language,
            "imageName": image_name,
        }
        body = {key: value for key, value in values.items() if value}
        if not body:
            return _error("nothing to update: pass at least one setting")
        return self._call("POST", f"/project/{_seg(project_id)}/settings", json=body)

    def overleaf_download_project_zip(self, project_id: str, output_path: str) -> str:
        """Download a whole project as a zip archive.

        Args:
            project_id: The project id.
            output_path: Local file to write the zip to.

        Returns:
            JSON string with ok status, path, and byte count.
        """
        return self._save(f"/Project/{_seg(project_id)}/download/zip", output_path)

    # ------------------------------------------------------------------- files

    def overleaf_list_files(self, project_id: str) -> str:
        """List every doc, file, and folder of a project with entity ids.

        Args:
            project_id: The project id.

        Returns:
            JSON string with ok status, ``root_folder_id``, and entries
            (path, type ``doc``/``file``/``folder``, id).  If the
            real-time connection fails, falls back to paths only.
        """
        try:
            root_id, entries = self._tree(project_id)
        except PermissionError as e:
            return _error(str(e))
        except Exception as e:
            data, err = self._json_call("GET", f"/project/{_seg(project_id)}/entities")
            if err:
                return err
            note = f"entity ids unavailable (real-time connection failed: {e})"
            return _ok({"note": note, "entities": data.get("entities", [])})
        files = [{"path": e["path"], "type": e["type"], "id": e["id"]} for e in entries]
        return _ok({"root_folder_id": root_id, "entries": files})

    def overleaf_read_file(self, project_id: str, path: str) -> str:
        """Read a text file (``.tex``, ``.bib``, ...) of a project.

        Args:
            project_id: The project id.
            path: Project-relative path such as ``chapters/intro.tex``.

        Returns:
            JSON string with ok status and the content, cut so the whole
            result fits in 8000 characters (``truncated`` tells whether
            content was cut), or an error saying a binary file must be
            downloaded with overleaf_download_file.
        """
        try:
            content, err = self._entity_bytes(project_id, path)
            if content is None:
                return err
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                return _error(
                    f"{path!r} is binary ({len(content)} bytes); use overleaf_download_file"
                )
            return _text_result(path.strip().strip("/"), text)
        except Exception as e:
            return _error(str(e))

    def overleaf_download_file(self, project_id: str, path: str, output_path: str) -> str:
        """Save one project file (text or binary) to a local path.

        Args:
            project_id: The project id.
            path: Project-relative path of the file.
            output_path: Local destination file.

        Returns:
            JSON string with ok status, output path, and byte count.
        """
        try:
            content, err = self._entity_bytes(project_id, path)
            if content is None:
                return err
            Path(output_path).expanduser().write_bytes(content)
            return json.dumps({"ok": True, "path": output_path, "bytes": len(content)})
        except Exception as e:
            return _error(str(e))

    def overleaf_write_file(self, project_id: str, path: str, content: str) -> str:
        """Create or replace a text file in a project (missing folders are created).

        Args:
            project_id: The project id.
            path: Project-relative path such as ``chapters/intro.tex``.
            content: The complete new file content.

        Returns:
            JSON string with ok status and the entity id and type.
        """
        folder, name = posixpath.split(path.strip().lstrip("/"))
        if not name:
            return _error("path must name a file")
        try:
            folder_id = self._ensure_folder(project_id, folder)
            return self._upload(project_id, folder_id, name, content.encode("utf-8"), "text/plain")
        except Exception as e:
            return _error(str(e))

    def overleaf_upload_file(self, project_id: str, local_path: str, folder_path: str = "") -> str:
        """Upload a local file (image, PDF, ``.bib``, ...) into a project folder.

        Args:
            project_id: The project id.
            local_path: Path of the local file; its base name is kept.
            folder_path: Destination folder ("" is the root; missing
                folders are created).  A same-named file is replaced.

        Returns:
            JSON string with ok status and the entity id and type.
        """
        try:
            local = Path(local_path).expanduser()
            content = local.read_bytes()
            folder_id = self._ensure_folder(project_id, folder_path)
            return self._upload(
                project_id, folder_id, local.name, content, "application/octet-stream"
            )
        except Exception as e:
            return _error(str(e))

    def overleaf_create_folder(self, project_id: str, path: str) -> str:
        """Create a folder (and any missing parents) in a project.

        Args:
            project_id: The project id.
            path: Project-relative folder path such as ``figures/plots``.

        Returns:
            JSON string with ok status and the folder id.
        """
        try:
            return json.dumps({"ok": True, "folder_id": self._ensure_folder(project_id, path)})
        except Exception as e:
            return _error(str(e))

    def overleaf_rename_entity(self, project_id: str, path: str, new_name: str) -> str:
        """Rename a doc, file, or folder in place.

        Args:
            project_id: The project id.
            path: Project-relative path of the entity.
            new_name: New base name (no ``/``).

        Returns:
            JSON string with ok status.
        """
        try:
            kind, entity_id = self._resolve(project_id, path)
            return self._call(
                "POST",
                f"/project/{_seg(project_id)}/{kind}/{_seg(entity_id)}/rename",
                json={"name": new_name},
            )
        except Exception as e:
            return _error(str(e))

    def overleaf_move_entity(self, project_id: str, path: str, new_folder_path: str) -> str:
        """Move a doc, file, or folder into another folder.

        Args:
            project_id: The project id.
            path: Project-relative path of the entity.
            new_folder_path: Destination folder path ("" is the root;
                missing folders are created).

        Returns:
            JSON string with ok status.
        """
        try:
            kind, entity_id = self._resolve(project_id, path)
            folder_id = self._ensure_folder(project_id, new_folder_path)
            return self._call(
                "POST",
                f"/project/{_seg(project_id)}/{kind}/{_seg(entity_id)}/move",
                json={"folder_id": folder_id},
            )
        except Exception as e:
            return _error(str(e))

    def overleaf_delete_entity(self, project_id: str, path: str) -> str:
        """Delete a doc, file, or folder (with its contents) from a project.

        Destructive: ask the user for explicit confirmation first.

        Args:
            project_id: The project id.
            path: Project-relative path of the entity.

        Returns:
            JSON string with ok status.
        """
        try:
            kind, entity_id = self._resolve(project_id, path)
            return self._call("DELETE", f"/project/{_seg(project_id)}/{kind}/{_seg(entity_id)}")
        except Exception as e:
            return _error(str(e))

    # ----------------------------------------------------------------- compile

    def overleaf_compile(
        self, project_id: str, draft: bool = False, stop_on_first_error: bool = False
    ) -> str:
        """Compile a project and report the status, output files, and log tail.

        Args:
            project_id: The project id.
            draft: Compile in draft mode (faster, images omitted).
            stop_on_first_error: Stop at the first LaTeX error.

        Returns:
            JSON string with ok status, compile ``status`` (``success``,
            ``failure``, ``timedout``, ...), outputFiles (path, type,
            url, build), and the last 4000 characters of ``output.log``.
        """
        data, err = self._compile(project_id, draft, stop_on_first_error)
        if err:
            return err
        outputs = [
            {key: f.get(key) for key in ("path", "type", "url", "build")}
            for f in data.get("outputFiles", [])
        ]
        result: dict[str, Any] = {"ok": True, "status": data.get("status"), "outputFiles": outputs}
        for output in outputs:
            if output["path"] == "output.log":
                resp, log_err = self._request("GET", output["url"], accept="*/*")
                result["log_tail"] = log_err if resp is None else resp.text[-_LOG_TAIL:]
        return json.dumps(result)[:_MAX_OUTPUT]

    def overleaf_download_pdf(self, project_id: str, output_path: str) -> str:
        """Compile a project and save the resulting PDF.

        Args:
            project_id: The project id.
            output_path: Local file to write the PDF to.

        Returns:
            JSON string with ok status, path, and byte count, or the
            compile status when no PDF was produced.
        """
        data, err = self._compile(project_id, False, False)
        if err:
            return err
        for output in data.get("outputFiles", []):
            if output.get("type") == "pdf":
                return self._save(output["url"], output_path)
        return _error(f"compile produced no PDF (status: {data.get('status')})")

    def overleaf_word_count(self, project_id: str, file: str = "") -> str:
        """Count the words of a project (or one of its ``.tex`` files).

        Args:
            project_id: The project id.
            file: Optional project-relative ``.tex`` path; "" counts the
                main document.

        Returns:
            JSON string with ok status and Overleaf's texcount result.
        """
        params = {"file": file} if file else None
        return self._call("GET", f"/project/{_seg(project_id)}/wordcount", params=params)

    def overleaf_clear_compile_cache(self, project_id: str) -> str:
        """Delete a project's cached compile output (fixes stale-build errors).

        Args:
            project_id: The project id.

        Returns:
            JSON string with ok status and the HTTP status code.
        """
        return self._ack("DELETE", f"/project/{_seg(project_id)}/output")

    # ----------------------------------------------------------- collaboration

    def overleaf_list_members(self, project_id: str) -> str:
        """List a project's collaborators and (for the owner) pending invites.

        Args:
            project_id: The project id.

        Returns:
            JSON string with ok status, members (_id, email, names,
            privileges), and invites (or ``"unavailable: ..."`` when the
            user may not see them).
        """
        members, err = self._json_call("GET", f"/project/{_seg(project_id)}/members")
        if err:
            return err
        invites, inv_err = self._json_call("GET", f"/project/{_seg(project_id)}/invites")
        out = {
            "ok": True,
            "members": members.get("members", []),
            "invites": f"unavailable: {inv_err}" if inv_err else invites.get("invites", []),
        }
        return json.dumps(out)[:_MAX_OUTPUT]

    def overleaf_invite_collaborator(
        self, project_id: str, email: str, privileges: str = "readAndWrite"
    ) -> str:
        """Invite someone by email to collaborate on a project.

        Args:
            project_id: The project id.
            email: The invitee's email address.
            privileges: ``readOnly``, ``readAndWrite``, or ``review``.

        Returns:
            JSON string with ok status and the invite.
        """
        if privileges not in _PRIVILEGES:
            return _error(f"privileges must be one of {', '.join(_PRIVILEGES)}")
        return self._call(
            "POST",
            f"/project/{_seg(project_id)}/invite",
            json={"email": email, "privileges": privileges},
        )

    def overleaf_set_collaborator_privileges(
        self, project_id: str, user_id: str, privileges: str
    ) -> str:
        """Change a collaborator's access level.

        Args:
            project_id: The project id.
            user_id: The collaborator's user id (from overleaf_list_members).
            privileges: ``readOnly``, ``readAndWrite``, or ``review``.

        Returns:
            JSON string with ok status.
        """
        if privileges not in _PRIVILEGES:
            return _error(f"privileges must be one of {', '.join(_PRIVILEGES)}")
        return self._call(
            "PUT",
            f"/project/{_seg(project_id)}/users/{_seg(user_id)}",
            json={"privilegeLevel": privileges},
        )

    def overleaf_remove_collaborator(self, project_id: str, user_id: str) -> str:
        """Remove a collaborator from a project.

        Args:
            project_id: The project id.
            user_id: The collaborator's user id.

        Returns:
            JSON string with ok status.
        """
        return self._call("DELETE", f"/project/{_seg(project_id)}/users/{_seg(user_id)}")

    def overleaf_revoke_invite(self, project_id: str, invite_id: str) -> str:
        """Revoke a pending collaboration invite.

        Args:
            project_id: The project id.
            invite_id: The invite id (from overleaf_list_members).

        Returns:
            JSON string with ok status.
        """
        return self._call("DELETE", f"/project/{_seg(project_id)}/invite/{_seg(invite_id)}")

    def overleaf_get_sharing_links(self, project_id: str) -> str:
        """Get a project's link-sharing URLs (read-only and read-write).

        Args:
            project_id: The project id.

        Returns:
            JSON string with ok status and ``read_only_url`` /
            ``read_write_url`` for whichever tokens exist.
        """
        tokens, err = self._json_call("GET", f"/project/{_seg(project_id)}/tokens")
        if err:
            return err
        links: dict[str, Any] = {"ok": True}
        if tokens.get("readOnly"):
            links["read_only_url"] = f"{self._host}/read/{tokens['readOnly']}"
        if tokens.get("readAndWrite"):
            links["read_write_url"] = f"{self._host}/{tokens['readAndWrite']}"
        return json.dumps(links)

    def overleaf_leave_project(self, project_id: str) -> str:
        """Leave a project someone else shared with the user.

        Args:
            project_id: The project id.

        Returns:
            JSON string with ok status.
        """
        return self._call("POST", f"/project/{_seg(project_id)}/leave")

    def overleaf_transfer_ownership(self, project_id: str, user_id: str) -> str:
        """Make a collaborator the project owner (the user becomes an editor).

        Destructive: ask the user for explicit confirmation first.

        Args:
            project_id: The project id.
            user_id: The collaborator who becomes the owner.

        Returns:
            JSON string with ok status.
        """
        return self._call(
            "POST", f"/project/{_seg(project_id)}/transfer-ownership", json={"user_id": user_id}
        )

    # -------------------------------------------------------------------- chat

    def overleaf_get_chat_messages(self, project_id: str, limit: int = 50, before: str = "") -> str:
        """Read a project's chat messages, newest first.

        Args:
            project_id: The project id.
            limit: Maximum messages to return (default 50).
            before: Timestamp (ms) to page back from; "" for the latest.

        Returns:
            JSON string with ok status and messages (id, content,
            timestamp, user).
        """
        params = {"limit": str(limit)}
        if before:
            params["before"] = before
        return self._call("GET", f"/project/{_seg(project_id)}/messages", params=params)

    def overleaf_send_chat_message(self, project_id: str, content: str) -> str:
        """Post a message to a project's chat.

        Args:
            project_id: The project id.
            content: The message text.

        Returns:
            JSON string with ok status.
        """
        if not content.strip():
            return _error("content cannot be empty")
        return self._call(
            "POST", f"/project/{_seg(project_id)}/messages", json={"content": content}
        )

    # -------------------------------------------------------------------- tags

    def overleaf_list_tags(self) -> str:
        """List the user's project tags (dashboard folders).

        Returns:
            JSON string with ok status and tags (_id, name, color,
            project_ids).
        """
        return self._call("GET", "/tag")

    def overleaf_create_tag(self, name: str, color: str = "") -> str:
        """Create a project tag.

        Args:
            name: The tag name.
            color: Optional hex color such as ``#43A7F0``.

        Returns:
            JSON string with ok status and the tag.
        """
        body = {"name": name, "color": color} if color else {"name": name}
        return self._call("POST", "/tag", json=body)

    def overleaf_edit_tag(self, tag_id: str, name: str, color: str = "") -> str:
        """Rename a tag and optionally change its color.

        Args:
            tag_id: The tag id.
            name: The new tag name.
            color: Optional new hex color.

        Returns:
            JSON string with ok status.
        """
        body = {"name": name, "color": color} if color else {"name": name}
        return self._call("POST", f"/tag/{_seg(tag_id)}/edit", json=body)

    def overleaf_delete_tag(self, tag_id: str) -> str:
        """Delete a tag (its projects are kept).

        Args:
            tag_id: The tag id.

        Returns:
            JSON string with ok status.
        """
        return self._call("DELETE", f"/tag/{_seg(tag_id)}")

    def overleaf_tag_project(self, tag_id: str, project_id: str, remove: bool = False) -> str:
        """Add a project to a tag, or remove it from the tag.

        Args:
            tag_id: The tag id.
            project_id: The project id.
            remove: Remove the project from the tag instead of adding it.

        Returns:
            JSON string with ok status.
        """
        method = "DELETE" if remove else "POST"
        return self._call(method, f"/tag/{_seg(tag_id)}/project/{_seg(project_id)}")

    # ----------------------------------------------------------------- history

    def overleaf_get_history(self, project_id: str, min_count: int = 25, before: str = "") -> str:
        """List a project's history updates, newest first.

        Args:
            project_id: The project id.
            min_count: Minimum number of updates to return (default 25).
            before: ``nextBeforeTimestamp`` from a previous call to page back.

        Returns:
            JSON string with ok status and updates (fromV, toV, meta
            users/timestamps, labels, pathnames) plus nextBeforeTimestamp.
        """
        params = {"min_count": str(min_count)}
        if before:
            params["before"] = before
        return self._call("GET", f"/project/{_seg(project_id)}/updates", params=params)

    def overleaf_list_labels(self, project_id: str) -> str:
        """List a project's labelled history versions.

        Args:
            project_id: The project id.

        Returns:
            JSON string with ok status and labels (id, comment, version,
            user_id, created_at).
        """
        return self._call("GET", f"/project/{_seg(project_id)}/labels")

    def overleaf_create_label(self, project_id: str, version: int, comment: str) -> str:
        """Label a history version (like a named snapshot).

        Args:
            project_id: The project id.
            version: The history version number (``toV`` of an update).
            comment: The label text.

        Returns:
            JSON string with ok status and the label.
        """
        return self._call(
            "POST",
            f"/project/{_seg(project_id)}/labels",
            json={"comment": comment, "version": version},
        )

    def overleaf_delete_label(self, project_id: str, label_id: str) -> str:
        """Delete a history label.

        Args:
            project_id: The project id.
            label_id: The label id.

        Returns:
            JSON string with ok status.
        """
        return self._call("DELETE", f"/project/{_seg(project_id)}/labels/{_seg(label_id)}")

    def overleaf_get_diff(
        self, project_id: str, pathname: str, from_version: int, to_version: int
    ) -> str:
        """Show how one file changed between two history versions.

        Args:
            project_id: The project id.
            pathname: Project-relative file path.
            from_version: Older version number.
            to_version: Newer version number.

        Returns:
            JSON string with ok status and the diff (text chunks marked
            ``i`` inserted / ``d`` deleted / unchanged).
        """
        params = {"pathname": pathname, "from": str(from_version), "to": str(to_version)}
        return self._call("GET", f"/project/{_seg(project_id)}/diff", params=params)

    def overleaf_download_version_zip(self, project_id: str, version: int, output_path: str) -> str:
        """Download a past history version of a project as a zip.

        Args:
            project_id: The project id.
            version: The history version number.
            output_path: Local file to write the zip to.

        Returns:
            JSON string with ok status, path, and byte count.
        """
        return self._save(f"/project/{_seg(project_id)}/version/{version}/zip", output_path)

    def overleaf_restore_file(self, project_id: str, pathname: str, version: int) -> str:
        """Restore one file to its content at a history version.

        Args:
            project_id: The project id.
            pathname: Project-relative file path.
            version: The history version to restore from.

        Returns:
            JSON string with ok status.
        """
        return self._call(
            "POST",
            f"/project/{_seg(project_id)}/restore_file",
            json={"version": version, "pathname": pathname},
        )

    def overleaf_revert_project(self, project_id: str, version: int) -> str:
        """Revert the whole project to a history version.

        Destructive: ask the user for explicit confirmation first.

        Args:
            project_id: The project id.
            version: The history version to revert to.

        Returns:
            JSON string with ok status.
        """
        return self._call(
            "POST", f"/project/{_seg(project_id)}/revert-project", json={"version": version}
        )

    # ----------------------------------------------------------- notifications

    def overleaf_list_notifications(self) -> str:
        """List the user's Overleaf notifications (invites, system messages).

        Returns:
            JSON string with ok status and notifications.
        """
        return self._call("GET", "/notifications")

    # --------------------------------------------------------------------- git

    def _git_remote_url(self, project_id: str) -> str:
        """Return the Git bridge URL of a project for the configured host."""
        if self._host in (_DEFAULT_HOST, "https://overleaf.com"):
            return f"https://git.overleaf.com/{_seg(project_id)}"
        return f"{self._host}/git/{_seg(project_id)}"

    def _git(self, args: list[str], cwd: str) -> tuple[int, str]:
        """Run git with the Git token as a Basic auth header.

        The header is passed through ``GIT_CONFIG_*`` environment
        variables, so the token never lands in ``.git/config``, the
        remote URL, or the process arguments.

        Args:
            args: git arguments, e.g. ``["push"]``.
            cwd: Working directory.

        Returns:
            ``(returncode, combined stdout+stderr)``.
        """
        auth = base64.b64encode(f"git:{self._git_token}".encode()).decode()
        env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraHeader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
        }
        proc = subprocess.run(
            ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()

    def overleaf_git_clone(self, project_id: str, dest_dir: str) -> str:
        """Clone a project through the Overleaf Git bridge (premium feature).

        Args:
            project_id: The project id.
            dest_dir: Local directory to clone into (must not exist yet).

        Returns:
            JSON string with ok status and git's output.
        """
        if not self._git_token:
            return _NO_GIT_TOKEN
        try:
            dest = str(Path(dest_dir).expanduser())
            rc, out = self._git(["clone", self._git_remote_url(project_id), dest], ".")
            return json.dumps({"ok": rc == 0, "path": dest, "output": out[-_MAX_OUTPUT:]})
        except Exception as e:
            return _error(str(e))

    def overleaf_git_sync(self, repo_dir: str, commit_message: str = "") -> str:
        """Commit local changes in a Git-bridge clone, pull (rebase), and push.

        Args:
            repo_dir: A directory created by overleaf_git_clone.
            commit_message: Commit message (default "Update from KISS Sorcar").

        Returns:
            JSON string with ok status and each step's return code and output.
        """
        if not self._git_token:
            return _NO_GIT_TOKEN
        try:
            cwd = str(Path(repo_dir).expanduser())
            rc, out = self._git(["add", "-A"], cwd)
            steps = [{"step": "add", "rc": rc, "output": out}]
            if rc == 0 and self._git(["diff", "--cached", "--quiet"], cwd)[0] != 0:
                message = commit_message or "Update from KISS Sorcar"
                rc, out = self._git(["commit", "-m", message], cwd)
                steps.append({"step": "commit", "rc": rc, "output": out})
            for args in (["pull", "--rebase"], ["push"]):
                if rc != 0:
                    break
                rc, out = self._git(args, cwd)
                steps.append({"step": args[0], "rc": rc, "output": out})
            return json.dumps({"ok": rc == 0, "steps": steps})[:_MAX_OUTPUT]
        except Exception as e:
            return _error(str(e))


class OverleafAgent(BaseChannelAgent):
    """Channel agent with Overleaf project, file, compile, sharing, and history tools."""

    _backend: OverleafChannelBackend

    channel_system_prompt = (
        "You are operating the user's Overleaf account through their browser "
        "session cookie (Overleaf has no OAuth or public API). Tools: "
        "overleaf_whoami; projects (overleaf_list_projects, "
        "overleaf_create_project, overleaf_upload_project_zip, "
        "overleaf_rename_project, overleaf_clone_project, "
        "overleaf_set_project_state, overleaf_update_project_settings, "
        "overleaf_download_project_zip); files by project-relative path "
        "(overleaf_list_files, overleaf_read_file, overleaf_write_file, "
        "overleaf_upload_file, overleaf_download_file, overleaf_create_folder, "
        "overleaf_rename_entity, overleaf_move_entity, overleaf_delete_entity); "
        "compiling (overleaf_compile, overleaf_download_pdf, overleaf_word_count, "
        "overleaf_clear_compile_cache); sharing (overleaf_list_members, "
        "overleaf_invite_collaborator, overleaf_set_collaborator_privileges, "
        "overleaf_remove_collaborator, overleaf_revoke_invite, "
        "overleaf_get_sharing_links, overleaf_leave_project, "
        "overleaf_transfer_ownership); chat, tags, history (versions, labels, "
        "diffs, restore, revert), notifications; and the Git bridge "
        "(overleaf_git_clone, overleaf_git_sync), which needs a premium plan "
        "and a Git token. overleaf_set_project_state with action 'delete', "
        "overleaf_delete_entity, overleaf_revert_project and "
        "overleaf_transfer_ownership are destructive: get explicit user "
        "confirmation before calling them."
    )

    def __init__(self) -> None:
        super().__init__("Overleaf Agent")
        self._backend = OverleafChannelBackend()
        self._backend.connect()

    def _is_authenticated(self) -> bool:
        """Return True if an Overleaf session cookie is configured."""
        return bool(self._backend._cookie)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_overleaf_auth() -> str:
            """Check whether the Overleaf session is configured and still valid.

            Returns:
                The signed-in user's email and name, an expired-session
                error, or step-by-step instructions to authenticate.
            """
            if not agent._is_authenticated():
                return (
                    "Not configured for Overleaf. Overleaf has no OAuth or public "
                    "API, so the agent uses your browser session:\n"
                    "1. Sign in at https://www.overleaf.com/login in your own "
                    "browser (any method: email/password, Google, ORCID, IEEE, or "
                    "institutional SSO).\n"
                    "2. Open DevTools -> Application (Chrome) or Storage (Firefox) "
                    "-> Cookies -> https://www.overleaf.com and copy the value of "
                    "the overleaf_session2 cookie.\n"
                    "3. Optional: create a Git authentication token at "
                    "https://www.overleaf.com/user/settings (Git integration "
                    "section, premium plans) for overleaf_git_clone / overleaf_git_sync.\n"
                    "4. Call authenticate_overleaf(session_cookie=..., git_token=...).\n"
                    "The cookie grants full access to your Overleaf account and "
                    "stops working when you log out of that browser session."
                    + "\n"
                    + portal_handoff("https://www.overleaf.com/login")
                )
            info, err = agent._backend._json_call("GET", "/user/personal_info")
            if err:
                return err
            name = f"{info.get('first_name', '')} {info.get('last_name', '')}".strip()
            return json.dumps(
                {
                    "ok": True,
                    "email": info.get("email", ""),
                    "name": name,
                    "host": agent._backend._host,
                }
            )

        def authenticate_overleaf(
            session_cookie: str, git_token: str = "", host: str = _DEFAULT_HOST
        ) -> str:
            """Verify and store the Overleaf session cookie (and optional Git token).

            Args:
                session_cookie: Value of the ``overleaf_session2`` cookie
                    (a pasted ``overleaf_session2=...`` is accepted).
                git_token: Optional Git authentication token for the Git bridge.
                host: Overleaf base URL; change it only for Server Pro /
                    Community Edition installations.

            Returns:
                Configuration result or error message.
            """
            cookie = _normalize_cookie(session_cookie)
            if not cookie:
                return _error("session_cookie cannot be empty")
            base = host.strip().rstrip("/")
            if not base.startswith(("http://", "https://")):
                return _error("host must start with http:// or https://")
            candidate = OverleafChannelBackend()
            candidate._configure(cookie, git_token, base)
            info, err = candidate._json_call("GET", "/user/personal_info")
            if err:
                return err
            if not isinstance(info, dict) or not (info.get("email") or info.get("id")):
                return _error("Overleaf did not return a user profile; cookie not saved")
            try:
                _config.save(
                    {"session_cookie": cookie, "git_token": git_token.strip(), "host": base}
                )
            except Exception as e:
                return _error(f"could not save config: {e}")
            agent._backend._configure(cookie, git_token, base)
            return json.dumps(
                {"ok": True, "message": f"Overleaf configured for {info.get('email', '')}."}
            )

        def clear_overleaf_auth() -> str:
            """Clear the stored Overleaf session cookie and Git token.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._configure("")
            return "Overleaf configuration cleared."

        return [check_overleaf_auth, authenticate_overleaf, clear_overleaf_auth]


def main() -> None:
    """Run the OverleafAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): Overleaf has no
    inbound message stream to poll.
    """
    channel_main(OverleafAgent, "kiss-overleaf", channel_name="Overleaf", make_backend=None)


def tools() -> list:
    """Return the Overleaf channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return OverleafAgent()._get_tools()


if __name__ == "__main__":
    main()
