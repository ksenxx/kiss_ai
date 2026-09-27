# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Minimal socket.io v0.9 client for Overleaf's real-time editor service.

Overleaf's web routes expose file *paths* (``/project/<id>/entities``)
but not the entity ids needed to download, rename, move, or delete a
single file.  The editor gets those ids from the ``joinProjectResponse``
event its socket.io connection receives right after connecting with a
``projectId`` query parameter.  This module performs that handshake
with the browser session cookie and returns the project (including the
``rootFolder`` tree).
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import quote

import requests
from websockets.sync.client import connect


def fetch_file_tree(
    session: requests.Session, host: str, project_id: str, timeout: float
) -> dict[str, Any]:
    """Join a project over socket.io and return its project document.

    Args:
        session: A ``requests`` session carrying the Overleaf cookies.
        host: Overleaf base URL, e.g. ``https://www.overleaf.com``.
        project_id: The project id.
        timeout: Seconds to wait for the handshake and each frame.

    Returns:
        The ``project`` dict of the ``joinProjectResponse`` event; its
        ``rootFolder[0]`` is the root folder of the file tree.

    Raises:
        PermissionError: The handshake was refused for the session (HTTP
            401 or a redirect to ``/login``).
        requests.HTTPError: The socket.io handshake failed otherwise.
        RuntimeError: The server sent a socket.io error frame.
        TimeoutError: No frame arrived within *timeout* seconds.
    """
    resp = session.get(
        f"{host}/socket.io/1/",
        params={"projectId": project_id, "t": str(int(time.time() * 1000))},
        timeout=timeout,
        allow_redirects=False,
    )
    status = resp.status_code
    if status == 401 or (status in (302, 303) and "/login" in resp.headers.get("Location", "")):
        raise PermissionError(f"Overleaf real-time handshake refused the session (HTTP {status})")
    resp.raise_for_status()
    sid = resp.text.split(":", 1)[0]
    ws_url = (
        "ws"
        + host.removeprefix("http")
        + f"/socket.io/1/websocket/{sid}?projectId={quote(project_id, safe='')}"
    )
    cookie = "; ".join(f"{c.name}={c.value}" for c in session.cookies)
    with connect(ws_url, additional_headers={"Cookie": cookie}, open_timeout=timeout) as ws:
        while True:
            frame = ws.recv(timeout=timeout, decode=True)
            assert isinstance(frame, str)
            if frame.startswith("7:"):
                raise RuntimeError(f"Overleaf real-time error: {frame}")
            if frame.startswith("2:"):
                ws.send("2::")
            elif frame.startswith("5:"):
                # socket.io v0.9 frame: "<type>:<id>:<endpoint>:<json>".
                event = json.loads(frame.split(":", 3)[3])
                if event.get("name") == "joinProjectResponse":
                    project: dict[str, Any] = event["args"][0]["project"]
                    return project


def flatten_tree(folder: dict[str, Any], prefix: str = "") -> list[dict[str, str]]:
    """Flatten an Overleaf folder tree into path entries.

    Args:
        folder: A folder dict with ``_id``, ``docs``, ``fileRefs`` and
            ``folders`` lists (e.g. ``project["rootFolder"][0]``).
        prefix: Path prefix of *folder* (``""`` for the root folder).

    Returns:
        One ``{"path", "type", "id", "parent_folder_id"}`` dict per doc,
        file, and folder below *folder* (the folder itself excluded);
        ``type`` is ``"doc"``, ``"file"``, or ``"folder"`` and paths
        are ``/``-separated and relative to the root.
    """
    entries: list[dict[str, str]] = []
    for kind, key in (("doc", "docs"), ("file", "fileRefs"), ("folder", "folders")):
        for item in folder.get(key, []):
            path = prefix + item["name"]
            entries.append(
                {"path": path, "type": kind, "id": item["_id"], "parent_folder_id": folder["_id"]}
            )
            if kind == "folder":
                entries.extend(flatten_tree(item, path + "/"))
    return entries
