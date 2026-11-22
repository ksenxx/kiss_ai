# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A local stand-in for skills.sh, GitHub's archive host and the model that judges a skill.

One HTTP server answers what the Skills panel asks of the two hosts
(``KISS_SKILLS_REGISTRY_URL`` and ``KISS_SKILLS_ARCHIVE_URL`` point at
it): the search, the all-time leaderboard, the audits of a skill and a
repository's tarball, each from what the test put into the
:class:`Registry`.  A path in ``replies`` gets that canned reply
instead: an HTTP error, a body that is not JSON, a redirect, a reply
held back until the test lets it go.  Given a directory for its
certificates, the server speaks HTTPS.

Its ``/v1`` is an OpenAI-compatible endpoint that answers every chat
completion with the scripted replies of ``Registry.judge``; a model
registered for it (:func:`register_judge`) and named by
``KISS_SKILL_JUDGE_MODEL`` is the judge of the skill check.
"""

from __future__ import annotations

import io
import json
import ssl
import tarfile
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from kiss.core.models.model_info import delete_custom_model, save_custom_model
from kiss.core.tls_certs import generate_local_ca, issue_server_cert, server_cert_names
from kiss.tests.agents.sorcar.local_model_server import text_body

JUDGE = "skill-judge"
"""The custom model :func:`register_judge` registers as the skill judge."""


@dataclass
class Reply:
    """One canned HTTP reply.

    Attributes:
        body: The body.
        status: The HTTP status.
        hold: When set, the reply is kept back until the event is set.
        location: The address a redirect (status 302) names.
        unsent: Bytes the reply claims after its body and never sends;
            the connection stays open until the client hangs up.
    """

    body: bytes = b""
    status: int = 200
    hold: threading.Event | None = None
    location: str = ""
    unsent: int = 0


@dataclass
class Registry:
    """What the stand-in serves, filled in by the test.

    Attributes:
        url: Its base URL.
        searches: Query -> the ``skills`` of ``/api/search``.
        leaderboard: The ``skills`` of ``/api/skills/all-time/0``.
        audits: Skill id -> the ``audits`` of its audit (HTTP 404 without one).
        archives: ``owner/repo`` -> its tarball (HTTP 404 without one).
        replies: Path with query -> the canned reply served instead.
        paths: Every path requested with GET, in order.
        judge: The model's replies in order, the last repeating; a
            judge without one finds nothing in any skill.
        judge_status: The HTTP status of the model's replies.
        judged: Every request the model received.
    """

    url: str = ""
    searches: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    audits: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    archives: dict[str, bytes] = field(default_factory=dict)
    replies: dict[str, Reply] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    judge: list[bytes] = field(default_factory=list)
    judge_status: int = 200
    judged: list[dict[str, Any]] = field(default_factory=list)


def _json(value: object) -> Reply:
    return Reply(json.dumps(value).encode())


class _RegistryHandler(BaseHTTPRequestHandler):
    """Answer a GET as skills.sh or codeload.github.com would, a POST as the judge."""

    registry = Registry()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the access log."""

    def _reply(self) -> Reply:
        registry = type(self).registry
        url = urlsplit(self.path)
        if self.path in registry.replies:
            return registry.replies[self.path]
        if url.path == "/api/search":
            query = parse_qs(url.query)
            found = registry.searches.get(query["q"][0], [])
            return _json({"skills": found[: int(query["limit"][0])]})
        if url.path == "/api/skills/all-time/0":
            return _json({"skills": registry.leaderboard, "hasMore": True})
        skill = url.path.removeprefix("/api/v1/skills/audit/")
        if skill in registry.audits:
            return _json({"audits": registry.audits[skill]})
        source = url.path.removesuffix("/tar.gz/HEAD").strip("/")
        if source in registry.archives:
            return Reply(registry.archives[source])
        return Reply(b'{"error":"not_found"}', 404)

    def _send(self, reply: Reply) -> None:
        if reply.hold is not None:
            reply.hold.wait(60)
        try:
            self.send_response(reply.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply.body) + reply.unsent))
            if reply.location:
                self.send_header("Location", reply.location)
            self.end_headers()
            self.wfile.write(reply.body)
            if reply.unsent:
                self.connection.settimeout(60)
                self.rfile.read(1)
        except OSError:
            pass  # the client hung up on a body it refuses

    def do_GET(self) -> None:  # noqa: N802
        type(self).registry.paths.append(self.path)
        self._send(self._reply())

    def do_POST(self) -> None:  # noqa: N802
        registry = type(self).registry
        registry.judged.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        script = registry.judge or [judge_reply()]
        body = script[min(len(registry.judged), len(script)) - 1]
        self._send(Reply(body, registry.judge_status))


@contextmanager
def serve_registry(tls: Path | None = None) -> Iterator[Registry]:
    """Serve an empty :class:`Registry` on a local port for the block's duration.

    Args:
        tls: An existing directory.  When given, the server speaks
            HTTPS, with a certificate of an authority whose own
            certificate it writes there as ``ca.pem``.
    """
    registry = Registry()
    handler = type("Handler", (_RegistryHandler,), {"registry": registry})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    scheme = "http"
    if tls is not None:
        authority = generate_local_ca(tls / "ca.pem", tls / "ca.key")
        issue_server_cert(tls / "cert.pem", tls / "key.pem", *authority, *server_cert_names())
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(tls / "cert.pem", tls / "key.pem")
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    # Polled often: the default half second per shutdown adds up over a module's tests.
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    registry.url = f"{scheme}://127.0.0.1:{server.server_port}"
    try:
        yield registry
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)


@contextmanager
def register_judge(registry: Registry, name: str = JUDGE) -> Iterator[None]:
    """Register the model of *registry* as *name* in this process's home for the block."""
    delete_custom_model(name)  # one a killed run left behind
    assert save_custom_model(name, endpoint=registry.url + "/v1", api_key="local") is None
    try:
        yield
    finally:
        assert delete_custom_model(name) is None


def skill_md(name: str, body: str = "", description: str = "Says hello to the user.") -> bytes:
    """Return a ``SKILL.md`` named *name* whose instructions are *body*."""
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n{body}".encode()


def tarball(files: dict[str, bytes], extra: Sequence[tarfile.TarInfo] = ()) -> bytes:
    """Return a repository tarball as GitHub makes one.

    Args:
        files: Path in the repository -> content; every path goes under
            the one top-level directory ``repo-main/``, a ``.sh`` file
            as an executable.
        extra: Members added as given, after the files and without
            content: links, devices, directories, paths outside the
            top-level directory, a file whose header claims a size.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(f"repo-main/{name}")
            info.size = len(data)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
        assert archive.fileobj is not None
        for info in extra:
            # The header alone: ``addfile`` refuses a file that claims content it is not given.
            archive.fileobj.write(info.tobuf(archive.format, archive.encoding, archive.errors))
    return buffer.getvalue()


def member(name: str, kind: bytes, target: str = "") -> tarfile.TarInfo:
    """Return a tarball member *name* that is no regular file (``tarfile.SYMTYPE``, ...)."""
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = target
    return info


def judge_reply(*findings: dict[str, Any]) -> bytes:
    """Return the model reply that reports *findings* (none: a harmless skill)."""
    answer = {"reviewed": True, "findings": list(findings), "summary": "Reviewed."}
    return text_body(json.dumps(answer))


def judge_refusal() -> bytes:
    """Return the reply of a model whose content filter refused to read the skill."""
    body = json.loads(text_body(""))
    body["choices"][0]["finish_reason"] = "content_filter"
    return json.dumps(body).encode()
