# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the imported `harness` and `recorder` fixtures are
#   intentionally shadowed by test parameters of the same name)
"""The daemon side of the "Open in Browser" button of an HTML content tab.

``openInBrowser`` has the daemon open the tab's file in the default
browser of its own machine; ``stageHtmlView`` uploads the document for
the browser tab the page opens on ``GET /view/<token>/<name>``
(:mod:`kiss.server.html_views`).  A real daemon throughout: commands
over ``wss://``, pages over ``https://``.  The opener honours
``$BROWSER``, so the launch tests point it at a script that records the
address it was given: a real launch, no mocks.

``openWithDefaultApp`` has the daemon open a file no tab can show with
the application of its type, through the platform opener, never
``$BROWSER``: its tests put a script of the opener's name first on
``PATH``.  Not covered, and why: the Windows opener (``os.startfile``,
``%PATHEXT%``) and the macOS rule for a file without an extension run
on those hosts only.

A request for a well-formed token that is not staged waits ten seconds
before its 404, so the tests that show a document is gone ask for all
of them at once.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from websockets.asyncio.client import connect

from kiss.core.brand import PRODUCT_NAME
from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import no_verify_ssl
from kiss.tests.server.test_content_tab_file_links import (
    _HTML_SOURCE,
    _ServerHarness,
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_open_external_command import (
    recorder,  # noqa: F401  (fixture used by param name)
)

SANDBOX = (
    "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox "
    "allow-forms allow-modals allow-downloads"
)
GONE = "This page is no longer available"


async def _exchange(
    harness: _ServerHarness, payloads: list[dict], reply_type: str, count: int,
) -> dict:
    """Send *payloads* on one connection; the *count* *reply_type* replies, by ``path``."""
    replies: dict = {}
    async with connect(harness.ws_url, ssl=no_verify_ssl()) as ws:
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        while json.loads(await asyncio.wait_for(ws.recv(), 30)).get("type") != "auth_ok":
            pass
        for payload in payloads:
            await ws.send(json.dumps(payload))
        while len(replies) < count:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == reply_type:
                replies[msg["path"]] = msg
    return replies


def _open_in_browser(harness: _ServerHarness, commands: list[dict]) -> dict:
    """Send ``openInBrowser`` *commands*; their replies, by the path each echoes."""
    payloads = [{"type": "openInBrowser", **command} for command in commands]
    return harness.run(_exchange(harness, payloads, "openInBrowserResult", len(payloads)))


def _stage(harness: _ServerHarness, documents: list[dict]) -> None:
    """Send ``stageHtmlView`` *documents* and return once the daemon has handled them.

    ``stageHtmlView`` has no reply, and a connection's commands are
    handled in order: the ``openFile`` sent after them is answered last.
    """
    payloads = [{"type": "stageHtmlView", **document} for document in documents]
    payloads.append({"type": "openFile", "path": str(harness.work_dir / "page.html")})
    harness.run(_exchange(harness, payloads, "fileContent", 1))


def _get(harness: _ServerHarness, path: str) -> tuple[int, dict[str, str], bytes, float]:
    """GET *path* from the daemon: status, headers, body and the seconds it took."""
    started = time.monotonic()
    try:
        with urllib.request.urlopen(
            harness.base_url + path, context=no_verify_ssl(), timeout=60,
        ) as response:
            status, headers, body = response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        status, headers, body = exc.code, dict(exc.headers), exc.read()
    return status, headers, body, time.monotonic() - started


def _token() -> str:
    """A token as the page mints it: 16 random bytes in hex."""
    return secrets.token_hex(16)


@posix_only("POSIX $BROWSER script")
def test_the_daemon_opens_the_html_file_in_its_default_browser(
    harness: _ServerHarness, recorder: Path,
) -> None:
    page = harness.work_dir / "page.html"
    spaced = harness.work_dir / "My Report.HTM"
    spaced.write_text(_HTML_SOURCE, encoding="utf-8")
    replies = _open_in_browser(harness, [
        {"path": "page.html", "workDir": str(harness.work_dir), "tabId": "t-1"},
        {"path": str(spaced), "token": "browser:7"},
    ])
    assert replies["page.html"] == {
        "type": "openInBrowserResult", "opened": True, "path": "page.html",
    }, "the reply echoes the path as sent"
    assert replies[str(spaced)] == {
        "type": "openInBrowserResult", "opened": True, "path": str(spaced),
        "token": "browser:7",
    }, "and the token of the click it answers, so the page finds the tab that asked"
    assert sorted(recorder.read_text(encoding="utf-8").splitlines()) == sorted(
        [page.as_uri(), spaced.as_uri()],
    ), "the browser is handed the file:// address of the resolved file"


@posix_only("POSIX $BROWSER script")
def test_only_an_existing_html_file_is_opened(harness: _ServerHarness, recorder: Path) -> None:
    (harness.work_dir / "folder.html").mkdir(exist_ok=True)
    # "" names the work dir itself; 7 is no path at all and reads as "".
    refused = ["missing.html", "sample.py", "notes.md", "folder.html", ""]
    commands = [{"path": path, "workDir": str(harness.work_dir)} for path in [*refused, 7]]
    replies = _open_in_browser(harness, commands[:-1]) | _open_in_browser(harness, commands[-1:])
    assert {path: reply["opened"] for path, reply in replies.items()} == dict.fromkeys(
        refused, False,
    )
    assert not recorder.exists(), "nothing was launched"


@posix_only("POSIX $BROWSER script")
def test_a_headless_daemon_opens_nothing(harness: _ServerHarness, recorder: Path) -> None:
    os.environ["KISS_HEADLESS"] = "1"
    path = str(harness.work_dir / "page.html")
    assert _open_in_browser(harness, [{"path": path}])[path]["opened"] is False
    assert not recorder.exists(), "nothing was launched"


@pytest.fixture
def opener(recorder: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The platform opener as a script, first on ``PATH``, that logs the file it is handed.

    Like ``xdg-open``, it fails with status 3 for a type no application
    opens (here ``.noapp``).
    """
    log = recorder.with_name("app-opened.log")
    script = recorder.with_name("open" if sys.platform == "darwin" else "xdg-open")
    script.write_text(
        f'#!/bin/sh\ncase "$1" in *.noapp) exit 3;; esac\nprintf \'%s\\n\' "$1" >> "{log}"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    return log


def _open_with_default_app(harness: _ServerHarness, commands: list[dict]) -> dict:
    """Send ``openWithDefaultApp`` *commands*; their replies, by the path each echoes."""
    payloads = [{"type": "openWithDefaultApp", **command} for command in commands]
    return harness.run(_exchange(harness, payloads, "openWithDefaultAppResult", len(payloads)))


@posix_only("POSIX opener script")
def test_the_daemon_opens_a_file_with_its_default_application(
    harness: _ServerHarness, recorder: Path, opener: Path,
) -> None:
    spaced = harness.work_dir / "My Report.docx"
    spaced.write_bytes(b"PK\x03\x04\x00")
    replies = _open_with_default_app(harness, [
        {"path": "binary.bin", "workDir": str(harness.work_dir), "tabId": "t-1"},
        {"path": str(spaced), "token": "app:7"},
    ])
    assert replies["binary.bin"] == {
        "type": "openWithDefaultAppResult", "opened": True, "path": "binary.bin",
    }, "the reply echoes the path as sent"
    assert replies[str(spaced)] == {
        "type": "openWithDefaultAppResult", "opened": True, "path": str(spaced),
        "token": "app:7",
    }, "and the token of the request it answers"
    assert sorted(opener.read_text(encoding="utf-8").splitlines()) == sorted(
        [str(harness.work_dir / "binary.bin"), str(spaced)],
    ), "the platform opener is handed the resolved file"
    assert not recorder.exists(), "a file is no page: $BROWSER is not asked"


@posix_only("POSIX opener script")
def test_a_program_is_never_started_nor_a_file_that_cannot_be_opened(
    harness: _ServerHarness, opener: Path,
) -> None:
    folder = harness.work_dir / "programs"
    folder.mkdir()
    try:
        # Native executables (ELF, PE, the five Mach-O forms) and a
        # script that names its interpreter, whatever their names.
        magics = [b"\x7fELF", b"MZ", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf"]
        magics += [b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"#!/bin/sh\n"]
        for number, magic in enumerate(magics):
            (folder / f"program{number}.dat").write_bytes(magic + b"\0\0\0\0")
        # Types a shell runs, by name alone; a type no application opens.
        named = ("App.JAR", "module.pyc", "setup.msi", "run.command", "menu.desktop")
        for name in (*named, "viewer.noapp"):
            (folder / name).write_bytes(b"PK\x03\x04\x00")
        # A link is the file it leads to, whatever the link is called.
        (folder / "notes.docx").symlink_to(folder / "run.command")
        (folder / "folder.bin").mkdir()
        if os.geteuid():  # root reads any file
            (folder / "unreadable.bin").write_bytes(b"\0")
            (folder / "unreadable.bin").chmod(0)
        refused = [*sorted(path.name for path in folder.iterdir()), "missing.bin", ""]
        replies = _open_with_default_app(
            harness, [{"path": path, "workDir": str(folder)} for path in refused],
        )
        assert {path: reply["opened"] for path, reply in replies.items()} == dict.fromkeys(
            refused, False,
        )
        assert not opener.exists(), "nothing was started"
    finally:
        shutil.rmtree(folder)


@posix_only("POSIX opener script")
def test_a_headless_daemon_starts_no_application(harness: _ServerHarness, opener: Path) -> None:
    os.environ["KISS_HEADLESS"] = "1"
    path = str(harness.work_dir / "binary.bin")
    assert _open_with_default_app(harness, [{"path": path}])[path]["opened"] is False
    assert not opener.exists(), "nothing was started"


def test_a_staged_document_is_served_sandboxed(harness: _ServerHarness) -> None:
    token = _token()
    # A lone surrogate (legal in JSON) has no UTF-8 form.
    document = "<!DOCTYPE html><title>t</title><p>caf\u00e9 \u4e2d\u6587 \ud800</p>"
    _stage(harness, [{"token": token, "name": "page.html", "html": document}])
    status, headers, body, _ = _get(harness, f"/view/{token}/page.html")
    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Content-Security-Policy"] == SANDBOX
    assert "no-store" in headers["Cache-Control"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert body == document.replace("\ud800", "?").encode("utf-8")
    # What follows the token only names the browser tab.
    assert _get(harness, f"/view/{token}/sub/other%20name.html")[2] == body
    assert _get(harness, f"/view/{token}")[2] == body
    # Staged again, the token's document is replaced.
    _stage(harness, [{"token": token, "name": "page.html", "html": "<p>second</p>"}])
    assert _get(harness, f"/view/{token}/page.html")[2] == b"<p>second</p>"


def test_a_malformed_token_stages_nothing_and_is_refused_at_once(
    harness: _ServerHarness,
) -> None:
    shortest, longest = "s" * 16, "l" * 64
    bad_tokens = ["s" * 15, "l" * 65, "c" * 15 + "!", "..", "", 7]
    _stage(harness, [
        {"token": token, "html": f"<p>{token}</p>"} for token in [shortest, longest, *bad_tokens]
    ])
    for token in (shortest, longest):
        assert _get(harness, f"/view/{token}/page.html")[2] == f"<p>{token}</p>".encode()
    for bad in bad_tokens:
        status, headers, body, seconds = _get(harness, f"/view/{bad}/page.html")
        assert status == 404, bad
        assert GONE in body.decode("utf-8") and PRODUCT_NAME in body.decode("utf-8")
        assert "Content-Security-Policy" not in headers
        assert seconds < 5, "a token that can never be staged is not waited for"
    # No file is served on this route, whatever the path names.
    assert _get(harness, "/view/../media/main.js")[0] == 404
    assert _get(harness, "/view/")[0] == 404


def test_the_oldest_documents_are_dropped_first(harness: _ServerHarness) -> None:
    # By size: two documents of 33 MiB do not fit the 64 MiB the daemon keeps.
    big = [_token(), _token()]
    filler = "x" * (33 * 1024 * 1024)
    for token in big:
        _stage(harness, [{"token": token, "name": "big.html", "html": token + filler}])
    status, _, body, _ = _get(harness, f"/view/{big[1]}/big.html")
    assert (status, len(body)) == (200, len(big[1]) + len(filler))
    # By count: the ninth small document pushes the first one out.
    small = [_token() for _ in range(9)]
    _stage(harness, [{"token": token, "html": f"<p>{token}</p>"} for token in small])
    for token in small[1:]:
        assert _get(harness, f"/view/{token}/x.html")[2] == f"<p>{token}</p>".encode()
    with ThreadPoolExecutor(max_workers=2) as pool:
        dropped = [
            pool.submit(_get, harness, f"/view/{token}/x.html") for token in (big[0], small[0])
        ]
        assert [answer.result(60)[0] for answer in dropped] == [404, 404]


def test_a_request_waits_for_a_late_upload_then_gives_up(harness: _ServerHarness) -> None:
    waiting = [_token() for _ in range(8)]
    not_text = _token()
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = [pool.submit(_get, harness, f"/view/{token}/late.html") for token in waiting]
        deadline = time.monotonic() + 8
        while harness.server._html_views.waiting < 8:
            assert time.monotonic() < deadline, "the eight requests never came to wait"
            time.sleep(0.02)
        # Eight requests wait: a ninth is answered at once.  So is one
        # for a token whose upload carried no text: nothing was staged.
        _stage(harness, [{"token": not_text, "html": 7}])
        for token in (_token(), not_text):
            status, _, _, seconds = _get(harness, f"/view/{token}/late.html")
            assert status == 404 and seconds < 5
        # The upload of the first request's document arrives after the request.
        _stage(harness, [{"token": waiting[0], "name": "late.html", "html": "<p>late</p>"}])
        status, _, body, _ = answers[0].result(30)
        assert (status, body) == (200, b"<p>late</p>")
        for answer in answers[1:]:
            status, _, body, seconds = answer.result(30)
            assert status == 404 and GONE in body.decode("utf-8")
            assert seconds >= 9.5, "the request waited ten seconds for its document"
    assert harness.server._html_views.waiting == 0
