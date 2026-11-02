# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a machine link's answer is the frame answering its own request.

A real remote daemon (:mod:`kiss.tests.remote_daemon`) registered as
machine ``box``, and the real links to it (:func:`machines.link_for`).
Another client of the daemon adds custom models without pause, each
add pushed to every connection (the link's included) as an unstamped
``myModelsData``; meanwhile the link saves models of its own, and
every answer carries the model it just saved.

Before the fix the link took the first ``myModelsData`` after its
request: a list pushed for the other client's edit while the request
was on its way (in flight, beyond the reach of the link's drain of the
frames already received) passed for the answer, and the real answer,
arriving later, for the next request's.
``test_remote_machine_settings.py::test_broadcasts_queued_on_the_link_
do_not_answer_a_later_request`` failed that way under load.  Now the
daemon echoes the request's ``generation`` on the frame answering it,
and on that frame only: the list a save pushes to every client is
unstamped, the requester gets besides a copy echoing its generation
(a first version stamped the pushed list itself, which another link,
counting from 0 like every link, could be waiting for).  The link
takes no other frame of that type, when the daemon said it echoes
(``echoesGeneration`` in its ``auth_ok``); an older daemon's first
frame of the type is read as the answer, as before.
"""

from __future__ import annotations

import json
import ssl
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from websockets.sync.client import connect as ws_connect
from websockets.sync.server import Server, ServerConnection, serve

from kiss.agents.sorcar import machines as mach
from kiss.tests.local_ws import make_test_tls
from kiss.tests.remote_daemon import start_remote_daemon

PASSWORD = "pw-remote"
SAVES = 40
"""Saves over the link; each is a chance for a pushed list to pass for the answer."""


def _client(url: str) -> Any:
    """Return an authenticated websocket client of the daemon at *url* (a context manager)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ws = ws_connect(mach.ws_url(url), ssl=ctx, open_timeout=30)
    ws.send(json.dumps({"type": "auth", "password": PASSWORD}))
    assert json.loads(ws.recv(timeout=30))["type"] == "auth_ok"
    return ws


def _answer(
    ws: Any, cmd: dict[str, Any], reply_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Send *cmd* on *ws* and return its *reply_type* answer and the pushes before it.

    The answer is the *reply_type* frame echoing the command's
    ``generation`` (the first one, for a command carrying none); the
    pushes are the generation-free *reply_type* frames received before
    it (a frame echoing another generation would be a defect).
    """
    ws.send(json.dumps({**cmd, "password": PASSWORD}))
    generation = cmd.get("generation")
    pushed: list[dict[str, Any]] = []
    while True:
        msg: dict[str, Any] = json.loads(ws.recv(timeout=30))
        if msg.get("type") != reply_type:
            continue
        if generation is None or msg.get("generation") == generation:
            return msg, pushed
        assert "generation" not in msg, f"a frame of another request's generation: {msg}"
        pushed.append(msg)


def _names(frame: dict[str, Any]) -> list[str]:
    """Return the model names a ``myModelsData`` *frame* lists."""
    return [m["name"] for m in frame["models"]]


def _lists_pushed_to(ws: Any) -> list[dict[str, Any]]:
    """Return the ``myModelsData`` frames *ws* received before the ``pong`` of a ping sent now."""
    ws.send(json.dumps({"type": "ping", "password": PASSWORD}))
    frames: list[dict[str, Any]] = []
    while True:
        msg: dict[str, Any] = json.loads(ws.recv(timeout=30))
        if msg.get("type") == "pong":
            return frames
        if msg.get("type") == "myModelsData":
            frames.append(msg)


def _add_models_without_pause(url: str, stop: threading.Event) -> int:
    """Add custom models on the daemon at *url* as another client until *stop*; the count."""
    added = 0
    with _client(url) as ws:
        while not stop.is_set():
            ws.send(json.dumps({
                "type": "saveMyModel", "name": f"other-{added}", "password": PASSWORD,
            }))
            while json.loads(ws.recv(timeout=30)).get("type") != "myModelsData":
                pass
            added += 1
    return added


def _save_over(link: mach.MachineLink, tag: str, outcome: dict[str, Any]) -> None:
    """Save :data:`SAVES` models ``<tag>-<k>`` over *link*, each answer checked for its own.

    A failure is left in *outcome* under *tag* (thread body).
    """
    try:
        for k in range(SAVES):
            reply = link.request({"type": "saveMyModel", "name": f"{tag}-{k}"}, "myModelsData")
            assert not reply.get("error"), reply
            names = _names(reply)
            assert f"{tag}-{k}" in names, f"{tag}: save {k} answered without it: {names}"
    except BaseException as exc:  # noqa: BLE001 - reported by the test
        outcome[tag] = exc


def test_a_save_over_the_link_is_answered_with_its_own_list(tmp_path: Path) -> None:
    remote = start_remote_daemon(tmp_path / "remote-home", password=PASSWORD)
    try:
        mach.save_machines([])
        link = mach.link_for(mach.add_machine("box", remote.url, PASSWORD))
        # The link's socket is up before the other client starts.
        assert link.request({"type": "getMyModels"}, "myModelsData")["models"] == []
        stop = threading.Event()
        outcome: dict[str, Any] = {}

        def other_client() -> None:
            try:
                outcome["added"] = _add_models_without_pause(remote.url, stop)
            except BaseException as exc:  # noqa: BLE001 - reported by the test below
                outcome["error"] = exc

        other = threading.Thread(target=other_client, name="other-client", daemon=True)
        other.start()
        try:
            _save_over(link, "mine", outcome)
        finally:
            stop.set()
            other.join(timeout=60)
        assert "error" not in outcome, outcome["error"]
        assert "mine" not in outcome, outcome["mine"]
        # The other client's pushes did flow while the link was asking.
        assert outcome["added"] > 0, outcome
    finally:
        mach.save_machines([])
        remote.stop()


def test_two_links_counting_alike_each_get_their_own_answer(tmp_path: Path) -> None:
    """Two lanes of one machine save without pause; each answer carries the lane's own model.

    Every link counts its generations from 0, so the two stamp their
    requests with the same numbers: a daemon that stamped the pushed
    list itself with the requester's generation (the first version of
    the echo) had one lane take the other's list for its answer.
    """
    remote = start_remote_daemon(tmp_path / "remote-home", password=PASSWORD)
    try:
        mach.save_machines([])
        machine = mach.add_machine("box", remote.url, PASSWORD)
        links = [mach.link_for(machine), mach.link_for(machine, "panels")]
        assert links[0] is not links[1]
        for link in links:  # both sockets up, both counters at 1
            assert link.request({"type": "getMyModels"}, "myModelsData")["models"] == []
        outcome: dict[str, Any] = {}
        threads = [
            threading.Thread(target=_save_over, args=(link, tag, outcome), name=tag, daemon=True)
            for link, tag in zip(links, ("lane-a", "lane-b"), strict=True)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not any(thread.is_alive() for thread in threads), outcome
        assert outcome == {}, outcome
    finally:
        mach.save_machines([])
        remote.stop()


def test_on_the_wire_only_the_requester_gets_the_echo(tmp_path: Path) -> None:
    """A ``myModelsData`` or ``machines`` answer echoes the request's ``generation``.

    A save's list is pushed to every client unstamped (the requester's
    connection included, where it precedes the requester's stamped
    copy); a client that asked nothing sees each mutation once, without
    a generation; a page's request, which carries none, is answered as
    before, without one.
    """
    remote = start_remote_daemon(tmp_path / "remote-home", password=PASSWORD)
    try:
        with _client(remote.url) as ws, _client(remote.url) as other:
            listed, pushed = _answer(ws, {"type": "getMyModels", "generation": 5}, "myModelsData")
            assert listed["generation"] == 5 and listed["models"] == [] and pushed == []
            save = {"type": "saveMyModel", "name": "m", "generation": 6}
            saved, pushed = _answer(ws, save, "myModelsData")
            assert saved["generation"] == 6 and _names(saved) == ["m"]
            assert [p.get("generation") for p in pushed] == [None] and _names(pushed[0]) == ["m"]
            delete = {"type": "deleteMyModel", "name": "m", "generation": 7}
            gone, pushed = _answer(ws, delete, "myModelsData")
            assert gone["generation"] == 7 and gone["models"] == []
            assert [p.get("generation") for p in pushed] == [None] and pushed[0]["models"] == []
            seen = _lists_pushed_to(other)
            assert [(_names(f), f.get("generation")) for f in seen] == [(["m"], None), ([], None)]
            machines, _ = _answer(ws, {"type": "getMachines", "generation": 8}, "machines")
            assert machines["generation"] == 8 and machines["action"] == ""
            refused, _ = _answer(
                ws, {"type": "machineAction", "action": "remove", "name": "nowhere",
                     "generation": 9},
                "machines",
            )
            assert refused["generation"] == 9 and refused["action"] == "remove"
            assert refused["error"], refused
            # A page's request: no generation, none echoed.
            page, _ = _answer(ws, {"type": "getMyModels"}, "myModelsData")
            assert "generation" not in page
            assert "generation" not in _answer(ws, {"type": "getMachines"}, "machines")[0]
    finally:
        remote.stop()


def _stand_in(
    tmp: Path, auth_ok: dict[str, Any], answer: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> tuple[Server, mach.Machine, Path]:
    """Serve a daemon stand-in over ``wss://``; its machine entry and pinned CA.

    Answers the link's ``auth`` with *auth_ok* and every later command
    with the frames *answer* returns for it.  The caller shuts the
    server down.
    """
    certfile, keyfile, ca_file = make_test_tls(tmp)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    # As ``web_server._create_ssl_context``: a TLS 1.3 session ticket
    # read by the sync client's receive thread while its main thread
    # writes the upgrade intermittently loses the upgrade.
    ctx.num_tickets = 0

    def handler(ws: ServerConnection) -> None:
        ws.recv()  # the link's auth frame
        ws.send(json.dumps(auth_ok))
        for raw in ws:
            for frame in answer(json.loads(raw)):
                ws.send(json.dumps(frame))

    server = serve(handler, "127.0.0.1", 0, ssl=ctx)
    threading.Thread(target=server.serve_forever, name="stand-in-daemon", daemon=True).start()
    url = f"https://127.0.0.1:{server.socket.getsockname()[1]}"
    return server, mach.Machine(name="stand-in", url=url, password="pw"), ca_file


def _older_daemon_answer(cmd: dict[str, Any]) -> list[dict[str, Any]]:
    """An older daemon's answer: the list, echoing no generation."""
    return [{"type": "myModelsData", "models": [{"name": "old"}]}]


def _push_then_echo(cmd: dict[str, Any]) -> list[dict[str, Any]]:
    """A current daemon's frames for a link's save: the unstamped push, then its echo."""
    return [
        {"type": "myModelsData", "models": [{"name": "pushed"}]},
        {"type": "myModelsData", "models": [{"name": "mine"}], "generation": cmd["generation"]},
    ]


def test_a_link_reads_an_older_daemon_as_before(tmp_path: Path) -> None:
    """Without ``echoesGeneration`` in the ``auth_ok``, the type's first frame is the answer."""
    server, machine, ca_file = _stand_in(
        tmp_path, {"type": "auth_ok", "local": False}, _older_daemon_answer,
    )
    link = mach.MachineLink(machine, cert_file=ca_file)
    try:
        reply = link.request({"type": "getMyModels"}, "myModelsData", timeout=10)
        assert _names(reply) == ["old"] and "generation" not in reply
    finally:
        link.close()
        server.shutdown()


def test_a_link_skips_the_push_before_its_echo(tmp_path: Path) -> None:
    """With ``echoesGeneration``, a generation-free frame of the type is not the answer."""
    server, machine, ca_file = _stand_in(
        tmp_path, {"type": "auth_ok", "local": False, "echoesGeneration": True}, _push_then_echo,
    )
    link = mach.MachineLink(machine, cert_file=ca_file)
    try:
        reply = link.request({"type": "saveMyModel", "name": "mine"}, "myModelsData", timeout=10)
        assert _names(reply) == ["mine"] and reply["generation"] == 1
    finally:
        link.close()
        server.shutdown()
