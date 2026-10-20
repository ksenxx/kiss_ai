# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A message's files are decoded by the thread that uses them, never by the handler.

Audit 2026-10-20 (C4-F1, review R2).  Decoding an attachment
(``Attachment.from_bytes``, a HEIC transcode of hundreds of
milliseconds) first ran under ``STATE_LOCK``, then was moved in front
of it, which opened two windows:

* ``_cmd_run`` decoded BEFORE registering a fresh run, so a Stop sent
  meanwhile found no run (``stop_ack`` with ``accepted=False``) and the
  run started anyway;
* ``_cmd_append_user_message`` decoded BEFORE choosing the running
  owner, so a task finishing meanwhile made the follow-up (prompt and
  files) vanish.

Now the handlers decode nothing: a running owner takes lazily decoded
``PendingAttachment``s in the same locked step that chooses it, and a
fresh run's worker decodes its own after the run is registered.

Each test parks a thread INSIDE the real decoder with an
:class:`EntryGate` (a profile hook: nothing is replaced) so the
interleaving is forced rather than timed.  The runs execute a real SEA
whose ``settings`` hook blocks on a flag file and then raises, so no
model is ever called.
"""

from __future__ import annotations

import base64
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.core.models.model import Attachment
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import EntryGate, clear_tab_states
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

_TAB = "tab-decode-race"
_PDF = b"%PDF-1.4 decode race"
_UNKNOWN_MODEL = "decode-race-no-such-model"

_BLOCKING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def settings(self, settings):
            \"\"\"Block until released, then raise so no model is called.\"\"\"
            (_DIR / "entered").write_text("1", encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline:
                if (_DIR / "release").exists():
                    raise RuntimeError("released")
                time.sleep(0.02)
            raise RuntimeError("timed out waiting for the release")
    """
)


class _RecordingRunsServer(AgentServer):
    """A real server that keeps every ``run`` command it is given."""

    def __init__(self, printer: JsonPrinter) -> None:
        super().__init__(printer=printer)
        self.runs: list[dict[str, Any]] = []

    def _cmd_run(self, cmd: dict[str, Any]) -> None:
        self.runs.append(dict(cmd))
        super()._cmd_run(cmd)


class _CapturingPrinter(JsonPrinter):
    """Real printer that also keeps a copy of every broadcast."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(dict(event))
        super().broadcast(event)

    def find(self, event_type: str, **fields: Any) -> dict[str, Any] | None:
        """The first captured *event_type* event for the test tab matching *fields*."""
        with self._lock:
            for ev in self.events:
                if (
                    ev.get("type") == event_type
                    and ev.get("tabId") == _TAB
                    and all(ev.get(k) == v for k, v in fields.items())
                ):
                    return ev
        return None

    def wait(self, event_type: str, timeout: float, **fields: Any) -> dict[str, Any] | None:
        """:meth:`find`, polled for up to *timeout* seconds."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ev = self.find(event_type, **fields)
            if ev is not None:
                return ev
            time.sleep(0.02)
        return None


def _wait_file(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"{path.name} never appeared")
        time.sleep(0.02)


def _wire(data: bytes) -> dict[str, str]:
    return {
        "name": "spec.pdf",
        "mimeType": "application/pdf",
        "data": base64.b64encode(data).decode("ascii"),
    }


class TestAttachmentDecodeRace:
    def setup_method(self) -> None:
        self.home = IsolatedKissHome("kiss-decode-race-")
        clear_tab_states(_TAB)
        self.dir = self.home.tmpdir / "sea"
        self.dir.mkdir()
        self.script = self.dir / "agent.py"
        self.script.write_text(_BLOCKING_SCRIPT, encoding="utf-8")
        self.work_dir = self.dir / "work"
        self.work_dir.mkdir()
        self.printer = _CapturingPrinter()
        self.server = _RecordingRunsServer(self.printer)
        self.gate = EntryGate(Attachment.from_bytes.__code__)

    def teardown_method(self) -> None:
        threading.setprofile(None)
        self.gate.release.set()
        (self.dir / "release").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            with agent_state.STATE_LOCK:
                st = agent_state.find_by_tab(_TAB)
                thread = st.task_thread if st is not None else None
            if thread is None:
                break
            thread.join(timeout=1)
        clear_tab_states(_TAB)
        self.home.cleanup()

    def _run_cmd(self, prompt: str, attachments: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "type": "run",
            "tabId": _TAB,
            "prompt": prompt,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "isParallel": False,
            "autoCommit": False,
            "classifyTasks": False,
            "seaPath": str(self.script),
            "attachments": attachments,
        }

    def test_stop_during_a_fresh_runs_attachment_decode_is_accepted(self) -> None:
        """The run is registered before its files are decoded, so the Stop lands."""
        # No SEA: the worker reaches the decode right after setup, and
        # the unknown model makes the run fail before any network call
        # should the Stop not get it first.
        cmd = {**self._run_cmd("go", [_wire(_PDF)]), "model": _UNKNOWN_MODEL}
        del cmd["seaPath"]
        handler = self.gate.start(self.server._cmd_run, cmd)
        assert self.gate.reached.wait(30), "the attachment was never decoded"

        # A thread is inside the decoder.  Only a registered run can
        # be stopped: with the decode in the handler, before the run
        # exists, the Stop found nothing and the run started anyway.
        self.server._stop_task(_TAB)
        ack = self.printer.find("stop_ack")
        assert ack is not None, "the stop was not acknowledged"
        assert ack["accepted"] is True, (
            "BUG: the Stop sent while the fresh run's attachment was being "
            "decoded found no run to stop"
        )

        self.gate.release.set()
        handler.join(timeout=30)
        assert not handler.is_alive()
        result = self.printer.wait("result", timeout=30)
        assert result is not None, "the stopped run never ended"
        assert result["success"] is False
        assert self.printer.wait("status", timeout=30, running=False) is not None

    def test_followup_sent_while_the_owner_finishes_is_not_lost(self) -> None:
        """A follow-up with a file is queued the moment it arrives, before any decode."""
        self.server._cmd_run(self._run_cmd("first run", []))
        _wait_file(self.dir / "entered")
        with agent_state.STATE_LOCK:
            owner = agent_state.find_by_tab(_TAB)
        assert owner is not None and owner.running()

        followup = {
            "tabId": _TAB,
            "prompt": "also read this",
            "attachments": [_wire(_PDF)],
        }
        handler = self.gate.start(self.server._cmd_append_user_message, followup)
        # The handler either queued the message at once (fixed) or is
        # parked in the decoder with the owner not yet chosen (bug).
        while handler.is_alive() and not self.gate.reached.is_set():
            time.sleep(0.01)

        # The owner finishes while the follow-up's file is decoded.
        # Queued already, the message is a leftover that
        # ``_release_run_state`` hands to the re-dispatch; still being
        # decoded, it finds no owner afterwards and is dropped.
        (self.dir / "release").write_text("1", encoding="utf-8")
        assert self.printer.wait("status", timeout=30, running=False) is not None
        self.gate.release.set()
        handler.join(timeout=30)
        assert not handler.is_alive()

        deadline = time.monotonic() + 10
        while len(self.server.runs) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(self.server.runs) == 2, (
            "BUG: the follow-up sent while its owner finished was dropped instead of re-submitted"
        )
        redispatched = self.server.runs[1]
        assert redispatched["prompt"] == "also read this"
        assert [
            (a["mimeType"], base64.b64decode(a["data"])) for a in redispatched["attachments"]
        ] == [("application/pdf", _PDF)]
        # The re-submitted run itself ends (its SEA raises at once).
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            results = [e for e in self.printer.events if e.get("type") == "result"]
            if len(results) >= 2:
                break
            time.sleep(0.02)
        assert len([e for e in self.printer.events if e.get("type") == "result"]) == 2


@pytest.mark.parametrize(
    "raw",
    [
        {"mimeType": "application/pdf", "data": "%%%not base64"},
        {"mimeType": "application/pdf", "data": "AAA"},
    ],
)
def test_pending_attachment_with_undecodable_base64_is_empty(raw: dict[str, str]) -> None:
    """Base64 that passes the shape check but fails to decode yields an empty file."""
    from kiss.server.task_runner import PendingAttachment

    att = PendingAttachment(raw)
    with pytest.raises(AttributeError):
        _ = att.name
    assert att.data == b""
    assert att.mime_type == "application/octet-stream"


def test_pending_attachments_shape_check() -> None:
    """Wrong-shaped entries are dropped up front; a good one decodes on first read."""
    from kiss.server.task_runner import PendingAttachment, pending_attachments

    assert pending_attachments("nope") is None
    assert pending_attachments([]) is None
    assert pending_attachments([1, {"data": 3}, {"mimeType": 4}]) is None
    pending = pending_attachments([_wire(_PDF), "junk"])
    assert pending is not None and len(pending) == 1
    att = pending[0]
    assert isinstance(att, PendingAttachment)
    assert att.__dict__ == {"_raw": _wire(_PDF)}
    assert att.data == _PDF
    assert att.mime_type == "application/pdf"
    assert att.to_data_url() == Attachment(_PDF, "application/pdf").to_data_url()
