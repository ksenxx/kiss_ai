# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-16 (server-core): a Stop landing inside ``/xxx help``.

``/xxx help`` is answered by the task runner itself from the SEA's
``description()`` (``help_text_if_command``), untrusted code that runs
on the task thread.  The loader converts every raise there —
``BaseException`` included — into a ``SeaError``, and the help branch
of ``_run_task_inner`` catches that error as "the SEA is broken".  A
Stop whose watchdog injected its ``KeyboardInterrupt`` while
``description()`` ran was therefore swallowed:

* the run was answered with a persisted ``task_error`` row reading
  ``"description() of SEA '...' raised: KeyboardInterrupt"`` instead
  of ``"Task stopped by user"``;
* ``_cancel_outcome`` never ran, so the stop was never acknowledged
  and the watchdog's 5-second retry could land a second interrupt in
  the row persistence.

``_run_task`` already treats the same wrapped interrupt from
``_apply_sea`` as a cancellation (``_stop_interrupt_wrapped``); the
help branch now re-raises it to that handler.

Everything here is real: a ``RemoteAccessServer`` on its local
endpoint, a registered user SEA, the real worker thread and the real
watchdog.  The interrupt lands deterministically inside
``description()`` because the getter blocks (polling for a release
file) until the injection arrives; releasing it makes it raise, so a
run that survives the stop still fails without touching a model.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import local_endpoint, persistence, sea_commands
from kiss.core.config import kiss_home
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer

_STOP_LABEL = "Task stopped by user"

_BLOCKING_DESCRIPTION = textwrap.dedent(
    """
    import pathlib
    import time

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def description(self):
            \"\"\"Block until interrupted; raise if released or timed out.\"\"\"
            (_DIR / "entered").write_text("1", encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline:
                if (_DIR / "release").exists():
                    raise RuntimeError("released before the stop landed")
                time.sleep(0.02)
            raise RuntimeError("timed out waiting for the stop")
    """
)


class _LocalClient:
    """JSON client of the server's local endpoint, reading on a thread."""

    def __init__(self, endpoint_file: Path) -> None:
        self.ws = local_endpoint.connect(endpoint_file)
        self.events: list[dict[str, Any]] = []
        self._cond = threading.Condition()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        while True:
            try:
                message = self.ws.recv()
            except (ConnectionClosed, OSError):
                return
            with self._cond:
                self.events.append(json.loads(message))
                self._cond.notify_all()

    def send(self, cmd: dict[str, Any]) -> None:
        self.ws.send(json.dumps(cmd))

    def wait_for(
        self, event_type: str, tab_id: str, timeout: float = 30.0, **fields: Any,
    ) -> dict[str, Any]:
        """Return the first *event_type* event for *tab_id* matching *fields*."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for ev in self.events:
                    if (
                        ev.get("type") == event_type
                        and ev.get("tabId") == tab_id
                        and all(ev.get(k) == v for k, v in fields.items())
                    ):
                        return ev
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"no {event_type!r} for {tab_id!r}; got "
                        f"{[e.get('type') for e in self.events]}",
                    )
                self._cond.wait(remaining)

    def close(self) -> None:
        self.ws.close()


class TestStopDuringHelp(TestCase):
    """A stop swallowed by the help loader is still a user stop."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit1016-help-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        sea = self.tmp / "seas" / "slowhelp" / "slowhelp_sea.py"
        sea.parent.mkdir(parents=True)
        sea.write_text(_BLOCKING_DESCRIPTION, encoding="utf-8")
        self.sea_dir = sea.parent
        home = kiss_home()
        home.mkdir(parents=True, exist_ok=True)
        seas_md = home / "SEAS.md"
        seas_md.write_text(str(self.tmp / "seas") + "\n", encoding="utf-8")
        self.addCleanup(sea_commands._reset_for_tests)
        self.addCleanup(seas_md.unlink)
        sea_commands.refresh_registry()
        assert sea_commands.get_command("slowhelp") == sea.resolve()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.remote = RemoteAccessServer(
            local_endpoint_file=self.tmp / "sorcar-local.json",
            work_dir=str(self.work_dir),
        )
        asyncio.run_coroutine_threadsafe(
            self.remote.start_private_async(), self.loop,
        ).result(timeout=30)
        self.client = _LocalClient(self.tmp / "sorcar-local.json")

    def tearDown(self) -> None:
        (self.sea_dir / "release").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        self.client.close()
        asyncio.run_coroutine_threadsafe(
            self.remote.stop_async(), self.loop,
        ).result(timeout=60)
        self.loop.call_soon_threadsafe(self.loop.stop)
        agent_state.agent_states.clear()

    def _run_and_stop(self, prompt: str, tab_id: str) -> None:
        """Send a real ``run`` of *prompt*, park it in ``description()``, stop it."""
        run_token = f"tok-{tab_id}"
        self.client.send({
            "type": "run",
            "prompt": prompt,
            "tabId": tab_id,
            "taskId": run_token,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "isParallel": False,
            "autoCommit": False,
        })
        entered = self.sea_dir / "entered"
        deadline = time.monotonic() + 30
        while not entered.exists():
            if time.monotonic() > deadline:
                raise AssertionError("description() never ran")
            time.sleep(0.02)
        self.client.send({"type": "stop", "tabId": tab_id, "taskId": run_token})
        ack = self.client.wait_for("stop_ack", tab_id)
        self.assertTrue(ack["accepted"])
        result = self.client.wait_for("result", tab_id)
        self.assertEqual(
            result["text"],
            _STOP_LABEL,
            "BUG: the SEA loader folded the stop's KeyboardInterrupt "
            f"into the answer: {result['text']!r}",
        )
        self.assertFalse(result["success"])
        self.client.wait_for("status", tab_id, running=False)
        # No help row quoting the interrupt was persisted for the
        # stopped run.
        rows = [e for e in persistence._load_history() if e.get("task") == prompt]
        self.assertEqual(
            [e["result"] for e in rows if "KeyboardInterrupt" in str(e.get("result") or "")],
            [],
        )

    def test_stop_during_help_description_is_a_user_stop(self) -> None:
        """KI inside ``description()`` of ``/slowhelp help`` (``SeaError`` path)."""
        self._run_and_stop("/slowhelp help", "help-stop-tab")

    def test_stop_during_check_is_a_user_stop(self) -> None:
        """KI inside ``description()`` of ``/slowhelp check`` (report-text path).

        ``sea_check`` catches the loader's ``SeaError`` itself and
        returns ``"/slowhelp is broken: … KeyboardInterrupt"`` as a
        SUCCESSFUL answer, so the runner cannot see the wrapped
        interrupt at all: only the requested stop tells it apart.
        """
        self._run_and_stop("/slowhelp check", "check-stop-tab")
