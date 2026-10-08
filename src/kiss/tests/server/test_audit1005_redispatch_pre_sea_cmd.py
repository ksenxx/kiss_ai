# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a prompt re-dispatched after a ``/xxx`` run is a plain run.

``_run_task`` re-submits the prompts left in ``pending_user_messages``
as the tab's next run (``_redispatch_leftover_prompts``) — "what would
have happened had the user typed one second later".  The follow-up is
built from the finished run's ``run`` command, which ``_apply_sea``
had rewritten IN PLACE: a ``/xxx text`` slash command had become a run
of the SEA ``xxx`` (``seaPath`` set, ``prompt`` replaced by the
trailing text) and the SEA's pinned settings had overwritten the
user's.  A follow-up typed during a ``/sh ls`` run was therefore
re-submitted as another run of the ``sh`` SEA — the SEA executed
again, with the follow-up text as its task — instead of as the plain
prompt the user typed.

The fix snapshots the command before the SEA pipeline touches it and
re-dispatches from that snapshot.

Harness: a real ``VSCodeServer`` with a real ``JsonPrinter`` subclass
that submits the late prompt through the REAL ``_cmd_run`` the moment
the run's ``status running:true`` is broadcast (the worker thread is
installed, so the prompt is queued and never drained), a user-level
SEA registered through the ``SEAS.md`` of a private ``KISS_HOME``
whose module body appends a line to a marker file every time the SEA
is executed, and a model name that is not available, so both runs
take the runner's "No model available" exit without an LLM.  No mocks
or patches.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

_UNKNOWN_MODEL = "audit1005-redispatch-no-such-model"
_WAIT_S = 60.0
_TAB = "tab-a1005-redispatch"
_COMMAND = "a1005marker"

_SEA_BODY = """
from kiss.agents.seas.base.base_sea import BaseSea

# Module level: runs once per execution of the SEA file (one per run
# that names it), however many times the launcher calls ``settings``.
with open({marker!r}, "a", encoding="utf-8") as fh:
    fh.write("executed\\n")


class Sea(BaseSea):
    def description(self):
        return "appends a line to the marker file each time it is executed"

    def settings(self, settings):
        return settings | {{"use_worktree": False, "auto_commit": False}}
"""


class _LatePromptPrinter(JsonPrinter):
    """Real printer that submits *late_prompt* once the run has started."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.server: VSCodeServer | None = None
        self.late_prompt = ""
        self._submitted = False

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*; on the first ``status running:true`` queue the late prompt."""
        self.events.append(dict(event))
        super().broadcast(event)
        if (
            event.get("type") == "status"
            and event.get("running") is True
            and not self._submitted
            and self.server is not None
        ):
            self._submitted = True
            self.server._cmd_run({"tabId": _TAB, "prompt": self.late_prompt})


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path) -> Iterator[Path]:
    """Register the marker SEA in a private ``KISS_HOME``; drop every state of the tab.

    The ``SEAS.md`` lives in a throwaway home (``IsolatedKissHome``
    redirects ``KISS_HOME``, the config and the history DB there), so
    the suite-wide home's ``SEAS.md`` is never written or removed.
    """
    marker = tmp_path / "executed.txt"
    sea = tmp_path / "seas" / _COMMAND / f"{_COMMAND}_sea.py"
    sea.parent.mkdir(parents=True)
    sea.write_text(_SEA_BODY.format(marker=str(marker)), encoding="utf-8")
    home = IsolatedKissHome("kiss-audit1005-redispatch-")
    try:
        (home.kiss_home / "SEAS.md").write_text(
            str(tmp_path / "seas") + "\n", encoding="utf-8",
        )
        sea_commands._reset_for_tests()
        sea_commands.refresh_registry()
        assert sea_commands.get_command(_COMMAND) is not None
        _clear_tab_states()
        yield marker
    finally:
        _clear_tab_states()
        sea_commands._reset_for_tests()
        home.cleanup()


def _clear_tab_states() -> None:
    with agent_state.STATE_LOCK:
        for st in agent_state.snapshot():
            if st.tab_id == _TAB:
                agent_state.unregister(st.task_id, st)


def _wait_idle() -> None:
    """Wait until no worker thread is installed on the tab (joining each in turn)."""
    deadline = time.time() + _WAIT_S
    while time.time() < deadline:
        with agent_state.STATE_LOCK:
            st = agent_state.find_by_tab(_TAB)
            thread = st.task_thread if st is not None else None
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.time()))
            assert not thread.is_alive(), "worker thread did not finish"
            continue
        time.sleep(0.3)
        with agent_state.STATE_LOCK:
            st = agent_state.find_by_tab(_TAB)
            if st is None or st.task_thread is None:
                return
    raise AssertionError("tab never went idle")


def test_followup_after_slash_run_is_a_plain_run(
    tmp_path: Path, _isolate: Path,
) -> None:
    """The re-dispatched follow-up runs the user's prompt, not the SEA again."""
    marker = _isolate
    printer = _LatePromptPrinter()
    server = VSCodeServer(printer=printer)
    printer.server = server
    late = "and now a plain follow-up"
    printer.late_prompt = late
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    server._cmd_run({
        "type": "run",
        "tabId": _TAB,
        "prompt": f"/{_COMMAND} do the thing",
        "workDir": str(work_dir),
        "model": _UNKNOWN_MODEL,
        "useWorktree": False,
        "autoCommit": False,
        "classifyTasks": False,
    })
    _wait_idle()
    # Both runs ended on the runner's no-model exit: the first (the
    # slash command) and the re-dispatched follow-up.
    results = [e for e in printer.events if e.get("type") == "result"]
    assert [("No model available" in str(r.get("text", ""))) for r in results] == [
        True, True,
    ], results
    clears = [e for e in printer.events if e.get("type") == "clear"]
    assert len(clears) == 2, clears
    st = agent_state.find_by_tab(_TAB)
    assert st is not None and st.task_thread is None
    assert st.last_user_prompt == late
    # The SEA was executed by the slash command only; the follow-up is
    # the user's plain prompt (no ``seaPath``), so it did not run the
    # SEA a second time.
    assert marker.read_text(encoding="utf-8").count("executed") == 1
