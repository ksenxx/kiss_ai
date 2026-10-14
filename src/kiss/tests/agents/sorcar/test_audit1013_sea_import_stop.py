# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop that lands inside a SEA import is a stop, not a script error.

``execute_python_file`` wraps every import-time failure of a SEA script
(including a ``KeyboardInterrupt`` the script raises itself, F5-07) into
a ``SeaError``.  Two production cancellations are injected into the
task thread with ``PyThreadState_SetAsyncExc`` and can land while the
SEA executes: the runner's Stop (``KeyboardInterrupt``, after the
thread's stop event is set) and the per-panel Stop of
:mod:`kiss.core.tool_interrupt` (``ToolCallInterrupted``).  Either one
swallowed into a ``SeaError`` made the task carry on as if the SEA were
broken, and ``_registers_as_model`` cached "not a model" for the
healthy file until its mtime changed.  The loader now re-raises both.

The scripts below park on a gate the test opens; they run on a real
thread with a real stop binding or a real registered tool call, and
the per-tool case uses the real interrupt watchdog.
"""

from __future__ import annotations

import builtins
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_settings import SeaError, execute_python_file
from kiss.core.stop_signal import set_thread_stop_event
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    begin_tool_call,
    end_tool_call,
    interrupt_tool_call,
    unregister_tool_call,
)

_SCRIPT = """
import builtins
builtins._audit1013_gate.wait(10)
raise KeyboardInterrupt("stop landed here")
"""

_MODEL_SEA = """
import builtins
from kiss.agents.seas.base.base_sea import BaseSea

builtins._audit1013_gate.wait(10)


class GatedModelSea(BaseSea):
    def register_as_model(self):
        return True
"""


def _import_on_task_thread(path: Path, stop_event: threading.Event, outcome: dict) -> None:
    """Run the SEA import on a thread bound to *stop_event* and record what escaped."""
    set_thread_stop_event(stop_event)
    try:
        execute_python_file(str(path))
    except BaseException as exc:  # noqa: BLE001 — the test records whatever escaped
        outcome["error"] = exc
    finally:
        set_thread_stop_event(None)


def _run(tmp_path: Path, stop_before_raise: bool) -> BaseException | None:
    """Import the gated script on a stoppable thread, with or without a pending stop."""
    path = tmp_path / "sea.py"
    path.write_text(_SCRIPT)
    gate = threading.Event()
    builtins._audit1013_gate = gate  # type: ignore[attr-defined]
    stop_event = threading.Event()
    outcome: dict = {}
    thread = threading.Thread(target=_import_on_task_thread, args=(path, stop_event, outcome))
    try:
        thread.start()
        if stop_before_raise:
            stop_event.set()
        gate.set()
        thread.join(10)
    finally:
        del builtins._audit1013_gate  # type: ignore[attr-defined]
    assert not thread.is_alive()
    return outcome.get("error")


def test_stop_during_sea_import_propagates_as_keyboard_interrupt(tmp_path: Path) -> None:
    error = _run(tmp_path, stop_before_raise=True)
    assert isinstance(error, KeyboardInterrupt), repr(error)


def test_script_raising_keyboard_interrupt_without_a_stop_is_a_sea_error(tmp_path: Path) -> None:
    error = _run(tmp_path, stop_before_raise=False)
    assert isinstance(error, SeaError), repr(error)
    assert "KeyboardInterrupt" in str(error)


def _lookup_inside_tool_call(path: Path, outcome: dict) -> None:
    """Run ``_registers_as_model`` inside a registered ``run_agent`` call; record the outcome."""
    token = begin_tool_call("run_agent")
    outcome["token"] = token
    try:
        outcome["verdict"] = sea_commands._registers_as_model(path)
    except BaseException as exc:  # noqa: BLE001 — the test records whatever escaped
        outcome["error"] = exc
        unregister_tool_call(token)  # as the agent's ``except ToolCallInterrupted`` does
    else:
        end_tool_call(token)


def test_per_tool_stop_during_model_sea_import_does_not_poison_cache(tmp_path: Path) -> None:
    """The panel's Stop inside the import escapes, and the healthy SEA stays a model."""
    path = tmp_path / "gated_sea.py"
    path.write_text(_MODEL_SEA)
    gate = threading.Event()
    builtins._audit1013_gate = gate  # type: ignore[attr-defined]
    outcome: dict = {}
    thread = threading.Thread(target=_lookup_inside_tool_call, args=(path, outcome))
    try:
        thread.start()
        while "token" not in outcome:
            time.sleep(0.01)
        assert thread.ident is not None
        assert interrupt_tool_call(thread.ident, "run_agent")
        # The real watchdog injects after its grace period.
        deadline = time.monotonic() + 10
        while not outcome["token"].injected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert outcome["token"].injected
        gate.set()
        thread.join(10)
        assert not thread.is_alive()
        assert isinstance(outcome.get("error"), ToolCallInterrupted), outcome
        # Nothing was cached for the file: the next, uncancelled lookup
        # imports it (the gate stays open) and finds the model.
        assert sea_commands._registers_as_model(path) is True
    finally:
        gate.set()
        del builtins._audit1013_gate  # type: ignore[attr-defined]


if __name__ == "__main__":
    pytest.main([__file__])
