# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing in SEA code the loader runs escapes, and never strands the file.

``load_sea`` instantiates the SEA class, and ``_check_text``,
``_check_tools`` and ``_check_value`` call methods of values a SEA
returned; each wraps whatever that code raises into a ``SeaError``.  A
Stop landing there (the runner's ``KeyboardInterrupt`` while the
thread's stop event is set, or the panel's ``ToolCallInterrupted``) was
wrapped too: the task carried on as if the SEA were broken, and
``_registers_as_model`` cached a healthy model-picker SEA as "not a
model" until its file changed.  Each now re-raises the Stop unchanged.

A Stop injected in ``sea_settings._executing`` after the claim was
recorded but before its ``try`` leaves the file's execution record
behind.  It was healed only once the owning thread died, but the
per-tool Stop leaves that thread alive (the agent catches
``ToolCallInterrupted`` and carries on), so every other loader of the
file waited for as long as the task ran.  A record whose thread no
longer executes the file is now dropped while the thread lives.
"""

from __future__ import annotations

import builtins
import dis
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import sea_commands, sea_settings
from kiss.agents.sorcar._concurrency import locked_call
from kiss.agents.sorcar.sea_settings import SeaError, execute_python_file
from kiss.core.stop_signal import set_thread_stop_event
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    end_tool_call,
    interrupt_tool_call,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
)

_PICKER_SEA = """
import builtins
from kiss.agents.seas.base.base_sea import BaseSea


class GatedPicker(BaseSea):
    def __init__(self):
        builtins._sweep1030_parked.set()
        builtins._sweep1030_gate.wait(10)
        super().__init__()

    def register_as_model(self):
        return True
"""

_STOPPED_INIT_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea


class StoppedInit(BaseSea):
    def __init__(self):
        raise KeyboardInterrupt("stop landed in __init__")
"""

_BROKEN_VALUES_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea


class Text(str):
    def __str__(self):
        raise KeyboardInterrupt("stop landed in str()")


class Tools(list):
    def __iter__(self):
        raise KeyboardInterrupt("stop landed in iter()")


class Budget(float):
    def __float__(self):
        raise KeyboardInterrupt("stop landed in float()")


class BrokenValues(BaseSea):
    def settings(self, settings):
        return settings | {"max_budget": Budget(1.0)}

    def prompt(self, task):
        return Text(task)

    def tools(self, tools):
        return Tools(tools)
"""

_PLAIN_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea


class Plain(BaseSea):
    pass
"""

_PARKED_IMPORT_SEA = """
import builtins
builtins._sweep1030_parked.set()
builtins._sweep1030_gate.wait(10)

from kiss.agents.seas.base.base_sea import BaseSea


class Parked(BaseSea):
    pass
"""

_GATE_CODE = sea_settings._executing.__wrapped__.__code__  # type: ignore[attr-defined]


def _wait_until(done: Callable[..., bool], *args: Any) -> None:
    """Poll ``done(*args)`` every 10 ms for up to 10 s; fail when it never holds."""
    deadline = time.monotonic() + 10
    while not done(*args):
        assert time.monotonic() < deadline, f"{done.__name__}{args} never held"
        time.sleep(0.01)


def _injected(outcome: dict[str, Any]) -> bool:
    """Whether the watchdog injected the Stop into the recorded tool call."""
    return bool(outcome["token"].injected)


def _waiting_for(loader: threading.Thread, module_name: str) -> bool:
    """Whether *loader* waits in ``_claim`` for the file *module_name*."""
    return sea_settings._EXECUTING_WAIT.get(loader) == module_name


def _raised(stopping: bool, fn: Callable[..., Any], *args: Any) -> BaseException:
    """Return what ``fn(*args)`` raises on this thread, its stop event set when *stopping*."""
    event = threading.Event()
    if stopping:
        event.set()
    set_thread_stop_event(event)
    try:
        fn(*args)
    except BaseException as exc:  # noqa: BLE001 — the test inspects whatever escaped
        return exc
    finally:
        set_thread_stop_event(None)
    raise AssertionError(f"{fn.__name__} raised nothing")


def _verdict_inside_tool_call(path: Path, outcome: dict[str, Any]) -> None:
    """Run ``_registers_as_model`` inside a registered ``run_agent`` call; record what escaped."""
    token = new_tool_call("run_agent")
    register_tool_call(token)
    outcome["token"] = token
    try:
        outcome["verdict"] = sea_commands._registers_as_model(path)
    except BaseException as exc:  # noqa: BLE001 — the test records whatever escaped
        outcome["error"] = exc
        unregister_tool_call(token)  # as the agent's ``except ToolCallInterrupted`` does
        return
    try:
        end_tool_call(token)  # re-raises the Stop the lookup swallowed
    except ToolCallInterrupted:
        outcome["drained"] = True


def test_per_tool_stop_in_sea_init_escapes_and_keeps_the_picker(tmp_path: Path) -> None:
    """The panel's Stop inside ``__init__`` escapes the lookup; the SEA stays a model."""
    path = tmp_path / "picker_sea.py"
    path.write_text(_PICKER_SEA, encoding="utf-8")
    gate, parked = threading.Event(), threading.Event()
    builtins._sweep1030_gate = gate  # type: ignore[attr-defined]
    builtins._sweep1030_parked = parked  # type: ignore[attr-defined]
    outcome: dict[str, Any] = {}
    thread = threading.Thread(target=_verdict_inside_tool_call, args=(path, outcome), daemon=True)
    try:
        thread.start()
        assert parked.wait(10), "the lookup never reached GatedPicker.__init__"
        assert thread.ident is not None
        assert interrupt_tool_call(thread.ident, "run_agent")
        _wait_until(_injected, outcome)  # the real watchdog injects after its grace period
        gate.set()
        thread.join(10)
        assert not thread.is_alive()
        assert isinstance(outcome.get("error"), ToolCallInterrupted), outcome
        # Nothing was cached for the file: the next lookup instantiates it and finds the model.
        assert sea_commands._registers_as_model(path) is True
    finally:
        gate.set()
        del builtins._sweep1030_gate, builtins._sweep1030_parked  # type: ignore[attr-defined]


@pytest.mark.parametrize("stopping", [True, False])
def test_keyboard_interrupt_in_sea_init_is_a_stop_only_while_stopping(
    tmp_path: Path, stopping: bool
) -> None:
    """``load_sea`` re-raises a Stop from ``__init__``; the same raise unprompted is a SeaError."""
    path = tmp_path / "init_sea.py"
    path.write_text(_STOPPED_INIT_SEA, encoding="utf-8")
    error = _raised(stopping, sea_commands.load_sea, path)
    if stopping:
        assert type(error) is KeyboardInterrupt, repr(error)
    else:
        assert isinstance(error, SeaError), repr(error)
        assert "StoppedInit() raised: KeyboardInterrupt" in str(error)


def _prompt(sea: Any) -> str:
    """Fold the sample task through *sea*'s ``prompt`` (``_check_text``)."""
    return sea_commands.base_prompt([sea], "the task")


def _tools(sea: Any) -> list[Any]:
    """Fold an empty toolset through *sea*'s ``tools`` (``_check_tools``)."""
    return sea_commands.base_tools([sea], [])


def _settings(sea: Any) -> dict[str, Any]:
    """Resolve *sea*'s settings (``_check_value`` on its ``max_budget``)."""
    return sea_commands.base_settings([sea])


@pytest.mark.parametrize(
    ("evaluate", "broken"),
    [
        (_prompt, "returned a broken value: KeyboardInterrupt: stop landed in str()"),
        (_tools, "returned a broken list: KeyboardInterrupt: stop landed in iter()"),
        (_settings, "returned a broken value: KeyboardInterrupt: stop landed in float()"),
    ],
)
@pytest.mark.parametrize("stopping", [True, False])
def test_stop_in_a_returned_value_escapes_its_check(
    tmp_path: Path, evaluate: Callable[[Any], Any], broken: str, stopping: bool
) -> None:
    """A Stop raised by a SEA-returned value's own method escapes the value check."""
    path = tmp_path / "values_sea.py"
    path.write_text(_BROKEN_VALUES_SEA, encoding="utf-8")
    error = _raised(stopping, evaluate, sea_commands.load_sea(path))
    if stopping:
        assert type(error) is KeyboardInterrupt, repr(error)
    else:
        assert isinstance(error, SeaError), repr(error)
        assert broken in str(error)


def _claim_boundary_offset() -> int:
    """Return the offset of the ``STORE_FAST depth`` that follows the claim in ``_executing``.

    The owner record is written and the lock released, the ``try`` is not
    entered: ``PyThreadState_SetAsyncExc`` delivers a Stop at the
    eval-breaker check right after the claim's ``locked_call`` returns,
    which is this boundary.
    """
    instructions = list(dis.get_instructions(_GATE_CODE))
    first_yield = next(i for i, ins in enumerate(instructions) if ins.opname == "YIELD_VALUE")
    return next(
        ins.offset
        for ins in reversed(instructions[:first_yield])
        if ins.opname == "STORE_FAST" and ins.argval == "depth"
    )


def _per_tool_stop_at_claim_boundary(frame: FrameType, event: str, arg: Any) -> Any:
    """Trace function raising ``ToolCallInterrupted`` at the claim boundary of ``_executing``."""
    if frame.f_code is not _GATE_CODE:
        return None
    frame.f_trace_opcodes = True
    if event == "opcode" and frame.f_lasti == _claim_boundary_offset():
        sys.settrace(None)
        raise ToolCallInterrupted("per-tool Stop between the claim and the try")
    return _per_tool_stop_at_claim_boundary


def _load_through_per_tool_stop(
    path: Path, outcome: dict[str, Any], stranded: threading.Event, release: threading.Event
) -> None:
    """Thread body: load *path*, take the per-tool Stop at the claim boundary, carry on.

    Like an agent whose ``run_agent`` call was stopped, the thread lives
    on (until *release*) doing something other than loading the file.
    """
    sys.settrace(_per_tool_stop_at_claim_boundary)
    try:
        execute_python_file(str(path))
    except ToolCallInterrupted:
        outcome["stopped"] = True
    finally:
        sys.settrace(None)
    stranded.set()
    release.wait(10)


def _load(path: Path, outcome: dict[str, Any]) -> None:
    """Thread body: execute *path* and record that it finished."""
    execute_python_file(str(path))
    outcome["loaded"] = True


def _record_of(path: Path) -> str:
    """Return the ``_EXECUTING_OWNER`` key of the SEA file *path*."""
    prefix = f"_kiss_sea_{path.stem}_"
    return next(name for name in sea_settings._EXECUTING_OWNER if name.startswith(prefix))


def test_record_stranded_by_a_per_tool_stop_does_not_block_loaders(tmp_path: Path) -> None:
    """A loader proceeds while the thread that stranded the record is still alive."""
    path = tmp_path / "stranded_live_sea.py"
    path.write_text(_PLAIN_SEA, encoding="utf-8")
    stopped: dict[str, Any] = {}
    stranded, release = threading.Event(), threading.Event()
    stopper = threading.Thread(
        target=_load_through_per_tool_stop, args=(path, stopped, stranded, release), daemon=True
    )
    stopper.start()
    try:
        assert stranded.wait(10)
        assert stopped == {"stopped": True}, stopped
        module_name = _record_of(path)  # the Stop left the record behind
        assert sea_settings._EXECUTING_OWNER[module_name][0] is stopper
        loaded: dict[str, Any] = {}
        loader = threading.Thread(target=_load, args=(path, loaded), daemon=True)
        loader.start()
        loader.join(timeout=10)
        assert loaded == {"loaded": True}, "the load waited for a thread that stopped loading"
        assert stopper.is_alive()
        assert module_name not in sea_settings._EXECUTING_OWNER
    finally:
        release.set()
        stopper.join(timeout=10)


def test_record_of_a_dead_thread_is_dropped(tmp_path: Path) -> None:
    """A record whose thread exited is dropped, even when a new thread reuses its ident."""
    path = tmp_path / "stranded_dead_sea.py"
    path.write_text(_PLAIN_SEA, encoding="utf-8")
    stopped: dict[str, Any] = {}
    stranded, release = threading.Event(), threading.Event()
    release.set()  # the stopper exits right after the Stop
    stopper = threading.Thread(
        target=_load_through_per_tool_stop, args=(path, stopped, stranded, release), daemon=True
    )
    stopper.start()
    stopper.join(timeout=10)
    assert stopped == {"stopped": True}, stopped
    module_name = _record_of(path)
    loaded: dict[str, Any] = {}
    loader = threading.Thread(target=_load, args=(path, loaded), daemon=True)
    loader.start()
    loader.join(timeout=10)
    assert loaded == {"loaded": True}, "the load waited for a dead thread"
    assert module_name not in sea_settings._EXECUTING_OWNER


def test_thread_executing_the_file_keeps_its_claim(tmp_path: Path) -> None:
    """A live thread inside the file's import is the executor: a second loader waits for it."""
    path = tmp_path / "parked_import_sea.py"
    path.write_text(_PARKED_IMPORT_SEA, encoding="utf-8")
    gate, parked = threading.Event(), threading.Event()
    builtins._sweep1030_gate = gate  # type: ignore[attr-defined]
    builtins._sweep1030_parked = parked  # type: ignore[attr-defined]
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    owner = threading.Thread(target=_load, args=(path, first), daemon=True)
    loader = threading.Thread(target=_load, args=(path, second), daemon=True)
    try:
        owner.start()
        assert parked.wait(10), "the first load never reached the import"
        module_name = _record_of(path)
        loader.start()
        _wait_until(_waiting_for, loader, module_name)
        record = locked_call(sea_settings._EXECUTING, sea_settings._executing_owner, module_name)
        assert record is not None and record[0] is owner  # not mistaken for a stale record
        assert not second
        gate.set()
        owner.join(timeout=10)
        loader.join(timeout=10)
        assert first == second == {"loaded": True}
        assert module_name not in sea_settings._EXECUTING_OWNER
    finally:
        gate.set()
        del builtins._sweep1030_gate, builtins._sweep1030_parked  # type: ignore[attr-defined]
