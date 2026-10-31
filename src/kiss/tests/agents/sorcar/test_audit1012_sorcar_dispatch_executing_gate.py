# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The per-file SEA execution gate heals a record its owner could not remove.

The daemon stops a task by injecting ``KeyboardInterrupt`` into the
task thread at an arbitrary bytecode boundary
(``PyThreadState_SetAsyncExc``, see ``task_runner._stop_task``).  In
``sea_settings._executing`` the return of ``locked_call(_EXECUTING,
_claim, ...)`` is such a boundary, and it lies AFTER the thread recorded
itself as the file's executor and released the lock but BEFORE the
``try: yield`` whose ``finally`` removes the record.  A stop landing
there leaves the record behind; nothing of the stopped thread runs
again to remove it.

The test delivers the interrupt at exactly that boundary through an
opcode trace in the loading thread (the instruction after the claim
returned, before the ``try`` is entered, which is where the injected
exception would be raised), then asserts
that a later execution of the same file — on another thread — does not
wait forever.
"""

from __future__ import annotations

import dis
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Any

from kiss.agents.sorcar import sea_settings

_SEA = """\
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"use_memory": False}
"""

_GATE_CODE = sea_settings._executing.__wrapped__.__code__  # type: ignore[attr-defined]


def _after_lock_release_offset() -> int:
    """Return the offset of the instruction right after the claim's ``locked_call`` returns.

    The ``STORE_FAST`` of ``depth``, just before the first ``yield``:
    the owner record is written and the lock released, the ``try`` is
    not entered.  ``PyThreadState_SetAsyncExc`` delivers an injected
    stop at the eval-breaker check that follows the ``locked_call``
    call, which is this very boundary.
    """
    instructions = list(dis.get_instructions(_GATE_CODE))
    first_yield = next(i for i, ins in enumerate(instructions) if ins.opname == "YIELD_VALUE")
    store = next(
        ins
        for ins in reversed(instructions[:first_yield])
        if ins.opname == "STORE_FAST" and ins.argval == "depth"
    )
    return store.offset


def _interrupt_after_lock_release(frame: FrameType, event: str, arg: Any) -> Any:
    """Trace function raising ``KeyboardInterrupt`` at the hazard boundary of ``_executing``.

    Returned as the local trace of ``_executing``'s frame only; the
    exception is raised at the instruction after the condition's lock
    was released and the owner record written, before the ``try``.
    """
    if frame.f_code is not _GATE_CODE:
        return None
    frame.f_trace_opcodes = True
    if event == "opcode" and frame.f_lasti == _after_lock_release_offset():
        sys.settrace(None)
        raise KeyboardInterrupt("stop injected between the lock release and the try")
    return _interrupt_after_lock_release


def _load_with_injected_stop(
    sea: Path,
    outcome: dict[str, Any],
    stranded: threading.Event,
    release: threading.Event,
) -> None:
    """Thread body: execute *sea* with the stop injected at the hazard boundary.

    Sets *stranded* once the stop was caught and stays alive until
    *release*, so a loader started meanwhile gets a thread ident of
    its own (a dead thread's ident is reused by the next thread).
    """
    sys.settrace(_interrupt_after_lock_release)
    try:
        sea_settings.execute_python_file(str(sea))
    except KeyboardInterrupt:
        outcome["stopped"] = True
    finally:
        sys.settrace(None)
    stranded.set()
    release.wait(10)


def _load(sea: Path, outcome: dict[str, Any]) -> None:
    """Thread body: execute *sea* and record that it finished."""
    sea_settings.execute_python_file(str(sea))
    outcome["loaded"] = True


def test_stranded_execution_record_is_healed_for_later_loads(tmp_path: Path) -> None:
    """A load after a stop stranded the file's execution record does not hang."""
    sea = tmp_path / "stranded_sea.py"
    sea.write_text(_SEA, encoding="utf-8")
    stopped: dict[str, Any] = {}
    stranded, release = threading.Event(), threading.Event()
    stopper = threading.Thread(
        target=_load_with_injected_stop,
        args=(sea, stopped, stranded, release),
        daemon=True,
    )
    stopper.start()
    assert stranded.wait(10)
    assert stopped == {"stopped": True}, stopped
    # The stop left the thread's record behind: the gate's finally never ran.
    module_name = next(
        name for name in sea_settings._EXECUTING_OWNER if name.startswith("_kiss_sea_stranded_sea_")
    )

    loaded: dict[str, Any] = {}
    loader = threading.Thread(target=_load, args=(sea, loaded), daemon=True)
    loader.start()
    # The stopped thread no longer executes the file, so the loader drops the
    # stranded record at once instead of queueing until that thread ends.
    loader.join(timeout=10)
    assert stopper.is_alive()
    release.set()
    stopper.join(timeout=10)
    assert loaded == {"loaded": True}, "the load waited for the stopped thread"
    assert module_name not in sea_settings._EXECUTING_OWNER
    # The file executes again normally afterwards, on this thread too.
    assert "Sea" in sea_settings.execute_python_file(str(sea))
