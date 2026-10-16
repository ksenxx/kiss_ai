# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-16 (core): ``locked_call`` must not swallow kwargs named ``lock``/``fn``.

``locked``'s wrapper forwards every keyword argument into
``locked_call(lock, fn, *args, **kwargs)``.  While ``lock`` and ``fn``
were ordinary parameters, a decorated function whose own parameters are
called ``lock`` or ``fn`` could not be called by keyword: Python raised
``TypeError: got multiple values for argument 'lock'``.  They are now
positional-only, so such keywords pass through to the wrapped function.
"""

from __future__ import annotations

import threading

from kiss.core.locked import locked, locked_call

_LOCK = threading.RLock()


def _describe(lock: str, fn: str, extra: str = "") -> str:
    """Return the arguments joined, proving they reached the wrapped function."""
    assert _LOCK._is_owned()  # type: ignore[attr-defined]
    return f"{lock}|{fn}|{extra}"


describe = locked(_LOCK)(_describe)


def test_decorated_function_accepts_lock_and_fn_keywords() -> None:
    """Keywords named like locked_call's own parameters reach the wrapped function."""
    assert describe(lock="L", fn="F", extra="E") == "L|F|E"
    assert describe("L", fn="F") == "L|F|"
    assert describe("L", "F", extra="E") == "L|F|E"
    assert not _LOCK._is_owned()  # type: ignore[attr-defined]


def test_locked_call_forwards_lock_and_fn_keywords() -> None:
    """Calling ``locked_call`` directly forwards ``lock=``/``fn=`` to the section."""
    assert locked_call(_LOCK, _describe, lock="a", fn="b") == "a|b|"
