# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A Stop that lands right after a ``talk`` synthesis still bills it.

October 2026 cost audit: the ``talk`` tool attributed the TTS spend
after a ``try/except Exception``, so a ``KeyboardInterrupt`` (the
daemon's Stop, delivered asynchronously) unwinding through the tool
after ``synthesize_talk_audio`` had already banked its spend in
``usage_out`` discarded that spend with the local dict.  The
attribution now runs in a ``finally``.

The Stop is injected with a real ``KeyboardInterrupt`` raised by a
line tracer at the first line of ``synthesize_talk_audio`` executed
after its own ``finally`` filled ``usage_out`` — the same delivery
model as ``PyThreadState_SetAsyncExc``, pinned to the one window
where the spend is known but not yet attributed.  The synthesis is a
live gpt-audio call (the talk tests already make one).
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import _agent_usage
from kiss.core.speech_synthesis import synthesize_talk_audio
from kiss.tests.agents.sorcar.test_talk_tool import _find_tool, _make_agent
from kiss.tests.server._memory_printer import MemoryPrinter


def test_stop_after_synthesis_still_charges_the_tts_spend() -> None:
    """The task pays for the synthesized audio even when stopped right after it."""
    printer = MemoryPrinter()
    agent = _make_agent(printer)
    talk = _find_tool(agent._get_tools(), "talk")
    synthesis_code = synthesize_talk_audio.__code__
    injected = False

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal injected
        if frame.f_code is not synthesis_code:
            return None
        if event == "line" and not injected:
            usage = frame.f_locals.get("usage_out")
            if isinstance(usage, dict) and "total_tokens_used" in usage:
                injected = True
                raise KeyboardInterrupt("injected stop")
        return tracer

    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt):
            talk("en-US", "Hello there.")
    finally:
        sys.settrace(None)

    assert injected, "the synthesis never banked its spend; nothing was tested"
    budget, tokens, steps = _agent_usage(agent)
    assert budget > 0.0
    assert tokens > 0
    assert steps == 0
    assert not [e for e in printer.emitted if e.get("type") == "talk"]
