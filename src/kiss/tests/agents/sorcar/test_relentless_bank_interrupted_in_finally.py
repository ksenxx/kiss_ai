# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A stop that lands inside the session bank itself must not lose the spend.

``RelentlessAgent.perform_task`` banks every session's spend in a
``finally`` right after ``executor.run`` returns.  The server's stop
watchdog delivers ``KeyboardInterrupt`` asynchronously, so it can land
INSIDE that ``_accumulate_usage`` call, before its atomic ledger
append.  The ``except BaseException`` handler then banks the executor
again (exact-once by session key) so the terminal ``result`` event and
the ``task_history`` row carry the session's real spend.

Real scripted model run, real ``KeyboardInterrupt`` raised by a trace
function at the first call of the real bank; nothing is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import FrameType

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve

_BANK_CODE = RelentlessAgent._accumulate_usage.__code__


def _run(tmp_path: Path, interrupt_first_bank: bool) -> tuple[tuple[float, int, int], bool]:
    """Run one scripted session; optionally stop it inside its first bank.

    Returns:
        ``(usage snapshot, a KeyboardInterrupt escaped the run)``.
    """
    agent = RelentlessAgent("bank-in-finally")
    fired = False

    def stop_at_first_bank(frame: FrameType, event: str, arg: object) -> None:
        nonlocal fired
        if event == "call" and frame.f_code is _BANK_CODE and not fired:
            fired = True
            raise KeyboardInterrupt("injected stop")

    escaped = False
    with serve([finish_body("<p>ok</p>", prompt_tokens=500)]) as (url, _requests):
        if interrupt_first_bank:
            sys.settrace(stop_at_first_bank)
        try:
            agent.run(
                prompt_template="Say ok.",
                model_name=MODEL,
                work_dir=str(tmp_path),
                max_steps=2,
                max_budget=1.0,
                model_config={"base_url": url, "api_key": "local"},
                verbose=False,
            )
        except KeyboardInterrupt:
            escaped = True
        finally:
            sys.settrace(None)
    assert fired == interrupt_first_bank
    return agent.usage_snapshot(), escaped


def test_a_stop_inside_the_session_bank_still_banks_the_session(tmp_path: Path) -> None:
    """The spend of a session whose bank was interrupted equals an uninterrupted run's."""
    expected, escaped = _run(tmp_path / "plain", interrupt_first_bank=False)
    assert not escaped
    assert expected[1] > 0 and expected[2] >= 1, expected
    banked, escaped = _run(tmp_path / "stopped", interrupt_first_bank=True)
    assert escaped, "the injected stop must propagate out of the run"
    assert banked == expected, (banked, expected)
