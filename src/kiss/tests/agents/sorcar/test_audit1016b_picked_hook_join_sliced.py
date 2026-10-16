# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``run_picked_hook`` waits for the hook in slices so a Stop lands at once.

Audit 2026-10-16 (sorcar-tools F1).  The task runner calls
``run_picked_hook(..., wait=True)`` on the task thread before the agent
starts; it joined the hook thread with one unsliced
``Thread.join(15.0)``, a single C-level wait that the daemon's forced
Stop (``PyThreadState_SetAsyncExc``) cannot land in.  With a slow hook
the Stop button was dead for up to 15 s.  Real SEA file registered
through ``$KISS_HOME/SEAS.md``, real hook thread, real injection; no
mocks.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.core.tool_interrupt import inject_async_exception


@pytest.fixture
def slow_picker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Register a user SEA whose ``on_picked_as_model`` blocks for 30 s; yield its name."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    sea_dir = tmp_path / "user_seas" / "slowpicker"
    sea_dir.mkdir(parents=True)
    (sea_dir / "slowpicker_sea.py").write_text(
        "import time\n"
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
        "class Sea(BaseSea):\n"
        "    def description(self):\n        return 'slow picker'\n"
        "    def register_as_model(self):\n        return True\n"
        "    def on_picked_as_model(self, work_dir):\n"
        "        time.sleep(30)\n        return 'done'\n",
        encoding="utf-8",
    )
    (home / "SEAS.md").write_text(f"{tmp_path / 'user_seas'}\n", encoding="utf-8")
    sea_commands._reset_for_tests()
    sea_commands.refresh_registry()
    assert sea_commands.model_sea("slowpicker") == (sea_dir / "slowpicker_sea.py").resolve()
    yield "slowpicker"
    sea_commands._reset_for_tests()


def test_forced_stop_lands_while_the_hook_is_still_running(
    slow_picker: str, tmp_path: Path
) -> None:
    """An injected ``KeyboardInterrupt`` ends the waiting task thread within ~1 s, not 15 s."""
    outcome: dict[str, object] = {}

    def task_thread() -> None:
        try:
            sea_commands.run_picked_hook(slow_picker, str(tmp_path))
            outcome["returned"] = True
        except KeyboardInterrupt:
            outcome["interrupted"] = time.monotonic()

    thread = threading.Thread(target=task_thread, daemon=True)
    thread.start()
    time.sleep(0.5)
    assert thread.ident is not None
    injected = time.monotonic()
    assert inject_async_exception(thread.ident, KeyboardInterrupt) == 1
    thread.join(timeout=20)
    assert not thread.is_alive(), "the join never let the Stop land"
    assert "interrupted" in outcome, outcome
    landed = float(outcome["interrupted"]) - injected  # type: ignore[arg-type]
    assert landed < 1.5, f"Stop deferred by the unsliced join: landed after {landed:.1f} s"
