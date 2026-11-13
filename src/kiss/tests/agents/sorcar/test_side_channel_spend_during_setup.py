# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Side-channel spend charged while a task is still setting up.

A task's row becomes visible (and its tab can start a ``/update`` or
``/ask`` side channel) the moment ``ChatSorcarAgent.run`` allocates it,
but ``RelentlessAgent._reset`` used to start the run's usage epoch only
later, after the task classifier ran.  A side channel that captured the
agent's epoch inside that window (``run_task_update_sea`` and the
``/ask`` dispatcher both capture it when they start) banked its spend
into a ledger the reset then discarded, so the task's persisted cost,
and every ancestor's, silently omitted the side channel's spend
(observed in ``~/.kiss/history.db``: a $0.0373 task update lost from its
parent).  The run's epoch now starts at row allocation.

Drives a real :class:`ChatSorcarAgent` against a real local
chat-completions server; no mocks.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _get_db
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.task_update import charge_side_channel_usage
from kiss.tests.agents.sorcar.test_task_settings_info import (
    _DBRedirect,
    _FinishHandler,
)
from kiss.tests.core.test_budget_enforcement_e2e import _start_server


def _charge_side_channel(
    agent: ChatSorcarAgent, stop: bool, task_id: str, _chat_id: str,
) -> None:
    """Bank a side channel's spend under the epoch captured right now.

    Used as the run's ``_on_task_id_allocated`` callback, i.e. inside the
    window between row allocation and ``_reset``.  With *stop*, then
    interrupts the run the way a user stop during setup does.
    """
    epoch = agent._usage_epoch()
    charge_side_channel_usage(None, agent, task_id, 0.5, 70, 1, epoch=epoch)
    if stop:
        raise KeyboardInterrupt("stopped during setup")


def _broken_system_prompt(prompt: str) -> str:
    """A SEA ``system_prompt`` method that fails: the run ends in its setup."""
    raise RuntimeError("system prompt failed")


class TestSideChannelSpendDuringSetup(_DBRedirect):
    """Spend banked between row allocation and ``_reset`` must count."""

    def _run(self, agent: ChatSorcarAgent, on_alloc: Any) -> None:
        """Run *agent* once against a fake model that finishes at once."""
        srv, url = _start_server(_FinishHandler)
        try:
            agent.run(
                prompt_template="say hi",
                model_name="gpt-4o-mini",
                work_dir=self.tmpdir,
                max_steps=3,
                web_tools=False,
                is_parallel=False,
                append_basic_tools=False,
                verbose=False,
                model_config={"base_url": url, "api_key": "test-key"},
                _on_task_id_allocated=on_alloc,
            )
        finally:
            srv.shutdown()

    def _row(self, task_id: str) -> tuple[float, int, int]:
        """The persisted ``(cost, tokens, steps)`` of *task_id*."""
        row = _get_db().execute(
            "SELECT cost, tokens, steps FROM task_history WHERE id = ?",
            (task_id,),
        ).fetchone()
        return float(row[0]), int(row[1]), int(row[2])

    def test_epoch_captured_at_allocation_is_billed_to_the_task(self) -> None:
        """A side channel started right after allocation is charged in full.

        Runs the agent twice: the first run gives the reused agent a
        non-empty previous ledger (the real tab-reuse case), the second
        charges a side channel whose epoch was captured inside the
        allocation callback, exactly when ``getTaskUpdate`` can fire.
        """
        agent = ChatSorcarAgent("setup-window")
        self._run(agent, None)
        first_cost, first_tokens, first_steps = self._row(agent.last_task_id)
        assert first_cost > 0

        self._run(agent, partial(_charge_side_channel, agent, False))
        cost, tokens, steps = self._row(agent.last_task_id)
        # Rows store cost rounded to 6 decimals, hence the tolerance.
        assert cost == pytest.approx(first_cost + 0.5, abs=2e-6)
        assert tokens == first_tokens + 70
        assert steps == first_steps + 1

    def test_stop_during_setup_still_persists_the_side_channel_spend(
        self,
    ) -> None:
        """A run stopped before ``super().run`` keeps the banked spend."""
        agent = ChatSorcarAgent("setup-stop")
        self._run(agent, None)
        with pytest.raises(KeyboardInterrupt):
            self._run(agent, partial(_charge_side_channel, agent, True))
        assert self._row(agent.last_task_id) == (0.5, 70, 1)

    def test_setup_failure_of_a_reused_plain_agent_reports_no_usage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A plain ``SorcarAgent`` starts its epoch before its setup too.

        The second run fails before ``_reset`` and makes no model call:
        its counters are its own, zero, not the first run's.  The run has
        no console (no printer, the default ``verbose``): its failure is
        returned, not printed; a verbose run prints it.
        """
        agent = SorcarAgent("setup-failure")
        srv, url = _start_server(_FinishHandler)
        settings: dict[str, Any] = {
            "model_name": "gpt-4o-mini",
            "work_dir": self.tmpdir,
            "web_tools": False,
            "append_basic_tools": False,
            "use_memory": False,
            "model_config": {"base_url": url, "api_key": "test-key"},
        }
        try:
            agent.run(prompt_template="say hi", **settings)
            assert agent.usage_snapshot()[0] > 0
            result = agent.run(
                prompt_template="say hi again",
                system_prompt_hook=_broken_system_prompt,
                **settings,
            )
            quiet = capsys.readouterr().out
            agent.run(
                prompt_template="say hi again",
                system_prompt_hook=_broken_system_prompt,
                verbose=True,
                **settings,
            )
        finally:
            srv.shutdown()
        assert yaml.safe_load(result)["success"] is False
        assert "system prompt failed" in result
        assert agent.usage_snapshot() == (0.0, 0, 0)
        assert quiet == ""
        assert "Status: FAILED" in capsys.readouterr().out
