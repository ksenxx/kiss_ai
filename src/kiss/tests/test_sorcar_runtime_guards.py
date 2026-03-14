"""Tests for Sorcar runtime guard helper behavior."""

from kiss.agents.sorcar.sorcar import _new_utility_agent, _should_warn_no_changes


def test_new_utility_agent_disables_trajectory_saving() -> None:
    agent = _new_utility_agent("Autocomplete")
    assert agent.save_trajectory is False


def test_warn_no_changes_only_for_completed_tasks() -> None:
    assert _should_warn_no_changes({"type": "task_done"}, {"error": "No changes"})
    assert not _should_warn_no_changes({"type": "task_error"}, {"error": "No changes"})
    assert not _should_warn_no_changes({"type": "task_done"}, {"status": "opened"})
    assert not _should_warn_no_changes(None, {"error": "No changes"})
