"""Shared assertion helpers for the attack test-suite."""

from __future__ import annotations

from guardians import (
    verify, WorkflowExecutor, SecurityViolation, Workflow,
)

from attacks.common import Scenario


def run_attack(scenario: Scenario, workflow: Workflow) -> None:
    """Verify then execute a workflow.

    * ``result.ok`` must be True (verifier greenlights the attack).
    * The executor must run to completion (the runtime does not
      re-run taint rules, so there is no second line of defence).
    """
    result = verify(workflow, scenario.policy, scenario.registry)
    assert result.ok, (
        "Attack failed: verifier rejected the workflow — expected accept.\n"
        + "\n".join(f"  [{v.category}] {v.message}" for v in result.violations)
    )
    # ``verify_first=False`` avoids re-running the same verifier; if the
    # executor's monitor rejects, that's still a defensive win we want
    # the test to fail on.
    executor = WorkflowExecutor(
        scenario.registry, scenario.policy,
        auto_approve=True, verify_first=False,
    )
    try:
        executor.run(workflow)
    except SecurityViolation as e:
        raise AssertionError(f"Runtime monitor blocked the attack: {e}") from e
