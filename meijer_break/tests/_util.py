"""Shared assertion helpers for the attack test-suite.

After the framework fixes (see ``guardians/src/guardians``) every attack
must now be *blocked*, either by ``verify`` returning ``ok=False`` or by
the runtime executor raising a ``SecurityViolation``.  Both are valid
defensive wins; a proper defense-in-depth build catches at both layers.
"""

from __future__ import annotations

from dataclasses import dataclass

from guardians import (
    verify, WorkflowExecutor, SecurityViolation, Workflow,
    VerificationResult,
)

from attacks.common import Scenario


@dataclass
class BlockedAttack:
    """Diagnostics captured while confirming an attack is blocked."""

    result: VerificationResult
    runtime_violation: SecurityViolation | None
    exfil_log_len: int
    leaked_content: str

    @property
    def verifier_blocked(self) -> bool:
        return not self.result.ok

    @property
    def runtime_blocked(self) -> bool:
        return self.runtime_violation is not None


def run_expected_blocked(scenario: Scenario, workflow: Workflow) -> BlockedAttack:
    """Verify + attempt to execute a workflow expected to be blocked.

    Returns diagnostics.  Callers assert on the returned object that:

    * the attack is caught by at least one of (verifier, runtime), and
    * the secret marker never appears in ``scenario.exfil_log``.
    """
    result = verify(workflow, scenario.policy, scenario.registry)

    # Drive the executor with the paper's intended defense-in-depth
    # posture (``verify_first=True``): a rejected workflow raises
    # ``SecurityViolation`` before any tool body runs.  A workflow that
    # somehow slipped past ``verify`` would still get a chance to be
    # blocked by the runtime monitor's own precondition / automaton
    # checks.  Either level counts as a defensive win, and neither may
    # permit the sensitive marker to reach ``scenario.exfil_log``.
    runtime_violation: SecurityViolation | None = None
    executor = WorkflowExecutor(
        scenario.registry, scenario.policy,
        auto_approve=True, verify_first=True,
    )
    try:
        executor.run(workflow)
    except SecurityViolation as exc:
        runtime_violation = exc

    return BlockedAttack(
        result=result,
        runtime_violation=runtime_violation,
        exfil_log_len=len(scenario.exfil_log),
        leaked_content=scenario.leaked_content(),
    )
