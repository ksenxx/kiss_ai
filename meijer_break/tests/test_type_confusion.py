"""End-to-end test for Attack 2: type-confusion on the ``to`` parameter."""

from __future__ import annotations

from guardians import verify

from attacks import attack_type_confusion
from attacks.common import SECRET_MARKER, build_scenario

from ._util import run_attack


def test_to_equals_empty_list_bypasses_domain_automaton() -> None:
    scenario = build_scenario()
    workflow = attack_type_confusion.build(scenario)

    # First: the verifier accepts.  The Z3 backend gives up on
    # ``domain_of(to) in allowed_domains`` (only a warning, not a
    # violation) and the automaton guard evaluates to False under
    # safe_eval because ``[] not in [...]`` is False.
    result = verify(workflow, scenario.policy, scenario.registry)
    assert result.ok, (
        "Verifier unexpectedly rejected: "
        + "\n".join(f"  [{v.category}] {v.message}" for v in result.violations)
    )

    # A skipped precondition surfaces only as a warning.
    warnings_str = "\n".join(result.warnings)
    assert "Could not parse precondition" in warnings_str, (
        "Expected a 'skipped precondition' warning — the verifier is "
        "silently discarding the domain check. "
        f"Got warnings: {result.warnings!r}"
    )

    # Runtime: executor happily calls send_email with ``to=[]`` — no
    # automaton fires, no precondition failure, no taint recheck.
    run_attack(scenario, workflow)

    assert len(scenario.exfil_log) == 1
    call = scenario.exfil_log[0]

    # The domain-check said [] was a valid recipient set.
    assert call["to"] == []

    # And the payload still shipped in the subject.
    assert SECRET_MARKER in repr(call["subject"])
