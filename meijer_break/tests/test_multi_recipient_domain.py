"""End-to-end test for Attack 4: multi-recipient domain bypass."""

from __future__ import annotations

from guardians import verify

from attacks import attack_multi_recipient_domain
from attacks.common import SECRET_MARKER, SECRET_MARKER_2, build_scenario

from ._util import run_attack


def test_comma_separated_recipient_list_hides_external_address() -> None:
    scenario = build_scenario()
    workflow = attack_multi_recipient_domain.build(scenario)

    # The verifier accepts because both the Z3 condition encoding and the
    # automaton's safe_eval guard treat the whole string as having the final
    # @company.com suffix.
    result = verify(workflow, scenario.policy, scenario.registry)
    assert result.ok, (
        "Verifier unexpectedly rejected: "
        + "\n".join(f"  [{v.category}] {v.message}" for v in result.violations)
    )
    assert result.warnings == []

    run_attack(scenario, workflow)

    assert len(scenario.exfil_log) == 1
    call = scenario.exfil_log[0]

    # The sink received an address field containing an external recipient,
    # even though the no_external_send automaton approved it.
    assert call["to"] == attack_multi_recipient_domain.RECIPIENT_LIST
    assert attack_multi_recipient_domain.EXTERNAL_RECIPIENT in call["to"]
    assert attack_multi_recipient_domain.INTERNAL_COVER_RECIPIENT in call["to"]

    # The stolen inbox still left through the unchecked subject field.
    subject_repr = repr(call["subject"])
    assert SECRET_MARKER in subject_repr
    assert SECRET_MARKER_2 in subject_repr
