"""End-to-end test for Attack 1: subject-line exfiltration."""

from __future__ import annotations

from attacks import attack_subject_exfil
from attacks.common import SECRET_MARKER, SECRET_MARKER_2, build_scenario

from ._util import run_attack


def test_subject_field_exfiltrates_despite_ok_verification() -> None:
    scenario = build_scenario()
    workflow = attack_subject_exfil.build(scenario)

    run_attack(scenario, workflow)

    # Exactly one send_email call reached the sink.
    assert len(scenario.exfil_log) == 1
    call = scenario.exfil_log[0]

    # ``to`` cleared the domain automaton, so it looks like a
    # legitimate internal address.
    assert call["to"] == "attacker@company.com"

    # The body carried nothing sensitive — that is what fooled the rule.
    assert call["body"] == "See subject."

    # The subject carried the full stolen inbox verbatim.
    subject_repr = repr(call["subject"])
    assert SECRET_MARKER in subject_repr, (
        "Expected the first mail body to leak through the subject line, "
        f"got: {subject_repr!r}"
    )
    assert SECRET_MARKER_2 in subject_repr, (
        "Expected the second mail body to leak through the subject line, "
        f"got: {subject_repr!r}"
    )

    # Sanity: the leaked_content oracle also flags this as a leak.
    leaked = scenario.leaked_content()
    assert SECRET_MARKER in leaked
    assert SECRET_MARKER_2 in leaked
