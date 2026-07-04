"""End-to-end test for Attack 3: sanitizer confusion."""

from __future__ import annotations

from attacks import attack_sanitizer_confusion
from attacks.common import SECRET_MARKER, SECRET_MARKER_2, build_scenario

from ._util import run_attack


def test_sanitizer_list_is_trusted_by_name_not_by_semantics() -> None:
    scenario = build_scenario()
    workflow = attack_sanitizer_confusion.build(scenario)

    # Sanity: the attack modified the policy to declare
    # ``summarize_emails`` a sanitizer.
    assert "summarize_emails" in scenario.policy.taint_rules[0].sanitizers

    run_attack(scenario, workflow)

    # Two tool calls actually mattered for the sink: only ``send_email``
    # writes to exfil_log.
    assert len(scenario.exfil_log) == 1
    call = scenario.exfil_log[0]

    # ``body`` here is the fake-sanitizer output.  It concatenates every
    # raw field of the stolen inbox, so both markers appear verbatim.
    body_str = str(call["body"])
    assert SECRET_MARKER in body_str, (
        f"Expected M&A secret to leak via faux-sanitized body; got:\n{body_str!r}"
    )
    assert SECRET_MARKER_2 in body_str, (
        f"Expected HR secret to leak via faux-sanitized body; got:\n{body_str!r}"
    )
