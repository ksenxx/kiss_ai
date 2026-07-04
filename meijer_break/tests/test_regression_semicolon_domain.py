"""Round-2 regression test for non-comma recipient-list separators."""

from __future__ import annotations

from attacks import attack_regression_semicolon_domain
from attacks.common import build_scenario

from ._util import run_expected_blocked


def _assert_domain_policy_blocked(to: str) -> None:
    scenario = build_scenario()
    workflow = attack_regression_semicolon_domain.build_workflow(to)

    outcome = run_expected_blocked(scenario, workflow)

    assert outcome.verifier_blocked, (
        f"Verifier must reject recipient-list separator bypass: {to!r}"
    )
    categories = {v.category for v in outcome.result.violations}
    assert "precondition" in categories, (
        "Expected the strengthened Z3 precondition to reject the separator"
    )
    assert "automaton" in categories, (
        "Expected the domain automaton to fire on the external recipient"
    )
    assert outcome.exfil_log_len == 0


def test_semicolon_recipient_list_is_blocked() -> None:
    _assert_domain_policy_blocked(
        attack_regression_semicolon_domain.SEMICOLON_RECIPIENT_LIST
    )


def test_whitespace_recipient_list_is_blocked() -> None:
    _assert_domain_policy_blocked(
        attack_regression_semicolon_domain.WHITESPACE_RECIPIENT_LIST
    )
