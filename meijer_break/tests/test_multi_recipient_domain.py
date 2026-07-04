"""End-to-end test for Attack 4 (comma-separated recipient list).

Post-fix expectation
--------------------
* The Z3 precondition encoding for ``domain_of(to) in allowed_domains``
  now requires ``Not(Contains(to, ","))``, so a string with a comma is
  rejected at verification time.
* The runtime ``safe_eval._domain_of_str`` now splits multi-recipient
  strings on ``,`` and returns a *list* of domains; ``in``/``not in``
  over list-in-list checks all elements, so the automaton fires on the
  attacker's external address.
* The taint side-channel on ``subject`` is also caught (Fix 1).
"""

from __future__ import annotations

from attacks import attack_multi_recipient_domain
from attacks.common import SECRET_MARKER, SECRET_MARKER_2, build_scenario

from ._util import run_expected_blocked


def test_multi_recipient_domain_is_blocked() -> None:
    scenario = build_scenario()
    workflow = attack_multi_recipient_domain.build(scenario)

    outcome = run_expected_blocked(scenario, workflow)

    assert outcome.verifier_blocked, (
        "Verifier must reject a to=... string containing a comma"
    )

    categories = {v.category for v in outcome.result.violations}
    # (a) Z3 precondition catches the comma.
    assert "precondition" in categories, (
        "Expected the strengthened Z3 precondition to reject the comma"
    )
    # (b) The automaton reaches error via the safe_eval list-in-list
    # semantics: ``[evil.com, company.com] not in [company.com]`` is True.
    assert "automaton" in categories, (
        "Expected the domain automaton to fire on the multi-recipient list"
    )

    # No exfiltration.
    assert outcome.exfil_log_len == 0
    assert SECRET_MARKER not in outcome.leaked_content
    assert SECRET_MARKER_2 not in outcome.leaked_content
