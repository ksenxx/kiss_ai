"""End-to-end test for Attack 2 (``to=[]`` type confusion).

Post-fix expectation
--------------------
Two independent fixes each block the attack, so we assert both fire:

1. ``verify`` runs a concrete argument-type check.  ``to=[]`` violates
   ``ParamSpec.type='str'`` and is rejected as ``type_mismatch``.
2. ``safe_eval`` no longer treats ``[] in [...]`` as True (vacuous
   subset), so the automaton's ``domain_of(to) not in allowed_domains``
   guard fires; the automaton reaches its error state.
"""

from __future__ import annotations

from attacks import attack_type_confusion
from attacks.common import SECRET_MARKER, SECRET_MARKER_2, build_scenario

from ._util import run_expected_blocked


def test_type_confusion_is_blocked() -> None:
    scenario = build_scenario()
    workflow = attack_type_confusion.build(scenario)

    outcome = run_expected_blocked(scenario, workflow)

    # Static verification must reject.
    assert outcome.verifier_blocked, "Verifier must reject to=[]"

    categories = {v.category for v in outcome.result.violations}

    # (a) Explicit type-mismatch diagnostic on the ``to`` argument.
    assert "type_mismatch" in categories, (
        "Expected a type_mismatch violation for the list-into-str slot"
    )

    # (b) The automaton's error state must also be reachable now that
    # ``[] not in [...]`` evaluates correctly.
    assert "automaton" in categories, (
        "Expected the domain-allowlist automaton to reach its error state "
        "once ``[] in [...]`` is no longer treated as vacuously true"
    )

    # No exfiltration ever happened.
    assert outcome.exfil_log_len == 0
    assert SECRET_MARKER not in outcome.leaked_content
    assert SECRET_MARKER_2 not in outcome.leaked_content
