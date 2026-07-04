"""End-to-end test for Attack 3 (sanitizer confusion).

Post-fix expectation
--------------------
The verifier no longer accepts a name-based sanitizer listing.  A tool
declared as a sanitizer for a taint rule is only honored when its
``ToolSpec.redacts_labels`` covers the rule's source labels.

The break-scenario's ``summarize_emails`` deliberately has no
``redacts_labels`` claim, so:

* ``verify`` emits a ``sanitizer_contract`` violation.
* Because sanitization is not applied, taint propagates through the
  summary and the downstream ``send_email.body`` is caught by the plain
  taint rule as well.
* The executor, running with ``verify_first=False`` to isolate the
  runtime monitor, also raises ``SecurityViolation`` at the summarizer
  step — a defense-in-depth win.
"""

from __future__ import annotations

from attacks import attack_sanitizer_confusion
from attacks.common import SECRET_MARKER, SECRET_MARKER_2, build_scenario

from ._util import run_expected_blocked


def test_sanitizer_confusion_is_blocked() -> None:
    scenario = build_scenario()
    workflow = attack_sanitizer_confusion.build(scenario)

    outcome = run_expected_blocked(scenario, workflow)

    # Static verification must reject.
    assert outcome.verifier_blocked, (
        "Verifier must reject an undeclared sanitizer"
    )
    categories = {v.category for v in outcome.result.violations}
    assert "sanitizer_contract" in categories, (
        "Expected an explicit sanitizer_contract violation"
    )
    # And the plain taint rule must still fire, because sanitization is
    # not applied to a spec that made no redaction claim.
    assert "taint" in categories, (
        "Sanitization must not be applied when the redacts_labels contract "
        "is not met, so the downstream body taint must still fire"
    )

    # The runtime executor also refuses at the summarizer step: a mail
    # send never happened.
    assert outcome.runtime_blocked, (
        "Runtime monitor should also refuse the misdeclared sanitizer"
    )
    assert outcome.exfil_log_len == 0
    assert SECRET_MARKER not in outcome.leaked_content
    assert SECRET_MARKER_2 not in outcome.leaked_content
