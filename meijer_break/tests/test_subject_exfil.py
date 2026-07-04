"""End-to-end test for Attack 1 (subject-line exfiltration).

Post-fix expectation
--------------------
The verifier's taint pass now enforces the taint rule over *every*
argument of the sink tool, not only the ``sink_param`` named in the
policy.  Data from ``fetch_mail`` reaching ``send_email.subject`` is
therefore rejected even though the paper's canonical rule uses
``sink_param="body"``.
"""

from __future__ import annotations

from attacks import attack_subject_exfil
from attacks.common import SECRET_MARKER, SECRET_MARKER_2, build_scenario

from ._util import run_expected_blocked


def test_subject_field_exfiltration_is_blocked() -> None:
    scenario = build_scenario()
    workflow = attack_subject_exfil.build(scenario)

    outcome = run_expected_blocked(scenario, workflow)

    # Static verification must reject the attack.
    assert outcome.verifier_blocked, (
        "Expected the verifier to reject subject-field exfiltration"
    )

    # The taint diagnostic must name the specific side-channel that leaked
    # (subject) rather than only the paper's declared sink (body).
    taint = [v for v in outcome.result.violations if v.category == "taint"]
    assert taint, "Expected a taint violation for the subject side-channel"
    assert any("subject" in v.message for v in taint), (
        f"Expected a taint violation naming 'subject'; got: "
        + "; ".join(v.message for v in taint)
    )

    # The workflow never executed a `send_email`, so no secret material
    # can have leaked to the runtime sink.
    assert outcome.exfil_log_len == 0
    assert SECRET_MARKER not in outcome.leaked_content
    assert SECRET_MARKER_2 not in outcome.leaked_content
