"""Run every attack end-to-end and show the framework now blocks it.

Usage:
    source guardians/.venv/bin/activate
    python demo.py

Before the fixes (see ``guardians/src/guardians/`` for the diff) each
attack produced ``verify.ok = True`` and delivered the secret marker to
``scenario.exfil_log``.  After the fixes:

* every attack has ``verify.ok = False`` with a specific violation, and
* the runtime executor refuses to run the workflow, so
  ``scenario.exfil_log`` stays empty.
"""

from __future__ import annotations

from guardians import verify, WorkflowExecutor, SecurityViolation

from attacks import (
    attack_subject_exfil,
    attack_type_confusion,
    attack_sanitizer_confusion,
    attack_multi_recipient_domain,
)
from attacks.common import SECRET_MARKER, SECRET_MARKER_2, build_scenario


_ATTACKS = [
    ("Attack 1 — subject-line exfil (verifier now sweeps all params)",
     attack_subject_exfil),
    ("Attack 2 — type confusion (concrete arg type + [] in [...] fix)",
     attack_type_confusion),
    ("Attack 3 — sanitizer confusion (redacts_labels contract required)",
     attack_sanitizer_confusion),
    ("Attack 4 — multi-recipient list (Z3 comma-reject + list-domain check)",
     attack_multi_recipient_domain),
]


def main() -> None:
    for title, mod in _ATTACKS:
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        scenario = build_scenario()
        workflow = mod.build(scenario)

        result = verify(workflow, scenario.policy, scenario.registry)
        print(f"verify.ok = {result.ok}   "
              f"violations={len(result.violations)}  "
              f"warnings={len(result.warnings)}")
        for w in result.warnings:
            print(f"    ~ warn: {w}")
        for v in result.violations:
            print(f"    ! [{v.category}] {v.message}")

        # The paper's canonical posture: executor refuses to run a workflow
        # that failed verification (``verify_first=True`` by default).
        try:
            WorkflowExecutor(
                scenario.registry, scenario.policy, auto_approve=True,
            ).run(workflow)
        except SecurityViolation as exc:
            print(f"  runtime executor refused: {exc}")

        # Prove the runtime sink stayed clean.
        leaked = scenario.leaked_content()
        secret_present = SECRET_MARKER in leaked or SECRET_MARKER_2 in leaked
        print(f"  exfil_log entries: {len(scenario.exfil_log)}   "
              f"secret_marker_leaked: {secret_present}")


if __name__ == "__main__":
    main()
