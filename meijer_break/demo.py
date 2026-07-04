"""Run every attack end-to-end and print the runtime exfiltration log.

Usage:
    source guardians/.venv/bin/activate
    python demo.py
"""

from __future__ import annotations

import textwrap

from guardians import verify, WorkflowExecutor

from attacks import (
    attack_subject_exfil,
    attack_type_confusion,
    attack_sanitizer_confusion,
    attack_multi_recipient_domain,
)
from attacks.common import build_scenario


_ATTACKS = [
    ("Attack 1 — subject-line exfil (sink_param='body' gap)",
     attack_subject_exfil),
    ("Attack 2 — type confusion (to=[] bypasses domain checks)",
     attack_type_confusion),
    ("Attack 3 — sanitizer confusion (name-trusted 'summarizer')",
     attack_sanitizer_confusion),
    ("Attack 4 — recipient-list parser gap (external plus internal)",
     attack_multi_recipient_domain),
]


def _fmt(value: object, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


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
            print(f"    ! {v.category}: {v.message}")
        if not result.ok:
            print("  (verifier rejected — attack fails)")
            continue

        WorkflowExecutor(
            scenario.registry, scenario.policy,
            auto_approve=True, verify_first=False,
        ).run(workflow)

        for i, call in enumerate(scenario.exfil_log, 1):
            print(f"  runtime send_email #{i}:")
            for k, v in call.items():
                print(textwrap.indent(f"{k}: {_fmt(v)}", "      "))


if __name__ == "__main__":
    main()
