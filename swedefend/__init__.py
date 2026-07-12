# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SWEDefend — a 4-layer defense harness against SWExploit attacks on SWE-agents.

A SWE-agent takes an *issue* (a natural-language problem statement, often mirrored
verbatim from an untrusted GitHub issue) plus a *codebase* and produces a *patch*.
:mod:`swedefend.swexploit` shows how an attacker weaponizes the issue text
(indirect prompt injection, OWASP LLM01) or the patch itself (command-injection /
data-exfiltration backdoors, OWASP CWE-77/78) to make the agent emit malicious code.

:class:`swedefend.pipeline.SWEDefendPipeline` chains four independent defenses:

1. :class:`~swedefend.defense.issue_sanitizer.IssueSanitizer` — strips and flags
   instruction-override / injection text in the (untrusted) issue.
2. :class:`~swedefend.defense.masked_reproduction.MaskedReproductionRepair` — masks
   untrusted spans, then verifies the patch against the *behavioral* reproduction
   (tests) instead of trusting attacker-supplied reproduction steps.
3. :class:`~swedefend.defense.provenance.ProvenanceScanner` — runs the verified
   ``./.venv/bin/bandit`` and ``./.venv/bin/semgrep`` binaries over the patched
   code and maps dangerous taint sinks (shell, eval, pickle, network exfil) to a
   provenance report.
4. :class:`~swedefend.defense.intent_judge.IntentAlignmentJudge` — an LLM-as-judge
   that vetoes patches whose behavior is misaligned with the *sanitized* issue
   intent (backdoors, exfil, scope creep).
"""

from swedefend.defense.intent_judge import IntentAlignmentJudge, IntentVerdict
from swedefend.defense.issue_sanitizer import IssueSanitizer, SanitizationResult
from swedefend.defense.masked_reproduction import (
    MaskedReproductionRepair,
    ReproductionResult,
)
from swedefend.defense.provenance import ProvenanceFinding, ProvenanceScanner
from swedefend.pipeline import LayerResult, PipelineVerdict, SWEDefendPipeline
from swedefend.swexploit import AttackPayload, SWExploitHarness, SWExploitReport

__all__ = [
    "AttackPayload",
    "IntentAlignmentJudge",
    "IntentVerdict",
    "IssueSanitizer",
    "LayerResult",
    "MaskedReproductionRepair",
    "PipelineVerdict",
    "ProvenanceFinding",
    "ProvenanceScanner",
    "ReproductionResult",
    "SWExploitHarness",
    "SWExploitReport",
    "SWEDefendPipeline",
    "SanitizationResult",
]
