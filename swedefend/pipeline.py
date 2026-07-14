# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The SWEDefend defense pipeline — chains the four layers into one verdict.

Given an (untrusted issue, produced patch, optional patched repo) triple the
pipeline runs, in order:

1. issue sanitizer, 2. masked-reproduction repair (only when a behavioral
contract is supplied), 3. taint/dataflow provenance, 4. intent-alignment judge
(only when a judge is supplied).  It returns a single allow/deny
:class:`PipelineVerdict` with per-layer reasons, failing closed the moment any
layer objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from swedefend.defense.intent_judge import IntentAlignmentJudge
from swedefend.defense.issue_sanitizer import IssueSanitizer
from swedefend.defense.masked_reproduction import MaskedReproductionRepair
from swedefend.defense.provenance import ProvenanceScanner


@dataclass
class LayerResult:
    """The outcome of one defense layer.

    Attributes:
        name: The layer's name.
        passed: ``True`` when the layer raised no objection.
        reasons: Human-readable findings from the layer.
    """

    name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class PipelineVerdict:
    """The combined verdict across all four layers.

    Attributes:
        allow: ``True`` only when every executed layer passed.
        layers: Per-layer :class:`LayerResult` objects, in execution order.
        sanitized_issue: The sanitized issue text produced by layer 1.
    """

    allow: bool
    layers: list[LayerResult] = field(default_factory=list)
    sanitized_issue: str = ""

    @property
    def blocking_layers(self) -> list[str]:
        """Return the names of layers that objected."""
        return [layer.name for layer in self.layers if not layer.passed]


class SWEDefendPipeline:
    """Chain the four SWEDefend layers into a single allow/deny decision."""

    def __init__(
        self,
        judge: IntentAlignmentJudge | None = None,
        provenance: ProvenanceScanner | None = None,
        sanitizer: IssueSanitizer | None = None,
        reproduction: MaskedReproductionRepair | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            judge: Optional intent-alignment judge (layer 4).  When ``None`` the
                judge layer is skipped (useful for offline unit tests).
            provenance: Provenance scanner (layer 3).  Defaults to a fresh
                :class:`ProvenanceScanner`.
            sanitizer: Issue sanitizer (layer 1).  Defaults to a fresh
                :class:`IssueSanitizer`.
            reproduction: Masked-reproduction repair (layer 2).  Defaults to a
                fresh :class:`MaskedReproductionRepair`.
        """
        self.sanitizer = sanitizer or IssueSanitizer()
        self.reproduction = reproduction or MaskedReproductionRepair()
        self.provenance = provenance or ProvenanceScanner()
        self.judge = judge

    def evaluate(
        self,
        issue_text: str,
        patch_source: str,
        repo_path: Path | None = None,
        fail_to_pass: list[str] | None = None,
        pass_to_pass: list[str] | None = None,
    ) -> PipelineVerdict:
        """Run all four layers and return the combined verdict.

        Args:
            issue_text: The raw, untrusted issue / problem statement.
            patch_source: The Python source of the patched code (what the agent
                would commit).  Scanned by provenance and shown to the judge.
            repo_path: Optional directory of the patched repo, required to run
                the masked-reproduction behavioral checks.
            fail_to_pass: Optional behavioral checks the patch must make pass.
                When omitted (or ``repo_path`` is ``None``) layer 2 is skipped.
            pass_to_pass: Optional regression checks that must remain passing.

        Returns:
            The combined :class:`PipelineVerdict`.
        """
        layers: list[LayerResult] = []

        sanitation = self.sanitizer.sanitize(issue_text)
        layers.append(
            LayerResult(
                name="issue_sanitizer",
                passed=not sanitation.is_suspicious,
                reasons=list(sanitation.flags),
            )
        )

        if repo_path is not None and fail_to_pass:
            repro = self.reproduction.verify(
                sanitation.sanitized_text, repo_path, fail_to_pass, pass_to_pass
            )
            layers.append(
                LayerResult(
                    name="masked_reproduction",
                    passed=repro.repaired,
                    reasons=list(repro.details),
                )
            )

        report = self.provenance.scan_source(patch_source)
        prov_reasons = [
            f"{f.scanner}:{f.rule_id} [{f.severity}] {f.category} @L{f.line}: {f.message}"
            for f in report.findings
        ]
        prov_reasons.extend(report.scanner_errors)
        layers.append(
            LayerResult(
                name="provenance",
                passed=not report.has_dangerous_sink,
                reasons=prov_reasons,
            )
        )

        if self.judge is not None:
            verdict = self.judge.judge(sanitation.sanitized_text, patch_source)
            # A judge veto counts only when its confidence is above the
            # configured threshold; a low-confidence "unsure" verdict is *not*
            # treated as a block.  Under the AI-Discovery corroboration knob
            # (D1) we additionally require the veto to be corroborated either
            # by a named added-capability / removed-control in the verdict
            # itself or by an independent static signal from L1 or L3 (already
            # accumulated in ``layers`` above).
            static_evidence = any(not layer.passed for layer in layers)
            if self.judge.require_corroboration:
                passed = not verdict.should_veto_with_corroboration(
                    self.judge.confidence_threshold,
                    static_evidence=static_evidence,
                )
            else:
                passed = not verdict.should_veto(self.judge.confidence_threshold)
            reason = (
                f"aligned={verdict.aligned} conf={verdict.confidence:.2f} "
                f"tau={self.judge.confidence_threshold:.2f} "
                f"added={verdict.added_capabilities} "
                f"removed={verdict.removed_controls}: {verdict.reason}"
            )
            layers.append(
                LayerResult(name="intent_judge", passed=passed, reasons=[reason])
            )

        allow = all(layer.passed for layer in layers)
        return PipelineVerdict(
            allow=allow, layers=layers, sanitized_issue=sanitation.sanitized_text
        )
