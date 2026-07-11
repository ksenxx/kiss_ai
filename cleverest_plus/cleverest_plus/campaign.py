"""Campaign driver: run the fixed-budget portfolio over every Issue and write results."""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .benchmark import Issue, load_benchmark
from .dispatcher import PortfolioMember, run_issue
from .loop import TrialResult


@dataclass(frozen=True)
class CampaignConfig:
    """Configuration for one end-to-end campaign."""

    benchmark_root: Path
    output_dir: Path
    total_trials: int
    timeout_s: float = 30.0


def _serialize_trial(trial: TrialResult) -> dict[str, object]:
    return {
        "issue_id": trial.issue_id,
        "portfolio_member": trial.portfolio_member,
        "model": trial.model,
        "trial_index": trial.trial_index,
        "ordinal_code": trial.ordinal_code,
        "outcome": trial.outcome,
        "iterations": [dataclasses.asdict(i) for i in trial.iterations],
    }


def _issue_summary(issue: Issue, trials: Sequence[TrialResult]) -> dict[str, object]:
    codes = [t.ordinal_code for t in trials]
    return {
        "issue_id": issue.identifier,
        "subject": issue.subject,
        "bug_id": issue.bug_id,
        "scenario": issue.scenario,
        "trials_run": len(codes),
        "mean_code": sum(codes) / len(codes) if codes else 0.0,
        "bug_any": int(any(c == 3 for c in codes)),
        "outcomes": [t.outcome for t in trials],
        "portfolio_breakdown": {
            member: sum(1 for t in trials if t.portfolio_member == member and t.ordinal_code == 3)
            for member in {t.portfolio_member for t in trials}
        },
    }


def run_campaign(
    config: CampaignConfig,
    portfolio: Sequence[PortfolioMember],
) -> dict[str, object]:
    """Run the campaign, write per-issue and campaign JSONL/JSON, return summary."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = config.output_dir / "trials.jsonl"
    summary_path = config.output_dir / "campaign_summary.json"
    issues = list(load_benchmark(config.benchmark_root))
    if not issues:
        raise FileNotFoundError(f"no issues found under {config.benchmark_root}")

    per_issue_summaries: list[dict[str, object]] = []
    started_at = time.time()
    with trials_path.open("w") as trials_fh:
        for issue_idx, issue in enumerate(issues, start=1):
            print(f"[campaign] issue {issue_idx}/{len(issues)}: {issue.identifier}", file=sys.stderr, flush=True)
            issue_tmp = config.output_dir / "work" / issue.subject / issue.bug_id / issue.scenario
            issue_started = time.time()
            def _emit(r: TrialResult, _fh=trials_fh, _iid=issue.identifier) -> None:
                _fh.write(json.dumps(_serialize_trial(r)) + "\n")
                _fh.flush()
                print(f"[campaign] {_iid} trial={r.trial_index} member={r.portfolio_member} "
                      f"outcome={r.outcome} code={r.ordinal_code} iters={len(r.iterations)}",
                      file=sys.stderr, flush=True)
            trials = run_issue(
                issue=issue,
                portfolio=portfolio,
                total_trials=config.total_trials,
                tmp_root=issue_tmp,
                timeout_s=config.timeout_s,
                on_trial=_emit,
            )
            print(f"[campaign] issue {issue.identifier} done in {time.time()-issue_started:.0f}s",
                  file=sys.stderr, flush=True)
            per_issue_summaries.append(_issue_summary(issue, trials))

    bic = [s for s in per_issue_summaries if s["scenario"] == "BIC"]
    fix = [s for s in per_issue_summaries if s["scenario"] == "FIX"]
    summary: dict[str, object] = {
        "config": {
            "benchmark_root": str(config.benchmark_root),
            "output_dir": str(config.output_dir),
            "total_trials": config.total_trials,
            "timeout_s": config.timeout_s,
            "portfolio": [
                {"name": m.name, "iteration_cap": m.iteration_cap} for m in portfolio
            ],
        },
        "duration_s": time.time() - started_at,
        "counts": {
            "issues": len(per_issue_summaries),
            "bic_issues": len(bic),
            "fix_issues": len(fix),
            "bic_bug_any": sum(int(s["bug_any"]) for s in bic),
            "fix_bug_any": sum(int(s["bug_any"]) for s in fix),
        },
        "per_issue": per_issue_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
