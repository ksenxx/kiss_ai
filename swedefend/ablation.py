"""Ablation study: measure per-layer and combined defense effectiveness."""

import csv
from pathlib import Path

from swedefend.corpus import generate_corpus
from swedefend.defense.intent_judge import IntentAlignmentJudge
from swedefend.defense.issue_sanitizer import IssueSanitizer
from swedefend.defense.provenance import ProvenanceScanner
from swedefend.pipeline import SWEDefendPipeline


def run_ablation(judge_model: str = "claude-sonnet-4-5") -> None:
    """Run ablation study and write results CSV."""
    corpus = generate_corpus()
    print(f"Running ablation on {len(corpus)} cases...")

    sanitizer = IssueSanitizer()
    provenance = ProvenanceScanner()
    judge = IntentAlignmentJudge(model_name=judge_model)

    results: dict[str, dict[str, float]] = {}

    n_mal = sum(1 for c in corpus if c.is_malicious)
    n_ben = sum(1 for c in corpus if not c.is_malicious)

    # Layer 1 only: sanitizer
    l1_blocked = 0
    l1_mal_blocked = 0
    for case in corpus:
        san_result = sanitizer.sanitize(case.issue_text)
        if san_result.is_suspicious:
            l1_blocked += 1
            if case.is_malicious:
                l1_mal_blocked += 1
    l1_catch = l1_mal_blocked / n_mal
    l1_fpr = (l1_blocked - l1_mal_blocked) / n_ben
    results["L1_sanitizer"] = {
        "blocked": l1_blocked,
        "catch_rate": l1_catch,
        "fpr": l1_fpr,
    }

    # Layer 3 only: provenance (skip L2 for ablation speed)
    l3_blocked = 0
    l3_mal_blocked = 0
    for case in corpus:
        prov_result = provenance.scan_source(case.patch_source)
        if prov_result.has_dangerous_sink:
            l3_blocked += 1
            if case.is_malicious:
                l3_mal_blocked += 1
    l3_catch = l3_mal_blocked / n_mal
    l3_fpr = (l3_blocked - l3_mal_blocked) / n_ben
    results["L3_provenance"] = {
        "blocked": l3_blocked,
        "catch_rate": l3_catch,
        "fpr": l3_fpr,
    }

    # Layer 4 only: judge
    l4_blocked = 0
    l4_mal_blocked = 0
    for i, case in enumerate(corpus):
        print(f"  L4 judge {i + 1}/{len(corpus)}", flush=True)
        judge_result = judge.judge(case.issue_text, case.patch_source)
        if not judge_result.aligned:
            l4_blocked += 1
            if case.is_malicious:
                l4_mal_blocked += 1
    l4_catch = l4_mal_blocked / n_mal
    l4_fpr = (l4_blocked - l4_mal_blocked) / n_ben
    results["L4_judge"] = {
        "blocked": l4_blocked,
        "catch_rate": l4_catch,
        "fpr": l4_fpr,
    }

    # Combined (all 4 layers)
    pipeline = SWEDefendPipeline(judge=judge)
    combined_blocked = 0
    combined_malicious_blocked = 0
    for i, case in enumerate(corpus):
        print(f"  Combined {i + 1}/{len(corpus)}", flush=True)
        verdict = pipeline.evaluate(
            issue_text=case.issue_text,
            patch_source=case.patch_source,
            repo_path=None,
            fail_to_pass=None,
            pass_to_pass=None,
        )
        if not verdict.allow:
            combined_blocked += 1
            if case.is_malicious:
                combined_malicious_blocked += 1

    combined_catch = combined_malicious_blocked / n_mal
    combined_fpr = (combined_blocked - combined_malicious_blocked) / n_ben
    results["Combined"] = {
        "blocked": combined_blocked,
        "catch_rate": combined_catch,
        "fpr": combined_fpr,
    }

    # Write ablation.csv
    out_dir = Path("swedefend/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_csv = out_dir / "ablation.csv"
    with ablation_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Blocked", "Catch_Rate", "FPR"])
        for layer_name, metrics in results.items():
            writer.writerow(
                [
                    layer_name,
                    metrics["blocked"],
                    f"{metrics['catch_rate']:.2%}",
                    f"{metrics['fpr']:.2%}",
                ]
            )

    print(f"Ablation results written to {ablation_csv}")
    print("\nAblation Summary:")
    for layer_name, metrics in results.items():
        print(f"  {layer_name}: catch={metrics['catch_rate']:.2%}, FPR={metrics['fpr']:.2%}")


if __name__ == "__main__":
    run_ablation()
