"""Run quick ablation on a small subset for speed."""

import csv
from pathlib import Path

from swedefend.corpus import generate_corpus
from swedefend.defense.issue_sanitizer import IssueSanitizer
from swedefend.defense.provenance import ProvenanceScanner
from swedefend.pipeline import SWEDefendPipeline

corpus = generate_corpus()
# Take first 10 malicious + 5 benign for quick eval
subset = [c for c in corpus if c.is_malicious][:10] + [c for c in corpus if not c.is_malicious][:5]
print(f"Quick eval on {len(subset)} cases (10 mal + 5 benign)...")

sanitizer = IssueSanitizer()
provenance = ProvenanceScanner()

# L1 only
l1_mal_blocked = sum(
    1 for c in subset if c.is_malicious and sanitizer.sanitize(c.issue_text).is_suspicious
)
l1_ben_blocked = sum(
    1 for c in subset if not c.is_malicious and sanitizer.sanitize(c.issue_text).is_suspicious
)

# L3 only
l3_mal_blocked = sum(
    1
    for c in subset
    if c.is_malicious and provenance.scan_source(c.patch_source).has_dangerous_sink
)
l3_ben_blocked = sum(
    1
    for c in subset
    if not c.is_malicious and provenance.scan_source(c.patch_source).has_dangerous_sink
)

# Combined (L1+L3, skip L2/L4 for speed)
pipeline_basic = SWEDefendPipeline(judge=None)  # no judge
combined_mal_blocked = sum(
    1
    for c in subset
    if c.is_malicious
    and not pipeline_basic.evaluate(c.issue_text, c.patch_source, None, None, None).allow
)
combined_ben_blocked = sum(
    1
    for c in subset
    if not c.is_malicious
    and not pipeline_basic.evaluate(c.issue_text, c.patch_source, None, None, None).allow
)

n_mal = sum(1 for c in subset if c.is_malicious)
n_ben = sum(1 for c in subset if not c.is_malicious)

results = [
    ("L1_sanitizer", l1_mal_blocked, l1_mal_blocked / n_mal, l1_ben_blocked / n_ben),
    ("L3_provenance", l3_mal_blocked, l3_mal_blocked / n_mal, l3_ben_blocked / n_ben),
    (
        "Combined_L1+L3",
        combined_mal_blocked,
        combined_mal_blocked / n_mal,
        combined_ben_blocked / n_ben,
    ),
]

out_dir = Path("swedefend/results")
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "ablation_quick.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Layer", "Mal_Blocked", "Catch_Rate", "FPR"])
    for layer, mal_blk, catch, fpr in results:
        writer.writerow([layer, mal_blk, f"{catch:.2%}", f"{fpr:.2%}"])

print("\nQuick Ablation Summary:")
for layer, mal_blk, catch, fpr in results:
    print(f"  {layer}: catch={catch:.2%}, FPR={fpr:.2%}")
print("\nWrote swedefend/results/ablation_quick.csv")
