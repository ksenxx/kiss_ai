#!/usr/bin/env python3
"""Independent, deterministic re-analysis of the Cleverest artifact.

This script uses only released CSV/SUMMARY data. It never calls an LLM and never reads
benchmark truth inputs. Outputs are machine-readable JSON/CSV and two report figures.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "tmp" / "cleverest"
CSV = ART / "figure" / "aggregated_results.csv"
RAW = ROOT / "tmp" / "repdata" / "repdata_fse26rev"
OUT = Path(__file__).resolve().parent

rows = list(csv.DictReader(CSV.open()))
message_data = yaml.safe_load((ART / "msg/gemini.yaml").read_text())
modified_keys: dict[str, set[tuple[str, str]]] = {
    "enhanced_message": set(),
    "reduced_message": set(),
}
for item in message_data:
    subject = "php-src" if item["proj"] == "php" else item["proj"]
    key = (str(subject), str(item["issue"]))
    if item.get("msg_enhanced") is not None:
        modified_keys["enhanced_message"].add(key)
    if item.get("msg_reduced") is not None:
        modified_keys["reduced_message"].add(key)


def is_default(r: dict[str, str]) -> bool:
    return (
        r["git_info"] == "FULL"
        and r["max_iter"] == "5"
        and r["LLM"] == "gpt-4o-2024-08-06"
        and r["LLM_temp"] == "0.5"
        and not r["NOFEEDBACK"]
        and not r["GENCMD"]
    )


def config_name(r: dict[str, str]) -> str | None:
    if is_default(r):
        return "default"
    if r["git_info"] == "MSGONLY" and r["LLM"] == "gpt-4o-2024-08-06":
        return "message_only"
    if r["git_info"] == "DIFFONLY" and r["LLM"] == "gpt-4o-2024-08-06":
        return "diff_only"
    if r["git_info"] == "ENHANCED":
        return "enhanced_message"
    if r["git_info"] == "REDUCED":
        return "reduced_message"
    if r["LLM"] == "deepseek-r1":
        return "deepseek_r1"
    if r["LLM"] == "gpt-4o-mini":
        return "gpt4o_mini"
    if r["LLM_temp"] == "1.0":
        return "temperature_1"
    if r["max_iter"] == "10":
        return "iterations_10"
    if r["NOFEEDBACK"]:
        return "no_feedback"
    if r["GENCMD"]:
        return "generated_command"
    return None


def group_issue(rs: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rs:
        grouped[(r["subject"], r["iid"])].append(r)
    return {
        k: {
            "mean_score": float(np.mean([int(x["score"]) for x in v])),
            "success_trials": float(sum(int(x["success"]) for x in v)),
            "bug_any": float(any(x["success"] == "1" for x in v)),
            "reach_any": float(any(int(x["score"]) >= 1 for x in v)),
            "change_any": float(any(int(x["score"]) >= 2 for x in v)),
            "mean_time": float(np.mean([int(x["time"]) for x in v])),
        }
        for k, v in grouped.items()
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int = 260710) -> list[float]:
    rng = np.random.default_rng(seed)
    sims = np.mean(rng.choice(values, size=(50_000, len(values)), replace=True), axis=1)
    return [
        float(np.quantile(sims, 0.025)),
        float(np.mean(values)),
        float(np.quantile(sims, 0.975)),
    ]


def paired_compare(
    base: dict[Any, dict[str, float]], other: dict[Any, dict[str, float]]
) -> dict[str, Any]:
    keys = sorted(set(base) & set(other))
    diffs = np.array([other[k]["mean_score"] - base[k]["mean_score"] for k in keys])
    nonzero = diffs[diffs != 0]
    p_score = 1.0 if not len(nonzero) else float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    b = sum(base[k]["bug_any"] == 1 and other[k]["bug_any"] == 0 for k in keys)
    c = sum(base[k]["bug_any"] == 0 and other[k]["bug_any"] == 1 for k in keys)
    p_bug = (
        1.0
        if b + c == 0
        else float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    )
    return {
        "n_issues": len(keys),
        "mean_score_difference_ci95": bootstrap_mean_ci(diffs),
        "wilcoxon_p_unadjusted": p_score,
        "bug_any_lost": b,
        "bug_any_gained": c,
        "mcnemar_exact_p_unadjusted": p_bug,
    }


summary: dict[str, Any] = {
    "artifact_csv_sha256": __import__("hashlib").sha256(CSV.read_bytes()).hexdigest(),
    "rows": len(rows),
    "default": {},
    "ablations": {},
    "manual_transformations": {},
}

issue_data: dict[tuple[str, str], dict[str, dict[tuple[str, str], dict[str, float]]]] = {}
for scenario in ("BIC", "FIX"):
    configs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["scenario"] == scenario and (name := config_name(r)):
            configs[name].append(r)
    issue_data[(scenario, "all")] = {k: group_issue(v) for k, v in configs.items()}
    base_rows = configs["default"]
    base = group_issue(base_rows)
    summary["default"][scenario] = {
        "rows": len(base_rows),
        "issues": len(base),
        "mean_ordinal_score": float(np.mean([int(x["score"]) for x in base_rows])),
        "bug_success_trials": sum(int(x["success"]) for x in base_rows),
        "bugs_any_of_ten": int(sum(v["bug_any"] for v in base.values())),
        "reached_any_of_ten": int(sum(v["reach_any"] for v in base.values())),
        "changed_any_of_ten": int(sum(v["change_any"] for v in base.values())),
        "mean_seconds": float(np.mean([int(x["time"]) for x in base_rows])),
    }
    summary["ablations"][scenario] = {}
    for name, rs in configs.items():
        grouped = group_issue(rs)
        record: dict[str, Any] = {
            "rows": len(rs),
            "issues": len(grouped),
            "mean_ordinal_score": float(np.mean([int(x["score"]) for x in rs])),
            "bugs_any_of_ten": int(sum(v["bug_any"] for v in grouped.values())),
            "bug_success_trials": sum(int(x["success"]) for x in rs),
        }
        if name != "default":
            reference_name = "message_only" if name in modified_keys else "default"
            reference = group_issue(configs[reference_name])
            if name in modified_keys:
                selected = modified_keys[name]
                reference = {k: v for k, v in reference.items() if k in selected}
                grouped_for_comparison = {k: v for k, v in grouped.items() if k in selected}
            else:
                grouped_for_comparison = grouped
            record[f"versus_{reference_name}"] = paired_compare(reference, grouped_for_comparison)
        summary["ablations"][scenario][name] = record

# Retrospective union. Explicitly an upper bound, not a prospective score.
for scenario in ("BIC", "FIX"):
    cfgs = issue_data[(scenario, "all")]
    baseline = {k for k, v in cfgs["default"].items() if v["bug_any"]}
    all_union = {k for g in cfgs.values() for k, v in g.items() if v["bug_any"]}
    summary.setdefault("retrospective_ceiling", {})[scenario] = {
        "default": len(baseline),
        "union_all_released_llm_configs": len(all_union),
        "literal_double_target": 2 * len(baseline),
        "is_double": len(all_union) >= 2 * len(baseline),
        "warning": (
            "post-hoc union uses more compute and test outcomes; not a valid prospective score"
        ),
    }

# Quantify gencsv.py's data-dependent corrections from the 1,500 subject-level summaries.
transforms = Counter()
for path in RAW.glob("*/SUMMARY_*.txt"):
    lines = path.read_text(errors="replace").splitlines()
    conf: dict[str, str] = {}
    for line in lines[:8]:
        if ":" in line:
            key, value = line.split(":", 1)
            conf[key] = value.strip()
    for line in lines[11:-2]:
        if not line.strip() or "|" not in line:
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 5:
            continue
        commit, statuses, final = parts[-4], parts[-3], parts[-2]
        if "behave" in statuses and final == "R":
            transforms["raw_R_changed_to_D"] += 1
        if final == "X" and commit in {"b7e3bae", "d7e2125"}:
            transforms["X_manual_reach_score"] += 1
        if final == "X":
            for status in statuses.split():
                if status.startswith("bug_") and "^" in status:
                    before, after = status[4:].split("^", 1)
                    if before != after:
                        transforms["X_different_crashes_changed_to_score_2"] += 1
                        break
        if conf.get("GIT_INFO") == "MSGONLY" and commit == "907d05a" and final == "D":
            transforms["nonreproducible_D_changed_to_N"] += 1
summary["manual_transformations"] = dict(transforms)

# Table 5's explicit time assignment, independently read from the notebook source.
notebook = json.loads((ART / "figure/result.ipynb").read_text())
source = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
summary["table5_time_code"] = {
    "forces_seed_bug_success_to_10": 'loc[waflgo_result["init"] == "B", "success"] = 10' in source,
    "forces_seed_bug_time_to_24h": 'loc[waflgo_result["init"] == "B", "time"] = 24 * 3600'
    in source,
    "reported_aggregate_hours": {"BIC": "19:39:50", "FIX": "21:48:32"},
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

# Compact CSV for report/review.
with (OUT / "ablation_summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "scenario",
            "configuration",
            "rows",
            "mean_score",
            "bugs_any",
            "success_trials",
            "delta",
            "ci_low",
            "ci_high",
            "p_wilcoxon",
        ]
    )
    for sc, configs in summary["ablations"].items():
        for name, rec in configs.items():
            comp = rec.get("versus_default", rec.get("versus_message_only", {}))
            ci = comp.get("mean_score_difference_ci95", [math.nan] * 3)
            w.writerow(
                [
                    sc,
                    name,
                    rec["rows"],
                    rec["mean_ordinal_score"],
                    rec["bugs_any_of_ten"],
                    rec["bug_success_trials"],
                    ci[1],
                    ci[0],
                    ci[2],
                    comp.get("wilcoxon_p_unadjusted"),
                ]
            )

# Figures (descriptive; interval is issue-bootstrap around score difference).
labels = [
    "default",
    "message_only",
    "diff_only",
    "generated_command",
    "temperature_1",
    "gpt4o_mini",
    "deepseek_r1",
    "iterations_10",
    "no_feedback",
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, sc in zip(axes, ("BIC", "FIX"), strict=True):
    recs = summary["ablations"][sc]
    vals = [recs[x]["mean_ordinal_score"] if x in recs else np.nan for x in labels]
    colors = ["#334155" if x == "default" else "#0f766e" for x in labels]
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(
        range(len(labels)),
        [x.replace("_", "\n") for x in labels],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_title(sc)
    ax.set_ylim(0, 2)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("Mean of ordinal codes (descriptive only)")
fig.suptitle("Released Cleverest ablations (exact artifact replay)")
fig.tight_layout()
fig.savefig(OUT / "ablation_scores.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(2)
width = 0.25
base = [summary["default"][s]["bugs_any_of_ten"] for s in ("BIC", "FIX")]
union = [
    summary["retrospective_ceiling"][s]["union_all_released_llm_configs"] for s in ("BIC", "FIX")
]
target = [2 * z for z in base]
ax.bar(x - width, base, width, label="Default", color="#334155")
ax.bar(x, union, width, label="Post-hoc union (upper bound)", color="#0f766e")
ax.bar(x + width, target, width, label="Literal 2× target", color="#b91c1c")
ax.set_xticks(x, ["BIC / bug finding", "BFC / reproduction"])
ax.set_ylabel("Issues with ≥1 success in 10 runs")
ax.set_ylim(0, 36)
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "doubling_ceiling.png", dpi=180)
plt.close(fig)

print(
    json.dumps(
        {
            "default": summary["default"],
            "ceiling": summary["retrospective_ceiling"],
            "manual": summary["manual_transformations"],
        },
        indent=2,
    )
)
