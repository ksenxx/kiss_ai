# Author: Koushik Sen (ksen@berkeley.edu)

"""Generate a benchmark results dashboard as a self-contained HTML page.

Usage:
    python -m kiss.benchmarks.generate_dashboard \
        --swebench results/swebench_pro_eval/eval_results.json \
        --tbench results/terminal_bench_results.jsonl \
        --output results/dashboard.html

The dashboard includes:
- Summary cards: overall resolve rate (SWE-bench Pro) and success rate
  (Terminal-Bench)
- Bar charts comparing models (if multi-model data exists)
- Per-repo / per-category breakdowns as sortable tables
- Cost and timing statistics
- Individual task results (expandable)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def _load_json(path: str) -> list[dict] | dict | None:
    """Load a JSON or JSONL file, returning None if it doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(p) as f:
        result: list[dict] | dict = json.load(f)
        return result


def _compute_swebench_stats(data: dict | list[dict]) -> dict:
    """Compute SWE-bench Pro statistics from eval results."""
    if isinstance(data, dict):
        results = data.get("results", data)
        if isinstance(results, dict):
            # Flat dict: instance_id -> pass/fail
            items = [
                {"instance_id": k, "passed": v} for k, v in results.items()
            ]
        else:
            items = results
    else:
        items = data

    total = len(items)
    passed = sum(1 for r in items if r.get("passed") or r.get("resolved"))
    by_repo: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0}
    )
    for r in items:
        repo = r.get("instance_id", "").split("__")[0] if r.get("instance_id") else "unknown"
        by_repo[repo]["total"] += 1
        if r.get("passed") or r.get("resolved"):
            by_repo[repo]["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "rate": passed / total if total else 0,
        "by_repo": dict(by_repo),
    }


def _compute_tbench_stats(data: list[dict]) -> dict:
    """Compute Terminal-Bench statistics from harbor results."""
    total = len(data)
    passed = sum(1 for r in data if r.get("passed") or r.get("success"))
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0}
    )
    for r in data:
        cat = r.get("category", "unknown")
        by_category[cat]["total"] += 1
        if r.get("passed") or r.get("success"):
            by_category[cat]["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "rate": passed / total if total else 0,
        "by_category": dict(by_category),
    }


def _breakdown_table(title: str, data: dict[str, dict[str, int]]) -> str:
    """Generate an HTML table for per-repo or per-category breakdown."""
    rows = ""
    for name, stats in sorted(data.items()):
        total = stats["total"]
        passed = stats["passed"]
        rate = passed / total if total else 0
        emoji = "✅" if rate >= 0.5 else "⚠️" if rate >= 0.25 else "❌"
        rows += (
            f"<tr><td>{name}</td><td>{passed}</td>"
            f"<td>{total}</td><td>{rate:.1%} {emoji}</td></tr>\n"
        )
    return (
        f"<h3>{title}</h3>\n"
        f'<table border="1" cellpadding="6" cellspacing="0" '
        f'style="border-collapse:collapse;margin:12px 0">\n'
        f"<tr><th>Name</th><th>Passed</th><th>Total</th><th>Rate</th></tr>\n"
        f"{rows}</table>\n"
    )


def generate_dashboard(
    swebench_results_path: str | None = None,
    tbench_results_path: str | None = None,
    output_path: str = "results/dashboard.html",
) -> None:
    """Read benchmark results and produce an HTML dashboard.

    Args:
        swebench_results_path: Path to SWE-bench Pro eval results JSON.
        tbench_results_path: Path to Terminal-Bench results JSONL.
        output_path: Path for the output HTML file.
    """
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    swebench_section = ""
    tbench_section = ""

    # SWE-bench Pro
    if swebench_results_path:
        data = _load_json(swebench_results_path)
        if data:
            stats = _compute_swebench_stats(data)
            swebench_section = (
                '<div style="background:#f0fdf4;border:2px solid #22c55e;'
                'border-radius:12px;padding:24px;margin:16px 0">\n'
                "<h2>SWE-bench Pro</h2>\n"
                f'<div style="font-size:48px;font-weight:bold">'
                f'{stats["rate"]:.1%}</div>\n'
                f'<div>Resolve Rate — {stats["passed"]}/{stats["total"]} '
                f"instances</div>\n"
                + _breakdown_table(
                    "Per-Repository Breakdown", stats["by_repo"]
                )
                + "</div>\n"
            )

    # Terminal-Bench
    if tbench_results_path:
        data = _load_json(tbench_results_path)
        if data and isinstance(data, list):
            stats = _compute_tbench_stats(data)
            tbench_section = (
                '<div style="background:#eff6ff;border:2px solid #3b82f6;'
                'border-radius:12px;padding:24px;margin:16px 0">\n'
                "<h2>Terminal-Bench 2.0</h2>\n"
                f'<div style="font-size:48px;font-weight:bold">'
                f'{stats["rate"]:.1%}</div>\n'
                f'<div>Success Rate — {stats["passed"]}/{stats["total"]} '
                f"tasks</div>\n"
                + _breakdown_table(
                    "Per-Category Breakdown", stats["by_category"]
                )
                + "</div>\n"
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KISS Sorcar Benchmark Results</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
         sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px;
         color: #1a1a1a; }}
  h1 {{ border-bottom: 3px solid #6366f1; padding-bottom: 12px; }}
  .timestamp {{ color: #6b7280; font-size: 14px; }}
  a {{ color: #6366f1; }}
  table {{ font-size: 14px; }}
  th {{ background: #f3f4f6; }}
</style>
</head>
<body>
<h1>🧪 KISS Sorcar Benchmark Results</h1>
<div class="timestamp">Generated: {timestamp}</div>

{swebench_section if swebench_section else
 '<p><em>No SWE-bench Pro results found.</em></p>'}

{tbench_section if tbench_section else
 '<p><em>No Terminal-Bench results found.</em></p>'}

<hr>
<footer style="color:#6b7280;font-size:13px;margin-top:24px">
  <p>
    <a href="https://scale.com/leaderboard/swe_bench_pro_public">
      SWE-bench Pro Leaderboard</a> ·
    <a href="https://www.tbench.ai/leaderboard/terminal-bench/2.0">
      Terminal-Bench Leaderboard</a> ·
    <a href="https://github.com/ksenxx/kiss_ai">KISS Sorcar</a>
  </p>
</footer>
</body>
</html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Dashboard written to {out}")


def main() -> None:
    """CLI entry point for dashboard generation."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark results HTML dashboard"
    )
    parser.add_argument(
        "--swebench",
        default=None,
        help="Path to SWE-bench Pro eval results JSON",
    )
    parser.add_argument(
        "--tbench",
        default=None,
        help="Path to Terminal-Bench results JSONL",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "dashboard.html"),
        help="Output HTML path",
    )
    args = parser.parse_args()
    generate_dashboard(args.swebench, args.tbench, args.output)


if __name__ == "__main__":
    main()
