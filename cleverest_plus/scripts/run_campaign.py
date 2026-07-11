#!/usr/bin/env python3
"""Run the Cleverest+ end-to-end campaign against a benchmark directory.

Portfolio (pre-specified before observing outcomes):
  1. DeepSeek-R1 via OpenRouter (Novita), temperature=0.0, 5-iteration cap.
  2. GPT-4o via OpenAI, temperature=1.0, 5-iteration cap.
  3. GPT-4o via OpenAI, temperature=1.0, 1-iteration cap (single-shot).

Fixed budget T=10 trials, allocation (4, 3, 3) as computed by allocate_budget.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure package importability whether run from repo or installed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cleverest_plus.campaign import CampaignConfig, run_campaign
from cleverest_plus.dispatcher import PortfolioMember
from cleverest_plus.llm import openai_client, openrouter_client


def build_portfolio() -> list[PortfolioMember]:
    """Instantiate the three-member portfolio described in the paper."""
    return [
        PortfolioMember(
            name="deepseek-r1-t0",
            iteration_cap=5,
            client_factory=lambda: openrouter_client(
                model="deepseek/deepseek-r1",
                temperature=0.0,
                response_format_json=True,
                provider_pin="novita",
            ),
        ),
        PortfolioMember(
            name="gpt4o-t1-iter5",
            iteration_cap=5,
            client_factory=lambda: openai_client(
                model="gpt-4o-2024-08-06",
                temperature=1.0,
                response_format_json=True,
            ),
        ),
        PortfolioMember(
            name="gpt4o-t1-iter1",
            iteration_cap=1,
            client_factory=lambda: openai_client(
                model="gpt-4o-2024-08-06",
                temperature=1.0,
                response_format_json=True,
            ),
        ),
    ]


def main() -> int:
    """Parse CLI arguments and run the campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmark",
                        help="benchmark root (default: bundled mini-benchmark)")
    parser.add_argument("--output", type=Path, default=ROOT / "campaign_results",
                        help="directory to write trials.jsonl and campaign_summary.json")
    parser.add_argument("--trials", type=int, default=10,
                        help="total trials per issue (default: 10)")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENROUTER_API_KEY"):
        print("Both OPENAI_API_KEY and OPENROUTER_API_KEY must be set.", file=sys.stderr)
        return 2

    config = CampaignConfig(
        benchmark_root=args.benchmark,
        output_dir=args.output,
        total_trials=args.trials,
        timeout_s=args.timeout,
    )
    portfolio = build_portfolio()
    summary = run_campaign(config, portfolio)
    counts = summary["counts"]
    print(f"Wrote: {args.output / 'campaign_summary.json'}")
    print(f"BIC bug-any: {counts['bic_bug_any']}/{counts['bic_issues']}")
    print(f"FIX bug-any: {counts['fix_bug_any']}/{counts['fix_issues']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
