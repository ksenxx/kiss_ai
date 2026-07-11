#!/usr/bin/env python3
"""Small-budget campaign for wiring validation: gpt-4o-mini only, T=3."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cleverest_plus.campaign import CampaignConfig, run_campaign
from cleverest_plus.dispatcher import PortfolioMember
from cleverest_plus.llm import openai_client


def main() -> int:
    """Run the small validation campaign."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY must be set", file=sys.stderr)
        return 2
    portfolio = [
        PortfolioMember(
            name="gpt4o-mini",
            iteration_cap=3,
            client_factory=lambda: openai_client(
                model="gpt-4o-mini", temperature=1.0, response_format_json=True,
            ),
        ),
    ]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "campaign_smoke"
    config = CampaignConfig(
        benchmark_root=ROOT / "benchmark",
        output_dir=output,
        total_trials=3,
        timeout_s=30.0,
    )
    summary = run_campaign(config, portfolio)
    counts = summary["counts"]
    print(f"BIC {counts['bic_bug_any']}/{counts['bic_issues']}  "
          f"FIX {counts['fix_bug_any']}/{counts['fix_issues']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
