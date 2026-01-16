# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# Cursor AI (cursor@cursor.com)
# add your name here

"""HotPotQA benchmark for GEPA prompt optimization."""

from kiss.agents.gepa.hotpotqa.hotpotqa_benchmark import (
    HotPotQABenchmark,
    evaluate_hotpotqa_result,
)

__all__ = ["HotPotQABenchmark", "evaluate_hotpotqa_result"]
