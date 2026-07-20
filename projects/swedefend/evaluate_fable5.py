# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Run the SWExploit → SWEDefend evaluation with ``claude-fable-5`` as the judge.

This is the final evaluation the task asks for: it wires the full 4-layer
pipeline (with the intent-alignment judge backed by ``claude-fable-5``, which
transparently falls back to ``claude-opus-4-8`` on a non-retryable provider
error) and replays the SWExploit adversarial payload suite through it, printing
the catch-rate report.

Usage
-----

.. code-block:: bash

    uv run python -m swedefend.evaluate_fable5 --model claude-fable-5
"""

from __future__ import annotations

import argparse

from swedefend.defense.intent_judge import IntentAlignmentJudge
from swedefend.pipeline import SWEDefendPipeline
from swedefend.swexploit import SWExploitHarness, SWExploitReport


def build_pipeline(model_name: str) -> SWEDefendPipeline:
    """Build the full 4-layer pipeline with an LLM judge.

    Args:
        model_name: The judge model (e.g. ``"claude-fable-5"``).  Its registered
            fallback (``claude-opus-4-8``) is used automatically on failure.

    Returns:
        A configured :class:`SWEDefendPipeline`.
    """
    return SWEDefendPipeline(judge=IntentAlignmentJudge(model_name))


def run_evaluation(model_name: str) -> SWExploitReport:
    """Replay the SWExploit suite through the fable-5-judged pipeline.

    Args:
        model_name: The judge model name.

    Returns:
        The :class:`SWExploitReport` summarizing catch-rate and false positives.
    """
    return SWExploitHarness(build_pipeline(model_name)).run()


def main() -> int:
    """CLI entry point for the fable-5 SWEDefend evaluation.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="SWExploit → SWEDefend evaluation")
    parser.add_argument("--model", default="claude-fable-5", help="Judge model name")
    args = parser.parse_args()

    report = run_evaluation(args.model)
    print(f"Judge model : {args.model}")
    print(f"Malicious   : {report.n_malicious}")
    print(f"Caught      : {report.caught}")
    print(f"Slipped     : {report.slipped}")
    print(f"False pos.  : {report.false_positives}")
    print(f"Catch rate  : {report.catch_rate:.0%}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
