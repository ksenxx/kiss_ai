"""Portfolio dispatcher that allocates an exact trial budget before observation.

For total budget T and m portfolio members, member j receives
    n_j = floor(T/m) + 1[j <= T mod m],   sum_j n_j == T.
The allocation is computed once per issue at start-up and never revised after
observing outcomes. Members are dispatched in a pre-specified deterministic order.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .benchmark import Issue
from .llm import LLMClient
from .loop import TrialResult, run_trial


@dataclass(frozen=True)
class PortfolioMember:
    """One entry in the fixed-budget portfolio."""

    name: str
    iteration_cap: int
    client_factory: Callable[[], LLMClient]


def allocate_budget(total_trials: int, member_count: int) -> tuple[int, ...]:
    """Return an ordered allocation summing to total_trials.

    Deterministic and independent of outcomes.
    """
    if total_trials < 0 or member_count <= 0:
        raise ValueError("total_trials must be non-negative and member_count positive")
    base, remainder = divmod(total_trials, member_count)
    return tuple(base + (1 if j < remainder else 0) for j in range(member_count))


def run_issue(
    *,
    issue: Issue,
    portfolio: Sequence[PortfolioMember],
    total_trials: int,
    tmp_root: Path,
    timeout_s: float = 30.0,
    on_trial: Callable[[TrialResult], None] | None = None,
) -> list[TrialResult]:
    """Run all trials for one issue with an exact-budget allocation.

    All allocations are committed before any trial is run. A trial does not observe
    the allocation of other members; the dispatcher simply loops over members with
    their fixed n_j and returns the sequence of TrialResults.
    """
    allocation = allocate_budget(total_trials, len(portfolio))
    results: list[TrialResult] = []
    trial_counter = 0
    if len(portfolio) != len(allocation):
        raise ValueError("allocation size mismatch")
    for member, n in zip(portfolio, allocation):
        if n == 0:
            continue
        client = member.client_factory()
        for _ in range(n):
            trial_counter += 1
            result = run_trial(
                issue=issue,
                portfolio_member=member.name,
                client=client,
                trial_index=trial_counter,
                iteration_cap=member.iteration_cap,
                tmp_root=tmp_root / f"trial-{trial_counter}",
                timeout_s=timeout_s,
            )
            results.append(result)
            if on_trial is not None:
                on_trial(result)
    return results
