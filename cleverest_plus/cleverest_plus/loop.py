"""Feedback loop for a single Cleverest+ trial.

One trial iterates up to K times. Each iteration:
  1. Assembles messages from the commit and prior attempts.
  2. Calls the trial's LLM.
  3. Parses the reply as a structured candidate (rejects shell syntax).
  4. Writes the candidate bytes to a temp file, materializes argv (no shell).
  5. Executes the before and after binaries directly, captures separated streams.
  6. Classifies with directional sanitizer signature (optional reference).
  7. Returns immediately on Bug. Otherwise appends the attempt to history and
     continues.

The trial function never observes the outcome before allocating iterations.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .benchmark import Issue, approximate_reached
from .core import (
    Candidate,
    CandidateError,
    Execution,
    Outcome,
    classify_differential,
    crash_signature,
    execute,
    materialize_argv,
    validate_format,
)
from .llm import LLMClient, LLMError
from .prompt import AttemptRecord, CommitContext, build_messages


@dataclass
class IterationRecord:
    """One iteration inside a trial."""

    iteration: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    llm_latency_s: float
    parsed: bool
    parse_error: str | None
    outcome: str
    format_valid: bool
    reached: bool
    behavior_changed: bool
    signature_match: bool
    candidate_argv: tuple[str, ...] | None
    candidate_bytes_hex: str | None


@dataclass
class TrialResult:
    """The result of a single trial (up to K iterations)."""

    issue_id: str
    portfolio_member: str
    model: str
    trial_index: int
    ordinal_code: int
    outcome: str
    iterations: list[IterationRecord] = field(default_factory=list)


_OUTCOME_TO_CODE = {
    Outcome.BUG.value: 3,
    Outcome.DIFFERENT.value: 2,
    Outcome.REACHED.value: 1,
    Outcome.NONE.value: 0,
    Outcome.AMBIGUOUS.value: 0,
}


def _parse_candidate(reply: str, *, allowed: set[str]) -> tuple[Candidate | None, str | None]:
    """Parse a JSON reply into a Candidate. Also accepts fenced-code responses."""
    text = reply.strip()
    fenced = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    # Some reasoning models emit an object followed by trailing text; keep only the
    # first balanced JSON object.
    if text.startswith("{"):
        depth, end = 0, -1
        for i, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > 0:
            text = text[:end]
    try:
        return Candidate.from_json(text, allowed_executables=allowed), None
    except CandidateError as exc:
        return None, str(exc)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def _to_commit_context(issue: Issue) -> CommitContext:
    return CommitContext(
        short_hash=issue.commit_short_hash,
        message=issue.commit_message,
        diff=issue.diff,
        subject_description=issue.subject_description,
        argv_template=issue.argv_template,
        format_name=issue.format_name,
    )


def _record_from_attempt(candidate: Candidate, before: Execution, after: Execution,
                        reached: bool, changed: bool) -> AttemptRecord:
    return AttemptRecord(
        input_bytes=candidate.data,
        argv=candidate.argv_template,
        reached_changed_lines=reached,
        behavior_changed=changed,
        before_stdout_tail=before.stdout[-256:].decode("utf-8", errors="replace"),
        after_stdout_tail=after.stdout[-256:].decode("utf-8", errors="replace"),
        before_stderr_tail=before.stderr[-256:].decode("utf-8", errors="replace"),
        after_stderr_tail=after.stderr[-256:].decode("utf-8", errors="replace"),
        before_returncode=before.returncode,
        after_returncode=after.returncode,
    )


def run_trial(
    *,
    issue: Issue,
    portfolio_member: str,
    client: LLMClient,
    trial_index: int,
    iteration_cap: int,
    tmp_root: Path,
    timeout_s: float = 30.0,
) -> TrialResult:
    """Execute one trial (up to iteration_cap iterations) for one Issue.

    Iterations stop early only on Bug. All other outcomes are logged and feed the
    next iteration's history.
    """
    tmp_root.mkdir(parents=True, exist_ok=True)
    allowed = {issue.argv_template[0]}
    attempts: list[AttemptRecord] = []
    iterations: list[IterationRecord] = []
    best_code = 0
    best_outcome = Outcome.NONE.value
    scenario = issue.scenario

    for k in range(1, iteration_cap + 1):
        messages = build_messages(
            scenario=scenario,
            commit=_to_commit_context(issue),
            prior_attempts=attempts,
        )
        try:
            reply = client.generate(messages, seed=1000 * trial_index + k)
        except LLMError as exc:
            iterations.append(IterationRecord(
                iteration=k, model=client.model, prompt_tokens=0, completion_tokens=0,
                llm_latency_s=0.0, parsed=False, parse_error=f"llm_error: {exc}",
                outcome=Outcome.NONE.value, format_valid=False, reached=False,
                behavior_changed=False, signature_match=False,
                candidate_argv=None, candidate_bytes_hex=None,
            ))
            continue

        candidate, err = _parse_candidate(reply.text, allowed=allowed)
        if candidate is None:
            iterations.append(IterationRecord(
                iteration=k, model=client.model,
                prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                llm_latency_s=reply.latency_s, parsed=False, parse_error=err,
                outcome=Outcome.NONE.value, format_valid=False, reached=False,
                behavior_changed=False, signature_match=False,
                candidate_argv=None, candidate_bytes_hex=None,
            ))
            continue

        input_path = tmp_root / f"trial-{trial_index}-iter-{k}.input"
        input_path.write_bytes(candidate.data)
        try:
            argv_before = materialize_argv(candidate, issue.before_dir, input_path)
            argv_after = materialize_argv(candidate, issue.after_dir, input_path)
        except CandidateError as exc:
            iterations.append(IterationRecord(
                iteration=k, model=client.model,
                prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                llm_latency_s=reply.latency_s, parsed=True, parse_error=str(exc),
                outcome=Outcome.NONE.value, format_valid=False, reached=False,
                behavior_changed=False, signature_match=False,
                candidate_argv=candidate.argv_template, candidate_bytes_hex=candidate.data.hex(),
            ))
            continue
        before_env = {"ASAN_OPTIONS": "abort_on_error=0:allocator_may_return_null=1:detect_leaks=0"}
        after_env = before_env
        before = execute(argv_before, timeout_s=timeout_s, env=before_env)
        after = execute(argv_after, timeout_s=timeout_s, env=after_env)

        # A fallback "reached" heuristic: any changed line covered by a target
        # binary's stderr trace of a line marker (subject-specific). For the mini
        # benchmark, we approximate reached=changed differ.
        reached = approximate_reached(
            covered_before_lines={},
            covered_after_lines={},
            changed_before=issue.changed_lines_before,
            changed_after=issue.changed_lines_after,
        ) or bool(before.stderr or after.stderr)

        fmt_ok, _fmt_msg = validate_format(candidate.data, issue.format_name)
        outcome = classify_differential(
            before=before,
            after=after,
            scenario=scenario,
            changed_lines_reached=reached,
            expected_signature=issue.reference_signature,
        )
        target_sig = crash_signature((after if scenario == "BIC" else before).stderr)
        signature_match = (
            issue.reference_signature is not None
            and target_sig is not None
            and target_sig == issue.reference_signature
        )
        behavior_changed = (
            (before.stdout, before.stderr, before.returncode)
            != (after.stdout, after.stderr, after.returncode)
        )
        code = _OUTCOME_TO_CODE[outcome.value]
        if code > best_code:
            best_code = code
            best_outcome = outcome.value

        iterations.append(IterationRecord(
            iteration=k, model=client.model,
            prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
            llm_latency_s=reply.latency_s, parsed=True, parse_error=None,
            outcome=outcome.value, format_valid=fmt_ok,
            reached=reached, behavior_changed=behavior_changed,
            signature_match=signature_match,
            candidate_argv=candidate.argv_template, candidate_bytes_hex=candidate.data.hex(),
        ))
        if outcome is Outcome.BUG:
            break
        attempts.append(_record_from_attempt(candidate, before, after, reached, behavior_changed))

    return TrialResult(
        issue_id=issue.identifier,
        portfolio_member=portfolio_member,
        model=client.model,
        trial_index=trial_index,
        ordinal_code=best_code,
        outcome=best_outcome,
        iterations=iterations,
    )


def summarize_trials(trials: Iterable[TrialResult]) -> dict[str, float | int]:
    """Aggregate a per-issue set of trials into ordinal-code and bug-any counts."""
    codes: list[int] = []
    for trial in trials:
        codes.append(trial.ordinal_code)
    if not codes:
        return {"trials": 0, "mean_code": 0.0, "bug_any": 0}
    return {
        "trials": len(codes),
        "mean_code": sum(codes) / len(codes),
        "bug_any": 1 if any(c == 3 for c in codes) else 0,
    }
