"""Cleverest-style prompt assembly, adapted for structured JSON candidates.

Prompts follow the four-block structure of Cleverest (task, commit info, attempt
history, upgrade instruction) but the response-format specification demands the exact
Cleverest+ candidate JSON schema. This is a design decision documented in the paper:
the model's reply is data, not shell syntax.
"""
from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass

from .llm import ChatMessage


@dataclass(frozen=True)
class CommitContext:
    """The commit-related information a prompt exposes to the model."""

    short_hash: str
    message: str
    diff: str
    subject_description: str
    argv_template: tuple[str, ...]
    format_name: str


@dataclass(frozen=True)
class AttemptRecord:
    """One prior attempt in this trial's feedback history."""

    input_bytes: bytes
    argv: tuple[str, ...]
    reached_changed_lines: bool
    behavior_changed: bool
    before_stdout_tail: str
    after_stdout_tail: str
    before_stderr_tail: str
    after_stderr_tail: str
    before_returncode: int
    after_returncode: int


_SYSTEM = (
    "You generate a single regression test as a JSON object. Do not include "
    "any prose, code fences, or comments. Reply with exactly one JSON object "
    "with the keys \"input_b64\", \"argv\", and \"rationale\". \n"
    "  - \"input_b64\": standard base64 of the raw input bytes (no data URI prefix).\n"
    "  - \"argv\": an array of strings starting with the program executable basename, "
    "containing exactly one \"@@\" token where the input file path should appear.\n"
    "  - \"rationale\": a short natural-language explanation.\n"
    "Do not embed shell metacharacters. Every argv element is one argument. \n"
    "The input bytes may be malformed on purpose if the target bug requires it."
)


_SCENARIO_GOAL = {
    "BIC": (
        "This commit is suspected to INTRODUCE a bug. Construct an input that causes "
        "a sanitizer failure in the program AFTER the commit but NOT in the program "
        "BEFORE the commit."
    ),
    "FIX": (
        "This commit is suspected to FIX a bug. Construct an input that causes a "
        "sanitizer failure in the program BEFORE the commit (still buggy) but NOT in "
        "the program AFTER the commit (already fixed)."
    ),
}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} bytes truncated]"


def _tail_stream(stream: bytes, limit: int = 400) -> str:
    text = stream.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


def _format_history_block(attempts: Sequence[AttemptRecord]) -> str:
    lines = [
        "Some previously generated inputs FAILED to trigger the intended bug. I will "
        "show them below for your reference, together with the observed before/after "
        "behavior. Generate a different, more targeted input."
    ]
    for i, attempt in enumerate(attempts, start=1):
        try:
            preview = attempt.input_bytes.decode("utf-8")
        except UnicodeDecodeError:
            preview = base64.b64encode(attempt.input_bytes).decode() + " (base64)"
        preview = _truncate(preview, 400)
        lines.append(f"\nPrevious try #{i} ({'reached-change' if attempt.reached_changed_lines else 'not-reached'}"
                     f"{', changed-output' if attempt.behavior_changed else ''}):")
        lines.append(f"  argv: {list(attempt.argv)}")
        lines.append(f"  input:\n{preview}")
        lines.append(f"  before returncode={attempt.before_returncode}, "
                     f"after returncode={attempt.after_returncode}")
        lines.append(f"  before stderr tail: {attempt.before_stderr_tail}")
        lines.append(f"  after  stderr tail: {attempt.after_stderr_tail}")
    return "\n".join(lines)


def build_messages(
    *,
    scenario: str,
    commit: CommitContext,
    prior_attempts: Sequence[AttemptRecord],
) -> list[ChatMessage]:
    """Assemble the messages for one iteration.

    Follows Cleverest's block ordering but adds explicit JSON-only response contract
    and directional target-side wording.
    """
    if scenario not in {"BIC", "FIX"}:
        raise ValueError("scenario must be 'BIC' or 'FIX'")
    task_lines = [
        f"You are a software expert testing {commit.subject_description}.",
        _SCENARIO_GOAL[scenario],
        f"The program is executed with argv template {list(commit.argv_template)} "
        f"where \"@@\" is replaced with the input file path.",
        f"The input format is {commit.format_name}.",
    ]
    commit_lines = [
        f"Here is commit {commit.short_hash}:",
        commit.message.strip(),
        "",
        "Diff:",
        _truncate(commit.diff, 6000),
    ]
    blocks = ["\n".join(task_lines), "\n".join(commit_lines)]
    if prior_attempts:
        blocks.append(_format_history_block(prior_attempts))
    blocks.append(
        "You should generate a new and better answer. If prior attempts are on the "
        "right track, refine them; otherwise, produce a diverse alternative. Return "
        "only the JSON object as specified."
    )
    return [
        ChatMessage(role="system", content=_SYSTEM),
        ChatMessage(role="user", content="\n\n".join(blocks)),
    ]
