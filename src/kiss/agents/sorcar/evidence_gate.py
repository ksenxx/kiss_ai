# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Evidence gate: a ``finish(success=True)`` must be backed by a check, not a claim.

An unattended agent claims success far more often than a check agrees:
its own summary says "educated guess, no answer key" and then calls
``finish(success=True)``, and a refused finish is usually re-issued at
once with nothing checked in between.  The gate refuses such a finish
once, with the concrete reason, so the model spends one more round on
evidence instead of on a claim.  A second ``finish`` always passes: the
agent is never trapped.

Two reasons are mechanical enough to check from the tool calls alone:

* the summary hedges (:data:`HEDGE_PHRASES`): it says the result was
  guessed, untested or unverified while ``success=True`` says it was;
* a file was changed (``Edit``, ``Write`` or a shell command that writes,
  :data:`WRITING_COMMAND`) and no shell command ran after the last change,
  so whatever was verified was verified before the final edit.

The interactive Sorcar agent (:meth:`SorcarAgent._guard_tool_call`) applies
the first reason to every run with the edit tools; the containerised coding
SEA applies both (``track_changes=True``), since an unattended trial has no
one else to notice an unchecked final edit.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Phrases that deny verification; a summary containing one contradicts ``success=True``.
HEDGE_PHRASES: tuple[str, ...] = (
    "educated guess", "my guess", "best guess", "guessed", "guesswork",
    "not verified", "unverified", "could not verify", "couldn't verify", "did not verify",
    "didn't verify", "unable to verify", "cannot verify", "can't verify",
    "untested", "not tested", "did not test", "didn't test", "could not test", "couldn't test",
    "unable to test", "cannot confirm", "can't confirm", "could not confirm", "not confirmed",
    "unable to confirm", "should work", "probably works", "likely works", "no answer key",
    "no reference output", "no reference solution", "no ground truth",
    # A reading of the task left open at the end: a summary that conditions the
    # result on an unseen checker names the requirement it did not settle.
    "if the grader", "if the hidden test", "if the checker", "judgement call",
    "judgment call", "could cost points", "may cost points", "might cost points",
    "if instead the task", "if the task instead", "if they instead", "open question",
    "left open", "remains ambiguous", "still ambiguous",
)

#: Shell commands that change files: redirections (not to ``/dev/null``), here
#: documents, copy/move/patch and in-place editors, package and build tools.
WRITING_COMMAND = re.compile(
    r"(?<![<>&\d])>{1,2}(?!&)\s*(?!/dev/null)\S"
    r"|<<"
    r"|\b(?:tee|cp|mv|rsync|install|patch|touch|mkdir|dd|truncate|chmod|chown)\b"
    r"|\bsed\s+(?:-[a-zA-Z]*i|--in-place)"
    r"|\bgit\s+(?:apply|checkout|stash|reset|revert|cherry-pick|merge|rebase|pull)\b"
    r"|\b(?:pip|pip3|uv|npm|yarn|pnpm|cargo|apt|apt-get|make|cmake|ninja)\b"
)

#: Tool names that change files outright.
EDIT_TOOLS = frozenset({"Edit", "Write"})
#: Tool names that run shell commands.
SHELL_TOOLS = frozenset({"Bash", "run_commands_parallel"})

#: The refusal; ``{reasons}`` is the numbered list of what is missing.
EVIDENCE_VERDICT = (
    "Not finished: finish(success=True) claims every requirement is met and verified, "
    "but:\n{reasons}\n"
    "Fix this, then call finish again: verify each requirement with a command run now, "
    "from a fresh shell, and put in summary_in_html a checklist of the requirements with "
    "the command that checked each and its result. If something cannot be verified, "
    "finish with success=False and say exactly what is unverified."
)
HEDGE_REASON = (
    "the summary itself says the result is unverified ({phrases}); a guess is not a "
    "verified result"
)
UNCHECKED_REASON = (
    "files were changed after the last shell command, so nothing that was verified "
    "covers the final state of the work"
)


def hedges_in(text: str) -> list[str]:
    """The :data:`HEDGE_PHRASES` found in *text* (case-insensitive), in catalogue order.

    Args:
        text: The summary to search.
    """
    lowered = text.lower()
    return [phrase for phrase in HEDGE_PHRASES if phrase in lowered]


def writes_files(args: dict[str, Any]) -> bool:
    """Whether a ``Bash`` / ``run_commands_parallel`` call looks like it changes files.

    Args:
        args: The tool call's arguments (``command`` or a JSON list ``commands``).
    """
    text = args.get("command") if isinstance(args.get("command"), str) else None
    if text is None:
        commands = args.get("commands")
        if isinstance(commands, str):
            try:
                parsed = json.loads(commands)
            except ValueError:
                parsed = [commands]
            text = "\n".join(c for c in parsed if isinstance(c, str)) if isinstance(
                parsed, list) else commands
        else:
            text = ""
    return bool(WRITING_COMMAND.search(text))


class EvidenceGate:
    """Tracks whether a check followed the last change; refuses one unsupported finish.

    Call :meth:`observe` for every tool call that runs and :meth:`verdict` for
    every ``finish``; the first ``finish(success=True)`` whose summary hedges or
    that follows an unchecked change is refused with the reason, later ones pass.
    """

    def __init__(self, track_changes: bool = False) -> None:
        """Create a gate.

        Args:
            track_changes: Also refuse a finish that follows a file change no
                shell command checked (:data:`UNCHECKED_REASON`).
        """
        self.track_changes = track_changes
        self.used = False
        #: A file changed and no shell command ran since.
        self.unchecked_change = False

    def observe(self, name: str, args: dict[str, Any]) -> None:
        """Record a tool call that is about to run.

        Args:
            name: The tool name.
            args: Its arguments.
        """
        if name in EDIT_TOOLS:
            self.unchecked_change = True
        elif name in SHELL_TOOLS:
            self.unchecked_change = writes_files(args)

    def reasons(self, args: dict[str, Any]) -> list[str]:
        """The reasons a ``finish`` with *args* is unsupported (empty when it is not).

        Args:
            args: The ``finish`` call's arguments.
        """
        if str(args.get("success")).lower() != "true":
            return []
        found = hedges_in(str(args.get("summary_in_html", "")))
        reasons = []
        if found:
            reasons.append(HEDGE_REASON.format(phrases=", ".join(f'"{p}"' for p in found)))
        if self.track_changes and self.unchecked_change:
            reasons.append(UNCHECKED_REASON)
        return reasons

    def verdict(self, args: dict[str, Any]) -> str | None:
        """The refusal text for a ``finish`` call, or ``None`` to let it through.

        Args:
            args: The ``finish`` call's arguments.
        """
        if self.used:
            return None
        reasons = self.reasons(args)
        if not reasons:
            return None
        self.used = True
        numbered = "\n".join(f"{i}. {reason}" for i, reason in enumerate(reasons, 1))
        return EVIDENCE_VERDICT.format(reasons=numbered)
