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
  :func:`writes_files`) and no shell command ran after the last change,
  so whatever was verified was verified before the final edit.

The interactive Sorcar agent (:meth:`SorcarAgent._guard_tool_call`) applies
the first reason to every run with the edit tools; the containerised coding
SEA applies both (``track_changes=True``), since an unattended trial has no
one else to notice an unchecked final edit.
"""

from __future__ import annotations

import json
import posixpath
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

#: A word after ``make``/``cmake``/``ninja`` that builds nothing: a check
#: target, an option or a variable assignment.
_MAKE_WORD = r"(?:(?:test|check)\b|-\S+|\w+=\S*)"
#: The rest of a Make (CMake, Ninja) command when it builds nothing: only
#: such words up to the end of the command, one a check target (``make test``,
#: ``make -k check V=1``).
_ONLY_CHECKS = (
    rf"(?=(?:\s+{_MAKE_WORD})*?\s+(?:test|check)\b)"
    rf"(?:\s+{_MAKE_WORD})+\s*(?:[;&|)#\n]|\d*[<>]|\Z)"
)

#: Shell commands that change files: redirections (not to ``/dev/null``), here
#: documents, copy/move/patch and in-place editors, ``--fix`` runs and builds
#: (Make, CMake, Ninja), unless every Make target is a check (``make test``).
#: Package tools are read word by word instead (:data:`WRITING_SUBCOMMANDS`).
WRITING_COMMAND = re.compile(
    r"(?<![<>&\d])>{1,2}(?!&)\s*(?!/dev/null)\S"
    r"|<<"
    r"|--fix\b"
    r"|\b(?:tee|cp|mv|rsync|install|patch|touch|mkdir|dd|truncate|chmod|chown)\b"
    r"|\bsed\s+(?:-[a-zA-Z]*i|--in-place)"
    rf"|\b(?:make|cmake|ninja)\b(?!{_ONLY_CHECKS})"
)

_PIP = re.compile(r"(?:install|uninstall)\b")
_NODE = re.compile(r"(?:install|i|ci|add|remove|uninstall|update|build)\b")
_APT = re.compile(r"(?:install|remove|purge|upgrade)\b")
#: Package tools, by program name, and the start of their subcommands that
#: change files (``git checkout-index`` too); their other subcommands (``uv
#: run pytest``, ``npm test``, ``git diff``) are checks.  ``npm run build``
#: (Yarn's, pnpm's) builds too.
WRITING_SUBCOMMANDS: dict[str, re.Pattern[str]] = {
    "git": re.compile(r"(?:apply|checkout|stash|reset|revert|cherry-pick|merge|rebase|pull)\b"),
    "pip": _PIP,
    "pip3": _PIP,
    "uv": re.compile(r"(?:add|remove|sync|lock|build)\b"),
    "npm": _NODE,
    "yarn": _NODE,
    "pnpm": _NODE,
    "cargo": re.compile(r"(?:build|install|add|remove|update|fmt|fix)\b"),
    "apt": _APT,
    "apt-get": _APT,
}
_RUN_BUILD = frozenset({"npm", "yarn", "pnpm"})
_BUILD = re.compile(r"build\b")
#: A shell word (unquoted characters, escapes and whole quoted strings, so a
#: quoted value with blanks is one word) or a character that ends a command.
_SHELL_TOKEN = re.compile(
    r"""(?:[^\s"'\\;&|()`]++|\\.|"(?:[^"\\]|\\.)*+"?|'[^']*+'?)++|[;&|()`\n]""", re.DOTALL)
_COMMAND_ENDS = frozenset(";&|()`\n")
#: What one level of quote removal changes: an escape, a ``"..."`` or a ``'...'`` string.
_QUOTING = re.compile(r"""\\(.)|"((?:[^"\\]|\\.)*+)"?|'([^']*+)'?""", re.DOTALL)
#: The escapes a ``"..."`` string keeps special.
_DOUBLE_QUOTED_ESCAPE = re.compile(r'\\([$`"\\])')
_BLANK = re.compile(r"\s")

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


def _unquote_match(match: re.Match[str]) -> str:
    """Return what one escape or quoted string of :data:`_QUOTING` stands for."""
    escaped, double, single = match.groups()
    if escaped is not None:
        return escaped
    if double is not None:
        return _DOUBLE_QUOTED_ESCAPE.sub(r"\1", double)
    return single


def _runs_writing_subcommand(text: str) -> bool:
    """Return whether the shell *text* runs a subcommand of :data:`WRITING_SUBCOMMANDS`.

    Every word naming a package tool starts a scan of the words after it,
    to the end of its command: an option (``-C``, ``--prefix``, Cargo's
    ``+stable``) may take the next word as its value, and the first word
    that is neither is the subcommand.  Words are split as the shell splits
    them, so a quoted or escaped value with blanks is one word, and a word
    with blanks after quote removal (``bash -c "npm ci"``) is read as
    commands too.  The scans of one tool in one state go on alike, so one
    pass keeps only the tools whose next plain word is the subcommand
    (*expect*) and those whose next plain word may be an option's value
    (*optioned*): the time is linear in the words, however options repeat.
    """
    expect: set[str] = set()
    optioned: set[str] = set()
    after_run = False  # the last word was a Node tool's ``run``
    for token in _SHELL_TOKEN.findall(text.replace("\\\n", " ")):
        if token in _COMMAND_ENDS:
            expect, optioned, after_run = set(), set(), False
            continue
        word = _QUOTING.sub(_unquote_match, token)
        # Blanks are left only by quotes or escapes, which quote removal
        # drops: *word* is shorter than *token*, so the recursion ends.
        if _BLANK.search(word) and _runs_writing_subcommand(word):
            return True
        if word.startswith(("-", "+")):
            expect, optioned, after_run = set(), expect | optioned, False
        else:
            tools = expect | optioned
            if (after_run and _BUILD.match(word)) or any(
                    WRITING_SUBCOMMANDS[tool].match(word) for tool in tools):
                return True
            after_run = word == "run" and not tools.isdisjoint(_RUN_BUILD)
            expect, optioned = optioned, set()
        name = posixpath.basename(word)
        if name in WRITING_SUBCOMMANDS:
            expect.add(name)
    return False


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
    return bool(WRITING_COMMAND.search(text)) or _runs_writing_subcommand(text)


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
