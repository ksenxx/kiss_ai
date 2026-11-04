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
* a file was changed (``Edit``, ``Write`` or a shell command that changes
  files, :func:`file_changes`) and no shell command ran after the last
  change, so whatever was verified was verified before the final edit.

What a shell command changes is read from its simple commands
(:func:`~kiss.agents.sorcar.shell_scan.simple_commands`), the way the
shell reads them: a program such as ``rm`` or ``cp`` counts in command
position only (``cat install.sh`` and ``git commit -m "rm stale"`` run
nothing of the kind), a here-document is input to the program that
reads it, and a file under a ``tmp`` directory is the run's scratch
(a saved probe output, a copied log), not a change of the work.

The interactive Sorcar agent (:meth:`SorcarAgent._guard_tool_call`) applies
the first reason to every run with the edit tools; the containerised coding
SEA applies both (``track_changes=True``), since an unattended trial has no
one else to notice an unchecked final edit.
"""

from __future__ import annotations

import itertools
import json
import posixpath
import re
from typing import Any

from kiss.agents.sorcar.shell_scan import UV_VALUED_OPTIONS, SimpleCommand, Word, simple_commands

#: Phrases that deny verification; a summary containing one contradicts ``success=True``.
HEDGE_PHRASES: tuple[str, ...] = (
    "educated guess", "my guess", "best guess", "guessed", "guesswork",
    "unverified", "could not verify", "couldn't verify", "did not verify",
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

#: What a summary cites rather than says, left out of the hedge search: code
#: (``<code>``, ``<pre>``, ``<kbd>``, ``<samp>`` elements and backtick spans)
#: and quoted text (straight, HTML-escaped or curly quotes; a single-quoted
#: string opens after a blank or a bracket, so an apostrophe does not).  A
#: UI label ("Not Verified"), a command, or the gate's own wording quoted
#: back is not the agent's claim about its work.
_CITED = re.compile(
    r"<(code|pre|kbd|samp)\b[^>]*>.*?</\1\s*>"
    r"|`[^`\n]*`"
    r'|"[^"\n]*"|&quot;.*?&quot;|“[^”\n]*”|‘[^’\n]*’'
    r"|(?<![\w'])'[^'\n]*'(?!\w)",
    re.IGNORECASE | re.DOTALL,
)

#: Programs that change the files their operands name, with the operands
#: they change: ``last`` (the destination of a copy), ``all`` (every path
#: operand) or ``rest`` (every operand but the first, a mode or an owner).
#: ``dd`` changes the files its ``of=`` operands name.  A call naming only
#: scratch files (:func:`is_scratch_path`) changes nothing of the work; one
#: naming no file (``patch < fix.diff``) is read as a change.
WRITING_PROGRAMS: dict[str, str] = {
    "cp": "last", "rsync": "last", "install": "last",
    "tee": "all", "mv": "all", "rm": "all", "rmdir": "all", "unlink": "all", "touch": "all",
    "mkdir": "all", "shred": "all", "truncate": "all", "patch": "all", "dd": "all",
    "chmod": "rest", "chown": "rest", "chgrp": "rest",
}
#: The copies whose ``-t DIR`` / ``--target-directory DIR`` names the destination,
#: and that option: the value attached (``-tDIR``, ``--target-directory=DIR``)
#: or in the next word, ``t`` possibly last in a cluster of short options.
_TARGET_PROGRAMS = frozenset({"cp", "mv", "install"})
_TARGET_OPTION = re.compile(r"(?:-[a-su-zA-Z]*t|--target-directory=?)(?P<value>.*)")
#: A python interpreter in command position, whose ``-m MODULE`` runs a tool.
_PYTHON = re.compile(r"python[0-9.]*$")
#: ``sed``'s in-place option, among its options in any position.  An
#: in-place edit of scratch files only (``sed -i s/a/b/ tmp/x/f``) changes
#: nothing of the work, like a copy into ``tmp/``.
_SED_INPLACE = re.compile(r"-[a-zA-Z]*i|--in-place")
#: The build tools whose bare call builds; a call naming only check targets
#: (``make test``, ``make -k check V=1``) builds nothing.
BUILD_TOOLS = frozenset({"make", "cmake", "ninja"})
_CHECK_TARGET = re.compile(r"(?:test|check)\b")
#: The option that fixes files in place (``ruff check --fix``, ``check --full --fix``).
_FIX_OPTION = re.compile(r"--fix\b")
#: The shells whose ``-c`` (``-lc``, ``-ec``) argument is shell text.
SHELLS = frozenset({"bash", "sh", "dash", "zsh", "ksh"})
_COMMAND_OPTION = re.compile(r"-\w*c")
#: ``find``'s actions that run a command on each file found.
_FIND_EXEC = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
#: A path under a ``tmp`` directory: the scratch directory of the run (``tmp/x``,
#: ``./tmp/x``, ``/home/u/repo/tmp/x``) or the system's (``/tmp/x``).
_SCRATCH = re.compile(r"(?:^|/)tmp(?:/|$)")

_PIP = re.compile(r"(?:install|uninstall)\b")
_NODE = re.compile(r"(?:install|i|ci|add|remove|uninstall|update|build)\b")
_APT = re.compile(r"(?:install|remove|purge|upgrade)\b")
#: Package tools, by program name, and the start of their subcommands that
#: change files (``git checkout-index`` too); their other subcommands (``uv
#: run pytest``, ``npm test``, ``git diff``) are checks.  ``npm run build``
#: (Yarn's, pnpm's) builds too.
WRITING_SUBCOMMANDS: dict[str, re.Pattern[str]] = {
    "git": re.compile(r"(?:apply|checkout|stash|reset|revert|cherry-pick|merge|rebase|pull"
                      r"|rm|mv|clean|restore|switch)\b"),
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
#: A tool's subcommand that is another tool: ``uv pip install`` is a ``pip install``.
_NESTED_TOOLS = {("uv", "pip"): "pip"}
#: A tool's subcommand that runs its next plain word as a command: ``uv run pip
#: install x`` installs, ``uv run pytest`` checks.
_LAUNCHERS = frozenset({("uv", "run")})
_LAUNCH = "<launch>"
_OUTER_TOOLS = frozenset(tool for tool, _ in _LAUNCHERS | frozenset(_NESTED_TOOLS))

#: Tool names that change files outright.
EDIT_TOOLS = frozenset({"Edit", "Write"})
#: Tool names that run shell commands.
SHELL_TOOLS = frozenset({"Bash", "run_commands_parallel"})
#: The ``bash_job`` action that reads a background command's outcome.
JOB_WAIT = "wait"

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

    Code and quoted text (:data:`_CITED`) are not searched: a summary
    that quotes a label or a command does not thereby hedge.

    Args:
        text: The summary to search.
    """
    lowered = _CITED.sub(" ", text).lower()
    return [phrase for phrase in HEDGE_PHRASES if phrase in lowered]


def is_scratch_path(path: str) -> bool:
    """Whether the shell word *path* names a file under a ``tmp`` directory.

    Args:
        path: A path operand after quote removal; a known variable is already
            substituted (``$S/log`` after ``S=tmp/x`` in the same command).
            ``tmp/../a.py`` names ``a.py``: the path is normalised first.
    """
    return bool(_SCRATCH.search(posixpath.normpath(path)))


def _in_scratch(cwd: str | None, path: str) -> bool:
    """Whether *path*, named in the directory *cwd*, is a scratch file (:func:`is_scratch_path`).

    Args:
        cwd: The absolute directory the command runs in, as the scanner
            followed the ``cd`` before it; ``None`` when only the shell
            knows it, and *path* is judged as written.
        path: A path operand after quote removal.  One the shell still
            expands (``$OUT``, ``~/x``) may name any file: judged as written.
    """
    if cwd is None or path.startswith("~") or "$" in path or "`" in path:
        return is_scratch_path(path)
    return is_scratch_path(posixpath.join(cwd, path))


def shell_commands(args: Any) -> list[str]:
    """The command strings of a ``Bash`` or ``run_commands_parallel`` call's *args*.

    Args:
        args: The call's arguments as the trajectory recorded them.

    Returns:
        ``[command]`` for ``Bash``; the list behind ``commands`` (a JSON list
        or a list) for ``run_commands_parallel``; empty for anything else.
    """
    if not isinstance(args, dict):
        return []
    if isinstance(args.get("command"), str):
        return [args["command"]]
    commands = args.get("commands")
    if isinstance(commands, str):
        try:
            commands = json.loads(commands)
        except ValueError:
            return [commands]
    if isinstance(commands, list):
        return [c for c in commands if isinstance(c, str)]
    return []


def _outputs(command: SimpleCommand) -> list[Word]:
    """The files the standard output of *command* is redirected to: its own and its groups'."""
    outputs = list(command.outputs)
    group = command.group
    while group is not None:
        outputs.extend(group.outputs)
        group = group.group
    return outputs


def _shell_script(words: list[str]) -> str | None:
    """The command string a shell's arguments *words* run (``bash -c "..."``), or ``None``.

    An option may take the next word as its value (``-o pipefail``); a
    word that is no option is a script file, not shell text.
    """
    index = 0
    while index < len(words):
        if _COMMAND_OPTION.fullmatch(words[index]):
            return words[index + 1] if index + 1 < len(words) else None
        if not words[index].startswith("-"):
            return None
        index += 1
        if index < len(words) and not words[index].startswith("-"):
            index += 1
    return None


def runs_subcommand(words: list[str], tools: dict[str, re.Pattern[str]]) -> bool:
    """Whether the command *words* run a subcommand of one of *tools*.

    A tool named in command position (``pip``, or a python module:
    ``python -m pip``) starts a scan of the words after it: an option
    (``-C``, ``--prefix``, Cargo's ``+stable``) may take the next word as
    its value, and the first word that is neither is the subcommand.  A
    tool named as an argument (``echo git commit``) runs nothing.  The
    scans of one tool in one state go on alike, so one pass keeps only
    the tools whose next plain word is the subcommand (*expect*) and
    those whose next plain word may be an option's value (*optioned*).
    A subcommand that is itself a tool (``uv pip``) is scanned as that
    tool; a launcher's (``uv run``) puts its next plain word in command
    position, after the launcher's options (only those known to take a
    value, :data:`~kiss.agents.sorcar.shell_scan.UV_VALUED_OPTIONS`, may
    take the next word: ``uv run --no-sync echo pip install`` runs
    ``echo``).

    Args:
        words: The command's words after quote removal, wrappers dropped.
        tools: By tool name, the start of its subcommands that count
            (:data:`WRITING_SUBCOMMANDS`); a Node tool's ``run build`` counts too.
    """
    expect: set[str] = set()
    optioned: set[str] = set()
    after_run = False  # the last word was a Node tool's ``run``
    program = 0  # the index of the word in command position
    for index, word in enumerate(words):
        launched = False
        if word.startswith(("-", "+")):
            launch = {_LAUNCH} & (expect | optioned)
            expect, optioned, after_run = set(), (expect | optioned) - launch, False
            if word in UV_VALUED_OPTIONS:
                optioned |= launch
            else:
                expect |= launch
        else:
            scanning = expect | optioned
            if (after_run and _BUILD.match(word)) or any(
                    tools[tool].match(word) for tool in scanning if tool in tools):
                return True
            after_run = word == "run" and not scanning.isdisjoint(_RUN_BUILD)
            launched = _LAUNCH in scanning
            nested = {_NESTED_TOOLS[tool, word] for tool in scanning
                      if (tool, word) in _NESTED_TOOLS and _NESTED_TOOLS[tool, word] in tools}
            if any((tool, word) in _LAUNCHERS for tool in scanning):
                nested.add(_LAUNCH)
            expect, optioned = optioned | nested, set()
        if index == 0 or launched:
            program = index
        name = posixpath.basename(word)
        if (name in tools or name in _OUTER_TOOLS) and (index == program or (
                words[index - 1] == "-m" and _PYTHON.match(posixpath.basename(words[program])))):
            expect.add(name)
    return False


def _named_operands(program: str, words: list[str]) -> list[str]:
    """The files the writing *program* changes among its arguments *words*.

    See :data:`WRITING_PROGRAMS`.  A copy's ``-t DIR`` /
    ``--target-directory DIR`` names its destination (``cp -t src a.py``
    changes ``src``, not ``a.py``).
    """
    if program == "dd":
        return [word[3:] for word in words if word.startswith("of=")]
    operands: list[str] = []
    target: str | None = None
    value_next = False
    for word in words:
        if value_next:
            target, value_next = word, False
        elif program in _TARGET_PROGRAMS and (match := _TARGET_OPTION.match(word)):
            # ``-t DIR``, ``-tDIR``, ``-rt DIR``, ``--target-directory[=| ]DIR``
            target = match["value"] or None
            value_next = target is None
        elif not word.startswith("-"):
            operands.append(word)
    rule = WRITING_PROGRAMS[program]
    if rule == "last":
        return operands[-1:] if target is None else [target]
    if rule == "rest":
        return operands[1:]
    return operands if target is None else [target, *operands]


def _sed_files(words: list[str]) -> list[str]:
    """The files an in-place ``sed`` edits among its arguments *words*.

    The first plain operand is the script unless an ``-e`` / ``-f``
    option (``--expression``, ``--file``; its value the next word,
    attached as in ``-es/a/b/`` or ``-nes/a/b/``, or after ``=``)
    supplied it; the plain operands after it are the files.  ``-l`` /
    ``--line-length`` take a value too; ``-i`` only an attached suffix.
    """
    files: list[str] = []
    script_given = value_next = False
    for word in words:
        if value_next:
            value_next = False
        elif word in ("--expression", "--file", "--line-length"):
            script_given = script_given or word != "--line-length"
            value_next = True
        elif word.startswith(("--expression=", "--file=")):
            script_given = True
        elif word.startswith("--"):
            continue
        elif word.startswith("-"):
            for index, letter in enumerate(word[1:], 2):  # a cluster: -ne 's/a/b/', -nes/a/b/
                if letter in "ef":
                    script_given = True
                    value_next = index == len(word)
                    break
                if letter == "l":
                    value_next = index == len(word)
                    break
                if letter == "i":  # the suffix, if any, is attached
                    break
        elif script_given:
            files.append(word)
        else:
            script_given = True  # the script itself
    return files


def _program_changes(argv: list[str], cwd: str | None) -> list[str]:
    """The files the command *argv* (wrappers dropped) changes: ``[]`` when it changes none.

    The program's name is listed first, then the operands it changes,
    so a change of unnamed files (``make``, ``pip install x``) is listed
    too.  A nested shell's command string is read as commands.  A
    program whose named files are all scratch files in *cwd* (the
    directory it runs in, ``None`` when unknown) changes nothing of the
    work: ``cd /w/tmp/x && cp a b`` as ``cp tmp/x/a tmp/x/b``.
    """
    if not argv:
        return []
    program = posixpath.basename(argv[0])
    if program == "find":
        # The files ``-exec``'s program (or ``-delete``) changes are the matches,
        # under the start paths (``.`` when none): ``{}`` stands for them, and
        # ``-execdir`` runs in their directories.
        starts = list(itertools.takewhile(lambda w: not w.startswith(("-", "(", "!")), argv[1:]))
        starts = starts or ["."]
        if "-delete" in argv:
            return [] if all(_in_scratch(cwd, s) for s in starts) else [program, *starts]
        for index, word in enumerate(argv):
            if word in _FIND_EXEC:
                rest = [w for w in argv[index + 1:] if w not in (";", "+")]
                files = [s for w in rest for s in (starts if w == "{}" else [w])]
                if word in ("-execdir", "-okdir"):
                    cwd = (posixpath.join(cwd, starts[0]) if cwd is not None
                           and all(_in_scratch(cwd, s) for s in starts) else None)
                return _program_changes(files, cwd)
        return []
    if program in SHELLS:
        script = _shell_script(argv[1:])
        if script is None:
            return []
        return [change for command in simple_commands(script, cwd, optimistic=True)
                for change in _changes(command)]
    operands = [word for word in argv[1:] if not word.startswith("-")]
    if program in WRITING_PROGRAMS:
        named = _named_operands(program, argv[1:])
        if named and all(_in_scratch(cwd, path) for path in named):
            return []
        return [program, *named]
    if program == "sed":
        if not any(_SED_INPLACE.match(w) for w in argv[1:]):
            return []
        files = _sed_files(argv[1:])
        if files and all(_in_scratch(cwd, path) for path in files):
            return []
        return [program, *operands]
    if program in BUILD_TOOLS:
        targets = argv[1:]
        only_checks = any(_CHECK_TARGET.match(t) for t in targets) and all(
            _CHECK_TARGET.match(t) or t.startswith("-") or "=" in t for t in targets)
        return [] if only_checks else [program, *operands]
    if any(_FIX_OPTION.match(w) for w in argv[1:]) or runs_subcommand(argv, WRITING_SUBCOMMANDS):
        return [program, *operands]
    return []


def _changes(command: SimpleCommand) -> list[str]:
    """The files one simple *command* changes: its redirection targets, then the program's.

    A redirection into a device (``/dev/null``) or a scratch file (in
    the directory the command runs in: ``cd /w/tmp/x && cat > f``) is a
    check's output, not a change.
    """
    changed = [word.text for word in _outputs(command)
               if not word.text.startswith("/dev/") and not _in_scratch(command.cwd, word.text)]
    changed.extend(_program_changes([word.text for word in command.argv()], command.cwd))
    return changed


def file_changes(args: dict[str, Any]) -> list[str]:
    """The files a ``Bash`` / ``run_commands_parallel`` call changes, as its commands name them.

    Args:
        args: The tool call's arguments (``command`` or a JSON list ``commands``).

    Returns:
        For each simple command that changes files, as the shell reads
        the call's text (:func:`~kiss.agents.sorcar.shell_scan.simple_commands`,
        read as it was meant to run: a variable assigned after ``&&``
        is known to the whole command, ``cd d && T=tmp/x && a; cp b $T/``):
        the targets of its output redirections, the name of a program
        that changes files and the operands it changes
        (:func:`_program_changes`); empty when the call changes nothing
        (a check, a probe, a command naming only scratch files).
    """
    return [change for text in shell_commands(args)
            for command in simple_commands(text, None, optimistic=True)
            for change in _changes(command)]


def writes_files(args: dict[str, Any]) -> bool:
    """Whether a ``Bash`` / ``run_commands_parallel`` call changes files (:func:`file_changes`).

    Args:
        args: The tool call's arguments (``command`` or a JSON list ``commands``).
    """
    return bool(file_changes(args))


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
        elif name == "bash_job" and str(args.get("action", "tail")).strip().lower() == JOB_WAIT:
            # The outcome of a background check (a test suite, a build) is
            # read here, as a foreground shell command's would be.
            self.unchecked_change = False

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
