# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""What a shell command changes: the file-change reader of the coding gate.

The coding gate (:mod:`kiss.agents.sorcar.coding_gate`) and the
containerised coding SEA (:mod:`kiss.agents.seas.coding.coding_sea`) need
to know when the agent's work changed: to arm the gate on a code file a
shell command wrote, to count the changes between a passing run of the
verifier and a finish (the SEA reuses the agent's own run when nothing
changed, the gate its own last one), and to review only after a change.
What a shell command changes is read from its simple commands
(:func:`~kiss.agents.sorcar.shell_scan.simple_commands`), the way the
shell reads them: a program such as ``rm`` or ``cp`` counts in command
position only (``cat install.sh`` and ``git commit -m "rm stale"`` run
nothing of the kind), a here-document is input to the program that
reads it, and a file under a ``tmp`` directory is the run's scratch
(a saved probe output, a copied log), not a change of the work.

The module once also refused a ``finish`` whose summary hedged or that
followed an edit no command checked.  Those refusals were answered by a
rewording of the summary far more often than by a check, and the
measurement of 2026-11-04 found no pass that depended on them, so the
gate no longer refuses on the summary's wording.
"""

from __future__ import annotations

import itertools
import posixpath
import re
from typing import Any

from kiss.agents.sorcar.fanout_guard import SUBAGENT_DIR_PREFIX, parse_tasks_json
from kiss.agents.sorcar.shell_scan import (
    SHELLS,
    UV_VALUED_OPTIONS,
    SimpleCommand,
    Word,
    command_text,
    shell_script_index,
    simple_commands,
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
#: ``chmod``'s (``chown``'s, ``chgrp``'s) option naming the file whose mode or
#: owner to copy, attached (``--reference=RFILE``) or in the next word: the
#: mode is then no operand, and every operand is a file.
_REFERENCE_OPTION = re.compile(r"--reference=?(?P<value>.*)")
#: A ``chmod`` mode that starts with a dash (``-x``, ``-rw,o=r``): its first
#: operand, not an option (the characters GNU ``chmod`` reads a mode from).
_CHMOD_MODE = re.compile(r"-[rwxXstugoa,+=0-7]")
#: A python interpreter in command position, whose ``-m MODULE`` runs a tool.
_PYTHON = re.compile(r"python[0-9.]*$")
#: ``sed``'s in-place option, among its options in any position (``--i``:
#: any prefix of ``--in-place``, its only long option in ``i``).  An
#: in-place edit of scratch files only (``sed -i s/a/b/ tmp/x/f``) changes
#: nothing of the work, like a copy into ``tmp/``.
_SED_INPLACE = re.compile(r"-[a-zA-Z]*i|--i")
#: The build tools whose bare call builds; a call naming only check targets
#: (``make test``, ``make -k check V=1``) builds nothing.
BUILD_TOOLS = frozenset({"make", "cmake", "ninja"})
_CHECK_TARGET = re.compile(r"(?:test|check)\b")
#: The option that fixes files in place (``ruff check --fix``, ``check --full --fix``).
_FIX_OPTION = re.compile(r"--fix\b")
#: ``find``'s actions that run a command on each file found.
_FIND_EXEC = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
#: A path under a ``tmp`` directory: the scratch directory of the run (``tmp/x``,
#: ``./tmp/x``, ``/home/u/repo/tmp/x``) or the system's (``/tmp/x``); or one in a
#: sub-agent's private folder, wherever the system keeps its temp directory.
_SCRATCH = re.compile(rf"(?:^|/)(?:tmp(?:/|$)|{SUBAGENT_DIR_PREFIX}[^/]*/)")

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
        ``[command]`` for ``Bash``.  For ``run_commands_parallel``, the
        commands as the tool decodes them
        (:func:`~kiss.agents.sorcar.fanout_guard.parse_tasks_json`); text
        the tool refuses (no array of commands: it runs nothing) is read
        as one command, and a list gives its strings.  Empty for anything
        else.
    """
    if not isinstance(args, dict):
        return []
    if isinstance(args.get("command"), str):
        return [args["command"]]
    commands = args.get("commands")
    if isinstance(commands, str):
        try:
            return parse_tasks_json(commands)
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

    See :data:`WRITING_PROGRAMS`.  Every word after a ``--`` is an
    operand.  A copy's ``-t DIR`` / ``--target-directory DIR`` names its
    destination (``cp -t src a.py`` changes ``src``, not ``a.py``), and
    a copy into a directory writes a file of each source's name there,
    listed too (``src/a.py``): for a directory by the words (``-t``,
    several sources, a trailing ``/``) and, the words not telling, for
    a destination whose name has no suffix.  A ``chmod
    --reference=RFILE a.py`` takes its mode from ``RFILE`` (read, not
    changed) and changes ``a.py``, its first operand; a ``chmod`` mode
    that starts with a dash (``chmod -x a.py``) is the first operand,
    not an option.
    """
    if program == "dd":
        return [word[3:] for word in words if word.startswith("of=")]
    operands: list[str] = []
    target: str | None = None
    value_next = skip_next = False
    rule = WRITING_PROGRAMS[program]
    for position, word in enumerate(words):
        if skip_next:
            skip_next = False
        elif value_next:
            target, value_next = word, False
        elif word == "--":
            operands.extend(words[position + 1:])
            break
        elif program in _TARGET_PROGRAMS and (match := _TARGET_OPTION.match(word)):
            # ``-t DIR``, ``-tDIR``, ``-rt DIR``, ``--target-directory[=| ]DIR``
            target = match["value"] or None
            value_next = target is None
        elif rule == "rest" and (match := _REFERENCE_OPTION.match(word)):
            # ``--reference[=| ]RFILE`` supplies the mode or owner: every operand is a file
            rule, skip_next = "all", not match["value"]
        elif not word.startswith("-") or (program == "chmod" and _CHMOD_MODE.match(word)):
            operands.append(word)
    if rule == "last":
        directory = target is not None  # ``-t DIR``
        if target is None:
            if not operands:
                return []
            target = operands.pop()
        if directory or len(operands) > 1 or not posixpath.splitext(posixpath.basename(target))[1]:
            names = map(posixpath.basename, operands)  # ``""`` for a source named ``dir/``
            return [target, *(posixpath.join(target, name) for name in names if name)]
        return [target]
    if rule == "rest":
        return operands[1:]
    return operands if target is None else [target, *operands]


def _sed_files(words: list[str]) -> list[str]:
    """The files an in-place ``sed`` edits among its arguments *words*.

    GNU sed reads its options wherever they stand among the operands,
    up to a ``--``, after which every word is an operand.  The first
    operand is the script unless an ``-e`` / ``-f`` option
    (``--expression``, ``--file`` or an unambiguous prefix of either;
    its value the next word, attached as in ``-es/a/b/`` or
    ``-nes/a/b/``, or after ``=``) supplied it, before or after it;
    the other operands are the files.  ``-l`` / ``--line-length`` take
    a value too; ``-i`` only an attached suffix.
    """
    operands: list[str] = []
    script_given = value_next = False
    for position, word in enumerate(words):
        if value_next:
            value_next = False
        elif word == "--":
            operands.extend(words[position + 1:])
            break
        elif word.startswith("--"):
            name, attached, _ = word[2:].partition("=")
            # ``--f`` is no script option: it is ``--follow-symlinks`` as much as ``--file``
            script = "expression".startswith(name) or (name != "f" and "file".startswith(name))
            script_given = script_given or script
            value_next = not attached and (script or "line-length".startswith(name))
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
        else:
            operands.append(word)
    return operands if script_given else operands[1:]


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
        changes: list[str] = []
        if "-delete" in argv and not all(_in_scratch(cwd, s) for s in starts):
            changes = [program, *starts]
        # A match's directory is the start's own parent at the least (the
        # start itself is a match), so an ``-execdir`` command whose
        # operands stay below it (no ``..``) names scratch files when every
        # such parent is scratch; else they are judged as written.
        execdir_cwd = (posixpath.join(cwd, starts[0]) if cwd is not None and all(
            is_scratch_path(posixpath.dirname(posixpath.normpath(posixpath.join(cwd, s))))
            for s in starts) else None)
        index = 0
        while index < len(argv):
            word = argv[index]
            index += 1
            if word not in _FIND_EXEC:
                continue
            # The clause's command runs up to its ``;`` or ``+``; every clause counts.
            end = next((i for i in range(index, len(argv)) if argv[i] in (";", "+")), len(argv))
            files = [s for w in argv[index:end] for s in (starts if w == "{}" else [w])]
            clause_cwd = cwd
            if word in ("-execdir", "-okdir"):
                clause_cwd = None if any(".." in w for w in argv[index:end]) else execdir_cwd
            changes.extend(_program_changes(files, clause_cwd))
            index = end + 1
        return changes
    if program in SHELLS:
        script = shell_script_index(argv[1:])
        if script is None:
            return []
        return [change for command in simple_commands(argv[script + 1], cwd, optimistic=True)
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
        # a file ``--`` protects may start with a dash
        return [program, *operands, *(path for path in files if path.startswith("-"))]
    if program in BUILD_TOOLS:
        targets = argv[1:]
        only_checks = any(_CHECK_TARGET.match(t) for t in targets) and all(
            _CHECK_TARGET.match(t) or t.startswith("-") or "=" in t for t in targets)
        return [] if only_checks else [program, *operands]
    if any(_FIX_OPTION.match(w) for w in argv[1:]) or runs_subcommand(argv, WRITING_SUBCOMMANDS):
        return [program, *operands]
    if program == "uv":
        # ``uv [options] run [options] command...`` runs that command, whose
        # changes are its own (``uv run sed -i s/a/b/ f`` edits ``f``).
        index = 1
        while index < len(argv) and argv[index].startswith("-"):
            index += 2 if argv[index] in UV_VALUED_OPTIONS else 1
        if index < len(argv) and argv[index] == "run":
            index += 1
            while index < len(argv) and argv[index].startswith("-"):
                index += 2 if argv[index] in UV_VALUED_OPTIONS else 1
            return _program_changes(argv[index:], cwd)
    return []


def _changes(command: SimpleCommand) -> list[str]:
    """The files one simple *command* changes: its redirection targets, then the program's.

    A redirection into a device (``/dev/null``) or a scratch file (in
    the directory the command runs in: ``cd /w/tmp/x && cat > f``) is a
    check's output, not a change.
    """
    argv = command.argv()
    text = command_text(argv)
    if text is not None and simple_commands(text, command.cwd, optimistic=True):
        # ``eval 'text'``, ``bash -c 'text'``: the scanner read the text as
        # commands of its own, which carry this command's redirections too.
        # A text that runs no command (a comment, blanks) carries nothing:
        # the redirections are this command's, which still truncate.
        return []
    changed = [word.text for word in _outputs(command)
               if not word.text.startswith("/dev/") and not _in_scratch(command.cwd, word.text)]
    changed.extend(_program_changes([word.text for word in argv], command.cwd))
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
