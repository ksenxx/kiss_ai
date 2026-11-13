# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Deterministic guards shared by every Sorcar shell tool.

Two mechanisms, both born in the HarnessTax container harness
(:mod:`kiss.agents.seas.coding.coding_sea`) and applied to the host and
Docker ``Bash`` / ``run_commands_parallel`` tools alike:

* :func:`destructive_command_guard` refuses the few shell commands that end
  the task itself: killing every process (``kill -1``, ``pkill -f .``; the
  agent and its own shell included) or deleting the root or the working
  directory.  A model that meant "stop my server" and typed ``kill -9 -1``
  otherwise ends the run with nothing to deliver.
* :func:`lift_install_timeout` raises the timeout of package installs and
  builds to at least :data:`INSTALL_TIMEOUT_SECONDS`.  A short tool timeout
  leaves them half done (broken dpkg state, partial builds) and the model
  then spends turns repairing and retrying; lifting the deadline in place is
  cheaper than letting the model guess it.
"""

from __future__ import annotations

import functools
import os
import posixpath
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath

#: ``_END`` closes a shell word: whitespace, end of command, a separator, a
#: redirection, a comment or a closing backquote.  ``_WORD`` is one argument
#: of a simple command: it stops at a separator, a redirection or a newline and cannot start a
#: comment, so ``_ARGS`` (zero or more arguments) never crosses into the next
#: command or a trailing ``# comment``.  ``_CMD`` is where a command name may
#: start: the beginning or a separator (a backquote too; a ``)``, which
#: closes a ``case`` pattern, a function's ``()`` or a substitution), then
#: the reserved words a command follows (``{``, spelled ``\x7b`` in this
#: format template, ``!``, ``if``, ``then``, ``do``...), optional
#: environment assignments, one wrapper (``sudo``, ``env``, ``nohup``,
#: ``timeout 5``...) with its own arguments (``sudo -n``, ``sudo -u root``,
#: ``env DEBUG=1``; the ``name {`` of a ``function`` or ``coproc``, read as
#: wrappers too), or one shell's ``-c`` with the quote opening its text
#: (``bash -c 'kill -1'``; the shell's options may go on after the ``-c``
#: cluster, ``bash -c -e 'kill -1'``, ``bash -co posix 'kill -1'``, ``bash -c
#: -- 'kill -1'``), and an optional directory prefix.  ``_CMD`` is
#: a template: ``{commands}`` is the alternation of the command names the
#: pattern after it matches.  Each group tokenises the text in exactly one
#: way (words are delimited by blanks and assignments contain ``=``), so
#: matching stays linear in the command length; a repeated wrapper (``sudo
#: -n sudo -n …``) is absorbed by the single wrapper's argument list instead
#: of a nested repetition, which backtracked exponentially.  The wrapper's
#: arguments (``_WRAPPER_ARGS``) end at the first of those command names
#: (under a directory prefix too), the shell's at its first ``-c`` cluster
#: (``_SHELL_C``: the ``c`` is the cluster's first ``c``, so ``-ccc…c`` is
#: read once rather than once per ``c``) and the options after the cluster
#: (``_SHELL_ARGS``) at the first command name, quoted or not: as an
#: arbitrary word list every later word was a candidate command, and each
#: ``rm`` candidate of ``sudo echo rm rm rm …`` rescanned the whole suffix
#: for a protected operand, quadratic in the command length.  The pattern
#: after the prefix reads every word of the simple command, so its first
#: occurrence stands for every later one (``xargs -I rm rm -rf /``): one
#: scan of the suffix per command name.  That holds only when ``{commands}``
#: names the one command the pattern after it matches: a wrapper's option
#: values may be spelled like any number of other guarded commands
#: (``timeout -s kill 5 rm -rf /``, ``env -u kill -u pkill rm -rf /``), and
#: stopping at one of them hid the real command, so each guarded command
#: gets its own prefix.  The blanks after a separator stop at a newline,
#: which is a separator of its own: letting them run across newlines made
#: every newline of a blank-line command swallow the rest of the text and
#: backtrack through it, quadratic in the command length.  An assignment's
#: value and the directory prefix stop at a separator too
#: (``_UNSEPARATED``), where the next command may start: as ``\S*`` they
#: ran from every separator of ``:;:;:;...`` to the end of the text and
#: backtracked through it, quadratic again.  A backquote is such a
#: separator, for them and for ``_WORD``: read across, every backquote of
#: a run rescanned the rest of the text.
_END = r"(?=\s|$|[;&|)\"'>#`])"
_WORD = r"[^\s;&|<>()#`][^\s;&|<>()`]*"
_ARGS = r"(?:" + _WORD + r"[ \t]+)*"
_ARGS_LAZY = r"(?:" + _WORD + r"[ \t]+)*?"
_UNSEPARATED = r"[^\s;&|()`]*"
_WRAPPER = r"sudo|env|exec|nohup|nice|time|command|builtin|xargs|timeout|function|coproc"
_COMMAND_NAME = r"(?:" + _UNSEPARATED + r"/)?(?:{commands})"
_WRAPPER_ARGS = r"(?:[ \t]+(?!" + _COMMAND_NAME + _END + r")" + _WORD + r")*[ \t]+"
_SHELL_C = r"-[abd-zA-Z]*c[a-zA-Z]*"
_SHELL_ARGS = r"(?:[ \t]+(?!['\"]?" + _COMMAND_NAME + _END + r")" + _WORD + r")*[ \t]+['\"]?"
_CMD = (
    r"(?:^|[;&|()`\n])[^\S\n]*(?:(?:\x7b|!|if|then|elif|else|while|until|do)[ \t]+)*"
    r"(?:\w+=" + _UNSEPARATED + r"[ \t]+)*"
    r"(?:(?:" + _WRAPPER + r")" + _WRAPPER_ARGS
    + r"|(?:bash|sh|zsh|dash|ksh)(?:[ \t]+(?!" + _SHELL_C + r"[ \t])" + _WORD + r")*"
    r"[ \t]+" + _SHELL_C + _SHELL_ARGS + r")?"
    r"(?:" + _UNSEPARATED + r"/)?"
)
#: A path operand: quoted on both sides or not at all, so ``"/work backup"`` is
#: not read as ``/work``; the ``{path}`` placeholder takes the path pattern.
_QUOTED_OR_BARE = r"(?:'{path}'|\"{path}\"|{path})"

#: Template of the destructive-command pattern; ``{workdir}`` is the escaped
#: working directory without its trailing slash (``/`` for the root).
#:
#: * ``kill -1`` targets every process unless ``-1`` is the signal of a
#:   ``kill -1 <pid>``: a following pid, ``$var``, ``%job`` or backquote
#:   (optionally quoted) exempts ``-1`` only when it is the first argument
#:   (after ``kill -9 -1 1234`` the signal is taken, so ``-1`` is the
#:   all-processes target).  The ``-1`` may be quoted (``kill -9 "-1"``).
#: * ``pkill -f .`` may carry options before and after ``-f``, and ``-f`` may
#:   be combined with other short flags (``pkill -ef .``).
#: * The ``rm`` operand may follow options and other operands of the same
#:   command (``rm -rf a b /``), with an optional trailing ``/``, ``*`` or
#:   ``/*``; the working directory counts through its parent too
#:   (``{workdir}/..``, which holds it, and ``{workdir}/../{name}``), as does
#:   the root as ``/..``.
#:
#: The patterns are plain regexes on the command text: a quoted mention such as
#: ``echo 'a; rm -rf /'`` is refused too, and the model rephrases it.
_PKILL_FULL = r"(?:-[a-zA-Z0-9]*f[a-zA-Z0-9]*|--full)"
_MINUS_ONE = _QUOTED_OR_BARE.format(path="-1") + _END
_KILL_ALL = (
    r"kill[ \t]+(?:" + _MINUS_ONE + r"(?![ \t]+['\"]?[\d$%`])|" + _WORD + r"[ \t]+" + _ARGS
    + _MINUS_ONE + r")"
)
# The words before the first ``-f`` are matched greedily with a lookahead
# that stops at it, so the split is unique and the match stays linear.
_PKILL_ALL = (
    r"pkill[ \t]+(?:(?!" + _PKILL_FULL + r"[ \t])" + _WORD + r"[ \t]+)*" + _PKILL_FULL
    + r"[ \t]+" + _ARGS_LAZY + _QUOTED_OR_BARE.format(path=r"\.(?:\*)?") + _END
)
_RM_PROTECTED = (
    r"rm[ \t]+" + _ARGS
    + _QUOTED_OR_BARE.format(path=r"(?:/(?:\.\.)?|{workdir}(?:/\.\.(?:/{name})?)?)/?\*?")
    + _END
)
DESTRUCTIVE_COMMANDS = r"|".join(
    _CMD.format(commands=name) + body
    for name, body in (("kill", _KILL_ALL), ("pkill", _PKILL_ALL), ("rm", _RM_PROTECTED))
)
#: The working directory named from inside it, where every call starts:
#: ``*`` and ``./*`` (its content), and ``$PWD``, ``$(pwd)`` or
#: ``../{name}`` with the trailing ``/``, ``*`` or ``/*`` of the absolute
#: spellings.  The delete runs elsewhere only when it is ``entered``: a
#: plain ``rm`` right after ``cd DIR &&`` (or ``pushd``; or ``cd DIR; then``),
#: so it runs only once the ``cd`` succeeded.  ``_ENTERS`` is that ``cd``:
#: it starts a command of the shell that runs the ``rm`` (the text, or what
#: follows a ``;``, ``&`` or ``(`` that no backslash escapes and that is no
#: part of a redirection or a pipe, ``>&``, ``|&``; not a new line, which
#: the line before, or a here-document's, may continue into, nor a ``!``,
#: which inverts it, nor a wrapper, which runs it in a child) and DIR is
#: one literal path, not an option, ``-``, ``~`` or an expansion.
#: ``_CD`` is any mention of a ``cd``: one before the ``cd`` that enters
#: (an earlier ``cd``, a function of that name) and nothing is proven.
_ENTERS = (
    r"(?:^|(?<![\\<>|])[;&(])[^\S\n]*(?:(?:\x7b|if|then|elif|else|while|until|do)[ \t]+)*"
    r"(?:cd|pushd)[ \t]+(?P<quote>['\"]?)(?P<entered>(?![-+])/?[\w.+-]+(?:/[\w.+-]+)*/?)(?P=quote)"
    r"[ \t]*(?:&&|;[ \t]*then[ \t])[ \t]*"
)
_RM_HERE = (
    r"(?:" + _ENTERS + r"|" + _CMD.format(commands="rm") + r")rm[ \t]+" + _ARGS
    + _QUOTED_OR_BARE.format(
        path=r"(?:(?:\./)?\*|(?:\$PWD|\$\{{PWD\}}|\$\(pwd\)|`pwd`|\.\./{name})/?\*?)")
    + _END
)
_CD = re.compile(r"\b(?:cd|pushd)\b")
#: What every delete the guard refuses holds (``_RM_PROTECTED``, ``_RM_HERE``): only for a
#: command with one does it matter what the work directory resolves to.
_DELETES = re.compile(r"\brm[ \t]")

#: Text the model sees instead of running a destructive command.
DESTRUCTIVE_VERDICT = (
    "REFUSED: the Bash call did NOT run: it would kill every process (your own shell and "
    "the agent included) or delete the working directory, which ends the task with nothing "
    "to deliver. Kill only the processes you started, by pid or exact name, and delete "
    "only files you created."
)

#: Commands whose runtime is dominated by package managers or compilers.  Any
#: arguments may separate the tool from its subcommand (``pip --cache-dir
#: /tmp/pip install``, ``uv pip install``, ``npm run build``).  An installer's
#: subcommand may follow any installer name, so the installers share one
#: prefix; ``python -m pip`` holds the installer name ``pip`` and needs none of
#: its own.  The builders, matched by name alone, get their own prefix.
_INSTALLERS = r"apt-get|apt|dpkg|pip3?|uv|conda|mamba|npm|yarn|pnpm|cargo|gem|go"
_BUILDERS = r"make|cmake|ninja"
INSTALL_COMMANDS = re.compile(
    _CMD.format(commands=_INSTALLERS)
    + r"(?:" + _INSTALLERS + r"|python3?[ \t]+-m[ \t]+pip)[ \t]+" + _ARGS_LAZY
    + r"(?:install|ci|sync|add|update|upgrade|build|configure)" + _END
    + r"|" + _CMD.format(commands=_BUILDERS) + r"(?:" + _BUILDERS + r")" + _END
)

#: Minimum timeout, in seconds, of an install or build command.
INSTALL_TIMEOUT_SECONDS = 900


@functools.lru_cache(maxsize=64)
def destructive_pattern(workdir: str) -> re.Pattern[str]:
    """Return the compiled destructive-command pattern for *workdir*.

    Args:
        workdir: The working directory whose deletion must be refused
            (``""`` or ``"/"`` protect the root only).
    """
    escaped = re.escape(workdir.rstrip("/") or "/")
    name = re.escape(Path(workdir).name)
    return re.compile(DESTRUCTIVE_COMMANDS.format(workdir=escaped, name=name))


@functools.lru_cache(maxsize=64)
def _rm_here(name: str) -> re.Pattern[str]:
    """Return the compiled pattern of a delete of the directory *name* from inside it."""
    return re.compile(_RM_HERE.format(name=re.escape(name)))


def _slash_path(path: str) -> str:
    """Return *path* with shell separators, preserving any Windows drive prefix.

    Only Windows paths are rewritten: on POSIX a backslash is a legal
    file-name character, not a separator.
    """
    return path.replace("\\", "/") if os.name == "nt" else path


def _walk_realpath(path: str, realpath: Callable[[str], str]) -> str:
    """Resolve *path* left to right so links before ``..`` keep Bash semantics.

    Windows' :func:`os.path.realpath` removes ``..`` lexically before it
    resolves a preceding directory link.  Git Bash instead follows that link
    first when the normalized logical path does not exist, as POSIX realpath
    does.  Resolving each prefix before walking up gives both hosts the same
    answer and also works with the container callbacks used by the guard.
    """
    if re.match(r"^[A-Za-z]:/", path):
        current, tail = path[:3], path[3:]
    elif path.startswith("//"):
        current, tail = "//", path[2:]
    elif path.startswith("/"):
        current, tail = "/", path[1:]
    else:
        current, tail = "", path
    for part in tail.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            current = posixpath.dirname(_slash_path(realpath(current)))
        else:
            current = posixpath.join(current, part)
    return _slash_path(realpath(current))


def _deleted_from_inside(
    command: str, spelling: str, work: PurePosixPath,
    isdir: Callable[[str], bool], realpath: Callable[[str], str],
) -> bool:
    """Return whether *command* deletes the work directory *spelling* by a name from inside it.

    A delete that is not entered (``_RM_HERE``) runs where the call
    starts, in the work directory.  An entered one runs in the directory
    its ``cd`` names (a ``cd`` mentioned before it, ``_CD``, leaves
    nothing proven).  It deletes the work directory when that directory
    is the work directory, one above it, or one beside it, where
    ``../<name>`` is the work directory.  The directory is taken from
    the work directory by the text, and as the filesystem that holds
    it names it (*isdir* and *realpath* ask it; *work* is the work
    directory so named; a link back into the work directory is the
    work directory): where bash's ``cd`` lands, which takes ``..`` by
    the text, or the path as given when that one does not exist.
    """
    name = PurePosixPath(spelling).name
    for delete in _rm_here(name).finditer(command):
        entered = delete["entered"]
        if entered is None or _CD.search(command, 0, delete.start()):
            return True
        given = posixpath.join(spelling, entered)
        here = posixpath.normpath(given)
        beside = posixpath.join(posixpath.dirname(here), name)
        if any(map(PurePosixPath(spelling).is_relative_to, (here, beside))):
            return True
        landed = (
            _slash_path(realpath(here)) if isdir(here) else _walk_realpath(given, realpath)
        )
        beside = _slash_path(realpath(posixpath.join(posixpath.dirname(landed), name)))
        if any(map(work.is_relative_to, (landed, beside))):
            return True
    return False


def destructive_command_guard(
    command: str, work_dir: str | None,
    isdir: Callable[[str], bool] = os.path.isdir,
    realpath: Callable[[str], str] = os.path.realpath,
) -> str | None:
    """Refuse a shell command that would end the task itself.

    Both the spelling of *work_dir* the agent was given and its resolved
    path are protected, so a symlinked work directory cannot be deleted
    through either name, nor from inside (``rm -rf *``): every call
    starts in it, and only a ``cd`` proven to have left it moves the
    delete elsewhere (:func:`_deleted_from_inside`).

    Args:
        command: The shell command line the model wants to run.
        work_dir: The agent's working directory (``None`` protects only the
            root and the process table).
        isdir: Whether a path is a directory where the command runs.
        realpath: The path the filesystem there gives a path, links
            resolved as far as it exists (``os.path.realpath``).  Both
            default to this host's; a container's tools pass its own:
            its work directory is no path of this host.  Asked only
            about a command that deletes (:data:`_DELETES`).

    Returns:
        :data:`DESTRUCTIVE_VERDICT` when the command is refused, else
        ``None``.
    """
    if not work_dir or not _DELETES.search(command):
        refused = destructive_pattern(work_dir or "/").search(command)
        return DESTRUCTIVE_VERDICT if refused else None
    spelling = _slash_path(work_dir)
    resolved_native = realpath(work_dir)
    resolved = _slash_path(resolved_native)
    # Keep the native spelling too: callers and tests may put a Windows
    # path with backslashes into the command even though Git Bash normally
    # receives its slash spelling.
    protected = {work_dir, spelling, resolved_native, resolved}
    if any(destructive_pattern(path).search(command) for path in protected):
        return DESTRUCTIVE_VERDICT
    for path in {spelling, resolved}:
        if _deleted_from_inside(
                command, path, PurePosixPath(resolved), isdir, realpath):
            return DESTRUCTIVE_VERDICT
    return None


def lift_install_timeout(command: str, timeout_seconds: float) -> float:
    """Return the timeout to run *command* with.

    Args:
        command: The shell command line (or one command of a parallel batch).
        timeout_seconds: The timeout the model asked for.

    Returns:
        ``timeout_seconds``, raised to :data:`INSTALL_TIMEOUT_SECONDS` when
        the command is a package install or a build and asked for less.
    """
    if INSTALL_COMMANDS.search(command):
        return max(timeout_seconds, INSTALL_TIMEOUT_SECONDS)
    return timeout_seconds


def refuse_or_lift(
    command: str, work_dir: str | None, timeout_seconds: float,
    isdir: Callable[[str], bool] = os.path.isdir,
    realpath: Callable[[str], str] = os.path.realpath,
) -> tuple[str | None, float]:
    """Apply :func:`destructive_command_guard` and :func:`lift_install_timeout` together.

    Shared by the host ``Bash`` tool and the Docker ``Bash`` / parallel
    exec so a command is refused, and its deadline raised, identically
    on every path.

    Args:
        command: The shell command the model wants to run.
        work_dir: The directory whose deletion is refused.
        timeout_seconds: The caller's deadline.
        isdir: As for :func:`destructive_command_guard`.
        realpath: As for :func:`destructive_command_guard`.

    Returns:
        ``(refusal, timeout_seconds)``: the guard's error message (``None``
        when the command may run) and the possibly lifted deadline.
    """
    guard = destructive_command_guard(command, work_dir, isdir, realpath)
    return guard, lift_install_timeout(command, timeout_seconds)
