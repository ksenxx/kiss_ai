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
import re
from pathlib import Path

#: ``_END`` closes a shell word: whitespace, end of command, a separator, a
#: redirection or a comment.  ``_CMD`` is where a command name may start: the
#: beginning or a separator, then optional ``sudo`` or environment assignments
#: and an optional directory prefix.
_END = r"(?=\s|$|[;&|)\"'>#])"
_CMD = r"(?:^|[;&|(\n])\s*(?:sudo\s+|\w+=\S*\s+)*(?:\S*/)?"

#: Template of the destructive-command pattern; ``{workdir}`` is the escaped
#: working directory without its trailing slash (``/`` for the root).  The
#: ``rm`` operand may follow options and other operands of the same shell
#: command (``rm -rf a b /``; ``_OPERAND`` stops at a separator, a redirection
#: or a newline), with an optional trailing ``/``, ``*`` or ``/*``.  The
#: patterns are plain regexes on the command text: a quoted mention such as
#: ``echo 'a; rm -rf /'`` is refused too, and the model rephrases it.
_OPERAND = r"(?:[^\s;&|<>()]+[ \t]+)*"
DESTRUCTIVE_COMMANDS = (
    _CMD + r"(?:kill\s+(?:\S+\s+)*?-1" + _END + r"(?!\s+[\d$%`])"
    + r"|pkill\s+(?:-\S+\s+)*(?:-f|--full)\s+['\"]?\.(?:\*)?['\"]?" + _END
    + r"|rm[ \t]+" + _OPERAND + r"['\"]?(?:/|{workdir})/?\*?['\"]?" + _END + r")"
)

#: Text the model sees instead of running a destructive command.
DESTRUCTIVE_VERDICT = (
    "Blocked: this command would kill every process (your own shell and the agent "
    "included) or delete the working directory, which ends the task with nothing to "
    "deliver. Kill only the processes you started, by pid or exact name, and delete "
    "only files you created."
)

#: Commands whose runtime is dominated by package managers or compilers.
INSTALL_COMMANDS = re.compile(
    _CMD + r"(?:(?:apt-get|apt|dpkg|pip3?|python3?\s+-m\s+pip|uv\s+pip|uv|conda|mamba|npm|yarn|"
    r"pnpm|cargo|gem|go)\s+(?:-\S+\s+)*(?:run\s+)?"
    r"(?:install|ci|sync|add|update|upgrade|build|configure)"
    + _END + r"|(?:make|cmake|ninja)" + _END + r")"
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
    return re.compile(DESTRUCTIVE_COMMANDS.format(workdir=escaped))


def destructive_command_guard(command: str, work_dir: str | None) -> str | None:
    """Refuse a shell command that would end the task itself.

    Both the spelling of *work_dir* the agent was given and its resolved
    path are protected, so a symlinked work directory cannot be deleted
    through either name.

    Args:
        command: The shell command line the model wants to run.
        work_dir: The agent's working directory (``None`` protects only the
            root and the process table).

    Returns:
        :data:`DESTRUCTIVE_VERDICT` when the command is refused, else
        ``None``.
    """
    spellings = {work_dir or "/"}
    if work_dir:
        spellings.add(str(Path(work_dir).resolve()))
    for spelling in spellings:
        if destructive_pattern(spelling).search(command):
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
