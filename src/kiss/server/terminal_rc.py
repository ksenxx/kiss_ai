# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Make a terminal tab's command history outlive the tab.

bash and zsh read their history file once, when they start, and write
it once, when they exit.  A terminal tab opened while another tab's
shell is still running (a second tab, a page reloaded while the old
shell sits out its re-attach grace, a daemon that was killed outright)
therefore never saw that shell's commands: Up at the new prompt
recalled only what some earlier, long-closed shell had written.

This module gives the shell of a tab (:mod:`kiss.server.terminal_tab`)
start-up files that run the user's own dotfiles first and then make
every command land in the history file the moment it is entered, the
way VS Code's integrated terminal hooks its shells:

* **bash** starts with ``--rcfile`` pointing at :data:`BASH_RC`, which
  sources ``~/.bashrc`` (or, where the shell stands in for a login
  shell, ``/etc/profile`` and the first of ``~/.bash_profile``,
  ``~/.bash_login``, ``~/.profile``), sets ``histappend`` and adds
  ``history -a`` to ``PROMPT_COMMAND``.
* **zsh** starts with ``ZDOTDIR`` pointing at a directory of four
  shims (``.zshenv``, ``.zprofile``, ``.zshrc``, ``.zlogin``), each
  running the user's file of the same name; ``.zshrc`` then configures
  a history file when the user has none and sets ``INC_APPEND_HISTORY``.
  The user's ``ZDOTDIR`` is restored before the first prompt.
* Any other shell is started as it is (fish keeps its history on its
  own; sh and dash have none).

The files are written under ``<kiss home>/terminal/`` the first time a
tab opens and rewritten whenever their content here changes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from kiss.core.config import kiss_home
from kiss.core.utils import atomic_write_text

logger = logging.getLogger(__name__)

BASH_RC = r"""# Start-up file of a terminal tab (kiss/server/terminal_rc.py).
#
# bash reads this instead of ~/.bashrc (it was started with --rcfile),
# so the user's own files come first.  Then the tab's history is made
# to outlive the tab: bash reads $HISTFILE once at start and writes it
# once at exit, so a tab opened while another one's shell was still
# running never saw that shell's commands.  Here every command is
# appended as soon as it was entered, so Up/Down in a new tab recall
# the commands of every earlier one.
if [ -n "${KISS_TERMINAL_LOGIN:-}" ]; then
  # Stands in for `bash -l` (a login shell ignores --rcfile): the
  # login files, in the order and with the first-found rule bash
  # applies.  As in VS Code's terminal, the shell is not a login shell
  # proper (`shopt -q login_shell` is false, ~/.bash_logout is not run).
  unset KISS_TERMINAL_LOGIN
  if [ -r /etc/profile ]; then . /etc/profile; fi
  if [ -r ~/.bash_profile ]; then . ~/.bash_profile
  elif [ -r ~/.bash_login ]; then . ~/.bash_login
  elif [ -r ~/.profile ]; then . ~/.profile
  fi
elif [ -r ~/.bashrc ]; then
  . ~/.bashrc
fi
# Append to $HISTFILE at exit instead of overwriting it (the other
# tabs' commands are in there too), and after every command.
shopt -s histappend
# bash before 4.0 (macOS's /bin/bash 3.2) appends nothing with
# `history -a` in a shell whose history list started out empty (no
# history file yet, so nothing was read at start-up): such a shell
# appends each newest entry itself.  `fc` cannot list the newest
# entry from PROMPT_COMMAND and $HISTCMD is 1 there, so the entry
# comes from `history 1` (its number and text, no time stamp).
__kiss_terminal_append() {
  local entry
  entry=$(HISTTIMEFORMAT= history 1)
  if [ -n "$entry" ] && [ "$entry" != "${__kiss_terminal_entry:-}" ]; then
    __kiss_terminal_entry=$entry
    printf '%s\n' "${entry#*[0-9][ *] }" >> "$HISTFILE"
  fi
}
if (( BASH_VERSINFO[0] < 4 )) && [ -n "${HISTFILE:-}" ] && [ ! -s "$HISTFILE" ]; then
  __kiss_terminal_hook='__kiss_terminal_append'
else
  unset -f __kiss_terminal_append
  __kiss_terminal_hook='history -a'
fi
# bash 5.1 and later run every element of a PROMPT_COMMAND array;
# earlier versions run only ${PROMPT_COMMAND[0]}, which is what the
# scalar branch appends to.
if (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) \
  && case "$(declare -p PROMPT_COMMAND 2>/dev/null)" in "declare -a"*) true ;; *) false ;; esac
then
  PROMPT_COMMAND+=("$__kiss_terminal_hook")
elif [ -n "${PROMPT_COMMAND:-}" ]; then
  PROMPT_COMMAND="$PROMPT_COMMAND"$'\n'"$__kiss_terminal_hook"
else
  PROMPT_COMMAND="$__kiss_terminal_hook"
fi
unset __kiss_terminal_hook
"""

# The user's dotfiles are sourced at the top level of each shim, never
# inside a function: a `typeset` in a file sourced from a function
# would declare a function-local variable (`typeset -U path` would
# lose the user's PATH additions on return).  Every expansion of a
# variable the user may have unset is unset-safe (`${VAR:-}`), for a
# `setopt nounset` in their files.
ZSH_ZSHENV = r"""# Start-up files of a terminal tab (kiss/server/terminal_rc.py).
#
# zsh reads its dotfiles from $ZDOTDIR, which the tab points at this
# directory so that, after the user's own files, .zshrc can make the
# tab's history outlive the tab.  Each file here runs the user's file
# of the same name with ZDOTDIR as the user has it (set or not, value
# and export state, as their earlier files left it), and the user's
# ZDOTDIR is back in place as soon as no file of this directory is
# left to read.
KISS_TERMINAL_ZDOTDIR="${ZDOTDIR:-}"
# The user's ZDOTDIR: set (an empty one counts: zsh then looks for
# /.zshrc, not ~/.zshrc) and exported when it came from the
# environment (KISS_USER_ZDOTDIR) or they set and export it.
KISS_USER_ZDOTDIR_SET=${+KISS_USER_ZDOTDIR}
KISS_USER_ZDOTDIR_EXPORTED=$KISS_USER_ZDOTDIR_SET
# Exported, but only by an `export ZDOTDIR` of an unset one (see the capture).
KISS_USER_ZDOTDIR_UNASSIGNED=0
KISS_USER_ZDOTDIR="${KISS_USER_ZDOTDIR:-}"

# Put the user's ZDOTDIR in place for one of their files.  (The
# functions set variables with `typeset -g`: a plain assignment in a
# function is warned about under the user's `setopt warn_nested_var`,
# one that creates the variable under their `warn_create_global`, and
# a `typeset +x` without it acts on a variable of the function's own
# under their `unsetopt global_export`.)
__kiss_terminal_prepare() {
  if (( KISS_USER_ZDOTDIR_UNASSIGNED )); then
    # `export` leaves an unset one unset under posix_builtins, and
    # empty without.
    setopt local_options
    if (( KISS_USER_ZDOTDIR_SET )); then unsetopt posix_builtins; else setopt posix_builtins; fi
    unset ZDOTDIR
    export ZDOTDIR
  elif (( KISS_USER_ZDOTDIR_SET )); then
    typeset -g ZDOTDIR="$KISS_USER_ZDOTDIR"
    if (( KISS_USER_ZDOTDIR_EXPORTED )); then export ZDOTDIR; else typeset -g +x ZDOTDIR; fi
  else
    unset ZDOTDIR
  fi
}

# Note what the user's file did to ZDOTDIR; point zsh back here for
# the next shim (unexported: no child of a later file sees this dir).
# Where ZDOTDIR cannot do that (their file made it read-only or
# something other than a plain scalar, or switched zsh to its sh or
# ksh emulation, which reads the files of $HOME whatever ZDOTDIR
# says), and in a restricted shell (which lets the tab neither try
# that in silence nor set HISTFILE), it stays as it is, zsh reads the
# rest of their files itself, and the shim is told (a non-zero status)
# to stop with nothing of the tab's left.
__kiss_terminal_capture() {
  # A subshell tries, under posix_builtins (an unset ZDOTDIR that was
  # made read-only or exported under it still is, whatever the option
  # says now): a ZDOTDIR that cannot be assigned ends it with nothing
  # printed, and one that is no plain scalar keeps another type.
  case "$( [[ -o restricted ]] || { setopt posix_builtins; unsetopt all_export
             typeset -g ZDOTDIR="$KISS_TERMINAL_ZDOTDIR"
             builtin print -rn -- "${(t)ZDOTDIR} "; builtin emulate; } 2>/dev/null )" in
    ("scalar-export "[cz]sh) typeset -g KISS_USER_ZDOTDIR_EXPORTED=1 ;;
    ("scalar "[cz]sh) typeset -g KISS_USER_ZDOTDIR_EXPORTED=0 ;;
    (*) __kiss_terminal_forget; return 1 ;;
  esac
  typeset -g KISS_USER_ZDOTDIR="${ZDOTDIR:-}" KISS_USER_ZDOTDIR_SET=${+ZDOTDIR}
  # `export ZDOTDIR` of an unset one leaves it out of the environment
  # until it is assigned (and unset still under posix_builtins, empty
  # without): put back the same way.  Only a child process can tell,
  # whatever PATH is (`command -p`); one that did not run tells nothing.
  typeset -g KISS_USER_ZDOTDIR_UNASSIGNED=0
  if (( KISS_USER_ZDOTDIR_EXPORTED )) && [[ -z "$KISS_USER_ZDOTDIR" \
    && "$(command -p sh -c 'echo "${ZDOTDIR-unset}"' 2>/dev/null)" == unset ]]; then
    typeset -g KISS_USER_ZDOTDIR_UNASSIGNED=1
  fi
  typeset -g ZDOTDIR="$KISS_TERMINAL_ZDOTDIR"
  typeset -g +x ZDOTDIR
}

# Drop the tab's variables and functions.
__kiss_terminal_forget() {
  unset KISS_USER_ZDOTDIR KISS_USER_ZDOTDIR_SET KISS_USER_ZDOTDIR_EXPORTED \
    KISS_USER_ZDOTDIR_UNASSIGNED KISS_TERMINAL_ZDOTDIR
  unfunction __kiss_terminal_prepare __kiss_terminal_capture __kiss_terminal_forget \
    __kiss_terminal_restore
}

# Put the user's ZDOTDIR back for good and drop the tab's variables.
__kiss_terminal_restore() {
  __kiss_terminal_prepare
  __kiss_terminal_forget
}

__kiss_terminal_prepare
if [[ -r "${ZDOTDIR-$HOME}/.zshenv" ]]; then . "${ZDOTDIR-$HOME}/.zshenv"; fi
__kiss_terminal_capture || return 0
# No further file is read with `unsetopt rcs`, nor by a shell that is
# neither a login shell nor interactive.
if [[ ! -o rcs || ( ! -o login && ! -o interactive ) ]]; then __kiss_terminal_restore; fi
"""

ZSH_ZPROFILE = r"""# See .zshenv in this directory.
__kiss_terminal_prepare
if [[ -r "${ZDOTDIR-$HOME}/.zprofile" ]]; then . "${ZDOTDIR-$HOME}/.zprofile"; fi
__kiss_terminal_capture || return 0
if [[ ! -o rcs ]]; then __kiss_terminal_restore; fi
"""

ZSH_ZSHRC = r"""# See .zshenv in this directory.
# A system rc that ran before this file (macOS's /etc/zshrc) derives
# HISTFILE from the ZDOTDIR of the moment, which was this directory:
# move such a file to where that rc has it for the user (`:-`: with an
# empty ZDOTDIR too it is $HOME's), before their .zshrc looks at it.
if [[ "${HISTFILE:-}" == "$KISS_TERMINAL_ZDOTDIR/"* ]]; then
  HISTFILE="${KISS_USER_ZDOTDIR:-$HOME}/${HISTFILE#"$KISS_TERMINAL_ZDOTDIR"/}"
fi
__kiss_terminal_prepare
if [[ -r "${ZDOTDIR-$HOME}/.zshrc" ]]; then . "${ZDOTDIR-$HOME}/.zshrc"; fi
__kiss_terminal_capture || return 0
# History that outlives the tab.  zsh keeps no history file unless
# told to; then the usual one.
if [[ -z "${HISTFILE:-}" ]]; then
  HISTFILE="${KISS_USER_ZDOTDIR:-$HOME}/.zsh_history"
  (( SAVEHIST > 0 )) || SAVEHIST=10000
  (( HISTSIZE >= SAVEHIST )) || HISTSIZE=$SAVEHIST
fi
# Every command is written as soon as it was entered (zsh writes the
# file once, at exit, by default), so a new tab recalls earlier ones.
if [[ ! -o sharehistory && ! -o incappendhistorytime ]]; then
  setopt incappendhistory
fi
# A login shell still has .zlogin to read from here (unless rcs is off).
if [[ ! -o rcs || ! -o login ]]; then __kiss_terminal_restore; fi
"""

ZSH_ZLOGIN = r"""# See .zshenv in this directory.
__kiss_terminal_prepare
if [[ -r "${ZDOTDIR-$HOME}/.zlogin" ]]; then . "${ZDOTDIR-$HOME}/.zlogin"; fi
__kiss_terminal_capture || return 0
__kiss_terminal_restore
"""

_ZSH_FILES = {
    ".zshenv": ZSH_ZSHENV,
    ".zprofile": ZSH_ZPROFILE,
    ".zshrc": ZSH_ZSHRC,
    ".zlogin": ZSH_ZLOGIN,
}


def rc_dir() -> Path:
    """Return the directory the start-up files are written to."""
    return kiss_home() / "terminal"


def _write(path: Path, text: str) -> None:
    """Write *text* to *path* unless it is there already.

    A file that cannot be read, or is not UTF-8 (truncated, edited by
    hand), is simply rewritten.
    """
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except (OSError, UnicodeDecodeError):
        pass
    # Staged and renamed into place (unique temp name, so two daemons
    # sharing the home never clobber each other's staging file); the
    # shell is the only reader, so the private default mode is fine.
    atomic_write_text(path, text)


def with_persistent_history(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """Return the command and the extra environment that start *argv* with history hooks.

    Args:
        argv: The shell command as :func:`kiss.server.terminal_tab.default_shell`
            builds it (``-l`` makes it a login shell).

    Returns:
        ``(argv, env)``: the command to run and the variables to add to
        its environment.  A shell this module knows nothing about, or
        start-up files that cannot be written, give *argv* back
        unchanged with no extra variables (history then works as the
        shell's own dotfiles have it).
    """
    name = os.path.basename(argv[0])
    try:
        if name == "bash":
            rc = rc_dir() / "bashrc"
            _write(rc, BASH_RC)
            # A login shell ignores --rcfile: the rc file stands in
            # for -l when told to.
            login = "-l" in argv[1:]
            rest = [a for a in argv[1:] if a != "-l"]
            env = {"KISS_TERMINAL_LOGIN": "1"} if login else {}
            return [argv[0], "--rcfile", str(rc), *rest], env
        if name == "zsh":
            zdotdir = rc_dir() / "zdotdir"
            for filename, text in _ZSH_FILES.items():
                _write(zdotdir / filename, text)
            env = {"ZDOTDIR": str(zdotdir)}
            # An empty one is passed along too: to zsh it is not an unset one.
            user_zdotdir = os.environ.get("ZDOTDIR")
            if user_zdotdir is not None:
                env["KISS_USER_ZDOTDIR"] = user_zdotdir
            return list(argv), env
    except OSError as exc:
        logger.warning("terminal history start-up files not written: %s", exc)
    return list(argv), {}


__all__ = [
    "BASH_RC",
    "ZSH_ZLOGIN",
    "ZSH_ZPROFILE",
    "ZSH_ZSHENV",
    "ZSH_ZSHRC",
    "rc_dir",
    "with_persistent_history",
]
