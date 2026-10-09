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
# of the same name with ZDOTDIR as the user has it (value and export
# state, as their earlier files left it), and the user's ZDOTDIR is
# back in place as soon as no file of this directory is left to read.
KISS_TERMINAL_ZDOTDIR="${ZDOTDIR:-}"
# The user's ZDOTDIR: empty when they have none; exported when it came
# from the environment (KISS_USER_ZDOTDIR) or they export it.
KISS_USER_ZDOTDIR="${KISS_USER_ZDOTDIR:-}"
KISS_USER_ZDOTDIR_EXPORTED=0
if [[ -n "$KISS_USER_ZDOTDIR" ]]; then KISS_USER_ZDOTDIR_EXPORTED=1; fi

# Put the user's ZDOTDIR in place for one of their files.
__kiss_terminal_prepare() {
  if [[ -n "$KISS_USER_ZDOTDIR" ]]; then
    ZDOTDIR="$KISS_USER_ZDOTDIR"
    if (( KISS_USER_ZDOTDIR_EXPORTED )); then export ZDOTDIR; else typeset +x ZDOTDIR; fi
  else
    unset ZDOTDIR
  fi
}

# Note what the user's file did to ZDOTDIR; point zsh back here for
# the next shim (unexported: no child of a later file sees this dir).
__kiss_terminal_capture() {
  KISS_USER_ZDOTDIR="${ZDOTDIR:-}"
  KISS_USER_ZDOTDIR_EXPORTED=0
  if [[ "${(t)ZDOTDIR-}" == *export* ]]; then KISS_USER_ZDOTDIR_EXPORTED=1; fi
  ZDOTDIR="$KISS_TERMINAL_ZDOTDIR"
  typeset +x ZDOTDIR
}

# Put the user's ZDOTDIR back for good and drop the tab's variables.
__kiss_terminal_restore() {
  __kiss_terminal_prepare
  unset KISS_USER_ZDOTDIR KISS_USER_ZDOTDIR_EXPORTED KISS_TERMINAL_ZDOTDIR
  unfunction __kiss_terminal_prepare __kiss_terminal_capture __kiss_terminal_restore
}

__kiss_terminal_prepare
if [[ -r "${ZDOTDIR:-$HOME}/.zshenv" ]]; then . "${ZDOTDIR:-$HOME}/.zshenv"; fi
__kiss_terminal_capture
# No further file is read with `unsetopt rcs`, nor by a shell that is
# neither a login shell nor interactive.
if [[ ! -o rcs || ( ! -o login && ! -o interactive ) ]]; then __kiss_terminal_restore; fi
"""

ZSH_ZPROFILE = r"""# See .zshenv in this directory.
__kiss_terminal_prepare
if [[ -r "${ZDOTDIR:-$HOME}/.zprofile" ]]; then . "${ZDOTDIR:-$HOME}/.zprofile"; fi
__kiss_terminal_capture
if [[ ! -o rcs ]]; then __kiss_terminal_restore; fi
"""

ZSH_ZSHRC = r"""# See .zshenv in this directory.
__kiss_terminal_prepare
if [[ -r "${ZDOTDIR:-$HOME}/.zshrc" ]]; then . "${ZDOTDIR:-$HOME}/.zshrc"; fi
__kiss_terminal_capture
# History that outlives the tab.  A system rc that ran before this
# file (macOS's /etc/zshrc) derives HISTFILE from the ZDOTDIR of the
# moment, which was this directory: move such a file to the user's.
if [[ "${HISTFILE:-}" == "$KISS_TERMINAL_ZDOTDIR/"* ]]; then
  HISTFILE="${KISS_USER_ZDOTDIR:-$HOME}/${HISTFILE#"$KISS_TERMINAL_ZDOTDIR"/}"
fi
# zsh keeps no history file unless told to; then the usual one.
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
if [[ -r "${ZDOTDIR:-$HOME}/.zlogin" ]]; then . "${ZDOTDIR:-$HOME}/.zlogin"; fi
__kiss_terminal_capture
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
    """Write *text* to *path* unless it is there already."""
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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
            user_zdotdir = os.environ.get("ZDOTDIR", "")
            if user_zdotdir:
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
