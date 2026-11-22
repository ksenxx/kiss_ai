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
  agent and its own shell included) or deleting the root, the home or the
  working directory.  A model that meant "stop my server" and typed
  ``kill -9 -1`` otherwise ends the run with nothing to deliver.  It judges
  the words of every simple command bash would run, as
  :mod:`~kiss.agents.sorcar.shell_scan` reads them (:class:`_Guard`).
* :func:`lift_install_timeout` raises the timeout of package installs and
  builds to at least :data:`INSTALL_TIMEOUT_SECONDS`.  A short tool timeout
  leaves them half done (broken dpkg state, partial builds) and the model
  then spends turns repairing and retrying; lifting the deadline in place is
  cheaper than letting the model guess it.
"""

from __future__ import annotations

import collections
import fnmatch
import itertools
import os
import posixpath
import re
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field

from kiss.agents.sorcar.shell_scan import (
    SHELLS,
    TOO_DEEP_VERDICT,
    Group,
    SimpleCommand,
    TooDeepError,
    Word,
    bash_path,
    command_text,
    host_path,
    reads_script,
    receiver,
    shell_script_index,
    simple_commands,
)

#: The patterns of :data:`INSTALL_COMMANDS`, which read the command as text.
#: ``_END`` closes a shell word: whitespace, end of command, a separator, a
#: redirection, a comment or a closing backquote.  ``_WORD`` is one argument
#: of a simple command: it stops at a separator, a redirection or a newline and cannot start a
#: comment, so ``_ARGS_LAZY`` (zero or more arguments) never crosses into the next
#: command or a trailing ``# comment``.  ``_CMD`` is where a command name may
#: start: the beginning or a separator (a backquote too; a ``)``, which
#: closes a ``case`` pattern, a function's ``()`` or a substitution), then
#: the reserved words a command follows (``{``, spelled ``\x7b`` in this
#: format template, ``!``, ``if``, ``then``, ``do``...), optional
#: environment assignments, one wrapper (``sudo``, ``env``, ``nohup``,
#: ``timeout 5``...) with its own arguments (``sudo -n``, ``sudo -u root``,
#: ``env DEBUG=1``; the ``name {`` of a ``function`` or ``coproc``, read as
#: wrappers too), or one shell's ``-c`` with the quote opening its text
#: (``bash -c 'make'``; the shell's options may go on after the ``-c``
#: cluster, ``bash -c -e 'make'``, ``bash -co posix 'make'``, ``bash -c
#: -- 'make'``), and an optional directory prefix.  ``_CMD`` is
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
#: ``npm`` candidate of ``sudo echo npm npm npm …`` rescanned the whole
#: suffix for a subcommand, quadratic in the command length.  The pattern
#: after the prefix reads every word of the simple command, so its first
#: occurrence stands for every later one: one scan of the suffix per
#: command name.  That holds only when ``{commands}`` names the commands
#: the pattern after it matches: a wrapper's option values may be spelled
#: like other commands (``env -u make pip install x``), and stopping at one
#: of them hid the real command, so the installers and the builders each
#: get their own prefix.  The blanks after a separator stop at a newline,
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
#: A line continuation, which bash drops before it reads the command (``pip
#: \<newline>install`` is ``pip install``): a backslash-newline whose backslash
#: no other one escapes.  Group 1 is what stays as written: any other escaped
#: character and, in ``_CONTINUED_CODE``, a comment (a ``#`` that starts a
#: word, to the end of its line), whose last backslash continues nothing: the
#: next line is a command of its own.
_ESCAPED = r"\\[^\n]"
_CONTINUED = re.compile(r"(" + _ESCAPED + r")|\\\n")
_CONTINUED_CODE = re.compile(r"(" + _ESCAPED + r"|(?<![^\s;&|()`])#[^\n]*)|\\\n")
#: What bash's quoting makes characters of a word: an escaped character (group
#: 2, with a ``#`` after it, which then starts no comment) and what a ``$'…'``,
#: a single-quoted or a double-quoted string holds (groups 3 to 5).  Group 1 is
#: a comment, where a quote opens nothing.  ``_SYNTAX`` is what the patterns
#: would read in there as more than a character of a word: an escaped character
#: of a string, a blank, a separator, a ``#``, a quote, a backslash or a ``*``.
#: What may expand in a string stays (a ``$``, a backquote: the patterns do
#: not tell the quotes apart either).
_QUOTING = re.compile(
    r"((?<![^\s;&|()`])#[^\n]*)|\\([^\n]#*)|\$'((?:[^'\\]|\\[\s\S])*)'|'([^']*)'"
    r'|"((?:[^"\\]|\\[\s\S])*)"'
)
_SYNTAX = re.compile(r"\\[\s\S]|[\s;&|<>()#'\"*\\]")

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


def _unquoted(quoting: re.Match[str]) -> str:
    """Return what the quoting *quoting* matched (``_QUOTING``) stands for in its word.

    The characters it holds, each one the patterns would read as more
    than a character of a word (``_SYNTAX``) replaced by a NUL, which is a
    character of a word to every pattern: an escaped ``$`` or backquote
    too, which expands nothing.  A comment stays as written.
    """
    held: str = quoting[quoting.lastindex or 1]
    if quoting.lastindex == 1:
        return held
    if quoting.lastindex == 2:
        held = held.replace("$", "\0").replace("`", "\0")
    return _SYNTAX.sub("\0", held)


def _readings(command: str) -> set[str]:
    """Return the texts a shell may read *command* as: an install in any of them is one.

    The command as written, and with its line continuations removed
    (``_CONTINUED``), everywhere and outside comments: the quotes decide
    which ``#`` starts a comment, the patterns do not follow them, and
    what the command quotes may be the text of another shell (``bash -c
    '…'``).  Then each of these with its quoting removed
    (:func:`_unquoted`), as bash removes it from the words of the
    command it runs: ``X="a b" pip install "x # y"`` is an assignment,
    then ``pip install`` with one operand.
    """
    texts = {command}
    if "\\\n" in command:
        texts |= {_CONTINUED.sub(r"\1", command), _CONTINUED_CODE.sub(r"\1", command)}
    return texts | {_QUOTING.sub(_unquoted, text) for text in texts}


def _inherited_cdpath() -> str:
    """Return the ``CDPATH`` a shell that this process starts inherits (``""`` when unset)."""
    return os.environ.get("CDPATH", "")


def _slash_path(path: str) -> str:
    """Return *path* with shell separators, preserving any Windows drive prefix.

    Only Windows paths are rewritten: on POSIX a backslash is a legal
    file-name character, not a separator.
    """
    return path.replace("\\", "/") if os.name == "nt" else path


#: The programs the guard judges (:meth:`_Guard.program`): those that end the task by their
#: arguments, and those that run a text of commands.
_JUDGED = frozenset({"rm", "kill", "pkill", "killall", "killall5", "eval"}) | SHELLS
#: What a command must mention, its quotes and backslashes aside, to be read at all: a
#: program the guard judges, an expansion, a file name pattern (``/bin/r?``), or a shell
#: (a word that ends in ``sh``), which may run a text the command puts together from parts
#: that mention nothing (``printf 'r%s -rf ~' m | sh``); or spell by a brace expansion
#: (``{r,}m``, ``{r..r}m``).
#: :data:`_MORE_SHELLS` are read as ``sh``; :data:`_NAMES` are the judged programs a
#: pattern in a command word may name (:func:`_named`).
_MORE_SHELLS = frozenset({"rbash", "rzsh", "rksh", "ash", "mksh", "pdksh", "yash", "posh"})
_NAMES = ("rm", "kill", "pkill", "killall", "killall5", *sorted(SHELLS | _MORE_SHELLS))
#: The commands whose own field separator reaches nothing else (``IFS=, read -r a b``).  Any
#: other assignment to ``IFS``, and a name reference (``declare -n``), make bash build
#: words in a way the guard does not model (:meth:`_Guard.shades`).
_SPLITTERS = frozenset({"read", "mapfile", "readarray"})
#: ``exec`` with an input redirection: what a later ``read`` gets may be text the call wrote.
#: A startup file of a shell that is a process substitution (``BASH_ENV=<(printf ..) bash
#: -c :``; a stream by its name, ``BASH_ENV=/dev/stdin``, is refused where the shell runs,
#: :meth:`_Guard.program`), and a ``source`` of what the call prints: texts the guard does
#: not read.  A command word that may name the shell (:func:`_reader`).
_EXEC_INPUT = re.compile(r"\bexec\b[^;&|\n]*<")
_STARTUP = re.compile(
    r"\bBASH_ENV=[^\s;&|]*<\(|\bBASH_FUNC_\w+%%=|\bxargs\b[^;&|\n]*(?:-a\s*|--arg-file[= ]\s*)<\("
    r"|(?:\bsource|(?<![\w./-])\.)\s+<\(\s*(?:echo|printf|cat)\b")
_SHELL_WORD = re.compile(r"\$\(|`|\$\{?(?:SHELL|BASH|0)\b|\$\{\w+:?[-=+]")
#: What makes a value more than the word the guard reads, once its shell has pasted it into
#: the lines of a here-document that another shell runs.
_PASTED = re.compile(r"[;&|<>()'\"\\`$\n~{]")
#: Where an extended pattern starts (``rm -rf !(keep)``, with ``extglob`` set by any means:
#: the scanner ends the word at its parenthesis, :meth:`_Guard.patterned`), and a variable
#: an ``unset`` may leave as it is.
_EXTENDED = re.compile(r"[@!?*+]\(")
_READONLY = re.compile(r"\breadonly\b|\b(?:declare|local|typeset)(?:\s+-\w+)*\s+-\w*r")
#: The builtins before which an assignment stays in POSIX mode (``sh -c 'D=$HOME :'``).
_SPECIAL = frozenset({
    ":", ".", "source", "break", "continue", "eval", "exec", "exit", "return", "set",
    "shift", "times", "trap", "unset"})
#: ``${NAME:=`` wherever it nests: one :data:`_DEFAULTED` does not read assigns any value.
_DEFAULT_START = re.compile(r"\$\{(!?)([A-Za-z_]\w*)(?:\[[^\]]*\])?:?=")
#: Where a command substitution starts in a word, and one in backticks.
_SUBSTITUTED = re.compile(r"\$\((?!\()")
_BACKTICKED = re.compile(r"`([^`]*)`")
_MENTIONED = re.compile(r"rm|kill|[$`*?\[]|sh(?![\w.-])|\bsource\b|(?<![\w./-])\.\s")
_QUOTES = re.compile(r"\\\n|[\\'\"]")
_SPELLED = re.compile(r"\{[^\s{}]*(?:,|\.\.)")
#: Commands that run their operands as a command.  The scanner drops the ones it knows
#: (``sudo``, ``env``...) with their options; an option it misreads (``sudo --user root rm``)
#: or a wrapper it does not know (``setsid``) leaves the program among the words, so the
#: guard judges every guarded program or shell named after one of these.
_RUNNERS = frozenset({
    "sudo", "doas", "env", "nice", "nohup", "timeout", "stdbuf", "xargs", "exec", "command",
    "builtin", "time", "setsid", "strace", "ltrace", "flock", "ionice", "chrt", "taskset",
    "busybox", "toybox", "su", "runuser", "chroot", "unshare", "nsenter", "fakeroot", "watch",
    "script", "unbuffer", "numactl", "systemd-run", "firejail", "caffeinate", "parallel",
    "find", "setpriv", "prlimit", "pkexec", "gosu", "xvfb-run", "dumb-init", "tini",
    "valgrind", "eatmydata", "proxychains", "torsocks", "uv", "uvx", "poetry", "pipx",
    "pipenv", "npx", "bunx", "tmux", "screen",
})
#: The runners that take the command as one word of shell text (``su -c 'text'``).
_TEXT_RUNNERS = frozenset({
    "su", "runuser", "watch", "script", "parallel", "flock", "tmux", "screen"})
#: A runner's options that move the program to another directory (``env -C dir``).
_CHDIR_OPTIONS = ("-C", "-D", "--chdir", "--cwd")
#: Builtins whose operands name variables they assign from what they read.
_ASSIGNERS = frozenset({"read", "mapfile", "readarray", "getopts", "printf"})
#: Builtins whose ``NAME=value`` operands are assignments (``declare D=$HOME``), and those
#: of them that make the variable a function's own.
_DECLARERS = frozenset({"export", "declare", "local", "typeset", "readonly"})
_LOCALS = frozenset({"declare", "local", "typeset"})
#: The commands that keep their words in the shell, where bash may expand them later
#: (:meth:`_Guard.quoted`).
_KEEPERS = _DECLARERS | _ASSIGNERS | _SPECIAL | {"let", "test", "[", "[[", "alias"}
#: The programs that hand on what they read as it is, and the commands the guard gives a
#: meaning of its own: a function of the call by one of these names runs in its place
#: (:meth:`_Guard.shades`).
_COPIERS = frozenset({"cat", "tee"})
_MODELLED = _KEEPERS | _COPIERS | {"cd", "pushd", "popd", "echo", "pwd", "command", "builtin"}
#: The first word of a command that assigns an element by a subscript the scanner does not
#: read (``A[${#A[@]}]=x``).
_ELEMENT = re.compile(r"([A-Za-z_]\w*)\[.*\]\+?=", re.DOTALL)
#: Where bash evaluates arithmetic in a word: ``$((``; the subscript and the offsets of an
#: ``expansion`` (``${NAME[i]:offset}``); the subscript of an ``element`` that is assigned
#: (``A[i]=x``, ``A=([i]=x)``; any other ``name[..]`` is a pattern).  Besides: an arithmetic
#: command and a compound assignment (:data:`_COMMANDED`), the operators of ``[[`` that
#: compare numbers, the subscript of a name a builtin is given (:data:`_NAMERS`).  An
#: expansion that reads the variable its variable names (``${!NAME}``) or expands its value
#: as a prompt (``${NAME@P}``) evaluates that value too.  A name an expression assigns
#: (``P=-1``, ``P--``, ``++P``: :data:`_COUNT`) holds a number nobody knows; a name it reads
#: (:data:`_NAME`; a positional parameter, :data:`_POSITIONAL`) is an expression too
#: (:meth:`_Guard.counts`).
_ARITHMETIC = re.compile(
    r"\$\(\(|(?P<expansion>\$\{[!#]?(?:\w+|[@*]))|(?<![\w$\]])(?P<element>(?:[A-Za-z_]\w*)?\[)")
#: What the scanner shows no word of: an arithmetic command where a command starts
#: (``(( .. ))``, the header of a ``for``) and the words of a compound assignment
#: (``NAME=( .. )``); a here-document its shell expands (the delimiter is not quoted).
_COMMANDED = re.compile(
    r"(?:^|[;&|(){\n!]|\b(?:then|do|else|elif|if|while|until|time|for|coproc))[ \t]*(\(\()"
    r"|(?<![\w$])[A-Za-z_]\w*(?:\[[^\]]*\])?\+?=(\()")
_EXPANDED_DOCUMENT = re.compile(
    r"(?<!<)<<-?[ \t]*([A-Za-z_][\w.+-]*)[^\n]*\n(.*?)(?:^\t*\1$|\Z)", re.DOTALL | re.MULTILINE)
#: A word no call holds: put at a place of a text, it tells whether the scanner reads the
#: words of a command there (:meth:`_Guard.stands`).  Past :data:`_MAX_PROBES` places of a
#: call, each is taken for one.
_PROBE = "kiss_guard_probe_word"
_MAX_PROBES = 32
#: The variables bash expands as prompts: a substitution one holds runs.  Those it keeps as
#: integers: what the call assigns one is an expression.
_PROMPTS = frozenset({"PS0", "PS1", "PS2", "PS4", "PROMPT_COMMAND"})
_INTEGERS = frozenset({"OPTIND", "RANDOM", "SRANDOM", "HISTCMD", "SECONDS", "LINENO"})
_COMPARES = frozenset({"-eq", "-ne", "-lt", "-le", "-gt", "-ge"})
_LENGTH = re.compile(r"\$\{#\w+(?:\[[@*]\])?\}")
#: The builtins that read an operand as the name of a variable (:func:`_names`): bash
#: evaluates its subscript (``read 'A[i]'``, ``printf -v 'A[i]'``, ``[[ -v A[i] ]]``).
_NAMERS = _ASSIGNERS | _DECLARERS | {"unset", "wait", "test", "[", "[["}
_COUNT = re.compile(
    r"(?:\+\+|--)\s*\$?\{?([A-Za-z_]\w*)"
    r"|([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)?(?:\+\+|--|(?:[-+*/%&|^]|<<|>>)?=(?!=))")
_NAME = re.compile(r"[A-Za-z_]\w*")
_POSITIONAL = re.compile(r"\$\{?[\d@*]")
#: A word of an ``echo`` or a ``printf`` that prints nothing of its own: an option, a format
#: of ``%s`` and newlines.
_BLANK = re.compile(r"(?:%s|\\[nt]|\s|-[neE]+|--)*")
#: A word ``env`` is given that is its ``-S`` option (the rest of it, or the next word, is a
#: command line), and one that is a variable a shell runs at its start: a function it
#: imports, or a name nobody knows.
_SPLITS = re.compile(r"-[A-Za-z]*S(.*)|--split-string(?:=(.*))?", re.DOTALL)
_STARTED = re.compile(r"BASH_FUNC_|[^=]*[\x00\x04-\x07][^=]*=")
#: The name of a function a text defines (``function name``, ``name()``).
_DEFINED = re.compile(
    r"\bfunction\s+([^\s(){};]+)"
    r"|(?<![^\s(){};&|<>$`'\"=])([^\s(){};&|<>$`'\"=]+)\s*\(\s*\)")
#: An assignment word: its name, its subscript, the ``+`` of ``+=`` and its value.
_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)(\[[^\]]*\])?(\+?)=(.*)", re.DOTALL)
#: What follows the ``NAME=(`` of an array assignment, to its ``)``: the scanner reads no
#: word of it.  One more pair of parentheses may nest in it (a substitution).
_ARRAY_PART = r"""[^()'"\\]|\\[\s\S]|'[^']*'|"(?:[^"\\]|\\[\s\S])*\""""
_ARRAY_BODY = re.compile(
    r"((?:" + _ARRAY_PART + r"|\((?:" + _ARRAY_PART + r"|\([^()]*\))*\))*)\)")
#: Constructs that leave no simple command (an arithmetic command, a bare ``time``): a ``&&``
#: list may pass through one unseen, so a command that holds one outside its quotes and
#: here-documents proves no ``cd``.  The header of an arithmetic ``for`` is no command: its
#: loop is a group the scanner shows (:data:`_FOR_HEADER`).
_UNSEEN = re.compile(r"(?<!\$)\(\(|\btime\s*(?:$|[;&|)\n])")
_FOR_HEADER = re.compile(r"\bfor\s*\(\(")
#: What ends a branch of an ``if`` or a ``case`` (the scanner shows the commands of all
#: branches as one group), and the commands the guard writes before each one to tell the
#: branches apart (:meth:`_Guard.refuses`); ``||`` before an ``if``, which then may not run.
_BRANCH_END = re.compile(r"\b(else|elif)\b|;;&?|;&")
_ELSE, _BRANCH = "__kiss_else__", "__kiss_branch__"
_OR_IF = re.compile(r"\|\|(?:\s|\\\n|!|time\b|-p\b|#[^\n]*)*if\b")
#: The runners that hand what they read on to the program they run, and an option that
#: names the text they replace with each line of it (``-I{}``, ``-0I %``, ``--replace=X``).
_FED_RUNNERS = frozenset({"xargs", "parallel"})
_REPLACED = re.compile(r"-[0-9A-Za-z]*?[Ii](.*)|--replace(?:=(.*))?", re.DOTALL)
#: A substitution that starts with ``echo``, ``printf`` or a program that prints its operand
#: or the directory of it, and a process substitution that starts with one of the first
#: two: text the Bash call itself writes (:meth:`_Guard.printed`).
_PRINTED = re.compile(r"(\$\(|`)\s*(?:echo|printf|dirname|basename|realpath|readlink)\b")
_PRINTED_INPUT = re.compile(r"<\(\s*(?:echo|printf)\b")
#: ``||`` before a brace group.  The scanner shows such a group like the body of a function's
#: definition (a group that may not run); in a text without one, that group is a body.
_OR_GROUP = re.compile(r"\|\|(?:\s|\\\n|!|time\b|-p\b|#[^\n]*)*\{")
#: The kinds of a group that is a substitution: its commands are part of the command that
#: holds it, and its input is not the one of the group around it.  With a subshell, the
#: groups that run in a shell of their own.
_SUBSTITUTIONS = ("$(", "`")
_SUBSHELLS = ("(", *_SUBSTITUTIONS)
#: The kinds of a group that is a loop: its commands run again where the turn before ended.
_LOOPS = frozenset({"for", "while", "until", "select"})
#: ``pkill`` / ``killall`` options that take the next word, and those that select by user.
_PKILL_VALUED = frozenset({
    "-u", "-U", "-g", "-G", "-P", "-s", "-t", "-F", "--signal", "--ns", "--nslist", "--euid",
    "--uid", "--group", "--pgroup", "--parent", "--session", "--terminal", "--pidfile",
    "--cgroup",
})
_PKILL_USERS = ("-u", "-U", "--euid", "--uid")
#: ``pkill``'s inverse (``-v`` alone or in a cluster of letters): everything but the pattern.
_PKILL_INVERSE = re.compile(r"-[a-z]*v[a-z]*|--inverse")
#: What of a ``pkill`` pattern demands nothing of a name: a bracket expression, an escape.
_ANY_CHARACTER = re.compile(r"\[[^\]]*\]|\\.")
#: ``-1`` as ``kill`` reads a pid: with leading zeros too, and blanks around it.
_EVERY_PID = re.compile(r"\s*-0*1\s*")
#: What stands in an expanded word for the directory the command runs in, the home
#: directory, a value that is not empty but unknown (what a program prints or reads: it
#: names nothing protected), an edited directory (``${PWD##*/}``, ``${HOME%/}``: the
#: directory, its parent or its own name) and a value the guard cannot bound (a command
#: that deletes or signals by one is refused).  :data:`_COMPUTED` and :data:`_COUNTED`
#: stand in a word for a command substitution and for arithmetic (:func:`_marked`).
#: :data:`_FOUND` is what a program printed that the command told a protected directory
#: (:func:`_tells`): unknown like :data:`_SOME`, but a path that may be such a directory.
_HERE, _HOME, _SOME, _EDITED, _ANY = "\x01", "\x02", "\x00", "\x03", "\x04"
_COMPUTED, _COUNTED, _FOUND = "\x05", "\x06", "\x07"
#: Unknown values next to each other (``"$pids $!"``): one unknown value.
_RUN = re.compile(r"\x00(?:\s*\x00)+")
#: What arithmetic makes that is not one number the guard reads: a number nobody knows,
#: which may be the pid that stands for every process, or the digit of it.
_A_NUMBER = (_SOME, "-1", "1")
#: An arithmetic expansion that adds numbers which are not negative: small constants,
#: special parameters and variables that hold nothing else (:data:`_ADDEND`).  The guard
#: reads the number it makes (:meth:`_Guard.summed`).
_TERM = r"(?:\$?[A-Za-z_]\w*|\$\{[A-Za-z_]\w*\}|\$[0-9#?!$]|0|[1-9]\d{0,8})"
_SUM = re.compile(r"\$\(\(\s*" + _TERM + r"(?:\s*\+\s*" + _TERM + r")*\s*\)\)")
_ADDEND = re.compile(r"\d{0,9}|\x00")
#: The values a variable may hold (``None``: it is unset): of one nobody assigned, of one a
#: program set, of one that is empty, of one the guard cannot bound.
_UNSET: frozenset[str | None] = frozenset({None})
_UNKNOWN: frozenset[str | None] = frozenset({_SOME})
_EMPTY: frozenset[str | None] = frozenset({""})
_ANYTHING: frozenset[str | None] = frozenset({_ANY})
#: What a number the shell itself counts may be (``$?``, ``$#``, ``${#NAME}``, a variable of
#: :data:`_COUNTERS` the call did not assign): not negative, and 0 or 1 among others, so
#: ``-$?`` may be the pid that stands for every process and ``${?:1}`` nothing at all.
_COUNTS: frozenset[str | None] = frozenset({_SOME, "0", "1"})
_COUNTERS = frozenset({
    "LINENO", "OPTIND", "OPTERR", "BASH_SUBSHELL", "SHLVL", "RANDOM", "SRANDOM", "SECONDS",
    "HISTCMD", "FUNCNEST", "COLUMNS", "LINES", "UID", "EUID", "PPID", "EPOCHSECONDS"})
#: A value longer than this is any value; a slice of a value the call wrote that is longer
#: than :data:`_MAX_SLICED` is any text (:func:`_parts`).
_MAX_VALUE = 4096
_MAX_SLICED = 8
#: What makes the output of a substitution depend on what it sets itself: a loop or a
#: ``case`` (:data:`_LOOPED`), a builtin that assigns (:data:`_SETTERS`).
_LOOPED = re.compile(r"(?:^|[;&|(\n])\s*(?:for|select|case|while|until)\b")
#: A file that is a descriptor of the shell: it may carry text the Bash call wrote.
_DESCRIPTOR = re.compile(r"/dev/(?:fd/|std)|/proc/[^/]+/fd/")
#: Programs that never read their standard input.
_DEAF = frozenset({
    "basename", "dirname", "pwd", "cd", "date", "mktemp", "echo", "printf", "which", "nproc",
    "uname", "id", "whoami", "hostname", "realpath", "readlink", "true", "false", ":"})
_SETTERS = _DECLARERS | _SPLITTERS | {
    "set", "shift", "getopts", "eval", "source", ".", "let", "unset"}
_QUOTE_MARKS = re.compile(r"""[\\'"]""")
#: The expansions of a word, read as a fresh shell expands them (:meth:`_Guard.expansion`):
#: ``$(pwd)``; a parameter, plain or in braces with its ``prefix``, ``subscript``, operator
#: and the operator's word (one more ``${...}`` may nest in it); arithmetic; a ``$'...'``
#: string; any other substitution (:func:`_marked` puts :data:`_COMPUTED` and
#: :data:`_COUNTED` where the word holds one).
#: :data:`_OPERAND` is the operator's word: each of its characters is read in one way only,
#: so an unclosed expansion costs one pass.  :data:`_UNREAD` is what starts an expansion
#: none of these read.
_OPERAND = r"(?:[^{}$]|\$(?!\{)|\$?\{[^{}]*\})*"
_PWD = re.compile(r"\$\(\s*pwd[^()`]*\)")
_EXPANSION = re.compile(
    r"(?P<here>\$\(\s*pwd[^()`]*\)|`\s*pwd[^`]*`)"
    r"|\$(?P<name>[A-Za-z_]\w*|[0-9@*#?!$-])"
    r"|\$\{(?P<prefix>[!#]?)(?P<braced>[A-Za-z_]\w*|[0-9]+|[@*#?!$-])(?P<subscript>\[[^\]{}]*\])?"
    r"(?P<operator>:?[-=?+]|[%#/^,:@])?(?P<word>" + _OPERAND + r")\}"
    r"|(?P<some>\x06)"
    r"|\$'(?P<ansi>(?:[^'\\]|\\.)*)'"
    r"|\x05|`[^`]*`|\$\{[^{}]*\}",
    re.DOTALL,
)
_UNREAD = re.compile(r"\$[({]|`")
#: What the commands of a substitution mention when their output may name a protected
#: directory (``$(cd ~ && pwd)``, ``$(printenv HOME)``); ``mktemp`` alone makes a new path.
_TOLD = re.compile(r"\bHOME\b|PWD\b|~|\b(?:pwd|dirs|pushd|popd|cd)\b")
_MAKES = re.compile(r"\s*mktemp\b[^;&|\n`()<>]*")
#: A ``$'...'`` string, a word that is all the positional parameters (``"$@"``), and what
#: makes several words of one: all the parameters or all the elements of an array.
_ANSI = re.compile(r"\$'((?:[^'\\]|\\.)*)'", re.DOTALL)
_ALL_ARGUMENTS = re.compile(r"\$(?:[@*]|\{[@*]\})")
_SEVERAL = re.compile(r"\$[@*]|\$\{[@*]|\[[@*]\]")
#: What bash splits an unquoted value at, or matches file names with.
_SPLIT = re.compile(r"[\s*?\[]")
#: An expansion that assigns its word to a variable without a value (``${D:=word}``).
_DEFAULTED = re.compile(r"\$\{([A-Za-z_]\w*):?=(" + _OPERAND + r")\}")
#: A backslash escape of a ``$'...'`` string: a letter, a number (octal, hex, Unicode), a
#: control character (``\cJ``).
_ESCAPE = re.compile(
    r"\\(?:(?P<letter>[abeEfnrtv\\'\"?])|(?P<octal>[0-7]{1,3})|x(?P<hex>[0-9a-fA-F]{1,2})"
    r"|u(?P<short>[0-9a-fA-F]{1,4})|U(?P<long>[0-9a-fA-F]{1,8})|c(?P<control>[\s\S]))")
_LETTERS = {"a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n", "r": "\r",
            "t": "\t", "v": "\v"}
#: An arithmetic expression that is one number (``-1``; a leading zero would make it octal).
_NUMBER = re.compile(r"\s*(-?)\s*(0|[1-9]\d{0,8})\s*")
#: A tilde-prefix (``~``, ``~name``, ``~+``, ``~-``); and one where bash expands it in the
#: value of an assignment: at its start and after each ``:`` (``PATH=$PATH:~/bin``).
_TILDE = re.compile(r"~(?:[+-]?\d+|\+|[\w.-]*)", re.ASCII)
_ASSIGNED_TILDE = re.compile(r"(?:^|(?<=:))~(?:[+-]?\d+|\+|[\w.-]*)(?=[/:]|$)", re.ASCII)
#: A tilde-prefix that names a directory of the stack ``pushd`` keeps (``~1``, ``~-0``).
_STACKED = re.compile(r"~[+-]?\d+")
#: A character class in a bracket expression (``[[:alpha:]]``, ``[[.w.]]``, ``[[=w=]]``).
_CLASS = re.compile(r"\[[:.=]")
#: A locale string (``$"text"`` is ``"text"``).
_LOCALE = re.compile(r'(?<![\\$])\$"')
#: The variables bash itself sets, to what the guard does not follow.
_BASH_SET = frozenset({
    "BASH_REMATCH", "OPTARG", "DIRSTACK", "BASH_ARGV", "BASH_ARGV0", "BASH_COMMAND",
    "BASH_EXECUTION_STRING", "BASH_SOURCE", "FUNCNAME", "COPROC", "READLINE_LINE"})
#: A value that holds a substitution naming a judged program: bash may expand it later
#: (arithmetic, a prompt, ``${X@P}``), whatever built it (``X=$(printf 'a[$(r%sm ..)]')``).
_ARMED = re.compile(
    r"(?:\$\(|`)[^)`]*\b(?:rm|kill|pkill|killall|killall5)\b"
    r"|(?:\$\(|`)\s*(?:\S*/)?(?:(?:ba|da|z|k)?sh|eval)\b")
#: A ``printf`` format the guard reads exactly: text, ``%s``, ``%%`` and the escapes of a
#: newline, a tab and a backslash; its parts around each ``%s``.
_EXACT_FORMAT = re.compile(r"(?:[^%\\]|%[s%]|\\[nt\\])*")
_FORMAT_PART = re.compile(r"(%s)|%%|\\[nt\\]")
_FORMAT_TEXT = {"%%": "%", "\\n": "\n", "\\t": "\t", "\\\\": "\\"}
#: An ``echo`` option, and the pattern of a ``${NAME%pattern}`` the guard matches itself.
_ECHO_OPTION = re.compile(r"-[neE]+")
_PLAIN_PATTERN = re.compile(r"[^$`\\(){}]*")
#: An escaped character of a ``$'...'`` string.
_ANSI_ESCAPE = re.compile(r"\\([\s\S])")
#: A brace expansion: its parts (``{a,b}``), or the ends of its sequence (``{1..3}``); and a
#: sequence the guard does not expand.
_BRACES = re.compile(
    r"(?<!\$)\{(?:([^{}]*,[^{}]*)|(-?\d{1,3}|[A-Za-z])\.\.(-?\d{1,3}|[A-Za-z]))\}")
_SEQUENCE = re.compile(r"(?<!\$)\{[^{}\s,]*\.\.[^{}\s,]*\}")
_GLOB = re.compile(r"[*?\[]")
#: A last path component that matches every entry of its directory (``*``, ``.*``, ``?*``).
_EVERY = re.compile(r"[*?.\[\]!^]*\*[*?.\[\]!^]*")
#: A here-document operator with its unquoted delimiter.
_HEREDOC = re.compile(r"(?<!<)<<-?[ \t]*[^\s;&|<>()]+")
#: How many texts within texts (``bash -c "bash -c ..."``) the guard rescans, how many
#: words it reads for one word (the alternatives of a brace expansion, the values of its
#: variables) and how many values for one variable, how many paths it asks the filesystem
#: about for one command, how many pipes and groups it follows an output through, how many
#: times it reads a command whose loops read what later commands assign (from the
#: :data:`_WIDENED` th time on a variable that still gets new values holds any value), and
#: how long a value it edits itself.
_MAX_TEXTS = 4
_MAX_ALTERNATIVES = 64
_MAX_RESOLVED = 16
_MAX_PIPES = 8
_MAX_PASSES = 10
_WIDENED = 4
_MAX_EDITED = 256


def _mentions(text: str) -> bool:
    """Return whether *text* mentions or spells what the guard reads a command for
    (:data:`_MENTIONED`, :data:`_SPELLED`)."""
    return (_MENTIONED.search(_QUOTES.sub("", text)) is not None
            or _SPELLED.search(text) is not None)


def _branch_mark(end: re.Match[str]) -> str:
    """Return the branch end *end* (:data:`_BRANCH_END`) after a command that marks it."""
    if end[1]:
        return f"{_ELSE if end[1] == 'else' else _BRANCH}; {end[0]}"
    return f"; {_BRANCH}; {end[0]}"


def _ansi_escape(escape: re.Match[str]) -> str:
    """Return the *escape* of a ``$'...'`` string (:data:`_ANSI_ESCAPE`), a quote as ``\\x27``."""
    return "\\x27" if escape[1] == "'" else str(escape[0])


def _requoted(quoting: re.Match[str]) -> str:
    """Return the *quoting* (:data:`_QUOTING`) as written, a ``$'...'`` without ``\\'`` in it.

    The scanner ends such a string at its first escaped quote and reads the rest of the
    line as part of the word or as a comment (``echo $'it\\'s'; rm -rf /``); spelled
    ``\\x27`` the quote is the same character and ends nothing.
    """
    if quoting.lastindex != 3:
        return str(quoting[0])
    return "$'" + _ANSI_ESCAPE.sub(_ansi_escape, quoting[3]) + "'"


def _ansi(string: re.Match[str]) -> str:
    """Return the characters the ``$'...'`` *string* stands for (its last group holds them).

    Up to its first NUL, where bash ends the string.
    """
    return _ESCAPE.sub(_escape, str(string[string.lastindex or 0])).partition("\0")[0]


def _escape(escape: re.Match[str]) -> str:
    """Return the character the backslash *escape* (:data:`_ESCAPE`) stands for."""
    if escape["letter"]:
        return _LETTERS.get(escape["letter"], escape["letter"])
    if escape["control"]:
        return chr(ord(escape["control"]) & 0x1F)
    if escape["octal"]:
        return chr(int(escape["octal"], 8) & 0xFF)
    number = int(escape["hex"] or escape["short"] or escape["long"], 16)
    return chr(number) if number < 0x110000 else ""


def _decoded(text: str) -> str:
    """Return the word *text* as bash hands it on as shell text: its ``$'...'`` decoded."""
    return _ANSI.sub(_ansi, text)


def _braced(braces: re.Match[str]) -> list[str]:
    """Return the words the brace expansion *braces* (:data:`_BRACES`) stands for.

    Its parts, or the numbers or the letters of its sequence; itself where bash expands
    nothing (a number to a letter).
    """
    if braces[1] is not None:
        return str(braces[1]).split(",")
    first, last = str(braces[2]), str(braces[3])
    numeric = first.lstrip("-").isdigit()
    if numeric != last.lstrip("-").isdigit():
        return [str(braces[0])]
    start, end = (int(first), int(last)) if numeric else (ord(first), ord(last))
    step = 1 if end >= start else -1
    return [str(at) if numeric else chr(at) for at in range(start, end + step, step)]


def _alternatives(text: str) -> list[str]:
    """Return the words bash's brace expansion makes of *text* (``/tmp/{a,b}``, ``x{1..3}``).

    More than :data:`_MAX_ALTERNATIVES` of them when the guard cannot read them all: the
    braces left stay as written.
    """
    words = [text]
    while len(words) <= _MAX_ALTERNATIVES:
        expanded: list[str] = []
        for word in words:
            braces = _BRACES.search(word)
            if braces is None:
                expanded.append(word)
            else:
                expanded.extend(
                    word[:braces.start()] + part + word[braces.end():]
                    for part in _braced(braces))
        if expanded == words:
            break
        words = expanded
    return words


def _closing(text: str, start: int, opener: str = "(", closer: str = ")") -> int:
    """Return the index past the ``)`` that closes the substitution whose text starts at *start*.

    ``-1`` for one that is not closed.  A parenthesis in quotes or after a backslash
    closes nothing.  Another *opener* and *closer* tell where a subscript or a ``${...}`` ends.
    """
    depth, quote, index = 1, "", start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 1
        elif quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if not depth:
                return index + 1
        index += 1
    return -1


def _modern(text: str) -> str:
    """Return *text* with each old arithmetic expansion (``$[-1]``) spelled ``$((-1))``."""
    start = text.find("$[")
    while start >= 0:
        end = _closing(text, start + 2, "[", "]")
        if end > 0:  # else in a string that ends before it closes: the next one may close
            text = f"{text[:start]}$(({text[start + 2:end - 1]})){text[end:]}"
        start = text.find("$[", start + 2)
    return text


def _reckoned(text: str, assigns: bool = True) -> list[str]:
    """Return the parts of the word *text* bash may evaluate as arithmetic (:data:`_ARITHMETIC`).

    Each to where its bracket closes (to the end of *text* when nothing closes it), with
    the ones that nest in it.  No element of a text that *assigns* nothing (a here-document).
    """
    regions: list[str] = []
    covered = 0
    for start in _ARITHMETIC.finditer(text):
        if start.start() < covered:
            continue
        position = start.end()
        if start["element"]:
            end = _closing(text, position, "[", "]")
            if assigns and end > 0 and text.startswith(("=", "+="), end) and (
                    not text.startswith("==", end)):
                covered = end
                regions.append(text[position:end])
            continue
        if not start["expansion"]:
            end = _closing(text, start.start() + 2)
            # to the end of the word when the guard cannot tell where it closes (no ``))``)
            covered = end if end > 0 and text[end - 2:end] == "))" else len(text)
            regions.append(text[position:covered])
            continue
        end = _closing(text, start.start() + 2, "{", "}")
        end = end if end > 0 else len(text)
        name = start["expansion"][2:]
        if name.startswith("!"):
            regions.append(name[1:])  # the name it holds, with the subscript of it
        if text.startswith("[", position):
            close = _closing(text, position + 1, "[", "]")
            close = close if 0 < close <= end else end
            regions.append(text[position + 1:close])
            covered = position = close
        if text.startswith("@P", position):
            regions.append(name.lstrip("!#"))
        elif text.startswith(":", position) and text[position + 1:position + 2] not in "-=+?":
            regions.append(text[position + 1:end])
            covered = end
    return regions


def _evaluated(commands: list[SimpleCommand]) -> list[str]:
    """Return the arithmetic bash evaluates in the words of the *commands* (:func:`_reckoned`):
    in each word that is not literal, and in what a command or a group is redirected to."""
    nodes: dict[int, SimpleCommand | Group] = {}
    for command in commands:
        node: SimpleCommand | Group | None = command
        while node is not None and id(node) not in nodes:
            nodes[id(node)] = node
            node = node.group
    regions: list[str] = []
    for node in nodes.values():
        words = [word for _, _, word in node.redirections]
        for word in [*words, *node.words] if isinstance(node, SimpleCommand) else words:
            if not word.literal or word.assignment:
                regions += _reckoned(word.text)
    return regions


def _names(argv: list[Word]) -> list[Word]:
    """Return the operands the builtin *argv* reads as names of variables (:data:`_NAMERS`).

    The word after ``-v`` (``-p`` of ``wait``) for one that names a variable by an option,
    else each word that is no option.
    """
    name = argv[0].text
    if name in _DECLARERS or name in _SPLITTERS or name in ("getopts", "unset"):
        return [word for word in argv[1:] if not word.text.startswith(("-", "+"))]
    option = "-p" if name == "wait" else "-v"
    return [word for before, word in zip(argv, argv[1:]) if before.text == option] + [
        Word(word.text[2:], word.literal) for word in argv[1:]
        if word.text.startswith(option) and len(word.text) > 2]


def _parts(value: str) -> set[str]:
    """Return what a slice of *value* (``${NAME:offset:length}``) may be.

    Nothing or the value for what a program found; any part of a short value the call
    wrote; any text for a longer one.
    """
    if not value or _SOME in value:
        return {value, ""}
    if len(value) > _MAX_SLICED or not value.isprintable():
        return {_ANY}
    return {value[start:end] for start in range(len(value) + 1)
            for end in range(start, len(value) + 1)}


def _defined(text: str) -> list[str]:
    """Return the names of the functions *text* may define (:data:`_DEFINED`)."""
    return [first or second for first, second in _DEFINED.findall(text)]


def _coded(find: Callable[[str], list[str]], texts: list[str],
           commands: list[SimpleCommand]) -> list[str]:
    """Return what *find* finds in the *texts* bash reads as code.

    Less what it finds in the here-documents of *commands* that no shell runs here: data,
    counted out one by one (the same words may stand in the code too).
    """
    found = collections.Counter(item for text in texts for item in find(text))
    for command in commands:
        if command.heredoc_body and command.script is None:
            found -= collections.Counter(find(command.heredoc_body))
    return list(found)


def _held(text: str) -> list[str]:
    """Return the commands of each command substitution the word *text* holds."""
    held: list[str] = _BACKTICKED.findall(text)
    for start in _SUBSTITUTED.finditer(text):
        end = _closing(text, start.end())
        if end > 0:
            held.append(text[start.end():end - 1])
    return held


def _tells(commands: str) -> bool:
    """Return whether what the *commands* of a substitution print may name a protected
    directory: they mention one (:data:`_TOLD`) and are not one ``mktemp``."""
    # without the quotes that may split a name (``p""wd``)
    return _TOLD.search(_QUOTE_MARKS.sub("", commands)) is not None and (
        _MAKES.fullmatch(commands) is None)


def _marked(text: str, telling: bool = True) -> str:
    """Return the word *text* with a mark where it holds a substitution or arithmetic.

    :data:`_COMPUTED` for a command substitution, whatever nests in it, or (*telling*)
    :data:`_FOUND` for one that may print a protected directory (:func:`_tells`); for
    arithmetic the number it spells, else :data:`_COUNTED`.  ``$(pwd)`` stays, and one
    that is not closed, which :data:`_UNREAD` then finds.
    """
    start = text.find("$(")
    made: list[str] = []
    position = 0
    while start >= 0:
        end = start + 2 if _PWD.match(text, start) else _closing(text, start + 2)
        if end < 0:
            break
        if end > start + 2:
            number = _NUMBER.fullmatch(text, start + 3, end - 2) if text.startswith(
                "$((", start) and text.endswith("))", start, end) else None
            mark = _COUNTED if text.startswith("$((", start) else (
                _FOUND if telling and _tells(text[start + 2:end - 1]) else _COMPUTED)
            made += (text[position:start], mark if number is None else number[1] + number[2])
            position = end
        start = text.find("$(", end)
    return "".join(made) + text[position:]


def _trimmed(value: str, operator: str, pattern: str) -> str:
    """Return *value* without the prefix (*operator* ``#``) or suffix (``%``) *pattern* matches.

    The shortest one, or the longest when the operator is doubled (it then starts the
    pattern): ``${f%.c}``, ``${path##*/}``.
    """
    longest = pattern[:1] == operator
    glob = pattern[1:] if longest else pattern
    cuts = range(len(value) + 1)
    if (operator == "#") == longest:
        cuts = range(len(value), -1, -1)
    for cut in cuts:
        if operator == "#" and fnmatch.fnmatchcase(value[:cut], glob):
            return value[cut:]
        if operator == "%" and fnmatch.fnmatchcase(value[cut:], glob):
            return value[:cut]
    return value


def _edited(value: str | None, operator: str, pattern: str) -> tuple[str, ...]:
    """Return what bash may make of *value* with an editing *operator* (``${NAME%pattern}``).

    Nothing of nothing; of a directory the guard knows :data:`_EDITED`; of what a program
    printed something as unknown; of any other value what the guard can match itself (a
    prefix or a suffix removed), a part of a plain name (or nothing), else :data:`_ANY`:
    a replacement (``${D/a/$HOME}``), a change of case or a transformation (``${X,}``,
    ``${X@P}``) or a part of a path may name any directory.
    """
    if not value:
        return ("",)
    if value[0] in (_HERE, _HOME, "/"):
        return (_EDITED,)
    if operator == "@" and pattern == "Q":
        return (value,)  # quoted: read as the value itself
    if operator in "/^,@" or _ANY in value:
        return (_ANY,)
    if _FOUND in value:
        return (_FOUND,)
    if _SOME in value:
        return (_SOME,)
    if operator in "%#" and len(value) <= _MAX_EDITED and _PLAIN_PATTERN.fullmatch(pattern):
        return (_trimmed(value, operator, pattern),)
    if ".." in value or (operator != "%" and "/" in value):
        return (_ANY,)
    return (_SOME, "")


def _placed(texts: Iterable[str], here: str | None) -> list[str]:
    """Return the expanded *texts* with ``$PWD`` the directory *here*, where it is known."""
    return [text.replace(_HERE, here) for text in texts] if here else list(texts)


def _matches(name: str, pattern: str, glob: bool) -> bool:
    """Return whether the path component *pattern* names *name* (as a *glob*, or literally).

    A pattern in any case of its letters: ``nocaseglob`` may be set.
    """
    if not glob or _GLOB.search(pattern) is None:
        return name == pattern
    hidden = name.startswith(".") and not pattern.startswith(".")
    if _CLASS.search(pattern):
        return not hidden  # a class fnmatch does not read: any name
    return not hidden and fnmatch.fnmatchcase(name.lower(), pattern.lower().replace("[^", "[!"))


def _takes(path: str, glob: bool, protected: list[list[str]]) -> bool:
    """Return whether deleting the absolute *path* takes one of the *protected* directories.

    It names the directory or one above it, or (as a *glob*) every entry of it.
    """
    parts = [part for part in posixpath.normpath(path).split("/") if part]
    for names in protected:
        every = (glob and len(parts) == len(names) + 1
                 and _EVERY.fullmatch(parts[-1]) is not None)
        if (len(parts) <= len(names) or every) and all(
                _matches(name, part, glob) for name, part in zip(names, parts)):
            return True
    return False


def _kills_all(args: list[str]) -> bool:
    """Return whether ``kill args`` signals every process: a ``-1`` that is a pid.

    The first argument may be the signal (``kill -1 1234`` hangs up one process); a lone
    ``kill -1`` is refused, as a typo of the other.
    """
    if args[:1] in (["-s"], ["-n"]):
        args = args[2:]
    elif args[:1] and args[0].startswith("-") and args[0] != "--":
        if args == ["-1"]:
            return True
        args = args[1:]
    return any(_EVERY_PID.fullmatch(arg) for arg in args)


def _matches_empty(pattern: str) -> bool:
    """Return whether the regular expression *pattern* matches the empty string: every name.

    A bracket expression and an escape stand for one character (Python reads a POSIX class
    as a nested set); a pattern Python cannot read is judged by its letters alone (``False``).
    """
    pattern = _ANY_CHARACTER.sub("x", pattern)
    if "[" in pattern:
        return False
    try:
        return re.search(pattern, "") is not None
    except (re.error, OverflowError, RecursionError):
        return False


def _kills_any(args: list[str], inverts: bool) -> bool:
    """Return whether ``pkill args`` (or ``killall``) selects processes by nothing of their own.

    Its pattern, as bash expands it (a variable nobody set is empty), holds no letter, digit
    or unknown value outside brackets and escapes (``.``, ``.*``, ``''``, ``[a-z]``) or
    matches the empty string (``node|``, ``x*``, ``(x)?``), it is inverted (``pkill -v``;
    *inverts* tells that ``-v`` does that), or there is none and it selects by user.
    """
    user = False
    index = 0
    while index < len(args):
        text = args[index]
        index += 1
        if text in _PKILL_VALUED:
            index += 1
        if text.startswith(_PKILL_USERS):
            user = True
        elif inverts and _PKILL_INVERSE.fullmatch(text):
            return True
        elif not text.startswith("-"):
            return (re.search(r"[A-Za-z0-9\0-\x03]", _ANY_CHARACTER.sub("", text)) is None
                    or _matches_empty(text))
    return user


def _level(command: SimpleCommand, anchor: Group | None) -> SimpleCommand | Group | None:
    """Return what of the group *anchor* the *command* is part of: itself, or a group around it.

    ``None`` for a command outside that group (*anchor* ``None`` is the top level).
    """
    node: SimpleCommand | Group = command
    while node.group is not anchor:
        if node.group is None:
            return None
        node = node.group
    return node


def _ids(groups: list[Group]) -> frozenset[int]:
    """Return the ``id`` of each of the *groups*: what :func:`_within` looks a group up in."""
    return frozenset(map(id, groups))


def _within(command: SimpleCommand, groups: frozenset[int]) -> bool:
    """Return whether *command* is a command of one of the *groups* (:func:`_ids`)."""
    group = command.group
    while group is not None:
        if id(group) in groups:
            return True
        group = group.group
    return False


def _programs(words: list[Word], argv: list[Word], always: bool = False) -> list[list[Word]]:
    """Return the programs a command may run, each with its arguments: *argv*, or what
    follows a runner.

    After a command that runs its operands (the first of its *words* is one of
    :data:`_RUNNERS`; *always*: whatever it is), every later word that names a judged
    program (:data:`_JUDGED`), with the words up to the next one or a lone ``;`` or ``+``
    (the end of ``find -exec``).  The words after a shell's ``-c`` text are its parameters,
    whatever they name (``xargs sh -c '...' sh``).
    """
    if not always and posixpath.basename(words[0].text) not in _RUNNERS:
        return [argv]
    programs: list[list[Word]] = []
    program: list[Word] | None = None
    texted = False  # the program is a shell that has its ``-c`` text
    for word in words[1:]:
        if word.text in (";", "+"):
            program, texted = None, False
        elif not texted and (  # a pattern bash expands names a program where one may start
                _named(word.text, not word.literal) if program is None
                else posixpath.basename(word.text)) in _JUDGED:
            texted = (program is not None and _named(program[0].text) in SHELLS
                      and shell_script_index([part.text for part in program[1:]]) is not None)
            if program is None or not texted:
                program = [word]
                programs.append(program)
            else:
                program.append(word)
        elif program is not None:
            program.append(word)
    return programs


def _named(text: str, pattern: bool = True) -> str:
    """Return the name of the program the command word *text* runs, as the guard judges it.

    Its base name; ``sh`` for a shell by another name (:data:`_MORE_SHELLS`); for a file
    name *pattern* the judged program it may name (``/bin/ba?h``, ``r?``).
    """
    base = posixpath.basename(text)
    if pattern and _GLOB.search(base) is not None and base not in ("[", "[["):
        base = next((name for name in _NAMES if fnmatch.fnmatchcase(name, base)), base)
    return "sh" if base in _MORE_SHELLS else base


def _deaf(command: SimpleCommand) -> bool:
    """Return whether *command* cannot read its standard input: one of :data:`_DEAF` or a
    ``git rev-parse``, by a name as written, or no program and no redirection at all."""
    argv = command.argv()
    if not argv:
        return not command.redirections
    name = posixpath.basename(argv[0].text)
    return argv[0].literal and (
        name in _DEAF or (name == "git" and [word.text for word in argv[1:2]] == ["rev-parse"]))


def _expands(word: Word) -> bool:
    """Return whether bash makes the word *word* of a value: it holds an expansion."""
    return not word.literal and re.search(r"[$`]", _ANSI.sub("", word.text)) is not None


def _reader(word: Word) -> str:
    """Return the program the command word *word* runs, as :func:`_named` names it: ``sh``
    for one that a substitution prints or the environment names (``| $(command -v bash)``,
    ``| $SHELL``)."""
    computed = not word.literal and _SHELL_WORD.search(word.text) is not None
    return "sh" if computed else _named(word.text)


def _turns(command: SimpleCommand) -> bool:
    """Return whether *command* may change the directory of a loop it is in (``cd``, ``eval``).

    Each turn of that loop starts where the one before ended, so no ``cd`` tells where its
    commands run.  A subshell or a substitution keeps the change to itself.
    """
    group = command.group
    while group is not None and group.kind not in _LOOPS:
        if group.kind in _SUBSHELLS:
            return False
        group = group.group
    return group is not None and (
        command.unknown or command.program() in ("cd", "pushd", "popd", "eval", "source", "."))


def _exits(commands: list[SimpleCommand], index: int) -> bool:
    """Return whether the shell goes on past the ``cd`` at *index* of *commands* only when it
    succeeded: the ``cd`` runs whenever the commands after it do, and ``|| exit`` follows it.

    A plain ``exit`` (``exit``, ``exit 1``, ``exit $?``): nothing of it may fail, or run first.
    """
    command = commands[index]
    if command.conditional or command.piped or command.reader is not None or command.background:
        return False
    held = _ids(command.substitutions)
    for position in range(index + 1, len(commands)):
        other = commands[position]
        if _within(other, held):
            continue
        words = other.words
        return (other.conditional and not other.after_and and other.group is command.group
                and other.script is None and other.reader is None
                and not (other.background or other.piped or other.heredoc
                         or other.redirections or other.substitutions)
                and 1 <= len(words) <= 2 and words[0].literal and words[0].text == "exit"
                and all(word.text.isdigit() or word.text == "$?" for word in words[1:]))
    return False


def _shell_readers(commands: list[SimpleCommand]) -> dict[int, bool]:
    """Return the ids of what hands its standard input to a shell as its script, each with
    whether that shell is the one that runs the command (a ``source``).

    Each shell of *commands* without ``-c`` (behind a runner too), the groups around it and
    the command whose text or function it is a command of (``f(){ bash; }; echo .. | f``):
    it inherits their input, or a command of the group hands it on (``echo .. | { bash; }``,
    ``echo .. | (cat | bash)``).
    """
    readers: dict[int, bool] = {}
    callers = {id(command.script): command for command in commands
               if command.script is not None}
    for command in commands:
        argv = command.argv()
        if not argv:
            continue
        words = [word for word in command.words if not word.assignment]
        sourced = _sources_input(argv)
        if not (sourced or any(
                reads_script([Word(_reader(program[0]), True), *program[1:]])
                for program in _programs(words, argv))):
            continue
        node: SimpleCommand | Group | None = command
        while node is not None and not (
                isinstance(node, Group) and node.kind in _SUBSTITUTIONS):
            readers[id(node)] = sourced or readers.get(id(node), False)
            node = callers.get(id(node), node.group)
    return readers


def _placeholders(words: list[Word]) -> set[str]:
    """Return the texts a runner among *words* may replace with each line it reads: ``{}``,
    and what an option names (:data:`_REPLACED`; one of the program it runs may look alike)."""
    marks = {"{}"}
    for index, word in enumerate(words):
        option = _REPLACED.fullmatch(word.text)
        if option is not None:
            named = option[1] or option[2] or ""
            if not named and word.text.endswith("I"):
                named = "".join(following.text for following in words[index + 1:index + 2])
            marks.add(named or "{}")
    return marks


def _marks(command: SimpleCommand) -> bool:
    """Return whether *command* is one the guard wrote where a branch ends (:data:`_ELSE`)."""
    return len(command.words) == 1 and command.words[0].text in (_ELSE, _BRANCH)


def _sources_input(argv: list[Word]) -> bool:
    """Return whether the command *argv* is a ``source`` (``.``) of its standard input or of
    a process substitution: text the Bash call hands on, as the script of ``sh`` is."""
    return argv[0].text in ("source", ".") and reads_script([Word("sh", True), *argv[1:]])


def _receivers(command: SimpleCommand) -> list[tuple[SimpleCommand | Group, bool]]:
    """Return what the standard output of *command* reaches through a pipe, the nearest
    first: each command or group with whether it gets the text as *command* wrote it.

    Through a pipe or a process substitution (``bash <(echo ..)``), from the group it is in
    or through the commands in between: a ``cat`` or a ``tee`` hands the text on as it is
    (``echo .. | tee log | sh``), any other command makes another text of it.
    """
    reached: list[tuple[SimpleCommand | Group, bool]] = []
    node: SimpleCommand | Group | None = command
    plain = True
    for _ in range(_MAX_PIPES):
        if node is None:
            break
        piped = node.reader is not None
        group = node.group
        if not piped and group is not None and group.asynchronous and (
                group.kind in _SUBSTITUTIONS):
            node = group  # a process substitution: a stream, which its reader gets
            continue
        node = receiver(node)
        if piped and node is not None:
            reached.append((node, plain))
            plain = plain and isinstance(node, SimpleCommand) and node.program() in _COPIERS
    return reached


def _feeds_shell(command: SimpleCommand, readers: dict[int, bool]) -> tuple[bool | None, bool]:
    """Return whether the standard output of *command* is the script of a ``source``
    (``False``: of a shell of its own; ``None`` when it is nobody's script), and whether
    that script is the text as *command* wrote it.

    The script of the nearest of the *readers* (:func:`_shell_readers`) it reaches
    (:func:`_receivers`).
    """
    for node, plain in _receivers(command):
        if id(node) in readers:
            return readers[id(node)], plain
    return None, True


def _aside(command: SimpleCommand) -> bool:
    """Return whether *command* may run in a shell of its own: in a pipeline, or after ``&``.

    What it assigns is then lost, or kept (the last command of a pipeline, in a shell
    that runs it itself).
    """
    return bool(command.piped or command.reader is not None or command.background
                or command.asynchronous)


def _operated(argv: list[Word]) -> bool:
    """Return whether the shell *argv* has operands: the parameters of a script it reads."""
    return any(not word.text.startswith(("-", "+")) for word in argv[1:])


#: The commands whose output reaches a command or a group (by its id) through a pipe, each
#: with whether its text arrives as written (:func:`_receivers`).
_Feeds = dict[int, list[tuple[SimpleCommand, bool]]]


class _Scope(dict[str, "frozenset[str | None]"]):
    """The variables one shell wrote, over those of the shell it was started from.

    A variable it did not write is its parent's; a *child* that is another program
    (``bash -c``) gets it only when it was exported, so there it may also be unset.  A
    variable no shell wrote is unset.
    """

    def __init__(self, parent: _Scope | None, child: bool = False) -> None:
        super().__init__()
        self.parent = parent
        self.child = child
        #: How many shells that are programs of their own it is in: what one of them ever
        #: assigned is nothing the shells that started it read (:attr:`_Guard.ever`).
        self.level: int = (parent.level if parent is not None else 0) + child

    def __missing__(self, name: str) -> frozenset[str | None]:
        """Return the values of *name* in the parent shell, as this shell sees them."""
        if self.parent is None:
            return _UNSET
        values = self.parent[name]
        return values | _UNSET if self.child else values


@dataclass
class _Frame:
    """What a group of commands, a ``&&`` list or a text assigned, until it ends.

    Attributes:
        group: The group it covers; for a list, the group the list is in.
        listing: Whether it is a ``&&`` list: the commands after its first run only when
            the ones before succeeded.
        tentative: Whether its first command may follow ``||``: what it assigns is not
            sure even for the next command of the list.
        merges: Whether its commands may not have run when it ends: a variable then
            holds what they assigned or what it held before.  Else it holds what they
            assigned.
        scope: The variables to go back to when it ends: its commands ran in a shell
            of their own.
        prefix: The positional parameters to go back to (:attr:`_Guard.prefix`).
        late: ``1`` for a loop or a trap, whose commands also run after the ones
            written after them; ``2`` for the body of a function's definition, which
            runs when something calls it.
        function: Whether it is a function's call: a variable its commands declare is
            the function's own (:attr:`own`).
        own: Those variables.
        saved: What each variable its commands assigned held before.
        branches: Whether its commands are those of several branches the scanner does
            not tell apart (a ``case``, an ``if`` with an ``else``): what one assigns adds
            to what a variable may hold (:meth:`_Guard.branched`).
        header: Whether it is a ``for`` or a ``select`` whose first command, its header,
            the walk has not reached yet.
        wrote: Every value its commands assigned each variable.
        ends: What each variable held where a branch of it ended (an ``if`` or a ``case``
            whose branches the guard tells apart); ``None`` before the first end.
        closed: Whether one of its branches runs for sure (the last one is an ``else``):
            a variable then holds what one of them left.
        apart: For the text of a shell of its own, that text and whether its caller
            builds words in a way the guard does not model (:attr:`_Guard.opaque`: the
            text starts without it).
    """

    group: Group | None
    listing: bool = False
    tentative: bool = False
    merges: bool = False
    scope: _Scope | None = None
    prefix: str | None = None
    late: int = 0
    function: bool = False
    own: set[str] = field(default_factory=set)
    saved: dict[str, frozenset[str | None]] = field(default_factory=dict)
    branches: bool = False
    header: bool = False
    wrote: dict[str, frozenset[str | None]] = field(default_factory=dict)
    ends: dict[str, frozenset[str | None]] | None = None
    closed: bool = False
    apart: tuple[str, bool] | None = None


class _Guard:
    """The judge of one Bash call: whether a command of it ends the task.

    It allows a call only when every word bash may make of each ``rm`` operand, ``kill``
    pid and ``pkill`` pattern is safe: a word is read as all the texts its expansions may
    make, and one the guard cannot bound (:data:`_ANY`) refuses the call.

    Every simple command bash would run is judged by its words after quote removal
    (:func:`~kiss.agents.sorcar.shell_scan.simple_commands`), so what a command quotes cannot
    hide or split a word.  A variable holds a set of values (:attr:`values`): unset until
    a command the walk has passed assigns it (``rm -rf $BUILD/*`` deletes ``/*``), every
    value it may hold after commands that may not run (:class:`_Frame`), and what a
    program printed or read (``d=$(mktemp -d)``, ``read d``) is unknown and names nothing
    protected.  ``$PWD`` and ``$(pwd)`` are the directory the command runs in, ``~`` and
    ``$HOME`` the home directory.

    Read as commands too: the text of a shell's ``-c``, of ``eval`` and of ``trap``, what a
    here-document, a here-string or a pipe hands a shell, and what follows a command that
    runs its operands (:data:`_RUNNERS`).  Not read: a script file, a remote shell
    (``ssh``), text or operands a program prints (``eval "$(...)"``, ``xargs``), a program
    only the shell or the filesystem can name (``$(which rm)``, ``/bin/r?``).
    """

    def __init__(
        self, work_dir: str | None, isdir: Callable[[str], bool],
        realpath: Callable[[str], str], cdpath: Callable[[], str],
    ) -> None:
        self.isdir = isdir
        self.realpath = realpath
        self.cdpath = cdpath
        #: The work directory as bash spells it, where every call starts.
        self.start = bash_path(_slash_path(work_dir)) if work_dir else None
        self.home = bash_path(_slash_path(os.path.expanduser("~")))
        self.resolved: dict[str, str] = {}
        self.protected: list[list[str]] | None = None
        self.scans: dict[tuple[str, str | None], list[SimpleCommand]] = {}
        #: The variables the call assigns an array, and the functions it defines.  A call of
        #: one that the scanner does not follow (in a trap, by a variable) has :attr:`strayed`:
        #: the body of every definition is then read with the words of :attr:`loose` as its
        #: parameters, at any time.
        self.arrays: set[str] = set()
        self.functions: set[str] = set()
        self.loose: set[str] = set()
        self.strayed = False
        #: Every value a command of the call may assign each variable of the shells at each
        #: level (:attr:`_Scope.level`): what a command that runs again or later
        #: (:attr:`_Frame.late`) may also read.  :attr:`grown` tells that the reading under
        #: way added to it, :attr:`delayed` that it held such a command.  A variable that
        #: still gets new values once the readings are :attr:`widening` is :attr:`widened`:
        #: it holds any value.  :attr:`unsettled`: every variable does, read late.
        self.ever: dict[tuple[int, str], set[str | None]] = {}
        self.widened: set[tuple[int, str]] = set()
        self.grown = self.delayed = self.unsettled = self.widening = False
        #: Whether the text being read has branches the guard could not mark
        #: (:data:`_BRANCH_END`: each ``if`` and ``case`` of it then assigns as
        #: :meth:`branched` tells) or marked (:meth:`turns`), an arithmetic ``for``, whose
        #: header the scanner does not show, and ``||`` before an ``if``.
        self.forked = self.marked = self.counted = self.orred = False
        #: Whether a substitution that may print a protected directory is any text
        #: (:func:`_tells`).  Not while :meth:`texts` expands a text for the shell that
        #: hands it on: the shell that runs it reads the substitution as written.
        self.telling = True
        #: Whether the call builds words in a way the guard does not model (:meth:`shades`:
        #: a delete, a signal, a command word or a text for a shell that is not literal is
        #: then refused), whether an ``exec`` of it redirects input (:data:`_EXEC_INPUT`),
        #: and whether the call mentions readonly variables.  The text being read
        #: (:meth:`patterned`).
        self.opaque = self.rewired = self.readonly = False
        self.written = ""
        #: The texts of shells of their own that build words that way (:meth:`reads`).
        self.shaded: set[str] = set()
        #: Whether the call may run what the guard does not read: a variable holds a
        #: substitution that names a judged program (:data:`_ARMED`), arithmetic evaluates
        #: any text (:meth:`counts`), the call names programs itself (``BASH_ALIASES``).
        self.armed = False
        #: The aliases the call defines: a command by the name of one runs its text.
        self.aliases: set[str] = set()
        #: Whether the call declares a name reference: what it assigns may go to any
        #: variable, one that bash evaluates too (:meth:`store`).
        self.referenced = False
        #: The variables bash's arithmetic assigns somewhere in the call (each then holds a
        #: number nobody knows, wherever it is read), and the names its expressions read
        #: (``@``: the positional parameters): the value of one is an expression too
        #: (:meth:`counts`).
        self.numbers: set[str] = set()
        self.reckoned: set[str] = set(_INTEGERS)
        #: Whether the scanner reads a command at a place of a text (:meth:`stands`).
        self.stood: dict[tuple[str, int], bool] = {}
        #: Whether the command being read has text the call wrote on its standard input
        #: (:meth:`supplies`), whether the text it is a command of has (a ``-c`` text of a
        #: shell that is fed one), and the commands that call the functions and the texts
        #: of the text being read.
        self.inlet = self.early = self.inherits = False
        self.callers: dict[int, SimpleCommand] = {}
        #: The texts each arithmetic expression with an expansion made when it was last read
        #: (:meth:`settled`).
        self.reckonings: dict[str, list[str]] = {}
        #: The values each variable may hold where the command being read runs, and the
        #: frames that are open there.  The positional parameters are the variables
        #: ``<prefix>0``, ``<prefix>1``... of the text or the call being read, with
        #: ``<prefix>@`` all of them and ``<prefix>*`` what each may be once their
        #: positions are unknown (after ``shift``, in a function's definition).
        self.values = _Scope(None)
        self.frames: list[_Frame] = []
        self.prefix = ""
        self.serial = 0
        #: How many open frames are loops or traps, and how many bodies of definitions.
        self.late = self.defining = 0
        #: Whether the command being read may not assign (:func:`_aside`), and what it
        #: puts in the environment of the program it runs (``D=x bash -c ...``).
        self.weak = False
        self.exports: list[tuple[str, frozenset[str | None]]] = []

    def real(self, path: str) -> str:
        """Return *path* as the filesystem names it (the first :data:`_MAX_RESOLVED` asked);
        as it is when no file can have that name (a NUL, a lone surrogate)."""
        if path not in self.resolved:
            if len(self.resolved) >= _MAX_RESOLVED:
                return path
            try:
                self.resolved[path] = bash_path(_slash_path(self.realpath(host_path(path))))
            except ValueError:
                self.resolved[path] = path
        return self.resolved[path]

    def physical(self, path: str) -> str:
        """Return the absolute *path* with the links of the directories ``rm`` walks resolved.

        Not its last name (``rm -rf link`` removes the link) unless a ``/`` follows it, and
        nothing from the first glob on.
        """
        names = [name for name in path.split("/") if name not in ("", ".")]
        current = "/"
        for index, name in enumerate(names):
            last = index == len(names) - 1 and not path.endswith(("/", "/."))
            if name == "..":
                current = posixpath.dirname(self.real(current))
            elif last or _GLOB.search(name) is not None or _SOME in name:
                return posixpath.join(self.real(current), *names[index:])
            else:
                current = posixpath.join(current, name)
        return self.real(current)

    def directory(self, path: str) -> bool:
        """Return whether the absolute bash *path* is a directory, as bash's ``cd`` finds it.

        The host is asked by its own spelling of the path; on Windows with the links before
        each ``..`` followed (:meth:`physical`), as Git Bash walks it: Windows itself drops a
        ``..`` with the name before it, a link or not.
        """
        if os.name == "nt" and ".." in path.split("/"):
            path = self.physical(path + "/")
        return self.isdir(host_path(path))

    def takes(self, path: str, glob: bool) -> bool:
        """Return whether deleting *path* takes the work directory, the home directory or ``/``.

        By its text, and by the path the filesystem makes of it: a link back into the work
        directory is the work directory.
        """
        if self.protected is None:
            paths = {"/", self.home}
            if self.start:
                paths |= {self.start, self.real(self.start)}
            self.protected = [
                [name for name in posixpath.normpath(path).split("/") if name]
                for path in paths if path.startswith("/")]
        return _takes(path, glob, self.protected) or _takes(
            self.physical(path), glob, self.protected)

    def store(self, name: str, values: Iterable[str | None]) -> None:
        """Record that the variable *name* holds one of *values* from here on.

        Until the frame on top ends (:meth:`leaves`); or one of those it held, for a
        command that may not assign (:attr:`weak`).  More than :data:`_MAX_ALTERNATIVES`
        values, or one the guard cannot bound, are any value.
        """
        if name == "IFS":
            self.shades()
        if name in ("BASH_ALIASES", "BASH_CMDS") or any(
                value and (_ARMED.search(value) or (name in _PROMPTS and _ANY in value))
                for value in values):
            self.armed = True
        if any(value and len(value) > _MAX_VALUE for value in values):
            values = _ANYTHING  # too long to follow
        held = frozenset(
            _RUN.sub(_SOME, value) if value and _SOME in value else value for value in values)
        if self.weak or self.branched():
            held |= self.values[name]
        key = (self.values.level, name)
        if (key in self.widened or len(held) > _MAX_ALTERNATIVES or not self.telling
                or any(value and _ANY in value for value in held)):
            held = _ANYTHING
        ever = self.ever.setdefault(key, set())
        if not held <= ever:
            if self.widening:  # its values do not settle
                self.widened.add(key)
                held = _ANYTHING
            ever |= held
            self.grown = True
        frame = self.frames[-1]
        if name not in frame.saved:
            frame.saved[name] = self.values[name]
        frame.wrote[name] = frame.wrote.get(name, frozenset()) | held
        self.values[name] = held
        if self.referenced or name in self.reckoned or (":" in name and "@" in self.reckoned):
            self.counts(value for value in held if value)

    def counts(self, expressions: Iterable[str]) -> None:
        """Note what bash's arithmetic does when it evaluates the *expressions*.

        A name one assigns (:data:`_COUNT`) is one of :attr:`numbers`; the call is read
        again, for the commands read before.  A name one reads is :attr:`reckoned`: every
        value the call assigns it is an expression too (``X='P=-1'; (( X ))``).  One the
        guard cannot bound may hold any substitution: the call is refused (:attr:`armed`).
        """
        pending = [*expressions]
        while pending:
            expression = pending.pop()
            if _ANY in expression:
                self.armed = True  # any text: bash runs the substitutions it holds
            assigned = {first or second for first, second in _COUNT.findall(expression)}
            if not assigned <= self.numbers:
                self.numbers |= assigned
                self.grown = self.delayed = True
            read = set(_NAME.findall(expression))
            if _POSITIONAL.search(expression):
                read.add("@")
            for name in read - self.reckoned:
                self.reckoned.add(name)
                if name in _BASH_SET:
                    self.armed = True  # text bash sets, which the guard does not follow
                pending += [value for (_, key), held in self.ever.items()
                            if (":" in key if name == "@" else key == name)
                            for value in held if value]

    def shades(self) -> None:
        """Note that the call builds words in a way the guard does not model (:attr:`opaque`).

        It is read again: a command read before may run after this one.
        """
        if not self.opaque:
            self.opaque = True
            self.grown = self.delayed = True

    def patterned(self, word: Word) -> bool:
        """Return whether an extended pattern goes on where the scanner ended *word*
        (``rm -rf /tmp/@(w)``: the word is ``/tmp/@``).

        By the last part of its path: the text being read holds it right before a ``(``.
        """
        tail = _QUOTES.sub("", word.text).rpartition("/")[2]
        return tail.endswith(("@", "!", "?", "*", "+")) and re.search(
            r"(?<![^\s;&|'\"/])" + re.escape(tail) + r"\(", self.written) is not None

    def holds(self, name: str) -> frozenset[str | None]:
        """Return the values the variable *name* may hold where the command being read runs.

        In a loop, a trap or the body of a function's definition also every value the
        call assigns it anywhere, in the shell that runs the command or one that started
        it: the command runs again or later, after the ones written after it.
        """
        values = self.values[name]
        if self.late or self.defining:
            for level in range(self.values.level + 1):
                written = self.ever.get((level, name))
                if written:
                    if self.unsettled:
                        return _ANYTHING
                    values |= written
        return values

    def branched(self) -> bool:
        """Return whether the command being read is one of a branch the guard cannot tell
        from the other branches of its ``if`` or ``case`` (:attr:`_Frame.branches`).

        What it assigns then adds to what a variable may hold; a group or a call in a
        branch runs its own commands in order, and may not run.
        """
        for frame in reversed(self.frames):
            if not frame.listing:
                return frame.branches
        return False

    def elements(self) -> frozenset[str]:
        """Return the values a positional parameter of the text being read may hold."""
        pool = self.holds(self.prefix + "*")
        if pool == _UNSET:
            pool = self.holds(self.prefix + "@")
        return frozenset(value for value in pool if value is not None)

    def parameter(self, name: str) -> frozenset[str | None]:
        """Return the values the parameter *name* may hold in the shell that runs the command.

        A variable by what the commands so far assigned (:meth:`holds`), ``PWD``, ``HOME``
        and ``OLDPWD`` (where the call started) by their directory until a command assigns
        them; a positional parameter by the words its text was called with, any of them
        once their positions are unknown; ``$_`` any value; another special parameter a
        number or a name, unknown.
        """
        if name.isdigit():
            if name == "0":
                return frozenset(value or _SOME for value in self.holds(self.prefix + "0"))
            pool = self.holds(self.prefix + "*")
            return self.holds(self.prefix + name[:9]) if pool == _UNSET else pool
        if name in "@*":
            return self.elements() or _EMPTY
        if name == "_":
            return _ANYTHING
        if name in ("?", "#"):
            return _COUNTS
        if not (name.isidentifier() and name.isascii()):
            return _UNKNOWN
        if name in _BASH_SET:
            return _ANYTHING
        values = self.holds(name)
        if name in self.numbers:
            values |= frozenset(_A_NUMBER)
        if None in values and name in _COUNTERS:
            values = values - _UNSET | _COUNTS
        if None in values and name in ("PWD", "HOME", "OLDPWD", "SHELL", "BASH"):
            inherited = {"PWD": _HERE, "HOME": _HOME, "SHELL": "bash", "BASH": "bash"}.get(
                name, self.start or "")
            values = values - _UNSET | {inherited}
        elif None in values and self.defining and not self.strayed:
            # the calls the scanner follows are judged with what the variable holds then
            values = values - _UNSET or _UNKNOWN
        return values

    def expansion(self, match: re.Match[str], computed: str) -> Collection[str]:
        """Return the texts the expansion *match* (:data:`_EXPANSION`) may stand for in its word.

        A parameter by its values (:meth:`parameter`) and its operator, as bash reads them:
        ``${X:-word}`` is the word when X is empty or unset (``${X-word}``: when unset),
        ``${X:=word}`` assigns it too, ``${X:?}`` is X or an error (unknown), ``${X:+word}``
        the word when X is not empty; any other operator edits the value (:func:`_edited`).
        What a program prints is *computed*; another variable's value (``${!X}``) and an
        expansion the guard does not read are any value.
        """
        if match["here"]:
            return (_HERE,)
        if match["ansi"] is not None:
            return (_ansi(match),)
        if match["some"]:  # arithmetic that is not one number
            return _A_NUMBER
        name = match["name"] or match["braced"]
        if not name:
            if match[0].startswith("${"):
                return (_ANY,)
            told = match[0].startswith("`") and self.telling and _tells(match[0][1:-1])
            return (_FOUND if told else computed,)
        if match["prefix"]:  # a length, the indices of an array; or another variable's value
            if match["prefix"] == "#" or match["subscript"]:
                return (_SOME, "0", "1")
            named = self.parameter(name)
            if match["operator"] or not all(one and one.isidentifier() for one in named):
                return (_ANY,)
            return frozenset(value or "" for one in named if one for value in self.parameter(one))
        values = self.parameter(name)
        if match["subscript"]:  # an element, or none
            values |= _EMPTY
        operator = match["operator"]
        if not operator:
            return {value or "" for value in values}
        if operator.startswith(":") and operator[-1] not in "-=?+":  # a slice: a part of it
            return {part for value in values for part in _parts(value or "")}
        if operator[-1] not in "-=?+":
            return {text for value in values for text in _edited(value, operator, match["word"])}
        # without its colon an operator asks whether the variable is set, empty or not
        colon = len(operator) == 2
        held = {value for value in values if value is not None and (value or not colon)}
        missing = len(held) < len(values)
        # the operator's word as written, and without the quotes bash removes from it
        spellings = dict.fromkeys((match["word"], _QUOTES.sub("", match["word"])))
        if operator[-1] == "+":
            words = {made for word in spellings if held for made in self.expanded(word, computed)}
            return words | {""} if missing else words
        if operator[-1] == "?":
            return held or (_SOME,)
        if missing:
            held |= {made for word in spellings for made in self.expanded(word, computed)}
            if operator[-1] == "=" and name.isidentifier():
                self.store(name, held)
        return held

    def tilde(self, prefix: re.Match[str]) -> str:
        """Return what the tilde-prefix *prefix* (:data:`_TILDE`) stands for: the variable
        bash reads for it, to expand (the call may have assigned it), or a user's home."""
        if _STACKED.fullmatch(prefix[0]):
            return _ANY
        named = {"~": "${HOME}", "~+": "${PWD}", "~-": "${OLDPWD}"}.get(prefix[0])
        if named is None:
            named = bash_path(_slash_path(os.path.expanduser(prefix[0])))
        return named

    def expanded(self, text: str, computed: str = "", depth: int = 0) -> list[str]:
        """Return the texts bash may make of *text*: its tilde-prefix and its expansions replaced.

        ``~`` is the home directory, ``~+`` the directory the command runs in, ``~-`` the
        one before it (read as where the call started), ``~name`` the home of that user.
        One text per combination of the values its expansions may have (an expansion
        written twice has one value); :data:`_ANY` alone past :data:`_MAX_ALTERNATIVES` of
        them, and for a text with an expansion the guard does not read.  A substitution
        whose output the guard reads (:meth:`printed`, :meth:`output`) is replaced by each
        text it may print, *depth* of them so far, and the rest expanded in turn.
        """
        printed = self.printed(text)
        if printed is None:
            for commands in _held(text):
                lines = self.output(commands)
                if lines is None:
                    return [_ANY]
                if lines:  # what its programs find, or one of the lines it writes itself
                    written = f"$({commands})" if f"$({commands})" in text else f"`{commands}`"
                    mark = _FOUND if self.telling and _tells(commands) else _COMPUTED
                    printed = [text.replace(written, line.rstrip("\n")) for line in (mark, *lines)]
                    break
        if printed is not None:
            if depth >= _MAX_RESOLVED:  # a value that holds the substitution that prints it
                return [_ANY]
            made: list[str] = []
            for output in printed:
                made += self.expanded(output, computed, depth + 1)
                if _ANY in made or len(made) > _MAX_ALTERNATIVES:
                    return [_ANY]
            return list(dict.fromkeys(made))
        user, slash, rest = text.partition("/")
        prefix = _TILDE.fullmatch(user)
        if prefix is not None:
            text = self.tilde(prefix) + slash + rest
        text = _marked(_SUM.sub(self.summed, text), self.telling)
        options: dict[str, tuple[str, ...]] = {}
        pieces: list[str] = []
        count, position = 1, 0
        for match in _EXPANSION.finditer(text):
            if _UNREAD.search(text, position, match.start()):
                return [_ANY]
            pieces += (text[position:match.start()], match[0])
            position = match.end()
            if match[0] not in options:
                options[match[0]] = tuple(self.expansion(match, computed))
                count *= len(options[match[0]])
                if count > _MAX_ALTERNATIVES:
                    return [_ANY]
        if _UNREAD.search(text, position):
            return [_ANY]
        made = []
        for values in itertools.product(*options.values()):
            chosen = dict(zip(options, values))
            made.append("".join(
                chosen[piece] if index % 2 else piece for index, piece in enumerate(pieces)
            ) + text[position:])
        return list(dict.fromkeys(made))

    def summed(self, total: re.Match[str]) -> str:
        """Return what stands for the arithmetic expansion *total* (:data:`_SUM`).

        The number it makes when each term is one number (a variable that is not set
        counts 0); a number a program found (:data:`_SOME`) when one term is nothing but
        such a number and the others are not negative.  Else the expansion as written: a
        number nobody knows.
        """
        terms = [frozenset({term}) if term.isdigit()
                 else self.parameter(term.lstrip("$").strip("{}"))
                 for term in map(str.strip, total[0][3:-2].split("+"))]
        if not all(value is None or _ADDEND.fullmatch(value) for held in terms for value in held):
            return total[0]
        if all(len(held) == 1 and held != _UNKNOWN for held in terms):
            return str(sum(int(value or 0) for held in terms for value in held))
        return _SOME if _UNKNOWN in terms else total[0]

    def printed(self, text: str) -> list[str] | None:
        """Return *text* with what its first substitution that runs ``echo`` or ``printf``
        alone may print in its place (:meth:`prints`); ``None`` when it holds none.

        That output is text the Bash call itself writes, not what a program found; so is
        the operand ``realpath`` and ``readlink`` print (where a link leads too) and the
        directory of it ``dirname`` prints.  More such substitutions than the guard
        follows are any text.
        """
        for count, start in enumerate(_PRINTED.finditer(text)):
            if count >= _MAX_RESOLVED:
                return [_ANY]
            opened = start.end(1)
            end = text.find("`", opened) + 1 if start[1] == "`" else _closing(text, opened)
            if end <= 0:
                continue
            commands = self.scan(text[opened:end - 1], None)
            top = [command for command in commands if command.group is None]
            if len(top) != 1 or top[0].reader is not None:
                continue
            held = _ids(top[0].substitutions)
            if not all(command is top[0] or _within(command, held) for command in commands):
                continue
            argv = top[0].argv()
            name = posixpath.basename(argv[0].text)
            if name in ("echo", "printf"):
                outputs = self.prints(argv, _HERE) or [""]
            else:
                operands = [operand for word in argv[1:] if not word.text.startswith("-")
                            for operand in self.expansions(word, _SOME)]
                outputs = [made for operand in operands for made in self.pathed(name, operand)]
                if name == "basename":  # without the suffix another operand may be
                    outputs += [made[:-len(suffix)] for made in outputs for suffix in operands
                                if suffix and made != suffix and made.endswith(suffix)]
            return [text[:start.start()] + output.rstrip("\n") + text[end:]
                    for output in outputs or [_SOME]]
        return None

    def output(self, commands: str) -> list[str] | None:
        """Return what a substitution that runs *commands* may print besides what its programs
        find: the lines its ``echo`` commands write (``$(git rev-parse .. || echo .)``);
        ``None`` for any text.

        Any text: one of the commands is a function of the call, reads a here-document or a
        here-string, or is an ``echo`` or a ``printf`` in a pipe, in a substitution of its
        own or without a newline at its end (the next one's text goes on from there).  Not
        one that hands on nothing but what programs found (:meth:`found`).
        """
        lines: list[str] = []
        if _LOOPED.search(commands):
            return None  # it may print what it sets itself
        scanned = self.scan(commands, None)
        if self.early and not all(map(_deaf, scanned)):
            return None  # it may print the text on the command's input
        programs = any(command.argv() for command in scanned)  # none: it prints nothing
        for command in scanned:
            argv = command.argv()
            name = posixpath.basename(argv[0].text) if argv else ""
            if name in self.functions or command.reads_heredoc() or name in _SETTERS or any(
                    _DESCRIPTOR.search(word.text) for word in command.words) or any(
                    word.assignment and programs for word in command.words) or any(
                    operator == "<&" for _, operator, _ in command.redirections) or (
                    name == "printf" and any(word.text == "-v" for word in argv[1:2])):
                return None
            nested = False
            node: SimpleCommand | Group | None = command
            while node is not None:
                if any(operator == "<<<" and not self.found(word)
                       for _, operator, word in node.redirections):
                    return None
                nested = nested or (isinstance(node, Group) and node.kind in _SUBSTITUTIONS)
                node = node.group
            if name in ("echo", "printf") and not all(map(self.found, argv[1:])):
                printed = self.prints(argv, _HERE)
                if nested or _receivers(command) or not all(
                        line.endswith("\n") for line in printed):
                    return None
                lines += printed
        return lines

    def supplies(self, command: SimpleCommand, fed: _Feeds, own: bool = True) -> bool:
        """Return whether the standard input of *command* is text the Bash call itself writes.

        A here-string or a here-document, its own or of a group around it, a descriptor the
        call opened (``<&3``), or what reaches it through a pipe (*fed*) from an ``echo``, a
        ``printf`` (a process substitution that starts with one) or a command that reads
        such text itself.  Not a here-string or an ``echo`` of nothing but what programs
        found (:meth:`found`: ``read major minor <<< "$(cat VERSION)"``).  Without *own*,
        the input while bash expands the command's words: it makes the command's own
        redirections only after that (``"$(sed .. FILE)python" - <<EOF``).
        """
        node: SimpleCommand | Group | None = command
        while node is not None:
            if ((own or node is not command) and (node.heredoc or any(
                    operator == "<&" or _PRINTED_INPUT.match(word.text) or (
                        operator.startswith("<<")
                        and not (operator == "<<<" and self.found(word)))
                    for _, operator, word in node.redirections))) or any(
                    not all(map(self.found, writer.argv()[1:]))
                    if writer.program() in ("echo", "printf") else self.supplies(writer, {})
                    for writer, _ in fed.get(id(node), ())):
                return True
            node = self.callers.get(id(node), node.group)  # of a function: the call's
        return False

    def found(self, word: Word) -> bool:
        """Return whether the word *word* holds nothing but what programs found, or blanks."""
        if word.literal:
            return _BLANK.fullmatch(word.text) is not None
        return not any(made.strip(_SOME + " \t\n") for made in self.expansions(word, _SOME))

    def pathed(self, name: str, operand: str) -> tuple[str, ...]:
        """Return what ``dirname`` (``basename``, ``realpath``, ``readlink``: *name*) may print
        for *operand*: the operand, and its directory, its name or the path its links lead to."""
        if _ANY in operand or _FOUND in operand:
            return (_ANY if _ANY in operand else _FOUND,)
        if name == "dirname" and operand.rstrip("/") in (_HERE, _HOME, _EDITED):
            return (_EDITED,)  # the directory above one the guard knows
        if name == "dirname":
            parent = posixpath.dirname(operand.rstrip("/"))
            if not parent and _SOME in operand:  # a name the guard does not know
                return (operand,)
            return (operand, parent or ("/" if operand.startswith("/") else "."))
        if name == "basename":
            last = posixpath.basename(operand.rstrip("/"))
            # the name of a directory the guard knows, as any name: after ``../``, that
            # directory (``../$(basename "$PWD")``); with more after it, another one
            return ("./" + _EDITED,) if last in (_HERE, _HOME, _EDITED) else (operand, last)
        if operand.startswith("/") and _SOME not in operand:
            return (operand, self.real(operand))
        return (operand,)

    def words(self, word: Word, computed: str = "") -> list[list[str]]:
        """Return the words bash makes of *word*, each as the texts it may be.

        One word per part of a brace expansion; a word that is all the positional
        parameters (``"$@"``) is any of them, or no word.
        """
        if word.literal:
            return [[word.text]]
        if _ALL_ARGUMENTS.fullmatch(word.text):
            return [list(self.elements())]
        parts = _alternatives(word.text)
        if len(parts) > _MAX_ALTERNATIVES or any(map(_SEQUENCE.search, parts)):
            return [[_ANY]]
        return [self.expanded(part, computed) for part in parts]

    def expansions(self, word: Word, computed: str = "") -> list[str]:
        """Return every text a word bash makes of *word* may be (:meth:`words`)."""
        return [text for texts in self.words(word, computed) for text in texts]

    def bases(self, cwd: str | None, trusted: bool, searched: bool, text: str) -> list[str]:
        """Return the directories a relative operand of a command run in *cwd* may be read from.

        The one the ``cd`` commands lead to when they are proven (*trusted*) and ``CDPATH``
        cannot have redirected a bare ``cd name`` (*searched*); else that one and the work
        directory, where the call started.
        """
        if trusted and searched and ("CDPATH" in text or self.cdpath()):
            trusted = False
        if trusted or not self.start:
            return [cwd] if cwd else []
        works = [self.start, self.real(self.start)]
        if cwd:
            return [*works, cwd]
        # nobody knows the directory: the work directory's own name may be it, from above
        return [*works, *map(posixpath.dirname, works)]

    def removes(self, args: list[Word], cwd: str | None, trusted: bool,
                searched: bool, text: str) -> bool:
        """Return whether ``rm args``, run in *cwd*, may take a protected directory (:meth:`takes`).

        By any text an operand may be, and by any word bash splits an unquoted one into.
        """
        bases: list[str] | None = None
        for word in args:
            if word.text.startswith("-"):
                continue
            operands = self.expansions(word)
            if not word.literal:
                operands = list(dict.fromkeys(
                    part for operand in operands for part in (operand, *operand.split())))
            for operand in operands:
                if not operand:
                    continue  # an empty word: every cleanup script has one
                # Git Bash reads ``C:\x`` and ``C:/x`` as ``/c/x``, and a ``\`` as a ``/``
                operand = bash_path(_slash_path(operand))
                if (_ANY in operand or _FOUND in operand or operand[0] == _EDITED
                        or (not word.literal and _EXTENDED.search(operand))):
                    # any value; an edit of a known directory: it, or one above it; a value
                    # that is an extended pattern
                    return True
                operand = operand.replace(_EDITED, "*")  # elsewhere: any name
                operand = operand[0] + operand[1:].replace(_HOME, self.home)
                if operand[0] in (_HOME, "/") and _HERE not in operand:
                    paths = [self.home + operand[1:] if operand[0] == _HOME else operand]
                    if (operand[0] == "/" and cwd and self.start and cwd != self.start
                            and not trusted and "PWD" in text and operand.startswith(cwd)):
                        # the scanner's ``$PWD``, had the ``cd`` failed
                        paths.append(self.start + operand[len(cwd):])
                else:
                    if bases is None:
                        bases = self.bases(cwd, trusted, searched, text)
                    paths = []
                    for base in bases:
                        made = operand.replace(_HERE, base)
                        if made[0] == _HOME:
                            made = self.home + made[1:]
                        paths.append(made if made[0] == "/" else posixpath.join(base, made))
                if any(self.takes(path, not word.literal) for path in paths):
                    return True
        return False

    def signals(self, name: str, args: list[Word]) -> bool:
        """Return whether ``kill args`` (``pkill``, ``killall``: *name*) may signal every process.

        With any of the texts each argument may be, as written and as bash splits one
        that is not literal; more combinations than the guard reads refuse the call.
        """
        choices: list[list[str | None]] = []
        count = 1
        for word in args:
            options: list[str | None] = [*self.expansions(word)]
            choices.append(options or [None])  # no word
            count *= len(choices[-1])
            if count > _MAX_ALTERNATIVES:
                return True
        for chosen in itertools.product(*choices):
            made = [(word, text) for word, text in zip(args, chosen) if text is not None]
            texts = [text for _, text in made]
            if any(_ANY in text for text in texts):
                return True
            split = [part for word, text in made
                     for part in ([text] if word.literal else text.split() or [text])]
            for read in (texts, split) if split != texts else (texts,):
                if _kills_all(read) if name == "kill" else _kills_any(read, name == "pkill"):
                    return True
        return False

    def texts(self, segments: list[str | Word], here: str) -> list[str]:
        """Return the texts the *segments* (words, and what joins them) may hand a shell.

        As written: the shell that runs the text expands it.  And as the shell that hands
        it on expands a word that is not literal first, with ``$PWD`` the directory *here*:
        one text per combination of values, :data:`_ANY` when there are more than the
        guard reads.
        """
        raw = "".join(
            part if isinstance(part, str) else _decoded(part.text) for part in segments)
        if all(isinstance(part, str) or part.literal for part in segments):
            return [raw]
        if self.opaque and any(
                not isinstance(part, str) and _expands(part) for part in segments):
            return [_ANY]
        made: list[list[str]] = [[]]
        telling, self.telling = self.telling, False
        try:
            for part in segments:
                options = [part] if isinstance(part, str) else self.expansions(part)
                if len(made) * len(options) > _MAX_ALTERNATIVES:
                    return [_ANY]
                made = [[*pieces, option] for pieces in made for option in options]
        finally:
            self.telling = telling
        expanded = ("".join(pieces).replace(_HERE, here).replace(_HOME, self.home)
                    for pieces in made)
        return list(dict.fromkeys([raw, *expanded]))

    def prints(self, argv: list[Word], here: str) -> list[str]:
        """Return the texts the command *argv* may print, where the guard reads them as commands.

        ``echo`` prints its arguments and a newline (none with ``-n``), and ``printf`` puts
        them where its format has ``%s``, reading the format again while arguments are
        left; with a backslash for ``echo``, or any other directive or escape in the
        format, the text is one the guard cannot tell (:data:`_ANY`).  Any other command
        is read as if it printed its arguments.
        """
        name = posixpath.basename(argv[0].text)
        words = argv[1:]
        if name != "printf":
            newline = "\n" if name == "echo" else ""
            while name == "echo" and words and _ECHO_OPTION.fullmatch(words[0].text):
                if "n" in words[0].text:
                    newline = ""
                words = words[1:]
            texts = self.texts([part for word in words for part in (" ", word)][1:], here)
            if name == "echo" and any("\\" in text for text in texts):
                return [_ANY]
            return [text + newline for text in texts]
        if words[:1] and words[0].text == "--":
            words = words[1:]
        if not words or words[0].text == "-v":
            return []
        if not words[0].literal or _EXACT_FORMAT.fullmatch(words[0].text) is None:
            return [_ANY]
        parts = _FORMAT_PART.split(words[0].text)[::2]
        marks = list(_FORMAT_PART.finditer(words[0].text))
        takes = any(mark[1] for mark in marks)
        segments: list[str | Word] = []
        arguments = iter(words[1:])
        left = len(words) - 1
        while True:
            for text, mark in itertools.zip_longest(parts, marks):
                segments.append(text)
                if mark is None:
                    continue
                if mark[1] is None:
                    segments.append(_FORMAT_TEXT[mark[0]])
                else:
                    segments.append(next(arguments, ""))
                    left -= 1
            if left <= 0 or not takes:
                return self.texts(segments, here)

    def strings(self, command: SimpleCommand, here: str) -> list[str]:
        """Return the texts of the here-strings *command* may read: its own, and those of the
        groups around it (``{ bash; } <<< text``, the call of a function that runs it)."""
        texts: list[str] = []
        node: SimpleCommand | Group | None = command
        while node is not None:
            texts += [text for _, operator, word in node.redirections if operator == "<<<"
                      for text in self.texts([word], here)]
            node = node.group
        return texts

    def scripts(self, command: SimpleCommand, argv: list[Word], readers: dict[int, bool],
                fed: _Feeds, here: str) -> list[str] | None:
        """Return the texts *command* hands a shell as its script; ``None`` for one nobody reads.

        A shell without ``-c`` runs its standard input: its here-string (:meth:`strings`),
        its here-document or that of a group around it (``{ bash; } <<EOF``, the call of a
        function that runs it), or what a command pipes to it.  That is what an ``echo`` or
        a ``printf`` prints (:meth:`prints`), the here-strings and the here-document a
        ``cat`` or a ``tee``
        copies, and any text (:data:`_ANY`) once another command made something else of
        text the call wrote (``tr X m <<< 'rX ..' | sh``, ``echo .. | sed .. | sh``); a
        command that reads no such text is read as if it printed its arguments.
        The scanner follows a shell's own here-document, but hands on no here-document
        the parent shell expands first.  What an ``echo`` or a ``printf`` of a group
        prints without a newline at its end goes on with what the next command of the
        group prints: a text the guard cannot tell.  So is one that the call hands a
        ``source``, whose assignments stay.
        """
        shell = id(command) in readers
        sourced, plain = (None, True) if shell else _feeds_shell(command, readers)
        if not shell and sourced is None:
            return []
        reader = command.heredoc_reader()  # itself, or a group around it
        if reader is not None and reader.heredoc_body is None and (shell or reader is command):
            return None
        if shell:
            texts = self.strings(command, here)
            if reader is not None and reader.heredoc_body:
                texts.append(reader.heredoc_body)
            if not texts and self.supplies(command, {}):
                texts.append(_ANY)  # a descriptor the call opened (``bash 0<&3``)
            return texts
        name = posixpath.basename(argv[0].text)
        writes = name in ("echo", "printf")
        if not writes and self.supplies(command, fed):
            if sourced or not plain or name not in _COPIERS:
                return [_ANY]
            return [*self.strings(command, here),
                    *([reader.heredoc_body or _ANY] if reader is not None else [])]
        texts = self.prints(argv, here)
        if writes and any(texts) and (sourced or not plain or (
                command.reader is None and not all(text.endswith("\n") for text in texts))):
            return [_ANY]
        return texts

    def handed(self, command: SimpleCommand, fed: _Feeds, here: str) -> list[Word]:
        """Return the texts ``xargs`` (*command*) reads that the Bash call itself writes:
        more arguments of the program it runs, or what it puts in them (:meth:`judges`).

        Its here-strings and its here-document, or those of a group around it, and what
        the commands whose output reaches it through a pipe write (*fed*, :meth:`wrote`).
        """
        texts = self.strings(command, here)
        reader = command.heredoc_reader()
        if reader is not None:
            texts.append(reader.heredoc_body or _ANY)
        node: SimpleCommand | Group | None = command
        while node is not None:
            for writer, plain in fed.get(id(node), ()):
                texts += self.wrote(writer, plain, here)
            node = node.group
        if not texts and (self.rewired or self.supplies(command, fed)):
            texts.append(_ANY)  # a descriptor the call opened: text the guard does not follow
        return [Word(text, False) for text in texts]

    def wrote(self, command: SimpleCommand, plain: bool, here: str) -> list[str]:
        """Return the texts the Bash call itself writes that *command* hands what reads it.

        What an ``echo`` or a ``printf`` prints; the here-strings and the here-document a
        ``cat`` or a ``tee`` copies; nothing for a program that reads no such text (it
        hands on what it found); any text once a command made something else of text the
        call wrote: another program that reads it, or one on the way (not *plain*).
        """
        argv = command.argv()
        name = posixpath.basename(argv[0].text) if argv else ""
        if name in ("echo", "printf"):
            texts = self.prints(argv, here)
        elif not self.supplies(command, {}):
            return []
        elif name in _COPIERS:
            reader = command.heredoc_reader()
            texts = self.strings(command, here)
            if reader is not None:
                texts.append(reader.heredoc_body or _ANY)
        else:
            return [_ANY]
        return texts if plain or not any(texts) else [_ANY]

    def passed(self, words: list[Word], here: str | None) -> tuple[list[frozenset[str]], bool]:
        """Return the values each word bash makes of the argument *words* may pass, and
        whether each of them is one word for sure.

        Not a word that may be empty, or that holds a blank or a glob, unless it is
        literal: unquoted, bash drops it, splits it or matches file names with it, and
        the parameters after it move.  Nor all the parameters or all the elements of an
        array.  ``$PWD`` is the directory *here* of the caller.
        """
        sets: list[frozenset[str]] = []
        exact = True
        for word in words:
            parts = self.words(word)
            exact = exact and len(parts) == 1 and (
                word.literal or _SEVERAL.search(word.text) is None)
            for texts in parts:
                values = frozenset(_placed(texts, here))
                exact = exact and (word.literal or not any(
                    not value or _SPLIT.search(value) for value in values))
                sets.append(values)
        return sets, exact

    def calls(self, frame: _Frame, sets: list[frozenset[str]], exact: bool,
              zero: frozenset[str | None], child: bool) -> None:
        """Start the positional parameters of a text or a call that *frame* covers.

        *sets* are the values of each (:meth:`places`), *zero* those of ``$0``.  A *child*
        is another shell: it gets the variables as a program its parent starts
        (:class:`_Scope`).
        """
        frame.prefix = self.prefix
        if child:
            frame.scope = self.values
            self.values = _Scope(self.values, True)
        self.serial += 1
        self.prefix = f"{self.serial}:"
        self.store(self.prefix + "0", zero if exact else zero.union(*sets))
        self.store(self.prefix + "#", {"0"})
        self.places(sets, exact)

    def places(self, sets: list[frozenset[str]], exact: bool) -> None:
        """Set the positional parameters of the text being read to the values *sets*.

        One by one while their positions are *exact* (:meth:`passed`) and those before
        were (``<prefix>#`` counts them).  Else each parameter is any of them, a word bash
        splits one into, or none (not when a program printed one: its words are unknown,
        and harmless).
        """
        prefix = self.prefix
        every: frozenset[str | None] = frozenset().union(*sets)
        counts = self.values[prefix + "#"]
        self.store(prefix + "@", every or _UNSET)
        if exact and None not in counts and self.values[prefix + "*"] == _UNSET:
            for number, values in enumerate(sets, 1):
                self.store(prefix + str(number), values)
            for number in range(len(sets) + 1, max(int(count or 0) for count in counts) + 1):
                self.store(prefix + str(number), _UNSET)
            self.store(prefix + "#", {str(len(sets))})
            return
        every |= {part for value in every if value for part in value.split()}
        printed = any(value and _SOME in value for value in every)
        self.store(prefix + "*", every if printed else every | _UNSET)

    def shifts(self, argv: list[Word]) -> None:
        """Move the positional parameters of the text being read as ``shift`` (*argv*) does.

        By its number while their positions are known and the command runs once; else
        each parameter is any of them, or none.
        """
        prefix = self.prefix
        if self.values[prefix + "*"] != _UNSET:
            return
        counts = self.values[prefix + "#"]
        count = next(iter(counts))
        by = argv[1].text if len(argv) == 2 else "1" if len(argv) == 1 else ""
        if count is None or len(counts) != 1 or not by.isdigit() or self.late:
            self.store(prefix + "*", self.elements() | _UNSET)
            return
        total, moved = int(count), int(by)
        if moved > total:
            return
        for number in range(1, total + 1):
            self.store(prefix + str(number), self.values[prefix + str(number + moved)]
                       if number + moved <= total else _UNSET)
        self.store(prefix + "#", {str(total - moved)})

    def strays(self, args: list[Word]) -> None:
        """Record a call, with *args*, of a function that the scanner did not follow.

        Not one in the body of a definition while no call outside the definitions strays:
        that body then runs through the calls the scanner follows only, each read with its
        own arguments (and two functions would hand each other's words on without end).
        """
        if self.defining and not self.strayed:
            return
        passed = {part for word in args for text in self.expansions(word)
                  for part in (text, *text.split())}
        if not (self.strayed and passed <= self.loose):
            self.strayed = self.grown = True
            self.loose |= passed

    def environment(self, command: SimpleCommand, argv: list[Word],
                    here: str | None) -> list[tuple[str, frozenset[str | None]]]:
        """Return what *command* puts in the environment of the program *argv* it runs.

        The assignments before its command word (``D=x bash -c ...``, ``env D=x sh``).
        """
        pairs: list[tuple[str, frozenset[str | None]]] = []
        for word in command.words[:len(command.words) - len(argv)]:
            assignment = _ASSIGNMENT.fullmatch(word.text)
            if assignment is not None:
                pairs.append((assignment[1], frozenset(
                    self.valued(Word(assignment[4], word.literal), here))))
        return pairs

    def opens(self, group: Group, caller: SimpleCommand | None, body: bool,
              here: str | None) -> None:
        """Open the frame of *group*, whose commands the walk reaches.

        The text or the function's body a *caller* runs starts with the caller's words
        as its positional parameters and what the caller exports; a shell's text, a
        subshell and a substitution run in a shell of their own.  The *body* of a
        function's definition runs when something calls it, with parameters nobody
        knows.  A compound command and a group in a pipeline or after ``||`` may not
        run; a loop runs again.
        """
        frame = _Frame(group)
        if caller is not None:
            argv = caller.argv()
            name = posixpath.basename(argv[0].text) if argv else ""
            exports = self.environment(caller, argv, here)
            if name in SHELLS:
                index = shell_script_index([word.text for word in argv[1:]])
                handed = argv[index + 2:] if index is not None else []
                sets, exact = self.passed(handed, here)
                self.frames.append(frame)
                self.calls(frame, sets[1:], exact, sets[0] if sets else _UNKNOWN, True)
                if self.opaque and any(_expands(word) for word in handed):
                    self.store(self.prefix + "*", _ANYTHING)  # words the guard does not model
                key = argv[index + 1].text if index is not None else caller.heredoc_body or ""
                frame.apart = (key, self.opaque)
                self.opaque = key in self.shaded
                if index is None:
                    # a here-document is its script: the scanner shows its commands after
                    # those written after it, and no one reads its operands
                    frame.late = 1
                    self.late += 1
                    self.delayed = True
                    if _operated(argv):
                        self.store(self.prefix + "*", _ANYTHING)
            elif name in ("eval", "source", "."):
                self.frames.append(frame)
                frame.merges = bool(exports) or _aside(caller)
            else:
                sets, exact = self.passed(argv[1:], here)
                zero = self.parameter("0")
                self.frames.append(frame)
                frame.function, frame.merges = True, bool(exports) or _aside(caller)
                self.calls(frame, sets, exact, zero, False)
            for exported, values in exports:
                self.store(exported, values)
            return
        self.frames.append(frame)
        if group.kind in _SUBSHELLS or (body and not self.strayed):
            frame.scope = self.values
            self.values = _Scope(self.values)
        if body:
            frame.late, frame.prefix = 2, self.prefix
            self.defining += 1
            self.delayed = True
            self.serial += 1
            self.prefix = f"{self.serial}:"
            self.store(self.prefix + "*", _UNKNOWN | self.loose | (
                _UNSET if self.strayed else frozenset()))
        elif group.kind in _LOOPS:
            frame.late = 1
            self.late += 1
            self.delayed = True
        frame.header = group.kind in ("for", "select")
        frame.branches = group.kind in ("if", "case") and self.forked
        frame.merges = group.kind != "{" or bool(
            group.conditional or group.piped or group.reader is not None
            or group.background or group.asynchronous)

    def leaves(self, depth: int) -> None:
        """End the frames above the first *depth* ones, the last one opened first.

        A frame whose commands ran in a shell of their own leaves the variables as they
        were.  Any other leaves a variable as its commands assigned it, or (a frame that
        :attr:`_Frame.merges`, a function's own variable) with that or what it held before.
        """
        while len(self.frames) > depth:
            frame = self.frames.pop()
            if frame.apart is not None:
                if self.opaque:
                    self.shaded.add(frame.apart[0])
                self.opaque = frame.apart[1]
            if frame.late:
                self.late -= frame.late == 1
                self.defining -= frame.late == 2
            if frame.prefix is not None:
                self.prefix = frame.prefix
            if frame.scope is not None:
                self.values = frame.scope
                continue
            outer = self.frames[-1]
            merges = frame.merges or self.branched()
            for name, before in frame.saved.items():
                wrote = frame.wrote[name]
                if frame.closed and frame.ends is not None:
                    held = frame.ends.get(name, before) | self.values[name]
                elif merges or name in frame.own:  # it may have ended after any command
                    held = before | wrote
                else:
                    held = self.values[name]
                self.values[name] = held if len(held) <= _MAX_ALTERNATIVES else _ANYTHING
                outer.saved.setdefault(name, before)
                outer.wrote[name] = outer.wrote.get(name, frozenset()) | wrote

    def turns(self, group: Group | None, last: bool) -> None:
        """End a branch of the ``if`` or the ``case`` *group*, whose commands are being read.

        The next branch starts with what each variable held before the group or was
        assigned in it: the conditions before it ran, and a branch may fall through.
        *last*: an ``else`` follows, so one branch runs for sure unless the group may not
        run or runs aside (:attr:`_Frame.closed`).
        """
        frame = self.frames[-1]
        if (group is None or frame.group is not group or frame.listing or frame.branches
                or group.kind not in ("if", "case")):
            return
        first = frame.ends is None
        ends = frame.ends = frame.ends or {}
        for name, before in frame.saved.items():
            ends[name] = ends.get(name, frozenset() if first else before) | self.values[name]
            held = before | frame.wrote[name]
            self.values[name] = held if len(held) <= _MAX_ALTERNATIVES else _ANYTHING
        frame.closed = last and not (
            self.orred or group.piped or group.reader is not None or group.background
            or group.asynchronous)

    def arrives(self, command: SimpleCommand, base: int, callers: dict[int, SimpleCommand],
                certain: bool, here: str | None) -> None:
        """Bring the frames above the first *base* ones in line with where *command* runs.

        The frames of the groups it is not in end; those of the groups it is in open
        (:meth:`opens`; *callers* maps the group of a text to the command that runs it).
        At each level a ``&&`` list is a frame of its own: what a command after ``&&``
        assigns holds for the rest of the list, and may not hold after it; a command
        that may follow ``||`` ends the list before it, and is a list by itself.  In a
        text that is *certain* to hold no ``||`` before a brace group, a brace group that
        may not run is the body of a function's definition.
        """
        chain: list[SimpleCommand | Group] = [command]
        while chain[-1].group is not None:
            chain.append(chain[-1].group)
        chain.reverse()
        depth, position = 0, base
        while position < len(self.frames):
            frame = self.frames[position]
            if not frame.listing:
                if frame.group is not chain[depth]:
                    break
                depth += 1
            position += 1
        self.leaves(position)
        for node in chain[depth:]:
            caller = callers.get(id(node))
            kind = node.kind if isinstance(node, Group) else None
            if caller is None and kind not in _SUBSTITUTIONS:
                fails = node.conditional and kind in (None, "{")
                if self.frames[-1].listing and (self.frames[-1].tentative or not node.after_and):
                    self.leaves(len(self.frames) - 1)
                if self.frames[-1].listing:
                    self.frames[-1].tentative = fails
                elif node.after_and or fails:
                    self.frames.append(_Frame(node.group, True, fails, True))
            if isinstance(node, Group):
                outer = node.group
                body = (certain and caller is None and node.kind == "{" and node.conditional
                        and (outer is None or outer.kind in _LOOPS or (
                            outer.kind in ("(", "{") and not outer.conditional
                            and id(outer) not in callers)))
                self.opens(node, caller, body, here)

    def reads(self, script: str, cwd: str | None, depth: int, kind: str,
              arguments: list[Word] | None, here: str | None, late: bool = False,
              alone: bool = True) -> bool:
        """Return whether the text *script*, run in *cwd* as a command hands it on, ends the task.

        *kind* tells who runs it: a ``shell`` of its own, with *arguments* as its ``$0``,
        ``$1``... (``None``: any value); ``eval``, in the shell that reads it; a ``trap``,
        later in that shell.  *late*: it may run before commands the walk has passed.
        Not *alone*: it is one of the texts its command may run, so it may not run.
        """
        mark, weak, exports = len(self.frames), self.weak, self.exports
        self.weak = False
        # a shell of its own builds its words its own way, whatever its caller did
        opaque, apart = self.opaque, kind == "shell" and not late
        if apart:
            self.opaque = script in self.shaded
        frame = _Frame(None, merges=kind == "trap" or weak or bool(exports) or not alone)
        if kind == "shell":
            sets, exact = self.passed(arguments or [], here)
            self.frames.append(frame)
            self.calls(frame, sets[1:], exact, sets[0] if sets else _UNKNOWN, True)
            if arguments is None:
                self.store(self.prefix + "*", _ANYTHING)
        else:
            self.frames.append(frame)
        if kind == "trap" or late:
            frame.late = 1
            self.late += 1
            self.delayed = True
        if kind != "trap":
            for name, values in exports:
                self.store(name, values)
        try:
            return self.refuses(script, cwd, depth + 1)
        finally:
            self.leaves(mark)
            self.weak, self.exports = weak, exports
            if apart:
                if self.opaque:
                    self.shaded.add(script)
                self.opaque = opaque

    def program(self, name: str, args: list[Word], cwd: str | None, trusted: bool,
                searched: bool, text: str, depth: int, followed: bool) -> bool:
        """Return whether the program *name*, run in *cwd* with *args*, ends the task.

        ``rm``, ``kill``, ``pkill`` and ``killall`` by their arguments; a shell's ``-c``,
        ``eval`` and ``trap`` by the texts they may run (:meth:`texts`), unless the scanner
        already *followed* the text.  The words after a ``-c`` text are its ``$0``, ``$1``...
        An ``alias`` by whether its text mentions what the guard reads (:func:`_mentions`).
        """
        if name in self.functions and not followed:
            self.strays(args)
        if name in ("killall5", "enable"):
            return True  # ``enable`` turns a builtin the guard reads into a program
        if name == "hash" and any(
                not word.literal or (word.text.startswith("-") and "p" in word.text)
                for word in args):
            return True  # ``hash -p /bin/rm name``: a name of its own for any program
        if name in ("mapfile", "readarray") and any(
                not word.literal or (word.text.startswith("-") and "C" in word.text)
                for word in args):
            return True  # ``mapfile -C command``: bash runs it with each line it reads
        if name == "rm" and any(self.patterned(word) for word in args):
            return True
        if self.opaque and name in ("rm", "kill", "pkill", "killall") and any(
                _expands(word) for word in args):
            return True
        if name == "rm":
            return self.removes(args, cwd, trusted, searched, text)
        if name in ("kill", "pkill", "killall"):
            return self.signals(name, args)
        if name == "alias":  # its text runs later, before words the guard does not join to it
            texts = [made for word in args for made in self.expansions(word)]
            names = {made.partition("=")[0] for made in texts if "=" in made}
            if not names <= self.aliases:
                self.aliases |= names
                self.grown = self.delayed = True  # a command read before may be one of them
            return any(_ANY in made or _mentions(made.partition("=")[2]) for made in texts)
        kind = name if name in ("eval", "trap") else "shell"
        arguments: list[Word] = []
        if name in SHELLS and any(
                value and (_ANY in value or value.startswith(("/dev/", "/proc/")))
                for values in (self.holds("BASH_ENV"), *(
                    held for key, held in self.exports if key == "BASH_ENV"))
                for value in values):
            return True  # its startup file is a stream: text the call may write
        if name in SHELLS:
            index = shell_script_index([word.text for word in args])
            if index is None:
                # without a script file it reads the input an ``exec`` set: text of the call
                return self.rewired and all(
                    word.text.startswith(("-", "+")) or word.text == "posix" for word in args)
            segments: list[str | Word] = [args[index]]
            arguments = args[index + 1:]
        elif name == "eval":  # the shell expands the words, then reads them as commands
            segments = [part for word in args for part in (" ", word)][1:]
        elif name == "trap":
            segments = list[str | Word](
                word for word in args if not word.text.startswith("-"))[:1]
        else:
            return False
        if followed:
            return False
        here = (cwd if trusted else self.start) or _HERE  # ``$PWD`` is the caller's
        scripts = [script for script in self.texts(segments, here) if script]
        # the words a shell the guard does not model makes of them are any parameters
        loose = self.opaque and any(_expands(word) for word in arguments)
        return any(
            self.reads(script, cwd if trusted else None, depth, kind,
                       None if loose else arguments, here, alone=len(scripts) == 1)
            for script in scripts)

    def runs(self, argv: list[Word], cwd: str | None, trusted: bool, searched: bool,
             text: str, depth: int, followed: bool) -> bool:
        """Return whether a program the command *argv* may run in *cwd* ends the task.

        The command word may be several texts, each split as bash splits an unquoted
        value (``CMD="rm -rf $HOME"; $CMD``); one the guard cannot bound may be a whole
        command line, or ``rm``, ``kill`` or ``pkill`` when a path ends it (``"$BIN/python"``:
        one word, unless a directory's name holds blanks).  The other words a brace
        expansion makes of it are its first arguments (``{rm,-rf,/}``); of all the
        positional parameters (``"$@"``) or all the elements of an array, each may be the
        program and all are its arguments.
        """
        if (self.opaque and _expands(argv[0])) or self.patterned(argv[0]):
            return True
        parts = self.words(argv[0])
        rest = [Word(made, False) for texts in parts[1:] for made in texts]
        if not argv[0].literal and _SEVERAL.search(argv[0].text):
            rest = [*(Word(made, False) for made in parts[0]), *rest]
        if not argv[0].literal and re.search(r"\$\{[@*][^}]|\[[@*]\][^}]", argv[0].text):
            return True  # a slice of the parameters or of an array: any of them
        for first in parts[0]:
            if not argv[0].literal:
                first = _QUOTES.sub("", first)  # of a default word: bash removes them
            if _ANY in first and not first.rpartition(_ANY)[2].startswith("/"):
                return True
            split = first.split()
            names = ("rm", "kill", "pkill") if _ANY in first else (
                _named(split[0]),) if split else ()
            args = [*(Word(made, False) for made in split[1:]), *rest, *argv[1:]]
            if any(self.program(name, args, cwd, trusted, searched, text, depth, followed)
                   for name in names):
                return True
        return False

    def judges(self, command: SimpleCommand, argv: list[Word], cwd: str | None, trusted: bool,
               searched: bool, text: str, depth: int, extra: list[Word]) -> bool:
        """Return whether the simple *command*, run in *cwd*, ends the task by what it runs.

        *argv* is its command word and arguments (:meth:`SimpleCommand.argv`), not empty.
        A command word that may be empty (``$SUDO rm -rf /``) leaves the words after it
        as the command: they are read as those of a runner.  *extra* are the texts a
        runner reads (:meth:`handed`): more arguments of its program, and each line of them
        in place of the text the runner may replace (:func:`_placeholders`); more lines
        than the guard reads are any text.
        """
        words = [word for word in command.words if not word.assignment]
        runner = posixpath.basename(words[0].text) in _RUNNERS
        if not runner:
            if self.runs(argv, cwd, trusted, searched, text, depth, command.script is not None):
                return True
            made = self.words(argv[0])[0]
            if argv[0].literal or (made and all(text.strip() for text in made)):
                return False
            words = argv
        if self.starts(words, depth):
            return True
        moved = texts = False
        stays: set[int] = set()  # the words before a runner's option that changes directory
        for word in words:
            moved = moved or word.text.startswith(_CHDIR_OPTIONS)
            if not moved:
                stays.add(id(word))
            if texts and " " in word.text and self.reads(
                    _decoded(word.text), None, depth, "shell", [], None):
                return True
            texts = texts or posixpath.basename(word.text) in _TEXT_RUNNERS
        lines = sorted({line.strip() for word in extra for line in word.text.splitlines()} - {""})
        if len(lines) > _MAX_ALTERNATIVES:
            lines = [_ANY]
        marks = _placeholders(words) if extra else ()
        for program in _programs(words, argv, not runner):
            # the text the scanner followed had none of the words the runner adds
            followed = program[0] is argv[0] and command.script is not None and not extra
            stays_put = trusted and id(program[0]) in stays
            if self.runs([*program, *extra], cwd, stays_put, searched, text, depth, followed):
                return True
            for mark in marks:
                # a text the scanner followed was the one before the lines were put in
                if any(mark in word.text for word in program) and any(
                        self.runs([Word(word.text.replace(mark, line), word.literal)
                                   for word in program], cwd, stays_put, searched, text, depth,
                                  False) for line in lines):
                    return True
        return False

    def starts(self, words: list[Word], depth: int) -> bool:
        """Return whether a runner's *words* end the task by what only their values tell.

        A word of ``env`` that may be its ``-S`` option (:data:`_SPLITS`): the text it
        splits is a command line, read as one.  A word given with a shell that may be a
        variable the shell runs at its start (:data:`_STARTED`).
        """
        shell = any(_named(word.text) in SHELLS for word in words[1:])
        splits = posixpath.basename(words[0].text) == "env"
        for index, word in enumerate(words[1:], 1):
            made = [word.text] if word.literal else self.expansions(word)
            if shell and any(_STARTED.match(text) for text in made):
                return True
            for option in filter(None, map(_SPLITS.fullmatch, made)) if splits else ():
                rest = option[1] or option[2]
                lines = [rest] if rest else self.texts([*words[index + 1:index + 2]], _HERE)
                if any(self.reads(line, None, depth, "shell", [], None)
                       for line in lines if line):
                    return True
        return False

    def valued(self, word: Word, here: str | None) -> list[str]:
        """Return the values the word *word* may assign: what a program prints is unknown
        (:data:`_SOME`), ``$PWD`` is the directory *here*, and a ``~`` at its start or after
        a ``:`` is as written (it was quoted) or the directory it names."""
        texts = [word.text] if word.literal else self.expanded(word.text, _SOME)
        tilded = _ASSIGNED_TILDE.sub(self.tilde, word.text)
        if tilded != word.text:
            texts = [*texts, *self.expanded(tilded, _SOME)]
        return list(dict.fromkeys(_placed(texts, here)))

    def array(self, name: str, text: str, here: str | None) -> frozenset[str | None]:
        """Return the elements the array assignments of *name* in *text* may hold.

        The scanner does not read the words of ``name=(...)``, so they are read from the
        text: those of every such assignment (the walk cannot tell which one a command
        is).  Any value when the guard cannot read one: an element by its index, or a
        reserved word the scanner reads as the start of a compound command.
        """
        self.arrays.add(name)
        values: set[str | None] = set()
        opened = False
        for start in re.finditer(r"(?<![\w$])" + name + r"\+?=\(", text):
            body = _ARRAY_BODY.match(text, start.end())
            if body is None:
                return _ANYTHING
            opened = True
            for command in self.scan(body[1], None):
                group = command.group
                while group is not None and group.kind not in _SUBSTITUTIONS:
                    group = group.group
                if group is not None:
                    continue  # a command of a substitution: what it prints is a word above
                if command.group is not None:
                    return _ANYTHING
                for word in command.words:
                    if word.text.startswith("[") and "]=" in word.text:
                        return _ANYTHING
                    values.update(_placed(self.expansions(word, _SOME), here))
        return frozenset(values) or _UNSET if opened else _ANYTHING

    def assigns(self, command: SimpleCommand, name: str, here: str | None, text: str,
                fed: _Feeds, header: bool) -> None:
        """Record what *command*, which runs the program *name* in *here*, assigns.

        The *header* of a ``for`` or a ``select`` makes its variable one of the words it
        loops over, an empty one too, or one of the positional parameters (``for a; do``);
        in a text with an arithmetic ``for``, whose header the scanner does not show, it
        may be the first command of that loop instead.  The operands of ``read``
        and its like are what a program wrote (unknown), or any value when the Bash call
        itself writes what they read (:func:`_written_input`, ``printf -v``).  ``unset``
        leaves a variable unset.  An assignment is read as bash expands it
        (:meth:`valued`, :meth:`array`); one before a program is that program's
        environment.  A declaration without a value may leave the variable unset, and
        makes it the function's own.
        """
        names = [word.text for word in command.words]
        group = command.group
        for word in command.argv():
            element = _ELEMENT.match(word.text)
            if element is None:
                break
            self.store(element[1], _ANYTHING)  # by a subscript the scanner does not read
        if name not in _SPLITTERS and any(
                word.assignment and word.text.startswith(("IFS=", "IFS+="))
                for word in command.words):
            self.shades()  # a field separator that a function, a text or a builtin may keep
        if name in _LOCALS and any(
                not word.literal or (word.text[:1] in "-+" and set("nluc") & set(word.text))
                for word in command.words if _ASSIGNMENT.fullmatch(word.text) is None):
            # a name reference: what is assigned to it goes to another variable; a
            # variable whose values change case
            self.shades()
        if not self.referenced and name in _LOCALS and any(
                not word.literal or (word.text[:1] in "-+" and "n" in word.text)
                for word in command.words if _ASSIGNMENT.fullmatch(word.text) is None):
            self.referenced = self.grown = self.delayed = True
        if name in _LOCALS and any(option[:1] == "-" and "i" in option for option in names[1:]):
            # an integer: bash evaluates what the call assigns it, here and wherever it does
            for word in command.argv()[1:]:
                if word.text[:1] not in "-+":
                    self.counts([word.text.partition("=")[0].rstrip("+") + "=0",
                                 *self.expansions(word)])
        if (header and group is not None and names[0].isidentifier()
                and (len(names) == 1 or names[1] == "in")):
            looped: set[str | None] = {
                made for word in command.words[2:]
                for made in _placed(self.expansions(word, _SOME), here)
            } if len(names) > 1 else set(self.elements())
            if group.kind == "select":
                looped.add("")
            weak, self.weak = self.weak, self.weak or self.counted
            self.store(names[0], looped or _UNKNOWN)
            self.weak = weak
            if not self.counted:
                return
        if name in _SPECIAL:  # what is assigned before one stays, in POSIX mode
            weak, self.weak = self.weak, True
            for word in command.words:
                prefixed = _ASSIGNMENT.fullmatch(word.text) if word.assignment else None
                if prefixed is not None:
                    self.store(prefixed[1], self.valued(Word(prefixed[4], word.literal), here))
            self.weak = weak
        if name in _ASSIGNERS:
            read = _ANYTHING if (
                name == "printf" or self.rewired or self.supplies(command, fed)
                or any(option.startswith("-u") for option in names[1:])) else _UNKNOWN
            targets = names[1:] if name != "printf" else names[names.index("-v") + 1:][:1] if (
                "-v" in names) else []
            for target in (*targets, "REPLY", "MAPFILE") if name != "printf" else targets:
                target = target.partition("[")[0]  # an element of it (``read 'A[1]'``)
                if target.isidentifier():
                    self.store(target, read)
            if name == "getopts":
                self.store("OPTARG", self.elements() | _UNSET)
            return
        if name == "unset":  # not with ``-f`` (a function), and a readonly variable stays
            kept = frozenset[str | None]()
            stays = self.readonly or any(  # ``unset -n NAME`` keeps what is no name reference
                option.startswith("-") and "n" in option for option in names[1:])
            for target in names[1:]:
                if target.startswith("-") and "f" in target:
                    return
                if target.isidentifier():
                    self.store(target, (self.values[target] if stays else kept) | _UNSET)
            return
        if name and name not in _DECLARERS:
            return
        for word in command.words:
            assignment = _ASSIGNMENT.fullmatch(word.text)
            if assignment is None:
                if name in _LOCALS and word.text.isidentifier() and not word.assignment:
                    self.store(word.text, self.values[word.text] | _UNSET)
                    self.owns(word.text)
                continue
            if not (word.assignment or name in _DECLARERS):
                continue
            target, subscript, added, value = assignment.groups()
            before = self.values[target]
            # ``NAME=(``...: the scanner keeps the name only, or the text as written
            listed = not word.literal and (not value or value.startswith("("))
            values: frozenset[str | None] = (
                self.array(target, text, here) if listed
                else frozenset(self.valued(Word(value, word.literal), here)))
            if added and not listed:
                joined = {(old or "") + (new or "") for old in before for new in values}
                values = frozenset(joined) | (before if target in self.arrays else frozenset())
            elif added or subscript:
                values |= before
            self.store(target, values)
            if name in _LOCALS:
                self.owns(target)

    def owns(self, name: str) -> None:
        """Make the variable *name* the own one of the function whose call is being read."""
        for frame in reversed(self.frames):
            if frame.function:
                frame.own.add(name)
                return

    def defaults(self, command: SimpleCommand, text: str) -> None:
        """Assign what the ``${NAME:=word}`` expansions *command* expands assign.

        Those of its words and of its redirections, as bash expands them.  Those of a
        here-document its shell expands for it are in *text* where the scanner does not
        show: every one of *text* then may have assigned.
        """
        for word in (*command.words, *(word for _, _, word in command.redirections)):
            if not word.literal and _DEFAULT_START.search(word.text):
                self.expansions(word)
                self.nested(word.text)
        reader = command.heredoc_reader()
        if reader is not None and reader.heredoc_body is None:
            weak, self.weak = self.weak, True
            for match in _DEFAULTED.finditer(text):
                self.expanded(match[0])
            self.nested(text)
            self.weak = weak

    def quoted(self, text: str, depth: int, spelled: bool) -> bool:
        """Return whether a command substitution the quoted word *text* holds ends the task.

        The shell that runs the call leaves it as written, and something may expand it
        later: a prompt under ``set -x``, arithmetic, ``${X@P}``, an ``eval``.
        """
        # *spelled*: by a ``printf``, which makes any command of a directive or an escape
        return any((spelled and ("%" in one or "\\" in one)) or (
            _mentions(one) and self.reads(one, None, depth, "shell", None, None, True))
                   for one in _held(text))

    def nested(self, text: str) -> None:
        """Assign any value to the variables a ``${NAME:=word}`` of *text* assigns that nests
        deeper than the guard reads (:data:`_DEFAULT_START`)."""
        read = {match.start() for match in _DEFAULTED.finditer(text)}
        for start in _DEFAULT_START.finditer(text):
            if start[1]:  # ``${!NAME:=word}`` assigns the variable NAME names
                self.shades()
            elif start.start() not in read or text.count("${", 0, start.start()) - text.count(
                    "}", 0, start.start()) > 1:  # deeper than an expansion is read
                self.store(start[2], _ANYTHING)

    def target(self, command: SimpleCommand, argv: list[Word],
               entered: str | None) -> tuple[str | None, bool]:
        """Return the directory the ``cd`` *command* (*argv*) enters where the scanner cannot
        tell, and whether its operand alone names it.

        An operand the scanner cannot read, as the guard reads it (:meth:`expansions`):
        below the home directory (``~/proj``), absolute, or below a value the command
        computes (``cd "$tmp"``), which is a directory of an unknown name.  A literal one
        without ``..`` below such a directory (*entered*).  ``None`` for what only the
        shell knows, and for a ``cd`` that may not run (it follows ``||``).
        """
        base = command.cwd if command.cwd is not None else entered
        texts = self.expansions(argv[1]) if len(argv) == 2 and not command.conditional else []
        text = texts[0] if len(texts) == 1 else ""
        if (not text or text.startswith("-") or _GLOB.search(text) is not None
                or _HERE in text or _EDITED in text or _ANY in text or _FOUND in text
                or ".." in text.split("/") or (self.opaque and _expands(argv[1]))
                or (argv[1].literal and (command.cwd is not None or text.startswith("/")))):
            return None, False
        if text[0] == _HOME:
            return posixpath.normpath(self.home + text[1:]), True
        if text[0] in ("/", _SOME):
            return posixpath.normpath("/" + text), True
        return (posixpath.normpath(posixpath.join(base, text)) if base else None), False

    def scan(self, text: str, cwd: str | None) -> list[SimpleCommand]:
        """Return the simple commands of *text* run in *cwd*, scanned once for every reading.

        Raises:
            TooDeepError: Its groups nest too deep to read.
        """
        key = (text, cwd)
        if key not in self.scans:
            self.scans[key] = simple_commands(text, cwd, isdir=self.directory, derived=True)
        return self.scans[key]

    def decides(self, text: str) -> bool:
        """Return whether the Bash call *text*, run where every call starts, ends the task.

        The call has no positional parameters (``$1`` is empty).  It is read again while
        its loops, traps and definitions (:attr:`delayed`) read values that later commands
        assign (:attr:`ever`), at most :data:`_MAX_PASSES` times; then those values are
        any value.  From the :data:`_WIDENED` th reading on, a variable that still gets a
        new value holds any value, so the readings end.

        Raises:
            TooDeepError: Its groups, or its texts within texts, nest too deep to read.
        """
        bare = _QUOTES.sub("", text)
        if _STARTUP.search(bare):
            return True
        self.opaque = self.armed = False
        self.shaded = set()
        self.rewired = _EXEC_INPUT.search(bare) is not None
        self.readonly = _READONLY.search(bare) is not None
        for attempt in range(_MAX_PASSES + 1):
            self.widening = attempt >= _WIDENED
            self.unsettled = attempt == _MAX_PASSES
            self.values = _Scope(None)
            self.frames = [_Frame(None)]
            self.prefix, self.serial = "0:", 0
            self.late = self.defining = 0
            self.grown = self.delayed = self.weak = False
            self.exports = []
            self.values["0:0"], self.values["0:#"] = frozenset({"bash"}), frozenset({"0"})
            if self.refuses(text, self.start) or self.armed:
                return True
            if not (self.grown and self.delayed):
                break
        return False

    def refuses(self, text: str, cwd: str | None, depth: int = 0) -> bool:
        """Return whether the command text *text*, run in *cwd*, ends the task.

        The directory the scanner gives a command assumes every ``cd`` before it succeeded.
        It is trusted only while each command since the ``cd`` is linked to it
        (:meth:`walks`), so the command runs only when the ``cd`` did, and nothing the
        scanner does not show stands in between (:data:`_UNSEEN`).  A text that holds a
        value the guard cannot bound is refused.

        Raises:
            TooDeepError: Its groups, or its texts within texts, nest too deep to read.
        """
        if depth > _MAX_TEXTS:
            raise TooDeepError
        if _ANY in text or _EDITED in text:
            return True
        if "\\'" in text:
            text = _QUOTING.sub(_requoted, text)
        if depth and _STARTUP.search(text):
            return True
        if depth and not self.rewired and _EXEC_INPUT.search(text):
            # a text the call puts together: the commands before it are read again
            self.rewired = True
            self.grown = self.delayed = True
        text = _modern(_LOCALE.sub('"', text))
        commands = self.scan(text, cwd)
        known = len(self.functions)
        # what bash reads as code here: the text, and the texts the scanner expanded and followed
        code = [text, *filter(None, (command_text(command.argv()) for command in commands
                                     if command.script is not None))]
        self.functions.update(_coded(_defined, code, commands))
        if "command_not_found_handle" in self.functions:
            return True  # every command the shell does not find runs it
        self.reckons([*_evaluated(commands), *(
            region for coded in code for region in self.commanded(coded))], commands)
        if depth and len(self.functions) > known:
            self.grown = True  # a command read before may have called one of them
        bare = text
        for command in commands:
            if command.heredoc_body:
                bare = bare.replace(command.heredoc_body, "")
        bare = _QUOTING.sub("", bare)
        unseen = _UNSEEN.search(_FOR_HEADER.sub("for ", bare)) is not None
        certain = _OR_GROUP.search(bare) is None
        flags = self.forked, self.marked, self.counted, self.orred, self.written
        self.written = text
        # anywhere in the text: the scanner follows the texts it quotes too
        self.forked = _BRANCH_END.search(text) is not None
        self.marked = False
        self.counted = _FOR_HEADER.search(text) is not None
        self.orred = _OR_IF.search(text) is not None
        if self.forked and _ELSE not in text and _BRANCH not in text:
            # the same commands, with one more where each branch ends
            marked = self.scan(_BRANCH_END.sub(_branch_mark, text), cwd)
            if [command.words for command in marked if not _marks(command)] == [
                    command.words for command in commands]:
                commands, self.forked, self.marked = marked, False, True
        readers = _shell_readers(commands)
        callers = {id(command.script): command for command in commands
                   if command.script is not None}
        fed: _Feeds = {}
        for command in commands:
            for node, plain in _receivers(command) if command.words else ():
                fed.setdefault(id(node), []).append((command, plain))
        base = len(self.frames)
        entered = self.inherits, self.callers, self.early
        self.inherits, self.callers = self.inlet, callers
        try:
            return self.walks(commands, unseen, readers, callers, certain, fed, text, depth)
        finally:
            self.leaves(base)
            self.forked, self.marked, self.counted, self.orred, self.written = flags
            self.inlet = self.inherits
            self.inherits, self.callers, self.early = entered

    def settled(self, key: str, made: list[str]) -> None:
        """Read the call again when the texts *made* of the expression *key* are new: what its
        variables hold later in the call is known only then."""
        if self.reckonings.get(key) != made:
            self.reckonings[key] = made
            self.delayed = True

    def shelled(self, argv: list[Word], depth: int) -> bool:
        """Return whether *argv*, whose command word only bash knows, ends the task as a shell.

        The word may name a shell (``"$@"``) that runs the text on its input; or nobody
        knows the name (``"$(which bash)" -c text``) and its ``-c`` text is a command line
        the guard refuses.  A text that follows several of its options is read once: each
        would run in a shell of its own.
        """
        made = self.expansions(argv[0], _SOME)
        if self.inlet and any(_named(text) in SHELLS for text in made):
            return True
        if not any(_SOME in posixpath.basename(text) or _ANY in text for text in made):
            return False  # a name the guard reads: judged as that program
        read: set[str] = set()
        for option, text in zip(argv[1:], argv[2:]):
            if option.literal and re.fullmatch(r"-[A-Za-z]*c", option.text):
                for line in self.texts([text], _HERE):
                    if line and line not in read:
                        read.add(line)
                        if self.reads(line, None, depth, "shell", [], None):
                            return True
        return False

    def stands(self, text: str, position: int) -> bool:
        """Return whether the scanner reads the words of a command at *position* of *text*.

        A word put there on a line of its own (:data:`_PROBE`) is then a word of one of its
        commands.  It is none in a string, in a comment, in arithmetic or in a
        here-document that is data.  A text the scanner cannot read holds a command
        anywhere.
        """
        key = (text, position)
        if key not in self.stood:
            if len(self.stood) >= _MAX_PROBES:
                return True
            try:
                commands = self.scan(f"{text[:position]} {_PROBE}\n{text[position:]}", None)
            except TooDeepError:
                return True
            self.stood[key] = any(
                word.text == _PROBE for command in commands for word in command.words)
        return self.stood[key]

    def commanded(self, text: str) -> list[str]:
        """Return the arithmetic of *text* that is in no word the scanner shows.

        The expression of each arithmetic command, what a compound assignment evaluates
        (:data:`_COMMANDED`) and what a here-document evaluates that its shell expands
        (:data:`_EXPANDED_DOCUMENT`): where the scanner reads a command (:meth:`stands`),
        not in a string or in a here-document that is data.  An expression goes on to the
        end of *text* when the guard cannot tell where it closes (no ``))`` there: a
        ``case`` in a substitution of it).
        """
        regions: list[str] = []
        for start in _COMMANDED.finditer(text):
            opened = start.end()
            if not self.stands(text, start.start(1) if start[1] else start.start()):
                continue
            if start[1]:
                end = _closing(text, opened - 1)
                closed = end > 0 and text[end - 2:end] == "))"
                regions.append(text[opened:end if closed else len(text)])
            else:
                end = _closing(text, opened)
                regions += _reckoned(text[opened:end - 1 if end > 0 else len(text)])
        for document in _EXPANDED_DOCUMENT.finditer(text):
            if self.stands(text, document.start()):
                regions += _reckoned(document[2], False)
        return regions

    def reckons(self, regions: list[str], commands: list[SimpleCommand]) -> None:
        """Note what the arithmetic of a text assigns (:meth:`counts`).

        Its *regions* (:func:`_reckoned`), the operands of each ``let`` and of each ``[[``
        that compares numbers among its *commands*, and the subscript of each name a
        builtin is given (:func:`_names`).  An expression with an expansion is every text
        bash may make of it, with every value its variables ever hold: the call is read
        again once those are known.
        """
        self.late += 1
        try:
            for command in commands:
                words = [word.text for word in command.words]
                if "let" in words or (words[:1] == ["[["] and _COMPARES.intersection(words)):
                    regions = [*regions, *words[1:]]
                argv = command.argv()
                for word in _names(argv) if argv and argv[0].text in _NAMERS else ():
                    named = [word.text] if word.literal else self.expansions(word)
                    self.settled("\x00" + word.text, named)
                    for made in named:
                        if _ANY in (made.partition("=")[0] if argv[0].text in _DECLARERS else made):
                            self.armed = True  # any name, with any subscript
                        regions.append(made.partition("[")[2])
            for region in regions:
                if "$" in region or "`" in region:
                    # as written too (but for a length, ``${#NAME}``): what it assigns in
                    # an expansion of its own, the parameters of its function or text
                    texts = [_LENGTH.sub("0", region), *self.expanded(region)]
                    self.settled(region, texts)
                    self.counts(texts)
                else:
                    self.counts([region])
        finally:
            self.late -= 1

    def walks(self, commands: list[SimpleCommand], unseen: bool, readers: dict[int, bool],
              callers: dict[int, SimpleCommand], certain: bool,
              fed: _Feeds, text: str, depth: int) -> bool:
        """Return whether one of the *commands* of *text* ends the task (see :meth:`refuses`).

        The commands are walked in the order they start.  After a ``cd``, *anchor* is the
        group it is in and *last* what of that group the command before was part of: a
        command is linked to the ``cd`` while it is part of the same thing (a compound
        command that follows ``&&`` runs whole or not at all), follows ``&&``, or the shell
        cannot have gone on without the ``cd`` (*sure*, :func:`_exits`); the commands of a
        substitution are part of the command that holds it.  *entered* is the
        directory of a ``cd`` whose operand the scanner cannot read and the guard can
        (:meth:`target`), until a command that may change it unseen; never in a text whose
        loops change directory (:func:`_turns`).  Each command first finds its frames
        (:meth:`arrives`) and assigns (:meth:`assigns`); one that holds substitutions
        (*holder*) assigns once their commands were read, as bash runs them first.
        """
        moved = proven = searched = sure = unfolds = False
        anchor: Group | None = None
        last: SimpleCommand | Group | None = None
        entered: str | None = None
        reads = not any(map(_turns, commands))
        base = len(self.frames)
        # the operands of a shell that reads its script from a pipe: parameters nobody reads
        loose = any(id(other) in readers and _operated(other.argv()) for other in commands)
        holder: tuple[SimpleCommand, str, str | None, int, bool] | None = None
        held: frozenset[int] = frozenset()  # the substitutions of the holder (:func:`_ids`)
        for index, command in enumerate(commands):
            if holder is not None and not _within(command, held):
                self.settles(*holder, text, fed)
                holder = None
            argv = command.argv()
            node = _level(command, anchor)
            if isinstance(node, Group) and node.kind in _SUBSTITUTIONS:
                linked = True  # a command of a substitution: part of the command that holds it
            else:
                linked = node is not None and (sure or node is last or node.after_and)
                last = node
            name = posixpath.basename(argv[0].text) if argv else ""
            where = command.cwd if command.cwd is not None else entered
            # ``$PWD`` here: that directory while the ``cd`` before is proven
            here = where if where is not None and (not moved or (proven and linked)) else self.start
            self.weak = False
            self.arrives(command, base, callers, certain, here)
            if self.marked and _marks(command):
                self.turns(command.group, command.words[0].text == _ELSE)
                continue
            header = False  # the first command of a ``for`` or a ``select``
            for frame in self.frames[base:]:
                if frame.header:
                    frame.header, header = False, frame is self.frames[-1]
            if header and [word.text for word in command.words[1:2]] in (["do"], ["{"]):
                return True  # ``for NAME do command``: the scanner shows them as one command
            if argv and argv[0].text in self.aliases:
                return True  # the text of an alias, with the words that follow it
            self.inlet = self.inherits or self.supplies(command, fed)
            # an ``exec`` that is a command of the text may have set the input of them all
            self.early = self.inherits or self.supplies(command, fed, False) or (
                self.rewired and any(self.stands(text, match.start())
                                     for match in _EXEC_INPUT.finditer(text)))
            if argv and not argv[0].literal and self.shelled(argv, depth):
                return True
            if name in _MODELLED and name in self.functions:
                self.shades()  # a command the guard reads may be a function of the call
            self.weak = _aside(command)
            self.defaults(command, text)
            # a quoted substitution: bash may expand it later where the shell keeps the
            # word (an assignment, a builtin, a function), not what another program gets
            outer = command.group
            while outer is not None and outer.kind not in _SUBSTITUTIONS:
                outer = outer.group
            kept = not argv or header or name in self.functions or (
                name in _KEEPERS and (name != "printf" or "-v" in (word.text for word in argv))
            ) or (name in ("echo", "printf") and (
                outer is not None or command.reader is not None))  # the shell reads it back
            strings = [word for _, operator, word in command.redirections if operator == "<<<"]
            if any(word.literal and (kept or _ASSIGNMENT.fullmatch(word.text) is not None)
                   and self.quoted(word.text, depth, kept and name == "printf")
                   for word in (*command.words, *strings)):
                return True
            if command.substitutions:
                holder = (command, name, here, len(self.frames), header)
                held = _ids(command.substitutions)
            else:
                self.assigns(command, name, here, text, fed, header)
            if not argv:
                proven = proven and linked
                continue
            self.exports = self.environment(command, argv, here)
            if name == "shift":
                self.shifts(argv)
            elif name == "set":
                operands = self.operands(argv)
                if operands is not None:
                    self.places(*self.passed(operands, here))
            if name in ("cd", "pushd", "popd"):
                operand = argv[-1].text if len(argv) > 1 else ""
                entered, rooted = (self.target(command, argv, entered)
                                   if name == "cd" and reads else (None, False))
                proven = not (unseen or command.negated) and (
                    rooted or operand.startswith("/") or not moved or (proven and linked))
                sure = proven and _exits(commands, index)
                searched = not (operand.startswith(("/", "./", "../")) or operand in (".", ".."))
                anchor, last, moved = command.group, command, True
                continue
            proven = proven and linked
            if command.unknown or name in ("eval", "source", "."):
                entered = None
            trusted = where is not None and (proven or not moved)
            scripts = self.scripts(command, argv, readers, fed, here or _HERE)
            if scripts is None:  # a here-document its shell expands: read below, once
                unfolds = True
            elif any(self.reads(script, None, depth,
                                "eval" if name in ("source", ".") else "shell",
                                None if loose else [], here, alone=len(scripts) == 1)
                     for script in scripts):
                return True
            extra = self.handed(command, fed, here or _HERE) if any(
                posixpath.basename(word.text) in _FED_RUNNERS for word in command.words) else []
            if self.judges(command, argv, where, trusted, searched and moved, text, depth,
                           extra):
                return True
        if holder is not None:
            self.settles(*holder, text, fed)
        # the lines of every here-document, read as commands of a shell of their own that
        # ran at any time, with any parameters
        if not unfolds:
            return False
        if any(value and _PASTED.search(value) for held in self.ever.values() for value in held):
            return True  # a value its shell pastes in is more than the word read here
        return self.reads(_HEREDOC.sub("", text), None, depth, "shell", None, None, True)

    def settles(self, command: SimpleCommand, name: str, here: str | None, depth: int,
                header: bool, text: str, fed: _Feeds) -> None:
        """Record what *command* assigns, now that the commands of its substitutions were read.

        *depth* is how many frames were open where it runs.
        """
        self.leaves(depth)
        self.weak = _aside(command)
        self.assigns(command, name, here, text, fed, header)

    def operands(self, argv: list[Word]) -> list[Word] | None:
        """Return the words ``set`` (*argv*) makes the positional parameters.

        ``None`` for a ``set`` that sets options only (``set -euo pipefail``).
        """
        skip = False
        for index, word in enumerate(argv[1:], 1):
            if skip:
                skip = False
            elif word.text in ("-", "--"):
                return argv[index + 1:]
            elif not word.text.startswith(("-", "+")):
                return argv[index:]
            else:
                skip = word.text.endswith("o")
        return None


def destructive_command_guard(
    command: str, work_dir: str | None,
    isdir: Callable[[str], bool] = os.path.isdir,
    realpath: Callable[[str], str] = os.path.realpath,
    cdpath: Callable[[], str] = _inherited_cdpath,
) -> str | None:
    """Refuse a shell command that would end the task itself.

    One whose ``rm`` names the work directory (as given or as the filesystem resolves it),
    the home directory, the root, a directory above one of them or every entry of one
    (``rm -rf *`` where the call starts), or that kills every process (``kill -9 -1``,
    ``pkill -f .``).  :class:`_Guard` reads the command, and refuses one whose ``rm`` or
    ``kill`` it cannot prove safe; programs that delete by their own means (``find
    -delete``, ``git clean``, ``python -c``) are not read.

    Args:
        command: The shell command line the model wants to run.
        work_dir: The agent's working directory (``None`` protects only the
            root, the home directory and the process table).
        isdir: Whether a path is a directory where the command runs.
        realpath: The path the filesystem there gives a path, links
            resolved as far as it exists (``os.path.realpath``).  Both
            default to this host's; a container's tools pass its own:
            its work directory is no path of this host.  Asked only
            about a command that deletes.
        cdpath: The ``CDPATH`` the command's shell inherits (``""`` for
            none): by default this process's, which the host ``Bash``
            tool hands its shell.  Asked only about a delete entered
            through a ``cd`` that ``CDPATH`` may redirect.

    Returns:
        :data:`DESTRUCTIVE_VERDICT` when the command is refused; the scanner's own verdict
        when it cannot read the command to its end
        (:class:`~kiss.agents.sorcar.shell_scan.TooDeepError`), which is
        :data:`~kiss.agents.sorcar.shell_scan.TOO_DEEP_VERDICT` when it nests too deep;
        else ``None``.
    """
    if not _mentions(command):
        return None
    guard = _Guard(work_dir, isdir, realpath, cdpath)
    try:
        refused = guard.decides(command)
    except TooDeepError as unread:
        return str(unread) or TOO_DEEP_VERDICT
    return DESTRUCTIVE_VERDICT if refused else None


def _is_directory(path: str) -> bool:
    """Return that *path* is a directory: the answer of a caller that cannot ask."""
    return True


def _as_written(path: str) -> str:
    """Return *path* itself: the answer of a caller that cannot resolve links."""
    return path


def _no_cdpath() -> str:
    """Return no ``CDPATH``: the answer of a caller whose shell inherits none it knows."""
    return ""


class _TextGuard:
    """The destructive-command guard of one work directory, asked like a compiled pattern."""

    def __init__(self, workdir: str) -> None:
        self.workdir = workdir

    def search(self, command: str) -> str | None:
        """Return the refusal of *command* (see :func:`destructive_command_guard`), or ``None``.

        By the command's text alone: no filesystem is asked.
        """
        return destructive_command_guard(
            command, self.workdir or None, _is_directory, _as_written, _no_cdpath)


def destructive_pattern(workdir: str) -> _TextGuard:
    """Return the destructive-command guard for *workdir*, for callers that only have the text.

    Args:
        workdir: The working directory whose deletion must be refused
            (``""`` protects the root and the home directory only).
    """
    return _TextGuard(workdir)


def lift_install_timeout(command: str, timeout_seconds: float) -> float:
    """Return the timeout to run *command* with.

    Args:
        command: The shell command line (or one command of a parallel batch).
        timeout_seconds: The timeout the model asked for.

    Returns:
        ``timeout_seconds``, raised to :data:`INSTALL_TIMEOUT_SECONDS` when
        the command (any of its :func:`_readings`) is a package install or
        a build and asked for less.
    """
    if any(INSTALL_COMMANDS.search(text) for text in _readings(command)):
        return max(timeout_seconds, INSTALL_TIMEOUT_SECONDS)
    return timeout_seconds


def refuse_or_lift(
    command: str, work_dir: str | None, timeout_seconds: float,
    isdir: Callable[[str], bool] = os.path.isdir,
    realpath: Callable[[str], str] = os.path.realpath,
    cdpath: Callable[[], str] = _inherited_cdpath,
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
        cdpath: As for :func:`destructive_command_guard`.

    Returns:
        ``(refusal, timeout_seconds)``: the guard's error message (``None``
        when the command may run) and the possibly lifted deadline.
    """
    guard = destructive_command_guard(command, work_dir, isdir, realpath, cdpath)
    return guard, lift_install_timeout(command, timeout_seconds)
