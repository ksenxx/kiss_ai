# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A small quote-aware scanner of bash command text, for the shell-tool guards.

:func:`simple_commands` splits a command line into its simple commands
the way bash reads them: quotes, backslashes, comments and line
continuations are honoured, and here-document bodies (any delimiter,
quoted or not, ``<<-`` too) are text, except for the substitutions of
an unquoted one, which run.  ``$(...)``, backticks, process
substitutions, subshells, brace groups, compound commands (``if``,
loops, ``case``) and arithmetic commands (``((...))``) are
:class:`Group` objects, so a redirection or a pipe after one applies to
the commands in it.  Each simple command records its words, the files
its standard output is redirected to, whether a here-document feeds it,
whether a pipe does, what its output is piped to, its group, and the
directory it runs in, followed through ``cd`` (unknown after a physical
one, ``cd -P`` or under ``set -P``).  A subshell, a
substitution, a pipeline stage and a ``&`` list keep their ``cd`` and
their assignments to themselves; a ``cd`` that may not run (after
``||``, or in an ``if``, a loop or a ``case``) leaves the directory
unknown, as does one after ``&&`` once its list ends (the rest of the
list runs only when it ran), and a variable set any other way than by a literal
assignment (``read``, ``unset``, ``declare``, ``let``, a loop variable,
an arithmetic command or expansion, an assignment before a special
builtin, ``eval``, ``source``, an assigning expansion ``${T=x}``) is
unknown afterwards, as is every variable once a name reference exists,
each that has an attribute changing what is assigned (``declare -i``,
``readonly``) and each that bash sets itself (``RANDOM``, ``_``; ``PWD``
once a ``cd`` follows its assignment), and the directory after ``eval`` or
``source``; the literal text
``eval`` runs is scanned for its commands, as is a child shell's
``-c`` text, which knows the exported variables only.  The
assignments a command starts with are read in order: a later one sees
the values the earlier ones gave the variables known so far.  A function body
is scanned as commands that may not run, and a call of a function leaves
unknown what a body may set: its variables, the directory, the shell's
output.  A bare ``exec > file`` is
the standard output of every later command that does not redirect its
own: the file as the ``exec`` opened it (a later ``cd`` does not move
it), in the shell that ran it (a pipeline stage's or a backgrounded
``exec`` ends with its subshell), and after an ``exec`` that may not
have run the earlier file is possible too.  The command of a ``coproc``
is read as a pipeline stage is, and the variable it names is unknown
afterwards.  Other commands are taken to succeed.

Nothing is expanded: a word holding a parameter, a command or
arithmetic substitution, a leading ``~`` (or one after the ``=`` or a
``:`` of an assignment), a glob or a brace expansion is marked as not
literal, as is an unquoted variable once the command set ``IFS``; the
operands of a program run in another directory by its wrapper (``env
-C``) are not literal either (a shell's ``-c`` text is scanned there,
in a directory only the shell knows).  Of an arithmetic expression only the
command substitutions are scanned (``<<`` in it is a shift).  It is a
scanner for guards, not a shell, and where bash versions disagree on
whether a substitution runs (the macOS system bash is 3.2) it is
scanned.
"""

from __future__ import annotations

import os
import posixpath
import re
import sys
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from kiss.core.processes import bash_version

_METACHARACTERS = frozenset(" \t\n;&|<>()")
# Words before the command word: wrappers that run the next word.
_PREFIXES = frozenset({
    "sudo", "doas", "command", "builtin", "exec", "nohup", "env", "nice", "timeout", "stdbuf",
    "xargs",
})
# The options of those wrappers that take the next word as their value.
_PREFIX_VALUE_OPTIONS = {
    "sudo": frozenset({
        "-C", "-D", "-g", "-h", "-p", "-R", "-r", "-T", "-t", "-U", "-u", "--close-from",
        "--chdir", "--group", "--host", "--prompt", "--chroot", "--role", "--command-timeout",
        "--type", "--other-user", "--user",
    }),
    "doas": frozenset({"-C", "-u"}),
    "exec": frozenset({"-a"}),
    "env": frozenset({"-a", "-C", "-u", "--argv0", "--chdir", "--unset"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "timeout": frozenset({"-k", "-s", "--kill-after", "--signal"}),
    "stdbuf": frozenset({"-e", "-i", "-o", "--error", "--input", "--output"}),
    "xargs": frozenset({
        "-a", "-d", "-E", "-e", "-I", "-L", "-l", "-n", "-P", "-s", "--arg-file", "--delimiter",
        "--eof", "--replace", "--max-lines", "--max-args", "--max-procs", "--max-chars",
        "--process-slot-var",
    }),
}
#: A cluster of a wrapper's short options that ends with one of those: ``-iu`` as ``-u``.
_PREFIX_VALUE_CLUSTER = {
    name: re.compile("-[^-{0}]*[{0}]".format(
        "".join(option[1] for option in options if len(option) == 2)))
    for name, options in _PREFIX_VALUE_OPTIONS.items()}
#: What cannot start a ``NAME=value`` word a wrapper takes before its program: ``env`` sets
#: any word with an ``=``, ``sudo`` any that is no absolute path; the other wrappers take none.
_PREFIX_ASSIGNMENTS = {"env": ("-",), "sudo": ("-", "/")}
#: The wrappers that are programs: the word after one names a program, not this shell's builtin.
_PROGRAM_PREFIXES = _PREFIXES - {"command", "builtin", "exec"}
#: The options (by their start: ``-C dir``, ``-Cdir``, ``--chdir=dir``) of the
#: wrappers that run the program in another directory: its operands are
#: paths there, which only the shell resolves.
_PREFIX_CHDIR_OPTIONS = {"env": ("-C", "--chdir"), "sudo": ("-D", "--chdir")}
#: ``env``'s option whose text is the command line it runs, with the words after it:
#: ``-S 'text'``, ``-iStext``, ``--split-string=text`` (``--s`` abbreviates it).
_SPLIT_STRING = re.compile(
    r"(?:-(?P<flags>[0iv]*)S|--s[a-z-]*(?P<equals>=)?)(?P<text>.*)", re.DOTALL)
#: ``uv`` options whose value is the next word: global ones before the
#: subcommand (``uv --directory . run``) and ``uv run``'s own.
UV_VALUED_OPTIONS = frozenset({
    "--project", "--directory", "--config-file", "--cache-dir", "--color",
    "--allow-insecure-host", "--python-preference", "--with", "--with-editable",
    "--with-requirements", "--python", "-p", "--env-file", "--extra", "--group",
    "--only-group", "--package", "--index", "--find-links", "-w", "--python-platform",
})
# Reserved words in command position: those that open or close a group, and
# those that only lead to the next command.
_OPENERS = frozenset({"{", "if", "while", "until", "for", "select", "case"})
_CLOSERS = frozenset({"}", "fi", "done", "esac"})
_RESERVED = _OPENERS | _CLOSERS | {
    "!", "then", "else", "elif", "do", "time", "function", "coproc"}
_BARE_WORD = re.compile(r"(?:[{}!]|[a-z]+)(?=[ \t\n;&|<>()]|\Z)")
#: After the word that follows ``coproc``: the compound command that makes the word its NAME.
_COPROC_COMMAND = re.compile(
    r"[ \t]*\(|[ \t]+(?:\[\[|\{|if|while|until|for|select|case)(?=[ \t\n;&|<>()]|\Z)")
#: The ``()`` after a function's name (``f() { ...; }``), and the name after ``function``.
_FUNCTION_PARENS = re.compile(r"\([ \t]*\)")
_FUNCTION_NAME = re.compile(r"[ \t]*([^\s;&|<>()]+)")
#: The ``[[`` that opens a conditional command, in command position.
_CONDITIONAL = re.compile(r"\[\[(?=[ \t\n])")
#: ``T=x``, ``T+=x`` and an array element's ``T[0]=x``; ``T=(x y)`` starts the same way.  A name
#: is ASCII, here and below: to bash ``Té=x`` is a command and ``$Té`` is ``$T`` and a letter.
_ASSIGNMENT = re.compile(
    r"(?P<name>[A-Za-z_]\w*)(?P<subscript>\[[^\]]*\])?(?P<append>\+)?=", re.ASCII)
#: The same as written, unquoted, where a word starts.
_WRITTEN_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*(?:\[[^\]]*\])?\+?=", re.ASCII)
_NAME = re.compile(r"[A-Za-z_]\w*", re.ASCII)
#: An array element as an operand names it (``read 'a[n++]'``): group 1 is its subscript.
_ELEMENT = re.compile(r"[A-Za-z_]\w*(\[[^\]]*\])", re.ASCII)
_BLANKS = re.compile(r"[ \t]*")
#: Builtins that assign the variables their operands name (``read T``,
#: ``unset T``, ``declare -x T=x``, ``mapfile T``): afterwards
#: the variable's value is unknown.
_ASSIGNING_BUILTINS = frozenset({"read", "readarray", "mapfile", "declare", "typeset", "local",
                                 "readonly", "unset", "getopts"})
#: ``read``'s options that take a value, attached (``-aT``) or next (``-a T``);
#: ``-a`` names the array read into.
_READ_VALUED = "adinNptu"
#: Builtins whose ``-n`` makes a name reference: assigning it assigns the
#: variable it names, so afterwards no variable is known.
_DECLARING_BUILTINS = frozenset({"declare", "typeset", "local"})
#: A word of theirs that may give a variable an attribute under which what is
#: assigned is not what it holds: an option with ``i``, ``l``, ``u`` or ``c`` (they
#: transform the value) or with ``r`` (it refuses it).  Group 1: the variable may be
#: any, by an option with ``n`` or a word that is no name as written (``$N``).
_ATTRIBUTE = re.compile(
    r"(-.*[n$`]|(?:[A-Za-z_]\w*)?[^\w=+\[-])|-.*[cilru]", re.DOTALL | re.ASCII)
#: The variable a builtin fills when no operand names one (``read <<<x``
#: sets REPLY, ``mapfile <<<x`` MAPFILE), unknown after every such call.
_DEFAULT_TARGETS = {"read": "REPLY", "readarray": "MAPFILE", "mapfile": "MAPFILE"}
#: Builtins that run text as commands: afterwards no variable is known.
_SOURCING_BUILTINS = frozenset({"eval", "source", "."})
#: A word of ``set`` or ``shopt``, or of a shell's own command line, that may turn
#: ``physical`` on (``-P``, ``-o physical``, a word the shell expands): ``cd`` then
#: follows the links.
_PHYSICAL = re.compile(r"-.*[P$`]|[$`]|physical$", re.DOTALL)
#: The options of ``set``, ``shopt`` and a shell's command line after which bash still runs
#: the text as the scan reads it.  After any other it may not (:meth:`_Scanner.unfollowed`):
#: ``lastpipe`` runs a pipeline's last stage in the shell itself, ``cdable_vars`` enters the
#: directory a variable names, ``-k`` takes the assignments out of a command's arguments,
#: ``extglob`` makes a pattern of a word without ``*``, ``?`` or ``[``, ``failglob`` and
#: ``posix`` change what a failure skips and what an assignment outlives.  (After ``noexec``
#: the shell runs nothing more: read as if it ran, the text is refused no less.)
_OPTIONS = frozenset(
    "allexport dotglob errexit errtrace functrace globstar inherit_errexit noclobber nocaseglob "
    "nocasematch noexec noglob nounset nullglob physical pipefail verbose xtrace".split())
#: Their letters (``set -eux``), with those of ``shopt`` itself (``-s``, ``-u``, ``-q``,
#: ``-p``) and ``-o`` / ``-O``, which take a name; a shell's command line has more.
_OPTION_LETTERS = frozenset("aefnpqsuvxCEPToO")
_SHELL_LETTERS = _OPTION_LETTERS | {"c"}
#: The conditions of a ``trap`` that runs when the shell ends.  Any other (``DEBUG``,
#: ``ERR``, ``RETURN``, ``CHLD``, a signal, which the text may send the shell itself:
#: ``kill -TERM $$``) may run between the commands that follow it.
_LATE_TRAPS = frozenset({"EXIT", "0"})


def _trap_operands(argv: list[Word]) -> list[Word]:
    """Return the action the ``trap`` command *argv* sets and its conditions; ``[]`` for none.

    Nothing runs later for a ``trap`` that prints (no operand, ``-p``,
    ``-l``), resets (``-``, or a condition alone) or ignores (the empty
    action: ``trap '' HUP``).  One operand only the shell knows may be
    an action and its conditions, when bash splits it.
    """
    operands = argv[1:]
    dashed = bool(operands) and operands[0].literal and operands[0].text == "--"
    if dashed:
        del operands[0]
    if not operands:
        return []
    first = operands[0]
    inert = first.literal and (first.text in ("", "-") or len(operands) == 1
                               or (not dashed and first.text.startswith("-")))
    return [] if inert else operands


#: The options with which ``command`` says what a name runs and runs nothing: ``-v``, ``-V``.
_DESCRIBING = re.compile(r"-p*[vV][pvV]*")


#: The programs that read their standard input and the files their operands name, and no
#: other descriptor: a here-document on one (``cat 3<<EOF``) is not their input.
_STDIN_READERS = frozenset({"cat", "tee"})


def _reads_heredoc(command: SimpleCommand) -> bool:
    """Return whether *command* may read a here-document it holds.

    It does when one is its standard input (``<<EOF``; ``11<<EOF`` too,
    as sh reads it).  One on another descriptor (``3<<EOF``) is out of
    reach of a ``cat`` or a ``tee`` named as written whose standard
    input is the one the shell gave it or a here-string, and for a
    ``cat`` one without a file to read: an operand may name the
    descriptor (``/dev/fd/3``, ``/dev/stderr``, a link to one), as a
    redirection of the standard input may (``<&3``, ``< /dev/fd/3``).
    Any other command may read it, or hand it to what it runs.
    """
    heredocs = [fd for fd, operator, _ in command.redirections if operator in ("<<", "<<-")]
    words = command.words
    if any(len(fd) != 1 or fd == "0" for fd in heredocs) or not (
            words and words[0].text in _STDIN_READERS and all(word.literal for word in words)):
        return True
    reads_files = words[0].text == "cat" and any(
        not word.text.startswith("-") for word in words[1:])
    return reads_files or any(
        descriptor(fd, operator) == "0" and not operator.startswith("<<")
        for fd, operator, _ in command.redirections)


def _describes(prefix: list[Word]) -> bool:
    """Return whether the wrappers *prefix* of a command only describe its command word.

    ``command -v name`` and ``command -V name`` print what the name is:
    bash runs nothing, whatever the name expands to.  Only these two
    words, after the assignments: ``command`` may be the value of
    another wrapper's option (``env -u command -v ln``).
    """
    wrappers = [word for word in prefix if not word.assignment]
    return (len(wrappers) == 2 and wrappers[0].literal and wrappers[0].text == "command"
            and wrappers[1].literal and _DESCRIBING.fullmatch(wrappers[1].text) is not None)


#: The variables under which a starting shell does not run its text as written: it reads
#: the file ``BASH_ENV`` names first, starts with the options the next two list, and
#: expands ``PS4`` (which may hold a command) before each command it traces.
_SHELL_STARTUP = frozenset({"BASH_ENV", "BASHOPTS", "SHELLOPTS", "PS4"})
#: A value of ``PS4`` that runs nothing when bash expands it.
_INERT = re.compile(r"[^$`\\]*")
#: The operand of a name reference that refers to a variable, not to an element whose
#: subscript bash evaluates at each use (``declare -n R='a[$(cmd)]'``).
_PLAIN_REFERENCE = re.compile(r"[A-Za-z_]\w*=[A-Za-z_]\w*", re.ASCII)
#: An indirect expansion's parameter (``${!T}``), group 1 its name; not the names with a
#: prefix (``${!T*}``, ``${!T@}``) nor an array's keys (``${!T[@]}``).
_INDIRECT = re.compile(r"!([A-Za-z_]\w*)(?![\w*@]|\[[@*]\])", re.ASCII)
#: A subscript of a compound array value (``a=([i++]=x)``), group 1 its expression.
_VALUE_SUBSCRIPT = re.compile(r"\[([^\]]*)\]\+?=")
#: The option that names the variable a builtin assigns: ``printf -v T``, ``wait -p T``.
_ASSIGNING_OPTION = {"printf": "v", "wait": "p"}
#: The wrappers that may reset the environment of the program they run
#: (``env -i``, ``env -u T``, ``sudo`` without ``-E``).
_ENVIRONMENT_WRAPPERS = frozenset({"env", "sudo", "doas"})
#: The shells whose ``-c`` (``-lc``, ``-ec``, ``-ce``) argument is shell text.
SHELLS = frozenset({"bash", "sh", "dash", "zsh", "ksh"})
#: Bash's long options (``--norc``, ``-norc`` too), which it reads before
#: the single-letter ones; two of them take the next word as their value.
_SHELL_LONG_VALUED = frozenset({"init-file", "rcfile"})
_SHELL_LONG_OPTIONS = _SHELL_LONG_VALUED | frozenset({
    "debugger", "dump-po-strings", "dump-strings", "help", "login", "noediting", "noprofile",
    "norc", "posix", "pretty-print", "restricted", "verbose", "version",
})


def shell_script_index(words: list[str]) -> int | None:
    """The index in a shell's arguments *words* of the command string it runs, or ``None``.

    ``bash -c "..."``: the text is the first operand after every option,
    read as bash reads them: its long options first (``--norc``,
    ``--rcfile file``), then clusters of letters (``-ec``, ``-co posix``;
    an ``o`` or ``O`` in one, or in a ``+`` cluster, takes the next word
    as its value), until ``--`` or ``-`` ends them or an operand starts.
    Without a ``c`` among the letters that operand is a script file, not
    shell text (``bash -a file -c "..."`` runs ``file``).

    Args:
        words: The arguments after the shell's name.
    """
    index, letters = _shell_operand(words)
    return index if "c" in letters and index < len(words) else None


def _shell_operand(words: list[str]) -> tuple[int, str]:
    """Return the index in a shell's arguments *words* of its first operand, and its option letters.

    The options are read as :func:`shell_script_index` describes; the
    index is past the last word when no operand follows them.
    """
    index, letters, long_options = 0, "", True
    while index < len(words):
        word = words[index]
        index += 1
        if word in ("--", "-"):
            break
        if len(word) < 2 or word[0] not in "-+":
            index -= 1
            break
        if long_options and word[0] == "-" and word.lstrip("-") in _SHELL_LONG_OPTIONS:
            if word.lstrip("-") in _SHELL_LONG_VALUED:
                index += 1
            continue
        if word.startswith("--"):  # a long option after a cluster: an error, or another shell's
            continue
        long_options = False
        letters += word[1:]
        index += word.count("o", 1) + word.count("O", 1)
    return index, letters


def _may_be_input(word: Word) -> bool:
    """Return whether the file operand *word* may name the standard input, not a file of its own.

    A word only the shell knows may (a process substitution too); one
    that ends in a literal name does when the name is a descriptor's
    (``/dev/stdin``, ``/dev/fd/0``).
    """
    name = posixpath.basename(word.text)
    return not (word.literal or word.path) or name == "stdin" or name.isdigit()


def reads_script(argv: list[Word]) -> bool:
    """Return whether the command *argv* is a shell whose script is its standard input.

    A shell (:data:`SHELLS`) without ``-c`` that has no script operand,
    has ``-s``, or whose script operand may name its standard input
    (``bash /dev/stdin``).

    Args:
        argv: The command word and its arguments (:meth:`SimpleCommand.argv`).
    """
    if posixpath.basename(argv[0].text) not in SHELLS:
        return False
    index, letters = _shell_operand([word.text for word in argv[1:]])
    if "c" in letters:
        return False
    return "s" in letters or index + 1 >= len(argv) or _may_be_input(argv[index + 1])


def command_text(argv: list[Word]) -> str | None:
    """The literal shell text the command *argv* runs as commands of its own, or ``None``.

    ``eval`` runs its arguments joined; a shell (:data:`SHELLS`) the
    text its ``-c`` names (:func:`shell_script_index`), an ANSI-C string
    in it decoded (:meth:`Word.passed`).  Text the shell expands first
    is only the shell's to read, as is an empty one.

    Args:
        argv: The command word and its arguments (:meth:`SimpleCommand.argv`).
    """
    if not argv:
        return None
    name = argv[0].text
    if name == "eval":
        if any(word.passed() is None for word in argv[1:]):
            return None
        return " ".join(word.passed() or "" for word in argv[1:]) or None
    if posixpath.basename(name) in SHELLS:
        script = shell_script_index([word.text for word in argv[1:]])
        if script is not None:
            return argv[script + 1].passed() or None
    return None


#: What ``env -S`` makes of a letter after a backslash, where it is not the letter itself.
_SPLIT_ESCAPES = {"_": " ", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v"}


def _split_string(text: str) -> list[str] | None:
    """Return the words GNU ``env -S`` splits *text* into; ``None`` when only ``env`` knows them.

    Blanks separate the words, quotes group them, ``\\_`` is a blank and ``\\t``
    a tab (:data:`_SPLIT_ESCAPES`); a ``#`` that starts a word starts a comment
    and ``\\c`` ends the text.  A ``$`` (``${NAME}`` is the environment's) and
    text ``env`` rejects leave the words unknown.
    """
    words: list[str] = []
    word: str | None = None  # ``None`` between two words
    quote = ""
    index = 0
    while index < len(text):
        char = text[index]
        index += 1
        if char in "'\"" and quote in ("", char):
            quote, word = ("" if quote else char), word or ""
        elif (char in " \t\n\v\f\r" and not quote) or (char == "#" and word is None):
            if word is not None:
                words.append(word)
            word = None
            if char == "#":
                break
        elif char == "\\" and (quote != "'" or text[index:index + 1] in ("\\", "'")):
            escaped = text[index:index + 1]
            index += 1
            if escaped in ("c", "_") and not quote:  # the end of the text, of the word
                if word is not None:
                    words.append(word)
                word = None
                if escaped == "c":
                    break
            elif escaped and escaped in "\"#$'\\_fnrtv":
                word = (word or "") + _SPLIT_ESCAPES.get(escaped, escaped)
            else:
                return None
        elif char == "$" and quote != "'":
            return None
        else:
            word = (word or "") + char
    if quote:
        return None
    return words if word is None else [*words, word]


#: The POSIX special builtins: an assignment before one (``T=x export Y``)
#: holds afterwards in sh, and not in bash.
_SPECIAL_BUILTINS = frozenset({"break", ":", "continue", ".", "eval", "exec", "exit", "export",
                               "readonly", "return", "set", "shift", "times", "trap", "unset"})
#: An assignment in an arithmetic expression: ``i=0``, ``i+=1``, ``i++``, ``--i``; any
#: subscripted name counts as one (``a[0]=1``, ``a[i]++``: the element makes ``a`` an array).
_ARITHMETIC_ASSIGNMENT = re.compile(
    r"(?:\+\+|--)\s*([A-Za-z_]\w*)"
    # a name is tried where it starts only: a long one is read once
    r"|(?<!\w)([A-Za-z_]\w*)\s*(?:\[|\+\+|--|(?:[-+*/%&|^]|<<|>>)?=(?!=))",
    re.ASCII)
#: The same where an expansion names the variable: ``$N=1``, ``${N}++``, ``--$N``.
_ARITHMETIC_EXPANDED_TARGET = re.compile(
    r"(?:\+\+|--)\s*[$`]|(?:\$\w+|[}`)])\s*(?:\+\+|--|(?:[-+*/%&|^]|<<|>>)?=(?!=))", re.ASCII)
#: Text of an arithmetic expression whose names are no operands: a command substitution's
#: (bash evaluates its output) and a length's (``${#x}``).
_NO_OPERAND = re.compile(r"\$\((?!\().*?\)|`[^`]*`|\$\{#[^}]*\}", re.DOTALL)
#: The value of a variable arithmetic assigned: an integer only the shell knows.  No text
#: of a word (no command line holds a NUL), and harmless as an operand of an expression.
NUMBER = "\0number"
#: The mark of a value only the shell knows that is one field ending in a literal ``/name``;
#: the value as written follows it (``D="$HOME/bin/python"``).  Quoted, as a command word,
#: it runs the program of that name (:attr:`Word.path`); unquoted, bash may split it.
PATH_VALUE = "\0path:"
#: The literal end of a value as written: no expansion holds a ``/`` and ends in these.
_LITERAL_END = re.compile(r"(?:/[\w.+-]+)+\Z", re.ASCII)
#: An integer as written: an operand that evaluates to itself.
#: (A decimal one: ``08`` is no number to bash, which reads a leading zero as octal.)
_DECIMAL = r"(?:0|[1-9]\d*)"
_INTEGER = re.compile(rf"-?{_DECIMAL}\Z")
#: An expansion in an arithmetic expression that may be any text: every one but ``$NAME`` and
#: ``${NAME}`` (read as the name is), ``$#``, ``$?``, ``$$`` and a length (``${#x}``).
_UNKNOWN_OPERAND = re.compile(r"`|\$(?![A-Za-z_]|\{[A-Za-z_]\w*\}|\{#[^}]*\}|[#?$])", re.ASCII)
#: A name an arithmetic expression reads or assigns: not the digits of ``16#ff`` or ``0x1f``.
_OPERAND = re.compile(r"(?<![\w#])[A-Za-z_]\w*", re.ASCII)
#: A name an expression assigns without reading it (``n = 5``, ``a[i] = 5``): group 1 is
#: the subscript, which it does read.
_PLAIN_TARGET = re.compile(r"(?<![\w#$\]}])[A-Za-z_]\w*(\[[^\]]*\])?\s*=(?!=)", re.ASCII)
#: An array element in an expression: assigning it leaves ``$name``, element 0, as it was.
_ARITHMETIC_ELEMENT = re.compile(r"(?<![\w#])([A-Za-z_]\w*)\s*\[", re.ASCII)
#: The variables bash computes at every use, whatever the environment held: numbers.
_NUMERIC = frozenset("BASHPID EPOCHSECONDS LINENO PPID RANDOM SECONDS SHLVL SRANDOM".split())
#: The operators of ``[[`` whose operands are arithmetic expressions.
_ARITHMETIC_TESTS = frozenset({"-eq", "-ne", "-lt", "-le", "-gt", "-ge"})
#: The loops: an iteration starts in the state the one before it left (:meth:`_Scanner.rehearse`).
_LOOPS = frozenset({"for", "select", "while", "until"})
#: How often a loop is scanned on trial before the scan gives up on what it leaves.
_TRIALS = 3
#: How deep the calls of functions are followed into their bodies, and how many in a scan.
_MAX_CALLS = 4
_MAX_FOLLOWED = 64
#: How many characters of the values of variables a scan reads (:meth:`_Scanner.read`), and
#: how long a directory it follows (Linux's ``PATH_MAX``): a value put into a word after
#: word and a directory that grows ``cd`` after ``cd`` cost their length each time, and
#: values built from one another double (``A="$A$A"``).  Past the first the scan ends as
#: one nested too deep does (:data:`UNREAD_VALUES_VERDICT`); past the second the directory
#: is one only the shell knows.
_MAX_READ = 1 << 20
_MAX_PATH = 4096
#: The list of a ``for`` loop over integers: ``in 1 2 3``, ``in {1..5}``.
_NUMBER_LIST = re.compile(
    rf"[ \t]+in(?:[ \t]+(?:-?{_DECIMAL}|\{{-?{_DECIMAL}\.\.-?{_DECIMAL}(?:\.\.-?\d+)?\}}))+"
    r"[ \t]*(?=[;\n])")
#: ``env``'s option that starts the program with an empty environment: ``-i``, ``-``.
_IGNORE_ENVIRONMENT = re.compile(r"-|-[a-zA-Z0]*i[a-zA-Z0]*|--i[a-z-]*")
#: A word of ``set`` after which every assignment is exported: ``-a``, ``-o allexport``.
_ALLEXPORT = re.compile(r"-\w*a|allexport$")
#: What can spell half of the here-document operator in text a command hands on: the
#: character, or an escape that bash (``$'...'``), ``printf`` or ``echo -e`` decode to it
#: (``\74``, ``\474``, ``\x3c``, ``\u003c``; U+FE64 and U+FF1C are variants a conversion may
#: turn into it).
_LESS = re.compile(
    r"<|\\0*[04]?74|\\x(?:\{[0-9A-Fa-f]*)?3[cC]|\\[uU]0*(?:3[cC]|[Ff][EeFf][0-9A-Fa-f]{2})")
#: The expansions that name the shell itself.
_SHELL_VALUED = frozenset({"$SHELL", "${SHELL}", "$BASH", "${BASH}", "$0", "${0}"})
#: Programs that run the command their operands name elsewhere: on another machine, in a
#: container.  The files named there are not this machine's.
ELSEWHERE = frozenset({"ssh", "docker", "podman", "kubectl"})
#: Commands that run none of their operands: a shell's name among them is data.
_RUNS_NO_OPERAND = frozenset({"echo", "printf", "grep", "which", "type", "test", "["})
#: Commands after which the shell holds their operands or their input: in a variable, in the
#: positional parameters (a script's too).  ``printf`` with ``-v`` only.
_KEEPS = SHELLS | {"read", "readarray", "mapfile", "set", "getopts", "declare", "typeset",
                   "local", "export", "readonly", "let", "printf"}
#: Builtins that run or evaluate what they are handed (a script, an action, a subscript), as
#: those of :data:`_KEEPS` do: one of them only the shell knows may run text unread.
_EVALUATES = _SOURCING_BUILTINS | {"trap", "test", "[", "[[", "unset", "wait"}
#: An option of ``xargs`` that puts its input into the words of the command it runs.
_REPLACES = re.compile(r"-[a-zA-Z0-9]*[Ii]|--replace")
#: Commands that write their operands or their input out as they are.
_COPIES = frozenset({"echo", "printf", "cat", "tee"})
#: A backslash-newline pair as bash's lexer reads one: after an even run of backslashes,
#: each of which escapes the next.  Group 1 is the pair.
_PAIR = re.compile(r"(?<!\\)(?:\\\\)*(\\\n)")
#: How many times a scan starts over with pairs kept (:class:`_Pairs`), and the characters
#: its passes read at most: past either the command is refused.
_MAX_RESTARTS = 64
_RESTART_WORK = 2_000_000
#: A value that is one arithmetic expansion, from its start to the end of the word.
_ARITHMETIC_VALUE = re.compile(r"\$\(\(((?:(?!\)\)).)*)\)\)\Z", re.DOTALL)
#: An expression that cannot fail once its names hold numbers: operands joined by operators
#: that take any number, or divided by a literal that is not zero.  (``1/0``, ``2**-1``,
#: ``08`` and a syntax error fail: nothing is assigned then.)
_UNFAILING = re.compile(
    rf"\s*-?\s*(?:{_DECIMAL}|[A-Za-z_]\w*)(?![\w#.])(?:\s*(?:"
    rf"(?:[-+*^]|<<|>>|[<>]=?|[=!]=|&&?|\|\|?)\s*-?\s*(?:{_DECIMAL}|[A-Za-z_]\w*)"
    rf"|[/%]\s*[1-9]\d*)(?![\w#.]))*\s*\Z", re.ASCII)
#: A ``${...}`` that cannot fail, as far as it is matched: a variable, an element by a plain
#: index or all of them, a positional or special parameter, a length, the indices, or one of
#: those with an operator every bash has, whose word is scanned for itself (``${x:-word}``,
#: ``${x#pattern}``, ``${x/a/b}``).  Not ``${x:offset}``, ``${a[i]}``, ``${!x}``, ``${x^^}``.
_ELEMENT_OR_NAME = r"[A-Za-z_]\w*(?:\[(?:[@*]|0|[1-9]\d*)\])?"
_UNFAILING_PARAMETER = re.compile(
    rf"\$\{{(?:#?{_ELEMENT_OR_NAME}\}}|![A-Za-z_]\w*\[[@*]\]\}}|#?(?:\d+|[@*#?$!-])\}}"
    rf"|(?:{_ELEMENT_OR_NAME}|\d+|[@*])(?::?[-+]|##?|%%?|//?))", re.ASCII)
#: The variables bash keeps read-only, and the subscript of an element by a plain index.
_READONLY = frozenset("BASHOPTS BASH_VERSINFO EUID PPID SHELLOPTS UID".split())
#: The variables bash starts with as integers: what one is assigned is evaluated as
#: arithmetic, as for one declared ``-i`` (``RANDOM=X`` evaluates the value of ``X``).
_INTEGERS = frozenset("BASHPID HISTCMD OPTIND RANDOM SECONDS SRANDOM".split())
_INDEX = re.compile(rf"\[{_DECIMAL}\]")
#: A redirection where a command ended: of the group, which does not run when it fails.
_REDIRECTED = re.compile(r"[ \t]*(?:\d*|\{\w+\})(?:[<>]|&>)")
# What may start at a token boundary: a process substitution, a redirection
# (with its file-descriptor number, or the ``{name}`` bash opens a new
# descriptor for) or a control operator.
_TOKEN = re.compile(
    r"(?P<substitution>[<>]\()"
    r"|(?P<fd>\d*|\{[A-Za-z_]\w*\})(?P<redirection><<<|<<-|<<|<>|<&|<|>>|>\||>&|>|&>>|&>)"
    r"|(?P<operator>;;&|;;|;&|&&|\|\||\|&|[;&|\n])",
    re.ASCII,
)
_PARAMETER = re.compile(r"\$(?:[A-Za-z_]\w*|[0-9@*#?$!-])", re.ASCII)
#: An expansion that is one variable and nothing else: ``$T`` or ``${T}``.
_VARIABLE = re.compile(r"\$(?:([A-Za-z_]\w*)|\{([A-Za-z_]\w*)\})\Z", re.ASCII)
#: The variables bash sets itself or refuses to assign (``RANDOM`` at every use, ``_`` after
#: every command, ``OPTARG`` in ``getopts``, ``BASH_CMDS`` in ``hash``, the read-only ``UID``):
#: what a command assigns
#: one is not what it expands to afterwards, so only the shell knows them.  (``PWD``, which
#: a ``cd`` sets, is followed up to the next one: :meth:`_Scanner.end_command`.)
_SHELL_SET = frozenset(
    "BASHOPTS BASHPID BASH_ALIASES BASH_ARGC BASH_ARGV BASH_CMDS BASH_COMMAND BASH_LINENO "
    "BASH_REMATCH BASH_SOURCE BASH_SUBSHELL BASH_VERSINFO COLUMNS DIRSTACK EPOCHREALTIME "
    "EPOCHSECONDS EUID FUNCNAME GROUPS HISTCMD LINENO LINES OLDPWD OPTARG OPTIND PIPESTATUS PPID "
    "RANDOM SECONDS SHELLOPTS SRANDOM UID _".split())
#: What makes bash split an unquoted expansion into words or expand it as a pattern.
_SPLIT_OR_GLOB = re.compile(r"[ \t\n*?\[]")
#: A quoted character (a backslash before a newline continues the line instead):
#: right after a ``~`` it keeps bash from expanding the tilde.
_QUOTED = re.compile(r"""['"]|\\[^\n]""")
#: A pattern character: bash expands an unquoted value that holds one as file names.
_GLOB = re.compile(r"[*?\[]")
#: A quote anywhere in a word as written, a backslash that ends the text too: with one, an
#: expansion of nothing leaves an empty word (:meth:`_Scanner.fields`).
_QUOTE = re.compile(r"""['"]|\\(?!\n)""")
# The parameter a ``${`` opens: a name, a positional number or a special
# parameter (``?``, ``#``, ``-``, ...; the characters that are operators
# after a parameter), after the ``!`` of ``${!x}``; or the ``#`` of
# ``${#x}`` with the parameter it measures, which is one only when that
# parameter ends the expansion (``${##}`` measures ``$#``; ``${##pat}``
# strips ``pat`` from it).
_BRACE_PARAMETER = re.compile(
    r"#(?:[A-Za-z_]\w*|[0-9]+|[@*#?$!-])(?=[}\[])|!?(?:[A-Za-z_]\w*|[0-9]+|[@*#?$!-])",
    re.ASCII,
)
# Redirections of standard output (``>&`` only when its operand is a file).
_STDOUT = frozenset({">", ">>", ">|", ">&"})
_BOTH = frozenset({"&>", "&>>"})
#: The operators that make descriptor 1 a file to write or a copy of another descriptor:
#: ``1<>file`` opens it read-write, and ``1<&3`` copies as ``1>&3`` does.
_STDOUT_OPENERS = _STDOUT | {"<>", "<&"}
#: The descriptor a duplication names: ``3``, ``03``, and ``3-``, which closes it afterwards.
_DUPLICATED = re.compile(r"(\d+)-?", re.ASCII)
# Groups that run in a subshell (a ``cd`` in them ends with them), and those
# whose output is a word of another command rather than a stream.
_SUBSHELLS = frozenset({"(", "$(", "`"})
_SUBSTITUTIONS = frozenset({"$(", "`"})
#: Groups and expansions within groups a scan follows; past them the text is
#: not a command line anyone runs, and the scan ends: a guard's with
#: :class:`TooDeepError`, a gate's with the commands found so far (a level
#: costs up to nine Python frames: some 460 of the interpreter's 1000 at most).
MAX_NESTING = 50


#: Text the model sees instead of running a command nested past :data:`MAX_NESTING`.
TOO_DEEP_VERDICT = (
    "REFUSED: the command did not run: it nests groups or expansions more than "
    f"{MAX_NESTING} levels deep, past what the shell guards read. Split it into simpler "
    "commands."
)


class TooDeepError(Exception):
    """The groups or expansions nest deeper than :data:`MAX_NESTING`.

    :func:`simple_commands` raises it for a guard, which must refuse
    what it cannot read; its message is :data:`TOO_DEEP_VERDICT`.  So
    it does, with :data:`UNREAD_DELIMITER_VERDICT`, for a here-document
    that ends where only bash knows: the text after it is unread too.
    And with :data:`UNREAD_VALUES_VERDICT` for variables that hold more
    text than a scan reads: the commands after them are unread as well.
    """


#: Text the model sees instead of running a command whose variables hold that much text.
UNREAD_VALUES_VERDICT = (
    "REFUSED: the command did not run: the values of its variables, counted each time one "
    f"is expanded, come to more than {_MAX_READ:,} characters, past what the shell guards "
    "read. Put the long text in a file, or split the command into simpler ones."
)


#: Text the model sees instead of running a command with such a here-document.
UNREAD_DELIMITER_VERDICT = (
    "REFUSED: the command did not run: a here-document delimiter in it is a $\"...\" string, "
    "or a $'...' string with an escape that bash versions decode differently (\\c, \\u, \\U, "
    "or a number below 2 or above 0x7e), so the shell guards cannot tell where the "
    "here-document ends. Quote the delimiter plainly: <<'EOF'."
)
#: An escape of an ANSI-C string ``$'...'``: a character's letter, one to three octal
#: digits, ``\xHH`` or ``\x{H...}``, which every bash from 3.2 on decodes alike; or ``\c``,
#: ``\u`` or ``\U``, which two versions or two locales decode differently.
_ANSI_C_ESCAPE = re.compile(
    r"\\(?:(?P<char>[abeEfnrtv\\'\"?])|(?P<octal>[0-7]{1,3})"
    r"|x(?P<hex>\{[0-9A-Fa-f]*\}?|[0-9A-Fa-f]{1,2})|[cuU])")
_ANSI_C_CHARACTERS = dict(zip("abeEfnrtv\\'\"?", "\a\b\x1b\x1b\f\n\r\t\v\\'\"?", strict=True))
#: One that may decode to syntax the text does not show: a number (``\x24``), ``\c`` (it
#: takes the character after it, a backslash too) or a backslash.
_DECODED = re.compile(r"\\[0-7xuUc\\]")


#: Text the model sees instead of running a command the scan cannot finish reading.
CONTINUED_VERDICT = (
    "REFUSED: the command did not run: it holds more backslash-newline pairs inside quoted "
    "strings, comments and quoted here-documents than the shell guards read. Join the "
    "continued lines, or split it into simpler commands."
)


class _KeptPairsError(Exception):
    """Backslash-newline pairs a scan removed where bash keeps them (:class:`_Pairs`).

    Attributes:
        raw: The text they are in, as written.
        origin: Where in it the backslash of each is.
    """

    def __init__(self, raw: str, origin: list[int]) -> None:
        """Name the pairs at *origin* in *raw*."""
        super().__init__()
        self.raw = raw
        self.origin = origin


class _Pairs:
    """A text as bash's lexer reads it: without its backslash-newline pairs.

    Bash drops such a pair while it reads characters, before it knows
    what token it is in (``<\\<newline><``, ``$\\<newline>(``,
    ``i\\<newline>f``, in double quotes, in backticks, in an unquoted
    here-document body), and keeps it in four places: a ``'...'``
    string, a ``$'...'`` string, a comment (which ends at the newline)
    and the body of a here-document with a quoted delimiter.  So every
    pair is removed before the scan, and the scanner tells where one
    must stay (:meth:`keep`, :meth:`keep_body`): the scan then starts
    over with those kept (:func:`simple_commands`).

    Attributes:
        raw: The text as written.
        text: It without the pairs removed.
        at: Where in *text* each removed pair was (the index of the
            character after it), ascending.
        origin: Where in *raw* the backslash of each is, parallel to *at*.
    """

    def __init__(self, raw: str = "", kept: Collection[int] = ()) -> None:
        """Remove the pairs of *raw*, but those whose backslash is at an index of *kept*."""
        self.raw = raw
        self.at: list[int] = []
        self.origin: list[int] = []
        parts: list[str] = []
        last = 0
        for pair in _PAIR.finditer(raw) if "\\\n" in raw else ():
            start = pair.start(1)
            if start not in kept:
                parts.append(raw[last:start])
                self.at.append(start - 2 * len(self.at))
                self.origin.append(start)
                last = start + 2
        self.text = "".join((*parts, raw[last:])) if parts else raw

    def keep(self, start: int, end: int, first: bool = False) -> None:
        """Start the scan over when a pair was removed from *start* to *end* of the text.

        Both inclusive: the content of a quoted string and the index of
        its closing quote, or a comment and the index of the newline
        that ends it.  A comment ends at its first pair (*first*): only
        that one is kept.

        Raises:
            _KeptPairsError: Those pairs.
        """
        low, high = bisect_left(self.at, start), bisect_right(self.at, end)
        if low < high:
            raise _KeptPairsError(self.raw, self.origin[low:low + 1 if first else high])

    def keep_body(self, start: int, delimiter: str, strip_tabs: bool) -> None:
        """Start the scan over when a pair was removed in the quoted here-document body at *start*.

        The body is read from the text as written, as bash reads it: to
        the line that is the *delimiter* (its leading tabs stripped when
        *strip_tabs*), or to the end.

        Raises:
            _KeptPairsError: The pairs of the body.
        """
        low = bisect_left(self.at, start)
        if low == len(self.at):
            return
        raw = self.raw
        pos = start + 2 * low  # where the body starts as written
        while pos < len(raw):
            newline = raw.find("\n", pos)
            end = len(raw) if newline < 0 else newline
            line = raw[pos:end]
            if (line.lstrip("\t") if strip_tabs and line != delimiter else line) == delimiter:
                break
            pos = end + 1
        high = bisect_left(self.origin, pos)
        if low < high:
            raise _KeptPairsError(raw, self.origin[low:high])


#: Those of a text that is read as it stands: a copy of text already joined.
_NO_PAIRS = _Pairs()


def _ansi_c_character(escape: re.Match[str]) -> str:
    """Return the character the *escape* (:data:`_ANSI_C_ESCAPE`) stands for.

    A backslash before any other character stays, and is not matched.

    Raises:
        TooDeepError: Only bash knows it: ``\\c``, ``\\u``, ``\\U`` (no digits here: 0), or
            a number that is no ASCII character bash keeps as it is (a NUL ends the
            string, 0x01 and 0x7f are marks of bash's own, 0x80 and above are bytes).
    """
    if escape["char"]:
        return _ANSI_C_CHARACTERS[escape["char"]]
    digits = escape["octal"] or (escape["hex"] or "").strip("{}") or "0"
    code = int(digits, 8 if escape["octal"] else 16) & 0xFF
    if code < 0x02 or code > 0x7E:
        raise TooDeepError(UNREAD_DELIMITER_VERDICT)
    return chr(code)


# On Windows host commands run in Git Bash, which spells ``C:\Users\x``
# as ``/c/Users/x`` and reads a ``C:/x`` word as that path.
_WINDOWS = sys.platform == "win32"
_DRIVE = re.compile(r"[A-Za-z]:(?:[/\\]|\Z)")
_BASH_DRIVE = re.compile(r"/[A-Za-z](?:/|\Z)")


@dataclass
class Word:
    """One shell word after quote removal.

    Attributes:
        text: The word without its quotes; expansions are kept as written.
        literal: Whether the shell passes *text* on unchanged: no
            parameter, command substitution, leading ``~`` or glob in it.
        assignment: Whether an unquoted ``NAME=``, ``NAME+=`` or
            ``NAME[...]=`` starts it: before the command word the shell
            assigns such a word, and runs any other (``'T=x'``, ``T\\=x``).
        path: Whether the word, though not literal, is one field that ends
            in a literal ``/name`` (``"$D/bin/python"``, ``~/bin/x``): as a
            command word it runs the program of that name, whatever the
            rest expands to.  Not an unquoted expansion's (bash may split
            it, or drop it), nor one with ``@`` (``"$@"/x`` is several
            words) or a glob.
        decoded: The text the shell passes on, where the word is literal
            but for its unquoted ``$'...'`` strings (*text* keeps them as
            written, and the word is not literal): each string as every
            bash decodes it.  ``None`` for any other word, and where only
            bash can decode a string.
    """

    text: str
    literal: bool
    assignment: bool = False
    path: bool = False
    decoded: str | None = field(default=None, compare=False, repr=False)

    def passed(self) -> str | None:
        """Return the text the shell passes on, or ``None`` when only the shell knows it."""
        return self.text if self.literal else self.decoded

    def value(self, assignment: re.Match[str]) -> str | None:
        """Return the value the word assigns, or ``None`` when only the shell knows it.

        Args:
            assignment: The match of :data:`_ASSIGNMENT` at the start of :attr:`text`.
        """
        passed = self.passed()
        return None if passed is None else passed[assignment.end():]


@dataclass
class Group:
    """Commands that share redirections and pipes.

    A subshell, a substitution, a brace group or a compound command: a
    redirection or a pipe after its end applies to the commands in it.

    Attributes:
        kind: ``"("`` (a subshell), ``"$("`` (a command or process
            substitution), ``"`"`` (a backtick substitution), ``"(("``
            (an arithmetic command, which holds no commands but those
            of its command substitutions), ``"[["`` (a conditional
            command: its one simple command and those of its
            substitutions) or the reserved word that opened it
            (``"{"``, ``"if"``, ``"for"``...).
        cwd: The directory it starts in, where its redirections resolve;
            ``None`` when only the shell knows it.
        group: The group it is in; ``None`` at the top level.
        conditional: Whether a ``cd`` in it may not run: it is a compound
            command, it follows ``||``, or its group is conditional (a
            subshell starts afresh).
        outputs: The files its standard output is redirected to.
        redirections: Every redirection after it, in order:
            ``(descriptor, operator, operand)``, the descriptor as
            written (``""`` for the operator's default).
        heredoc: Whether it holds a here-document its commands may read: its
            standard input, or one on another descriptor (``3<<EOF``, read
            through ``<&3`` or ``/dev/fd/3``).
        heredoc_body: The text the one that is its standard input hands on
            (see :attr:`SimpleCommand.heredoc_body`).
        piped: Whether a pipe is its standard input.
        reader: The command or group its standard output is piped to.
        after_and: Whether it (or the pipeline it is in) follows ``&&``.
        background: Whether ``&`` ends it (or the pipeline it ends).
        negated: Whether ``!`` precedes it (or the pipeline it starts):
            the shell inverts its exit status.
        asynchronous: Whether the shell goes on while it runs (see
            :attr:`SimpleCommand.asynchronous`).  A substitution that
            is, is a process substitution: its output is no word but
            a stream.  That of ``<(...)`` is read by its :attr:`reader`:
            the simple command that holds it, or the group it is the
            standard input of (``None`` on another descriptor of a
            group and in a loop's header); that of ``>(...)`` is the
            output the shell has where it expands it (no
            :attr:`reader`).
    """

    kind: str
    cwd: str | None
    group: Group | None
    conditional: bool = False
    outputs: list[Word] = field(default_factory=list)
    redirections: list[tuple[str, str, Word]] = field(default_factory=list)
    heredoc: bool = False
    heredoc_body: str | None = None
    piped: bool = False
    reader: SimpleCommand | Group | None = None
    after_and: bool = False
    background: bool = False
    negated: bool = False
    asynchronous: bool = False


@dataclass
class SimpleCommand:
    """One simple command of a command line.

    Attributes:
        cwd: The absolute directory it runs in; ``None`` when only the
            shell knows it (after ``cd "$D"``, say).
        group: The group it is in; ``None`` at the top level.
        words: Its words (assignments, wrappers, the command word and
            its arguments), without the redirections.
        outputs: The files its standard output is redirected to (those
            of a ``{name}>file`` or ``11>file`` too: bash truncates the
            file for the descriptor it opens, sh writes the output to it
            and reads the ``{name}`` or ``11`` as a word, kept in
            :attr:`words`).
        redirections: Every redirection of it, in order: ``(descriptor,
            operator, operand)``, the descriptor as written (``""`` for
            the operator's default, ``{name}`` for one bash opens), the
            operand of a here-document its delimiter.  What
            :attr:`outputs` and :attr:`heredoc` leave out (``< file``,
            ``2> file``, ``1>&2``) is here, for a guard that must know a
            command's every redirection.
        heredoc: Whether it holds a here-document it may read: its own
            standard input, or one on another descriptor (``3<<EOF``)
            unless it is a ``cat`` or a ``tee`` that reads its standard
            input and its operands only (:func:`_reads_heredoc`).
        heredoc_body: The text the here-document that is its standard
            input hands on, every line newline-terminated (the leading
            tabs of a ``<<-`` body stripped): the body as written after a
            quoted delimiter, and after an unquoted one the body without
            a ``$`` or a backtick, with the backslash of ``\\\\`` dropped
            and ``\\<newline>`` joined, as the shell hands it on.  ``None``
            when there is no such here-document, the shell would expand
            it, or the command does not end it.
        piped: Whether a pipe is its own standard input.
        reader: The command or group its standard output is piped to.
        conditional: Whether it follows ``||``: it runs only when the
            commands before it failed.
        after_and: Whether it (or the pipeline it is in) follows ``&&``:
            it runs only when the commands before it succeeded, and the
            list fails when it fails.
        background: Whether ``&`` ends it (or the pipeline it ends): the
            shell goes on without its exit status.
        negated: Whether ``!`` precedes it (or the pipeline it starts):
            the shell inverts its exit status.
        script: The group of the commands of the literal text it runs
            (``eval``, a shell's ``-c``; :func:`command_text`; the body of
            a function it calls), which share its redirections, its
            here-document and its pipes; ``None`` when it runs no such text.
        unknown: Whether only the shell knows what it runs: its command
            word is neither literal nor a path (``$X``, ``"$@"``: nothing,
            a wrapper, ``tee``, ``cd`` or ``eval``), bash no longer runs
            the text as written by then (:meth:`_Scanner.unfollowed`), or
            it evaluates arithmetic the scan cannot follow, which may run
            a command (``let "$X"``; one without words stands for such an
            evaluation in an expansion).  A guard takes it for a command
            that copies its input into any file.
        substitutions: The substitutions in its words: their commands
            read the pipe it reads, not its other redirections.
        derived: Whether it is not a command of the text where it stands:
            one of a function's body as a call runs it (in the call's
            :attr:`script`), or what stands for an evaluation or a call
            only the shell knows (no words).  :func:`simple_commands`
            returns those only when asked.
        asynchronous: Whether the shell goes on while it runs: it is a
            command of a list that ``&`` ends (each of ``a && b | c &``,
            where :attr:`background` marks ``c`` alone), or a coprocess.
            The commands written after it may run before it does.
    """

    cwd: str | None
    group: Group | None
    words: list[Word] = field(default_factory=list)
    outputs: list[Word] = field(default_factory=list)
    redirections: list[tuple[str, str, Word]] = field(default_factory=list)
    heredoc: bool = False
    heredoc_body: str | None = None
    piped: bool = False
    reader: SimpleCommand | Group | None = None
    conditional: bool = False
    after_and: bool = False
    background: bool = False
    negated: bool = False
    script: Group | None = None
    unknown: bool = False
    substitutions: list[Group] = field(default_factory=list)
    derived: bool = False
    asynchronous: bool = False

    def argv(self) -> list[Word]:
        """Return the command word and its arguments: assignments and wrappers dropped.

        A wrapper is matched by its basename (``/usr/bin/env``); its
        options, the value of an option that takes one (``sudo -u
        root``, ``timeout -s KILL``) and ``timeout``'s duration are
        dropped with it.  Under a wrapper that runs the program in
        another directory (``env -C dir``) the arguments are not literal:
        an operand is a path there, which only the shell resolves; the
        text of a shell's ``-c`` is no path and stays as written.  The
        words ``env -S 'text'`` makes of its text replace the option in
        :attr:`words` (:func:`_command_word`).
        """
        index, moved = _command_word(self.words)
        argv = self.words[index:]
        if not moved or not argv:
            return argv
        script = (shell_script_index([word.text for word in argv[1:]])
                  if posixpath.basename(argv[0].text) in SHELLS else None)
        return [argv[0], *(word if at == script else Word(word.text, literal=False)
                           for at, word in enumerate(argv[1:]))]

    def program(self) -> str:
        """Return the basename of the program it runs (``""`` for a bare redirection)."""
        argv = self.argv()
        return posixpath.basename(argv[0].text) if argv else ""

    def reads_heredoc(self) -> bool:
        """Return whether it may read a here-document: its own or a group's."""
        return self.heredoc_reader() is not None

    def heredoc_reader(self) -> SimpleCommand | Group | None:
        """Return the node that holds the here-document it may read, or ``None``.

        Itself, or the group it is in (:attr:`heredoc`).
        """
        node: SimpleCommand | Group | None = self
        while node is not None:
            if node.heredoc:
                return node
            node = node.group
        return None


def _command_word(words: list[Word]) -> tuple[int, bool]:
    """Return the index of the command word in *words*, and whether a wrapper moves its program.

    The index is past the last word when there is no command word; a
    wrapper moves the program it runs to another directory with one of
    :data:`_PREFIX_CHDIR_OPTIONS` (``env -C dir``).  The shell's own
    assignments start unquoted (:attr:`Word.assignment`); ``env`` and
    ``sudo`` take theirs before the program (:data:`_PREFIX_ASSIGNMENTS`:
    ``env 'T=a b' cmd``, ``env a-b=1 cmd``), and after another wrapper
    such a word is the program.  An option takes the next word as its
    value when it is one of :data:`_PREFIX_VALUE_OPTIONS` or a cluster
    that ends with one (``env -iu T``).  ``timeout``'s first operand is
    its duration.  ``env``'s split option (:data:`_SPLIT_STRING`) is
    replaced in *words* by the words ``env`` makes of its text, read on
    from there (``env -S 'tee a' b`` runs ``tee a b``); without its text
    (missing, or not read yet) there is no command word, and with text
    only the shell or ``env`` knows (:func:`_split_string`) the option
    stays and is the command word: a command no one can name.  It is no
    option once ``env``'s options ended, at ``--``, ``-`` or an assignment.
    """
    value_options: frozenset[str] = frozenset()
    cluster: re.Pattern[str] | None = None
    chdir_options: tuple[str, ...] = ()
    value_next = chdir = duration = ended = False
    wrapper = ""
    index = 0
    while index < len(words):
        word = words[index]
        text = word.text
        name = posixpath.basename(text)
        if wrapper in _PREFIX_ASSIGNMENTS:
            assigned = "=" in text and not text.startswith(_PREFIX_ASSIGNMENTS[wrapper])
        else:
            assigned = word.assignment and not wrapper
        if value_next or assigned:
            ended = ended or not value_next
            value_next = False
        elif duration and not text.startswith("-"):
            duration = False
        elif name in _PREFIXES:
            wrapper = name
            value_options = _PREFIX_VALUE_OPTIONS.get(name, frozenset())
            cluster = _PREFIX_VALUE_CLUSTER.get(name)
            chdir_options = _PREFIX_CHDIR_OPTIONS.get(name, ())
            duration, ended = name == "timeout", False
        elif (wrapper == "env" and not ended
              and (option := _SPLIT_STRING.fullmatch(text)) is not None):
            attached = bool(option["text"] or option["equals"])  # else the next word
            split = [Word(option["text"], word.literal)] if attached else words[index + 1:index + 2]
            if not split:
                return len(words), chdir
            parts = _split_string(split[0].text) if split[0].literal else None
            if parts is None:
                return index, chdir
            flags = [Word(f"-{option['flags']}", True)] if option["flags"] else []
            words[index:index + 2 - attached] = [*flags, *(Word(part, True) for part in parts)]
            continue
        elif text.startswith("-"):
            value_next = text in value_options or (
                cluster is not None and cluster.fullmatch(text) is not None)
            chdir = chdir or text.startswith(chdir_options)
            ended = ended or text in ("-", "--")
        else:
            return index, chdir
        index += 1
    return len(words), chdir


def handed_text(body: str, expands: bool) -> str | None:
    """Return the text a here-document *body* hands on, or ``None`` when the shell expands it.

    After a quoted delimiter (*expands* false) the body is handed on as
    written.  After an unquoted one the shell expands a ``$`` or a
    backtick, so the text is unknown; without them it only drops the
    backslash of ``\\\\`` and joins the lines of ``\\<newline>``.

    Args:
        body: The body, every line newline-terminated.
        expands: Whether the delimiter was unquoted.
    """
    if not expands:
        return body
    if "$" in body or "`" in body:
        return None
    parts: list[str] = []
    position = 0
    while position < len(body):
        char = body[position]
        if char == "\\" and body[position + 1:position + 2] in ("\\", "\n"):
            if body[position + 1] == "\\":
                parts.append("\\")
            position += 2
        else:
            parts.append(char)
            position += 1
    return "".join(parts)


def _continued(line: str) -> bool:
    """Return whether *line* ends in an unescaped backslash: the next line continues it.

    Bash removes a ``\\``-newline pair while reading an unquoted
    here-document line, before it looks for the delimiter; the backslash
    of a ``\\\\`` pair does not count (the pairs are read from the left,
    so an odd run at the end leaves one unpaired).
    """
    return (len(line) - len(line.rstrip("\\"))) % 2 == 1


def descriptor(fd: str, operator: str) -> str:
    """Return the descriptor the redirection *fd* *operator* is of, as bash numbers it.

    The number written before the operator, without its leading zeros
    (``01>`` is ``1>``); without one the operator's default: ``0`` for
    an input, ``1`` for an output.  ``""`` for one bash opens anew
    (``{name}>``).

    Args:
        fd: What stands before the operator (:attr:`SimpleCommand.redirections`).
        operator: The redirection operator.
    """
    if not fd:
        return "0" if operator.startswith("<") else "1"
    return str(int(fd)) if fd.isdigit() else ""


def duplicated(operator: str, operand: Word) -> str | None:
    """Return the descriptor the redirection *operator* *operand* makes a copy of.

    As bash numbers it: ``>&3`` and ``<&3`` copy descriptor 3, as do
    ``>&03`` and the move ``>&3-``, which closes it afterwards; ``"-"``
    for ``>&-``, which closes.  ``None`` for any other redirection: of
    a file, or of a word only the shell knows.

    Args:
        operator: The redirection operator.
        operand: Its operand.
    """
    if operator not in ("<&", ">&"):
        return None
    if operand.text == "-":
        return "-"
    match = _DUPLICATED.fullmatch(operand.text)
    return None if match is None else str(int(match[1]))


def inherited_cdpath() -> str:
    """Return the ``CDPATH`` a shell that this process starts inherits (``""`` when unset)."""
    return os.environ.get("CDPATH", "")


def _holds(words: list[Word]) -> bool:
    """Return whether the command of *words* is one of :data:`_KEEPS` (``printf``: with ``-v``)."""
    name = posixpath.basename(words[0].text)
    return name in _KEEPS and (name != "printf" or any(
        not word.literal or word.text.startswith("-v") for word in words[1:2]))


def _may_run_unread(command: SimpleCommand, functions: _Functions) -> bool:
    """Return whether *command* may run shell text the scan did not read.

    It is one only the shell knows (:attr:`SimpleCommand.unknown`) that
    nothing names (an evaluation), that an expansion names, or that runs
    or evaluates what it is handed (a function, :data:`_EVALUATES`,
    :data:`_KEEPS`); a program is not, when its name is all the shell
    knows better (a sourced file may have redefined it: a file's text is
    not the command's).  Or its operands name a shell while the scan
    read no script of it (``find -exec env sh -c ...``, ``flock f setsid
    -w bash -c ...``): whatever runs the operands, however deep the
    runners nest.  Not a command that runs none of its operands
    (:data:`_RUNS_NO_OPERAND`), nor one that runs them elsewhere
    (:data:`ELSEWHERE`).
    """
    words = command.words[_command_word(command.words)[0]:]
    if not words:
        return command.unknown
    name = posixpath.basename(words[0].text)
    prefix = [word.text for word in command.words[:len(command.words) - len(words)]]
    if "xargs" in map(posixpath.basename, prefix) and any(map(_REPLACES.match, prefix)):
        return True  # the command it runs holds text of its input
    if command.unknown:
        return (not (words[0].literal or words[0].path) or name in functions
                or words[0].text in functions or name in _EVALUATES or _holds(words))
    # a shell or an ``eval`` whose text the scan read runs that; a program that runs a shell
    # may hand it another text (``xargs -I``, ``find -exec`` with ``{}``)
    return (not (command.script is not None and words[0].literal
                 and name in SHELLS | _SOURCING_BUILTINS)
            and not (words[0].literal and name in _RUNS_NO_OPERAND | ELSEWHERE)
            and any(((word.literal or word.path) and posixpath.basename(word.text) in SHELLS)
                    or word.text in _SHELL_VALUED for word in words[1:]))


def _keeps(command: SimpleCommand, functions: _Functions) -> bool:
    """Return whether the shell keeps what *command* is handed, or may run it unread.

    A variable or the positional parameters hold it afterwards
    (:data:`_KEEPS`, a function's parameters), or the command may run it
    (:func:`_may_run_unread`, a command word only the shell knows); any
    other program takes its operands and its input as data.
    """
    words = command.words[_command_word(command.words)[0]:]
    if not words:
        return False
    return (not (words[0].literal or words[0].path)
            or posixpath.basename(words[0].text) in functions
            or _may_run_unread(command, functions) or _holds(words))


def _copies(command: SimpleCommand) -> bool:
    """Return whether *command* writes what it is handed to its output (:data:`_COPIES`)."""
    words = command.words[_command_word(command.words)[0]:]
    return bool(words) and (not (words[0].literal or words[0].path)
                            or posixpath.basename(words[0].text) in _COPIES)


def _nested_shell(argv: list[Word]) -> list[Word]:
    """Return the command line of the shell *argv* may run without being one, or ``[]``.

    The words from the first operand that names a shell on: a program
    may run the command its operands name (``setsid nohup sh -c text``,
    ``find -exec sh -c text {} \\;``), however deep the runners nest.
    *argv* itself as a shell's when an expansion names its command
    (``"$sh" -c text``): a shell would run that text.
    """
    if argv and not (argv[0].literal or argv[0].path):
        return [Word("sh", True), *argv[1:]]
    if argv and posixpath.basename(argv[0].text) in _RUNS_NO_OPERAND | ELSEWHERE:
        return []
    for index in range(1, len(argv)):
        word = argv[index]
        if (word.literal or word.path) and posixpath.basename(word.text) in SHELLS:
            return argv[index:]
    return []


def _script_words(argv: list[Word]) -> list[Word]:
    """Return the words of *argv* whose text :func:`command_text` returns."""
    if argv[0].text == "eval":
        return argv[1:]
    index = shell_script_index([word.text for word in argv[1:]])
    return [] if index is None else argv[index + 1:index + 2]


def _carries(command: SimpleCommand, functions: _Functions, seen: set[int],
             own: bool = True) -> bool:
    """Return whether the text *command* hands on may reach a shell the scan did not read.

    Through itself (:func:`_keeps`) or through what reads its output: a
    pipe or a process substitution, whatever *command* is, and a
    ``$(...)`` when it copies (:func:`_copies`).  What another program
    prints into a word is its own, as a file's text is: its operands are
    its patterns and its code (``n=$(awk 'NR<5' f)``).  *seen*: the
    commands and groups found to carry nothing so far, by ``id``; the
    commands that copy are asked first.  Not *own*: the command's text
    is a script the scan read (:attr:`_Functions.scripted`), so only
    what reads its output is asked.
    """
    node: SimpleCommand | Group | None = command
    while node is not None and id(node) not in seen:
        if own or node is not command:
            seen.add(id(node))
            if isinstance(node, SimpleCommand) and _keeps(node, functions):
                return True
        if isinstance(node.reader, Group):
            return True  # its commands read the pipe
        if node.reader is not None:
            node = node.reader
        else:
            node = node.group
            if node is not None and node.kind in _SUBSTITUTIONS:
                return node.asynchronous or _copies(command)
    return False


def simple_commands(command: str, cwd: str | None, optimistic: bool = False,
                    isdir: Callable[[str], bool] | None = None,
                    derived: bool = False,
                    cdpath: Callable[[], str] | None = None) -> list[SimpleCommand]:
    """Return the simple commands of the bash text *command*, in the order they start.

    Args:
        command: The shell command text, possibly several lines.
        cwd: The absolute POSIX directory it starts in; ``None`` when unknown.
        optimistic: Assume every list ran through: an assignment or a
            ``cd`` after ``&&`` or ``||`` holds for the rest of the
            command (``cd d && T=tmp/x && a; cp b $T/``), not only for
            the rest of its list, and a command nested past
            :data:`MAX_NESTING` is read up to the nesting.  A guard that
            must not let a write through keeps the default; a gate that
            judges what the agent changed reads the command as it was
            meant to run.
        isdir: Whether an absolute bash path is a directory where the
            command runs, its links followed; ``None`` when the caller
            cannot tell.  With it a ``cd`` through ``..`` whose logical
            path names no directory, while the path as written does,
            leaves a directory only the shell knows (:func:`_cd`).
        derived: Return the commands the text does not hold where they
            run too (:attr:`SimpleCommand.derived`): those of a
            function's body at each call, with the variables, the
            directory and the input of the call, and one without words
            for what only the shell knows.  For a guard that follows
            where the input of a command goes.
        cdpath: What tells the ``CDPATH`` the command's shell inherits
            where it runs (``""`` for none; :func:`inherited_cdpath`
            for a shell this process starts).  With one, bash looks
            there first for the operand of a ``cd`` that no ``/``, ``.``
            or ``..`` starts, so the directory is only the shell's to
            know unless the command assigned ``CDPATH`` itself.  Asked
            once, at the first such ``cd``.  ``None``: the caller
            accounts for an inherited ``CDPATH`` itself, or the shell
            inherits none.

    Raises:
        TooDeepError: The groups or expansions nest deeper than
            :data:`MAX_NESTING`, so the commands past them are unread,
            and the scan is not *optimistic*: a guard refuses the command.
            Likewise (:data:`CONTINUED_VERDICT`) when the scan started
            over more often than it may (:class:`_Pairs`), and
            (:data:`UNREAD_VALUES_VERDICT`) when the variables hold more
            text than it reads (:meth:`_Scanner.read`).
    """
    kept: dict[str, set[int]] = {}
    shared: Callable[[], str] | str | None = cdpath
    restarts = min(_MAX_RESTARTS, max(4, _RESTART_WORK // (len(command) + 1)))
    while True:
        functions = _Functions(shared, kept)
        scanner = _Scanner(command, cwd, None, optimistic=optimistic, isdir=isdir,
                           functions=functions)
        try:
            scanner.command_list()
            scanner.scripts()
        except _KeptPairsError as found:  # bash keeps them: the text is read again with them
            if restarts:
                restarts -= 1
                kept.setdefault(found.raw, set()).update(found.origin)
                shared = functions.cdpath
                continue
            if not optimistic:
                raise TooDeepError(CONTINUED_VERDICT) from None
        except TooDeepError:
            if not optimistic:
                raise
            # the gate's reading: the commands before the nesting
        break
    # Text that can spell a ``<`` and may reach a shell the scan did not read may be a
    # here-document there: how the text is put together (a ``printf`` format, ``+=``, an
    # array, two pieces of one ``<`` each) is no evidence.
    if not optimistic and any(_may_run_unread(simple, functions) for simple in scanner.commands):
        seen: set[int] = set()
        if functions.stored or any(
                _carries(holder, functions, seen)
                for holder in sorted(functions.holders.values(), key=_copies, reverse=True)
        ) or any(_carries(holder, functions, set(), own=False)
                 for holder in functions.scripted.values()):
            scanner.hidden()
    if derived:
        return scanner.commands
    return [simple for simple in scanner.commands if not simple.derived]


def receiver(node: SimpleCommand | Group) -> SimpleCommand | Group | None:
    """Return what gets the standard output of *node* when it is not redirected to a file.

    That is the command it is piped to, else its group, whose output it
    is part of; ``None`` at the top level and in a substitution, whose
    output is a word of another command rather than a stream.

    Args:
        node: A simple command or a group.
    """
    if node.reader is not None:
        return node.reader
    group = node.group
    return None if group is None or group.kind in _SUBSTITUTIONS else group


def resolve(cwd: str | None, word: Word) -> str | None:
    """Return the absolute path *word* names in the directory *cwd*.

    Args:
        cwd: The absolute POSIX directory; ``None`` when unknown.
        word: A path operand; a Windows drive path (``C:/x``) is
            absolute (see :func:`bash_path`).

    Returns:
        The normalised path, or ``None`` when only the shell knows it
        (an expansion in *word*, or a relative *word* in an unknown *cwd*).
        A ``..`` stays in it with what precedes it, for the filesystem to
        resolve: after a symbolic link it is not the directory the link
        is in (``link/../f``).  The guards tell a device and a descriptor
        by the path's text (``/dev/null``, ``/proc/self/fd/1``), which a
        ``..`` belies: such a path is ``None`` too.
    """
    if not word.literal:
        return None
    text = bash_path(word.text)
    if not text.startswith("/"):
        if cwd is None:
            return None
        text = posixpath.join(cwd, text)
    parts = text.split("/")
    if ".." not in parts:
        return posixpath.normpath(text)
    first = parts.index("..")
    path = posixpath.join(posixpath.normpath("/".join(parts[:first]) or "/"),
                          *(part for part in parts[first:] if part not in ("", ".")))
    by_name = any(("/" + spelt.lstrip("/")).startswith(("/dev/", "/proc/"))
                  for spelt in (path, posixpath.normpath(path)))
    return None if by_name else path


def bash_path(path: str) -> str:
    """Return the host *path* the way the bash that runs the commands spells it.

    On Windows that bash is Git Bash, where ``C:\\Users\\x`` is
    ``/c/Users/x``; any other path comes back unchanged.
    """
    if _WINDOWS and _DRIVE.match(path):
        return f"/{path[0].lower()}{path[2:]}".replace("\\", "/")
    return path


def host_path(path: str) -> str:
    """Return the host path the absolute bash *path* names: the inverse of :func:`bash_path`.

    Git Bash's ``/c/Users/x`` is ``C:/Users/x`` on Windows; any other
    path (Git Bash's own ``/tmp`` too) comes back unchanged.
    """
    if _WINDOWS and _BASH_DRIVE.match(path):
        return f"{path[1].upper()}:{path[2:] or '/'}"
    return path


def walks(path: str, isdir: Callable[[str], bool]) -> bool:
    """Return whether bash walks the absolute *path* by its names, as a logical ``cd`` does.

    Each ``..`` must leave a directory and the names left must be one:
    the ``cd`` then enters the normalised path.  Where it does not,
    bash takes the path as written, its links followed, and fails when
    that is no directory either.

    Args:
        path: The path of a ``cd`` as written (:func:`cd_path`).
        isdir: Whether an absolute bash path is a directory where the
            command runs, its links followed.
    """
    names: list[str] = []
    for part in path.split("/")[1:]:
        if part != "..":
            names.append(part)
        elif not isdir("/" + "/".join(names)):
            return False
        else:
            del names[-1:]
    return isdir("/" + "/".join(names))


def cd_path(cwd: str | None, args: list[Word], searched: bool = False) -> str | None:
    """Return the path ``cd args`` names in *cwd*, as written; ``None`` when only the shell knows.

    The shell alone knows ``$HOME`` and ``$OLDPWD`` (no operand, ``-``),
    what it makes of several operands (bash 4.4 and later fail, an
    older one enters the first), and where a physical ``cd -P`` ends:
    the directory the symbolic links in its path lead to.  The last of
    ``-L`` and ``-P`` decides (an option the shell expands may be
    ``-P``).  A ``..`` stays in the path with the name before it
    (:func:`resolve`): bash drops the two only where it walks the path
    (:func:`walks`).

    Args:
        cwd: The absolute POSIX directory the ``cd`` starts in; ``None`` when unknown.
        args: The words after ``cd``.
        searched: Whether ``CDPATH`` may be set: bash looks there first
            for an operand that no ``/``, ``.`` or ``..`` starts.
    """
    operands = [word for word in args if not word.text.startswith("-") or word.text == "-"]
    options = "".join(word.text for word in args if word.text.startswith("-"))
    physical = options.rfind("P") > options.rfind("L") or "$" in options or "`" in options
    if len(operands) != 1 or operands[0].text == "-" or physical:
        return None
    text = bash_path(operands[0].text)
    if searched and not (text.startswith(("/", "./", "../")) or text in (".", "..")):
        return None
    return resolve(cwd, operands[0])


def _cd(cwd: str | None, args: list[Word], searched: bool,
        isdir: Callable[[str], bool] | None) -> str | None:
    """Return the directory ``cd args`` moves *cwd* to; ``None`` when only the shell knows it.

    A logical ``cd``, the default, drops a ``..`` of its path
    (:func:`cd_path`) with the name before it, as bash does; where bash
    cannot walk the logical path (:func:`walks`; *isdir*, when the
    caller can tell) while the path as written is a directory, it
    follows the links of that one.
    """
    path = cd_path(cwd, args, searched)
    if path is None:
        return None
    logical = posixpath.normpath(path)
    if len(logical) > _MAX_PATH:  # no path names it: where relative steps led the shell
        return None
    if isdir is None or logical == path or walks(path, isdir) or not isdir(path):
        return logical
    return None


def _pipe(source: SimpleCommand | Group, reader: SimpleCommand | Group) -> None:
    """Pipe the standard output of *source* to *reader*: that of the text it runs too."""
    source.reader = reader
    if isinstance(source, SimpleCommand) and source.script is not None:
        source.script.reader = reader


def _target(operand: str) -> str | None:
    """Return the variable the *operand* of an assigning builtin names; ``None`` when unknown.

    The name starts the operand (``T``, ``T=x``, an element's ``T[0]``);
    ``""`` when none does.  A name the shell expands (``"$N"``, ``T$N=x``)
    is unknown: it may be any variable's.
    """
    named = _NAME.match(operand)
    name = named[0] if named else ""
    return None if operand.startswith(("$", "`"), len(name)) else name


class _Assigned(dict[str, "str | None"]):
    """The variables as the later assignments of a command see them (:meth:`_Scanner.assign`).

    ``changed`` once one is set, as the scan of a word does to make it unknown (``${T:=x}``).
    ``unknown`` holds the shell's variables with no value known, as Bash 3 expands those
    assignments (:meth:`_Scanner.assigning`), once the first of them is scanned that way.
    """

    changed = False
    unknown: _Assigned | None = None

    def __setitem__(self, name: str, value: str | None) -> None:
        self.changed = True
        super().__setitem__(name, value)


#: A subshell's own state (:meth:`_Scanner.scope`): the directory, the variables, the exported
#: names, the files of the shell's output, ``set -P``, whether a name reference may exist,
#: whether bash still runs the text as written, whether a startup variable may be exported,
#: and the text of each function's body.
_Scope = tuple[str | None, dict[str, str | None], set[str], tuple[Word, ...], bool, bool, bool,
               bool, dict[str, str | None]]


def _followed(words: list[Word], letters: frozenset[str], names_only: bool) -> bool:
    """Return whether bash still runs the text as the scan reads it after the options *words*.

    Those of ``set`` or of a shell's command line, up to the first word
    that is none (a positional parameter, the script), or with
    *names_only* those of ``shopt``: clusters of *letters* and, after
    ``-o`` or ``-O`` and for ``shopt``, the names of :data:`_OPTIONS`.
    (``--rcfile file`` names what an interactive shell alone reads.)
    """
    named = False  # the word names an option: it follows ``-o`` or ``-O``
    valued = False  # the word is the value of the option before it
    for word in words:
        text = word.text
        if valued:
            valued = False
        elif text.lstrip("-") in ("rcfile", "init-file") and text != text.lstrip("-"):
            valued = True
        elif named or (names_only and text[:1] not in ("-", "+")):
            if not (word.literal and text in _OPTIONS):
                return False
            named = False
        elif text[:1] not in ("-", "+") or text in ("-", "--"):
            return True
        elif text.startswith("--"):
            if text not in ("--norc", "--noprofile", "--login"):
                return False
        elif not (word.literal and set(text[1:]) <= letters):
            return False
        else:
            named = not names_only and bool(set(text[1:]) & {"o", "O"})
    return True


def _may_fail(redirection: tuple[str, str, Word]) -> bool:
    """Return whether opening *redirection* may fail: bash does not run its command then."""
    _, operator, target = redirection
    return not (operator.startswith("<<") or (
        target.literal and target.text in ("/dev/null", "-", "0", "1", "2")))


class _Functions(dict[str, tuple[frozenset[str], tuple[Word, ...], str | None, "Group | None"]]):
    """The functions a command defined (:attr:`_Scanner.functions`).

    For each the variables a call may set, the files it may make the
    shell's output, the text of its body (``None`` once two definitions
    differ) and the body as scanned where it was defined.

    Attributes:
        followed: How many calls the scan followed into a body
            (:meth:`_Scanner.call`), of :data:`_MAX_FOLLOWED`.
        read: How many characters of the values of variables the scan read
            (:meth:`_Scanner.read`), of :data:`_MAX_READ` at most.
        trapped: Whether a ``trap`` set an action, which may call any of
            them later (:meth:`_Scanner.trap`).
        heredocs: Whether a body defined so far holds a here-document
            (``<<``) or is given one where it is defined (``f() { cat; } <<EOF``),
            whatever defines the function now (:meth:`_Scanner.unseen`).
        bodies: Every body defined so far, as scanned: a redirection after
            one is the definition's.
        scripts: The shells whose script is a here-document, each with the
            scanner of that script (:meth:`_Scanner.scripts`).
        shells: Those shells, by ``id``: a here-document is looked up once,
            whatever their number (:meth:`_Scanner.heredoc_bodies`).
        cdpath: The ``CDPATH`` the command's shell inherits, or what tells
            it (:func:`simple_commands`) until :meth:`inherited_cdpath` asked.
        kept: For each text read as written, where the backslashes of the
            backslash-newline pairs bash keeps in it are (:class:`_Pairs`).
        stored: Whether the shell keeps text that can spell a ``<``
            (:data:`_LESS`): a variable's value, an array's, a loop's
            words, what no simple command reads (:meth:`_Scanner.hands_on`).
        holders: The simple commands whose words or here-document can spell
            one, by ``id``: data, unless it reaches a shell (:func:`_carries`).
        scripted: Those of them whose only such words are a script the scan
            read (:meth:`_Scanner.read_script`).
    """

    def __init__(self, cdpath: Callable[[], str] | str | None = None,
                 kept: dict[str, set[int]] | None = None) -> None:
        """Start with no function defined, in a shell that inherits what *cdpath* tells.

        *kept*: the pairs the scan found bash to keep so far (:attr:`kept`).
        """
        super().__init__()
        self.kept: dict[str, set[int]] = {} if kept is None else kept
        self.stored = False
        self.holders: dict[int, SimpleCommand] = {}
        self.scripted: dict[int, SimpleCommand] = {}
        self.followed = 0
        self.read = 0
        self.trapped = False
        self.heredocs = False
        self.bodies: list[Group] = []
        self.scripts: list[tuple[SimpleCommand, _Scanner]] = []
        self.shells: set[int] = set()
        self.cdpath: Callable[[], str] | str = "" if cdpath is None else cdpath

    def inherited_cdpath(self) -> str:
        """Return the ``CDPATH`` the command's shell inherits, asking for it once."""
        if not isinstance(self.cdpath, str):
            self.cdpath = self.cdpath()
        return self.cdpath

    def texts(self) -> dict[str, str | None]:
        """Return the text of the body of each function: what :meth:`demote` compares."""
        return {name: entry[2] for name, entry in self.items()}

    def demote(self, before: dict[str, str | None]) -> None:
        """Make the functions defined since *before* (:meth:`texts`) ones only the shell knows.

        Their definitions may not have run (a branch's), or ended with
        the subshell that ran them: a call runs the body, or the program
        of that name.
        """
        for name, (sets, sinks, text, body) in list(self.items()):
            if text is not None and before.get(name) != text:
                self[name] = (sets, sinks, None, body)


def _numeric(value: str | None) -> bool:
    """Return whether the variable value *value* is an integer for sure."""
    return value is not None and (value == NUMBER or _INTEGER.match(value) is not None)


def _unknown(value: str | None) -> bool:
    """Return whether *value* is no text the scan knows: none, or a mark (:data:`NUMBER`)."""
    return value is None or value.startswith("\0")


class _Scanner:
    """The position and state of one :func:`simple_commands` scan."""

    def __init__(self, text: str, cwd: str | None, group: Group | None,
                 depth: int = 0, optimistic: bool = False,
                 isdir: Callable[[str], bool] | None = None,
                 functions: _Functions | None = None, pairs: _Pairs | None = None) -> None:
        """Start at the beginning of *text*, in the directory *cwd*, inside *group*.

        *depth* is how many groups *group* is nested in (:data:`MAX_NESTING`).
        *pairs*: the backslash-newline pairs removed from *text*, a text
        already joined (:class:`_Pairs`); without them *text* is as
        written, and is joined here.
        """
        self.pos = 0
        self.cwd = cwd
        self.group = group
        self.depth = depth
        self.optimistic = optimistic  # see :func:`simple_commands`
        self.isdir = isdir  # see :func:`simple_commands`
        # Bash 3 expands every assignment prefix of a command from the
        # environment before that prefix, unlike Bash 4+; resolving a later
        # one through an earlier one could approve a write to another file.
        self.sequential_prefix_assignments = bash_version() >= (4, 0)
        # The functions the command defined so far, shared by the scanners of one
        # scan, with the variables the body of each may set and the files it may
        # make the shell's output (:meth:`function_body`): a call runs a body,
        # which may call any of them (:meth:`end_command`).
        self.functions = _Functions() if functions is None else functions
        if pairs is None:
            pairs = _Pairs(text, self.functions.kept.get(text, ()))
            text = pairs.text
        self.text, self.pairs = text, pairs
        # Where the scan read an operator in the text: a ``<`` there is no text handed on.
        self.operators: set[int] = set()
        # The functions whose bodies are being scanned as called, the innermost last, and
        # how many definitions the position is in: a body runs when it is called.
        self.calling: tuple[str, ...] = ()
        self.defining = 0
        # The variables the command assigned so far and their literal values:
        # ``None`` for one whose value only the shell knows (an expansion in
        # it, or an assignment that may not run).  Read back where a later
        # word expands ``$NAME`` / ``${NAME}``, so ``T=tmp/x && cat > $T/f``
        # names a literal path.  A ``-NAME`` key: NAME has an attribute, and stays unknown.
        self.variables: dict[str, str | None] = {}
        # The names ``export`` exported, for sure: a child shell's environment
        # holds those (:meth:`environment`).  An ``export`` that may not run
        # (after ``||``, in an ``if``) exports nothing here, and ``unset``
        # drops the attribute with the value: the child's value is unknown
        # either way.
        self.exported: set[str] = set()
        self.commands: list[SimpleCommand] = []
        # The here-documents opened on the current line: (delimiter, strip
        # leading tabs, whether the substitutions of the body run, the
        # node reading it as its standard input or None).
        self.heredocs: list[tuple[str, bool, bool, SimpleCommand | Group | None]] = []
        # What a ``|`` just piped, until the next command starts and reads it.
        self.pipe_from: SimpleCommand | Group | None = None
        # Whether the command or group being scanned follows ``||``, and
        # whether it follows ``&&``: an assignment there runs only when the
        # commands before it succeeded, as do the commands after it in the
        # same and-list (``cd d && T=tmp/x && cat > $T/f``), so its variable
        # is known to them and forgotten when the list ends any other way.
        self.after_or = False
        self.after_and = False
        self.and_assigned: set[str] = set()
        self.and_exported: set[str] = set()
        # Whether a ``cd`` after ``&&`` set the directory: known to the rest
        # of the and-list, unknown once the list ends any other way.
        self.and_cd = False
        # Whether a ``!`` was read and no command has taken it yet.
        self.negate = False
        # Whether a name reference exists (``declare -n R=T``): assigning
        # either name may assign the other, so no variable is known then.  Nor
        # is one once an attribute went to a variable only bash can name
        # (``readonly "$N"``, ``declare -u T{1,2}``), or an expansion
        # assigned a variable while its command's assignments were read
        # (:meth:`assigning`).
        self.nameref = False
        # Whether ``cd`` follows the links (``set -P``), to a directory only
        # the shell knows.
        self.physical = False
        # Whether bash no longer runs the text as written (:meth:`unfollowed`): every
        # later command of this shell is one only the shell knows.
        self.lost = False
        # Whether the text may have exported a variable of :data:`_SHELL_STARTUP`
        # with a value: a child shell then starts otherwise (:meth:`starts_otherwise`).
        self.startup = False
        # Whether a command from here to the end of the line may not run: bash skips the
        # rest of a line when an expansion or an assignment fails (``$((1/0))``, a
        # read-only variable), and ``return``, ``break`` and ``continue`` skip a body's.
        self.partial = False
        # The simple command whose words are being scanned (its substitutions read the
        # pipe it reads), and the one :meth:`end_command` follows: arithmetic it
        # evaluates runs with it (:meth:`blind`).
        self.current: SimpleCommand | None = None
        self.evaluating: SimpleCommand | None = None
        # Whether a loop is being scanned on trial (:meth:`rehearse`), and the state
        # the innermost compound command started in (:meth:`join`).
        self.trial = False
        self.entered: _Scope | None = None
        # Of the double-quoted string read last: where in the parts of its word its last
        # expansion of a value only the shell knows ends (-1 for none), and whether one
        # of those may be several words (``"$@"``).
        self.opaque: tuple[int, bool] = (-1, False)
        # Where a bare ``exec > file`` sent the shell's standard output: the
        # output of every later command, unless it redirects its own.  The
        # file as the ``exec`` opened it (an absolute path, or a word only the
        # shell resolves), so a later ``cd`` does not move it; several when an
        # ``exec`` that may not have run left the earlier one possible.
        self.stdout: tuple[Word, ...] = ()
        # Where in the text a ``((`` was found not to close as ``))``: shared
        # with the scanners of its arithmetic expressions, so no expression is
        # scanned twice (nested ones would double the work at each level).
        self.not_arithmetic: set[int] = set()

    def command_list(self) -> None:
        """Scan commands to the end of the text, or to the end of the group being scanned."""
        text = self.text
        current: SimpleCommand | Group | None = None
        # The variables as the next assignment *current* starts with sees them
        # (:meth:`assign`); ``None`` once one of its words is no assignment.
        assigned: _Assigned | None = None
        # The state where the current and-or list and pipeline started, and
        # whether the pipeline has several stages (each runs in a subshell of
        # its own, as does a backgrounded list, so its ``cd``, assignments and
        # ``exec`` end with it: the state is restored after it).
        list_scope = pipeline_scope = self.scope()
        piped = False
        # Whether the words being read are a ``case`` pattern: its ``|`` is no pipe.
        case = pattern = self.group is not None and self.group.kind == "case"
        # Whether they are the header of a loop (``x in a b``): no command either.
        header = self.group is not None and self.group.kind in ("for", "select")
        # The variables the ``coproc`` of the pipeline being read sets when it ends: the
        # array of its descriptors and ``NAME_PID``, its process id.
        coproc: tuple[str, ...] = ()
        # The commands and groups of the and-or list being read, and whether the one
        # being read is a coprocess: the shell does not wait for those of a list ``&``
        # ends, nor for a coprocess (:attr:`SimpleCommand.asynchronous`).
        listed: list[SimpleCommand | Group] = []
        coprocess = False
        while self.pos < len(text):
            char = text[self.pos]
            if char in " \t":
                self.pos += 1
            elif char == "#":
                self.comment()
            elif char == ")":
                self.pos += 1
                if self.group is not None and self.group.kind in ("(", "$("):
                    break
                self.end_command(current, pattern)  # a ``case`` pattern
                current, pattern = None, False
            elif char == "(":
                pattern = False  # ``(a|b)``: the pattern is read as a subshell
                if isinstance(current, SimpleCommand):
                    with self.assigning(assigned if len(current.words) > 1 else None):
                        if self.array_value(current):
                            continue
                if isinstance(current, SimpleCommand) and (name := self.function_name(current)):
                    current = self.function_body(name)
                else:
                    current = self.group_command("(")
            elif (not isinstance(current, SimpleCommand) and not pattern
                  and _CONDITIONAL.match(text, self.pos)):
                current, assigned = self.group_command("[["), None
            elif not isinstance(current, SimpleCommand) and (word := self.reserved_word()):
                if word == "!":
                    self.negate = True
                elif word == "function":
                    name = self.function_name()
                    current = self.function_body(name) if name else None
                elif word == "coproc":  # its command runs in a subshell, as a stage does
                    piped = coprocess = True
                    named, command = self.coproc()
                    coproc = (named, f"{named}_PID")
                    if command is not None:
                        current, assigned = command, None
                elif word in ("elif", "else"):  # the branch before it may not have run
                    if self.group is not None and self.group.kind == "if":
                        self.join(self.entered)
                elif word in _OPENERS:
                    current = self.group_command(word)
                elif word in _CLOSERS and self.group is not None and (
                        self.group.kind not in _SUBSHELLS):
                    break
            else:
                token = _TOKEN.match(text, self.pos)
                if token and token["operator"]:
                    operator = token["operator"]
                    self.pos = token.end()
                    self.end_command(current, pattern or header)
                    header = False
                    if operator == "|" and pattern:  # ``a|b)``: the next alternative
                        current = None
                        continue
                    if operator in (";;", ";&", ";;&"):
                        pattern = case
                        if case:  # the alternative before it may not have run
                            self.join(self.entered)
                    if current is not None:
                        listed.append(current)
                        current.asynchronous = coprocess
                    coprocess = False
                    if operator == "&":
                        if current is not None:
                            current.background = True
                        for node in listed:
                            node.asynchronous = True
                    if operator not in ("&&", "|", "|&"):
                        self.forget_and_assigned()
                    if operator in ("|", "|&"):
                        self.pipe_from, piped = current, True
                        self.restore(pipeline_scope)
                        self.line_breaks()
                    else:
                        if operator == "\n" and self.group is None:  # the next line runs
                            self.partial = False
                        if operator == "&":  # the list ran in the background
                            self.restore(list_scope)
                        elif piped:
                            self.restore(pipeline_scope)
                        for variable in coproc:
                            self.forget(variable)
                        coproc = ()
                        pipeline_scope, piped = self.scope(), False
                        if operator in ("&&", "||"):
                            self.line_breaks()
                        else:
                            list_scope, listed = self.scope(), []
                    self.after_or = operator == "||"
                    self.after_and = operator == "&&"
                    current = None
                    if operator == "\n":
                        self.heredoc_bodies()
                elif token and token["redirection"]:
                    self.pos = token.end()
                    if current is None:
                        current, assigned = self.start_command(), _Assigned()
                    self.operators.update(range(token.start(), token.end()))
                    self.redirection(current, token["fd"], token["redirection"])
                else:
                    if not isinstance(current, SimpleCommand):
                        current, assigned = self.start_command(), _Assigned()
                    if assigned is None or not _WRITTEN_ASSIGNMENT.match(text, self.pos):
                        assigned = None
                    elif current.words:
                        self.assign(assigned, current.words)
                    start = self.pos
                    with self.assigning(assigned if current.words else None):
                        scanned = self.substitution() if token else self.word()
                    if assigned is None:  # after a word that is none, one bash dropped too
                        scanned.assignment = False
                    current.words.extend(self.fields(scanned, start))
        self.end_command(current, pattern or header)
        if coprocess and current is not None:
            current.asynchronous = True
        # The list ended at no operator (the text's end, the closer of the group it is in):
        # its last stage's subshell ends with it, and a coprocess's variables are set.
        if piped:
            self.restore(pipeline_scope)
        for variable in coproc:
            self.forget(variable)

    def scope(self) -> _Scope:
        """Return a subshell's own state (:data:`_Scope`), the mutable parts copied."""
        return (self.cwd, dict(self.variables), set(self.exported), self.stdout, self.physical,
                self.nameref, self.lost, self.startup, self.functions.texts())

    def restore(self, scope: _Scope) -> None:
        """Restore the state :meth:`scope` returned: a pipeline stage or a ``&`` list ended."""
        (self.cwd, variables, exported, self.stdout, self.physical, self.nameref, self.lost,
         self.startup, functions) = scope
        self.variables, self.exported = dict(variables), set(exported)
        self.functions.demote(functions)

    def join(self, scope: _Scope | None) -> None:
        """Make unknown what differs from *scope*: the shell may be in either state.

        Where a branch that may not have run ends (``else``, ``elif``,
        the ``;;`` of a ``case``) and where a loop starts over.  A
        variable that is an integer in both is a number.
        """
        if scope is None or self.optimistic:
            return
        cwd, variables, exported, stdout, physical, nameref, lost, startup, functions = scope
        self.functions.demote(functions)
        if self.cwd != cwd:
            self.cwd = None
        for name in variables.keys() | self.variables.keys():
            ours, theirs = self.variables.get(name), variables.get(name)
            if ours != theirs or name not in self.variables:
                number = _numeric(self.read(ours)) and _numeric(self.read(theirs))
                self.variables[name] = NUMBER if number else None
        self.exported &= exported
        self.stdout = (*stdout, *(sink for sink in self.stdout if sink not in stdout))
        self.physical, self.nameref = self.physical or physical, self.nameref or nameref
        self.lost, self.startup = self.lost or lost, self.startup or startup

    def inherit(self, scanner: _Scanner) -> _Scanner:
        """Hand *scanner*, which scans an expansion of this shell's, what the scan is in."""
        scanner.lost, scanner.startup, scanner.current = self.lost, self.startup, self.current
        scanner.evaluating, scanner.calling = self.evaluating, self.calling
        scanner.defining, scanner.trial, scanner.partial = self.defining, self.trial, self.partial
        scanner.after_or, scanner.after_and = self.after_or, self.after_and
        return scanner

    def rehearse(self, word: str) -> None:
        """Join the state with what the loop *word* opens at the position leaves.

        An iteration starts in the state the one before it left, so the
        loop is scanned on trial, from the state so far joined with what
        the trial left (:meth:`join`), until a trial changes nothing:
        the scan that counts then holds for every iteration (``T=new;
        for i in 1 2; do cat > "$T"; T=old; done`` writes ``old`` too).
        After :data:`_TRIALS` of them nothing is known.  The functions a
        trial defined stay defined: a later iteration calls them.
        """
        start, commands, heredocs = self.pos, len(self.commands), list(self.heredocs)
        saved = (self.group, self.after_or, self.after_and, set(self.and_assigned),
                 set(self.and_exported), self.and_cd, self.current, self.entered)
        partial = self.partial
        for _ in range(_TRIALS):
            entry = self.scope()
            self.trial = True
            self.compound(word)
            self.trial, self.partial = False, partial
            left = self.scope()
            del self.commands[commands:]
            self.pos, self.heredocs = start, list(heredocs)
            (self.group, self.after_or, self.after_and, and_assigned, and_exported, self.and_cd,
             self.current, self.entered) = saved
            self.and_assigned, self.and_exported = set(and_assigned), set(and_exported)
            self.restore(entry)
            self.join(left)
            if self.scope() == entry:
                return
        self.unfollowed()

    def assign(self, assigned: _Assigned, words: list[Word]) -> None:
        """Record in *assigned* what the last of *words*, the assignments of a command, set.

        With the first of them *assigned* starts as the shell's variables.
        A value is ``None`` when only the shell knows it: an expansion in
        it, an appended one, an array element's.  Only a value that
        replaces a known one is followed: bash may hold another than the
        one written (an attribute set where this scan did not read, a
        variable bash sets itself: ``RANDOM``, ``_``), so a variable this
        scan does not know stays unknown (``PWD`` and ``IFS``, which it
        takes from the directory and the default, become so).
        """
        if len(words) == 1:
            assigned.update(self.variables)
        match = _ASSIGNMENT.match(words[-1].text)
        if match is not None and (
                match["name"] in self.variables or match["name"] in ("PWD", "IFS")):
            known = (not match["append"] and not match["subscript"]
                     and self.variables.get(match["name"]) is not None)
            assigned[match["name"]] = words[-1].value(match) if known else None

    @contextmanager
    def assigning(self, assigned: _Assigned | None) -> Iterator[None]:
        """Scan an assignment with the variables as the earlier ones of its command set them.

        Bash 4+ assigns the assignments a command starts with in order,
        so the expansions of a later one, and the commands in them, see
        what the earlier ones *assigned* (``T=a V=$T`` sets ``V`` to
        ``a``).  Bash 3 expands a command's whole assignment prefix from
        its earlier environment, so those expansions stay unresolved here;
        the command's other words and its redirections do not
        (``None``), nor does an assignment after a word that is none
        (``export T=a V=$T``).  A variable the scan makes unknown
        meanwhile (``${T:=x}``) may be one of them, which the end of the
        command takes from its word: no variable is known after that.
        """
        if assigned is None:
            yield
            return
        shell = self.variables
        if self.sequential_prefix_assignments:
            self.variables, assigned.changed = assigned, False
        else:
            # Until the parser reaches a command word it cannot know whether
            # Bash 3 will keep these as bare, sequential assignments or expand
            # all of them as one command prefix.  Unknown values are safe for
            # both readings; in particular, substitutions cannot be approved
            # against the wrong target.  Track an expansion that assigns one
            # (``${T:=x}``) so the command's later words stay unknown too.
            # One mapping serves the command's whole prefix: the scan of a word
            # only makes variables unknown in it, which they all are, and a copy
            # for each assignment took time quadratic in their number.
            unknown = assigned.unknown
            if unknown is None:
                unknown = _Assigned(dict.fromkeys([*shell, "PWD", "IFS", "CDPATH"]))
                assigned.unknown = unknown
            unknown.changed = False
            self.variables = unknown
        yield
        changed = self.variables.changed if isinstance(self.variables, _Assigned) else False
        self.variables = shell
        if changed:
            self.nameref = True

    def array_value(self, command: SimpleCommand) -> bool:
        """Skip the ``(...)`` at the position when it is the value of the assignment before it.

        ``T=(a b)`` and ``T+=(c)`` make ``T`` an array, whose words this
        scan does not follow, but for their substitutions, which run:
        the assignment's variable becomes unknown.  An operator in the
        value is a syntax error at which bash drops the line, with what
        it read of it, and goes on with the next one: nothing the line
        did is known then.  (The line as written: a backslash at its
        end continues nothing, :meth:`_Pairs.keep`.)  Returns whether
        the parenthesis was such a value.
        """
        if not command.words or not command.words[-1].text.endswith("="):
            return False
        if not _ASSIGNMENT.fullmatch(command.words[-1].text):
            return False
        text, depth, start, opened = self.text, 0, False, self.pos
        while self.pos < len(text):
            char = text[self.pos]
            if char == "#" and start:  # a comment, where a word of the value would start
                self.comment()
            elif char == "'":
                self.pos = self.closing(self.pos + 1, char)
            elif char == '"':
                self.double_quoted([])
            elif char in "$`":
                self.expansion()
            elif text.startswith(("<(", ">("), self.pos):
                self.substitution()
            elif char in ";&|<>" or (char == "(" and depth):
                newline = text.find("\n", self.pos)
                end = len(text) if newline < 0 else newline
                self.pairs.keep(self.pos + 1, end, first=True)
                self.pos = end
                self.heredocs.clear()
                self.cwd = None
                self.forget("$")
                break
            else:
                depth += {"(": 1, ")": -1}.get(char, 0)
                self.pos += 2 if char == "\\" else 1
                if not depth:
                    break
            start = char in " \t\n("
        if not any(word.text.startswith("-") and "A" in word.text for word in command.words):
            # an indexed array's subscripts are arithmetic (``a=([i++]=x)``)
            for subscript in _VALUE_SUBSCRIPT.finditer(text, opened, self.pos):
                self.arithmetic_assignments(subscript[1])
        # Keep the unread value: it may contain shell text later handed to eval or a shell.
        command.words[-1] = Word(command.words[-1].text + text[opened:self.pos], False,
                                 command.words[-1].assignment)
        self.hands_on(opened, self.pos, None, stored=True)
        return True

    def start_command(self) -> SimpleCommand:
        """Start a simple command at the position; it reads the pipe a ``|`` left open."""
        command = SimpleCommand(self.cwd, self.group, conditional=self.after_or,
                                after_and=self.after_and, negated=self.negate)
        command.outputs.extend(self.stdout)  # the shell's ``exec > file``
        self.negate = False
        self.commands.append(command)
        self.current = command
        if self.pipe_from is not None:
            command.after_and = self.pipe_from.after_and
            _pipe(self.pipe_from, command)
            self.pipe_from = None
            command.piped = True
        return command

    def group_command(self, opener: str) -> Group:
        """Scan the group *opener* starts (``(``, ``((`` or a reserved word); return it.

        The group reads the pipe a ``|`` left open: its commands read
        their standard input from it, not only the first of them.  For
        ``(`` the position is still at the parenthesis; for a reserved
        word it is past the word.
        """
        piped_from, self.pipe_from = self.pipe_from, None
        after_and = self.after_and if piped_from is None else piped_from.after_and
        negated, self.negate = self.negate, False
        group = Group("((" if opener == "(" else opener, self.cwd, self.group)
        end = 0
        if opener == "(" and self.text.startswith("((", self.pos):
            # the substitutions in the expression, and what it evaluates, read the
            # redirections after its ``))``
            self.group = group
            end = self.arithmetic(self.pos + 2)
            self.group = group.group
        if end:
            self.pos = end
        elif opener == "[[":  # its redirections are those of its substitutions too
            self.group = group
            self.conditional(self.start_command())
            self.group, self.current = group.group, None
        elif opener == "(":
            self.pos += 1
            group = self.subshell("(")
        else:
            group = self.compound(opener)
        group.after_and, group.negated = after_and, negated
        if piped_from is not None:
            _pipe(piped_from, group)
            group.piped = True
        return group

    def reserved_word(self) -> str:
        """Consume the reserved word at the position and return it; ``""`` when there is none."""
        match = _BARE_WORD.match(self.text, self.pos)
        if match is None or match[0] not in _RESERVED:
            return ""
        self.pos = match.end()
        return match[0]

    def coproc(self) -> tuple[str, SimpleCommand | None]:
        """Read the word after ``coproc``; return the variable set and the command the word starts.

        ``coproc NAME command`` runs a compound command (``{``, ``(``,
        ``if``, a loop, ``case``, ``[[``) and makes NAME, which bash
        expands, the array of its descriptors; ``COPROC`` without a NAME.
        Before anything else the word is the first of a simple command,
        which is returned; an assignment, a redirection, an operator or
        a reserved word is left at the position for the caller.
        """
        text = self.text
        blanks = _BLANKS.match(text, self.pos)
        self.pos = blanks.end() if blanks else self.pos
        bare = _BARE_WORD.match(text, self.pos)
        if (self.pos == len(text) or text[self.pos] in "()#" or _TOKEN.match(text, self.pos)
                or (bare and bare[0] in _RESERVED)
                or _CONDITIONAL.match(text, self.pos)
                or _WRITTEN_ASSIGNMENT.match(text, self.pos)):
            return "COPROC", None
        start = self.pos
        word = self.word()
        if _COPROC_COMMAND.match(text, self.pos):
            return word.text, None
        command = self.start_command()
        command.words.extend(self.fields(word, start))
        return "COPROC", command

    def read(self, value: str | None) -> str | None:
        """Return *value*, what a variable holds, counted toward what the scan reads.

        Raises:
            TooDeepError: The scan has read more than :data:`_MAX_READ`
                characters of values (:data:`UNREAD_VALUES_VERDICT`).  A
                value read as one only the shell knows would do instead for
                the scan, but not for a guard that looks for a text in the
                words (a ``sed`` script's newline): it would let through
                what it refuses with the value known.
        """
        if value:
            self.functions.read += len(value)
            if self.functions.read > _MAX_READ:
                raise TooDeepError(UNREAD_VALUES_VERDICT)
        return value

    def fields(self, word: Word, start: int) -> list[Word]:
        """Return the words bash makes of *word*, read from *start* on.

        None of an expansion of nothing: ``E=; $E tee f`` runs ``tee``,
        and the word after such a one is the command word.  A quote keeps
        the empty word (``"$E"``, ``$E''``): a command bash cannot find,
        an empty argument.  One for each part of a value the scan knows,
        when the word is that variable alone, unquoted (``G="python3
        x.py"; $G a``): bash splits it at its blanks and expands nothing
        in the parts, but a pattern, which is left to the shell.
        """
        if word.literal:
            dropped = not word.text and not _QUOTE.search(self.text, start, self.pos)
            return [] if dropped else [word]
        variable = _VARIABLE.match(self.text, start, self.pos)
        name = (variable[1] or variable[2]) if variable else ""
        value = (None if self.nameref or name in _SHELL_SET
                 else self.read(self.variables.get(name)))
        if (not value or _unknown(value) or _GLOB.search(value)
                or self.variables.get("IFS", " \t\n") != " \t\n"):
            return [word]
        return [Word(part, True) for part in re.split(r"[ \t\n]+", value) if part]

    def function_name(self, command: SimpleCommand | None = None) -> str:
        """Consume the name and ``()`` of a function definition; return the name, ``""`` for none.

        With *command* (``f() ...``: its one word is the name, read
        already, and the position is at the ``(``) the command is
        dropped, since a definition runs nothing.  Without it the
        position is after ``function``: the name and its optional ``()``
        are read.
        """
        text = self.text
        if command is not None:
            parens = _FUNCTION_PARENS.match(text, self.pos)
            if (parens is None or len(command.words) != 1 or not command.words[0].literal
                    or command.redirections):
                return ""
            if self.commands and self.commands[-1] is command:
                self.commands.pop()
            self.pos = parens.end()
            return command.words[0].text
        name = _FUNCTION_NAME.match(text, self.pos)
        if name is None:
            return ""
        blanks = _BLANKS.match(text, name.end())
        parens = _FUNCTION_PARENS.match(text, blanks.end() if blanks else name.end())
        self.pos = name.end() if parens is None else parens.end()
        return name[1]

    def function_body(self, name: str) -> Group | None:
        """Scan the body of the definition of the function *name* at the position; return it.

        The body (a brace group, a subshell, any compound command; blank
        lines and comments may precede it) runs only when the function is
        called: it is scanned as a group that may not run, so its
        commands are read, and a ``cd`` or an assignment in it leaves
        the directory or the variable unknown.  ``None`` without a body.
        What a call may set is kept (:attr:`functions`): the variables
        unknown after the body (those unknown before it are left out
        meanwhile, so that an assignment to one shows; ``$`` stands for
        every variable), ``PWD`` when the directory is unknown, both when
        the body runs a word only the shell knows (``"$@"`` may be any
        builtin), and the files a bare ``exec`` in it sends the output to.
        """
        self.line_breaks()
        after_or, self.after_or = self.after_or, True
        cwd, before, stdout, first = self.cwd, self.variables, self.stdout, len(self.commands)
        start, lost, partial = self.pos, self.lost, self.partial
        self.defining += 1
        self.variables = {key: value for key, value in before.items() if value is not None
                          or key.startswith("-") or key in ("PWD", "IFS", "CDPATH")}
        self.variables["$"] = ""
        if self.text.startswith("(", self.pos):
            group = self.group_command("(")
        elif (word := self.reserved_word()) in _OPENERS:
            group = self.group_command(word)
        else:
            group = None
        self.defining -= 1
        self.after_or, self.partial = after_or, partial
        sets = {key for key, value in self.variables.items()
                if value is None and not key.startswith("-")}
        if self.lost and not lost:  # what a call does (:meth:`call`), not the definition
            self.lost = False
            sets |= {"$", "PWD"}
        if self.variables.pop("$", None) is None:  # every variable, or a body defined another
            sets.add("$")
        if cwd is None or self.cwd is None:
            sets.add("PWD")
        if any(command.unknown and command.words for command in self.commands[first:]):
            sets |= {"$", "PWD"}
        self.variables = {**dict.fromkeys(before), **self.variables}
        body: str | None = self.text[start:self.pos]
        if "<<" in self.text[start:self.pos]:
            self.functions.heredocs = True
        if group is not None:
            self.functions.bodies.append(group)
        first_definition = name not in self.functions
        earlier, sinks, defined, _ = self.functions.get(name, (frozenset(), (), body, None))
        # two definitions that differ, or a first one that may not run: only the shell knows
        # what a call runs (one that repeats the text changes nothing, run or not)
        if defined != body or (first_definition and (
                after_or or self.after_and or self.partial or (
                    self.group is not None and self.group.conditional))):
            body = None
        self.functions[name] = (
            earlier | sets, (*sinks, *self.stdout[len(stdout):]), body, group)
        if name == "command_not_found_handle":  # bash calls it for a program it does not find
            self.unfollowed()
        if self.functions.trapped and "<<" in self.text[start:self.pos] and not self.optimistic:
            self.hidden()  # a trap may call it, in a state only the shell knows
        return group

    def call(self, command: SimpleCommand, name: str) -> None:
        """Scan the body of the function *name* as the call *command* runs it.

        With the variables, the directory and the standard input of the
        call: the body's commands are those of a group the call's
        redirections and pipes apply to, before the ones written after
        the definition (``f() { cat; } > file``), and are derived
        (:attr:`SimpleCommand.derived`).  A function that calls itself, a
        call nested past :data:`_MAX_CALLS`, one past
        :data:`_MAX_FOLLOWED` in a scan and one of a function defined
        twice run what only the shell knows.
        """
        _, _, text, body = self.functions[name]
        self.functions.followed += 1
        if (text is None or name in self.calling or len(self.calling) >= _MAX_CALLS
                or self.functions.followed > _MAX_FOLLOWED):
            command.unknown = True
            self.unfollowed()
            return
        redirections = [] if body is None else body.redirections
        outputs = (body.outputs if body is not None and body.outputs else command.outputs)
        group = command.script = Group(
            "(", self.cwd, self.group, outputs=outputs,
            redirections=[*command.redirections, *redirections],
            heredoc=command.heredoc or (body is not None and body.heredoc),
            piped=command.piped, after_and=command.after_and)
        scanner = _Scanner(text, self.cwd, group, self.depth, self.optimistic, self.isdir,
                           self.functions, _NO_PAIRS)
        # the assignments before the call are the body's variables (``T=x f``); bash takes
        # them back after it, or keeps them (``posix``): unknown then
        temporary: dict[str, str | None] = {}
        for word in command.words:
            match = _ASSIGNMENT.match(word.text) if word.assignment else None
            if match is None:
                break
            plain = not match["append"] and not match["subscript"]
            temporary[match["name"]] = word.value(match) if plain else None
        scanner.variables, scanner.exported = {**self.variables, **temporary}, set(self.exported)
        scanner.nameref, scanner.physical, scanner.stdout = self.nameref, self.physical, self.stdout
        scanner.lost, scanner.startup, scanner.trial = self.lost, self.startup, self.trial
        scanner.calling, scanner.partial = (*self.calling, name), self.partial
        scanner.nested_command_list()
        self.variables = {**scanner.variables, **dict.fromkeys(temporary)}
        self.exported = scanner.exported
        self.nameref, self.physical, self.stdout = scanner.nameref, scanner.physical, scanner.stdout
        self.lost, self.startup, self.partial = scanner.lost, scanner.startup, scanner.partial
        if scanner.cwd != self.cwd:
            self.cwd = None
        for simple in scanner.commands:
            simple.derived = True
        self.commands.extend(scanner.commands)

    def compound(self, word: str) -> Group:
        """Scan the group the reserved *word* opens to its closing word; return the group.

        A ``cd`` in it holds after it, unless it may not have run.  An
        assignment in it holds for the commands after it in the group;
        after a group that may not have run (a compound command, or a
        brace group after ``||``) the variable is unknown, and after a
        brace group after ``&&`` it is known to the rest of the and-list.
        """
        if word in _LOOPS and not (self.trial or self.optimistic):
            self.rehearse(word)
        outer = self.group
        conditional = word != "{" or self.after_or or (outer is not None and outer.conditional)
        may_not_run = conditional or self.after_and
        before = dict(self.variables)
        if word in ("for", "select"):  # the loop variable takes the values it loops over
            blanks = _BLANKS.match(self.text, self.pos)
            loop_variable = _NAME.match(self.text, blanks.end() if blanks else self.pos)
            if loop_variable:
                numbers = word == "for" and _NUMBER_LIST.match(self.text, loop_variable.end())
                self.variables[loop_variable[0]] = NUMBER if numbers else None
                if not numbers:  # an integer evaluates the words it loops over
                    self.assigns(loop_variable[0], True)
            if word == "select":  # the line read at each prompt
                self.variables["REPLY"] = None
        group = self.group = Group(word, self.cwd, outer, conditional)
        self.after_or = self.after_and = False
        and_assigned, self.and_assigned = self.and_assigned, set()
        and_cd, self.and_cd = self.and_cd, False
        scope = self.scope()
        entered, self.entered = self.entered, scope
        self.nested_command_list()
        self.group, self.entered = outer, entered
        if word == "{" and _REDIRECTED.match(self.text, self.pos):
            self.join(scope)  # with a redirection that fails, the group does not run
        self.and_assigned = and_assigned
        # a brace group after ``&&`` that changed directory: known to the rest of the and-list
        self.and_cd = and_cd or (may_not_run and not conditional and self.cwd != group.cwd)
        if may_not_run:
            for name in before.keys() | self.variables.keys():
                value, earlier = self.variables.get(name), before.get(name)
                if value != earlier:
                    if conditional:  # an integer either way: a number
                        number = _numeric(self.read(value)) and _numeric(self.read(earlier))
                        self.variables[name] = NUMBER if number else None
                    else:  # a brace group after ``&&``: known to the rest of the and-list
                        self.and_assigned.add(name)
        return group

    def forget_and_assigned(self) -> None:
        """End the and-list: the variables assigned after its ``&&`` may not have been.

        Nor may a ``cd`` there have run: the directory is unknown then.
        An optimistic scan keeps them: the list is assumed to have run through.
        """
        if not self.optimistic:
            for name in self.and_assigned:
                self.variables[name] = None
            self.exported -= self.and_exported
            if self.and_cd:
                self.cwd = None
        self.and_assigned = set()
        self.and_exported = set()
        self.and_cd = False

    def subshell(self, kind: str) -> Group:
        """Scan the commands of a subshell or a substitution to its ``)``; return its group.

        A ``cd`` in it ends with it.
        """
        saved = (self.cwd, self.group, self.after_or, self.after_and, self.variables,
                 self.exported, self.and_assigned, self.and_exported, self.and_cd, self.nameref,
                 self.stdout, self.physical, self.lost, self.startup, self.current, self.entered,
                 self.partial)
        functions = self.functions.texts()  # those it defines end with it too
        group = self.group = Group(kind, self.cwd, self.group)
        if kind in _SUBSTITUTIONS and self.current is not None:
            self.current.substitutions.append(group)
        self.after_or = self.after_and = False
        self.variables = dict(self.variables)  # its assignments end with it too
        self.exported = set(self.exported)
        self.and_assigned = set()
        self.and_exported = set()
        self.and_cd = False
        if kind in _SUBSTITUTIONS:  # its output is a word, not the shell's output
            self.stdout = ()
        self.nested_command_list()
        (self.cwd, self.group, self.after_or, self.after_and, self.variables,
         self.exported, self.and_assigned, self.and_exported, self.and_cd, self.nameref,
         self.stdout, self.physical, self.lost, self.startup, self.current, self.entered,
         self.partial) = saved
        self.functions.demote(functions)
        return group

    def nested_command_list(self) -> None:
        """Scan the commands of the group just entered, one level deeper."""
        self.depth = self.deeper()
        self.command_list()
        self.depth -= 1

    def deeper(self) -> int:
        """Return the depth of the group or expansion just entered; past the limit end the scan.

        Raises:
            TooDeepError: It would be deeper than :data:`MAX_NESTING`.
        """
        if self.depth >= MAX_NESTING:
            raise TooDeepError(TOO_DEEP_VERDICT)
        return self.depth + 1

    def substitution(self, fed: Group | None = None) -> Word:
        """Scan the ``$(...)``, ``<(...)`` or ``>(...)`` at the position as one word.

        Args:
            fed: The group a ``<(...)`` is the standard input of; without
                one, the simple command being scanned holds it.
        """
        start, holder = self.pos, fed or self.current
        self.pos += 2
        group = self.subshell("$(")
        # what its commands print is the word's text: they hand their own words on
        self.operators.update(range(start, self.pos))
        group.after_and = self.after_and  # as the word's command
        if self.text[start] != "$":  # a process substitution: a stream, not a word
            group.asynchronous = True
            group.reader = holder if self.text[start] == "<" else None
        return Word(self.text[start:self.pos], literal=False)

    def end_command(self, command: SimpleCommand | Group | None, header: bool = False) -> None:
        """Follow a finished ``cd`` or assignment; one that may not run leaves its state unknown.

        The words of a *header* (a loop's ``x in a b``, the subject and a
        pattern of a ``case``) are scanned as a command's and run nothing.

        A bare assignment (``T=tmp/x``, several at once) or an ``export``
        one records the variable's literal value for the words after it
        (:attr:`variables`); a value the shell expands, or an assignment
        after ``||`` or ``&&`` or in a compound command that may not run,
        makes the variable unknown, as does one before a special builtin
        or a wrapper running no command (``T=x export Y``, ``T=x exec
        >f``: sh keeps it, bash does not), as do the builtins that assign the
        variables their operands name (``read T``, ``unset T``, ``declare
        T=x``, ``printf -v T``...) or, named or not, their default
        (``read`` fills REPLY, ``mapfile`` MAPFILE); ``unset T`` drops T's
        export attribute too, and an ``export T`` exports T for a child
        shell (:attr:`exported`) only when it runs for sure, or for the
        rest of its and-list after ``&&``; after ``eval`` or
        ``source`` no variable is known, nor the directory, and the
        literal text ``eval`` runs is scanned (:meth:`run_text`); after
        a name reference no variable is known, nor after a call of a
        function what its body, or another's, may set (:attr:`functions`).
        A bare ``exec`` with a redirection of standard output to a file
        makes that file, as it opens it here, the output of the commands
        after it; one that may not run (after ``&&`` or ``||``, in a
        compound command, in a function body) leaves the earlier output
        possible too.  Behind a wrapper that is a program (``sudo cd dir``,
        ``env export T=x``) the word names a program, which sets nothing
        here.  (An assignment or an ``exec`` in a pipeline
        segment or a backgrounded list is recorded too and undone by
        :meth:`command_list` when the segment or the list ends.)
        """
        self.current = None
        if header and self.functions.holders.pop(id(command), None) is not None:
            self.functions.stored = True  # a loop's variable holds its words
        if not isinstance(command, SimpleCommand) or header:
            for stream in command.substitutions if isinstance(command, SimpleCommand) else ():
                stream.reader = None  # the commands after a header read its ``<(...)``
            return
        if command.heredoc and not _reads_heredoc(command):
            command.heredoc, command.heredoc_body = False, None
        argv = command.argv()
        prefix = command.words[:len(command.words) - len(argv)]
        program = any(not word.assignment and posixpath.basename(word.text) in _PROGRAM_PREFIXES
                      for word in prefix)
        name = argv[0].text if argv and not program else ""
        # a function with a wrapper's name is what bash runs for it (``timeout() { ...; }``)
        first = next((word for word in prefix if not word.assignment), None)
        shadowed = first is not None and first.literal and first.text in self.functions
        if first is not None and shadowed:
            name, program = first.text, False
        elif _describes(prefix) and not self.lost:
            return
        group = command.group
        self.evaluating = command
        # A command word only the shell knows may be nothing (the next word then runs), a
        # wrapper, ``tee``, ``eval`` or any builtin: its input may go anywhere, and run in
        # the shell itself it leaves nothing known.  Nor is a command known once bash no
        # longer runs the text as written.  (A function's body runs when it is called.)
        # (nor is a path's program, once a function may have a path's name: ``/bin/x() {``)
        if argv and (self.lost or not (argv[0].literal or argv[0].path) or (
                argv[0].path and any("/" in defined for defined in self.functions))):
            command.unknown = True
            if not (program or self.optimistic or self.defining):
                self.unfollowed()
        elif argv and posixpath.basename(argv[0].text) in SHELLS and not self.optimistic:
            # ``-c`` text the shell expands first: the child runs any command
            script = shell_script_index([word.text for word in argv[1:]])
            command.unknown = script is not None and argv[script + 1].passed() is None
            if reads_script(argv):
                # its standard input is its script: a here-document's body, read once the
                # bodies are (:meth:`scripts`), else text only the shell knows
                command.unknown = not command.heredoc
                if command.heredoc and not self.trial:
                    self.functions.scripts.append((command, self.shell(command, "", False)))
                    self.functions.shells.add(id(command))
        if any(self.fallible(word) for word in prefix):
            self.partial = True
        # one after a failure that skips the line may not run, nor one whose redirection fails
        conditional = (not self.optimistic and (self.after_or or self.partial or (
            bool(argv) and any(map(_may_fail, command.redirections))))) or (
            group is not None and group.conditional)
        if name in ("cd", "pushd", "popd"):
            self.and_cd |= self.after_and
            if "PWD" in self.variables:  # bash sets it anew: what was assigned to it is gone
                self.variables["PWD"] = None
        # ``command`` and ``builtin`` run the builtin where a function has its name
        function = shadowed or (name in self.functions and not any(
            word.text in ("command", "builtin") for word in prefix))
        bare = not argv and all(word.assignment for word in prefix)
        if not bare and (not argv or name in _SPECIAL_BUILTINS):  # in ``eval``'s text too
            for word in prefix:
                if (match := _ASSIGNMENT.match(word.text)) is not None:
                    self.variables[match["name"]] = None
        text = command_text(argv)
        trapped = _trap_operands(argv) if name == "trap" else []
        # what a function or the text of ``eval`` that may not run leaves in doubt
        entry = self.scope() if (conditional or self.after_and) and (
            function or text is not None) else None
        if text is not None:
            # ``eval``'s literal text, a child shell's literal ``-c`` text:
            # scanned here (the child's ``cd`` and assignments stay its own).
            self.run_text(command, text, name == "eval")
            self.read_script(command, _script_words(argv))
        elif not self.optimistic and command_text(_nested_shell(argv)) is not None:
            # the text of a shell that a runner or an expansion names: read as that shell's
            line = _nested_shell(argv)
            scanner = self.shell(command, command_text(line) or "", False, line[1:])
            scanner.nested_command_list()
            command.script = None  # no shell the scan knows: the guards judge it as before
            # (a text without a here-document adds nothing to what the guards judge)
            if command.heredoc or any(simple.heredoc for simple in scanner.commands):
                for simple in scanner.commands:
                    simple.derived = True
                self.commands.extend(scanner.commands)
            self.read_script(command, _script_words(line))
        if not argv and not program and any(word.text == "exec" for word in prefix):
            if command.heredoc and not self.optimistic:
                self.hidden()  # the shell's own from now on: any later command may read it
            for fd, operator, operand in command.redirections:
                # a file is the shell's output from now on; a duplicate (``>&3``) is the
                # file opened on that descriptor, which its name stands for
                copied = duplicated(operator, operand)
                if operator in _BOTH or (
                        descriptor(fd, operator) == "1" and operator in _STDOUT_OPENERS
                        and copied not in ("1", "-")):
                    path = resolve(self.cwd, operand) if copied is None else f"/dev/fd/{copied}"
                    sink = Word(operand.text, False) if path is None else Word(path, True)
                    may_not_run = conditional or (self.after_and and not self.optimistic)
                    self.stdout = (*self.stdout, sink) if may_not_run else (sink,)
        if name in ("mapfile", "readarray") and any(
                word.text.startswith("-") and "C" in word.text for word in argv[1:]):
            name = "eval"  # its ``-C`` callback is evaluated, line by line
        if function:  # a call runs a body, which may call any defined by now
            if not (self.optimistic or self.defining):
                self.call(command, name)
            for sets, sinks, _, _ in list(self.functions.values()):
                for variable in sets:
                    self.forget(variable)
                if "PWD" in sets:
                    self.cwd = None
                self.stdout = (*self.stdout, *(sink for sink in sinks if sink not in self.stdout))
        elif name == "cd":
            self.cwd = None if conditional or self.physical else self.cd(prefix, argv[1:])
        elif name == "pushd":
            # ``pushd DIR`` is a cd; ``pushd -n DIR`` only pushes and stays;
            # ``pushd`` alone or ``pushd +N`` / ``-N`` rotates the stack to a
            # directory only the shell knows.
            operands = [word for word in argv[1:] if word.text != "-n"]
            if (conditional or self.physical or not operands
                    or operands[0].text[:1] in ("+", "-")):
                self.cwd = None
            elif len(operands) == len(argv) - 1:  # no -n
                self.cwd = self.cd(prefix, operands)
        elif name == "popd":
            self.cwd = None
        elif name in ("set", "shopt"):
            if not _followed(argv[1:], _OPTION_LETTERS, name == "shopt"):
                self.unfollowed()
            for word in argv[1:]:
                if word.text == "--":  # the positional parameters follow
                    break
                self.physical |= _PHYSICAL.match(word.text) is not None
                # ``set -a`` exports what is assigned from here on
                self.startup |= name == "set" and _ALLEXPORT.match(word.text) is not None
        elif name in ("alias", "enable") or len(trapped) == 1 or any(
                not word.literal or word.text.upper().removeprefix("SIG") not in _LATE_TRAPS
                for word in trapped[1:]) or (
                # ``hash -p file name`` makes the name another program
                name == "hash" and any(
                    not word.literal or (word.text.startswith("-") and "p" in word.text)
                    for word in argv[1:])):
            command.unknown |= name == "alias"  # its words are text bash runs later
            self.unfollowed()
        elif name in ("test", "["):  # ``-v 'a[i++]'`` evaluates the subscript
            for before, word in zip(argv, argv[1:], strict=False):
                if (element := _ELEMENT.match(word.text)) is not None:
                    self.arithmetic_assignments(element[1])
                elif not word.literal and (before.text == "-v" or not before.literal):
                    self.forget("$")  # an element only the shell knows
                    self.blind()
        elif name in _SOURCING_BUILTINS:  # the text it ran may have set any variable
            self.forget("$")
            self.cwd = None  # and may have changed directory
            if name == "eval":
                unread = text is None and len(argv) > 1
            else:  # a file's commands are not the text's, but it may export a startup variable
                self.startup = True
                unread = not argv[1:] or _may_be_input(argv[1])
            if unread and not self.optimistic:
                # text only the shell knows: any command, with the input of this one, and
                # nothing is known after it
                command.unknown = True
                if not self.defining:
                    self.unfollowed()
        elif name in _ASSIGNING_BUILTINS:
            found = [match.lastindex for word in argv[1:] if (match := _ATTRIBUTE.match(word.text))]
            attribute = name == "readonly" or (name in _DECLARING_BUILTINS and bool(found))
            self.nameref |= attribute and 1 in found  # a name reference, or a name only bash reads
            if name in _DECLARING_BUILTINS and any(
                    word.text.startswith("-") and set(word.text) & set("n$`")
                    for word in argv[1:]) and not all(
                    word.text.startswith("-") or (
                        word.literal and _PLAIN_REFERENCE.fullmatch(word.text))
                    for word in argv[1:]):
                self.unfollowed()  # a reference to an element: each use evaluates its subscript
            if name in _DECLARING_BUILTINS and any(  # ``declare -x`` exports
                    word.text.startswith("-") and set(word.text) & set("x$`") for word in argv[1:]):
                self.startup |= self.exports_startup(argv[1:])
            integer = name in _DECLARING_BUILTINS and any(
                word.text.startswith("-") and set(word.text) & set("i$`") for word in argv[1:])
            for word in argv[1:]:
                if not word.text.startswith("-"):
                    target = self.assigns(word.text, name in (
                        "read", "readarray", "mapfile", "getopts"))
                    # an integer's value is arithmetic: ``declare -i U=T=5``, or U made one before
                    if integer or self.integer(target):
                        self.arithmetic_assignments(word.text.partition("=")[2])
                    if name == "unset":
                        self.exported.discard(target)
                    elif attribute:  # what is assigned to it is not what it holds
                        self.variables[f"-{target}"] = None
                elif name == "read":  # ``-aT``, ``-raT``: the array read into
                    for at, letter in enumerate(word.text[1:], 2):
                        if letter in _READ_VALUED:
                            if letter == "a":
                                self.assigns(word.text[at:], True)
                            break
            if name in _DEFAULT_TARGETS:
                self.variables[_DEFAULT_TARGETS[name]] = None
        elif name == "let":  # its operands are arithmetic expressions
            for word in argv[1:]:
                if word.literal:
                    self.arithmetic_assignments(word.text)
                else:  # one the shell expands may assign any variable, and run a command
                    self.forget("$")
                    self.blind()
        elif name in _ASSIGNING_OPTION:  # ``printf -v T ...``, ``wait -np T`` assign T
            texts = [word.text for word in argv[1:]]
            for text, following in zip(texts, [*texts[1:], ""], strict=False):
                if text.startswith("-") and _ASSIGNING_OPTION[name] in text:
                    self.assigns(text.partition(_ASSIGNING_OPTION[name])[2] or following, True)
        elif name == "export":  # ``export T``, ``export T=x``; ``export -n T`` undoes it
            options = "".join(word.text[1:] for word in argv[1:] if word.text.startswith("-"))
            names = {named[0] for word in argv[1:] if (named := _NAME.match(word.text))}
            if "f" not in options:  # exported functions are no variables
                if "n" in options:  # ran or not, the child's value is unknown
                    self.exported -= names
                else:
                    self.startup |= self.exports_startup(argv[1:])
                    if not conditional:  # one that may not run changes nothing for sure
                        if self.after_and:
                            self.and_exported |= names - self.exported
                        self.exported |= names
        self.join(entry)
        assigned = prefix if bare else argv[1:] if name == "export" and not function else []
        for word in assigned:
            match = _ASSIGNMENT.match(word.text)
            if match is None:
                if _target(word.text) is None:  # ``export "$N=x"``
                    self.assigns(word.text)
                continue
            # ``T+=x`` appends to a value this scan may not know; ``T[0]=x``
            # makes an array, whose ``$T`` is its first element.
            plain = (not match["append"] and not match["subscript"]
                     and f"-{match['name']}" not in self.variables)
            given = word.value(match)
            known = given is not None and not conditional and plain
            # the value of ``i=$((i + 1))`` is an integer, whatever its operands are
            # (one that held none is one when the assignment runs and cannot fail; a name
            # in capitals may be bash's own, which it sets itself)
            value = _ARITHMETIC_VALUE.match(word.text, match.end())
            number = plain and value is not None and (self.numeric(match["name"]) or (
                not (conditional or self.after_and) and match["name"] != match["name"].upper()
                and self.unfailing(value[1])))
            # one field, whatever its expansions hold, when its end is a literal ``/name``
            ends = (plain and not conditional and not word.literal
                    and _LITERAL_END.search(word.text, match.end()) is not None)
            self.variables[match["name"]] = (
                given if known else NUMBER if number
                else PATH_VALUE + word.text[match.end():] if ends else None)
            if match["name"] == "PS4" and not (
                    known and _INERT.fullmatch(word.text, match.end())):
                self.unfollowed()  # bash runs what it holds before a traced command
            if (known or ends) and self.after_and:
                self.and_assigned.add(match["name"])
        for word in [*prefix, *([] if bare else assigned)]:
            match = _ASSIGNMENT.match(word.text)
            if match is not None:
                # an element's subscript and what an integer variable is assigned are
                # arithmetic: ``a[n++]=x`` assigns ``n`` too
                self.arithmetic_assignments((match["subscript"] or "") + (
                    word.text[match.end():] if self.integer(match["name"]) else ""))
        if trapped and not (self.optimistic or self.defining):
            self.trap(command, trapped)
        if name in ("return", "break", "continue"):  # the rest of the body may not run
            self.partial = True
        self.evaluating = None

    def numeric(self, name: str) -> bool:
        """Return whether the variable *name* holds an integer for sure.

        One the text assigned an integer, or one bash computes
        (:data:`_NUMERIC`) that the text left alone; no other of the
        variables bash sets itself (:data:`_SHELL_SET`).
        """
        if self.nameref:
            return False
        if name in self.variables:
            return _numeric(self.read(self.variables[name])) and (
                name in _NUMERIC or name not in _SHELL_SET)
        return name in _NUMERIC

    def integer(self, name: str) -> bool:
        """Return whether bash evaluates what the variable *name* is assigned as arithmetic.

        One the text gave an attribute (``declare -i``: a ``-NAME`` key)
        or one bash starts with as an integer (:data:`_INTEGERS`).
        """
        return f"-{name}" in self.variables or name in _INTEGERS

    def unfailing(self, expression: str) -> bool:
        """Return whether evaluating the arithmetic *expression* cannot fail.

        Its operators take any number (:data:`_UNFAILING`) and each of
        its names holds one.
        """
        return (_UNFAILING.match(expression) is not None and "++" not in expression
                and "--" not in expression
                and all(self.numeric(name) for name in _OPERAND.findall(expression)))

    def fallible(self, word: Word) -> bool:
        """Return whether the assignment *word* may fail: bash then skips the rest of the line.

        One to a variable with an attribute (a read-only one, an
        integer's arithmetic), to one bash keeps read-only, to one it
        starts with as an integer unless the value is a number as
        written (``SECONDS=0``), or to an element whose subscript is no
        plain index.
        """
        match = _ASSIGNMENT.match(word.text)
        return match is not None and (
            f"-{match['name']}" in self.variables or match["name"] in _READONLY
            or (match["name"] in _INTEGERS and _INTEGER.match(word.text, match.end()) is None)
            or _INDEX.fullmatch(match["subscript"] or "[0]") is None)

    def trap(self, command: SimpleCommand, operands: list[Word]) -> None:
        """Follow the action the ``trap`` *command* sets: the first of its *operands*.

        (:func:`_trap_operands`.)  Bash runs it later, in a state only
        it knows, with the standard input the shell has then: its
        commands are unknown ones in the group of the trap
        (:attr:`SimpleCommand.derived`), and a function may be called
        unseen (:meth:`unseen`).  An action the shell expands is the
        trap's own to run.
        """
        self.functions.trapped = True
        self.unseen()
        action = operands[0].passed()
        if action is None:
            command.unknown = True
            return
        scanner = _Scanner(action, None, Group("(", None, self.group), self.depth,
                           isdir=self.isdir, functions=self.functions)
        scanner.lost = scanner.nameref = scanner.physical = True
        scanner.nested_command_list()
        for simple in scanner.commands:
            simple.derived = True
        self.commands.extend(scanner.commands)

    def assigns(self, operand: str, data: bool = False) -> str:
        """Forget the variable the *operand* of an assigning builtin names; return its name.

        As :meth:`forget` does.  A name only the shell knows may be an
        element's, whose subscript bash evaluates: it may run a command
        (:meth:`blind`).  So may text only the shell knows (*data*: what
        ``read``, ``mapfile``, ``getopts``, ``printf -v`` assign) given to
        a variable that may be an integer: bash evaluates it.
        """
        if _target(operand) is None:
            self.blind()
        name = self.forget(operand)
        if data and (self.nameref or self.integer(name)):
            self.forget("$")
            self.blind()
        return name

    def hidden(self) -> None:
        """Record a here-document the scan does not read as one, fed to a command only bash knows.

        A command without words stands for the two (:attr:`SimpleCommand.derived`).
        """
        self.commands.append(SimpleCommand(None, self.group, heredoc=True, unknown=True,
                                           derived=True))

    def unseen(self) -> None:
        """Record that a function defined by now may be called where the scan does not see it.

        By a command only the shell knows or by a trap, in a state only
        the shell knows: a here-document in a body (``<<``) is hidden then
        (:meth:`hidden`).
        """
        if self.functions.heredocs:
            self.hidden()

    def exports_startup(self, operands: list[Word]) -> bool:
        """Return whether exporting *operands* may hand a child shell a startup variable.

        One of :data:`_SHELL_STARTUP` that is not assigned the empty value
        there, a name only bash reads (``export "$N"``), or any name once
        a name reference may exist (``declare -n R=BASH_ENV; export R``).
        """
        for word in operands:
            if not word.text.startswith("-"):
                name = _target(word.text)
                if not name or self.nameref or (name in _SHELL_STARTUP and not (
                        word.literal and word.text == f"{name}=")):
                    return True
        return False

    def blind(self) -> None:
        """Record that arithmetic the scan cannot follow is evaluated here.

        Bash evaluates the value of an operand as an expression and
        expands the subscripts in it, so an operand only the shell knows
        may run a command (``a[$(cmd)]``).  That command reads what the
        evaluation reads: the input of the command evaluating it as it
        runs (``let``, ``declare -i``, ``test -v``: that command is
        unknown then), else that of the group or the pipeline stage
        whose word is being expanded, where a command without words
        stands for it (:attr:`SimpleCommand.unknown`).  A gate reads the
        command as it was meant to run: with numbers for operands.
        """
        if self.optimistic:
            return
        if self.evaluating is not None:
            self.evaluating.unknown = True
            return
        group = Group("$(", self.cwd, self.group)
        if self.current is not None:
            self.current.substitutions.append(group)
        self.commands.append(SimpleCommand(None, group, unknown=True, derived=True))

    def forget(self, operand: str) -> str:
        """Make the variable the *operand* of an assigning builtin names unknown; return its name.

        A name the shell expands (:func:`_target`) may be any variable's:
        none is known then, nor ``PWD``, ``IFS`` and ``CDPATH``, which this
        scan takes from the directory and the defaults while nothing assigns
        them; ``""`` is returned for it, and for an operand that names none.
        An element's subscript is arithmetic that may assign others
        (``read 'a[n++]'``).
        """
        name = _target(operand)
        if name is None:
            self.variables = dict.fromkeys([*self.variables, "PWD", "IFS", "CDPATH"])
        elif name:
            self.variables[name] = None
            if name == "PS4":  # bash runs what it holds before a traced command
                self.unfollowed()
        element = _ELEMENT.match(operand)
        if element is not None:
            self.arithmetic_assignments(element[1])
        return name or ""

    def unfollowed(self) -> None:
        """Give up on the variables and the directory: bash no longer runs the text as written.

        After an ``alias``, an ``enable``, a ``DEBUG``, ``ERR`` or
        ``RETURN`` trap or an option such as ``lastpipe``
        (one not in :data:`_OPTIONS`), what a later command sets or enters is not
        what its words say.  No variable is known from here on, nor the
        directory, here or after a ``cd``; a subshell's ends with it.
        What bash runs instead may call a function defined by now
        (:meth:`unseen`).
        """
        if not (self.lost or self.optimistic or self.defining):
            self.unseen()
        self.nameref = self.physical = self.lost = True
        self.cwd = None

    def arithmetic_assignments(self, expression: str) -> None:
        """Make the variables the arithmetic *expression* assigns unknown (``T=1``, ``i++``).

        Every variable, when it assigns one only the shell can name (``$N=1``, ``$N++``).
        Bash evaluates the value of a variable the expression reads as an
        expression too (``X='T=1'; ((X))`` assigns ``T``, as ``(($X))``
        does), so a value this scan knows that is no number is read the
        same way, with the values it names in turn; one holding an
        expansion (bash expands a subscript's: ``X='a[$(...)]'``) may
        assign any variable.  A value only the shell knows (a command
        substitution's output too) is not followed.
        """
        expressions, followed = [expression], set[str]()
        unknown = False
        while expressions:
            expression = expressions.pop()
            unknown = unknown or _UNKNOWN_OPERAND.search(expression) is not None
            # what its names hold before it assigns them: ``X++`` evaluates the value of
            # ``X``, ``X = 5`` does not
            read = _PLAIN_TARGET.sub(r"\1=", _NO_OPERAND.sub(" ", expression))
            for name in set(_OPERAND.findall(read)) - followed:
                followed.add(name)
                if self.numeric(name):
                    continue
                value = (None if self.nameref or name in _SHELL_SET
                         else self.read(self.variables.get(name)))
                if _unknown(value):  # any text
                    unknown = True
                elif value:
                    expressions.append(value)
                    if "$" in value or "`" in value:
                        self.forget("$")
                        unknown = True
            # a target is a number once it held one: an evaluation that fails (``1/0``, a
            # read-only operand) assigns nothing
            for prefixed, assigned in _ARITHMETIC_ASSIGNMENT.findall(expression):
                name = prefixed or assigned
                number = f"-{name}" not in self.variables and self.numeric(name)
                self.variables[name] = NUMBER if number else None
            for name in _ARITHMETIC_ELEMENT.findall(expression):
                self.variables[name] = None
            if _ARITHMETIC_EXPANDED_TARGET.search(expression):
                self.forget("$")
        if unknown and not self.optimistic:
            self.forget("$")
            self.blind()

    def cd(self, prefix: list[Word], args: list[Word]) -> str | None:
        """Return the directory ``cd args`` after the words *prefix* leads to (:func:`_cd`).

        ``CDPATH`` may be set once the command assigned it another value than
        the empty one (``CDPATH=dir cd sub``: for this ``cd`` alone), or may
        have assigned it through a name only bash reads.  One the command
        never assigned is the one its shell inherited
        (:meth:`_Functions.inherited_cdpath`), which is asked for an
        operand bash would look for there.
        """
        cdpath: str | None = None
        if not self.nameref:
            if "CDPATH" in self.variables:
                cdpath = self.variables["CDPATH"]
            elif cd_path(self.cwd, args, True) is None and cd_path(self.cwd, args) is not None:
                cdpath = self.functions.inherited_cdpath()
            else:
                cdpath = ""
        if _unknown(cdpath):  # a list only the shell knows
            cdpath = None
        for word in prefix:
            if word.text.startswith(("CDPATH=", "CDPATH+=")):
                cdpath = word.text[7:] if word.literal and word.text[6] == "=" else None
        return _cd(self.cwd, args, cdpath != "", self.isdir)

    def conditional(self, command: SimpleCommand) -> None:
        """Scan the words of the ``[[ ... ]]`` *command* at the position, to its ``]]``.

        Between them ``<``, ``>``, ``&&``, ``||`` and parentheses are operators
        of the expression, not redirections or lists; its expansions run, a
        comment is skipped, and a word may be an arithmetic expression that
        assigns (``[[ T=2 -eq 2 ]]``).  The pattern after ``=~``, ``==``,
        ``=`` or ``!=`` is one word with its parentheses, whatever is in
        them (``^(a b|#)$``, ``@(a|b)``).
        """
        text, pattern, operand = self.text, False, ""
        while self.pos < len(text):
            start, char = self.pos, text[self.pos]
            if char == "#":
                self.comment()
            elif text.startswith(("<(", ">("), start):
                command.words.append(self.substitution())
            elif char in _METACHARACTERS and not (pattern and char == "("):
                self.pos += 1
                if char == "\n":
                    self.heredoc_bodies()
            else:
                command.words.append(self.pattern_word() if pattern else self.word())
                written = text[start:self.pos]
                if written == "]]":
                    break
                # the operands of ``-eq`` and its like are arithmetic, as is the
                # subscript of the element ``-v`` tests
                word = command.words[-1].text
                if operand == "-v":
                    element = _ELEMENT.match(word)
                    self.arithmetic_assignments(element[1] if element else "")
                    if element is None and not command.words[-1].literal:
                        self.forget("$")  # an element only the shell knows
                        self.blind()
                elif operand:
                    self.arithmetic_assignments(word)
                elif written in _ARITHMETIC_TESTS and len(command.words) > 2:
                    self.arithmetic_assignments(command.words[-2].text)
                operand = written if written in _ARITHMETIC_TESTS or written == "-v" else ""
                pattern = written in ("=~", "==", "=", "!=")

    def pattern_word(self) -> Word:
        """Scan the pattern of a ``[[`` operator at the position: one word, kept as written.

        A blank or an operator ends it outside its parentheses only, and
        ``|`` is a character of it; its quotes are skipped and its
        expansions run.
        """
        text, start, depth = self.text, self.pos, 0
        while self.pos < len(text):
            char = text[self.pos]
            if char == "'":
                self.pos = self.closing(self.pos + 1, char)
            elif char == '"':
                self.double_quoted([])
            elif char in "$`":
                self.expansion()
            elif char in _METACHARACTERS and not depth and char not in "(|":
                break
            else:
                depth += {"(": 1, ")": -1}.get(char, 0)
                self.pos += 2 if char == "\\" else 1
        self.hands_on(start, self.pos, self.current)
        return Word(text[start:self.pos], literal=False)

    def run_text(self, command: SimpleCommand, text: str, evaluated: bool) -> None:
        """Scan *text*, the literal text *command* runs (:func:`command_text`), for its commands.

        They run in the command's directory (one only the shell knows
        under a wrapper that moves the program, ``env -C dir``), with its
        redirections, its here-document and its pipe, as the commands of
        a group would.  The ``eval`` of this shell's (*evaluated*) know the
        variables it knows; a child shell's know its environment
        (:meth:`environment`).  A child shell does not run its text as
        written (:meth:`unfollowed`) once a startup variable may reach it
        (:meth:`starts_otherwise`).
        """
        scanner = self.shell(command, text, evaluated)
        scanner.nested_command_list()
        if evaluated:  # what the text declared and set holds after it
            self.variables = scanner.variables  # its ``-NAME`` keys: the values end with ``eval``
            self.nameref, self.physical = scanner.nameref, scanner.physical
            self.lost, self.startup, self.partial = scanner.lost, scanner.startup, scanner.partial
        self.commands.extend(scanner.commands)

    def shell(self, command: SimpleCommand, text: str, evaluated: bool,
              arguments: list[Word] | None = None) -> _Scanner:
        """Return the scanner of *text*, which *command* runs, in the state it starts in.

        *arguments*: those of the shell that runs it, when the command
        is not that shell itself (:func:`_nested_shell`).

        As :meth:`run_text` describes; the group of the text's commands
        is the command's :attr:`SimpleCommand.script`.
        """
        group = command.script = Group(
            "(", self.cwd, self.group, outputs=command.outputs,
            redirections=command.redirections, heredoc=command.heredoc,
            piped=command.piped, after_and=command.after_and)
        cwd = None if _command_word(command.words)[1] else self.cwd
        scanner = _Scanner(text, cwd, group, self.depth, self.optimistic, self.isdir,
                           self.functions)
        if evaluated:
            scanner.variables, scanner.exported = dict(self.variables), set(self.exported)
            scanner.lost, scanner.startup, scanner.calling = self.lost, self.startup, self.calling
            scanner.partial = self.partial
        else:
            scanner.variables = self.environment(command)
            scanner.exported = set(scanner.variables)
        scanner.trial, scanner.defining = self.trial, self.defining
        scanner.nameref = self.nameref
        scanner.physical = self.physical or any(
            _PHYSICAL.match(word.passed() or word.text) for word in command.argv()[1:])
        if not evaluated:  # a child shell: the options before its text, and its environment
            arguments = command.argv()[1:] if arguments is None else arguments
            options = arguments[:shell_script_index([word.text for word in arguments])]
            if (self.starts_otherwise(command, scanner.variables)
                    or not _followed(options, _SHELL_LETTERS, False)):
                scanner.unfollowed()
        return scanner

    def scripts(self) -> None:
        """Scan the here-documents that are the scripts of shells (:func:`reads_script`).

        A body is the text of its shell, in the state the shell started
        in (:meth:`shell`), and the standard input of the text's commands
        too; those are derived (:attr:`SimpleCommand.derived`).  A body
        the parent shell expands first is text only the shell knows.
        """
        pending, index = self.functions.scripts, 0
        while index < len(pending):  # a script may feed another shell its own
            command, scanner = pending[index]
            index += 1
            if command.heredoc_body is None:
                command.unknown = True
                continue
            scanner.pairs = _Pairs(command.heredoc_body,
                                   self.functions.kept.get(command.heredoc_body, ()))
            scanner.text = scanner.pairs.text
            scanner.nested_command_list()
            for simple in scanner.commands:
                simple.derived = True
            self.commands.extend(scanner.commands)

    def starts_otherwise(self, command: SimpleCommand, environment: dict[str, str | None]) -> bool:
        """Return whether the child shell *command* runs may not run its text as written.

        It does not once a variable of :data:`_SHELL_STARTUP` may reach
        it with a value: one in its *environment* as the command sets it
        (``BASH_ENV=f bash -c``, ``env BASH_ENV=f bash -c``, an exported
        one; the empty value names no file and no option), one the text
        may have exported (:attr:`startup`) or, once a name reference may
        exist, any assignment before the command.  ``env -i`` drops what
        the shell exported; ``sh`` reads no ``BASH_ENV`` and ``dash``
        none of the three.
        """
        shell = command.program()
        if shell == "dash":
            return False
        names = _SHELL_STARTUP - {"BASH_ENV"} if shell == "sh" else _SHELL_STARTUP
        if any(name in names and value != "" for name, value in environment.items()):
            return True
        prefix = command.words[:len(command.words) - len(command.argv())]
        inherited = self.lost or self.startup
        wrapper, value_next = "", False
        for word in prefix:
            name = posixpath.basename(word.text)
            if wrapper and "=" in word.text and not word.text.startswith("-") and (
                    _ASSIGNMENT.match(word.text) is None):
                return True  # no variable's name: a function for the child (``BASH_FUNC_f%%``)
            if value_next:
                value_next = False
            elif name in _PREFIXES:
                wrapper = name
            elif not wrapper:
                if self.nameref and word.assignment:  # it may assign any variable
                    return True
            elif word.text.startswith("-"):
                cluster = _PREFIX_VALUE_CLUSTER.get(wrapper)
                value_next = word.text in _PREFIX_VALUE_OPTIONS.get(wrapper, ()) or (
                    cluster is not None and cluster.fullmatch(word.text) is not None)
                if wrapper == "env" and word.literal and _IGNORE_ENVIRONMENT.fullmatch(word.text):
                    inherited = False
        return inherited

    def environment(self, command: SimpleCommand) -> dict[str, str | None]:
        """Return the variables the child shell *command* runs starts with, and their values.

        The ones ``export`` exported, as the assignments before the
        command set them (``T= bash -c``, ``env T=x bash -c``); none
        after a wrapper that may reset the environment (``env -i``,
        ``sudo``).  A value only the shell knows is ``None``, as is every
        variable ``set -a`` or ``declare -x`` exported, which this scan
        does not follow (the child does not know them).  A starting
        shell takes the directory it starts in as its ``PWD`` whatever
        it was handed, and counts itself in ``SHLVL``.
        """
        argv = command.argv()
        variables = {name: value for name, value in self.variables.items()
                     if name in self.exported}
        for word in command.words[:len(command.words) - len(argv)]:
            if (match := _ASSIGNMENT.match(word.text)) is not None:
                known = (not match["append"] and not match["subscript"]
                         and f"-{match['name']}" not in self.variables)  # read-only: the old one
                variables[match["name"]] = word.value(match) if known else None
            elif posixpath.basename(word.text) in _ENVIRONMENT_WRAPPERS:
                variables = {}
        variables.pop("PWD", None)
        if "SHLVL" in variables:
            variables["SHLVL"] = None
        return variables

    def line_breaks(self) -> None:
        """Skip blanks, comments and newlines after ``&&``, ``||`` or ``|``: the list goes on.

        A function body may follow them too (:meth:`function_body`).
        """
        text = self.text
        while self.pos < len(text):
            char = text[self.pos]
            if char == "#":
                self.comment()
            elif char in " \t\n":
                self.pos += 1
                if char == "\n":
                    self.heredoc_bodies()
            else:
                break

    def comment(self) -> None:
        """Skip the comment at the position, to the newline that ends it.

        Bash keeps a backslash before that newline (:meth:`_Pairs.keep`).
        """
        newline = self.text.find("\n", self.pos)
        end = len(self.text) if newline < 0 else newline
        self.pairs.keep(self.pos + 1, end, first=True)
        self.pos = end

    def read_script(self, command: SimpleCommand, script: list[Word]) -> None:
        """Note that the scan read the words *script* of *command* as a script.

        A ``<`` in them is an operator the scan judged there, no text the
        command keeps (:attr:`_Functions.scripted`), unless another word
        or a here-string of the command can spell one too.
        """
        others = [word.text for word in command.words
                  if not any(word is part for part in script)]
        others += [operand.text for _, operator, operand in command.redirections
                   if operator == "<<<"]
        if not any(map(_LESS.search, others)) and id(command) in self.functions.holders:
            del self.functions.holders[id(command)]
            self.functions.scripted[id(command)] = command

    def hands_on(self, start: int, end: int, holder: SimpleCommand | Group | None,
                 stored: bool = False) -> None:
        """Note that *holder* hands on the text from *start* to *end*, when it can spell a ``<``.

        (:data:`_LESS`, where the scan read no operator.)  A word, an
        array's value, a here-document body.  The shell keeps it when
        *stored* (an assignment) and when no simple command holds it
        (:attr:`_Functions.stored`).
        """
        if any(match.start() not in self.operators
               for match in _LESS.finditer(self.text, start, end)):
            if stored or not isinstance(holder, SimpleCommand):
                self.functions.stored = True
            else:
                self.functions.holders[id(holder)] = holder

    def redirection(self, node: SimpleCommand | Group, fd: str, operator: str) -> None:
        """Scan the operand of a redirection and record what it does to *node*."""
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1
        start, scanned = self.pos, len(self.commands)
        if operator in ("<<", "<<-"):  # the shell expands nothing in a delimiter
            operand = self.word(expand=False)
        elif self.text.startswith(("<(", ">("), self.pos):
            stdin = operator == "<" and descriptor(fd, operator) == "0"
            operand = self.substitution(node if isinstance(node, Group) and stdin else None)
        else:
            operand = self.word()
        if node.heredoc and len(self.commands) > scanned and not self.optimistic:
            # bash expands an operand with the redirections before it done: a command
            # substituted in this one may read a here-document of them, where the scan sees
            # none.  The input of ``>(...)`` is the pipe: it has no here-document on
            # descriptor 0, unless a redirection opened or copied that descriptor.
            if not self.text.startswith(">(", start) or any(
                    mark in ("<", "<>", "<&", ">&")
                    or (mark in ("<<", "<<-") and descriptor(other, mark) != "0")
                    for other, mark, _ in node.redirections):
                self.hidden()
        node.redirections.append((fd, operator, operand))
        if fd.startswith("{"):  # bash assigns the variable the descriptor it opened
            self.variables[fd[1:-1]] = None
        numbers = {descriptor(fd, operator)}
        if len(fd) > 1:
            # ``11>x`` or ``{name}>x``: bash opens descriptor 11 or a new one,
            # sh reads a word before an unnumbered redirection.  A guard
            # must hold under both: the word is kept as an argument (before
            # the command word, where sh would run it, bash's command word
            # is the one that must be guarded), and the redirection is the
            # default descriptor's too (the stdout replay of a guard reads
            # the descriptor as written both ways).
            if isinstance(node, SimpleCommand) and node.argv():
                node.words.append(Word(fd, literal=True))
            numbers.add(descriptor("", operator))
        if operator in ("<<", "<<-"):
            # Quoting any of the delimiter keeps the body literal.
            expands = self.text[start:self.pos] == operand.text
            # The node reads it on whichever descriptor it is (``3<<EOF`` through ``<&3`` or
            # ``/dev/fd/3``); the text its standard input hands on is that of one on
            # descriptor 0, which a longer number is too as sh reads it.
            self.heredocs.append((operand.text, operator == "<<-", expands,
                                  node if "0" in numbers else None))
            # Who reads it is the shell's to know when bash names the descriptor (that one
            # outlives the command), and which of two is the standard input when bash and
            # sh number a descriptor differently (``bash <<A 11<<B``).
            if not self.optimistic and (fd.startswith("{") or (node.heredoc and any(
                    len(other) > 1 for other, mark, _ in node.redirections
                    if mark in ("<<", "<<-")))):
                self.hidden()
            node.heredoc = True
            if any(node is body for body in self.functions.bodies):
                # the here-document of a definition is every call's, a trap's too
                self.functions.heredocs = True
                if self.functions.trapped and not self.optimistic:
                    self.hidden()
        elif operator in _BOTH or ("1" in numbers and operator in _STDOUT_OPENERS
                                   and duplicated(operator, operand) is None):
            shell = len(self.stdout)
            if len(node.outputs) >= shell and all(
                    word is sink for word, sink in zip(node.outputs, self.stdout, strict=False)):
                del node.outputs[:shell]  # its own file replaces the shell's ``exec > file``
            node.outputs.append(operand)

    def heredoc_bodies(self) -> None:
        """Scan the bodies of the here-documents opened on the line that just ended.

        The text a body hands on is kept on the node reading it
        (:attr:`SimpleCommand.heredoc_body`).  Bash keeps the
        backslash-newline pairs of a body with a quoted delimiter
        (:meth:`_Pairs.keep_body`); it joined the lines of any other
        before it read the substitutions in them, so a quote or a
        comment there keeps none.
        """
        pending, self.heredocs = self.heredocs, []
        for delimiter, strip_tabs, expands, node in pending:
            lines: list[str] = []
            ended = False
            start, pairs = self.pos, self.pairs
            if expands:
                self.pairs = _NO_PAIRS
            else:
                pairs.keep_body(start, delimiter, strip_tabs)
            while not ended and self.pos < len(self.text):
                ended, line = self.body_line(delimiter, strip_tabs, expands)
                if not ended:
                    lines.append(line)
            self.pairs = pairs
            # (a shell's script is read as one, :meth:`scripts`, or refused unread)
            if id(node) not in self.functions.shells:
                self.hands_on(start, self.pos, node)
            if node is not None and ended:
                node.heredoc_body = handed_text("".join(f"{line}\n" for line in lines), expands)

    def body_line(self, delimiter: str, strip_tabs: bool, expands: bool) -> tuple[bool, str]:
        """Scan one line of a here-document body.

        A body is text, but the substitutions of an unquoted one
        (*expands*) run, so their commands are scanned.  (In such a
        body a line ending in a backslash was joined with the next
        where the text was: :class:`_Pairs`.)

        Returns:
            Whether the line was the delimiter, and the line (its leading
            tabs stripped when *strip_tabs*).
        """
        text = self.text
        start = self.pos
        newline = text.find("\n", self.pos)
        end = len(text) if newline < 0 else newline
        line = text[self.pos:end]
        if strip_tabs and line != delimiter:  # bash looks at the line as written first
            line = line.lstrip("\t")
        if line == delimiter:
            self.pos = end + 1
            return True, line
        if not expands:
            self.pos = end + 1
            return False, line
        while self.pos < len(text) and text[self.pos] != "\n":
            if text[self.pos] == "\\":
                self.pos += 2
            elif text.startswith(("$(", "`"), self.pos):
                self.expansion()
            elif text.startswith(("${", "$["), self.pos):
                self.assigning_expansion(end)
            else:
                self.pos += 1
        # the logical line: an expansion in it may go on over the next ones
        line = text[start:self.pos]
        self.pos += 1
        return False, line.lstrip("\t") if strip_tabs else line

    def assigning_expansion(self, end: int) -> None:
        """Scan the ``${...}`` or ``$[...]`` at the position, in a body line ending at *end*.

        For what it assigns (``${T:=x}``, ``${a[n++]}``, ``$[T=1]``); the
        commands in it are the line's (:meth:`body_line`).  It is read
        from a copy of the line, of a bounded length: the text after the
        body is commands.  One that does not close there goes on over
        the lines after it, where what it assigns is not read, so every
        variable is unknown then.
        """
        line = self.text[self.pos:min(end, self.pos + 1000)] + "\n"
        scanner = self.inherit(_Scanner(line, self.cwd, self.group, self.deeper(),
                                        self.optimistic, self.isdir, self.functions, _NO_PAIRS))
        scanner.variables, scanner.nameref = self.variables, self.nameref
        scanner.expansion(quoted=True)
        self.variables = scanner.variables  # another dictionary once it forgot them all
        self.partial = scanner.partial
        # what stands for an evaluation the scan cannot follow (:meth:`blind`)
        self.commands.extend(command for command in scanner.commands if not command.words)
        if scanner.pos >= len(line):  # it took the newline too: not closed in the line
            self.forget("$")
        self.pos += 2  # the commands in it are read where they are written

    def word(self, expand: bool = True) -> Word:
        """Scan the word at the position: quotes removed, expansions kept as written.

        *expand*: whether the shell expands the word (:meth:`expanded`);
        not a here-document delimiter, which gets quote removal only: a
        ``$`` or a backtick in it is a character of the delimiter, and no
        substitution in it runs.
        """
        text, start, literal, holder = self.text, self.pos, True, self.current
        parts: list[str] = []
        strings: list[int] = []  # where in *parts* its unquoted ``$'...'`` strings are
        # An unquoted ``{`` and, after it, an unquoted ``,`` or ``..``: a
        # brace expansion once a ``}`` closes it (``{a,b}``, ``{1..3}``).
        opened = listed = False
        # Whether the word is one field so far, and where in *parts* its literal
        # end starts: after its last expansion (:attr:`Word.path`).
        field, tail = True, 0
        while self.pos < len(text) and text[self.pos] not in _METACHARACTERS:
            char = text[self.pos]
            if char == "\\":  # an escaped character
                parts.append(text[self.pos + 1:self.pos + 2])
                self.pos += 2
            elif char == "'":
                end = self.closing(self.pos + 1, "'")
                parts.append(text[self.pos + 1:end - 1])
                self.pos = end
            elif char == '"':
                literal = self.double_quoted(parts, expand=expand) and literal
                opaque, several = self.opaque
                if opaque >= 0:
                    tail, field = opaque, field and not several
            elif char in "$`":
                string = expand and text.startswith("$'", self.pos)
                known = self.expanded(parts) if expand else self.unexpanded(parts)
                if string:  # a quoted string, kept as written (:attr:`Word.decoded`)
                    strings.append(len(parts) - 1)
                else:
                    literal = known and literal
                field = field and known  # unquoted, bash splits it, or drops it
            else:
                # a ``~`` after the ``=`` of an assignment or a ``:`` in its value expands too;
                # none does whose prefix is quoted (``~"/x"``, ``~\/x``: a directory named ``~``)
                glob = char in "*?[" or (
                    char == "~" and not _QUOTED.match(text, self.pos + 1) and (
                        self.pos == start or (text[self.pos - 1] in "=:"
                                              and _ASSIGNMENT.match(text, start) is not None)))
                if char == "{":
                    opened = True
                elif opened and (char == "," or text.startswith("..", self.pos)):
                    listed = True
                elif char == "}" and listed:
                    glob = True
                literal = literal and not glob
                parts.append(char)
                self.pos += 1
                if glob and char == "~":  # a home directory: one field
                    tail = len(parts)
                else:
                    field = field and not glob
        if text[start:self.pos] == "[":  # no pattern without its ``]``: the builtin's name
            literal = True
        assignment = _WRITTEN_ASSIGNMENT.match(text, start, self.pos) is not None
        if expand:
            self.hands_on(start, self.pos, holder, assignment)
        last = "".join(parts[tail:])
        if not strings:
            path = not literal and field and "/" in last and not last.endswith("/")
            return Word("".join(parts), literal, assignment, path)
        decoded = list(parts)
        for index in strings:
            try:
                decoded[index] = _ANSI_C_ESCAPE.sub(
                    _ansi_c_character, parts[index][2:].removesuffix("'"))
            except TooDeepError:  # only bash knows the string
                literal = False
        return Word("".join(parts), False, assignment,
                    decoded="".join(decoded) if literal else None)

    def double_quoted(self, parts: list[str], close: str = '"', expand: bool = True) -> bool:
        """Scan the ``"..."`` string at the position into *parts*; return whether it is literal.

        *close* ends it: the single quote of a ``'...'`` in an operand
        of a ``${...}`` that bash reads by the same rules
        (:meth:`parameter`).  *expand* is :meth:`word`'s.
        """
        text, literal = self.text, True
        opaque, several = -1, False
        self.pos += 1
        while self.pos < len(text) and text[self.pos] != close:
            char = text[self.pos]
            escaped = text[self.pos + 1:self.pos + 2]
            if char == "\\" and escaped and escaped in '$`"\\':
                parts.append(escaped)
                self.pos += 2
            elif char in "$`":
                known = (self.expanded(parts, True, True) if expand
                         else self.unexpanded(parts, True))
                literal = known and literal
                if not known:  # ``"$@"`` is a word for each parameter
                    opaque, several = len(parts), several or "@" in parts[-1]
                    variable = _VARIABLE.match(parts[-1])
                    value = (self.read(self.variables.get(variable[1] or variable[2]))
                             if variable else None)
                    end = _LITERAL_END.search(value or "")
                    if value and value.startswith(PATH_VALUE) and end and not self.nameref:
                        # one field that ends in a literal ``/name``: read as it was assigned
                        parts[-1:] = [value[len(PATH_VALUE):end.start()], end[0]]
            else:
                parts.append(char)
                self.pos += 1
        self.pos += 1
        self.opaque = (opaque, several)
        return literal

    def unexpanded(self, parts: list[str], quoted: bool = False) -> bool:
        """Skip the expansion at the position into *parts* as written; return ``True``.

        A here-document delimiter gets quote removal only: the ``$(...)``
        is text of the delimiter, so its commands are not commands of the
        line and its assignments do not happen.  A throwaway scanner
        finds its end.  Outside double quotes (*quoted*) bash's parser has
        made a quoted string of a ``$'...'``, decoded
        (:func:`_ansi_c_character`), and of a ``$"..."``, translated.

        Raises:
            TooDeepError: Only bash knows that string, so where the
                here-document ends (:data:`UNREAD_DELIMITER_VERDICT`).
        """
        scanner = _Scanner(self.text, self.cwd, self.group, self.depth, self.optimistic,
                           pairs=self.pairs)
        scanner.pos = self.pos
        written = scanner.expansion(quoted)
        self.pos = scanner.pos
        if not quoted and written.startswith("$'"):
            written = _ANSI_C_ESCAPE.sub(_ansi_c_character, written[2:].removesuffix("'"))
        elif not quoted and written == "$" and self.text.startswith('"', self.pos):
            raise TooDeepError(UNREAD_DELIMITER_VERDICT)
        parts.append(written)
        return True

    def expanded(self, parts: list[str], quoted: bool = False, expands: bool = False) -> bool:
        """Scan the expansion at the position into *parts*; return whether its value is known.

        An expansion that is one variable the command assigned a literal
        value (:attr:`variables`) contributes that value, as does
        ``$PWD`` where the directory is known and the command did not
        assign it, and the word stays literal; any other expansion is
        kept as written, as is a variable bash sets itself
        (:data:`_SHELL_SET`) and an unquoted one whose value bash would
        split into several words or expand as a pattern (blanks, ``*``,
        ``?``, ``[`` in it).
        """
        expansion = self.expansion(quoted, expands)
        variable = _VARIABLE.match(expansion)
        name = (variable[1] or variable[2]) if variable else ""
        if name in _SHELL_SET:
            value = None
        elif name == "PWD" and name not in self.variables:
            value = self.read(self.cwd)
        else:
            value = self.read(self.variables.get(name))
        # unquoted, bash splits the value on IFS, so only the default IFS is modelled
        # (nothing splits an empty one: the word, if it is all of it, is dropped)
        if self.nameref or _unknown(value) or (value and not quoted and (
                _SPLIT_OR_GLOB.search(value) or self.variables.get("IFS", " \t\n") != " \t\n")):
            value = None
        parts.append(expansion if value is None else value)
        return value is not None

    def expansion(self, quoted: bool = False, expands: bool = False) -> str:
        """Scan the expansion at the position and return its text.

        ``$x``, ``${x}``, ``$((...))``, ``$(...)``, ``$'...'`` or a backtick
        string.  In a double-quoted string (*quoted*) ``$'`` is a dollar
        and a quote character, not an ANSI-C string, which ends at its
        first quote no backslash escapes.  *expands*: bash reads the
        word of a ``${x:-word}`` double-quoted (:meth:`parameter`).
        """
        text, start = self.text, self.pos
        end = self.arithmetic(start + 3) if text.startswith("$((", start) else 0
        if end:
            self.pos = end
        elif text.startswith("$(", start):
            return self.substitution().text
        elif text.startswith("`", start):
            self.backtick()
        elif text.startswith("${", start):
            self.parameter(quoted, expands)
        elif text.startswith("$[", start):  # the older spelling of ``$((...))``
            self.pos, brackets = start + 2, 1
            self.depth = self.deeper()
            while self.pos < len(text) and brackets:  # read as :meth:`arithmetic` reads its own
                char = text[self.pos]
                if char in "$`":
                    self.expansion()
                elif char == '"':
                    self.double_quoted([])
                elif char == "'":
                    self.pos = self.closing(self.pos + 1, "'")
                else:
                    brackets += {"[": 1, "]": -1}.get(char, 0)
                    self.pos += 2 if char == "\\" else 1
            self.depth -= 1
            self.arithmetic_assignments(text[start + 2:self.pos - 1])
        elif text.startswith("$'", start) and not quoted:
            self.pos = self.closing(start + 2, "'", ansi_c=True)
        else:
            parameter = _PARAMETER.match(text, start)
            self.pos = parameter.end() if parameter else start + 1
        written = text[start:self.pos]
        # one that may fail: bash then skips the rest of the line (:attr:`partial`)
        if (end and not self.unfailing(written[3:-2])) or written.startswith("$[") or (
                written.startswith("${") and not _UNFAILING_PARAMETER.match(written)):
            self.partial = True
        return written

    def parameter(self, quoted: bool = False, expands: bool = False) -> None:
        """Scan the ``${...}`` at the position to its closing brace, and the commands in it.

        The word of ``${x:-word}`` (``:=``, ``:+``, ``:?``), the pattern
        and the replacement of ``${x#pat}``, ``${x%pat}`` and
        ``${x/pat/string}``, and the offset and length of
        ``${x:off:len}`` are expanded, so a command substitution or a
        backtick in them runs and is scanned.  As bash finds the end,
        quotes are skipped whole, a backslash escapes the character after
        it, a nested ``${`` closes with its own brace, and a bare ``{``
        does not nest.  The end of the text closes an unterminated one.
        Whether the single quotes of an operand quote (what is between
        them is text) or are characters to bash (what is between them
        expands by the double-quote rules, so its commands run and are
        scanned) depends on the operand: in the word of ``${x:-word}``
        (``-``, ``=``, ``+``) they are characters when bash reads the
        expansion double-quoted (*expands*: it is in a double-quoted
        string, or in such a word or offset of an outer one), in an
        offset or length always; in the message of ``${x:?word}`` and
        in a pattern they quote, and so in the expansions nested in
        them.  Two operands depend on the bash version, and are read as
        the version that runs their commands reads them: bash 3.2 (the
        macOS system bash) reads the replacement of a double-quoted
        ``${x/pat/string}`` by the double-quote rules (bash 4.3 and
        later remove its quotes), and it inserts the text of an ANSI-C
        ``$'...'`` in a double-quoted ``${...}`` (*quoted*) unquoted, so
        the commands in it run wherever the operand expands (bash 5.2
        single-quotes it in a pattern or a replacement; :meth:`ansi_c`).
        """
        text, opened, assigning = self.text, self.pos, False
        self.pos += 2
        self.depth = self.deeper()
        # The parameter comes first: the ``?`` of ``${?:+word}`` and the
        # ``#`` of ``${#:+word}`` name a parameter, not an operator.
        parameter = _BRACE_PARAMETER.match(text, self.pos)
        if parameter:
            self.pos = parameter.end()
        # ``${!T}`` expands the variable T names, or the element (its subscript is evaluated,
        # and may run a command); ``${T@P}`` expands T's value as a prompt, commands and all
        indirect = _INDIRECT.match(text, self.pos - len(parameter[0]) if parameter else self.pos)
        if text.startswith("@P", self.pos) or (indirect is not None and not _NAME.fullmatch(
                self.read(self.variables.get(indirect[1])) or "[")):
            self.forget("$")
            self.blind()
        # ``run``: the commands between single quotes run.  The first operator
        # character (``:`` with the one after it) decides it; ``slash`` is the
        # index after a ``/`` operator, where its pattern starts (a ``/`` there
        # is the operator's, ``${x//pat/string}``) and the next ``/`` ends it.
        run, operator, slash = expands, False, 0
        # Where arithmetic starts, which may assign (``${a[n++]}``, ``${x:n++}``):
        # at a subscript, or at the ``:`` of an offset.
        arithmetic = self.pos if text.startswith("[", self.pos) else 0
        while self.pos < len(text):
            char = text[self.pos]
            if char == "}":
                self.pos += 1
                break
            if char == "\\":
                self.pos += 2
            elif text.startswith("$'", self.pos) and (run or quoted):
                self.ansi_c(run)
            elif char in "$`":
                self.expansion(quoted, run)
            elif char == '"':
                self.double_quoted([])
            elif char == "'" and run:
                self.double_quoted([], "'")
            elif char == "'":
                self.pos = self.closing(self.pos + 1, "'")
            else:
                if not operator and char in "#%/^,:-=?+":
                    operator = True
                    if parameter and (char == "=" or text.startswith(":=", self.pos)):
                        # ``${T=word}``, ``${T:=word}``: T gets a value only the shell
                        # knows; ``${!T=word}`` the variable T names
                        assigning = True
                        if parameter[0].startswith("!"):
                            self.forget("$")
                        else:
                            self.variables[parameter[0]] = None
                    if char in "#%/^,":
                        run = False
                        slash = self.pos + 1 if char == "/" else 0
                    elif char == "?" or text.startswith(":?", self.pos):
                        run = False
                    elif char == ":" and text[self.pos + 1:self.pos + 2] not in ("-", "=", "+"):
                        run = True
                        arithmetic = arithmetic or self.pos
                elif slash and char == "/" and self.pos != slash:
                    run, slash = expands, 0  # the replacement
                self.pos += 1
        if arithmetic:
            self.arithmetic_assignments(text[arithmetic:self.pos])
        if assigning:  # the variable holds the word
            self.hands_on(opened, self.pos, None, stored=True)
        self.depth -= 1

    def ansi_c(self, expanded: bool) -> None:
        """Scan the ANSI-C string ``$'...'`` at the position, whose text bash expands.

        *expanded*: by the double-quote rules, in the word of a
        double-quoted ``${x:-word}``, where bash drops a
        backslash-newline pair of it; in a pattern it keeps one
        (:meth:`closing`).

        The parser ends it at its first quote no backslash escapes,
        whatever it holds; bash then decodes it and expands the text
        where it stands (:meth:`parameter`).  The expansions written in
        it are scanned by the double-quote rules, to that quote and no
        further: a here-document opened there has no body after it.
        What bash expands is another text where an escape decodes to
        syntax the string does not show (:data:`_DECODED`:
        ``\\x24(cmd)``) or stands in an expansion (``$(cat\\t>f)``): an
        evaluation the scan cannot follow (:meth:`blind`).  An expansion
        or a quote of that text may be left open, which fails the line
        (:attr:`partial`); plain text and the other escapes (``\\n``)
        are what they show.
        """
        text, start, heredocs = self.text, self.pos, self.heredocs
        end = self.closing(start + 2, "'", ansi_c=True, expanded=expanded)
        self.text, self.pos, self.heredocs = text[:end].removesuffix("'"), start + 2, []
        written, expands = self.text[start + 2:], False
        while self.pos < len(self.text):
            if self.text[self.pos] in "$`":
                self.expansion(True, True)
                expands = True
            else:
                self.pos += 2 if self.text[self.pos] == "\\" else 1
        self.text, self.pos, self.heredocs = text, end, heredocs
        if _DECODED.search(written) or (expands and "\\" in written):
            self.forget("$")
            self.blind()
        elif not expands and '"' not in written and "\\'" not in written:
            return
        self.partial = True

    def arithmetic(self, pos: int) -> int:
        """Scan the arithmetic expression from *pos* to its ``))``; return the index past it.

        *pos* is just past the opening ``((`` (or ``$((``).  ``0`` when
        the parentheses do not close as ``))``: then bash reads ``((`` as
        two subshells (``$((`` as a substitution starting with one) and
        the text is scanned again as commands.  The expression is no
        shell syntax (``<<`` is a shift), but bash runs the command
        substitutions in it, so their commands are scanned, and kept
        once the expression closes; the variables it assigns (``T=1``,
        ``i++``) are unknown afterwards.  Quotes are skipped whole, as
        bash does when it looks for the end.
        """
        if pos in self.not_arithmetic:
            return 0
        text = self.text
        scanner = self.inherit(_Scanner(text, self.cwd, self.group, self.deeper(),
                                        self.optimistic, self.isdir, self.functions, self.pairs))
        scanner.pos, scanner.not_arithmetic, depth = pos, self.not_arithmetic, 0
        scanner.variables, scanner.nameref = self.variables, self.nameref  # ``${T=1}`` in it
        scanner.physical, scanner.operators = self.physical, self.operators
        while scanner.pos < len(text):
            char = text[scanner.pos]
            if char == ")" and not depth:
                if not text.startswith("))", scanner.pos):
                    break
                self.commands.extend(scanner.commands)
                self.heredocs.extend(scanner.heredocs)
                self.variables = scanner.variables  # another dictionary once it forgot them all
                self.partial = scanner.partial
                self.arithmetic_assignments(text[pos:scanner.pos])
                self.operators.update(range(pos, scanner.pos))  # its value is a number
                return scanner.pos + 2
            if char in "$`":
                scanner.expansion()
            elif char == '"':
                scanner.double_quoted([])
            elif char == "'":
                scanner.pos = scanner.closing(scanner.pos + 1, "'")
            else:
                depth += {"(": 1, ")": -1}.get(char, 0)
                scanner.pos += 2 if char == "\\" else 1
        self.not_arithmetic.add(pos)
        return 0

    def backtick(self) -> None:
        """Scan the backtick substitution at the position, and the commands in it."""
        start = self.pos
        self.pos = self.closing(start + 1, "`")
        self.operators.update(range(start, self.pos))  # as in :meth:`substitution`
        body = self.text[start + 1:self.pos].removesuffix("`")
        group = Group("`", self.cwd, self.group, after_and=self.after_and)
        if self.current is not None:
            self.current.substitutions.append(group)
        scanner = self.inherit(_Scanner(re.sub(r"\\([$`\\])", r"\1", body), self.cwd, group,
                                        self.depth, self.optimistic, self.isdir, self.functions))
        scanner.current = scanner.evaluating = None  # its commands are its own
        scanner.after_or = scanner.after_and = False
        scanner.variables, scanner.nameref = dict(self.variables), self.nameref
        scanner.physical = self.physical
        scanner.nested_command_list()
        self.commands.extend(scanner.commands)

    def closing(self, pos: int, char: str, ansi_c: bool = False,
                expanded: bool = False) -> int:
        """Return the index just past the first *char* from *pos* on, or the end of the text.

        A backslash escapes the character after it, except when *char*
        is a single quote (nothing is escaped inside ``'...'``) that
        closes no ANSI-C string ``$'...'`` (*ansi_c*).  Bash keeps the
        backslash-newline pairs of a string a single quote closes
        (:meth:`_Pairs.keep`), unless it expands the text by the
        double-quote rules (*expanded*: :meth:`ansi_c`); it drops those
        between backticks.
        """
        text, start = self.text, pos
        while pos < len(text) and text[pos] != char:
            pos += 2 if text[pos] == "\\" and (char != "'" or ansi_c) else 1
        if char == "'" and not expanded:
            self.pairs.keep(start, pos)
        return min(pos + 1, len(text))
