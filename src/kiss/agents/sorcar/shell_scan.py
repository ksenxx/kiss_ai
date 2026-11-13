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

import posixpath
import re
import sys
from collections.abc import Callable, Iterator
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
    "sudo": frozenset({"-C", "-D", "-g", "-h", "-p", "-R", "-r", "-T", "-t", "-U", "-u"}),
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
#: The same as written, unquoted, where a word starts: a ``\<newline>`` may continue the name.
_WRITTEN_ASSIGNMENT = re.compile(
    r"[A-Za-z_](?:\w|\\\n)*(?:\[[^\]]*\])?(?:\\\n)*\+?(?:\\\n)*=", re.ASCII)
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
#: One after which bash runs later commands otherwise than their text says
#: (:meth:`_Scanner.unfollowed`): ``lastpipe`` runs a pipeline's last stage in the shell
#: itself, ``cdable_vars`` enters the directory a variable names, ``-k`` (``keyword``)
#: takes the assignments out of a command's arguments; a shell's own ``-O`` sets any.
_UNFOLLOWED = re.compile(r"-\w*[kO]|[$`]|(?:lastpipe|cdable_vars|keyword)$", re.DOTALL)
#: The conditions of a ``trap`` that runs when the shell ends, or on a signal from outside,
#: which no scan can place.  Any other (``DEBUG``, ``ERR``, ``RETURN``, ``CHLD``, one the
#: shell sends itself) may run between the commands that follow it.
_LATE_TRAPS = frozenset({"EXIT", "HUP", "INT", "QUIT", "TERM", "0", "1", "2", "3", "15"})
#: The variables under which a starting shell does not run its text as written: it reads
#: the file ``BASH_ENV`` names first, and starts with the options the other two list.
_SHELL_STARTUP = frozenset({"BASH_ENV", "BASHOPTS", "SHELLOPTS"})
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
    index, command_mode, long_options = 0, False, True
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
        command_mode = command_mode or "c" in word[1:]
        index += word.count("o", 1) + word.count("O", 1)
    return index if command_mode and index < len(words) else None


def command_text(argv: list[Word]) -> str | None:
    """The literal shell text the command *argv* runs as commands of its own, or ``None``.

    ``eval`` runs its arguments joined; a shell (:data:`SHELLS`) the
    text its ``-c`` names (:func:`shell_script_index`).  Text the shell
    expands first is only the shell's to read, as is an empty one.

    Args:
        argv: The command word and its arguments (:meth:`SimpleCommand.argv`).
    """
    if not argv:
        return None
    name = argv[0].text
    if name == "eval":
        if not all(word.literal for word in argv[1:]):
            return None
        return " ".join(word.text for word in argv[1:]) or None
    if posixpath.basename(name) in SHELLS:
        script = shell_script_index([word.text for word in argv[1:]])
        if script is not None and argv[script + 1].literal:
            return argv[script + 1].text or None
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
    what it cannot read; its message is :data:`TOO_DEEP_VERDICT`.
    """


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
    """

    text: str
    literal: bool
    assignment: bool = False


@dataclass
class Group:
    """Commands that share redirections and pipes.

    A subshell, a substitution, a brace group or a compound command: a
    redirection or a pipe after its end applies to the commands in it.

    Attributes:
        kind: ``"("`` (a subshell), ``"$("`` (a command or process
            substitution), ``"`"`` (a backtick substitution), ``"(("``
            (an arithmetic command, which holds no commands but those
            of its command substitutions) or the reserved word that
            opened it (``"{"``, ``"if"``, ``"for"``...).
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
        heredoc: Whether a here-document is its standard input.
        heredoc_body: The text that here-document hands on (see
            :attr:`SimpleCommand.heredoc_body`).
        piped: Whether a pipe is its standard input.
        reader: The command or group its standard output is piped to.
        after_and: Whether it (or the pipeline it is in) follows ``&&``.
        background: Whether ``&`` ends it (or the pipeline it ends).
        negated: Whether ``!`` precedes it (or the pipeline it starts):
            the shell inverts its exit status.
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
        heredoc: Whether a here-document is its own standard input.
        heredoc_body: The text that here-document hands on, every line
            newline-terminated (the leading tabs of a ``<<-`` body
            stripped): the body as written after a quoted delimiter, and
            after an unquoted one the body without a ``$`` or a backtick,
            with the backslash of ``\\\\`` dropped and ``\\<newline>``
            joined, as the shell hands it on.  ``None`` when there is no
            here-document, the shell would expand it, or the command
            does not end it.
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
            (``eval``, a shell's ``-c``; :func:`command_text`), which share
            its redirections, its here-document and its pipes; ``None``
            when it runs no such text.
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
        """Return whether a here-document is its standard input: its own or a group's."""
        return self.heredoc_reader() is not None

    def heredoc_reader(self) -> SimpleCommand | Group | None:
        """Return the node whose here-document is its standard input, or ``None``.

        Itself, or the group it is in.
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


def simple_commands(command: str, cwd: str | None, optimistic: bool = False,
                    isdir: Callable[[str], bool] | None = None) -> list[SimpleCommand]:
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

    Raises:
        TooDeepError: The groups or expansions nest deeper than
            :data:`MAX_NESTING`, so the commands past them are unread,
            and the scan is not *optimistic*: a guard refuses the command.
    """
    scanner = _Scanner(command, cwd, None, optimistic=optimistic, isdir=isdir)
    try:
        scanner.command_list()
    except TooDeepError:
        if not optimistic:
            raise
        # the gate's reading: the commands before the nesting
    return scanner.commands


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
    """

    changed = False

    def __setitem__(self, name: str, value: str | None) -> None:
        self.changed = True
        super().__setitem__(name, value)


#: A subshell's own state (:meth:`_Scanner.scope`): the directory, the variables, the exported
#: names, the files of the shell's output, ``set -P``, and whether a name reference may exist.
_Scope = tuple[str | None, dict[str, str | None], set[str], tuple[Word, ...], bool, bool]


class _Scanner:
    """The position and state of one :func:`simple_commands` scan."""

    def __init__(self, text: str, cwd: str | None, group: Group | None,
                 depth: int = 0, optimistic: bool = False,
                 isdir: Callable[[str], bool] | None = None,
                 functions: dict[str, tuple[frozenset[str], tuple[Word, ...]]] | None = None,
                 ) -> None:
        """Start at the beginning of *text*, in the directory *cwd*, inside *group*.

        *depth* is how many groups *group* is nested in (:data:`MAX_NESTING`).
        """
        self.text = text
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
        self.functions: dict[str, tuple[frozenset[str], tuple[Word, ...]]] = (
            {} if functions is None else functions)
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
        # The variables the ``coproc`` of the pipeline being read sets when it ends: the
        # array of its descriptors and ``NAME_PID``, its process id.
        coproc: tuple[str, ...] = ()
        while self.pos < len(text):
            char = text[self.pos]
            if char in " \t":
                self.pos += 1
            elif text.startswith("\\\n", self.pos):
                self.pos += 2
            elif char == "#":
                newline = text.find("\n", self.pos)
                self.pos = len(text) if newline < 0 else newline
            elif char == ")":
                self.pos += 1
                if self.group is not None and self.group.kind in ("(", "$("):
                    break
                self.end_command(current)  # a ``case`` pattern
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
                current, assigned = self.start_command(), None
                self.conditional(current)
            elif not isinstance(current, SimpleCommand) and (word := self.reserved_word()):
                if word == "!":
                    self.negate = True
                elif word == "function":
                    name = self.function_name()
                    current = self.function_body(name) if name else None
                elif word == "coproc":  # its command runs in a subshell, as a stage does
                    piped = True
                    named, command = self.coproc()
                    coproc = (named, f"{named}_PID")
                    if command is not None:
                        current, assigned = command, None
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
                    self.end_command(current)
                    if operator == "|" and pattern:  # ``a|b)``: the next alternative
                        current = None
                        continue
                    if operator in (";;", ";&", ";;&"):
                        pattern = case
                    if operator == "&" and current is not None:
                        current.background = True
                    if operator not in ("&&", "|", "|&"):
                        self.forget_and_assigned()
                    if operator in ("|", "|&"):
                        self.pipe_from, piped = current, True
                        self.restore(pipeline_scope)
                        self.line_breaks()
                    else:
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
                            list_scope = self.scope()
                    self.after_or = operator == "||"
                    self.after_and = operator == "&&"
                    current = None
                    if operator == "\n":
                        self.heredoc_bodies()
                elif token and token["redirection"]:
                    self.pos = token.end()
                    if current is None:
                        current, assigned = self.start_command(), _Assigned()
                    self.redirection(current, token["fd"], token["redirection"])
                else:
                    if not isinstance(current, SimpleCommand):
                        current, assigned = self.start_command(), _Assigned()
                    if assigned is None or not _WRITTEN_ASSIGNMENT.match(text, self.pos):
                        assigned = None
                    elif current.words:
                        self.assign(assigned, current.words)
                    with self.assigning(assigned if current.words else None):
                        current.words.append(self.substitution() if token else self.word())
        self.end_command(current)
        # The list ended at no operator (the text's end, the closer of the group it is in):
        # its last stage's subshell ends with it, and a coprocess's variables are set.
        if piped:
            self.restore(pipeline_scope)
        for variable in coproc:
            self.forget(variable)

    def scope(self) -> _Scope:
        """Return a subshell's own state (:data:`_Scope`), the mutable parts copied."""
        return (self.cwd, dict(self.variables), set(self.exported), self.stdout, self.physical,
                self.nameref)

    def restore(self, scope: _Scope) -> None:
        """Restore the state :meth:`scope` returned: a pipeline stage or a ``&`` list ended."""
        self.cwd, variables, exported, self.stdout, self.physical, self.nameref = scope
        self.variables, self.exported = dict(variables), set(exported)

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
            known = (words[-1].literal and not match["append"] and not match["subscript"]
                     and self.variables.get(match["name"]) is not None)
            assigned[match["name"]] = words[-1].text[match.end():] if known else None

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
            unknown = _Assigned(dict.fromkeys([*shell, "PWD", "IFS", "CDPATH"]))
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
        did is known then.  Returns whether the parenthesis was such a
        value.
        """
        if not command.words or not command.words[-1].text.endswith("="):
            return False
        if not _ASSIGNMENT.fullmatch(command.words[-1].text):
            return False
        text, depth, start = self.text, 0, False
        while self.pos < len(text):
            char = text[self.pos]
            if text.startswith("\\\n", self.pos):  # a continued line: no character of a word
                self.pos += 2
                continue
            if char == "#" and start:  # a comment, where a word of the value would start
                newline = text.find("\n", self.pos)
                self.pos = len(text) if newline < 0 else newline
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
                self.pos = len(text) if newline < 0 else newline
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
        command.words[-1] = Word(command.words[-1].text, False, command.words[-1].assignment)
        return True

    def start_command(self) -> SimpleCommand:
        """Start a simple command at the position; it reads the pipe a ``|`` left open."""
        command = SimpleCommand(self.cwd, self.group, conditional=self.after_or,
                                after_and=self.after_and, negated=self.negate)
        command.outputs.extend(self.stdout)  # the shell's ``exec > file``
        self.negate = False
        self.commands.append(command)
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
        arithmetic = opener == "(" and self.text.startswith("((", self.pos)
        end = self.arithmetic(self.pos + 2) if arithmetic else 0
        if end:
            group = Group("((", self.cwd, self.group)
            self.pos = end
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
                or text.startswith("\\\n", self.pos) or (bare and bare[0] in _RESERVED)
                or _CONDITIONAL.match(text, self.pos)
                or _WRITTEN_ASSIGNMENT.match(text, self.pos)):
            return "COPROC", None
        word = self.word()
        if _COPROC_COMMAND.match(text, self.pos):
            return word.text, None
        command = self.start_command()
        command.words.append(word)
        return "COPROC", command

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
        self.variables = {key: value for key, value in before.items() if value is not None
                          or key.startswith("-") or key in ("PWD", "IFS", "CDPATH")}
        self.variables["$"] = ""
        if self.text.startswith("(", self.pos):
            group = self.group_command("(")
        elif (word := self.reserved_word()) in _OPENERS:
            group = self.group_command(word)
        else:
            group = None
        self.after_or = after_or
        sets = {key for key, value in self.variables.items()
                if value is None and not key.startswith("-")}
        if self.variables.pop("$", None) is None:  # every variable, or a body defined another
            sets.add("$")
        if cwd is None or self.cwd is None:
            sets.add("PWD")
        if any(argv and not argv[0].literal
               for argv in map(SimpleCommand.argv, self.commands[first:])):
            sets |= {"$", "PWD"}
        self.variables = {**dict.fromkeys(before), **self.variables}
        earlier, sinks = self.functions.get(name, (frozenset(), ()))
        self.functions[name] = earlier | sets, (*sinks, *self.stdout[len(stdout):])
        return group

    def compound(self, word: str) -> Group:
        """Scan the group the reserved *word* opens to its closing word; return the group.

        A ``cd`` in it holds after it, unless it may not have run.  An
        assignment in it holds for the commands after it in the group;
        after a group that may not have run (a compound command, or a
        brace group after ``||``) the variable is unknown, and after a
        brace group after ``&&`` it is known to the rest of the and-list.
        """
        outer = self.group
        conditional = word != "{" or self.after_or or (outer is not None and outer.conditional)
        may_not_run = conditional or self.after_and
        if word in ("for", "select"):  # the loop variable takes the values it loops over
            blanks = _BLANKS.match(self.text, self.pos)
            loop_variable = _NAME.match(self.text, blanks.end() if blanks else self.pos)
            if loop_variable:
                self.variables[loop_variable[0]] = None
            if word == "select":  # the line read at each prompt
                self.variables["REPLY"] = None
        before = dict(self.variables)
        group = self.group = Group(word, self.cwd, outer, conditional)
        self.after_or = self.after_and = False
        and_assigned, self.and_assigned = self.and_assigned, set()
        and_cd, self.and_cd = self.and_cd, False
        self.nested_command_list()
        self.group = outer
        self.and_assigned = and_assigned
        # a brace group after ``&&`` that changed directory: known to the rest of the and-list
        self.and_cd = and_cd or (may_not_run and not conditional and self.cwd != group.cwd)
        if may_not_run:
            for name in before.keys() | self.variables.keys():
                if self.variables.get(name) != before.get(name):
                    if conditional:
                        self.variables[name] = None
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
                 self.stdout, self.physical)
        group = self.group = Group(kind, self.cwd, self.group)
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
         self.stdout, self.physical) = saved
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

    def substitution(self) -> Word:
        """Scan the ``$(...)``, ``<(...)`` or ``>(...)`` at the position as one word."""
        start = self.pos
        self.pos += 2
        self.subshell("$(").after_and = self.after_and  # as the word's command
        return Word(self.text[start:self.pos], literal=False)

    def end_command(self, command: SimpleCommand | Group | None) -> None:
        """Follow a finished ``cd`` or assignment; one that may not run leaves its state unknown.

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
        if not isinstance(command, SimpleCommand):
            return
        argv = command.argv()
        prefix = command.words[:len(command.words) - len(argv)]
        program = any(not word.assignment and posixpath.basename(word.text) in _PROGRAM_PREFIXES
                      for word in prefix)
        name = argv[0].text if argv and not program else ""
        group = command.group
        conditional = (self.after_or and not self.optimistic) or (
            group is not None and group.conditional)
        if name in ("cd", "pushd", "popd"):
            self.and_cd |= self.after_and
            if "PWD" in self.variables:  # bash sets it anew: what was assigned to it is gone
                self.variables["PWD"] = None
        # ``command`` and ``builtin`` run the builtin where a function has its name
        function = name in self.functions and not any(
            word.text in ("command", "builtin") for word in prefix)
        bare = not argv and all(word.assignment for word in prefix)
        if not bare and (not argv or name in _SPECIAL_BUILTINS):  # in ``eval``'s text too
            for word in prefix:
                if (match := _ASSIGNMENT.match(word.text)) is not None:
                    self.variables[match["name"]] = None
        text = command_text(argv)
        if text is not None:
            # ``eval``'s literal text, a child shell's literal ``-c`` text:
            # scanned here (the child's ``cd`` and assignments stay its own).
            self.run_text(command, text, name == "eval")
        if not argv and not program and any(word.text == "exec" for word in prefix):
            for fd, operator, operand in command.redirections:
                # a file is the shell's output from now on; a duplicate
                # (``>&2``) may be a file opened earlier, so it changes nothing
                if operator in _BOTH or (fd in ("", "1") and operator in _STDOUT and not (
                        operator == ">&" and (operand.text.isdigit() or operand.text == "-"))):
                    path = resolve(self.cwd, operand)
                    sink = Word(operand.text, False) if path is None else Word(path, True)
                    may_not_run = conditional or (self.after_and and not self.optimistic)
                    self.stdout = (*self.stdout, sink) if may_not_run else (sink,)
        if name in ("mapfile", "readarray") and any(
                word.text.startswith("-") and "C" in word.text for word in argv[1:]):
            name = "eval"  # its ``-C`` callback is evaluated, line by line
        if function:  # a call runs a body, which may call any defined by now
            for sets, sinks in self.functions.values():
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
            for word in argv[1:]:
                if word.text == "--":  # the positional parameters follow
                    break
                self.physical |= _PHYSICAL.match(word.text) is not None
                if _UNFOLLOWED.match(word.text):
                    self.unfollowed()
        elif name in ("alias", "enable") or (name == "trap" and any(
                not word.literal or word.text.upper().removeprefix("SIG") not in _LATE_TRAPS
                for word in argv[3 if argv[1:2] and argv[1].text == "--" else 2:])):
            self.unfollowed()
        elif name in ("test", "["):  # ``-v 'a[i++]'`` evaluates the subscript
            for word in argv[1:]:
                if (element := _ELEMENT.match(word.text)) is not None:
                    self.arithmetic_assignments(element[1])
        elif name in _SOURCING_BUILTINS:  # the text it ran may have set any variable
            self.forget("$")
            self.cwd = None  # and may have changed directory
        elif name in _ASSIGNING_BUILTINS:
            found = [match.lastindex for word in argv[1:] if (match := _ATTRIBUTE.match(word.text))]
            attribute = name == "readonly" or (name in _DECLARING_BUILTINS and bool(found))
            self.nameref |= attribute and 1 in found  # a name reference, or a name only bash reads
            integer = name in _DECLARING_BUILTINS and any(
                word.text.startswith("-") and set(word.text) & set("i$`") for word in argv[1:])
            for word in argv[1:]:
                if not word.text.startswith("-"):
                    target = self.forget(word.text)
                    # an integer's value is arithmetic: ``declare -i U=T=5``, or U made one before
                    if integer or f"-{target}" in self.variables:
                        self.arithmetic_assignments(word.text.partition("=")[2])
                    if name == "unset":
                        self.exported.discard(target)
                    elif attribute:  # what is assigned to it is not what it holds
                        self.variables[f"-{target}"] = None
                elif name == "read":  # ``-aT``, ``-raT``: the array read into
                    for at, letter in enumerate(word.text[1:], 2):
                        if letter in _READ_VALUED:
                            if letter == "a":
                                self.forget(word.text[at:])
                            break
            if name in _DEFAULT_TARGETS:
                self.variables[_DEFAULT_TARGETS[name]] = None
        elif name == "let":  # its operands are arithmetic expressions
            for word in argv[1:]:
                if word.literal:
                    self.arithmetic_assignments(word.text)
                else:  # one the shell expands may assign any variable
                    self.forget("$")
        elif name in _ASSIGNING_OPTION:  # ``printf -v T ...``, ``wait -np T`` assign T
            texts = [word.text for word in argv[1:]]
            for text, following in zip(texts, [*texts[1:], ""], strict=False):
                if text.startswith("-") and _ASSIGNING_OPTION[name] in text:
                    self.forget(text.partition(_ASSIGNING_OPTION[name])[2] or following)
        elif name == "export":  # ``export T``, ``export T=x``; ``export -n T`` undoes it
            options = "".join(word.text[1:] for word in argv[1:] if word.text.startswith("-"))
            names = {named[0] for word in argv[1:] if (named := _NAME.match(word.text))}
            if "f" not in options:  # exported functions are no variables
                if "n" in options:  # ran or not, the child's value is unknown
                    self.exported -= names
                elif not conditional:  # one that may not run changes nothing for sure
                    if self.after_and:
                        self.and_exported |= names - self.exported
                    self.exported |= names
        assigned = prefix if bare else argv[1:] if name == "export" and not function else []
        for word in assigned:
            match = _ASSIGNMENT.match(word.text)
            if match is None:
                if _target(word.text) is None:  # ``export "$N=x"``
                    self.forget(word.text)
                continue
            # ``T+=x`` appends to a value this scan may not know; ``T[0]=x``
            # makes an array, whose ``$T`` is its first element.
            known = (word.literal and not conditional and not match["append"]
                     and not match["subscript"] and f"-{match['name']}" not in self.variables)
            self.variables[match["name"]] = word.text[match.end():] if known else None
            if known and self.after_and:
                self.and_assigned.add(match["name"])
        for word in [*prefix, *([] if bare else assigned)]:
            match = _ASSIGNMENT.match(word.text)
            if match is not None:
                # an element's subscript and what an integer variable is assigned are
                # arithmetic: ``a[n++]=x`` assigns ``n`` too
                integer = f"-{match['name']}" in self.variables
                self.arithmetic_assignments(
                    (match["subscript"] or "") + (word.text[match.end():] if integer else ""))

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
        element = _ELEMENT.match(operand)
        if element is not None:
            self.arithmetic_assignments(element[1])
        return name or ""

    def unfollowed(self) -> None:
        """Give up on the variables and the directory: bash no longer runs the text as written.

        After an ``alias``, an ``enable``, a ``DEBUG``, ``ERR`` or
        ``RETURN`` trap or an option such as ``lastpipe``
        (:data:`_UNFOLLOWED`), what a later command sets or enters is not
        what its words say.  No variable is known from here on, nor the
        directory, here or after a ``cd``; a subshell's ends with it.
        """
        self.nameref = self.physical = True
        self.cwd = None

    def arithmetic_assignments(self, expression: str) -> None:
        """Make the variables the arithmetic *expression* assigns unknown (``T=1``, ``i++``).

        Every variable, when it assigns one only the shell can name (``$N=1``, ``$N++``).
        """
        for prefixed, assigned in _ARITHMETIC_ASSIGNMENT.findall(expression):
            self.variables[prefixed or assigned] = None
        if _ARITHMETIC_EXPANDED_TARGET.search(expression):
            self.forget("$")

    def cd(self, prefix: list[Word], args: list[Word]) -> str | None:
        """Return the directory ``cd args`` after the words *prefix* leads to (:func:`_cd`).

        ``CDPATH`` may be set once the command assigned it another value than
        the empty one (``CDPATH=dir cd sub``: for this ``cd`` alone), or may
        have assigned it through a name only bash reads.
        """
        cdpath = None if self.nameref else self.variables.get("CDPATH", "")
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
        text, pattern = self.text, False
        while self.pos < len(text):
            start, char = self.pos, text[self.pos]
            if char == "#":
                newline = text.find("\n", start)
                self.pos = len(text) if newline < 0 else newline
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
                self.arithmetic_assignments(command.words[-1].text)
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
        return Word(text[start:self.pos], literal=False)

    def run_text(self, command: SimpleCommand, text: str, evaluated: bool) -> None:
        """Scan *text*, the literal text *command* runs (:func:`command_text`), for its commands.

        They run in the command's directory (one only the shell knows
        under a wrapper that moves the program, ``env -C dir``), with its
        redirections, its here-document and its pipe, as the commands of
        a group would.  The ``eval`` of this shell's (*evaluated*) know the
        variables it knows; a child shell's know its environment
        (:meth:`environment`).
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
        else:
            scanner.variables = self.environment(command)
            scanner.exported = set(scanner.variables)
        scanner.nameref = self.nameref
        scanner.physical = self.physical or any(
            _PHYSICAL.match(word.text) for word in command.argv()[1:])
        if not evaluated:  # a child shell: the options before its text, and its environment
            arguments = command.argv()[1:]
            options = arguments[:shell_script_index([word.text for word in arguments])]
            if scanner.variables.keys() & _SHELL_STARTUP or any(
                    _UNFOLLOWED.match(word.text) for word in options):
                scanner.unfollowed()
        scanner.nested_command_list()
        if evaluated:  # what the text declared and set holds after it
            self.variables = scanner.variables  # its ``-NAME`` keys: the values end with ``eval``
            self.nameref, self.physical = scanner.nameref, scanner.physical
        self.commands.extend(scanner.commands)

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
                known = (word.literal and not match["append"] and not match["subscript"]
                         and f"-{match['name']}" not in self.variables)  # read-only: the old one
                variables[match["name"]] = word.text[match.end():] if known else None
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
                newline = text.find("\n", self.pos)
                self.pos = len(text) if newline < 0 else newline
            elif char in " \t\n":
                self.pos += 1
                if char == "\n":
                    self.heredoc_bodies()
            else:
                break

    def redirection(self, node: SimpleCommand | Group, fd: str, operator: str) -> None:
        """Scan the operand of a redirection and record what it does to *node*."""
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1
        start = self.pos
        if operator in ("<<", "<<-"):  # the shell expands nothing in a delimiter
            operand = self.word(expand=False)
        elif self.text.startswith(("<(", ">("), self.pos):
            operand = self.substitution()
        else:
            operand = self.word()
        node.redirections.append((fd, operator, operand))
        if fd.startswith("{"):  # bash assigns the variable the descriptor it opened
            self.variables[fd[1:-1]] = None
        if len(fd) > 1:
            # ``11>x`` or ``{name}>x``: bash opens descriptor 11 or a new one,
            # sh reads a word before an unnumbered redirection.  A guard
            # must hold under both: the word is kept as an argument (before
            # the command word, where sh would run it, bash's command word
            # is the one that must be guarded), the redirection is the
            # default descriptor's (the stdout replay of a guard merges the
            # two readings from the descriptor as written).
            if isinstance(node, SimpleCommand) and node.argv():
                node.words.append(Word(fd, literal=True))
            fd = ""
        # ``1<>file`` opens the file read-write on standard output: a write
        stdout = (fd == "1" and operator == "<>") or (fd in ("", "1") and operator in _STDOUT)
        duplicate = operator == ">&" and (operand.text.isdigit() or operand.text == "-")
        if operator in ("<<", "<<-"):
            # Quoting any of the delimiter keeps the body literal.
            expands = self.text[start:self.pos] == operand.text
            reads = fd in ("", "0")
            self.heredocs.append((operand.text, operator == "<<-", expands,
                                  node if reads else None))
            node.heredoc = node.heredoc or reads
        elif operator in _BOTH or (stdout and not duplicate):
            shell = len(self.stdout)
            if len(node.outputs) >= shell and all(
                    word is sink for word, sink in zip(node.outputs, self.stdout, strict=False)):
                del node.outputs[:shell]  # its own file replaces the shell's ``exec > file``
            node.outputs.append(operand)

    def heredoc_bodies(self) -> None:
        """Scan the bodies of the here-documents opened on the line that just ended.

        The text a body hands on is kept on the node reading it
        (:attr:`SimpleCommand.heredoc_body`).
        """
        pending, self.heredocs = self.heredocs, []
        for delimiter, strip_tabs, expands, node in pending:
            lines: list[str] = []
            ended = False
            while not ended and self.pos < len(self.text):
                ended, line = self.body_line(delimiter, strip_tabs, expands)
                if not ended:
                    lines.append(line)
            if node is not None and ended:
                node.heredoc_body = handed_text("".join(f"{line}\n" for line in lines), expands)

    def body_line(self, delimiter: str, strip_tabs: bool, expands: bool) -> tuple[bool, str]:
        """Scan one line of a here-document body.

        A body is text, but the substitutions of an unquoted one
        (*expands*) run, so their commands are scanned; and in such a
        body a line ending in a backslash is joined with the next before
        the delimiter is looked for, as the shell does.

        Returns:
            Whether the line was the delimiter, and the line (its leading
            tabs stripped when *strip_tabs*).
        """
        text = self.text
        start = self.pos
        newline = text.find("\n", self.pos)
        end = len(text) if newline < 0 else newline
        line = text[self.pos:end]
        while expands and _continued(line) and end < len(text):
            newline = text.find("\n", end + 1)
            line = line[:-1] + text[end + 1:len(text) if newline < 0 else newline]
            end = len(text) if newline < 0 else newline
        if strip_tabs:
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
        # the logical line: a ``\``-newline pair continued it on the next one
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
        scanner = _Scanner(line, self.cwd, self.group, self.deeper(), self.optimistic,
                           self.isdir, self.functions)
        scanner.variables, scanner.nameref = self.variables, self.nameref
        scanner.expansion(quoted=True)
        self.variables = scanner.variables  # another dictionary once it forgot them all
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
        text, start, literal = self.text, self.pos, True
        parts: list[str] = []
        # An unquoted ``{`` and, after it, an unquoted ``,`` or ``..``: a
        # brace expansion once a ``}`` closes it (``{a,b}``, ``{1..3}``).
        opened = listed = False
        while self.pos < len(text) and text[self.pos] not in _METACHARACTERS:
            char = text[self.pos]
            if char == "\\":  # an escaped character (a newline: a line continuation)
                parts.append(text[self.pos + 1:self.pos + 2].replace("\n", ""))
                self.pos += 2
            elif char == "'":
                end = self.closing(self.pos + 1, "'")
                parts.append(text[self.pos + 1:end - 1])
                self.pos = end
            elif char == '"':
                literal = self.double_quoted(parts, expand=expand) and literal
            elif char in "$`":
                literal = (self.expanded(parts) if expand else self.unexpanded(parts)) and literal
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
        assignment = _WRITTEN_ASSIGNMENT.match(text, start, self.pos) is not None
        return Word("".join(parts), literal, assignment)

    def double_quoted(self, parts: list[str], close: str = '"', expand: bool = True) -> bool:
        """Scan the ``"..."`` string at the position into *parts*; return whether it is literal.

        *close* ends it: the single quote of a ``'...'`` in an operand
        of a ``${...}`` that bash reads by the same rules
        (:meth:`parameter`).  *expand* is :meth:`word`'s.
        """
        text, literal = self.text, True
        self.pos += 1
        while self.pos < len(text) and text[self.pos] != close:
            char = text[self.pos]
            escaped = text[self.pos + 1:self.pos + 2]
            if char == "\\" and escaped and escaped in '$`"\\\n':
                parts.append(escaped.replace("\n", ""))
                self.pos += 2
            elif char in "$`":
                literal = (self.expanded(parts, True, True) if expand
                           else self.unexpanded(parts)) and literal
            else:
                parts.append(char)
                self.pos += 1
        self.pos += 1
        return literal

    def unexpanded(self, parts: list[str]) -> bool:
        """Skip the expansion at the position into *parts* as written; return ``True``.

        A here-document delimiter gets quote removal only: the ``$(...)``
        is text of the delimiter, so its commands are not commands of the
        line and its assignments do not happen.  A throwaway scanner
        finds its end.
        """
        scanner = _Scanner(self.text, self.cwd, self.group, self.depth, self.optimistic)
        scanner.pos = self.pos
        parts.append(scanner.expansion())
        self.pos = scanner.pos
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
            value = self.cwd
        else:
            value = self.variables.get(name)
        # unquoted, bash splits the value on IFS, so only the default IFS is modelled
        if self.nameref or (value is not None and not quoted and (
                _SPLIT_OR_GLOB.search(value) or self.variables.get("IFS", " \t\n") != " \t\n")):
            value = None
        parts.append(expansion if value is None else value)
        return value is not None

    def expansion(self, quoted: bool = False, expands: bool = False) -> str:
        """Scan the expansion at the position and return its text.

        ``$x``, ``${x}``, ``$((...))``, ``$(...)``, ``$'...'`` or a backtick
        string.  In a double-quoted string (*quoted*) ``$'`` is a dollar
        and a quote character, not an ANSI-C string.  *expands*: bash
        reads the word of a ``${x:-word}`` double-quoted
        (:meth:`parameter`).
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
            self.arithmetic_assignments(text[start:self.pos])
        elif text.startswith("$'", start) and not quoted:
            self.pos = self.closing(start + 2, "'")
        else:
            parameter = _PARAMETER.match(text, start)
            self.pos = parameter.end() if parameter else start + 1
        return text[start:self.pos]

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
        single-quotes it in a pattern or a replacement).
        """
        text = self.text
        self.pos += 2
        self.depth = self.deeper()
        # The parameter comes first: the ``?`` of ``${?:+word}`` and the
        # ``#`` of ``${#:+word}`` name a parameter, not an operator.
        parameter = _BRACE_PARAMETER.match(text, self.pos)
        if parameter:
            self.pos = parameter.end()
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
                self.pos += 1
                self.double_quoted([], "'")
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
        self.depth -= 1

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
        scanner = _Scanner(text, self.cwd, self.group, self.deeper(), self.optimistic,
                           self.isdir, self.functions)
        scanner.pos, scanner.not_arithmetic, depth = pos, self.not_arithmetic, 0
        scanner.variables, scanner.nameref = self.variables, self.nameref  # ``${T=1}`` in it
        scanner.physical = self.physical
        while scanner.pos < len(text):
            char = text[scanner.pos]
            if char == ")" and not depth:
                if not text.startswith("))", scanner.pos):
                    break
                self.commands.extend(scanner.commands)
                self.heredocs.extend(scanner.heredocs)
                self.variables = scanner.variables  # another dictionary once it forgot them all
                self.arithmetic_assignments(text[pos:scanner.pos])
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
        body = self.text[start + 1:self.pos].removesuffix("`")
        scanner = _Scanner(re.sub(r"\\([$`\\])", r"\1", body), self.cwd,
                           Group("`", self.cwd, self.group, after_and=self.after_and),
                           self.depth, self.optimistic, self.isdir, self.functions)
        scanner.variables, scanner.nameref = dict(self.variables), self.nameref
        scanner.physical = self.physical
        scanner.nested_command_list()
        self.commands.extend(scanner.commands)

    def closing(self, pos: int, char: str) -> int:
        """Return the index just past the first *char* from *pos* on, or the end of the text.

        A backslash escapes the character after it, except when *char*
        is a single quote (nothing is escaped inside ``'...'``).
        """
        text = self.text
        while pos < len(text) and text[pos] != char:
            pos += 2 if text[pos] == "\\" and char != "'" else 1
        return min(pos + 1, len(text))
