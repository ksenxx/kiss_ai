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
directory it runs in, followed through ``cd``.  A subshell, a
substitution, a pipeline stage and a ``&`` list keep their ``cd`` and
their assignments to themselves; a ``cd`` that may not run (after
``||``, or in an ``if``, a loop or a ``case``) leaves the directory
unknown, as does one after ``&&`` once its list ends (the rest of the
list runs only when it ran), and a variable set any other way than by a literal
assignment (``read``, ``unset``, ``declare``, ``let``, a loop variable,
an arithmetic command or expansion, an assignment before a special
builtin, ``eval``, ``source``, an assigning expansion ``${T=x}``) is
unknown afterwards, as is every variable once a name reference exists,
and the directory after ``eval`` or ``source``; the literal text
``eval`` runs is scanned for its commands, as is a child shell's
``-c`` text, which knows the exported variables only.  A function body
is scanned as commands that may not run.  A bare ``exec > file`` is
the standard output of every later command that does not redirect its
own: the file as the ``exec`` opened it (a later ``cd`` does not move
it), in the shell that ran it (a pipeline stage's or a backgrounded
``exec`` ends with its subshell), and after an ``exec`` that may not
have run the earlier file is possible too.  Other commands are taken
to succeed.

Nothing is expanded: a word holding a parameter, a command or
arithmetic substitution, a leading ``~`` (or one after the ``=`` or a
``:`` of an assignment), a glob or a brace expansion is marked as not
literal, as is an unquoted variable once the command set ``IFS``; the
operands of a program run in another directory by its wrapper (``env
-C``) are not literal either.  Of an arithmetic expression only the
command substitutions are scanned (``<<`` in it is a shift).  It is a
scanner for guards, not a shell, and where bash versions disagree on
whether a substitution runs (the macOS system bash is 3.2) it is
scanned.
"""

from __future__ import annotations

import posixpath
import re
import sys
from dataclasses import dataclass, field

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
    "env": frozenset({"-C", "-S", "-u", "--chdir", "--split-string", "--unset"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "timeout": frozenset({"-k", "-s", "--kill-after", "--signal"}),
    "stdbuf": frozenset({"-e", "-i", "-o", "--error", "--input", "--output"}),
    "xargs": frozenset({
        "-a", "-d", "-E", "-e", "-I", "-L", "-l", "-n", "-P", "-s", "--arg-file", "--delimiter",
        "--eof", "--replace", "--max-lines", "--max-args", "--max-procs", "--max-chars",
        "--process-slot-var",
    }),
}
#: The options (by their start: ``-C dir``, ``-Cdir``, ``--chdir=dir``) of the
#: wrappers that run the program in another directory: its operands are
#: paths there, which only the shell resolves.
_PREFIX_CHDIR_OPTIONS = {"env": ("-C", "--chdir"), "sudo": ("-D", "--chdir")}
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
_RESERVED = _OPENERS | _CLOSERS | {"!", "then", "else", "elif", "do", "time", "function"}
_BARE_WORD = re.compile(r"(?:[{}!]|[a-z]+)(?=[ \t\n;&|<>()]|\Z)")
#: The ``()`` after a function's name (``f() { ...; }``), and the name after ``function``.
_FUNCTION_PARENS = re.compile(r"\([ \t]*\)")
_FUNCTION_NAME = re.compile(r"[ \t]*([^\s;&|<>()]+)")
#: ``T=x``, ``T+=x`` and an array element's ``T[0]=x``; ``T=(x y)`` starts the same way.
_ASSIGNMENT = re.compile(r"(?P<name>[A-Za-z_]\w*)(?P<subscript>\[[^\]]*\])?(?P<append>\+)?=")
_NAME = re.compile(r"[A-Za-z_]\w*")
_BLANKS = re.compile(r"[ \t]*")
#: Builtins that assign the variables their operands name (``read T``,
#: ``unset T``, ``declare -x T=x``, ``mapfile T``, ``let T=1``): afterwards
#: the variable's value is unknown.
_ASSIGNING_BUILTINS = frozenset({"read", "readarray", "mapfile", "declare", "typeset", "local",
                                 "readonly", "unset", "getopts", "let"})
#: ``read``'s options that take a value, attached (``-aT``) or next (``-a T``);
#: ``-a`` names the array read into.
_READ_VALUED = "adinNptu"
#: Builtins whose ``-n`` makes a name reference: assigning it assigns the
#: variable it names, so afterwards no variable is known.
_DECLARING_BUILTINS = frozenset({"declare", "typeset", "local"})
#: The variable a builtin fills when no operand names one (``read <<<x``
#: sets REPLY, ``mapfile <<<x`` MAPFILE), unknown after every such call.
_DEFAULT_TARGETS = {"read": "REPLY", "readarray": "MAPFILE", "mapfile": "MAPFILE"}
#: Builtins that run text as commands: afterwards no variable is known.
_SOURCING_BUILTINS = frozenset({"eval", "source", "."})
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


#: The POSIX special builtins: an assignment before one (``T=x export Y``)
#: holds afterwards in sh, and not in bash.
_SPECIAL_BUILTINS = frozenset({"break", ":", "continue", ".", "eval", "exec", "exit", "export",
                               "readonly", "return", "set", "shift", "times", "trap", "unset"})
#: An assignment in an arithmetic expression: ``i=0``, ``i+=1``, ``i++``, ``--i``.
_ARITHMETIC_ASSIGNMENT = re.compile(
    r"(?:\+\+|--)\s*([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*(?:\+\+|--|(?:[-+*/%&|^]|<<|>>)?=(?!=))")
# What may start at a token boundary: a process substitution, a redirection
# (with its file-descriptor number, or the ``{name}`` bash opens a new
# descriptor for) or a control operator.
_TOKEN = re.compile(
    r"(?P<substitution>[<>]\()"
    r"|(?P<fd>\d*|\{[A-Za-z_]\w*\})(?P<redirection><<<|<<-|<<|<>|<&|<|>>|>\||>&|>|&>>|&>)"
    r"|(?P<operator>;;&|;;|;&|&&|\|\||\|&|[;&|\n])"
)
_PARAMETER = re.compile(r"\$(?:[A-Za-z_]\w*|[0-9@*#?$!-])")
#: An expansion that is one variable and nothing else: ``$T`` or ``${T}``.
_VARIABLE = re.compile(r"\$(?:([A-Za-z_]\w*)|\{([A-Za-z_]\w*)\})\Z")
#: What makes bash split an unquoted expansion into words or expand it as a pattern.
_SPLIT_OR_GLOB = re.compile(r"[ \t\n*?\[]")
# The parameter a ``${`` opens: a name, a positional number or a special
# parameter (``?``, ``#``, ``-``, ...; the characters that are operators
# after a parameter), after the ``!`` of ``${!x}``; or the ``#`` of
# ``${#x}`` with the parameter it measures, which is one only when that
# parameter ends the expansion (``${##}`` measures ``$#``; ``${##pat}``
# strips ``pat`` from it).
_BRACE_PARAMETER = re.compile(
    r"#(?:[A-Za-z_]\w*|[0-9]+|[@*#?$!-])(?=[}\[])|!?(?:[A-Za-z_]\w*|[0-9]+|[@*#?$!-])"
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
#: :class:`TooDeepError`, a gate's with the commands found so far (each level
#: costs a few Python frames, so the interpreter's limit stays far).
MAX_NESTING = 100


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
    """

    text: str
    literal: bool


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

    def argv(self) -> list[Word]:
        """Return the command word and its arguments: assignments and wrappers dropped.

        A wrapper is matched by its basename (``/usr/bin/env``); its
        options, the value of an option that takes one (``sudo -u
        root``, ``timeout -s KILL``) and its numbers (``timeout 5``) are
        dropped with it.  Under a wrapper that runs the program in
        another directory (``env -C dir``) the arguments are not literal:
        an operand is a path there, which only the shell resolves.
        """
        value_options: frozenset[str] = frozenset()
        chdir_options: tuple[str, ...] = ()
        value_next = chdir = False
        for index, word in enumerate(self.words):
            text = word.text
            name = posixpath.basename(text)
            if value_next or _ASSIGNMENT.match(text) or text[:1].isdigit():
                value_next = False
            elif name in _PREFIXES:
                value_options = _PREFIX_VALUE_OPTIONS.get(name, frozenset())
                chdir_options = _PREFIX_CHDIR_OPTIONS.get(name, ())
            elif text.startswith("-"):
                value_next = text in value_options
                chdir = chdir or text.startswith(chdir_options)
            elif chdir:
                return [word, *(Word(w.text, literal=False) for w in self.words[index + 1:])]
            else:
                return self.words[index:]
        return []

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
    """Return whether *line* ends in an unescaped backslash: the next line continues it."""
    return (len(line) - len(line.rstrip("\\"))) % 2 == 1


def simple_commands(command: str, cwd: str | None,
                    optimistic: bool = False) -> list[SimpleCommand]:
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

    Raises:
        TooDeepError: The groups or expansions nest deeper than
            :data:`MAX_NESTING`, so the commands past them are unread,
            and the scan is not *optimistic*: a guard refuses the command.
    """
    scanner = _Scanner(command, cwd, None, optimistic=optimistic)
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
    """
    if not word.literal:
        return None
    text = bash_path(word.text)
    if text.startswith("/"):
        return posixpath.normpath(text)
    return None if cwd is None else posixpath.normpath(posixpath.join(cwd, text))


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


def _cd(cwd: str | None, args: list[Word]) -> str | None:
    """Return the directory ``cd args`` moves *cwd* to; ``None`` when only the shell knows it."""
    operands = [word for word in args if not word.text.startswith("-") or word.text == "-"]
    if not operands or operands[0].text == "-":
        return None  # $HOME or $OLDPWD
    return resolve(cwd, operands[0])


class _Scanner:
    """The position and state of one :func:`simple_commands` scan."""

    def __init__(self, text: str, cwd: str | None, group: Group | None,
                 depth: int = 0, optimistic: bool = False) -> None:
        """Start at the beginning of *text*, in the directory *cwd*, inside *group*.

        *depth* is how many groups *group* is nested in (:data:`MAX_NESTING`).
        """
        self.text = text
        self.pos = 0
        self.cwd = cwd
        self.group = group
        self.depth = depth
        self.optimistic = optimistic  # see :func:`simple_commands`
        # The variables the command assigned so far and their literal values:
        # ``None`` for one whose value only the shell knows (an expansion in
        # it, or an assignment that may not run).  Read back where a later
        # word expands ``$NAME`` / ``${NAME}``, so ``T=tmp/x && cat > $T/f``
        # names a literal path.
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
        # either name may assign the other, so no variable is known then.
        self.nameref = False
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
        # The state where the current and-or list and pipeline started, and
        # whether the pipeline has several stages (each runs in a subshell of
        # its own, as does a backgrounded list, so its ``cd``, assignments and
        # ``exec`` end with it: the state is restored after it).
        list_scope = pipeline_scope = self.scope()
        piped = False
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
                current = None
            elif char == "(":
                if isinstance(current, SimpleCommand) and self.array_value(current):
                    continue
                if isinstance(current, SimpleCommand) and self.function_name(current):
                    current = self.function_body()
                else:
                    current = self.group_command("(")
            elif not isinstance(current, SimpleCommand) and (word := self.reserved_word()):
                if word == "!":
                    self.negate = True
                elif word == "function":
                    current = self.function_body() if self.function_name() else None
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
                    current = current or self.start_command()
                    self.redirection(current, token["fd"], token["redirection"])
                else:
                    if not isinstance(current, SimpleCommand):
                        current = self.start_command()
                    current.words.append(self.substitution() if token else self.word())
        self.end_command(current)

    def scope(self) -> tuple[str | None, dict[str, str | None], set[str], tuple[Word, ...]]:
        """Return the state a subshell keeps to itself: directory, variables, exports, output."""
        return self.cwd, dict(self.variables), set(self.exported), self.stdout

    def restore(self, scope: tuple[str | None, dict[str, str | None], set[str],
                                   tuple[Word, ...]]) -> None:
        """Restore the state :meth:`scope` returned: a pipeline stage or a ``&`` list ended."""
        self.cwd, variables, exported, self.stdout = scope
        self.variables, self.exported = dict(variables), set(exported)

    def array_value(self, command: SimpleCommand) -> bool:
        """Skip the ``(...)`` at the position when it is the value of the assignment before it.

        ``T=(a b)`` and ``T+=(c)`` make ``T`` an array, whose words this
        scan does not follow, but for their substitutions, which run:
        the assignment's variable becomes unknown.  Returns whether the
        parenthesis was such a value.
        """
        if not command.words or not command.words[-1].text.endswith("="):
            return False
        if not _ASSIGNMENT.fullmatch(command.words[-1].text):
            return False
        text, depth = self.text, 0
        while self.pos < len(text):
            char = text[self.pos]
            if char == "'":
                self.pos = self.closing(self.pos + 1, char)
            elif char == '"':
                self.double_quoted([])
            elif char in "$`":
                self.expansion()
            elif text.startswith(("<(", ">("), self.pos):
                self.substitution()
            else:
                depth += {"(": 1, ")": -1}.get(char, 0)
                self.pos += 2 if char == "\\" else 1
                if not depth:
                    break
        command.words[-1] = Word(command.words[-1].text, literal=False)
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
            self.pipe_from.reader, self.pipe_from = command, None
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
            piped_from.reader, group.piped = group, True
        return group

    def reserved_word(self) -> str:
        """Consume the reserved word at the position and return it; ``""`` when there is none."""
        match = _BARE_WORD.match(self.text, self.pos)
        if match is None or match[0] not in _RESERVED:
            return ""
        self.pos = match.end()
        return match[0]

    def function_name(self, command: SimpleCommand | None = None) -> bool:
        """Consume the name and ``()`` of a function definition; return whether there is one.

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
                return False
            if self.commands and self.commands[-1] is command:
                self.commands.pop()
            self.pos = parens.end()
            return True
        name = _FUNCTION_NAME.match(text, self.pos)
        if name is None:
            return False
        blanks = _BLANKS.match(text, name.end())
        parens = _FUNCTION_PARENS.match(text, blanks.end() if blanks else name.end())
        self.pos = name.end() if parens is None else parens.end()
        return True

    def function_body(self) -> Group | None:
        """Scan the body of the function definition at the position; return it.

        The body (a brace group, a subshell, any compound command; blank
        lines and comments may precede it) runs only when the function is
        called: it is scanned as a group that may not run, so its
        commands are read, and a ``cd`` or an assignment in it leaves
        the directory or the variable unknown.  ``None`` without a body.
        """
        self.line_breaks()
        after_or, self.after_or = self.after_or, True
        if self.text.startswith("(", self.pos):
            group = self.group_command("(")
        elif (word := self.reserved_word()) in _OPENERS:
            group = self.group_command(word)
        else:
            group = None
        self.after_or = after_or
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
                 self.stdout)
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
         self.stdout) = saved
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
        a name reference no variable is known.
        A bare ``exec`` with a redirection of standard output to a file
        makes that file, as it opens it here, the output of the commands
        after it; one that may not run (after ``&&`` or ``||``, in a
        compound command, in a function body) leaves the earlier output
        possible too.  (An assignment or an ``exec`` in a pipeline
        segment or a backgrounded list is recorded too and undone by
        :meth:`command_list` when the segment or the list ends.)
        """
        if not isinstance(command, SimpleCommand):
            return
        argv = command.argv()
        name = argv[0].text if argv else ""
        group = command.group
        conditional = (self.after_or and not self.optimistic) or (
            group is not None and group.conditional)
        if name in ("cd", "pushd", "popd") and self.after_and:
            self.and_cd = True
        text = command_text(argv)
        if text is not None:
            # ``eval``'s literal text, a child shell's literal ``-c`` text:
            # scanned here (the child's ``cd`` and assignments stay its own).
            self.run_text(command, text)
        if not argv and any(word.text == "exec" for word in command.words):
            for fd, operator, operand in command.redirections:
                # a file is the shell's output from now on; a duplicate
                # (``>&2``) may be a file opened earlier, so it changes nothing
                if operator in _BOTH or (fd in ("", "1") and operator in _STDOUT and not (
                        operator == ">&" and (operand.text.isdigit() or operand.text == "-"))):
                    path = resolve(self.cwd, operand)
                    sink = Word(operand.text, False) if path is None else Word(path, True)
                    may_not_run = conditional or (self.after_and and not self.optimistic)
                    self.stdout = (*self.stdout, sink) if may_not_run else (sink,)
        if name == "cd":
            self.cwd = None if conditional else _cd(self.cwd, argv[1:])
        elif name == "pushd":
            # ``pushd DIR`` is a cd; ``pushd -n DIR`` only pushes and stays;
            # ``pushd`` alone or ``pushd +N`` / ``-N`` rotates the stack to a
            # directory only the shell knows.
            operands = [word for word in argv[1:] if word.text != "-n"]
            if conditional or not operands or operands[0].text[:1] in ("+", "-"):
                self.cwd = None
            elif len(operands) == len(argv) - 1:  # no -n
                self.cwd = _cd(self.cwd, operands)
        elif name == "popd":
            self.cwd = None
        elif name in _SOURCING_BUILTINS:
            self.variables = dict.fromkeys(self.variables)
            self.cwd = None  # the text it ran may have changed directory
        elif name in _ASSIGNING_BUILTINS:
            for word in argv[1:]:
                if not word.text.startswith("-"):
                    if named := _NAME.match(word.text):
                        self.variables[named[0]] = None
                        if name == "unset":
                            self.exported.discard(named[0])
                elif name in _DECLARING_BUILTINS and "n" in word.text[1:]:
                    self.nameref = True  # a name reference
                elif name == "read":  # ``-aT``, ``-raT``: the array read into
                    for at, letter in enumerate(word.text[1:], 2):
                        if letter in _READ_VALUED:
                            if letter == "a" and (named := _NAME.match(word.text, at)):
                                self.variables[named[0]] = None
                            break
            if name in _DEFAULT_TARGETS:
                self.variables[_DEFAULT_TARGETS[name]] = None
        elif name == "printf":  # ``printf -v T ...`` assigns T
            texts = [word.text for word in argv]
            for index, text in enumerate(texts[1:], 1):
                if text == "-v" and index + 1 < len(texts):
                    self.variables[texts[index + 1]] = None
                elif text.startswith("-v"):
                    self.variables[text[2:]] = None
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
        prefix = command.words[:len(command.words) - len(argv)]
        bare = not argv and all(_ASSIGNMENT.match(word.text) for word in prefix)
        if not bare and (not argv or name in _SPECIAL_BUILTINS):
            for word in prefix:
                if (match := _ASSIGNMENT.match(word.text)) is not None:
                    self.variables[match["name"]] = None
        assigned = prefix if bare else argv[1:] if name == "export" else []
        for word in assigned:
            match = _ASSIGNMENT.match(word.text)
            if match is None:
                continue
            # ``T+=x`` appends to a value this scan may not know; ``T[0]=x``
            # makes an array, whose ``$T`` is its first element.
            known = (word.literal and not conditional
                     and not match["append"] and not match["subscript"])
            self.variables[match["name"]] = word.text[match.end():] if known else None
            if known and self.after_and:
                self.and_assigned.add(match["name"])

    def run_text(self, command: SimpleCommand, text: str) -> None:
        """Scan *text*, the literal text *command* runs (:func:`command_text`), for its commands.

        They run in the command's directory, with its redirections, its
        here-document and its pipe, as the commands of a group would.
        ``eval``'s know the variables this shell knows; a child shell's
        know its environment (:meth:`environment`).
        """
        group = Group("(", self.cwd, self.group, outputs=command.outputs,
                      redirections=command.redirections, heredoc=command.heredoc,
                      piped=command.piped, after_and=command.after_and)
        scanner = _Scanner(text, self.cwd, group, self.depth, self.optimistic)
        if command.argv()[0].text == "eval":
            scanner.variables, scanner.exported = dict(self.variables), set(self.exported)
        else:
            scanner.variables = self.environment(command)
            scanner.exported = set(scanner.variables)
        scanner.nameref = self.nameref
        scanner.nested_command_list()
        self.commands.extend(scanner.commands)

    def environment(self, command: SimpleCommand) -> dict[str, str | None]:
        """Return the variables the child shell *command* runs starts with, and their values.

        The ones ``export`` exported, as the assignments before the
        command set them (``T= bash -c``, ``env T=x bash -c``); none
        after a wrapper that may reset the environment (``env -i``,
        ``sudo``).  A value only the shell knows is ``None``, as is every
        variable ``set -a`` or ``declare -x`` exported, which this scan
        does not follow (the child does not know them).
        """
        argv = command.argv()
        variables = {name: value for name, value in self.variables.items()
                     if name in self.exported}
        for word in command.words[:len(command.words) - len(argv)]:
            if (match := _ASSIGNMENT.match(word.text)) is not None:
                known = word.literal and not match["append"] and not match["subscript"]
                variables[match["name"]] = word.text[match.end():] if known else None
            elif posixpath.basename(word.text) in _ENVIRONMENT_WRAPPERS:
                variables = {}
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
            else:
                self.pos += 1
        # the logical line: a ``\``-newline pair continued it on the next one
        line = text[start:self.pos]
        self.pos += 1
        return False, line.lstrip("\t") if strip_tabs else line

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
                # a ``~`` after the ``=`` of an assignment or a ``:`` in its value expands too
                glob = char in "*?[" or (char == "~" and (self.pos == start or (
                    text[self.pos - 1] in "=:" and _ASSIGNMENT.match(text, start) is not None)))
                if char == "{":
                    opened = True
                elif opened and (char == "," or text.startswith("..", self.pos)):
                    listed = True
                elif char == "}" and listed:
                    glob = True
                literal = literal and not glob
                parts.append(char)
                self.pos += 1
        return Word("".join(parts), literal)

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
        kept as written, as is an unquoted one whose value bash would
        split into several words or expand as a pattern (blanks, ``*``,
        ``?``, ``[`` in it).
        """
        expansion = self.expansion(quoted, expands)
        variable = _VARIABLE.match(expansion)
        name = (variable[1] or variable[2]) if variable else None
        value = self.variables.get(name) if name else None
        if name == "PWD" and name not in self.variables:
            value = self.cwd
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
                            for known in self.variables:
                                self.variables[known] = None
                        else:
                            self.variables[parameter[0]] = None
                    if char in "#%/^,":
                        run = False
                        slash = self.pos + 1 if char == "/" else 0
                    elif char == "?" or text.startswith(":?", self.pos):
                        run = False
                    elif char == ":" and text[self.pos + 1:self.pos + 2] not in ("-", "=", "+"):
                        run = True
                elif slash and char == "/" and self.pos != slash:
                    run, slash = expands, 0  # the replacement
                self.pos += 1
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
        scanner = _Scanner(text, self.cwd, self.group, self.deeper(), self.optimistic)
        scanner.pos, scanner.not_arithmetic, depth = pos, self.not_arithmetic, 0
        scanner.variables, scanner.nameref = self.variables, self.nameref  # ``${T=1}`` in it
        while scanner.pos < len(text):
            char = text[scanner.pos]
            if char == ")" and not depth:
                if not text.startswith("))", scanner.pos):
                    break
                self.commands.extend(scanner.commands)
                self.heredocs.extend(scanner.heredocs)
                for prefixed, assigned in _ARITHMETIC_ASSIGNMENT.findall(text[pos:scanner.pos]):
                    self.variables[prefixed or assigned] = None
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
                           self.depth, self.optimistic)
        scanner.variables, scanner.nameref = dict(self.variables), self.nameref
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
