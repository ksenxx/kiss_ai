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
substitution, a pipeline stage and a ``&`` list keep their ``cd`` to
themselves; a ``cd`` that may not run (after ``||``, or in an ``if``,
a loop or a ``case``) leaves the directory unknown.  Other commands are
taken to succeed.

Nothing is expanded: a word holding a parameter, a command or
arithmetic substitution, a leading ``~`` or a glob is marked as not
literal; of an arithmetic expression only the command substitutions
are scanned (``<<`` in it is a shift).  It is a scanner for guards, not
a shell, so syntax it does not model (``exec`` redirections) is read
approximately.
"""

from __future__ import annotations

import posixpath
import re
import sys
from dataclasses import dataclass, field

_METACHARACTERS = frozenset(" \t\n;&|<>()")
# Words before the command word: wrappers that run the next word.
_PREFIXES = frozenset({
    "sudo", "command", "builtin", "exec", "nohup", "env", "nice", "timeout", "stdbuf",
})
# The options of those wrappers that take the next word as their value.
_PREFIX_VALUE_OPTIONS = {
    "sudo": frozenset({"-C", "-D", "-g", "-h", "-p", "-R", "-r", "-T", "-t", "-U", "-u"}),
    "exec": frozenset({"-a"}),
    "env": frozenset({"-C", "-S", "-u", "--chdir", "--split-string", "--unset"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "timeout": frozenset({"-k", "-s", "--kill-after", "--signal"}),
    "stdbuf": frozenset({"-e", "-i", "-o", "--error", "--input", "--output"}),
}
# Reserved words in command position: those that open or close a group, and
# those that only lead to the next command.
_OPENERS = frozenset({"{", "if", "while", "until", "for", "select", "case"})
_CLOSERS = frozenset({"}", "fi", "done", "esac"})
_RESERVED = _OPENERS | _CLOSERS | {"!", "then", "else", "elif", "do", "time"}
_BARE_WORD = re.compile(r"(?:[{}!]|[a-z]+)(?=[ \t\n;&|<>()]|\Z)")
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*\+?=")
# What may start at a token boundary: a process substitution, a redirection
# (with its file-descriptor number) or a control operator.
_TOKEN = re.compile(
    r"(?P<substitution>[<>]\()"
    r"|(?P<fd>\d*)(?P<redirection><<<|<<-|<<|<>|<&|<|>>|>\||>&|>|&>>|&>)"
    r"|(?P<operator>;;&|;;|;&|&&|\|\||\|&|[;&|\n])"
)
_PARAMETER = re.compile(r"\$(?:[A-Za-z_]\w*|[0-9@*#?$!-])")
# Redirections of standard output (``>&`` only when its operand is a file).
_STDOUT = frozenset({">", ">>", ">|", ">&"})
_BOTH = frozenset({"&>", "&>>"})
# Groups that run in a subshell (a ``cd`` in them ends with them), and those
# whose output is a word of another command rather than a stream.
_SUBSHELLS = frozenset({"(", "$(", "`"})
_SUBSTITUTIONS = frozenset({"$(", "`"})
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
        heredoc: Whether a here-document is its standard input.
        piped: Whether a pipe is its standard input.
        reader: The command or group its standard output is piped to.
    """

    kind: str
    cwd: str | None
    group: Group | None
    conditional: bool = False
    outputs: list[Word] = field(default_factory=list)
    heredoc: bool = False
    piped: bool = False
    reader: SimpleCommand | Group | None = None


@dataclass
class SimpleCommand:
    """One simple command of a command line.

    Attributes:
        cwd: The absolute directory it runs in; ``None`` when only the
            shell knows it (after ``cd "$D"``, say).
        group: The group it is in; ``None`` at the top level.
        words: Its words (assignments, wrappers, the command word and
            its arguments), without the redirections.
        outputs: The files its standard output is redirected to.
        heredoc: Whether a here-document is its own standard input.
        piped: Whether a pipe is its own standard input.
        reader: The command or group its standard output is piped to.
    """

    cwd: str | None
    group: Group | None
    words: list[Word] = field(default_factory=list)
    outputs: list[Word] = field(default_factory=list)
    heredoc: bool = False
    piped: bool = False
    reader: SimpleCommand | Group | None = None

    def argv(self) -> list[Word]:
        """Return the command word and its arguments: assignments and wrappers dropped.

        A wrapper is matched by its basename (``/usr/bin/env``); its
        options, the value of an option that takes one (``sudo -u
        root``, ``timeout -s KILL``) and its numbers (``timeout 5``) are
        dropped with it.
        """
        value_options: frozenset[str] = frozenset()
        value_next = False
        for index, word in enumerate(self.words):
            text = word.text
            name = posixpath.basename(text)
            if value_next or _ASSIGNMENT.match(text) or text[:1].isdigit():
                value_next = False
            elif name in _PREFIXES:
                value_options = _PREFIX_VALUE_OPTIONS.get(name, frozenset())
            elif text.startswith("-"):
                value_next = text in value_options
            else:
                return self.words[index:]
        return []

    def program(self) -> str:
        """Return the basename of the program it runs (``""`` for a bare redirection)."""
        argv = self.argv()
        return posixpath.basename(argv[0].text) if argv else ""

    def reads_heredoc(self) -> bool:
        """Return whether a here-document is its standard input: its own or a group's."""
        node: SimpleCommand | Group | None = self
        while node is not None:
            if node.heredoc:
                return True
            node = node.group
        return False


def simple_commands(command: str, cwd: str | None) -> list[SimpleCommand]:
    """Return the simple commands of the bash text *command*, in the order they start.

    Args:
        command: The shell command text, possibly several lines.
        cwd: The absolute POSIX directory it starts in; ``None`` when unknown.
    """
    scanner = _Scanner(command, cwd, None)
    scanner.command_list()
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

    def __init__(self, text: str, cwd: str | None, group: Group | None) -> None:
        """Start at the beginning of *text*, in the directory *cwd*, inside *group*."""
        self.text = text
        self.pos = 0
        self.cwd = cwd
        self.group = group
        self.commands: list[SimpleCommand] = []
        # The here-documents opened on the current line: (delimiter, strip
        # leading tabs, whether the substitutions of the body run).
        self.heredocs: list[tuple[str, bool, bool]] = []
        # What a ``|`` just piped, until the next command starts and reads it.
        self.pipe_from: SimpleCommand | Group | None = None
        # Whether the command or group being scanned follows ``||``.
        self.after_or = False
        # Where in the text a ``((`` was found not to close as ``))``: shared
        # with the scanners of its arithmetic expressions, so no expression is
        # scanned twice (nested ones would double the work at each level).
        self.not_arithmetic: set[int] = set()

    def command_list(self) -> None:
        """Scan commands to the end of the text, or to the end of the group being scanned."""
        text = self.text
        current: SimpleCommand | Group | None = None
        # Where the current and-or list and pipeline started, and whether the
        # pipeline has several stages (each runs in a subshell of its own).
        list_cwd = pipeline_cwd = self.cwd
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
                current = self.group_command("(")
            elif not isinstance(current, SimpleCommand) and (word := self.reserved_word()):
                if word in _OPENERS:
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
                    if operator in ("|", "|&"):
                        self.pipe_from, self.cwd, piped = current, pipeline_cwd, True
                        self.line_breaks()
                    else:
                        if operator == "&":
                            self.cwd = list_cwd  # the list ran in the background
                        elif piped:
                            self.cwd = pipeline_cwd
                        pipeline_cwd, piped = self.cwd, False
                        if operator in ("&&", "||"):
                            self.line_breaks()
                        else:
                            list_cwd = self.cwd
                    self.after_or = operator == "||"
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

    def start_command(self) -> SimpleCommand:
        """Start a simple command at the position; it reads the pipe a ``|`` left open."""
        command = SimpleCommand(self.cwd, self.group)
        self.commands.append(command)
        if self.pipe_from is not None:
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

    def compound(self, word: str) -> Group:
        """Scan the group the reserved *word* opens to its closing word; return the group.

        A ``cd`` in it holds after it, unless it may not have run.
        """
        outer = self.group
        conditional = word != "{" or self.after_or or (outer is not None and outer.conditional)
        group = self.group = Group(word, self.cwd, outer, conditional)
        self.command_list()
        self.group = outer
        return group

    def subshell(self, kind: str) -> Group:
        """Scan the commands of a subshell or a substitution to its ``)``; return its group.

        A ``cd`` in it ends with it.
        """
        saved = self.cwd, self.group, self.after_or
        group = self.group = Group(kind, self.cwd, self.group)
        self.after_or = False
        self.command_list()
        self.cwd, self.group, self.after_or = saved
        return group

    def substitution(self) -> Word:
        """Scan the ``$(...)``, ``<(...)`` or ``>(...)`` at the position as one word."""
        start = self.pos
        self.pos += 2
        self.subshell("$(")
        return Word(self.text[start:self.pos], literal=False)

    def end_command(self, command: SimpleCommand | Group | None) -> None:
        """Follow a finished ``cd``; one that may not run leaves the directory unknown."""
        if not isinstance(command, SimpleCommand):
            return
        argv = command.argv()
        name = argv[0].text if argv else ""
        if name in ("cd", "pushd"):
            group = command.group
            conditional = self.after_or or (group is not None and group.conditional)
            self.cwd = None if conditional else _cd(self.cwd, argv[1:])
        elif name == "popd":
            self.cwd = None

    def line_breaks(self) -> None:
        """Skip the blanks and newlines after ``&&``, ``||`` or ``|``: the list goes on."""
        text = self.text
        while self.pos < len(text) and text[self.pos] in " \t\n":
            self.pos += 1
            if text[self.pos - 1] == "\n":
                self.heredoc_bodies()

    def redirection(self, node: SimpleCommand | Group, fd: str, operator: str) -> None:
        """Scan the operand of a redirection and record what it does to *node*."""
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1
        start = self.pos
        if self.text.startswith(("<(", ">("), self.pos):
            operand = self.substitution()
        else:
            operand = self.word()
        stdout = fd in ("", "1") and operator in _STDOUT
        duplicate = operator == ">&" and (operand.text.isdigit() or operand.text == "-")
        if operator in ("<<", "<<-"):
            # Quoting any of the delimiter keeps the body literal.
            expands = self.text[start:self.pos] == operand.text
            self.heredocs.append((operand.text, operator == "<<-", expands))
            node.heredoc = node.heredoc or fd in ("", "0")
        elif operator in _BOTH or (stdout and not duplicate):
            node.outputs.append(operand)

    def heredoc_bodies(self) -> None:
        """Scan the bodies of the here-documents opened on the line that just ended."""
        pending, self.heredocs = self.heredocs, []
        for delimiter, strip_tabs, expands in pending:
            ended = False
            while not ended and self.pos < len(self.text):
                ended = self.body_line(delimiter, strip_tabs, expands)

    def body_line(self, delimiter: str, strip_tabs: bool, expands: bool) -> bool:
        """Scan one line of a here-document body; return whether it was the delimiter.

        A body is text, but the substitutions of an unquoted one
        (*expands*) run, so their commands are scanned.
        """
        text = self.text
        newline = text.find("\n", self.pos)
        end = len(text) if newline < 0 else newline
        line = text[self.pos:end]
        if (line.lstrip("\t") if strip_tabs else line) == delimiter:
            self.pos = end + 1
            return True
        if not expands:
            self.pos = end
        while self.pos < len(text) and text[self.pos] != "\n":
            if text[self.pos] == "\\":
                self.pos += 2
            elif text.startswith(("$(", "`"), self.pos):
                self.expansion()
            else:
                self.pos += 1
        self.pos += 1
        return False

    def word(self) -> Word:
        """Scan the word at the position: quotes removed, expansions kept as written."""
        text, start, literal = self.text, self.pos, True
        parts: list[str] = []
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
                literal = self.double_quoted(parts) and literal
            elif char in "$`":
                parts.append(self.expansion())
                literal = False
            else:
                glob = char in "*?[" or (char == "~" and self.pos == start)
                literal = literal and not glob
                parts.append(char)
                self.pos += 1
        return Word("".join(parts), literal)

    def double_quoted(self, parts: list[str]) -> bool:
        """Scan the ``"..."`` string at the position into *parts*; return whether it is literal."""
        text, literal = self.text, True
        self.pos += 1
        while self.pos < len(text) and text[self.pos] != '"':
            char = text[self.pos]
            escaped = text[self.pos + 1:self.pos + 2]
            if char == "\\" and escaped and escaped in '$`"\\\n':
                parts.append(escaped.replace("\n", ""))
                self.pos += 2
            elif char in "$`":
                parts.append(self.expansion())
                literal = False
            else:
                parts.append(char)
                self.pos += 1
        self.pos += 1
        return literal

    def expansion(self) -> str:
        """Scan the expansion at the position and return its text.

        ``$x``, ``${x}``, ``$((...))``, ``$(...)``, ``$'...'`` or a backtick string.
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
            self.pos = self.closing(start + 2, "}")
        elif text.startswith("$'", start):
            self.pos = self.closing(start + 2, "'")
        else:
            parameter = _PARAMETER.match(text, start)
            self.pos = parameter.end() if parameter else start + 1
        return text[start:self.pos]

    def arithmetic(self, pos: int) -> int:
        """Scan the arithmetic expression from *pos* to its ``))``; return the index past it.

        *pos* is just past the opening ``((`` (or ``$((``).  ``0`` when
        the parentheses do not close as ``))``: then bash reads ``((`` as
        two subshells (``$((`` as a substitution starting with one) and
        the text is scanned again as commands.  The expression is no
        shell syntax (``<<`` is a shift), but bash runs the command
        substitutions in it, so their commands are scanned, and kept
        once the expression closes.  Quotes are skipped whole, as bash
        does when it looks for the end.
        """
        if pos in self.not_arithmetic:
            return 0
        text = self.text
        scanner = _Scanner(text, self.cwd, self.group)
        scanner.pos, scanner.not_arithmetic, depth = pos, self.not_arithmetic, 0
        while scanner.pos < len(text):
            char = text[scanner.pos]
            if char == ")" and not depth:
                if not text.startswith("))", scanner.pos):
                    break
                self.commands.extend(scanner.commands)
                self.heredocs.extend(scanner.heredocs)
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
                           Group("`", self.cwd, self.group))
        scanner.command_list()
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
