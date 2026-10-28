# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The quotes of a finish summary checked against what the run's tools returned.

A finish summary quotes commands and their output as evidence.  The finish
gate (:meth:`kiss.agents.seas.coding.coding_sea.ContainerHarness.quoted_evidence_verdict`)
refuses a summary whose quoted text appeared in no tool result of the run:
the agent cannot invent an output, and a reviewer reading the summary sees
where every quote came from.

What counts as a quote: the text of ``<pre>``, ``<code>``, ``<samp>``,
``<kbd>``, ``<tt>``, ``<q>`` and ``<blockquote>`` elements and of fenced or
single-backtick spans.  Each quote is
split into lines; a leading shell prompt (``$``, ``#``, ``>``) and the
elisions (``...``) are removed; every remaining piece of at least
:data:`MIN_QUOTE_CHARS` characters must be found, whitespace-normalised,
in one of the run's :class:`Source` texts.

What supports a piece:

- the task statement (quoting its words is how a reading is settled);
- a tool result, unless the piece is also written inside a shell command
  the agent ran (``echo 'All 27 tests passed'`` prints its argument, which
  proves nothing, whether the command ran in the foreground, in a
  background job or through ``run_commands_parallel``);
- a shell command the agent ran, from the start of the command or of one
  of its ``&&``, ``||``, ``;``, ``|``, ``&`` or ``(`` segments, separators
  inside quotes not counting (``cd /app && pytest -q`` supports ``pytest
  -q``; ``echo 'All 27 tests passed'`` and ``true 'x; All 27 tests passed'``
  do not support ``All 27 tests passed``).

The check establishes that a quote was observed, not that it means what the
summary says: a command can print any text it is given in a form the
substring rule does not see (``printf``, ``base64 -d``, a script the agent
wrote).  That is why every quote's provenance (the command, and whether it
names a file the agent wrote) is handed to the reviewer.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

#: Shortest quoted piece that is checked; shorter ones (a flag, a file name,
#: a number) are not evidence of anything on their own.
MIN_QUOTE_CHARS = 12
#: Most unsupported quotes named in a verdict, and the longest form of each.
MAX_QUOTES_SHOWN = 20
MAX_QUOTE_SHOWN_CHARS = 200

QUOTE_ELEMENTS = re.compile(r"<(pre|code|samp|kbd|tt|q|blockquote)\b[^>]*>(.*?)</\1\s*>",
                            re.S | re.I)
FENCED_SPAN = re.compile(r"```[^\n`]*\n?(.*?)```", re.S)
BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")
LINE_BREAK_TAGS = re.compile(r"<br\s*/?>|</(?:p|li|div|tr)\s*>", re.I)
TAGS = re.compile(r"<[^>]+>")
SHELL_PROMPT = re.compile(r"^(?:\$|#|>>>|>)\s+")
ELLIPSIS = re.compile(r"\.{3,}|…")
WHITESPACE = re.compile(r"\s+")
#: Shell separators after which a new command starts, longest first.
SEPARATORS = ("&&", "||", ";", "|", "&", "(", "\n")


@dataclass(frozen=True)
class Source:
    """One text a quote may come from.

    Attributes:
        kind: ``"statement"`` (the task statement), ``"output"`` (a tool
            result) or ``"command"`` (a shell command the agent ran).
        text: The text itself.
        label: How the source is named to the reviewer (``"output of: pytest -q"``).
    """

    kind: str
    text: str
    label: str = ""


def normalize(text: str) -> str:
    """*text* with HTML entities decoded and every whitespace run collapsed to one space."""
    return WHITESPACE.sub(" ", html.unescape(text)).strip()


def quoted_pieces(summary_html: str) -> list[str]:
    """The checkable pieces of the literal quotes in *summary_html*, in order, deduplicated.

    Args:
        summary_html: The ``summary_in_html`` argument of a finish call.

    Returns:
        Whitespace-normalised pieces of at least :data:`MIN_QUOTE_CHARS`
        characters: one per quoted line, split at elisions, with a leading
        shell prompt removed.
    """
    fragments = [match.group(2) for match in QUOTE_ELEMENTS.finditer(summary_html)]
    rest = QUOTE_ELEMENTS.sub(" ", summary_html)
    fragments += FENCED_SPAN.findall(rest)
    fragments += BACKTICK_SPAN.findall(FENCED_SPAN.sub(" ", rest))
    pieces: dict[str, None] = {}
    for fragment in fragments:
        text = html.unescape(TAGS.sub("", LINE_BREAK_TAGS.sub("\n", fragment)))
        for line in text.splitlines():
            for piece in ELLIPSIS.split(SHELL_PROMPT.sub("", line.strip())):
                piece = normalize(piece)
                if len(piece) >= MIN_QUOTE_CHARS:
                    pieces[piece] = None
    return list(pieces)


def command_starts(command: str) -> list[int]:
    """The offsets of *command* at which a command starts.

    Offset 0 and the offset after every separator of :data:`SEPARATORS`
    that is outside single or double quotes and not escaped.

    Args:
        command: A shell command line.
    """
    starts, quote, i = [0], "", 0
    while i < len(command):
        char = command[i]
        if char == "\\" and quote != "'":
            i += 2
            continue
        if quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        else:
            separator = next((s for s in SEPARATORS if command.startswith(s, i)), "")
            if separator:
                i += len(separator)
                starts.append(i)
                continue
        i += 1
    return starts


def command_supports(command: str, piece: str) -> bool:
    """Whether the normalised *piece* starts at a command start of *command*."""
    return any(command[start:].lstrip().startswith(piece) for start in command_starts(command))


def quote_sources(summary_html: str, sources: list[Source]) -> dict[str, str]:
    """Where each quoted piece of *summary_html* comes from.

    Args:
        summary_html: The ``summary_in_html`` argument of a finish call.
        sources: The run's sources, in the order they are searched.

    Returns:
        Each piece of :func:`quoted_pieces` mapped to the label of the first
        source that supports it, or to ``""`` when none does.
    """
    found = dict.fromkeys(quoted_pieces(summary_html), "")
    commands = [normalize(source.text) for source in sources if source.kind == "command"]
    for source in sources:
        if all(found.values()):
            break
        text = normalize(source.text)
        for piece, label in found.items():
            if label:
                continue
            if source.kind == "command":
                supported = command_supports(text, piece)
            elif source.kind == "output":
                supported = piece in text and not any(piece in command for command in commands)
            else:
                supported = piece in text
            if supported:
                found[piece] = source.label or source.kind
    return found


def unsupported_quotes(provenance: dict[str, str]) -> list[str]:
    """The pieces of :func:`quote_sources`'s answer that no source supports."""
    return [piece for piece, label in provenance.items() if not label]


def quotes_report(provenance: dict[str, str]) -> str:
    """The provenance of the summary's quotes as lines for the reviewer."""
    if not provenance:
        return "(the summary quotes nothing checkable)"
    lines = []
    for piece, label in provenance.items():
        shown = piece[:MAX_QUOTE_SHOWN_CHARS]
        lines.append(f'- "{shown}": {label or "NOT FOUND in any tool result or command"}')
    return "\n".join(lines)
