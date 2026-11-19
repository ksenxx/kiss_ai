# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Split a memory page into statements, the units searched inside a page.

A statement is one bullet (with its continuation lines), one paragraph, one
table row or one fenced code block of the page body.  A fixed rule cuts
them out of the page; no model rewrites them.  To be embedded on its own,
each statement is prefixed with the page title and the heading it sits
under (:func:`page_statements`).  :class:`~kiss.core.memoryfield.index.VectorIndex`
stores one vector per statement and ranks a page by its own vector and its
best statement's.
"""

import re

from kiss.core.memoryfield.pages import page_title, split_frontmatter

# Version of the page-to-statements mapping, stored with each page's
# statement vectors.  ``VectorIndex.sync`` re-embeds the statements of a
# page stored under another value: bump this whenever the split or the
# prefix below changes.
STATEMENT_FORMAT = "1"

# Byte limits of one statement's embedding input (prefix included) and
# the number of statements embedded per page.  Together with the page's
# own input (at most ``MAX_PAGE_BYTES``) they keep one page within one
# embedding request (``EMBED_BATCH_INPUTS`` / ``EMBED_BATCH_BYTES``).
# The prefix has a limit of its own, above any title and heading met in
# practice (under 700 bytes), so that an absurd title cannot push the
# statement out of its input.
MAX_STATEMENT_BYTES = 2000
MAX_CODE_BYTES = 1500
MAX_PREFIX_BYTES = 1000
MAX_PAGE_STATEMENTS = 100

# CommonMark's fence and ATX heading: three or more backticks or tildes;
# one to six ``#`` followed by white space or the end of the line.
_FENCE_RE = re.compile(r"`{3,}|~{3,}")
_HEADING_RE = re.compile(r"#{1,6}(?:\s|$)")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE_CHARS = frozenset("|-: ")

_Block = tuple[str, str, int]
"""One statement of a page body: its heading, its text and its byte limit."""


def _end_paragraph(blocks: list[_Block], heading: str, paragraph: list[str]) -> None:
    """Move the open *paragraph* (if any) into *blocks* as one statement."""
    if paragraph:
        blocks.append((heading, " ".join(paragraph), MAX_STATEMENT_BYTES))
        paragraph.clear()


def _blocks(body: str) -> list[_Block]:
    """Cut *body* into its first :data:`MAX_PAGE_STATEMENTS` statements, in page order."""
    blocks: list[_Block] = []
    heading = ""
    paragraph: list[str] = []
    fence = ""  # the marks that opened the fenced block being read
    code: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if fence:
            # Closed by a line of its own marks, at least as many: a longer
            # fence may hold a shorter one.
            if stripped.startswith(fence) and not stripped.strip(fence[0]):
                blocks.append((heading, "\n".join(code), MAX_CODE_BYTES))
                fence, code = "", []
            else:
                code.append(line)
        elif opening := _FENCE_RE.match(stripped):
            _end_paragraph(blocks, heading, paragraph)
            fence = opening.group()
        elif _HEADING_RE.match(stripped):
            _end_paragraph(blocks, heading, paragraph)
            heading = stripped.lstrip("#").strip()
        elif not stripped:
            _end_paragraph(blocks, heading, paragraph)
        elif _TABLE_ROW_RE.match(line):
            _end_paragraph(blocks, heading, paragraph)
            if not _TABLE_RULE_CHARS.issuperset(stripped):  # skip the |---|---| rule
                blocks.append((heading, stripped, MAX_STATEMENT_BYTES))
        else:
            if _BULLET_RE.match(line):
                _end_paragraph(blocks, heading, paragraph)
            paragraph.append(stripped)
    if fence:  # a fence left open runs to the end of the page
        blocks.append((heading, "\n".join(code), MAX_CODE_BYTES))
    _end_paragraph(blocks, heading, paragraph)
    return [block for block in blocks if block[1].strip()][:MAX_PAGE_STATEMENTS]


def page_statements(name: str, raw: str) -> list[str]:
    """Return the embedding input of every statement of a page, in page order.

    Each input is ``"<title> - <heading>\\n<statement>"`` (just the title
    when the statement sits under no heading), the first line cut to
    :data:`MAX_PREFIX_BYTES` and the whole to :data:`MAX_STATEMENT_BYTES`
    (:data:`MAX_CODE_BYTES` for a code block) of UTF-8.  A page whose body
    holds only headings, or nothing, has no statement.

    Args:
        name: Page name, the title of a page without one.
        raw: Complete page text including frontmatter.
    """
    frontmatter, body = split_frontmatter(raw)
    title = page_title(frontmatter, name)
    statements = []
    for heading, text, limit in _blocks(body):
        prefix = (f"{title} - {heading}" if heading else title).encode()[:MAX_PREFIX_BYTES]
        data = prefix + f"\n{text}".encode()
        statements.append(data[:limit].decode("utf-8", errors="ignore"))
    return statements


def statement_text(raw: str, ordinal: int) -> str:
    """Return statement *ordinal* of a page as written in the page, without its prefix.

    Args:
        raw: Complete page text including frontmatter.
        ordinal: Position of the statement in :func:`page_statements`.

    Returns:
        The statement's text, or ``""`` when the page has no such statement
        (it changed since the ordinal was computed).
    """
    blocks = _blocks(split_frontmatter(raw)[1])
    return blocks[ordinal][1] if 0 <= ordinal < len(blocks) else ""
