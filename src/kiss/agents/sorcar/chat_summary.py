# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``chat_summaries`` table: one row per chat, derived from ``task_history``.

A row holds a 6-8 word summary of the chat's tasks (:func:`summarize_chat`)
and ``last_launched``, the launch instant of its newest listable task.
The table is a cache: :func:`upsert_chat_summary` recomputes a chat's row
from scratch from the chat's ``task_history`` rows, and whoever changes
those rows -- the daemon when a task finishes, the metadata backfill,
``kiss.scripts.sync_db`` when it merges another machine's tasks -- calls
it afterwards.  Nothing here depends on the rest of the package: the
module is stdlib-only because ``sync_db`` ships it to the remote machine
together with itself and runs it there with a bare ``python3``.
"""

from __future__ import annotations

import re
import sqlite3

_SUMMARY_MIN_WORDS = 6
_SUMMARY_MAX_WORDS = 8

#: A leading politeness phrase ("please", "can you", ...) that carries no
#: meaning; stripped before summarising and before classifying a task.
POLITENESS_FILLER = re.compile(
    r"^(?:(?:hi|hello|hey)[,!. ]+)?"
    r"(?:(?:can|could|would|will) you(?: please)?|please|"
    r"i(?:'d| would) like(?: you)? to|i (?:want|need)(?: you)? to|help me(?: to)?|"
    r"let'?s|kindly)\s+",
    re.IGNORECASE,
)
_SLASH_COMMAND = re.compile(r"^/([A-Za-z0-9_]+)")
_SENTENCE_END = re.compile(r"[.!?;:]\s|\n")
_TRAILING_STOPWORDS = frozenset(
    "a an the of to and or for with on at by in from as is are be that this "
    "which into onto than then so if it its".split()
)
_STRIP_CHARS = "\"'`*#()[]{}<>,:;"
_URL = re.compile(r"^<?https?://([^/\s<>|]+)")
_GREETINGS = frozenset({"hi", "hello", "hey", "thanks", "ok", "okay"})
_MAX_WORD_CHARS = 40

#: ``task_history`` rows the history panel lists: the chat's own tasks,
#: not the sub-agent runs they dispatched.
LISTABLE = "chat_id = ? AND (parent_task_id IS NULL OR parent_task_id = '')"


def _summary_word(token: str) -> str:
    """Normalise one whitespace-delimited token: URLs become their host, long tokens are cut."""
    url = _URL.match(token)
    if url:
        return url.group(1)
    word = token.strip(_STRIP_CHARS).rstrip(".!?")
    if len(word) > _MAX_WORD_CHARS:
        return word[: _MAX_WORD_CHARS - 1] + "…"
    return word


def _summary_sentences(task: str) -> list[list[str]]:
    """Return *task*'s sentences as lists of significant words, in order.

    Slash-command syntax and a leading politeness filler are stripped;
    a task that is only a greeting ("hi", "thanks") has no sentences.
    """
    text = task.strip()
    command = _SLASH_COMMAND.match(text)
    if command:
        text = command.group(1).replace("_", " ") + "\n" + text[command.end() :]
    sentences = []
    for chunk in _SENTENCE_END.split(text):
        chunk = POLITENESS_FILLER.sub("", chunk.strip(), count=1)
        words = [w for w in (_summary_word(t) for t in chunk.split()) if w]
        if words and not all(w.lower() in _GREETINGS for w in words):
            sentences.append(words)
    return sentences


def summarize_chat(tasks: list[str]) -> str:
    """Summarise a chat's tasks in 6-8 words.

    The summary is the leading words of the chat's tasks in
    chronological order: the first sentence of the first task states
    the chat's intent, and further sentences and later tasks contribute
    only while the summary is shorter than six words.  Trailing function
    words (``the``, ``of``, ``in``, ...) are dropped so the summary does
    not end mid-phrase; a chat whose tasks hold fewer than six words in
    total ("hi") yields what there is.

    Args:
        tasks: The chat's listable task texts, oldest first.

    Returns:
        The summary; empty when *tasks* has no words at all.
    """
    words: list[str] = []
    for sentence in (s for task in tasks for s in _summary_sentences(task)):
        if len(words) >= _SUMMARY_MIN_WORDS:
            break
        words.extend(sentence[: _SUMMARY_MAX_WORDS - len(words)])
    while len(words) > 1 and words[-1].lower() in _TRAILING_STOPWORDS:
        words.pop()
    return " ".join(words)


def launch_ms(start_ts: int | str | None, timestamp: float | str | None) -> int:
    """Return a ``task_history`` row's launch instant in epoch milliseconds.

    ``start_ts`` (already ms) when the daemon recorded one, else the
    row's insertion ``timestamp`` (epoch seconds) -- the fallback the
    history panel applies too.

    Args:
        start_ts: The row's ``start_ts`` column.
        timestamp: The row's ``timestamp`` column.
    """
    start = int(start_ts or 0)
    if start > 0:
        return start
    return int(float(timestamp or 0.0) * 1000)


def upsert_chat_summary(db: sqlite3.Connection, chat_id: str) -> None:
    """Recompute and store the ``chat_summaries`` row of *chat_id*.

    The summary is built from the chat's oldest listable tasks
    (sub-agent rows excluded, as in the history panel) and
    ``last_launched`` is the launch instant of its newest listable task.
    A chat with no listable task gets no row.  Plain tuples are read, so
    the connection's row factory does not matter.

    Args:
        db: Open connection inside the caller's transaction.
        chat_id: The chat session id.
    """
    first = db.execute(
        f"SELECT task FROM task_history WHERE {LISTABLE} ORDER BY timestamp ASC, rowid ASC LIMIT 5",
        (chat_id,),
    ).fetchall()
    if not first:
        return
    start_ts, timestamp = db.execute(
        f"SELECT start_ts, timestamp FROM task_history WHERE {LISTABLE} "
        "ORDER BY timestamp DESC, rowid DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    db.execute(
        "INSERT INTO chat_summaries (chat_id, summary, last_launched) VALUES (?, ?, ?) "
        "ON CONFLICT(chat_id) DO UPDATE SET summary = excluded.summary, "
        "last_launched = excluded.last_launched",
        (chat_id, summarize_chat([r[0] or "" for r in first]), launch_ms(start_ts, timestamp)),
    )
