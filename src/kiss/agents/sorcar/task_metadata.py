"""Derived task metadata: classification tags, chat summaries, SEA names.

Everything here is a pure function of the ``task_history`` text columns
and runs inside the persistence write path when a task finishes
(``persistence._save_task_extra`` with ``endTs``), so it is deterministic,
offline and cheap: a handful of regular expressions per task, never a
model call.  :func:`backfill_task_metadata` applies the same functions
to every row of an existing database.

Columns produced:

* ``task_history.tags`` — comma-separated tags from :func:`classify_task_tags`:
  one of ``work`` / ``personal`` first, then ``secret``, ``chore`` and
  activity tags (``coding``, ``testing``, ``research``, ...).
* ``task_history.sea`` — the file stem of the SEA (agent script) that ran
  the task, from the run's ``agentPath``; backfilled for old rows from the
  parent trajectory's ``run_agent`` tool calls (:func:`sea_name_of_agent`).
* ``chat_summaries`` — per chat a 6-8 word summary (:func:`summarize_chat`)
  and the launch instant of the chat's latest task.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

#: Upper bound on the number of tags stored per task: the history panel
#: shows them inline before the "launched ..." label.
MAX_TAGS = 6

_SUMMARY_MIN_WORDS = 6
_SUMMARY_MAX_WORDS = 8

#: Rows written per transaction by :func:`backfill_task_metadata`.
_BACKFILL_BATCH = 500


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_PERSONAL = _rx(
    r"\b(grocer(?:y|ies)|recipes?|cook(?:ing)?|dinner|lunch|breakfast|restaurants?|"
    r"travel|flights?|hotels?|vacation|holidays?|trip|doctor|dentist|birthday|gifts?|"
    r"my (?:wife|husband|kids?|children|son|daughter|mom|dad|mother|father|family|"
    r"friends?|car|house|apartment|health)|movies?|netflix|playlist|workout|fitness|"
    r"gym|diet|wedding|party|amazon|shopping|personal|car|shops?|stores?|near me|"
    r"within \d+ miles)\b"
)
_SECRET = _rx(
    r"\b(passwords?|passwd|api[ _-]?keys?|secrets?|credentials?|access tokens?|"
    r"private keys?|ssh keys?|bank account|social security|ssn|credit card|otp|2fa|"
    r"confidential)\b|\.env\b"
)
_CHORE = _rx(
    r"\b(git pull|pull the latest|merge|rebase|sync|install|reinstall|upgrade|"
    r"update (?:the )?dependenc(?:y|ies)|bump|rename|clean ?up|reformat|format the|"
    r"lint|commit|push|restart|rerun|re-run|backup|purge|"
    r"what have the tasks? .* done|partial results|progress so far)\b"
)
_QUESTION = _rx(
    r"^(?:what|why|how|who|which|is|are|does|do|did|has|have|should|will|would|could|can|"
    r"explain|tell me)\b"
)

#: Only the opening of a task states its intent; the primary tags
#: (personal / secret / chore / question) are decided on this many
#: characters, the activity tags on :data:`_ACTIVITY_HEAD` — appended
#: boilerplate (model-routing protocols, sub-agent instructions) would
#: otherwise tag every task alike.
_INTENT_HEAD = 300
_ACTIVITY_HEAD = 1500

#: Activity tags in display order; a task gets every one whose pattern matches.
_ACTIVITY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "coding",
        _rx(
            r"\b(implement|refactor|function|class|module|fix|code|script|feature|"
            r"api|endpoint|compile|typecheck|webview|extension|migration|schema|"
            r"column|table|wire|wiring)\b"
        ),
    ),
    ("testing", _rx(r"\b(tests?|testing|pytest|coverage|adversarial)\b")),
    (
        "debugging",
        _rx(
            r"\b(debug|bugs?|errors?|crash(?:es)?|traceback|exceptions?|fails?|failing|"
            r"broken|not working|regression|flaky|race condition|deadlock)\b"
        ),
    ),
    ("review", _rx(r"\b(review|audit|verify|inspect|critique|check whether)\b")),
    (
        "research",
        _rx(
            r"\b(research|survey|literature|state of the art|sota|investigate|explore|"
            r"compare|find out|look up|search the (?:web|internet)|discover)\b"
        ),
    ),
    (
        "paper",
        _rx(r"\b(papers?|latex|manuscript|arxiv|iclr|neurips|icml|acl|abstract)\b|\.tex\b"),
    ),
    (
        "writing",
        _rx(
            r"\b(write (?:a |the |an )?(?:report|blog|essay|email|post|article|summary)|"
            r"draft|rewrite|proofread|blog post|article|slides|presentation)\b"
        ),
    ),
    ("docs", _rx(r"\b(readme|documentation|docstrings?|guide|tutorial|changelog)\b")),
    (
        "data",
        _rx(
            r"\b(datasets?|csv|dataframe|pandas|plot|chart|statistics|sqlite|database|"
            r"\.db|sql|query)\b"
        ),
    ),
    (
        "devops",
        _rx(
            r"\b(deploy|docker|kubernetes|github actions|ci|release|publish|daemon|"
            r"systemd|nginx|ssh|install)\b"
        ),
    ),
    (
        "messaging",
        _rx(
            r"\b(slack|discord|telegram|whatsapp|imessage|signal|matrix|sms|"
            r"email|gmail|notify|ntfy)\b"
        ),
    ),
    ("browsing", _rx(r"\b(browse|website|url|https?://|go to|web ?page|scrape|download)\b")),
    ("shopping", _rx(r"\b(buy|purchase|order (?:a|the|some)|shop|amazon|cart|prices?)\b")),
    (
        "finance",
        _rx(r"\b(invoice|bank|transfer money|stocks?|tax(?:es)?|expenses?|receipts?|payroll)\b"),
    ),
    (
        "scheduling",
        _rx(r"\b(cron|schedule|every (?:day|hour|minute|week)|remind(?:er)?|calendar|meeting)\b"),
    ),
)

#: Every tag :func:`classify_task_tags` can emit, in the order it emits
#: them.  The history panel's tag filter offers exactly this list.
ALL_TAGS: tuple[str, ...] = (
    "work",
    "personal",
    "secret",
    "chore",
    "question",
    *(tag for tag, _ in _ACTIVITY_RULES),
    "subagent",
    "failed",
)


def classify_task_tags(task: str, *, is_subagent: bool = False, failed: bool = False) -> list[str]:
    """Classify a task's text into tags.

    The first tag is always ``personal`` or ``work`` (``work`` is the
    default: a task with no personal-life signal is treated as work).
    ``secret`` marks tasks that handle credentials or private data,
    ``chore`` routine maintenance and status queries, ``question`` a
    request for information rather than an action; the activity tags
    (``coding``, ``testing``, ``debugging``, ``review``, ``research``,
    ``paper``, ``writing``, ``docs``, ``data``, ``devops``, ``messaging``,
    ``browsing``, ``shopping``, ``finance``, ``scheduling``) follow.
    ``subagent`` and ``failed`` record how the task ran.  At most
    :data:`MAX_TAGS` tags are returned, in this order.

    Args:
        task: The task text as typed or dispatched.
        is_subagent: Whether the task was a sub-agent of another task.
        failed: Whether the task's result records a failure.

    Returns:
        The tags, most significant first; never empty.
    """
    text = _FILLER.sub("", task.strip(), count=1)
    intent = text[:_INTENT_HEAD]
    head = text[:_ACTIVITY_HEAD]
    tags = ["personal" if _PERSONAL.search(intent) else "work"]
    if _SECRET.search(intent):
        tags.append("secret")
    if _CHORE.search(intent):
        tags.append("chore")
    if _QUESTION.search(intent):
        tags.append("question")
    for tag, pattern in _ACTIVITY_RULES:
        if pattern.search(head):
            tags.append(tag)
    if is_subagent:
        tags.append("subagent")
    if failed:
        tags.append("failed")
    return tags[:MAX_TAGS]


_FILLER = _rx(
    r"^(?:(?:hi|hello|hey)[,!. ]+)?"
    r"(?:(?:can|could|would|will) you(?: please)?|please|"
    r"i(?:'d| would) like(?: you)? to|i (?:want|need)(?: you)? to|help me(?: to)?|"
    r"let'?s|kindly)\s+"
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
        chunk = _FILLER.sub("", chunk.strip(), count=1)
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


def sea_name_of_agent(agent: str, channels: list[str]) -> str:
    """Return the canonical SEA name a ``run_agent(agent=...)`` argument resolves to.

    Mirrors ``agent_dispatch._run_agent``: a path names the SEA file
    (its stem), ``cron`` the bundled ``cron_agent``, a channel name its
    ``<channel>_sea`` script, and an empty value the default
    ``dummy_sea``.  The same stem is what a live run records from its
    ``agentPath``, so backfilled and live rows agree.

    Args:
        agent: The ``agent`` argument of the tool call.
        channels: Installed channel names (``agent_dispatch.available_channels``).

    Returns:
        The SEA file stem, or ``""`` when *agent* names nothing known.
    """
    requested = agent.strip()
    if not requested:
        return "dummy_sea"
    if requested.endswith(".py") or "/" in requested or "\\" in requested:
        return Path(requested).stem
    squashed = re.sub(r"[\s\-_]+", "", requested.lower())
    if squashed == "cron":
        return "cron_agent"
    for channel in channels:
        if re.sub(r"[\s\-_]+", "", channel.lower()) == squashed:
            return f"{channel}_sea"
    return ""


def launch_ms(row: sqlite3.Row) -> int:
    """Return a ``task_history`` row's launch instant in epoch milliseconds.

    ``start_ts`` (already ms) when the daemon recorded one, else the
    row's insertion ``timestamp`` (epoch seconds) — the fallback the
    history panel applies too.
    """
    start = int(row["start_ts"] or 0)
    if start > 0:
        return start
    return int(float(row["timestamp"] or 0.0) * 1000)


def infer_subagent_seas(db: sqlite3.Connection, parent_task_id: str) -> dict[str, str]:
    """Infer the SEA of each ``run_agent`` child of *parent_task_id* from its trajectory.

    Every ``tool_call`` event named ``run_agent`` in the parent's events
    carries the dispatched ``agent`` and ``task``; the child row is the
    parent's earliest not-yet-matched sub-agent whose task text contains
    that task (channel and cron dispatches prepend a preamble).  Paths
    are ignored in the comparison: ``agent_dispatch`` rewrites the
    parent repository's absolute paths to the worktree's before
    dispatch.  Every child takes part in the matching, so a prompt
    dispatched twice to different SEAs pairs up in order, but only
    children whose ``sea`` is still empty are returned.  Children
    reached by ``run_parallel`` have no such call and are not returned.

    Args:
        db: Open connection to ``sorcar.db``.
        parent_task_id: The parent row id.

    Returns:
        Mapping of child task id (``sea`` still empty) to SEA name.
    """
    from kiss.agents.sorcar.agent_dispatch import available_channels

    calls = db.execute(
        "SELECT event_json FROM events WHERE task_id = ? "
        "AND event_json LIKE '%\"run_agent\"%' ORDER BY seq",
        (parent_task_id,),
    ).fetchall()
    if not calls:
        return {}
    children = db.execute(
        "SELECT id, task, sea FROM task_history WHERE parent_task_id = ? ORDER BY timestamp, rowid",
        (parent_task_id,),
    ).fetchall()
    if not children:
        return {}
    channels = available_channels()
    child_keys = [(row[0], _dispatch_key(row[1] or ""), row[2] or "") for row in children]
    matched: set[str] = set()
    inferred: dict[str, str] = {}
    for (event_json,) in calls:
        try:
            event = json.loads(event_json)
        except ValueError:
            continue
        if event.get("type") != "tool_call" or event.get("name") != "run_agent":
            continue
        extras = event.get("extras") or {}
        key = _dispatch_key(str(extras.get("task") or ""))
        sea = sea_name_of_agent(str(extras.get("agent") or ""), channels)
        if not key or not sea:
            continue
        for child_id, child_key, child_sea in child_keys:
            if child_id not in matched and key in child_key:
                matched.add(child_id)
                if not child_sea:
                    inferred[child_id] = sea
                break
    return inferred


_PATH_TOKEN = re.compile(r"\S*[/\\]\S*")


def _dispatch_key(text: str) -> str:
    """Return *text* without path-like tokens and with whitespace collapsed."""
    return " ".join(_PATH_TOKEN.sub("", text).split())


def backfill_task_metadata(db: sqlite3.Connection) -> dict[str, int]:
    """Fill ``tags``, ``sea`` and ``chat_summaries`` for every row that lacks them.

    Tags are computed for every row whose ``tags`` is empty, the SEA for
    every sub-agent row whose parent's trajectory shows the dispatch,
    and a ``chat_summaries`` row is written for every chat that has
    none.  Writes are committed in short batches so a live daemon
    sharing the database is never locked out for long.

    Args:
        db: Open connection to ``sorcar.db`` with the current schema.

    Returns:
        Counts: ``{"tags": n, "sea": n, "chats": n}``.
    """
    from kiss.agents.sorcar.persistence import _is_failed_result

    db.row_factory = sqlite3.Row
    counts = {"tags": 0, "sea": 0, "chats": 0}
    rows = db.execute(
        "SELECT id, task, result, parent_task_id FROM task_history WHERE tags IS NULL OR tags = ''"
    ).fetchall()
    for start in range(0, len(rows), _BACKFILL_BATCH):
        with db:
            for row in rows[start : start + _BACKFILL_BATCH]:
                tags = classify_task_tags(
                    row["task"] or "",
                    is_subagent=bool(row["parent_task_id"]),
                    failed=_is_failed_result(row["result"] or ""),
                )
                db.execute(
                    "UPDATE task_history SET tags = ? WHERE id = ?",
                    (",".join(tags), row["id"]),
                )
    counts["tags"] = len(rows)
    parents = db.execute(
        "SELECT DISTINCT parent_task_id FROM task_history "
        "WHERE parent_task_id IS NOT NULL AND parent_task_id != '' "
        "AND (sea IS NULL OR sea = '')"
    ).fetchall()
    for (parent_id,) in parents:
        inferred = infer_subagent_seas(db, parent_id)
        with db:
            for child_id, sea in inferred.items():
                db.execute("UPDATE task_history SET sea = ? WHERE id = ?", (sea, child_id))
        counts["sea"] += len(inferred)
    chats = db.execute(
        "SELECT DISTINCT chat_id FROM task_history t "
        "WHERE (parent_task_id IS NULL OR parent_task_id = '') AND chat_id != '' "
        "AND NOT EXISTS (SELECT 1 FROM chat_summaries c WHERE c.chat_id = t.chat_id)"
    ).fetchall()
    for start in range(0, len(chats), _BACKFILL_BATCH):
        with db:
            for (chat_id,) in chats[start : start + _BACKFILL_BATCH]:
                upsert_chat_summary(db, chat_id)
    counts["chats"] = len(chats)
    return counts


def upsert_chat_summary(db: sqlite3.Connection, chat_id: str) -> None:
    """Recompute and store the ``chat_summaries`` row of *chat_id*.

    The summary is built from the chat's oldest listable tasks
    (sub-agent rows excluded, as in the history panel) and
    ``last_launched`` is the launch instant of its newest listable task.
    A chat with no listable task gets no row.

    Args:
        db: Open connection inside the caller's transaction.
        chat_id: The chat session id.
    """
    listable = "chat_id = ? AND (parent_task_id IS NULL OR parent_task_id = '')"
    first = db.execute(
        f"SELECT task FROM task_history WHERE {listable} ORDER BY timestamp ASC, rowid ASC LIMIT 5",
        (chat_id,),
    ).fetchall()
    if not first:
        return
    last = db.execute(
        f"SELECT timestamp, start_ts FROM task_history WHERE {listable} "
        "ORDER BY timestamp DESC, rowid DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    db.execute(
        "INSERT INTO chat_summaries (chat_id, summary, last_launched) VALUES (?, ?, ?) "
        "ON CONFLICT(chat_id) DO UPDATE SET summary = excluded.summary, "
        "last_launched = excluded.last_launched",
        (chat_id, summarize_chat([r[0] or "" for r in first]), launch_ms(last)),
    )
