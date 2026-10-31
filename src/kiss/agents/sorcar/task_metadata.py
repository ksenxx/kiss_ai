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
* ``task_history.sea`` — the file stem of the SEA that ran
  the task, from the run's ``seaPath``; backfilled for old rows from the
  parent trajectory's ``run_agent`` tool calls (:func:`sea_name_of_agent`).
* ``chat_summaries`` — per chat a 6-8 word summary and the launch instant
  of the chat's latest task; the code lives in
  :mod:`kiss.agents.sorcar.chat_summary` (stdlib-only, so ``sync_db`` can
  ship it to a remote machine and rebuild the rows there after a merge)
  and the backfill below calls it for chats that have no row yet.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

from kiss.agents.sorcar.chat_summary import (
    NOT_SUBAGENT,
    POLITENESS_FILLER,
    upsert_chat_summary,
)

#: Upper bound on the number of tags stored per task: the history panel
#: shows them inline after the "... ago" launch-age label.  The scope
#: tag, ``secret`` and ``failed`` are never dropped by the cap; the
#: optional tags (``chore``, ``question``, the activities, ``subagent``)
#: fill the remaining slots in display order.
MAX_TAGS = 3

#: Rows written per transaction by :func:`backfill_task_metadata`.
_BACKFILL_BATCH = 500


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_PERSONAL = _rx(
    r"\b(grocer(?:y|ies)|recipes?|cook(?:ing)?|dinner|lunch|breakfast|restaurants?|"
    r"travel|flights?|hotels?|vacation|holidays?|trip|doctor|dentist|birthday|gifts?|"
    r"my (?:wife|husband|kids?|children|son|daughter|mom|dad|mother|father|family|"
    r"friends?|house|apartment|health)|movies?|netflix|playlist|workout|fitness|"
    r"gym|diet|wedding|party|amazon|shopping|personal|car|shops?|stores?|near me|"
    r"within \d+ miles)\b"
)
_SECRET = _rx(
    r"\b(passwords?|passwd|api[ _-]?keys?|secrets?|credentials?|access tokens?|"
    r"private keys?|ssh keys?|bank account|social security|ssn|credit card|otp|2fa|"
    r"confidential)\b|\.env\b"
)
#: The secret scanner works on UTF-8 bytes: ``bytes.lower()`` folds
#: ASCII only and keeps every offset, so a match found on the
#: lower-cased copy can be read back from the original bytes.  Every
#: pattern starts with a literal (fast :mod:`re` prefix scan), has no
#: unbounded lookahead and uses possessive quantifiers, so a scan is
#: linear even on adversarial input (``"sk-" * 50_000``).  Group 1 of
#: each pattern is the candidate value.
#:
#: Well-known key formats, matched case-sensitively on the original
#: bytes: private-key blocks, OpenAI/Anthropic ``sk-`` keys, GitHub,
#: AWS, Slack and Google tokens, JWTs.
_SECRET_FORMATS: tuple[re.Pattern[bytes], ...] = tuple(
    re.compile(p)
    for p in (
        rb"(-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----)",
        rb"(sk-[A-Za-z0-9_-]{40,}+)",
        rb"(gh[pousr]_[A-Za-z0-9]{36,}+)",
        rb"(AKIA[0-9A-Z]{16})(?![0-9A-Za-z])",
        rb"(xox[abprs]-\d{10,}+-[A-Za-z0-9-]{10,}+)",
        rb"(AIza[0-9A-Za-z_-]{35})",
        rb"(eyJ[A-Za-z0-9_-]{10,}+\.eyJ[A-Za-z0-9_-]{10,}+\.[A-Za-z0-9_-]{10,}+)",
    )
)
#: Free-form credentials, matched on the lower-cased bytes: ``Bearer``
#: headers, ``scheme://user:password@`` URLs and ``password = ...`` /
#: ``"token": "..."`` / ``API_KEY=...`` assignments (a JSON key's closing
#: quote and the ``\"`` escaping of ``events.event_json`` are tolerated).
#: An assignment's value is either quoted (8+ characters, spaces
#: allowed) or bare (7+ characters of a credential alphabet, not
#: followed by ``(``, ``.`` or ``[`` so that ``token = getToken()`` and
#: ``password = os.environ[...]`` do not count).
_SECRET_ASSIGNMENTS: tuple[re.Pattern[bytes], ...] = (
    re.compile(rb"bearer ([A-Za-z0-9._~+/=-]{20,}+)"),
    re.compile(rb"://[A-Za-z0-9._%+-]{2,}+:([^\s/@\"'{}<>\\]{6,}+)@"),
    *(
        re.compile(
            key + rb"\\?[\"']?\s*[:=]\s*+"
            rb"(?:\\?[\"']([^\"'\\\n]{8,}+)\\?[\"']|([A-Za-z0-9_@#$%^&*!?+/=-]{7,}+)(?![(.\[]))"
        )
        for key in (rb"password", rb"passwd", rb"secret", rb"token", rb"api[_-]?key")
    ),
)
#: A free-form value that is clearly not a credential (searched in the
#: lower-cased value): a template (``<tok>``, ``${TOKEN}``, ``&lt;tok&gt;``),
#: a redaction (``****``, ``xxxx``, ``...``), code or a shell command
#: (brackets, ``;``), or a delimited word naming a fixture or the key
#: itself (``fc-test-key``, ``race-password``, ``pypi-your-token-here``).
_PLACEHOLDER = re.compile(
    rb"[<>$*{}\[\];]|&[lg]t;|x{4,}|\.\.\.|change[_-]?me|"
    rb"(?:^|[^a-z0-9])(?:test|fake|mock|sample|demo|dummy|invalid|example|placeholder|"
    rb"redacted|your|here|none|null|true|false|required|string|pass(?:word)?|secret|"
    rb"token|key|api)(?![a-z0-9])"
)
#: A free-form value whose segments (split on ``-``, ``_``, ``.``, ``/``,
#: ``=`` and space) are each a plain word or a plain number is a name, a
#: slug, a version, a date, a model id or a phrase (``get_token``,
#: ``Cloudflare``, ``gemini-2.0-flash``, ``2026-09-16``,
#: ``openrouter/anthropic/claude-opus-4-7``, ``not path safe``), not a
#: credential; real ones mix letters and digits inside a segment or
#: carry other symbols; a short letters-digits segment (``r1``, ``qwen3``,
#: ``gpt5``, ``480b``, ``fp8``) is a version mark, not a mix.  (A pass-phrase of
#: plain words is missed on purpose: prose after ``token:`` is far more
#: common in a transcript.)
_SLUG_SEGMENT = rb"(?:[a-z]+|\d+|[a-z]{0,5}\d{1,4}[a-z]{0,2})"
_SLUG = re.compile(
    rb"[-_. /=]*" + _SLUG_SEGMENT + rb"(?:[-_. /=]+" + _SLUG_SEGMENT + rb")*[-_. /=]*"
)


def _is_credential(raw: bytes, match: re.Match[bytes], free_form: bool) -> bool:
    """Vet a candidate secret found by one of the scanner patterns.

    Args:
        raw: The original (case-preserved) bytes the match offsets refer to.
        match: A match whose last group is the candidate value.
        free_form: Whether the match is a :data:`_SECRET_ASSIGNMENTS`
            value (checked against :data:`_PLACEHOLDER` and
            :data:`_SLUG`) rather than a :data:`_SECRET_FORMATS` key.

    Returns:
        True unless the value is a placeholder or a slug, a documentation
        example (``AKIAIOSFODNN7EXAMPLE``), or an ``sk-`` key without
        both an upper-case letter and a digit (branch names such as
        ``sk-cache-review-2026-09-16``).
    """
    group = match.lastindex or 1
    value = raw[match.start(group) : match.end(group)]
    lowered = value.lower()
    if free_form:
        return not (_PLACEHOLDER.search(lowered) or _SLUG.fullmatch(lowered))
    has_digit = any(c in value for c in b"0123456789")
    if value.startswith(b"sk-") and (value == lowered or not has_digit):
        return False
    return b"example" not in lowered


def has_secret_value(texts: Iterable[str]) -> bool:
    """Return whether any of *texts* contains a credential-shaped string.

    Looks for private-key blocks, well-known API-token formats
    (OpenAI/Anthropic, GitHub, AWS, Slack, Google), JWTs, ``Bearer``
    headers, ``user:password@`` URLs and ``password=`` / ``token:`` /
    ``api_key=`` assignments with a plausible value.  Templates,
    redactions, test fixtures and names (``<your-key>``, ``***``,
    ``fc-test-key``, ``changeme``, ``get_token``, ``claude-sonnet-5``)
    are not secrets.

    Args:
        texts: Fragments of a trajectory: the task text, the JSON of
            each persisted event, the result.  Scanned lazily and in
            order; the first hit ends the scan.

    Returns:
        True when some fragment contains a secret.
    """
    for text in texts:
        raw = text.encode("utf-8", "replace")
        lowered = raw.lower()
        for patterns, haystack, free_form in (
            (_SECRET_FORMATS, raw, False),
            (_SECRET_ASSIGNMENTS, lowered, True),
        ):
            for pattern in patterns:
                if any(_is_credential(raw, m, free_form) for m in pattern.finditer(haystack)):
                    return True
    return False


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


def classify_task_tags(
    task: str, *, is_subagent: bool = False, failed: bool = False, has_secret: bool = False
) -> list[str]:
    """Classify a task's text into tags.

    The first tag is always ``personal`` or ``work`` (``work`` is the
    default: a task with no personal-life signal is treated as work).
    ``secret`` marks tasks that handle credentials or private data:
    the task text asks for them, or *has_secret* says the trajectory
    contains one (see :func:`has_secret_value`).  ``chore`` marks routine
    maintenance and status queries, ``question`` a request for
    information rather than an action; the activity tags (``coding``,
    ``testing``, ``debugging``, ``review``, ``research``, ``paper``,
    ``writing``, ``docs``, ``data``, ``devops``, ``messaging``,
    ``browsing``, ``shopping``, ``finance``, ``scheduling``) follow.
    ``subagent`` and ``failed`` record how the task ran.

    At most :data:`MAX_TAGS` tags are returned, in the order of
    :data:`ALL_TAGS`.  The scope tag, ``secret`` and ``failed`` always
    survive the cap; the other tags fill the remaining slots in that
    order, so a secret-handling or failed task shows fewer activities.

    Args:
        task: The task text as typed or dispatched.
        is_subagent: Whether the task was a sub-agent of another task.
        failed: Whether the task's result records a failure.
        has_secret: Whether the task's trajectory contains a
            credential-shaped string.

    Returns:
        The tags, most significant first; never empty.
    """
    text = POLITENESS_FILLER.sub("", task.strip(), count=1)
    intent = text[:_INTENT_HEAD]
    head = text[:_ACTIVITY_HEAD]
    tags = ["personal" if _PERSONAL.search(intent) else "work"]
    if has_secret or _SECRET.search(intent):
        tags.append("secret")
    if failed:
        tags.append("failed")
    intent_rules = (("chore", _CHORE), ("question", _QUESTION))
    optional = [tag for tag, pattern in intent_rules if pattern.search(intent)]
    optional += [tag for tag, pattern in _ACTIVITY_RULES if pattern.search(head)]
    if is_subagent:
        optional.append("subagent")
    tags += optional[: max(0, MAX_TAGS - len(tags))]
    return sorted(tags, key=ALL_TAGS.index)


def sea_name_of_agent(agent: str) -> str:
    """Return the canonical SEA name a ``run_agent(agent=...)`` argument resolves to.

    A path names the SEA file: its stem, whether or not the file still
    exists (the backfill reads dispatches whose worktree may be gone).
    Every other spelling goes through
    :func:`~kiss.agents.sorcar.agent_dispatch.resolve_agent`, the
    resolver a live dispatch uses, so backfilled and live rows agree:
    empty or a generic label is ``sorcar_sea``, ``cron`` is
    ``cron_agent``, and a registered slash command (a channel, a
    bundled SEA such as ``write_paper``, a ``SEAS.md`` folder) is its
    script's stem.

    Args:
        agent: The ``agent`` argument of the tool call.

    Returns:
        The SEA file stem, or ``""`` when *agent* names nothing known.
    """
    from kiss.agents.sorcar.agent_dispatch import is_sea_path, resolve_agent

    requested = agent.strip()
    if is_sea_path(requested):
        return Path(requested).stem
    resolved = resolve_agent(requested, "")
    return Path(resolved[0]).stem if isinstance(resolved, tuple) else ""


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
        db: Open connection to ``history.db``.
        parent_task_id: The parent row id.

    Returns:
        Mapping of child task id (``sea`` still empty) to SEA name.
    """
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
    from kiss.agents.sorcar.persistence import _safe_str

    # ``_safe_str``: a BLOB task text (a hand-edited or third-party
    # database) is decoded, not split as ``bytes``.
    child_keys = [(row[0], _dispatch_key(_safe_str(row[1])), row[2] or "") for row in children]
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
        sea = sea_name_of_agent(str(extras.get("agent") or ""))
        if not key or not sea:
            continue
        for child_id, child_key, child_sea in child_keys:
            if child_id not in matched and key in child_key:
                matched.add(child_id)
                if not child_sea:
                    inferred[child_id] = sea
                break
    return inferred


def _dispatch_key(text: str) -> str:
    """Return *text* without path-like tokens and with whitespace collapsed."""
    return " ".join(t for t in text.split() if "/" not in t and "\\" not in t)


def trajectory_has_secret(
    db: sqlite3.Connection, task_id: str, task: object, result: object
) -> bool:
    """Return whether a task's trajectory contains a credential-shaped string.

    Scans the task text, every persisted event of the task (tool
    arguments and outputs included, as stored in ``events.event_json``)
    and the result with :func:`has_secret_value`; events are streamed
    from the database and the scan stops at the first hit.

    Args:
        db: Connection to ``history.db``.
        task_id: The ``task_history`` row id.
        task: The row's ``task`` column (``None`` tolerated).
        result: The row's ``result`` column (``None`` tolerated).

    Returns:
        True when the trajectory contains a secret.
    """
    events = db.execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq, id", (task_id,)
    )
    texts = (str(ev[0] or "") for ev in events)
    return has_secret_value(chain((str(task or ""), str(result or "")), texts))


def write_task_tags(
    db: sqlite3.Connection, task_id: str, row: sqlite3.Row, has_secret: bool | None = None,
) -> None:
    """Classify a task row and store its ``tags`` column.

    The result decides the ``failed`` tag and the persisted events
    whether the trajectory holds a secret.  Runs inside the caller's
    transaction; nothing is committed here.  The secret scan reads
    every event of the task, so a caller about to take a write lock
    runs :func:`trajectory_has_secret` first and passes its verdict.

    Args:
        db: Connection to ``history.db``.
        task_id: The ``task_history`` row id.
        row: The row's ``task``, ``result``, ``success`` and
            ``parent_task_id`` columns.
        has_secret: The verdict of :func:`trajectory_has_secret` for the
            row; ``None`` scans the trajectory here.
    """
    if has_secret is None:
        has_secret = trajectory_has_secret(db, task_id, row["task"], row["result"])
    db.execute(
        "UPDATE task_history SET tags = ? WHERE id = ?", (_task_tags(row, has_secret), task_id),
    )


def _task_tags(row: sqlite3.Row, has_secret: bool) -> str:
    """Return the ``tags`` column of a task row: :func:`classify_task_tags`, comma-joined.

    Args:
        row: The row's ``task``, ``result``, ``success`` and
            ``parent_task_id`` columns.
        has_secret: The :func:`trajectory_has_secret` verdict for the row.
    """
    from kiss.agents.sorcar.persistence import _is_failed_result, _safe_str

    # ``_safe_str``: a BLOB in a TEXT column (a hand-edited or
    # third-party database) is decoded, not rendered as ``b'...'``.
    return ",".join(classify_task_tags(
        _safe_str(row["task"]),
        is_subagent=bool(row["parent_task_id"]),
        failed=_is_failed_result(_safe_str(row["result"]), row["success"]),
        has_secret=has_secret,
    ))


def _scan_for_backfill(db: sqlite3.Connection, row: sqlite3.Row) -> tuple[int, bool]:
    """Return a task row's last event id and :func:`trajectory_has_secret` verdict.

    The id is read before the scan: an event committed after it makes
    :func:`backfill_task_metadata` skip the row, so no verdict it writes
    missed an event.
    """
    (last_event,) = db.execute(
        "SELECT COALESCE(MAX(id), 0) FROM events WHERE task_id = ?", (row["id"],)
    ).fetchone()
    return last_event, trajectory_has_secret(db, row["id"], row["task"], row["result"])


def backfill_task_metadata(db: sqlite3.Connection) -> dict[str, int]:
    """Fill ``tags``, ``sea`` and ``chat_summaries`` for every row that lacks them.

    Tags are computed for every row whose ``tags`` is empty, the SEA for
    every sub-agent row whose parent's trajectory shows the dispatch,
    and a ``chat_summaries`` row is written for every chat that has
    none or whose derived ``summary`` is empty (a ``--refresh`` blanks
    it; the user's ``user_summary`` is never touched).  Writes are
    committed in short batches, and the trajectory scans run outside
    them, so a live daemon sharing the database is never locked out
    for long.  A row's tags are written only while it is still
    untagged and has no event newer than its scan: a task that
    finished meanwhile wrote its own, and one still running is tagged
    when it finishes (or by the next backfill).

    Args:
        db: Open connection to ``history.db`` with the current schema.

    Returns:
        Counts of rows written: ``{"tags": n, "sea": n, "chats": n}``.
    """
    counts = {"tags": 0, "sea": 0, "chats": 0}
    # A cursor with its own row factory: the caller's connection keeps
    # whatever factory it was opened with.
    cursor = db.cursor()
    cursor.row_factory = sqlite3.Row
    rows = cursor.execute(
        "SELECT id, task, result, parent_task_id, success FROM task_history "
        "WHERE tags IS NULL OR tags = ''"
    ).fetchall()
    for start in range(0, len(rows), _BACKFILL_BATCH):
        batch = rows[start : start + _BACKFILL_BATCH]
        # The scans read every event of a task: they run before the
        # batch's first UPDATE opens its write transaction.
        scans = [_scan_for_backfill(db, row) for row in batch]
        with db:
            for row, (last_event, has_secret) in zip(batch, scans, strict=True):
                counts["tags"] += db.execute(
                    "UPDATE task_history SET tags = ? WHERE id = ? AND (tags IS NULL OR tags = '') "
                    "AND (SELECT COALESCE(MAX(id), 0) FROM events WHERE task_id = ?) = ?",
                    (_task_tags(row, has_secret), row["id"], row["id"], last_event),
                ).rowcount
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
        f"WHERE {NOT_SUBAGENT} AND chat_id != '' "
        "AND NOT EXISTS (SELECT 1 FROM chat_summaries c "
        "WHERE c.chat_id = t.chat_id AND c.summary != '')"
    ).fetchall()
    for start in range(0, len(chats), _BACKFILL_BATCH):
        with db:
            for (chat_id,) in chats[start : start + _BACKFILL_BATCH]:
                upsert_chat_summary(db, chat_id)
    counts["chats"] = len(chats)
    return counts
