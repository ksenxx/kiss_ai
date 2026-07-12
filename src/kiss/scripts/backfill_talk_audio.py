#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Report and optionally repair silent persisted ``talk`` tool_call events.

Demo-mode replay plays exactly ``extras.audioB64`` of the persisted
``talk`` ``tool_call`` event — the speech-synthesis fallback is gone.
Recordings made before the clip-persistence fix therefore replay their
narration completely silently.  This one-off admin command scans the
sorcar events database for ``talk`` ``tool_call`` rows lacking
``extras.audioB64``, prints a report, and (with ``--fix``)
re-synthesizes each utterance from its recorded ``text`` / ``language``
/ ``emotion`` extras via the same GPT-audio TTS the live ``talk`` tool
uses and amends the rows in place (an UPDATE, never an append — a
duplicate row would render the panel twice on replay).

Usage:
    uv run kiss-backfill-talk-audio [OPTIONS]

Options:
    --fix            Re-synthesize audio for silent rows and update them
                     in place (calls the paid GPT-audio TTS API once per
                     utterance; use --limit to cap cost)
    --limit N        Repair at most N rows in --fix mode
    --task-id ID     Only consider rows belonging to this task_history id
    --db PATH        Operate on this SQLite database instead of the
                     active ~/.kiss/sorcar.db (KISS_HOME respected)
    --json           Emit the report as JSON instead of a table
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.vscode.speech_synthesis import synthesize_talk_audio

# Rough length cap for text previews in the human-readable report.
_PREVIEW_CHARS = 60


@dataclass
class TalkCall:
    """One persisted ``talk`` ``tool_call`` event row.

    Attributes:
        row_id: Primary key of the ``events`` row.
        task_id: ``task_history`` row id the event belongs to.
        seq: Event sequence number within the task.
        task: Task description from ``task_history`` ("" when the task
            row is gone).
        text: The spoken utterance recorded in ``extras.text``.
        language: BCP-47 language tag from ``extras.language``.
        emotion: Delivery vibe from ``extras.emotion``.
        timestamp: Event timestamp (seconds since epoch, 0.0 if absent).
        has_audio: Whether ``extras.audioB64`` is already present.
    """

    row_id: int
    task_id: str
    seq: int
    task: str
    text: str
    language: str
    emotion: str
    timestamp: float
    has_audio: bool


def _parse_talk_event(event_json: str) -> dict | None:
    """Return the parsed event dict when it is a ``talk`` ``tool_call``.

    Args:
        event_json: Raw ``event_json`` column value.

    Returns:
        The event dict, or ``None`` for corrupt JSON or events that are
        not ``talk`` tool calls (the SQL LIKE prefilter over-matches,
        e.g. a Bash command whose text mentions "talk").
    """
    try:
        event = json.loads(event_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(event, dict):
        return None
    if event.get("type") != "tool_call" or event.get("name") != "talk":
        return None
    return event


def scan_talk_calls(task_id: str = "") -> list[TalkCall]:
    """Scan the active database for persisted ``talk`` tool_call events.

    Uses the persistence layer's own lock and per-thread connection so
    the scan is safe while a daemon holds the database open.

    Args:
        task_id: When non-empty, restrict the scan to this task.

    Returns:
        All ``talk`` ``tool_call`` rows (silent and with-audio alike),
        ordered by task then sequence.
    """
    _persistence._flush_chat_events()
    query = (
        "SELECT e.id, e.task_id, e.seq, e.event_json, e.timestamp, "
        "th.task AS task FROM events e "
        "LEFT JOIN task_history th ON th.id = e.task_id "
        "WHERE e.event_json LIKE '%\"tool_call\"%' "
        "AND e.event_json LIKE '%\"talk\"%'"
    )
    params: tuple = ()
    if task_id:
        query += " AND e.task_id = ?"
        params = (task_id,)
    query += " ORDER BY e.task_id, e.seq"
    with _persistence._rw_lock.read_lock():
        db = _persistence._get_db()
        rows = db.execute(query, params).fetchall()
    calls: list[TalkCall] = []
    for row in rows:
        event = _parse_talk_event(row["event_json"])
        if event is None:
            continue
        extras = event.get("extras") or {}
        calls.append(
            TalkCall(
                row_id=row["id"],
                task_id=str(row["task_id"]),
                seq=row["seq"] or 0,
                task=str(row["task"] or ""),
                text=str(extras.get("text") or ""),
                language=str(extras.get("language") or ""),
                emotion=str(extras.get("emotion") or ""),
                timestamp=float(row["timestamp"] or 0.0),
                has_audio=bool(extras.get("audioB64")),
            ),
        )
    return calls


def _set_row_audio(row_id: int, audio_b64: str, audio_mime: str) -> bool:
    """Attach a clip to one ``events`` row in place.

    Re-reads the row under the write lock (its JSON may have changed
    since the scan) and rewrites ``extras`` with the clip.  An UPDATE,
    never an append — a duplicate row would render the panel twice on
    demo replay.

    Args:
        row_id: Primary key of the ``events`` row to amend.
        audio_b64: Base64-encoded synthesized clip bytes.
        audio_mime: The clip's MIME type (e.g. ``"audio/mpeg"``).

    Returns:
        ``True`` when the row was amended, ``False`` when it no longer
        exists, is not an audio-less ``talk`` tool call, or the clip is
        empty.
    """
    if not audio_b64:
        return False
    _persistence._flush_chat_events()
    with _persistence._rw_lock.write_lock():
        db = _persistence._get_db()
        row = db.execute(
            "SELECT event_json FROM events WHERE id = ?", (row_id,),
        ).fetchone()
        if row is None:
            return False
        event = _parse_talk_event(row["event_json"])
        if event is None:
            return False
        extras = event.get("extras")
        if not isinstance(extras, dict):
            extras = {}
            event["extras"] = extras
        if extras.get("audioB64"):
            return False
        extras["audioB64"] = audio_b64
        extras["audioMime"] = audio_mime
        db.execute(
            "UPDATE events SET event_json = ? WHERE id = ?",
            (json.dumps(event), row_id),
        )
        db.commit()
    return True


def repair_silent_talk_calls(
    calls: list[TalkCall],
    limit: int = 0,
    synth: Callable[..., tuple[str, str] | None] = synthesize_talk_audio,
) -> tuple[int, int, int]:
    """Re-synthesize audio for silent talk calls and amend their rows.

    Args:
        calls: Rows from :func:`scan_talk_calls`; rows that already
            carry audio or have empty text are skipped.
        limit: Maximum number of rows to repair (0 = no limit).
        synth: Synthesizer ``(text, language, emotion) -> (b64, mime)
            | None``; defaults to the live GPT-audio TTS.

    Returns:
        ``(repaired, failed, skipped_empty)`` counts.
    """
    repaired = failed = skipped_empty = 0
    for call in calls:
        if call.has_audio:
            continue
        if limit and repaired >= limit:
            break
        if not call.text.strip():
            skipped_empty += 1
            print(
                f"  SKIP row {call.row_id} (task {call.task_id} seq "
                f"{call.seq}): empty text",
            )
            continue
        preview = _preview(call.text)
        result = synth(call.text, call.language, call.emotion)
        if not result:
            failed += 1
            print(
                f"  FAIL row {call.row_id} (task {call.task_id} seq "
                f"{call.seq}): synthesis failed — {preview!r}",
            )
            continue
        audio_b64, audio_mime = result
        if _set_row_audio(call.row_id, audio_b64, audio_mime):
            repaired += 1
            print(
                f"  OK   row {call.row_id} (task {call.task_id} seq "
                f"{call.seq}): {preview!r}",
            )
        else:
            failed += 1
            print(
                f"  FAIL row {call.row_id} (task {call.task_id} seq "
                f"{call.seq}): row changed or vanished — {preview!r}",
            )
    return repaired, failed, skipped_empty


def _preview(text: str) -> str:
    """Return *text* squeezed onto one line and capped for the report."""
    one_line = " ".join(text.split())
    if len(one_line) > _PREVIEW_CHARS:
        return one_line[: _PREVIEW_CHARS - 1] + "…"
    return one_line


def print_report(calls: list[TalkCall], as_json: bool = False) -> None:
    """Print the scan results.

    Args:
        calls: Rows from :func:`scan_talk_calls`.
        as_json: Emit machine-readable JSON instead of a table.
    """
    silent = [c for c in calls if not c.has_audio]
    if as_json:
        payload = {
            "total_talk_calls": len(calls),
            "with_audio": len(calls) - len(silent),
            "silent": len(silent),
            "distinct_tasks_with_silent": len({c.task_id for c in silent}),
            "silent_calls": [asdict(c) for c in silent],
        }
        print(json.dumps(payload, indent=2))
        return
    if silent:
        header = (
            f"{'row':>7} {'task_id':>8} {'seq':>5} {'lang':<7} "
            f"{'task':<30} text"
        )
        print(header)
        print("-" * len(header))
        for c in silent:
            print(
                f"{c.row_id:>7} {c.task_id:>8} {c.seq:>5} "
                f"{c.language:<7} {_preview(c.task)[:30]:<30} "
                f"{_preview(c.text)}",
            )
        print()
    print(f"Total persisted talk tool_call events: {len(calls)}")
    print(f"  with audio (replay sounds):          {len(calls) - len(silent)}")
    print(f"  SILENT (no extras.audioB64):         {len(silent)}")
    print(
        f"  distinct tasks with silent calls:    "
        f"{len({c.task_id for c in silent})}",
    )
    if silent:
        print(
            "\nRun with --fix to re-synthesize audio for these rows "
            "(paid TTS call per utterance; --limit N caps cost).",
        )


def _apply_db_override(db_path: str) -> None:
    """Point the persistence layer at *db_path* for this process.

    Mirrors the redirection test fixtures use: reassigning ``_DB_PATH``
    (and clearing the cached connection) makes ``_get_db`` open the
    override file, so scans and fixes go through the same lock and
    connection machinery as the default path.

    Args:
        db_path: Path to an existing sorcar SQLite database file.
    """
    from pathlib import Path

    path = Path(db_path)
    if not path.exists():
        raise SystemExit(f"error: database not found: {db_path}")
    _persistence._KISS_DIR = path.parent
    _persistence._DB_PATH = path
    _persistence._db_conn = None


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``kiss-backfill-talk-audio`` admin command.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 when --fix left failures.
    """
    parser = argparse.ArgumentParser(
        prog="kiss-backfill-talk-audio",
        description=(
            "Report (default) or repair (--fix) persisted talk tool_call "
            "events that lack the audioB64 clip demo replay needs."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="re-synthesize audio via TTS and update silent rows in place",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="repair at most N rows in --fix mode (0 = no limit)",
    )
    parser.add_argument(
        "--task-id",
        default="",
        metavar="ID",
        help="only consider rows belonging to this task_history id",
    )
    parser.add_argument(
        "--db",
        default="",
        metavar="PATH",
        help="operate on this database instead of ~/.kiss/sorcar.db",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON",
    )
    args = parser.parse_args(argv)
    if args.db:
        _apply_db_override(args.db)
    calls = scan_talk_calls(task_id=args.task_id)
    print_report(calls, as_json=args.json)
    if not args.fix:
        return 0
    silent = [c for c in calls if not c.has_audio]
    if not silent:
        print("\nNothing to fix.")
        return 0
    print(f"\nRepairing {len(silent)} silent talk call(s)...")
    repaired, failed, skipped_empty = repair_silent_talk_calls(
        silent, limit=args.limit,
    )
    print(
        f"\nRepaired: {repaired}, failed: {failed}, "
        f"skipped (empty text): {skipped_empty}",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
