# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Local stand-in for the whatsapp-mcp bridge REST API (POST-only ``/api/send``).

``with bridge_server({"success": True, "message": "sent"}) as server:`` runs
the fake on an ephemeral port; ``server.server_address[1]`` is the port to
pass as ``bridge_port``.  Every POST is recorded in ``server.requests``
(see ``JsonWebhookServer``); every other verb answers 405 exactly like the
Go bridge's mux, so a tool that stops using POST fails its test.

:func:`make_messages_db` writes the bridge's ``store/messages.db`` with
its real schema and the given rows.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from kiss.tests.agents.third_party_agents.recording_http import (
    JsonWebhookHandler,
    JsonWebhookServer,
    recording_server,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    jid TEXT PRIMARY KEY,
    name TEXT,
    last_message_time TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT,
    chat_jid TEXT,
    sender TEXT,
    content TEXT,
    timestamp TIMESTAMP,
    is_from_me BOOLEAN,
    media_type TEXT,
    filename TEXT,
    url TEXT,
    media_key BLOB,
    file_sha256 BLOB,
    file_enc_sha256 BLOB,
    file_length INTEGER,
    PRIMARY KEY (id, chat_jid),
    FOREIGN KEY (chat_jid) REFERENCES chats(jid)
);
"""


def make_messages_db(
    repo_dir: Path, chats: list[tuple[Any, ...]], messages: list[tuple[Any, ...]]
) -> Path:
    """Create ``<repo_dir>/whatsapp-bridge/store/messages.db`` with the bridge's schema.

    Args:
        repo_dir: The configured whatsapp-mcp checkout directory.
        chats: ``(jid, name, last_message_time)`` rows.
        messages: ``(id, chat_jid, sender, content, timestamp, is_from_me,
            media_type)`` rows.

    Returns:
        The database path.
    """
    store = repo_dir / "whatsapp-bridge" / "store"
    store.mkdir(parents=True, exist_ok=True)
    db = store / "messages.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    conn.executemany("INSERT INTO chats (jid, name, last_message_time) VALUES (?,?,?)", chats)
    conn.executemany(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me,"
        " media_type) VALUES (?,?,?,?,?,?,?)",
        messages,
    )
    conn.commit()
    conn.close()
    return db


class _BridgeHandler(JsonWebhookHandler):
    def do_GET(self) -> None:  # noqa: N802
        """Refuse every verb but POST like the Go bridge's mux."""
        self._reply(405, "Method not allowed\n", "text/plain")

    do_PUT = do_PATCH = do_DELETE = do_GET  # noqa: N815


@contextmanager
def bridge_server(response_body: dict[str, Any]) -> Iterator[JsonWebhookServer]:
    """Run a bridge stand-in answering *response_body* for the block."""
    with recording_server(_BridgeHandler, JsonWebhookServer) as server:
        server.response_body = response_body
        yield server
