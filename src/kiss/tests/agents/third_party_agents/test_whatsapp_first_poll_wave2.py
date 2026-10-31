# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""WhatsApp's first poll at the channel runner's starting cursor ``"0"``.

``ChannelRunner`` polls a fresh channel with cursor ``"0"``, and
``WhatsAppChannelBackend.poll_messages`` documents ``""`` and ``"0"`` alike
as "no cursor": the newest *limit* rows of the bridge database, oldest
first.  Read as a timestamp cursor, ``"0"`` would instead hand the runner
the oldest rows of the chat history on its first tick.  The test runs the
real backend on a bridge-schema SQLite database with more rows than the
limit, so the two readings return different rows.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.third_party_agents.whatsapp.whatsapp_sea import WhatsAppChannelBackend
from kiss.tests.agents.third_party_agents.whatsapp_bridge import make_messages_db

_CHAT = "14155550001@s.whatsapp.net"


def test_runner_start_cursor_polls_the_newest_rows(tmp_path: Path) -> None:
    """Cursor ``"0"`` returns the newest rows oldest-first, exactly like ``""``."""
    rows = [
        (f"m{i}", _CHAT, _CHAT, f"text {i}", f"2026-09-10 10:00:0{i}+00:00", 0, "")
        for i in (1, 2, 3)
    ]
    make_messages_db(tmp_path, [(_CHAT, "Alice", "2026-09-10 10:00:03+00:00")], rows)
    backend = WhatsAppChannelBackend(repo_dir=str(tmp_path))
    messages, cursor = backend.poll_messages(_CHAT, "0", limit=2)
    assert [m["id"] for m in messages] == ["m2", "m3"]
    assert cursor == "2026-09-10 10:00:03+00:00"
    assert backend.poll_messages(_CHAT, "", limit=2) == (messages, cursor)
