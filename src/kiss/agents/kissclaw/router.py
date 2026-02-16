"""Message formatting and outbound routing for KISSClaw."""

from __future__ import annotations

import re
from xml.sax.saxutils import escape as xml_escape

from kiss.agents.kissclaw.types import Message


def escape_xml(s: str) -> str:
    if not s:
        return ""
    return xml_escape(s, {'"': "&quot;"})


def format_messages(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        lines.append(
            f'  <message sender="{escape_xml(m.sender_name)}" '
            f'timestamp="{escape_xml(m.timestamp)}">'
            f"{escape_xml(m.content)}</message>"
        )
    return "<messages>\n" + "\n".join(lines) + "\n</messages>"


def strip_internal_tags(text: str) -> str:
    return re.sub(r"<internal>[\s\S]*?</internal>", "", text).strip()


def format_outbound(raw_text: str) -> str:
    text = strip_internal_tags(raw_text)
    return text if text else ""
