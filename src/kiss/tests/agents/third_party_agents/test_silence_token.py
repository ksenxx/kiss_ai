# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for Hermes-style silence-token reply suppression in channel agents."""

from __future__ import annotations

import yaml

from kiss.agents.third_party_agents._channel_agent_utils import (
    SILENCE_TOKENS,
    summary_for_reply,
)


def _result_yaml(summary: str) -> str:
    """Build a task-result YAML string the way the launcher does."""
    return str(yaml.safe_dump({"success": True, "summary": summary}))


def test_silence_tokens_suppress_reply() -> None:
    """A summary of exactly [SILENT] or NO_REPLY yields no reply."""
    for token in SILENCE_TOKENS:
        assert summary_for_reply(_result_yaml(token)) is None
        assert summary_for_reply(_result_yaml(f"  {token}  ")) is None


def test_html_wrapped_silence_token_suppresses_reply() -> None:
    """The daemon may HTML-wrap the summary; the token still silences."""
    assert summary_for_reply(_result_yaml("<p>[SILENT]</p>")) is None
    assert summary_for_reply(_result_yaml("<p>\nNO_REPLY\n</p>")) is None


def test_normal_summary_is_returned() -> None:
    """A normal summary is returned verbatim for the reply."""
    assert summary_for_reply(_result_yaml("All done.")) == "All done."


def test_token_embedded_in_longer_text_is_not_silenced() -> None:
    """Silence requires the summary to be exactly the token."""
    text = "The agent said [SILENT] but also did work."
    assert summary_for_reply(_result_yaml(text)) == text


def test_empty_summary_falls_back_to_raw_result() -> None:
    """An empty summary falls back to the raw result string."""
    raw = _result_yaml("")
    assert summary_for_reply(raw) == raw


def test_non_yaml_result_is_returned_verbatim() -> None:
    """A non-YAML raw string is used as the reply text."""
    assert summary_for_reply("plain text result") == "plain text result"


def test_yaml_scalar_result_falls_back_to_raw() -> None:
    """A YAML scalar (non-dict) result falls back to the raw string."""
    assert summary_for_reply("42") == "42"


def test_malformed_yaml_falls_back_to_raw() -> None:
    """Unparseable YAML falls back to the raw string instead of raising."""
    raw = "summary: [unclosed"
    assert summary_for_reply(raw) == raw
