"""Tests for Codex native SSE parsing behavior."""

from __future__ import annotations

import json

from kiss.core.models.codex_native_model import CodexNativeModel


class _DummyOAuthManager:
    def get_access_token(self) -> str:
        return "token"

    def get_account_id(self) -> str | None:
        return None

    def force_refresh_access_token(self) -> str | None:
        return None


def _evt(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n".encode()


def test_parse_sse_early_exits_on_tool_json_done_event() -> None:
    model = CodexNativeModel(
        "gpt-5.3-codex-spark",
        model_config={"early_tool_exit": True},
        oauth_manager=_DummyOAuthManager(),
    )
    lines = [
        _evt(
            {
                "type": "response.output_text.done",
                "text": (
                    '{"tool_calls":[{"name":"finish","arguments":{"success":true,"summary":"ok"}}]}'
                ),
            }
        ),
        _evt({"type": "response.completed", "response": {"status": "completed"}}),
    ]
    parsed = model._parse_sse_response(lines)
    assert parsed["output_text"].startswith('{"tool_calls"')
    assert parsed["response"] == {}


def test_default_timeout_tuned_for_spark() -> None:
    spark = CodexNativeModel(
        "gpt-5.3-codex-spark",
        oauth_manager=_DummyOAuthManager(),
    )
    regular = CodexNativeModel(
        "gpt-5.3-codex",
        oauth_manager=_DummyOAuthManager(),
    )
    assert spark._default_timeout_seconds() == 20.0
    assert regular._default_timeout_seconds() == 90.0
