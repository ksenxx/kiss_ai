# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for identity-linked Anthropic keys (workspace id).

The Anthropic API rejects a request made with an identity-linked API key
unless it carries the ``anthropic-workspace-id`` header naming the
workspace the request acts in::

    Error code: 400 - {'type': 'error', 'error': {'type':
    'invalid_request_error', 'message': 'anthropic-workspace-id is
    required when authenticating with an identity-linked API key; ...'}}

These tests drive the real ``anthropic`` SDK against a real local HTTP
endpoint (:class:`ScriptedAnthropicServer`) — no mocks, patches or test
doubles — and verify that:

* :class:`AnthropicModel` sends the header when ``ANTHROPIC_WORKSPACE_ID``
  is set, and does not send it when the variable is absent or blank;
* the API's opaque 400 is converted into an actionable, non-retryable
  :class:`KISSError` that names ``ANTHROPIC_WORKSPACE_ID`` and the places
  it can be set;
* any other 400 is left untouched;
* the settings store round-trips the variable: ``save_api_key`` persists
  it to ``$KISS_HOME/api_keys.env`` and ``load_api_keys`` (what the
  daemon runs at startup, on local and ``./rsorcar``-deployed installs
  alike) imports it into ``os.environ`` and the config singleton.
"""

import json
import os

import anthropic
import pytest
from anthropic import BadRequestError

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import MODEL_INFO, ModelInfo
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer

_MODEL = "claude-workspace-under-test"

_WORKSPACE_400 = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": (
            "anthropic-workspace-id is required when authenticating with "
            "an identity-linked API key; send the id of the workspace "
            "this request acts in."
        ),
    },
}

_OTHER_400 = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "max_tokens: must be greater than 0",
    },
}


def _run_one_turn(server: ScriptedAnthropicServer, monkeypatch) -> None:
    """Run one real model turn against *server* through the real SDK.

    Args:
        server: The local Anthropic endpoint under test.
        monkeypatch: The pytest fixture used to point the SDK at it.
    """
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
    model = AnthropicModel(_MODEL, api_key="test-key")
    model.initialize("Say ok.")
    model.generate()


def test_workspace_id_header_sent_when_env_var_set(monkeypatch) -> None:
    """ANTHROPIC_WORKSPACE_ID is sent as the anthropic-workspace-id header."""
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test123")
    with ScriptedAnthropicServer() as server:
        _run_one_turn(server, monkeypatch)
        headers = server.request_headers[-1]
    assert headers["anthropic-workspace-id"] == "wrkspc_test123"


def test_no_workspace_header_when_env_var_absent(monkeypatch) -> None:
    """No anthropic-workspace-id header is sent when the variable is unset."""
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    with ScriptedAnthropicServer() as server:
        _run_one_turn(server, monkeypatch)
        headers = server.request_headers[-1]
    assert "anthropic-workspace-id" not in headers


def test_no_workspace_header_when_env_var_blank(monkeypatch) -> None:
    """A whitespace-only ANTHROPIC_WORKSPACE_ID is treated as unset."""
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "   ")
    with ScriptedAnthropicServer() as server:
        _run_one_turn(server, monkeypatch)
        headers = server.request_headers[-1]
    assert "anthropic-workspace-id" not in headers


def test_workspace_400_becomes_actionable_error(monkeypatch) -> None:
    """The workspace-id 400 becomes a KISSError that says what to do.

    The raised message must name the ANTHROPIC_WORKSPACE_ID variable, the
    stores it can be set in, and the ./rsorcar redeploy step.
    """
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    body = json.dumps(_WORKSPACE_400).encode()
    with ScriptedAnthropicServer(
        chunks=[body], status=400, content_type="application/json"
    ) as server:
        with pytest.raises(KISSError) as excinfo:
            _run_one_turn(server, monkeypatch)
    message = str(excinfo.value)
    assert "ANTHROPIC_WORKSPACE_ID" in message
    assert "api_keys.env" in message
    assert "./rsorcar" in message
    assert "settings panel" in message


def test_other_400_is_not_rewritten(monkeypatch) -> None:
    """A 400 unrelated to the workspace id surfaces as the SDK's own error."""
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    body = json.dumps(_OTHER_400).encode()
    with ScriptedAnthropicServer(
        chunks=[body], status=400, content_type="application/json"
    ) as server:
        with pytest.raises(BadRequestError):
            _run_one_turn(server, monkeypatch)


def test_400_naming_but_not_requiring_the_header_is_not_rewritten(
    monkeypatch,
) -> None:
    """A 400 that merely mentions the header (e.g. malformed id) is kept.

    Rewriting it as "the workspace id is missing" would mislead a user
    who HAS configured an id — the match must require the API's full
    "anthropic-workspace-id is required" phrase.
    """
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_badvalue")
    malformed = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "anthropic-workspace-id: invalid workspace id format",
        },
    }
    body = json.dumps(malformed).encode()
    with ScriptedAnthropicServer(
        chunks=[body], status=400, content_type="application/json"
    ) as server:
        with pytest.raises(BadRequestError):
            _run_one_turn(server, monkeypatch)


def test_agent_run_fails_fast_with_workspace_hint(monkeypatch) -> None:
    """A real KISSAgent run fails on the FIRST request with the hint.

    ``_run_agentic_loop``'s ``except KISSError`` branch re-raises the
    rewritten error immediately (the model has no registered fallback),
    so exactly one HTTP request reaches the API — no retry of a request
    that can never succeed.
    """
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.setitem(
        MODEL_INFO,
        _MODEL,
        ModelInfo(
            context_length=128_000,
            input_price_per_million=0.0,
            output_price_per_million=0.0,
            is_function_calling_supported=True,
            is_embedding_supported=False,
            is_generation_supported=True,
            fallback=None,
            extended_thinking=False,
        ),
    )
    from kiss.core.config import DEFAULT_CONFIG

    monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "test-key", raising=False)
    body = json.dumps(_WORKSPACE_400).encode()
    with ScriptedAnthropicServer(
        chunks=[body], status=400, content_type="application/json"
    ) as server:

        def point_client_at_server(m) -> None:
            base = str(getattr(getattr(m, "client", None), "base_url", ""))
            if isinstance(m, AnthropicModel) and server.base_url not in base:
                m.client = anthropic.Anthropic(
                    api_key="test-key", base_url=server.base_url
                )

        agent = KISSAgent("test-workspace-hint")
        agent.pre_step_hook = point_client_at_server
        with pytest.raises(KISSError) as excinfo:
            agent.run(
                model_name=_MODEL,
                prompt_template="Say ok.",
                max_steps=5,
                max_budget=1.0,
                verbose=False,
            )
        assert "ANTHROPIC_WORKSPACE_ID" in str(excinfo.value)
        assert len(server.requests) == 1


def test_workspace_id_settings_store_roundtrip(tmp_path, monkeypatch) -> None:
    """save_api_key persists the id and load_api_keys re-imports it.

    This is the exact path a remote install takes: the settings panel (or
    ``./rsorcar``) writes ``$KISS_HOME/api_keys.env``, and the daemon's
    startup ``load_api_keys()`` turns it into ``os.environ`` and config
    state — which is where :class:`AnthropicModel` reads it from.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kiss"))
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    from kiss.core.config import DEFAULT_CONFIG
    from kiss.core.vscode_config import (
        api_keys_env_path,
        load_api_keys,
        save_api_key,
    )

    # save_api_key/load_api_keys mutate the process-global config
    # singleton; registering the field with monkeypatch restores its
    # pre-test value on teardown so later tests see unchanged state.
    monkeypatch.setattr(
        DEFAULT_CONFIG,
        "ANTHROPIC_WORKSPACE_ID",
        DEFAULT_CONFIG.ANTHROPIC_WORKSPACE_ID,
        raising=False,
    )

    save_api_key("ANTHROPIC_WORKSPACE_ID", "wrkspc_roundtrip")
    stored = api_keys_env_path().read_text(encoding="utf-8")
    assert "export ANTHROPIC_WORKSPACE_ID=wrkspc_roundtrip" in stored

    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    load_api_keys()
    assert os.environ["ANTHROPIC_WORKSPACE_ID"] == "wrkspc_roundtrip"
    assert DEFAULT_CONFIG.ANTHROPIC_WORKSPACE_ID == "wrkspc_roundtrip"

    # Deleting from the settings panel unsets it everywhere.
    save_api_key("ANTHROPIC_WORKSPACE_ID", "")
    assert "ANTHROPIC_WORKSPACE_ID" not in api_keys_env_path().read_text(
        encoding="utf-8"
    )
    assert "ANTHROPIC_WORKSPACE_ID" not in os.environ
    assert DEFAULT_CONFIG.ANTHROPIC_WORKSPACE_ID == ""
