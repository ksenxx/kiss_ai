# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for non-Claude models on Amazon Bedrock (``bedrock/<id>``).

A ``bedrock/<id>`` name whose id is not a Claude id (Kimi, GLM, Grok,
OpenAI GPT, DeepSeek, MiniMax, Qwen, ...) is routed by the ``model()``
factory to the OpenAI-compatible adapter against bedrock-runtime's
``/openai/v1`` endpoint.  These tests verify, with no mocks of the code
under test:

* the factory routing (non-Claude id -> OpenAICompatibleModel, Claude id
  stays on AnthropicModel) and the wire model name;
* the base URL follows the configured region;
* auth: a configured ``AWS_BEARER_TOKEN_BEDROCK`` is used as-is, else a
  short-term bearer token is SigV4-signed from the ambient AWS
  credentials (token signing is purely local, so fake credentials
  produce a fully checkable token), cached briefly and re-signed when
  the credentials rotate;
* a callable ``api_key`` is re-resolved whenever the OpenAI client is
  (re)built, so a rotated token forces a rebuild;
* a live generation and server-side automatic prompt caching against
  real Bedrock when credentials are available.

Run with::

    uv run pytest src/kiss/tests/core/models/test_bedrock_openai_compatible.py -v
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import sys
import threading

import pytest

from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models import model_info
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import model
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

_KIMI = "bedrock/us.moonshotai.kimi-k3"
_CLAUDE = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
_FAKE_KEY = "AKIAFAKE1234567890AB"
_FAKE_SECRET = "fakeSecretKeyfakeSecretKeyfakeSecret000"

# Captured at import (before the isolation fixture runs) so the live test
# can opt back into the developer's real AWS environment.
_AMBIENT_AWS_ENV = {k: v for k, v in os.environ.items() if k.startswith("AWS_")}


@pytest.fixture(autouse=True)
def _fresh_bedrock_state(monkeypatch):
    """Isolate each test from the process's AWS config and the token cache.

    The token cache is module state; the ambient environment may export
    real AWS variables (a developer's AWS_PROFILE, a pod's IRSA file),
    so every source boto3 could fall back to is redirected at empty
    files or unset, and tests opt into the credentials they need.
    """
    monkeypatch.setattr(model_info, "_bedrock_token_cache", None)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "AWS_BEARER_TOKEN_BEDROCK", "")
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "AWS_REGION", "")
    for var in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    yield


def _fake_env_credentials(monkeypatch, session_token: str | None = None) -> None:
    """Give the boto3 default chain a static fake key pair via the environment."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", _FAKE_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", _FAKE_SECRET)
    if session_token is not None:
        monkeypatch.setenv("AWS_SESSION_TOKEN", session_token)


def _decode_token(token: str) -> str:
    """Undo the ``bedrock-api-key-`` + base64 wrapping of a bearer token."""
    assert token.startswith("bedrock-api-key-")
    return base64.b64decode(token[len("bedrock-api-key-") :]).decode("utf-8")


def test_factory_routes_non_claude_bedrock_to_openai_compatible(monkeypatch) -> None:
    _fake_env_credentials(monkeypatch)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "AWS_REGION", "us-west-2")
    m = model(_KIMI)
    assert isinstance(m, OpenAICompatibleModel)
    assert m.base_url == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"
    assert m._api_model_name == "us.moonshotai.kimi-k3"


def test_factory_routes_single_region_model_id(monkeypatch) -> None:
    _fake_env_credentials(monkeypatch)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "AWS_REGION", "us-west-2")
    m = model("bedrock/deepseek.v3.2")
    assert isinstance(m, OpenAICompatibleModel)
    assert m._api_model_name == "deepseek.v3.2"


def test_factory_keeps_claude_on_anthropic_adapter(monkeypatch) -> None:
    _fake_env_credentials(monkeypatch)
    m = model(_CLAUDE)
    assert isinstance(m, AnthropicModel)


def test_base_url_follows_configured_region(monkeypatch) -> None:
    _fake_env_credentials(monkeypatch)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "AWS_REGION", "eu-west-1")
    m = model(_KIMI)
    assert isinstance(m, OpenAICompatibleModel)
    assert m.base_url == "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1"


def test_configured_bearer_token_is_used_verbatim(monkeypatch) -> None:
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG, "AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-configured"
    )
    m = model(_KIMI)
    assert isinstance(m, OpenAICompatibleModel)
    m._ensure_client()
    assert m.client.api_key == "bedrock-api-key-configured"


def test_signed_token_shape_and_cache(monkeypatch) -> None:
    _fake_env_credentials(monkeypatch)
    token1 = model_info._bedrock_bearer_token("us-west-2")
    raw = _decode_token(token1)
    assert raw.startswith("bedrock.amazonaws.com/?Action=CallWithBearerToken&")
    assert f"X-Amz-Credential={_FAKE_KEY}" in raw
    assert raw.endswith("&Version=1")
    # Same credentials inside the cache window: identical token, no re-sign.
    assert model_info._bedrock_bearer_token("us-west-2") == token1
    # A different region is a different signature.
    assert model_info._bedrock_bearer_token("eu-west-1") != token1


def test_token_resigned_when_credentials_rotate(monkeypatch) -> None:
    _fake_env_credentials(monkeypatch, session_token="session-one")
    token1 = model_info._bedrock_bearer_token("us-west-2")
    _fake_env_credentials(monkeypatch, session_token="session-two")
    token2 = model_info._bedrock_bearer_token("us-west-2")
    assert token1 != token2


def test_token_cache_expiry_triggers_resign(monkeypatch) -> None:
    _fake_env_credentials(monkeypatch)
    token1 = model_info._bedrock_bearer_token("us-west-2")
    assert model_info._bedrock_token_cache is not None
    key, _, _ = model_info._bedrock_token_cache
    monkeypatch.setattr(
        model_info,
        "_bedrock_token_cache",
        (key, 0.0, token1),  # signed in 1970: outside the cache window
    )
    model_info._bedrock_bearer_token("us-west-2")
    # The stale entry was replaced by a fresh signature (a same-second
    # re-sign yields an identical token, so the timestamp is the proof).
    assert model_info._bedrock_token_cache is not None
    assert model_info._bedrock_token_cache[1] > 0.0


def test_missing_credentials_raise_kiss_error(monkeypatch) -> None:
    with pytest.raises(KISSError, match="No AWS credentials"):
        model_info._bedrock_bearer_token("us-west-2")


def test_missing_boto3_raises_kiss_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)  # import boto3 -> ImportError
    with pytest.raises(KISSError, match="boto3 is required"):
        model_info._bedrock_bearer_token("us-west-2")


def test_callable_api_key_rebuilds_client_on_rotation() -> None:
    current = ["key-one"]
    m = OpenAICompatibleModel(
        "kimi-k3",
        base_url="http://localhost:9/v1",
        api_key=lambda: current[0],
    )
    m._ensure_client()
    first_client = m.client
    assert first_client.api_key == "key-one"
    m._ensure_client()  # unchanged key: the client is reused
    assert m.client is first_client
    current[0] = "key-two"
    m._ensure_client()  # rotated key: the client is rebuilt
    assert m.client is not first_client
    assert m.client.api_key == "key-two"


class _RecordingChatHandler(http.server.BaseHTTPRequestHandler):
    """Answers every chat completion with "ok" and records its bearer key."""

    seen_keys: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        self.seen_keys.append(self.headers["Authorization"])
        body = json.dumps(
            {
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def test_rotated_key_is_sent_on_the_next_turn_of_a_running_conversation() -> None:
    server = http.server.HTTPServer(("127.0.0.1", 0), _RecordingChatHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    current = ["key-one"]
    try:
        m = OpenAICompatibleModel(
            "kimi-k3",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key=lambda: current[0],
        )
        m.initialize("hi")
        assert m.generate()[0] == "ok"
        current[0] = "key-two"
        m.add_message_to_conversation("user", "again")
        assert m.generate()[0] == "ok"
    finally:
        server.shutdown()
        server.server_close()
    assert _RecordingChatHandler.seen_keys == ["Bearer key-one", "Bearer key-two"]


def test_bedrock_name_stripped_from_wire_name() -> None:
    m = OpenAICompatibleModel(
        "bedrock/us.xai.grok-4.7",
        base_url="http://localhost:9/v1",
        api_key="k",
    )
    assert m._api_model_name == "us.xai.grok-4.7"


def test_model_provider_label_stays_bedrock() -> None:
    assert model_info.get_model_provider(_KIMI) == "Bedrock"


@pytest.mark.skipif(
    not model_info.bedrock_credentials_configured(),
    reason="no AWS credentials configured",
)
def test_live_generation_and_automatic_caching(monkeypatch) -> None:
    """Real Bedrock round-trip: a repeated long prompt is served from cache.

    The OpenAI-compatible endpoint caches the prompt prefix server-side
    with no request markers; the second generation of an identical
    conversation must report cache reads in its usage.
    """
    # Opt back into the ambient AWS environment the fixture stripped.
    stripped = ("AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE", "AWS_EC2_METADATA_DISABLED")
    for var in stripped:
        monkeypatch.delenv(var, raising=False)
    for key, value in _AMBIENT_AWS_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG, "AWS_REGION", _AMBIENT_AWS_ENV.get("AWS_REGION", "")
    )
    big = " ".join(f"Fact {i}: shard {i} holds {i * 3} bytes." for i in range(400))
    m = model(_KIMI, model_config={"system_instruction": big, "max_tokens": 512})
    m.initialize("What does shard 350 hold? One line.")
    text1, _ = m.generate()
    assert text1.strip()
    m2 = model(_KIMI, model_config={"system_instruction": big, "max_tokens": 512})
    m2.initialize("What does shard 350 hold? One line.")
    text2, response2 = m2.generate()
    assert text2.strip()
    cache_read = m2.extract_input_output_token_counts_from_response(response2)[2]
    assert cache_read > 0
