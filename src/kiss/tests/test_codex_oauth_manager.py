"""Unit tests for native Codex OAuth credential management."""

from __future__ import annotations

import base64
import json
import time
import urllib.parse
from pathlib import Path

from kiss.core.models import codex_oauth


def _jwt(claims: dict[str, object]) -> str:
    def enc(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{enc({'alg': 'none', 'typ': 'JWT'})}.{enc(claims)}."


def test_manager_loads_credentials_from_codex_auth_file(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    access = _jwt({"exp": int(time.time()) + 3600})
    auth_file.write_text(
        json.dumps(
            {
                "last_refresh": "2026-03-04T12:00:00+00:00",
                "tokens": {
                    "access_token": access,
                    "refresh_token": "rtok",
                    "id_token": _jwt(
                        {
                            "exp": int(time.time()) + 3600,
                            "https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"},
                        }
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    cache_file = tmp_path / "cache.json"
    manager = codex_oauth.OpenAICodexOAuthManager(
        cache_file=str(cache_file),
        source_file=str(auth_file),
    )

    assert manager.has_credentials()
    assert manager.get_account_id() == "acct-123"
    assert manager.get_access_token() == access


def test_manager_refreshes_expired_token(tmp_path: Path, monkeypatch):
    cache_file = tmp_path / "cache.json"
    expired = _jwt({"exp": int(time.time()) - 10})
    cache_file.write_text(
        json.dumps(
            {
                "type": "openai-codex",
                "access_token": expired,
                "refresh_token": "refresh-old",
                "id_token": _jwt({"exp": int(time.time()) - 10}),
                "account_id": "acct-old",
                "expires_at": time.time() - 10,
                "updated_at": time.time() - 30,
            }
        ),
        encoding="utf-8",
    )

    refreshed_access = _jwt({"exp": int(time.time()) + 7200})
    refreshed_id = _jwt({"chatgpt_account_id": "acct-new", "exp": int(time.time()) + 7200})

    class _DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "access_token": refreshed_access,
                    "refresh_token": "refresh-new",
                    "id_token": refreshed_id,
                    "expires_in": 7200,
                }
            ).encode("utf-8")

    def _fake_urlopen(_request, timeout: float):  # noqa: ANN001
        assert timeout > 0
        return _DummyResponse()

    monkeypatch.setattr(codex_oauth.urllib.request, "urlopen", _fake_urlopen)

    manager = codex_oauth.OpenAICodexOAuthManager(
        cache_file=str(cache_file),
        source_file=str(tmp_path / "missing-auth.json"),
    )
    token = manager.get_access_token()

    assert token == refreshed_access
    assert manager.get_account_id() == "acct-new"

    persisted = json.loads(cache_file.read_text(encoding="utf-8"))
    assert persisted["refresh_token"] == "refresh-new"


def test_pkce_helpers_generate_valid_values():
    verifier = codex_oauth.generate_code_verifier()
    challenge = codex_oauth.generate_code_challenge(verifier)
    state = codex_oauth.generate_oauth_state()

    assert len(verifier) >= 43
    assert challenge
    assert len(state) == 32


def test_build_authorization_url_contains_required_params():
    url = codex_oauth.build_authorization_url("challenge-123", "state-abc")
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "auth.openai.com"
    assert params["client_id"][0] == codex_oauth.CODEX_OAUTH_CLIENT_ID
    assert params["response_type"][0] == "code"
    assert params["code_challenge_method"][0] == "S256"
    assert params["codex_cli_simplified_flow"][0] == "true"
    assert params["originator"][0] == "kiss-ai"


def test_manager_exchanges_authorization_code_and_persists(tmp_path: Path, monkeypatch):
    cache_file = tmp_path / "cache.json"
    refreshed_access = _jwt({"exp": int(time.time()) + 7200})
    refreshed_id = _jwt({"chatgpt_account_id": "acct-login", "exp": int(time.time()) + 7200})

    class _DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "access_token": refreshed_access,
                    "refresh_token": "refresh-login",
                    "id_token": refreshed_id,
                    "expires_in": 7200,
                }
            ).encode("utf-8")

    def _fake_urlopen(_request, timeout: float):  # noqa: ANN001
        assert timeout > 0
        return _DummyResponse()

    monkeypatch.setattr(codex_oauth.urllib.request, "urlopen", _fake_urlopen)

    manager = codex_oauth.OpenAICodexOAuthManager(
        cache_file=str(cache_file),
        source_file=str(tmp_path / "missing-auth.json"),
    )
    token = manager.exchange_authorization_code("code-123", "verifier-123")

    assert token == refreshed_access
    assert manager.get_account_id() == "acct-login"
    persisted = json.loads(cache_file.read_text(encoding="utf-8"))
    assert persisted["refresh_token"] == "refresh-login"
