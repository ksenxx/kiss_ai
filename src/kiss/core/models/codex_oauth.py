# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Native OAuth token management for OpenAI Codex subscription auth."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODEX_OAUTH_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_AUTHORIZATION_ENDPOINT = "https://auth.openai.com/oauth/authorize"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_REDIRECT_URI = "http://localhost:1455/auth/callback"
CODEX_OAUTH_SCOPES = "openid profile email offline_access"
CODEX_OAUTH_CALLBACK_PORT = 1455


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    """Decode JWT payload claims; return empty dict when token is malformed."""
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("ascii"))
        decoded = json.loads(raw.decode("utf-8"))
        if isinstance(decoded, dict):
            return decoded
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return {}


def generate_code_verifier() -> str:
    """Generate a PKCE code verifier (base64url without padding)."""
    raw = secrets.token_bytes(32)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_code_challenge(code_verifier: str) -> str:
    """Generate S256 PKCE code challenge from verifier."""
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def generate_oauth_state() -> str:
    """Generate random OAuth state for CSRF protection."""
    return secrets.token_hex(16)


def build_authorization_url(
    code_challenge: str,
    state: str,
    *,
    originator: str = "kiss-ai",
    redirect_uri: str = CODEX_OAUTH_REDIRECT_URI,
) -> str:
    """Build OpenAI Codex OAuth authorization URL with PKCE."""
    params = urllib.parse.urlencode(
        {
            "client_id": CODEX_OAUTH_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": CODEX_OAUTH_SCOPES,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "response_type": "code",
            "state": state,
            # Codex-specific flow flags used by known working implementations.
            "codex_cli_simplified_flow": "true",
            "originator": originator,
        }
    )
    return f"{CODEX_OAUTH_AUTHORIZATION_ENDPOINT}?{params}"


def _extract_account_id_from_claims(claims: dict[str, Any]) -> str | None:
    """Extract ChatGPT account id from known JWT claim locations."""
    root = claims.get("chatgpt_account_id")
    if isinstance(root, str) and root:
        return root

    nested = claims.get("https://api.openai.com/auth")
    if isinstance(nested, dict):
        nested_id = nested.get("chatgpt_account_id")
        if isinstance(nested_id, str) and nested_id:
            return nested_id

    organizations = claims.get("organizations")
    if isinstance(organizations, list) and organizations:
        first = organizations[0]
        if isinstance(first, dict):
            org_id = first.get("id")
            if isinstance(org_id, str) and org_id:
                return org_id
    return None


def _extract_expiry_timestamp(access_token: str, id_token: str | None = None) -> float | None:
    """Extract token expiration timestamp (seconds since epoch) from JWT claims."""
    for token in (id_token, access_token):
        if not token:
            continue
        claims = _decode_jwt_claims(token)
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            return float(exp)
    return None


@dataclass
class CodexOAuthCredentials:
    access_token: str
    refresh_token: str | None
    id_token: str | None
    account_id: str | None
    expires_at: float | None
    updated_at: float

    def is_expired(self, *, buffer_seconds: int = 300) -> bool:
        """Return True if the access token is expired (with clock-skew buffer)."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at - buffer_seconds


class OpenAICodexOAuthManager:
    """Loads and refreshes OpenAI Codex OAuth credentials."""

    def __init__(
        self,
        cache_file: str | None = None,
        source_file: str | None = None,
        token_endpoint: str = CODEX_OAUTH_TOKEN_ENDPOINT,
        client_id: str = CODEX_OAUTH_CLIENT_ID,
    ) -> None:
        cache_default = Path("~/.kiss/codex_oauth.json").expanduser()
        source_default = Path("~/.codex/auth.json").expanduser()
        self.cache_file = Path(cache_file).expanduser() if cache_file else cache_default
        self.source_file = Path(source_file).expanduser() if source_file else source_default
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self._credentials: CodexOAuthCredentials | None = None
        self._lock = threading.Lock()

    def has_credentials(self) -> bool:
        """Return True if a usable access token or refresh token is available."""
        creds = self._load_credentials()
        if creds is None:
            return False
        return bool(creds.access_token or creds.refresh_token)

    def get_account_id(self) -> str | None:
        """Return cached ChatGPT account id when available."""
        creds = self._load_credentials()
        return creds.account_id if creds else None

    def force_refresh_access_token(self) -> str | None:
        """Force refresh using the stored refresh token."""
        return self.get_access_token(force_refresh=True)

    def exchange_authorization_code(
        self,
        code: str,
        code_verifier: str,
        *,
        redirect_uri: str = CODEX_OAUTH_REDIRECT_URI,
    ) -> str | None:
        """Exchange OAuth authorization code for tokens and persist credentials."""
        form = urllib.parse.urlencode(
            {
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.token_endpoint,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        timeout_seconds = float(os.environ.get("KISS_CODEX_OAUTH_TIMEOUT_SECONDS", "30"))
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return None

        try:
            token_data = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(token_data, dict):
            return None

        with self._lock:
            credentials = self._credentials_from_token_response(token_data)
            if credentials is None:
                return None
            self._credentials = credentials
            self._save_credentials(credentials)
            return credentials.access_token

    def get_access_token(self, *, force_refresh: bool = False) -> str | None:
        """Return a valid access token, refreshing when needed."""
        with self._lock:
            creds = self._load_credentials()
            if creds is None:
                return None
            needs_refresh = force_refresh or creds.is_expired() or not creds.access_token
            if needs_refresh:
                refreshed = self._refresh_credentials(creds)
                if refreshed is None:
                    return None
                self._credentials = refreshed
                self._save_credentials(refreshed)
                creds = refreshed
            return creds.access_token

    def _load_credentials(self) -> CodexOAuthCredentials | None:
        if self._credentials is not None:
            return self._credentials

        creds = self._load_from_cache()
        if creds is not None:
            self._credentials = creds
            return creds

        creds = self._load_from_codex_auth_file()
        if creds is not None:
            self._credentials = creds
            # Persist to KISS cache so we can refresh without mutating Codex-managed files.
            self._save_credentials(creds)
        return creds

    def _load_from_cache(self) -> CodexOAuthCredentials | None:
        if not self.cache_file.exists():
            return None
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return self._credentials_from_dict(data)

    def _load_from_codex_auth_file(self) -> CodexOAuthCredentials | None:
        auth_file_env = os.environ.get("KISS_CODEX_AUTH_FILE")
        source_file = Path(auth_file_env).expanduser() if auth_file_env else self.source_file
        if not source_file.exists():
            return None
        try:
            data = json.loads(source_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        token_data: dict[str, Any] = {}
        if isinstance(data.get("tokens"), dict):
            token_data = dict(data["tokens"])
        else:
            token_data = data

        creds = self._credentials_from_dict(token_data)
        if creds is None:
            return None

        last_refresh = data.get("last_refresh")
        if isinstance(last_refresh, str):
            try:
                # "2026-03-03T18:35:39.038815400Z" -> epoch seconds
                ts = last_refresh.replace("Z", "+00:00")
                creds.updated_at = max(creds.updated_at, _iso_to_epoch(ts))
            except ValueError:
                pass
        return creds

    def _credentials_from_dict(self, data: dict[str, Any]) -> CodexOAuthCredentials | None:
        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return None

        refresh_token = data.get("refresh_token")
        if not isinstance(refresh_token, str):
            refresh_token = None

        id_token = data.get("id_token")
        if not isinstance(id_token, str):
            id_token = None

        account_id = data.get("account_id")
        if not isinstance(account_id, str):
            account_id = None
        if not account_id:
            account_id = _extract_account_id_from_claims(
                _decode_jwt_claims(id_token or access_token)
            )

        expires_at_val = data.get("expires_at")
        expires_at = float(expires_at_val) if isinstance(expires_at_val, (int, float)) else None
        if expires_at is None:
            expires_at = _extract_expiry_timestamp(access_token, id_token)

        updated_at_val = data.get("updated_at")
        updated_at = (
            float(updated_at_val)
            if isinstance(updated_at_val, (int, float))
            else time.time()
        )

        return CodexOAuthCredentials(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            account_id=account_id,
            expires_at=expires_at,
            updated_at=updated_at,
        )

    def _save_credentials(self, credentials: CodexOAuthCredentials) -> None:
        payload = {
            "type": "openai-codex",
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "id_token": credentials.id_token,
            "account_id": credentials.account_id,
            "expires_at": credentials.expires_at,
            "updated_at": credentials.updated_at,
        }
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            # Cache persistence failure should not break request flow.
            return

    def _refresh_credentials(
        self, credentials: CodexOAuthCredentials
    ) -> CodexOAuthCredentials | None:
        if not credentials.refresh_token:
            return None

        form = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": credentials.refresh_token,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.token_endpoint,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        timeout_seconds = float(os.environ.get("KISS_CODEX_OAUTH_TIMEOUT_SECONDS", "15"))

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return None

        try:
            token_data = json.loads(body)
        except json.JSONDecodeError:
            return None

        refreshed = self._credentials_from_token_response(
            token_data,
            fallback_refresh_token=credentials.refresh_token,
            fallback_id_token=credentials.id_token,
            fallback_account_id=credentials.account_id,
        )
        return refreshed

    def _credentials_from_token_response(
        self,
        token_data: dict[str, Any],
        *,
        fallback_refresh_token: str | None = None,
        fallback_id_token: str | None = None,
        fallback_account_id: str | None = None,
    ) -> CodexOAuthCredentials | None:
        access_token = token_data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return None

        refresh_token = token_data.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            refresh_token = fallback_refresh_token

        id_token = token_data.get("id_token")
        if not isinstance(id_token, str):
            id_token = fallback_id_token

        expires_in = token_data.get("expires_in")
        expires_at: float | None = None
        if isinstance(expires_in, (int, float)):
            expires_at = time.time() + float(expires_in)
        if expires_at is None:
            expires_at = _extract_expiry_timestamp(access_token, id_token)

        account_id = _extract_account_id_from_claims(_decode_jwt_claims(id_token or access_token))
        if not account_id:
            account_id = fallback_account_id

        return CodexOAuthCredentials(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            account_id=account_id,
            expires_at=expires_at,
            updated_at=time.time(),
        )


def _iso_to_epoch(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value).timestamp()
