"""LLM client abstraction.

Provides a uniform generate() interface for the three portfolio members used in the
Cleverest+ end-to-end campaign:
  - OpenAI GPT-4o at a caller-supplied temperature (used at 0.5, 1.0, and 1.0 for
    the single-shot member).
  - OpenRouter DeepSeek-R1 (reasoning model), pinned to a provider that supports
    response_format=json_object so structured candidates are enforced server-side.

The client returns the model text and a token/cost estimate. It never invokes a shell
and never persists API keys to disk.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class ChatMessage:
    """One message in the OpenAI-compatible chat.completions schema."""

    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Result of a single chat completion call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_s: float


class LLMError(RuntimeError):
    """A model call could not be completed after retries."""


class LLMClient:
    """OpenAI-compatible chat client.

    Two provider profiles are supported without changing the wire format because both
    OpenAI and OpenRouter expose the same schema.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_output_tokens: int = 4096,
        response_format_json: bool = True,
        request_timeout_s: float = 90.0,
        provider_pin: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._response_format_json = response_format_json
        self._timeout = request_timeout_s
        self._provider_pin = provider_pin
        self._extra_headers = dict(extra_headers) if extra_headers else {}

    @property
    def model(self) -> str:
        """Return the model identifier used for wire-format requests."""
        return self._model

    def generate(self, messages: Sequence[ChatMessage], *, seed: int | None = None) -> LLMResponse:
        """Perform one chat completion and return the assistant text.

        Retries transient HTTP errors with exponential backoff up to three attempts.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if self._response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if self._provider_pin is not None:
            payload["provider"] = {"order": [self._provider_pin], "allow_fallbacks": False}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)

        last_error: Exception | None = None
        for attempt in range(3):
            start = time.monotonic()
            try:
                req = urllib.request.Request(
                    f"{self._base_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read()
                latency = time.monotonic() - start
                body = cast(dict[str, Any], json.loads(raw.decode("utf-8")))
                choices = cast(list[dict[str, Any]], body.get("choices", []))
                if not choices:
                    raise LLMError(f"empty choices from {self._model}: {body}")
                message = cast(dict[str, Any], choices[0].get("message", {}))
                text = cast(str, message.get("content") or "")
                usage = cast(dict[str, Any], body.get("usage", {}))
                return LLMResponse(
                    text=text,
                    prompt_tokens=int(cast(int, usage.get("prompt_tokens", 0))),
                    completion_tokens=int(cast(int, usage.get("completion_tokens", 0))),
                    model=cast(str, body.get("model", self._model)),
                    latency_s=latency,
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise LLMError(f"HTTP {exc.code} from {self._model}: {detail}") from exc
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        raise LLMError(f"{self._model} failed after retries: {last_error!r}")


def openai_client(*, model: str, temperature: float, response_format_json: bool = True) -> LLMClient:
    """Instantiate a client for the OpenAI chat.completions endpoint."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY is not set")
    return LLMClient(
        base_url="https://api.openai.com/v1",
        api_key=key,
        model=model,
        temperature=temperature,
        response_format_json=response_format_json,
    )


def openrouter_client(
    *,
    model: str,
    temperature: float,
    response_format_json: bool = True,
    provider_pin: str | None = "novita",
) -> LLMClient:
    """Instantiate a client for the OpenRouter chat.completions endpoint."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set")
    return LLMClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        model=model,
        temperature=temperature,
        response_format_json=response_format_json,
        provider_pin=provider_pin,
        extra_headers={
            "HTTP-Referer": "https://github.com/ksenxx/kiss_ai",
            "X-Title": "Cleverest+ Campaign",
        },
    )
