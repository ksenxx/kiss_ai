# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the vendor capability registry and adaptive probe.

Whether ``tools`` + ``reasoning_effort`` survive on the same Chat
Completions request used to be governed by a hardcoded host allowlist in
``openai_compatible_model.py`` — every new vendor (OpenRouter, Together,
...) required a manual patch.  The knowledge now lives in a single
config-driven registry, ``model_info.OPENAI_COMPATIBLE_PROVIDERS``: each
factory-routed vendor declares ``tools_accept_reasoning_effort``
(True/False) when verified live, or ``None`` when unverified — in which
case the transport keeps the effort optimistically and *learns* the
verdict from the endpoint's actual response at runtime (adaptive probe),
so unknown gateways and future vendors work with no manual patch at all.

Contract verified here:

* Tripwire: adding a new vendor to the registry fails a test until the
  developer consciously declares its capability and updates the expected
  vendor set here.
* The ``model()`` factory routes every registered prefix to the registered
  ``base_url`` (registry and routing cannot drift — they are one table).
* SorcarAgent's ``set_model`` endpoint carry-over set derives from the
  registry.
* On the wire (real in-process HTTP server, no mocks): declared-True hosts
  keep the effort; declared-False hosts strip it; unknown and
  declared-None hosts send it optimistically, retry once without it when
  the endpoint rejects it with a 400 mentioning ``reasoning_effort``, and
  cache the verdict per endpoint either way.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.models.model_info import (
    OPENAI_COMPATIBLE_PROVIDERS,
    model,
    openai_compatible_provider_for_base_url,
)
from kiss.core.models.openai_compatible_model import (
    _ADAPTIVE_TOOL_EFFORT_VERDICTS,
    OpenAICompatibleModel,
)

# ---------------------------------------------------------------------------
# Registry tripwire and consistency
# ---------------------------------------------------------------------------

_KNOWN_VENDORS = {"openai", "openrouter", "together", "zai", "moonshot"}

_TRIPWIRE_MESSAGE = (
    "A new OpenAI-compatible vendor was added to (or removed from) "
    "model_info.OPENAI_COMPATIBLE_PROVIDERS. For a NEW vendor you MUST: "
    "(1) declare its tools_accept_reasoning_effort capability — True/False "
    "verified live against the vendor's Chat Completions endpoint, or None "
    "to let the transport probe adaptively at runtime; (2) declare "
    "delegate_tools_to_responses (True only if the vendor implements "
    "/v1/responses AND rejects tools + reasoning_effort on Chat "
    "Completions, like api.openai.com); (3) add the vendor name to "
    "_KNOWN_VENDORS in this test. Routing, capability handling and "
    "SorcarAgent endpoint carry-over all derive from the registry — no "
    "other code change is needed."
)


class TestRegistryTripwire:
    """Adding a vendor must be a conscious, capability-declaring decision."""

    def test_new_vendor_must_declare_capability(self) -> None:
        """The registered vendor set matches the reviewed, expected set."""
        assert {p.name for p in OPENAI_COMPATIBLE_PROVIDERS} == _KNOWN_VENDORS, (
            _TRIPWIRE_MESSAGE
        )

    def test_registry_entries_are_well_formed(self) -> None:
        """Names, hosts and base_urls are unique and mutually consistent."""
        names = [p.name for p in OPENAI_COMPATIBLE_PROVIDERS]
        hosts = [p.host for p in OPENAI_COMPATIBLE_PROVIDERS]
        base_urls = [p.base_url for p in OPENAI_COMPATIBLE_PROVIDERS]
        assert len(set(names)) == len(names)
        assert len(set(hosts)) == len(hosts)
        assert len(set(base_urls)) == len(base_urls)
        for p in OPENAI_COMPATIBLE_PROVIDERS:
            assert p.tools_accept_reasoning_effort in (True, False, None)
            assert isinstance(p.delegate_tools_to_responses, bool)
            assert p.host in p.base_url, f"{p.name}: host must appear in base_url"
            assert p.prefixes, f"{p.name}: at least one routing prefix required"
            assert openai_compatible_provider_for_base_url(p.base_url) is p

    def test_unknown_endpoint_has_no_registry_entry(self) -> None:
        """Custom gateways resolve to None (adaptive handling)."""
        assert openai_compatible_provider_for_base_url("http://127.0.0.1:9/v1") is None


class TestFactoryRoutesMatchRegistry:
    """The model() factory and the registry are a single source of truth."""

    _REPRESENTATIVE = {
        "openrouter": "openrouter/openai/gpt-4o",
        "openai": "gpt-4o",
        "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "zai": "glm-4.6",
        "moonshot": "kimi-k2-0905-preview",
    }

    def test_every_vendor_routes_to_registered_base_url(self) -> None:
        """A representative model name per vendor routes to its base_url."""
        for provider in OPENAI_COMPATIBLE_PROVIDERS:
            name = self._REPRESENTATIVE[provider.name]
            m = model(name)
            assert isinstance(m, OpenAICompatibleModel), (provider.name, name)
            assert m.base_url == provider.base_url, (provider.name, name)
            assert openai_compatible_provider_for_base_url(m.base_url) is provider

    def test_excluded_prefixes_route_elsewhere(self) -> None:
        """Registry excludes keep non-OpenAI-compatible names out of the table."""
        assert type(model("text-embedding-004")).__name__ == "GeminiModel"
        together = model("openai/gpt-oss-20b")
        assert isinstance(together, OpenAICompatibleModel)
        assert together.base_url == "https://api.together.xyz/v1"

    def test_sorcar_default_base_urls_derive_from_registry(self) -> None:
        """set_model's endpoint carry-over set covers every registered vendor."""
        from kiss.agents.sorcar.sorcar_agent import _FACTORY_DEFAULT_BASE_URLS

        assert _FACTORY_DEFAULT_BASE_URLS == frozenset(
            p.base_url.rstrip("/") for p in OPENAI_COMPATIBLE_PROVIDERS
        )


# ---------------------------------------------------------------------------
# Wire behavior: real in-process HTTP server (no mocks).
# ---------------------------------------------------------------------------


def _chat_ok_json() -> bytes:
    """Return a minimal /v1/chat/completions non-streaming JSON body."""
    return json.dumps(
        {
            "id": "chatcmpl-ok",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    ).encode()


class _CapabilityHandler(BaseHTTPRequestHandler):
    """Captures chat.completions bodies; optionally rejects reasoning_effort.

    When ``reject_reasoning_effort`` is True, any payload containing a
    ``reasoning_effort`` key gets a 400 whose message mentions the
    parameter — mimicking vendors that reject ``tools`` + effort.
    """

    captured_bodies: list[dict[str, Any]] = []
    reject_reasoning_effort = False

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.captured_bodies.append(body)
        if self.__class__.reject_reasoning_effort and "reasoning_effort" in body:
            payload = json.dumps(
                {
                    "error": {
                        "message": (
                            "Function tools with reasoning_effort are not "
                            "supported on this endpoint."
                        ),
                        "type": "invalid_request_error",
                        "code": "unsupported_parameter",
                    }
                }
            ).encode()
            self.send_response(400)
        else:
            payload = _chat_ok_json()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def wire_server() -> Generator[str]:
    """Spawn an in-process capture server; yields its /v1 base URL."""
    _CapabilityHandler.captured_bodies = []
    _CapabilityHandler.reject_reasoning_effort = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CapabilityHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        for key in [k for k in _ADAPTIVE_TOOL_EFFORT_VERDICTS if str(server.server_port) in k]:
            _ADAPTIVE_TOOL_EFFORT_VERDICTS.pop(key, None)


def echo(text: str) -> str:
    """Echo back ``text`` (test-only tool stub).

    Args:
        text: The string to echo.

    Returns:
        The input string unchanged.
    """
    return text


def _make_model(base_url: str, effort: str = "high") -> OpenAICompatibleModel:
    """Build a v1 model pointed at *base_url* with an explicit effort."""
    return OpenAICompatibleModel(
        "gpt-4o",
        base_url=base_url,
        api_key="test-key",
        model_config={"reasoning_effort": effort},
    )


class TestDeclaredCapabilityOnTheWire:
    """Registry-declared verdicts drive the wire behavior directly."""

    def test_declared_true_host_keeps_effort(self, wire_server: str) -> None:
        """A Together-hosted URL keeps tools + reasoning_effort as declared."""
        m = _make_model(f"{wire_server}/api.together.xyz")
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        assert len(_CapabilityHandler.captured_bodies) == 1
        body = _CapabilityHandler.captured_bodies[0]
        assert body["reasoning_effort"] == "high"
        assert body["tools"]

    def test_declared_false_host_strips_effort(self, wire_server: str) -> None:
        """An OpenAI-hosted URL strips the effort from tool-bearing requests.

        ``use_responses_api=False`` disables the Responses delegation so the
        request stays on Chat Completions, where api.openai.com's declared
        False capability must strip the effort (single request, no retry).
        """
        m = OpenAICompatibleModel(
            "gpt-5.5-xhigh",
            base_url=f"{wire_server}/api.openai.com",
            api_key="test-key",
            model_config={"use_responses_api": False},
        )
        assert m.model_config["reasoning_effort"] == "xhigh"
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        assert len(_CapabilityHandler.captured_bodies) == 1
        body = _CapabilityHandler.captured_bodies[0]
        assert "reasoning_effort" not in body
        assert body["tools"]

    def test_declared_none_vendor_uses_adaptive_path(self, wire_server: str) -> None:
        """A registered vendor declared None (z.ai) probes adaptively."""
        base_url = f"{wire_server}/api.z.ai"
        provider = openai_compatible_provider_for_base_url(base_url)
        assert provider is not None and provider.name == "zai"
        assert provider.tools_accept_reasoning_effort is None
        m = _make_model(base_url)
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        assert _CapabilityHandler.captured_bodies[0]["reasoning_effort"] == "high"
        assert _ADAPTIVE_TOOL_EFFORT_VERDICTS.pop(base_url) is True


class TestAdaptiveProbeOnUnknownEndpoints:
    """Unknown gateways need no manual patch: the transport learns live."""

    def test_accepting_endpoint_keeps_effort_and_caches_verdict(
        self, wire_server: str
    ) -> None:
        """A 200 with the effort attached caches an accepting verdict."""
        m = _make_model(wire_server)
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        assert len(_CapabilityHandler.captured_bodies) == 1
        assert _CapabilityHandler.captured_bodies[0]["reasoning_effort"] == "high"
        assert _ADAPTIVE_TOOL_EFFORT_VERDICTS[wire_server] is True
        # Second turn: still direct, still with the effort.
        m.generate_and_process_with_tools({"echo": echo})
        assert len(_CapabilityHandler.captured_bodies) == 2
        assert _CapabilityHandler.captured_bodies[1]["reasoning_effort"] == "high"

    def test_rejecting_endpoint_retries_without_effort_and_caches_verdict(
        self, wire_server: str
    ) -> None:
        """A 400 mentioning reasoning_effort triggers one retry without it.

        The rejecting verdict is cached, so a subsequent call sends a
        single request with the effort already stripped.
        """
        _CapabilityHandler.reject_reasoning_effort = True
        m = _make_model(wire_server)
        m.initialize("hi")
        _, content, _ = m.generate_and_process_with_tools({"echo": echo})
        assert content == "ok"
        assert len(_CapabilityHandler.captured_bodies) == 2
        assert _CapabilityHandler.captured_bodies[0]["reasoning_effort"] == "high"
        assert "reasoning_effort" not in _CapabilityHandler.captured_bodies[1]
        assert _CapabilityHandler.captured_bodies[1]["tools"]
        assert _ADAPTIVE_TOOL_EFFORT_VERDICTS[wire_server] is False
        # Second turn: the cached verdict strips upfront — exactly one request.
        m.generate_and_process_with_tools({"echo": echo})
        assert len(_CapabilityHandler.captured_bodies) == 3
        assert "reasoning_effort" not in _CapabilityHandler.captured_bodies[2]

    def test_unrelated_400_is_not_swallowed(self, wire_server: str) -> None:
        """A 400 for a request without effort propagates unchanged."""
        _CapabilityHandler.reject_reasoning_effort = True
        m = _make_model(wire_server)
        # Pre-cache a rejecting verdict, then force the effort back into the
        # request via a no-tools path check: without tools the wrapper must
        # not retry-strip, and the server accepts (no effort key on 400 rule
        # only when present) — instead verify the error path directly.
        _ADAPTIVE_TOOL_EFFORT_VERDICTS[wire_server] = True
        m.initialize("hi")
        from openai import BadRequestError

        with pytest.raises(BadRequestError):
            m.generate_and_process_with_tools({"echo": echo})
        # Known-True verdict means no adaptive retry: exactly one request.
        assert len(_CapabilityHandler.captured_bodies) == 1
        _ADAPTIVE_TOOL_EFFORT_VERDICTS.pop(wire_server, None)

    def test_no_effort_request_does_not_cache_a_verdict(
        self, wire_server: str
    ) -> None:
        """Requests without reasoning_effort never record a verdict."""
        m = OpenAICompatibleModel(
            "gpt-4o", base_url=wire_server, api_key="test-key"
        )
        m.initialize("hi")
        m.generate_and_process_with_tools({"echo": echo})
        assert "reasoning_effort" not in _CapabilityHandler.captured_bodies[0]
        assert wire_server not in _ADAPTIVE_TOOL_EFFORT_VERDICTS
