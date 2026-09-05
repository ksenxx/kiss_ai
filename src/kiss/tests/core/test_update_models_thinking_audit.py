# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cross-vendor regression audit for reasoning-effort alias generation.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.scripts.test_update_models_thinking_audit``; the non-core tests
remain there.

The scripted endpoint speaks both OpenAI-compatible transports: aliases whose
catalog entry carries ``use_responses_api: true`` (live-verified by
``scripts/update_responses_api_support``) stream from ``/v1/responses``, the
rest get a plain ``/v1/chat/completions`` JSON completion.  Effort assertions
are therefore shape-aware — ``reasoning.effort`` on the v2 wire,
``reasoning_effort`` on v1 — so a catalog refresh flipping an alias's
transport does not invalidate the audit.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
)


def _v1_completion(model_id: str, text: str = "hello") -> dict:
    """Build a minimal successful Chat Completions body echoing ``model_id``."""
    return {
        "id": "cmpl-wire",
        "object": "chat.completion",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        },
    }


def _v2_response(model_id: str, text: str = "hello") -> dict:
    """Build a minimal successful non-streaming ``/v1/responses`` body."""
    return {
        "id": "resp_wire",
        "object": "response",
        "created_at": 0,
        "model": model_id,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": text, "annotations": []}
                ],
            }
        ],
        "usage": {
            "input_tokens": 3,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 4,
        },
    }


def _responder(request: Request) -> Reply:
    """Answer ``/responses`` with a v2 body, anything else with v1 JSON."""
    model_id = request.body.get("model", "")
    if request.path.endswith("/responses"):
        return Reply(json_body=_v2_response(model_id))
    return Reply(json_body=_v1_completion(model_id))


@pytest.fixture
def effort_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[ScriptedOpenAIServer]:
    """Route the openrouter and together vendors at an in-process emulator."""
    import dataclasses

    from kiss.core import config as config_module
    from kiss.core.models import model_info

    server = ScriptedOpenAIServer(_responder)
    providers = tuple(
        dataclasses.replace(p, base_url=server.base_url)
        if p.name in ("openrouter", "together")
        else p
        for p in model_info.OPENAI_COMPATIBLE_PROVIDERS
    )
    monkeypatch.setattr(model_info, "OPENAI_COMPATIBLE_PROVIDERS", providers)
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "test-key", raising=False
    )
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG, "TOGETHER_API_KEY", "test-key", raising=False
    )
    try:
        yield server
    finally:
        server.stop()


def _generate_and_capture(server: ScriptedOpenAIServer, alias: str) -> Request:
    """Run a real ``generate()`` through ``alias`` and return the wire request."""
    from kiss.core.models.model_info import model as create_model

    m = create_model(alias)
    m.initialize("Say hello in one word.")
    text, _ = m.generate()
    assert text.strip() == "hello"
    requests = server.requests
    assert len(requests) == 1, (
        f"Exactly one request expected for {alias}; got {len(requests)}"
    )
    return requests[0]


def _wire_effort(request: Request) -> str:
    """Return the reasoning effort as it appeared on the wire.

    The v2 Responses transport nests it as ``reasoning.effort``; the v1
    Chat Completions transport sends a flat ``reasoning_effort``.
    """
    if request.path.endswith("/responses"):
        reasoning = request.body.get("reasoning") or {}
        return str(reasoning.get("effort", ""))
    return str(request.body.get("reasoning_effort", ""))


class TestWireShapePerVendorRoute:
    """Each new family's alias must put its reasoning effort on the wire."""

    def test_grok_4_5_high_via_openrouter(
        self, effort_wire: ScriptedOpenAIServer
    ) -> None:
        """openrouter/x-ai/grok-4.5-high → model=x-ai/grok-4.5, effort=high."""
        req = _generate_and_capture(effort_wire, "openrouter/x-ai/grok-4.5-high")
        assert req.body["model"] == "x-ai/grok-4.5", "Wire id must be the base model"
        assert _wire_effort(req) == "high"

    def test_glm_5_2_max_via_together(
        self, effort_wire: ScriptedOpenAIServer
    ) -> None:
        """zai-org/GLM-5.2-max → model=zai-org/GLM-5.2, effort=max."""
        req = _generate_and_capture(effort_wire, "zai-org/GLM-5.2-max")
        assert req.body["model"] == "zai-org/GLM-5.2", "Wire id must be the base model"
        assert _wire_effort(req) == "max"

    def test_glm_5_2_max_via_openrouter(
        self, effort_wire: ScriptedOpenAIServer
    ) -> None:
        """openrouter/z-ai/glm-5.2-max → model=z-ai/glm-5.2, effort=max."""
        req = _generate_and_capture(effort_wire, "openrouter/z-ai/glm-5.2-max")
        assert req.body["model"] == "z-ai/glm-5.2", "Wire id must be the base model"
        assert _wire_effort(req) == "max"

    def test_gpt_oss_120b_medium_via_together(
        self, effort_wire: ScriptedOpenAIServer
    ) -> None:
        """openai/gpt-oss-120b-medium → model=openai/gpt-oss-120b, effort=medium."""
        req = _generate_and_capture(effort_wire, "openai/gpt-oss-120b-medium")
        assert req.body["model"] == "openai/gpt-oss-120b", (
            "Wire id must be the base model"
        )
        assert _wire_effort(req) == "medium"

    def test_gpt_oss_20b_low_via_openrouter(
        self, effort_wire: ScriptedOpenAIServer
    ) -> None:
        """openrouter/openai/gpt-oss-20b-low → model=openai/gpt-oss-20b, effort=low."""
        req = _generate_and_capture(effort_wire, "openrouter/openai/gpt-oss-20b-low")
        assert req.body["model"] == "openai/gpt-oss-20b", (
            "Wire id must be the base model"
        )
        assert _wire_effort(req) == "low"
