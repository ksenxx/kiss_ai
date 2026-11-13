# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A model's family is its vendor whatever the route serves it (audit C2, F4).

``kimi-k3`` (Moonshot's own API) and ``moonshotai/Kimi-K3`` (Together) or
``openrouter/moonshotai/kimi-k3`` are one model; so are ``glm-4.7``
(Z.ai's API), ``zai-org/GLM-4.7`` (Together) and ``openrouter/z-ai/glm-4.7``.
The prefix table knew the routed spellings only, so the direct names fell
to the ``name.split("-")[0]`` rule (``kimi``, ``glm``) and a provider twin
passed as a second opinion or a fail-over of "another family".  No code is
patched: the credentials are set through the configuration the routing
reads.

Two configurations of one model are one worker exactly when its provider
is asked for the same reasoning: the last test sends real turns to local
endpoints and compares what they received with ``same_model_and_effort``.
"""

from __future__ import annotations

import contextlib
import itertools
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from google.genai import types

from kiss.core import config as config_module
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_families import (
    fallback_of_another_family,
    family_of,
    other_family_model,
    same_family,
    same_model_and_effort,
)
from kiss.core.models.model_info import (
    _NATIVE_PROVIDERS,
    OPENAI_COMPATIBLE_PROVIDERS,
    model,
    registered_bedrock_model,
)
from kiss.tests.core.models.anthropic_sse_harness import (
    ScriptedAnthropicServer,
    text_message_stream,
)
from kiss.tests.core.models.gemini_sse_harness import GeminiScript, chunk, serve, text_part
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

TWINS: tuple[tuple[str, str], ...] = (
    ("kimi-k3", "moonshotai/Kimi-K3"),
    ("kimi-k3", "openrouter/moonshotai/kimi-k3"),
    ("kimi-k2.5", "openrouter/~moonshotai/kimi-latest"),
    ("kimi-k3", "moonshot-v1-8k"),
    ("glm-4.7", "openrouter/z-ai/glm-4.7"),
    ("glm-4.7", "zai-org/GLM-4.7"),
    ("glm-5", "openrouter/~z-ai/glm-latest"),
)
"""The same vendor's models under the direct and the routed spellings."""


def _configure_only(monkeypatch: pytest.MonkeyPatch, *api_key_names: str) -> None:
    """Blank every HTTP provider credential, then set the named ones."""
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, provider.api_key_name, "")
    for _prefix, _label, api_key_name in _NATIVE_PROVIDERS:
        if api_key_name is not None:
            monkeypatch.setattr(config_module.DEFAULT_CONFIG, api_key_name, "")
    for api_key_name in api_key_names:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, api_key_name, "configured-for-test")


@pytest.mark.parametrize(("direct", "routed"), TWINS)
def test_direct_and_routed_spellings_are_one_family(direct: str, routed: str) -> None:
    """The direct name and its Together / OpenRouter twin share a family."""
    assert same_family(direct, routed), (family_of(direct), family_of(routed))


@pytest.mark.parametrize(
    ("model_name", "family"),
    [("kimi-k3", "moonshot"), ("moonshotai/Kimi-K3", "moonshot"), ("glm-4.7", "z-ai")],
)
def test_family_names(model_name: str, family: str) -> None:
    """The family is named after the vendor, as the routed spellings already were."""
    assert family_of(model_name) == family


def test_other_families_stay_distinct() -> None:
    """The new prefixes do not fold other vendors together."""
    assert not same_family("kimi-k3", "deepseek-ai/DeepSeek-V4-Pro")
    assert not same_family("glm-4.7", "kimi-k3")
    assert not same_family("glm-4.7", "Qwen/Qwen3-235B-A22B-Instruct")


def test_a_provider_twin_is_no_second_opinion(monkeypatch: pytest.MonkeyPatch) -> None:
    """With Moonshot and Together configured, Kimi's Together twin is not another family."""
    _configure_only(monkeypatch, "MOONSHOT_API_KEY", "TOGETHER_API_KEY")
    chosen = other_family_model("kimi-k3")
    assert chosen is not None and chosen != "moonshotai/Kimi-K3", chosen
    assert not same_family(chosen, "kimi-k3"), chosen


@pytest.mark.parametrize(
    ("model_name", "own_key", "twin"),
    [
        ("kimi-k3", "MOONSHOT_API_KEY", "openrouter/moonshotai/kimi-k3"),
        ("glm-4.7", "ZAI_API_KEY", "openrouter/z-ai/glm-4.7"),
    ],
)
def test_fail_over_prefers_a_configured_other_family_to_the_twin(
    monkeypatch: pytest.MonkeyPatch, model_name: str, own_key: str, twin: str
) -> None:
    """The declared OpenRouter twin loses to a configured model of another family.

    The caller's own provider, OpenRouter (the twin's route) and Anthropic
    are configured: the fail-over must be the Anthropic model, not the
    twin that shares the refusal, filter or cap that triggered it.
    """
    _configure_only(monkeypatch, own_key, "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")
    assert fallback_of_another_family(model_name) == "claude-opus-5-5", twin


_BEDROCK = "AWS_BEARER_TOKEN_BEDROCK"
_NATIVE = f"ZAI_API_KEY MOONSHOT_API_KEY {_BEDROCK}"


@pytest.mark.parametrize(
    ("model", "keys", "clis", "expected"),
    [
        ("gpt-6-astra", "OPENAI_API_KEY MOONSHOT_API_KEY", "", "kimi-k3"),
        ("gpt-6-astra", "OPENAI_API_KEY ZAI_API_KEY", "", "glm-4.7"),
        ("gpt-6-astra", f"OPENAI_API_KEY {_BEDROCK}", "", ""),
        ("gpt-6-astra", "OPENAI_API_KEY MOONSHOT_API_KEY", "claude", "cc/opus"),
        ("gpt-6-astra", "OPENAI_API_KEY ZAI_API_KEY", "claude", "cc/opus"),
        ("gpt-6-astra", f"OPENAI_API_KEY {_BEDROCK}", "claude", "cc/opus"),
        ("claude-opus-5-5", "ANTHROPIC_API_KEY MOONSHOT_API_KEY", "codex", "codex/default"),
        ("claude-opus-5-5", "ANTHROPIC_API_KEY ZAI_API_KEY", "codex", "codex/default"),
        ("gemini-3.1-pro-preview", f"GEMINI_API_KEY {_BEDROCK}", "codex", "codex/default"),
        ("gpt-6-astra", f"OPENAI_API_KEY {_NATIVE}", "claude codex", "cc/opus"),
    ],
)
def test_a_native_route_is_a_second_opinion_only_where_there_was_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: str, keys: str, clis: str, expected: str
) -> None:
    """Moonshot, Z.AI and Bedrock answer only an installation without another family (audit C5, F1).

    Without a CLI, OpenAI plus one of them was told no other family is
    configured (``""``: the strongest Bedrock model registered); with one,
    the CLI model stays the answer.  ``PATH`` holds only the named CLIs,
    as executables the selection looks up and never runs.
    """
    for cli in clis.split():
        executable = f"{cli}.cmd" if sys.platform == "win32" else cli
        (tmp_path / executable).touch(mode=0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    _configure_only(monkeypatch, *keys.split())
    assert other_family_model(model) == (expected or registered_bedrock_model(cheapest=False))


def _openai_reply(request: Request) -> Reply:
    """Responder: a complete reply of the API the request is for, Responses or Chat Completions."""
    if request.path.endswith("/responses"):
        text = {"type": "output_text", "text": "ok", "annotations": []}
        message = {"type": "message", "id": "m", "role": "assistant", "status": "completed"}
        return Reply(
            json_body={
                "id": "r",
                "object": "response",
                "status": "completed",
                "model": "gpt-6-astra",
                "output": [message | {"content": [text]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        )
    message = {"role": "assistant", "content": "ok"}
    choice = {"index": 0, "message": message, "finish_reason": "stop"}
    return Reply(
        json_body={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-6-astra",
            "choices": [choice],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


class _Endpoints:
    """Local endpoints of the three API styles, each recording the requests it receives."""

    def __init__(
        self, anthropic: ScriptedAnthropicServer, gemini: GeminiScript, openai: ScriptedOpenAIServer
    ) -> None:
        self.anthropic, self.gemini, self.openai = anthropic, gemini, openai

    def asked_for(self, name: str, config: dict[str, Any]) -> Any:
        """Run one turn of *name* under *config*; return the reasoning its endpoint was asked for.

        The adapter is the one the model factory builds for the pair.
        """
        turn = model(name, model_config=config)
        try:
            turn.initialize("hi")
            turn.generate()
        finally:
            turn.client.close()  # type: ignore[attr-defined]
        if isinstance(turn, AnthropicModel):
            return self.anthropic.requests[-1]["output_config"]["effort"]
        if isinstance(turn, GeminiModel):
            thinking = self.gemini.requests[-1]["generationConfig"]["thinkingConfig"]
            return thinking.get("thinking_level", thinking.get("thinking_budget"))
        body = self.openai.requests[-1].body
        # Chat Completions takes ``reasoning_effort``, the Responses API ``reasoning.effort``.
        return body.get("reasoning_effort", (body.get("reasoning") or {}).get("effort"))


@pytest.fixture
def endpoints(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Endpoints]:
    """The three local endpoints, configured as the endpoints of their providers."""
    _configure_only(monkeypatch, "ANTHROPIC_API_KEY", "GEMINI_API_KEY")
    with (
        ScriptedAnthropicServer(text_message_stream()) as anthropic,
        ScriptedOpenAIServer(_openai_reply) as openai,
        contextlib.contextmanager(serve)() as (gemini_url, gemini),
    ):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", anthropic.base_url)
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", gemini_url)
        gemini.play([chunk([text_part("ok")], finish_reason="STOP")])
        yield _Endpoints(anthropic, gemini, openai)


_LOW = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)

REASONING: dict[str, list[tuple[dict[str, Any], Any]]] = {
    "claude-opus-5-5": [
        ({}, "high"),
        ({"output_config": {"effort": "high"}}, "high"),
        ({"output_config": {"effort": "low"}}, "low"),
        ({"reasoning_effort": "low"}, "low"),
        ({"reasoning_effort": "max", "output_config": {"effort": "low"}}, "max"),
        ({"reasoning": {"effort": "low"}}, "high"),
        ({"reasoning_effort": None, "output_config": {"effort": None}}, "high"),
        ({"reasoning_effort": None, "output_config": {"effort": "low"}}, "low"),
    ],
    "gemini-3-flash-preview": [
        ({}, "HIGH"),
        ({"reasoning_effort": "xhigh"}, "HIGH"),
        ({"reasoning_effort": "max"}, "HIGH"),
        ({"reasoning_effort": "low"}, "LOW"),
        ({"thinking_config": {"thinking_level": "low"}}, "LOW"),
        ({"thinking_config": {"thinkingLevel": "low"}}, "LOW"),
        ({"thinking_config": _LOW}, "LOW"),
        ({"reasoning_effort": "max", "thinking_config": _LOW}, "HIGH"),
        ({"thinking_config": {"thinking_budget": 512}}, 512),
        ({"thinking_config": {"thinking_budget": 0}}, 0),
        ({"reasoning_effort": None, "thinking_config": {"thinking_level": None}}, "HIGH"),
        ({"reasoning_effort": "", "thinking_config": _LOW}, "LOW"),
        ({"reasoning_effort": None, "thinking_config": {"thinking_level": "low"}}, "LOW"),
    ],
    "gpt-6-astra": [
        ({}, "high"),
        ({"reasoning": {"effort": "low"}}, "low"),
        ({"reasoning_effort": "low"}, "low"),
        ({"reasoning_effort": "high", "reasoning": {"effort": "low"}}, "high"),
        ({"reasoning_effort": None}, None),
        ({"reasoning": {"effort": None}}, None),
        ({"reasoning_effort": None, "reasoning": {"effort": "low"}}, "low"),
    ],
}
"""Per model, each configuration with the reasoning its provider is asked for under it.

The catalog runs all three models at ``high``.  A provider's own spelling
of the effort (``output_config.effort``, ``thinking_config``,
``reasoning.effort``) ranks between the portable ``reasoning_effort`` and
the catalog's level, and means nothing to another provider's adapter;
Gemini has no level above ``high``.  A null setting is none (to Gemini,
an empty portable one too), except to the OpenAI adapters: they take a
key that is there for the caller's choice and ask for no effort, which
is the provider's default, unless the other spelling names a level.
"""


@pytest.mark.parametrize(
    ("name", "responses_api"),
    [
        ("claude-opus-5-5", None),
        ("gemini-3-flash-preview", None),
        ("gpt-6-astra", True),
        ("gpt-6-astra", False),
    ],
    ids=["anthropic", "gemini", "openai-responses", "openai-chat-completions"],
)
def test_one_effort_is_one_reasoning_asked_of_the_provider(
    name: str, responses_api: bool | None, endpoints: _Endpoints
) -> None:
    """Two configurations are one worker exactly when the provider is asked for the same reasoning.

    The effort was read from ``reasoning_effort`` and the catalog alone:
    a provider's own setting was ignored (two sub-agents at native ``low``
    and ``high`` passed for one worker, native and portable ``low`` for
    two) and Gemini's ``xhigh`` and ``max``, both sent as ``high``, for
    three.  ``gpt-6-astra`` goes through each of the two OpenAI APIs.
    """
    route: dict[str, Any] = {}
    if responses_api is not None:
        url = endpoints.openai.base_url
        route = {"base_url": url, "api_key": "k", "use_responses_api": responses_api}
    turns = [(route | config, asked) for config, asked in REASONING[name]]
    for config, asked in turns:
        assert endpoints.asked_for(name, config) == asked, config
    for (config_a, asked_a), (config_b, asked_b) in itertools.combinations(turns, 2):
        same = same_model_and_effort(name, name, config_a, config_b)
        assert same == (asked_a == asked_b), (config_a, config_b)


def test_the_codex_cli_reads_no_effort_but_the_portable_one() -> None:
    """``CodexModel._effort_level`` knows ``reasoning_effort`` and the catalog only."""
    native = {"reasoning": {"effort": "low"}, "output_config": {"effort": "low"}}
    assert same_model_and_effort("codex/default", "codex/default", native)
    assert not same_model_and_effort("codex/default", "codex/default", {"reasoning_effort": "low"})
