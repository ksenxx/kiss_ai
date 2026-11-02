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
"""

from __future__ import annotations

import pytest

from kiss.core import config as config_module
from kiss.core.models.model_families import (
    fallback_of_another_family,
    family_of,
    other_family_model,
    same_family,
)
from kiss.core.models.model_info import _NATIVE_PROVIDERS, OPENAI_COMPATIBLE_PROVIDERS

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
