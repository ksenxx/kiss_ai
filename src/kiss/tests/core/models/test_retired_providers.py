# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Providers the product no longer talks to: MiniMax, Together AI, Z.AI and Moonshot AI.

Their API keys are not settings, a key left in the environment configures
no provider, and the model names their own APIs and Together used
(``glm-4.7``, ``kimi-k3``, ``zai-org/GLM-5.2``, ``Qwen/Qwen3.7-Max``) are
unknown models.  Z.AI's and Moonshot's models stay reachable through
OpenRouter (``openrouter/z-ai/...``, ``openrouter/moonshotai/...``).

That the bundled catalog holds no entry of an unrouted name is checked in
``test_provider_registry_single_source.py``.
"""

from __future__ import annotations

import json

import pytest

from kiss.core import config as config_module
from kiss.core import user_config
from kiss.core.kiss_error import KISSError
from kiss.core.models import model_info

RETIRED_KEYS = ("MINIMAX_API_KEY", "TOGETHER_API_KEY", "ZAI_API_KEY", "MOONSHOT_API_KEY")

RETIRED_MODEL_NAMES = (
    "minimax-m2.5",
    "glm-4.7",
    "kimi-k3",
    "moonshot-v1-32k",
    "zai-org/GLM-5.2",
    "moonshotai/Kimi-K3",
    "Qwen/Qwen3.7-Max",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V4-Pro",
    "openai/gpt-oss-120b",
)


def test_a_retired_key_is_no_setting_and_configures_no_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in RETIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", config_module.Config())
    providers = model_info._configured_providers()
    available = model_info.get_available_models()
    default_model, fast_model = model_info.get_default_model(), model_info.get_fast_model()

    for key in RETIRED_KEYS:
        monkeypatch.setenv(key, "a-key-of-a-retired-provider")
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", config_module.Config())

    assert not set(RETIRED_KEYS) & set(config_module.Config.model_fields)
    assert not set(RETIRED_KEYS) & user_config.API_KEY_ENV_VARS
    assert not set(RETIRED_KEYS) & set(user_config.get_current_api_keys())
    assert model_info._configured_providers() == providers
    assert not {"MiniMax", "Together", "Z.AI", "Moonshot"} & set(providers)
    assert model_info.get_available_models() == available
    assert (model_info.get_default_model(), model_info.get_fast_model()) == (
        default_model,
        fast_model,
    )


@pytest.mark.parametrize("name", RETIRED_MODEL_NAMES)
def test_a_name_of_a_retired_route_is_an_unknown_model(name: str) -> None:
    assert model_info.get_model_provider(name) == "Unknown"
    assert name not in model_info.MODEL_INFO
    with pytest.raises(KISSError, match="Unknown model name"):
        model_info.model(name)


def test_the_catalog_keeps_the_openrouter_routes_of_zai_and_moonshot_but_none_of_minimax() -> None:
    catalog = json.loads(model_info.PACKAGE_MODEL_INFO_PATH.read_text(encoding="utf-8"))
    for prefix in ("openrouter/z-ai/", "openrouter/moonshotai/"):
        names = [name for name in catalog if name.startswith(prefix)]
        assert names, prefix
        assert {model_info.get_model_provider(name) for name in names} == {"OpenRouter"}
    assert not [name for name in catalog if "minimax" in name.lower()]
