# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Live OpenRouter model support through the SorcarAgent ``set_model`` tool.

OpenRouter exposes hundreds of models behind KISS's ``openrouter/<provider>/
<id>`` routing prefix (OpenAI-compatible transport, ``https://openrouter.ai/
api/v1``).  These tests verify — with real OpenRouter API calls, driven
through the **production SorcarAgent hand-off path** rather than raw model
objects — that:

* the ``model()`` factory routes ``openrouter/...`` names (including
  non-catalog ``:free`` variants) to :class:`OpenAICompatibleModel` with the
  OpenRouter endpoint and key;
* live tool calling works end to end on an OpenRouter model (tool call →
  ``role="tool"`` result → final answer);
* the SorcarAgent ``set_model`` tool can switch INTO an OpenRouter model
  mid-conversation and back OUT to a native OpenAI model without losing
  history (regression hotspot: ``set_model`` must not carry the previous
  provider's ``base_url``/``api_key`` across the switch — openrouter.ai is
  a factory-default endpoint and must be re-derived by routing).

The OpenRouter account may only have free-tier credit, so the tests use
``:free`` models with tool support and retry across several candidates on
upstream 429 rate limits, skipping only if every candidate is saturated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openai
import pytest

import kiss.server.vscode_config as vscode_config
from kiss.agents.sorcar import persistence as sorcar_persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.model_info import model as model_factory
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.conftest import (
    requires_openai_api_key,
    requires_openrouter_api_key,
)
from kiss.tests.core.models.test_multihop_model_switching import (
    _SECRETS,
    _run_tool_turn,
    reveal_secret,
)

# Free OpenRouter models that support tool calling (checked via
# https://openrouter.ai/api/v1/models ``supported_parameters``).  Free-tier
# upstreams rate-limit aggressively, so several candidates are tried.
_FREE_TOOL_MODELS = [
    "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/nvidia/nemotron-nano-9b-v2:free",
    "openrouter/openai/gpt-oss-120b:free",
]


def _find_tool(tools: list, name: str) -> Any:
    """Return the tool function named *name* from *tools*."""
    for t in tools:
        if callable(t) and t.__name__ == name:
            return t
    raise AssertionError(f"Tool {name!r} not found")


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path) -> Any:
    """Redirect vscode_config writes to a temp directory."""
    orig_dir = vscode_config.CONFIG_DIR
    orig_path = vscode_config.CONFIG_PATH
    test_dir = tmp_path / ".kiss"
    test_dir.mkdir()
    vscode_config.CONFIG_DIR = test_dir
    vscode_config.CONFIG_PATH = test_dir / "config.json"
    yield
    vscode_config.CONFIG_DIR = orig_dir
    vscode_config.CONFIG_PATH = orig_path


@pytest.fixture(autouse=True)
def _isolate_sorcar_db(tmp_path: Path) -> Any:
    """Redirect the sorcar SQLite DB to a temp file for each test."""
    orig_kiss_dir = sorcar_persistence._KISS_DIR
    orig_db_path = sorcar_persistence._DB_PATH
    test_dir = tmp_path / "sorcar_db_dir"
    test_dir.mkdir()
    sorcar_persistence._KISS_DIR = test_dir
    sorcar_persistence._DB_PATH = test_dir / "sorcar.db"
    sorcar_persistence._close_db()
    yield
    sorcar_persistence._close_db()
    sorcar_persistence._KISS_DIR = orig_kiss_dir
    sorcar_persistence._DB_PATH = orig_db_path


@requires_openrouter_api_key
@requires_openai_api_key
class TestSorcarOpenRouterLive:
    """Live OpenRouter tool calling and set_model hops via SorcarAgent."""

    def _switch(self, agent: Any, set_model: Any, model_name: str) -> None:
        """Call the real ``set_model`` tool and verify the swap landed.

        Args:
            agent: The SorcarAgent whose live model is being swapped.
            set_model: The ``set_model`` tool from ``agent._get_tools()``.
            model_name: The model to switch to.
        """
        old_schema = agent._cached_tools_schema
        result = set_model(model_name)
        assert f"to {model_name}" in result, result
        assert agent.model_name == model_name
        assert agent.model.model_name == model_name
        assert agent._cached_tools_schema is not old_schema
        names = {
            entry["function"]["name"]
            for entry in agent._cached_tools_schema
            if isinstance(entry, dict) and "function" in entry
        }
        assert "reveal_secret" in names

    def test_openrouter_hop_via_set_model_tool(self) -> None:
        """gpt-4o -> openrouter ``:free`` model -> gpt-4o, each hop making a
        live tool call; the final model recalls both secrets, proving the
        production hand-off works with OpenRouter in both directions."""
        agent: Any = SorcarAgent("openrouter-set-model-live")
        agent._use_web_tools = False
        tools = agent._get_tools()
        set_model = _find_tool(tools, "set_model")

        # Hop 1: live OpenAI model, exactly as a running executor holds it.
        agent.model = model_factory("gpt-4o")
        agent.model_name = "gpt-4o"
        agent.model.initialize(
            "We will play a memory game across this conversation. "
            "Follow each instruction exactly."
        )
        agent.function_map = {"reveal_secret": reveal_secret}
        agent._cached_tools_schema = agent.model._build_openai_tools_schema(
            agent.function_map
        )
        _run_tool_turn(
            agent.model,
            "Call the reveal_secret tool with index 1, then state the "
            "secret word you received.",
            1,
        )

        # Hop 2: switch INTO an OpenRouter model via the production
        # set_model tool.  Regression hotspot: the OpenAI endpoint/key must
        # NOT be carried over — routing must re-derive openrouter.ai and
        # OPENROUTER_API_KEY.  Free-tier upstreams 429 often, so several
        # candidate models are tried before skipping.
        hop2_done = False
        rate_limit_errors: list[str] = []
        for name in _FREE_TOOL_MODELS:
            self._switch(agent, set_model, name)
            assert isinstance(agent.model, OpenAICompatibleModel)
            assert "openrouter.ai" in agent.model.base_url, (
                f"set_model built {type(agent.model).__name__} with "
                f"base_url={agent.model.base_url!r} for an OpenRouter model"
            )
            try:
                _run_tool_turn(
                    agent.model,
                    "Call the reveal_secret tool with index 2, then state "
                    "the secret word you received.",
                    2,
                )
                hop2_done = True
                break
            except openai.RateLimitError as e:
                rate_limit_errors.append(f"{name}: {e}")
        if not hop2_done:
            pytest.skip(
                "All free OpenRouter tool-capable models are rate-limited "
                f"upstream: {rate_limit_errors}"
            )

        # Hop 3: switch back OUT to a native OpenAI model.  The OpenRouter
        # endpoint/key must not leak into the OpenAI model either.
        self._switch(agent, set_model, "gpt-4o")
        assert isinstance(agent.model, OpenAICompatibleModel)
        assert agent.model.base_url.startswith("https://api.openai.com"), (
            f"set_model kept base_url={agent.model.base_url!r} after "
            "switching back to a native OpenAI model"
        )

        # The full history — including the OpenRouter turn — must survive.
        agent.model.add_message_to_conversation(
            "user",
            "List every secret word you learned from the reveal_secret "
            "tool in this conversation, in order.",
        )
        final_text, _ = agent.model.generate()
        for index in (1, 2):
            assert _SECRETS[index] in final_text, (
                f"secret {index} ({_SECRETS[index]}) lost across the "
                f"OpenRouter hand-off; final answer: {final_text!r}"
            )
