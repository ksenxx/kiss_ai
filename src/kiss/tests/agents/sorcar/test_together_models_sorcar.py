# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Live Together.ai model support through the SorcarAgent ``set_model`` tool.

Together serves 270+ models behind KISS's prefix-based routing
(``_TOGETHER_PREFIXES``: ``meta-llama/``, ``Qwen/``, ``mistralai/``,
``openai/gpt-oss``, ... → OpenAI-compatible transport at
``https://api.together.xyz/v1`` with ``TOGETHER_API_KEY``).  These tests
verify — with real Together API calls, driven through the **production
SorcarAgent hand-off path** rather than raw model objects — that:

* the ``model()`` factory routes Together-prefixed names to
  :class:`OpenAICompatibleModel` with the Together endpoint and key;
* live tool calling works end to end on a Together model (tool call →
  ``role="tool"`` result → final answer);
* the SorcarAgent ``set_model`` tool can switch INTO a Together model
  mid-conversation and back OUT to a native OpenAI model without losing
  history (regression hotspot: ``set_model`` must not carry the previous
  provider's ``base_url``/``api_key`` across the switch — api.together.xyz
  is a factory-default endpoint and must be re-derived by routing).

Serverless Together models occasionally return transient errors, so the
tests retry across a couple of cheap tool-capable candidates and skip only
if every candidate is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openai
import pytest

import kiss.core.vscode_config as vscode_config
from kiss.agents.sorcar import persistence as sorcar_persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.model_info import model as model_factory
from kiss.core.models.openai_compatible_model import (
    OpenAICompatibleBase,
    OpenAICompatibleModel,
)
from kiss.tests.conftest import (
    requires_openai_api_key,
    requires_together_api_key,
)
from kiss.tests.core.models.test_multihop_model_switching import (
    _SECRETS,
    _run_tool_turn,
    reveal_secret,
)

_TOGETHER_TOOL_MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai/gpt-oss-120b",
]


def _expected_transport_class(name: str) -> type[OpenAICompatibleBase]:
    """Return the adapter class the factory must build for *name*.

    Catalog entries carrying the live-verified ``use_responses_api`` flag
    are built as the v2 Responses adapter; everything else stays on the
    Chat Completions v1 adapter.
    """
    from kiss.core.models.model_info import MODEL_INFO
    from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

    info = MODEL_INFO.get(name)
    if info is not None and info.use_responses_api is True:
        return OpenAICompatibleModel2
    return OpenAICompatibleModel


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


@requires_together_api_key
@requires_openai_api_key
class TestSorcarTogetherLive:
    """Live Together tool calling and set_model hops via SorcarAgent."""

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

    def test_together_hop_via_set_model_tool(self) -> None:
        """gpt-4o -> Together model -> gpt-4o, each hop making a live tool
        call; the final model recalls both secrets, proving the production
        hand-off works with Together in both directions."""
        agent: Any = SorcarAgent("together-set-model-live")
        agent._use_web_tools = False
        tools = agent._get_tools()
        set_model = _find_tool(tools, "set_model")

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

        hop2_done = False
        upstream_errors: list[str] = []
        no_tool_call: list[str] = []
        for name in _TOGETHER_TOOL_MODELS:
            self._switch(agent, set_model, name)
            assert isinstance(agent.model, _expected_transport_class(name))
            assert "api.together.xyz" in agent.model.base_url, (
                f"set_model built {type(agent.model).__name__} with "
                f"base_url={agent.model.base_url!r} for a Together model"
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
            except (openai.RateLimitError, openai.APIStatusError) as e:
                upstream_errors.append(f"{name}: {e}")
            except AssertionError as e:
                # Live-model nondeterminism: the candidate answered in
                # plain text instead of calling the tool.  Try the next
                # candidate, mirroring the upstream-error retry above.
                no_tool_call.append(f"{name}: {e}")
        if not hop2_done:
            if len(no_tool_call) == len(_TOGETHER_TOOL_MODELS):
                # Every reachable candidate declined to call the tool:
                # that points at a real hand-off/schema bug, not flakiness.
                raise AssertionError(
                    "No Together candidate ever called reveal_secret "
                    f"after set_model hand-off: {no_tool_call}"
                )
            pytest.skip(
                "All candidate Together tool-capable models are unavailable "
                f"or flaked upstream: {upstream_errors + no_tool_call}"
            )

        self._switch(agent, set_model, "gpt-4o")
        assert isinstance(agent.model, _expected_transport_class("gpt-4o"))
        assert agent.model.base_url.startswith("https://api.openai.com"), (
            f"set_model kept base_url={agent.model.base_url!r} after "
            "switching back to a native OpenAI model"
        )

        agent.model.add_message_to_conversation(
            "user",
            "List every secret word you learned from the reveal_secret "
            "tool in this conversation, in order.",
        )
        final_text, _ = agent.model.generate()
        for index in (1, 2):
            assert _SECRETS[index] in final_text, (
                f"secret {index} ({_SECRETS[index]}) lost across the "
                f"Together hand-off; final answer: {final_text!r}"
            )
