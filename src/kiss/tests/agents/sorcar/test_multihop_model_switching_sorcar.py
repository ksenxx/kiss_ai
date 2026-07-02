# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Live multi-hop model switching through the SorcarAgent ``set_model`` tool.

``test_multihop_model_switching.py`` (in ``kiss.tests.core.models``) proves
the *model-level* conversation hand-off chain is lossless across five hops.
This module proves the same property through the **production Sorcar code
path**: the ``set_model`` tool built by :meth:`SorcarAgent._get_tools`, which
is what actually swaps models during a real Sorcar run.  That tool does more
than the raw hand-off — it reconstructs ``model_config`` (endpoint carry-over),
routes through the ``model()`` factory, re-initializes the provider client,
transfers the live conversation, and rebuilds the cached tools schema — so a
bug in any of those steps is invisible to the model-level test.

Two live tests (all three provider API keys required):

* ``TestSorcarSetModelMultiHopLive`` drives the agent's live model directly
  (the exact objects ``KISSAgent._execute_step`` uses) through
  chat (gpt-4o) -> OpenAI Responses delegation (gpt-5.5, reasoning_effort
  preserved) -> Anthropic -> Gemini -> back to gpt-5.5, with a real tool call
  on every hop, switching via the real ``set_model`` tool each time.  The
  final model must recall every secret revealed on every earlier hop.

* ``TestSorcarAgentRunMultiHopLive`` runs the full ``SorcarAgent.run`` loop
  (RelentlessAgent -> per-session KISSAgent executor) on a task that
  instructs the agent to alternate ``Read`` calls with ``set_model`` calls
  across three providers and then report all secrets in ``finish``.

Regression: ``set_model`` used to copy the old model's ``base_url``/
``api_key`` into ``model_config`` unconditionally.  Because the ``model()``
factory bypasses provider routing whenever ``model_config`` contains
``base_url``, switching *from* any OpenAI-compatible model *to* an
Anthropic/Gemini model built an ``OpenAICompatibleModel`` that pointed e.g.
``claude-haiku-4-5`` at ``api.openai.com`` — every subsequent LLM call
failed.  These tests fail loudly if that regresses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import kiss.agents.vscode.vscode_config as vscode_config
from kiss.agents.sorcar import persistence as sorcar_persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import model as model_factory
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.conftest import (
    requires_anthropic_api_key,
    requires_gemini_api_key,
    requires_openai_api_key,
)
from kiss.tests.core.models.test_multihop_model_switching import (
    _SECRETS,
    _run_tool_turn,
    reveal_secret,
)


def _find_tool(tools: list, name: str) -> Any:
    """Return the tool function named *name* from *tools*."""
    for t in tools:
        if callable(t) and t.__name__ == name:
            return t
    raise AssertionError(f"Tool {name!r} not found")


# ---------------------------------------------------------------------------
# Fixtures — isolate config and the sorcar DB so tests never touch real state.
# ---------------------------------------------------------------------------

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


@requires_openai_api_key
@requires_anthropic_api_key
@requires_gemini_api_key
class TestSorcarSetModelMultiHopLive:
    """Five-hop live chain switched exclusively via SorcarAgent's set_model."""

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
        # The cached tools schema must be rebuilt against the new model.
        assert agent._cached_tools_schema is not old_schema
        names = {
            entry["function"]["name"]
            for entry in agent._cached_tools_schema
            if isinstance(entry, dict) and "function" in entry
        }
        assert "reveal_secret" in names

    def test_five_hop_chain_via_set_model_tool(self) -> None:
        """gpt-4o -> gpt-5.5 (Responses) -> Claude -> Gemini -> gpt-5.5,
        each hop making a live tool call; the final model recalls all
        secrets, proving the production hand-off is lossless."""
        agent: Any = SorcarAgent("multihop-set-model-live")
        agent._use_web_tools = False
        tools = agent._get_tools()
        set_model = _find_tool(tools, "set_model")

        # Hop 1: live OpenAI Chat Completions model, exactly as a running
        # executor would hold it (real endpoint, real key via routing).
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
        assert isinstance(agent.model, OpenAICompatibleModel)
        _run_tool_turn(
            agent.model,
            "Call the reveal_secret tool with index 1, then state the "
            "secret word you received.",
            1,
        )

        # Hop 2: OpenAI reasoning model.  The factory auto-defaults
        # reasoning_effort from MODEL_INFO, and with tools attached the
        # request is delegated to /v1/responses — the effort must survive
        # the set_model config reconstruction.
        self._switch(agent, set_model, "gpt-5.5")
        assert isinstance(agent.model, OpenAICompatibleModel)
        assert agent.model.base_url.startswith("https://api.openai.com")
        assert (agent.model.model_config or {}).get("reasoning_effort") == "high"
        _run_tool_turn(
            agent.model,
            "Call the reveal_secret tool with index 2, then state the "
            "secret word you received.",
            2,
        )

        # Hop 3: Anthropic.  Regression hotspot: set_model must NOT carry
        # the OpenAI base_url into the factory config, otherwise this
        # builds an OpenAICompatibleModel pointing Claude at OpenAI.
        self._switch(agent, set_model, "claude-haiku-4-5")
        assert isinstance(agent.model, AnthropicModel), (
            f"set_model built {type(agent.model).__name__} for a Claude "
            "model — the old base_url leaked into the factory config"
        )
        _run_tool_turn(
            agent.model,
            "Call the reveal_secret tool with index 3, then state the "
            "secret word you received.",
            3,
        )

        # Hop 4: Gemini on top of chat + Responses + Anthropic history.
        self._switch(agent, set_model, "gemini-2.5-flash")
        assert isinstance(agent.model, GeminiModel)
        _run_tool_turn(
            agent.model,
            "Call the reveal_secret tool with index 4, then state the "
            "secret word you received.",
            4,
        )

        # Hop 5: back to the OpenAI reasoning model; tools must still work
        # on the four-provider history.
        self._switch(agent, set_model, "gpt-5.5")
        assert isinstance(agent.model, OpenAICompatibleModel)
        _run_tool_turn(
            agent.model,
            "Call the reveal_secret tool with index 1 one more time to "
            "double-check it, then state the secret word you received.",
            1,
        )

        # Semantic losslessness: the final model can only list all four
        # secrets if every hop's tool result survived every conversion.
        agent.model.add_message_to_conversation(
            "user",
            "List ALL the secret words that were revealed by the "
            "reveal_secret tool at any point in this conversation, in "
            "order of their index. Reply with the words only.",
        )
        content, _resp = agent.model.generate()
        for secret in _SECRETS.values():
            assert secret in content, (
                f"final model failed to recall {secret!r}; got: {content!r}"
            )

        # Structural losslessness in the handed-off conversation.
        serialized = json.dumps(agent.model.conversation, default=str)
        for secret in _SECRETS.values():
            assert secret in serialized


@requires_openai_api_key
@requires_anthropic_api_key
@requires_gemini_api_key
class TestSorcarAgentRunMultiHopLive:
    """Full SorcarAgent.run loop switching providers mid-task via set_model."""

    def test_agent_run_switches_models_and_recalls_secrets(
        self, tmp_path: Path
    ) -> None:
        """The agent reads a secret file, switches provider, reads the next,
        and must report all three secrets after two live hand-offs."""
        secrets = {
            1: "QUARTZ-PELICAN",
            2: "VELVET-COMPASS",
            3: "IGUANA-FURNACE",
        }
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        for i, word in secrets.items():
            (work_dir / f"secret{i}.txt").write_text(f"The secret word is {word}\n")

        prompt = (
            "Perform these steps EXACTLY in order. Do not skip, reorder, or "
            "add steps.\n"
            f"1. Use the Read tool to read {work_dir}/secret1.txt.\n"
            "2. Call set_model with model_name 'claude-haiku-4-5'.\n"
            f"3. Use the Read tool to read {work_dir}/secret2.txt.\n"
            "4. Call set_model with model_name 'gemini-2.5-flash'.\n"
            f"5. Use the Read tool to read {work_dir}/secret3.txt.\n"
            "6. Call finish with success=True and a summary that lists all "
            "three secret words exactly as written in the files.\n"
        )
        agent = SorcarAgent("multihop-run-live")
        result = agent.run(
            model_name="gpt-4o",
            prompt_template=prompt,
            work_dir=str(work_dir),
            max_steps=20,
            max_budget=5.0,
            max_sub_sessions=1,
            web_tools=False,
            verbose=False,
        )

        parsed = yaml.safe_load(result)
        assert parsed.get("success") is True, result
        for word in secrets.values():
            assert word in result, (
                f"secret {word!r} missing from agent result: {result!r}"
            )
        # The last set_model call must have stuck on the agent's bookkeeping.
        assert agent.model_name == "gemini-2.5-flash"
