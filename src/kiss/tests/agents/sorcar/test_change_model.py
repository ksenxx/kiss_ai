# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the ``set_model`` tool on ``SorcarAgent``.

``set_model(model_name)`` swaps the live LLM model instance of the
agent's executor (the :class:`KISSAgent` a run holds in
``_current_executor``) in place so the very next LLM call goes to the
changed model.  These tests verify:

* The tool appears in the agent's tool list.
* Calling it without a live model just updates ``self.model_name``.
* Calling it with a live model swaps the executor's ``model`` to a
  fresh instance for the new model name while preserving the existing
  conversation history, ``model_config``, and callbacks.
* The cached tools schema is rebuilt so it stays consistent with the
  (possibly different) provider of the new model.
* The same-name no-op short-circuits.
* The model picker default is NOT persisted or otherwise changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import kiss.core.user_config as user_config
from kiss.agents.sorcar import persistence as sorcar_persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models import model_info
from kiss.core.models.model_info import model as model_factory
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve, tool_call_body
from kiss.tests.conftest import executor_of
from kiss.tests.persistence_helpers import close_db


def _find_tool(tools: list, name: str) -> Any:
    """Return the tool function named *name* from *tools*."""
    for t in tools:
        if callable(t) and t.__name__ == name:
            return t
    raise AssertionError(
        f"Tool {name!r} not found in "
        f"{[getattr(t, '__name__', None) for t in tools if callable(t)]}"
    )


def _make_agent() -> tuple[Any, list]:
    """Build a SorcarAgent with web tools disabled and return ``(agent, tools)``.

    The return is typed ``Any`` so tests can freely poke at runtime-
    attached attributes like ``model`` and ``_cached_tools_schema``
    that are set inside ``_reset`` rather than ``__init__``.
    """
    agent = SorcarAgent("test-change-model")
    agent._use_web_tools = False
    tools = agent._get_tools()
    return agent, tools


def _bootstrap_live_model(agent: Any, model_name: str) -> Any:
    """Give *agent* an executor holding a live ``OpenAICompatibleModel``; return it.

    A run keeps the live model on the :class:`KISSAgent` executor in
    ``agent._current_executor``, which ``set_model`` swaps.  The
    OpenAI-compatible model's constructor only stores the ``base_url``
    and ``api_key`` — it does not make any network request — so we can
    build one synchronously inside a test with a fake endpoint and a
    fake key.
    """
    config = {"base_url": "http://localhost:65535/v1", "api_key": "fake-key"}
    executor = executor_of(agent)
    executor.model = model_factory(model_name, model_config=config)
    executor.model_name = model_name
    agent.model_name = model_name  # as ``run(model_name=...)`` sets it
    return executor


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path) -> Any:
    """Redirect user_config writes to a temp directory."""
    orig_dir = user_config.CONFIG_DIR
    orig_path = user_config.CONFIG_PATH
    test_dir = tmp_path / ".kiss"
    test_dir.mkdir()
    user_config.CONFIG_DIR = test_dir
    user_config.CONFIG_PATH = test_dir / "config.json"
    yield
    user_config.CONFIG_DIR = orig_dir
    user_config.CONFIG_PATH = orig_path


@pytest.fixture(autouse=True)
def _isolate_sorcar_db(tmp_path: Path) -> Any:
    """Redirect the sorcar SQLite DB to a temp file for each test."""
    orig_kiss_dir = sorcar_persistence._KISS_DIR
    orig_db_path = sorcar_persistence._DB_PATH
    test_dir = tmp_path / "sorcar_db_dir"
    test_dir.mkdir()
    sorcar_persistence._KISS_DIR = test_dir
    sorcar_persistence._DB_PATH = test_dir / "history.db"
    close_db()
    yield
    close_db()
    sorcar_persistence._KISS_DIR = orig_kiss_dir
    sorcar_persistence._DB_PATH = orig_db_path


class TestChangeModelToolPresence:
    """``set_model`` appears in the agent's tool list."""

    def test_in_tools(self) -> None:
        _agent, tools = _make_agent()
        names = [t.__name__ for t in tools if callable(t)]
        assert "set_model" in names

    def test_signature_takes_model_name_string(self) -> None:
        import inspect

        _agent, tools = _make_agent()
        set_model = _find_tool(tools, "set_model")
        sig = inspect.signature(set_model)
        params = list(sig.parameters.values())
        assert len(params) == 1
        assert params[0].name == "model_name"
        ann = params[0].annotation
        assert ann is str or ann == "str"


class TestChangeModelNoLiveModel:
    """When ``self.model`` is unset, the tool just updates ``model_name``."""

    def test_updates_model_name(self) -> None:
        agent, tools = _make_agent()
        assert getattr(agent, "model", None) is None
        set_model = _find_tool(tools, "set_model")

        result = set_model("gpt-4o-mini")
        assert "gpt-4o-mini" in result
        assert agent.model_name == "gpt-4o-mini"

    def test_does_not_persist_or_change_model_picker_default(self) -> None:
        """Internal set_model must not change the user's picker default."""
        from kiss.agents.sorcar.persistence import _load_last_model, _save_last_model

        _save_last_model("claude-opus-4-7")
        agent, tools = _make_agent()
        set_model = _find_tool(tools, "set_model")

        set_model("gemini-2.5-pro")

        assert agent.model_name == "gemini-2.5-pro"
        assert _load_last_model() == "claude-opus-4-7"


class TestChangeModelLiveSwap:
    """With a live model, the tool replaces ``self.model`` with a fresh instance."""

    def test_swaps_to_new_instance(self) -> None:
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")
        set_model = _find_tool(tools, "set_model")

        old_model_obj = agent.model
        result = set_model("model-b")

        assert "model-b" in result
        assert agent.model is not old_model_obj
        assert agent.model.model_name == "model-b"
        assert agent.model_name == "model-b"

    def test_conversation_preserved(self) -> None:
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")
        agent.model.conversation = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        agent.model.usage_info_for_messages = "tokens=42"
        set_model = _find_tool(tools, "set_model")

        set_model("model-b")

        assert agent.model.conversation == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        assert agent.model.usage_info_for_messages == "tokens=42"

    def test_model_config_preserved(self) -> None:
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")
        original_base_url = getattr(agent.model, "base_url", None)
        original_api_key = getattr(agent.model, "api_key", None)
        set_model = _find_tool(tools, "set_model")

        set_model("model-b")

        assert getattr(agent.model, "base_url", None) == original_base_url
        assert getattr(agent.model, "api_key", None) == original_api_key

    def test_custom_endpoint_model_runs_against_its_own_endpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A switch to a ``MY_MODELS.json`` model with its own endpoint, key and headers
        uses them (as the task runner starts such a model) instead of carrying the
        launch endpoint or its key over, even within the Claude family; the entries
        are registered after the catalog was loaded, so a switch between two of them
        drops the old model's default effort; a switch away from one routes the new
        model to its own provider, as a sub-agent is (nothing of the custom model's
        endpoint, key or headers is carried)."""
        monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
        fields = {"context_length": 262144, "input_price_per_1M": 0.07, "output_price_per_1M": 0.15}
        router = {"endpoint": "http://localhost:65534/v1", "api_key": "router-token"}
        (tmp_path / "MY_MODELS.json").write_text(
            json.dumps(
                {
                    "claude-via-router": {**fields, **router},
                    "s10s-scout-high": {
                        **fields, **router, "thinking": "high", "headers": "X-Team: s10s",
                    },
                    "s10s-athena": {**fields, **router},
                }
            )
        )
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "claude-old")  # fake-key at localhost:65535
        set_model = _find_tool(tools, "set_model")

        set_model("claude-via-router")

        assert agent.model.base_url == "http://localhost:65534/v1"
        assert agent.model.api_key == "router-token"

        set_model("s10s-scout-high")

        assert agent.model.model_name == "s10s-scout-high"
        assert agent.model.base_url == "http://localhost:65534/v1"
        assert agent.model.api_key == "router-token"
        assert agent.model.model_config["extra_headers"] == {"X-Team": "s10s"}
        assert agent.model.model_config["reasoning_effort"] == "high"

        set_model("s10s-athena")

        assert "reasoning_effort" not in agent.model.model_config

        set_model("gpt-5.5-mini")

        assert agent.model.base_url == "https://api.openai.com/v1"
        assert agent.model.api_key != "router-token"
        assert "extra_headers" not in agent.model.model_config

    def test_callbacks_preserved(self) -> None:
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")
        tokens: list[str] = []

        def token_cb(t: str) -> None:
            tokens.append(t)

        agent.model.token_callback = token_cb
        set_model = _find_tool(tools, "set_model")

        set_model("model-b")
        assert agent.model.token_callback is token_cb

    def test_same_name_is_noop(self) -> None:
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")
        set_model = _find_tool(tools, "set_model")
        original = agent.model

        result = set_model("model-a")
        assert "already" in result.lower() or "no change" in result.lower()
        assert agent.model is original

    def test_does_not_persist_or_change_model_picker_default(self) -> None:
        """Swapping a live model must not change the user's picker default."""
        from kiss.agents.sorcar.persistence import _load_last_model, _save_last_model

        _save_last_model("claude-opus-4-7")
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")
        set_model = _find_tool(tools, "set_model")

        set_model("model-b")

        assert agent.model.model_name == "model-b"
        assert agent.model_name == "model-b"
        assert _load_last_model() == "claude-opus-4-7"


class TestChangeModelToolSchemaCache:
    """The cached tools schema is invalidated/rebuilt after a swap."""

    def test_cached_schema_rebuilt_when_function_map_populated(self) -> None:
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")

        def dummy_tool(x: str) -> str:
            """A dummy tool used to populate the schema cache."""
            return x

        agent.function_map = {"dummy_tool": dummy_tool}
        agent._cached_tools_schema = agent.model._build_openai_tools_schema(
            agent.function_map,
        )
        original_schema = agent._cached_tools_schema

        set_model = _find_tool(tools, "set_model")
        set_model("model-b")

        assert agent._cached_tools_schema is not None
        schema = agent._cached_tools_schema
        names = {
            entry["function"]["name"] for entry in schema
            if isinstance(entry, dict) and "function" in entry
        }
        assert "dummy_tool" in names
        assert agent._cached_tools_schema is not original_schema

    def test_no_function_map_no_crash(self) -> None:
        """Calling set_model before tools are wired up must not crash."""
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")
        agent.function_map = {}
        agent._cached_tools_schema = None  # type: ignore[assignment]
        set_model = _find_tool(tools, "set_model")

        set_model("model-b")
        assert agent._cached_tools_schema is None


class TestNextCallTargetsNewModel:
    """After set_model, ``self.model`` (which generate() targets) is the new one."""

    def test_model_instance_targets_new_name(self) -> None:
        """The instance the agent will call ``generate()`` on uses the new name.

        ``KISSAgent._execute_step`` invokes ``self.model.generate_*`` —
        so verifying ``self.model.model_name`` is the new name proves
        that the next LLM call would target the new model.
        """
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")
        set_model = _find_tool(tools, "set_model")

        set_model("model-c")

        assert agent.model.model_name == "model-c"
        assert agent.model_name == "model-c"


class TestPinnedModelDaemon:
    """A daemon that pins its tasks' model (a benchmark daemon) refuses ``set_model``."""

    def test_refused_and_model_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kiss.agents.sorcar.agent_dispatch import PIN_TASK_MODEL_ENV

        monkeypatch.setenv(PIN_TASK_MODEL_ENV, "1")
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "model-a")
        set_model = _find_tool(tools, "set_model")

        result = set_model("model-b")

        assert result.startswith("Refused") and "model-a" in result
        assert agent.model_name == "model-a" and agent.model.model_name == "model-a"
        # ``0`` and an empty value mean "not pinned": the switch goes through again
        monkeypatch.setenv(PIN_TASK_MODEL_ENV, "0")
        assert "model-b" in set_model("model-b")
        assert agent.model_name == "model-b"


class TestSwitchSurvivesContinuation:
    """The next session's executor keeps the routing a switch computed."""

    def test_custom_endpoint_switch_routes_the_continuation_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two sessions, two endpoints: after a switch to a ``MY_MODELS.json`` model the
        continuation's executor, built from the agent's ``model_config``, calls the
        custom endpoint, not the launch endpoint with the new model's name
        (audit 2026-11-04, S2 F1).  The launch config is not carried as the
        custom model's own, and the executor's per-session defaults are not
        frozen into the agent's config."""
        monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
        monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
        switched = "custom-endpoint-model"
        continue_body = tool_call_body("finish", {
            "success": False, "is_continue": True, "summary_in_html": "<p>Continue.</p>",
        }, 500)
        done_body = tool_call_body("finish", {
            "success": True, "is_continue": False, "summary_in_html": "<p>Done.</p>",
        }, 500)
        switch_body = tool_call_body("set_model", {"model_name": switched}, 500)
        with serve([switch_body, done_body]) as (launch, launch_requests):
            with serve([continue_body, done_body]) as (custom, custom_requests):
                (tmp_path / "MY_MODELS.json").write_text(json.dumps({switched: {
                    "endpoint": custom, "api_key": "custom-key", "context_length": 262144,
                    "input_price_per_1M": 0.1, "output_price_per_1M": 0.4,
                }}))
                agent = SorcarAgent("test-switch-continuation")
                agent.run(
                    model_name=MODEL,
                    model_config={"base_url": launch, "api_key": "launch-key"},
                    prompt_template="Call set_model, continue once, then finish.",
                    max_sub_sessions=2, max_budget=1, max_steps=10,
                    tool_profile="control", web_tools=False, use_memory=False,
                    work_dir=str(tmp_path), verbose=False,
                )
        assert [body["model"] for body in launch_requests] == [MODEL]
        assert [body["model"] for body in custom_requests] == [switched, switched]
        assert agent.model_name == switched
        assert agent.model_config == {"base_url": custom, "api_key": "custom-key"}
