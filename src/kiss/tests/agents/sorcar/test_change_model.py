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
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import kiss.core.user_config as user_config
from kiss.agents.sorcar import agent_dispatch
from kiss.agents.sorcar import persistence as sorcar_persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import config as config_module
from kiss.core.models import model_info
from kiss.core.models.model_info import model as model_factory
from kiss.core.models.openai_compatible_model import OpenAICompatibleBase, OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    ScriptedHandler,
    serve,
    text_body,
    tool_call_body,
    tool_calls_body,
)
from kiss.tests.conftest import clear_classification_cache, executor_of
from kiss.tests.persistence_helpers import close_db

OWN_ENDPOINT_MODEL = "own-endpoint-model"


class _DeletedInFlight(ScriptedHandler):
    """The settings panel deletes the custom model while its request is in flight."""

    def do_POST(self) -> None:  # noqa: N802
        for name in (OWN_ENDPOINT_MODEL, MODEL):  # whichever the test registered
            model_info.delete_custom_model(name)
        super().do_POST()


class _FirstRequestRefused(ScriptedHandler):
    """The endpoint refuses its first request for good: the model is not available."""

    def do_POST(self) -> None:  # noqa: N802
        if type(self).requests:
            return super().do_POST()
        length = int(self.headers.get("Content-Length", 0))
        type(self).requests.append(json.loads(self.rfile.read(length)))
        body = json.dumps({"error": {"message": "The model is not available"}}).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _refused(prompt: str) -> str:
    """A system-prompt hook that fails its run before the run's first session."""
    raise RuntimeError("no system prompt for this run")


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


def _openai_model(executor: Any) -> OpenAICompatibleBase:
    """Return *executor*'s live model, which must be an OpenAI-compatible one."""
    model = executor.model
    assert isinstance(model, OpenAICompatibleBase), type(model)
    return model


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

    def test_run_endpoint_stays_when_the_live_models_name_is_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The endpoint a run was launched with is the run's: registering the live model's
        name as a custom model while it runs does not make that endpoint the model's own,
        so the next model still inherits it and its key."""
        monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
        agent, tools = _make_agent()
        agent = _bootstrap_live_model(agent, "gpt-4o")  # fake-key at localhost:65535
        assert model_info.save_custom_model("gpt-4o", "http://localhost:65534/v1", "own") is None

        _find_tool(tools, "set_model")("gpt-4o-mini")

        # (the key is compared here, not by pytest: a failure must not print a real one)
        assert (agent.model.base_url, agent.model.api_key == "fake-key") == (
            "http://localhost:65535/v1", True,
        )

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

        # The same once the custom model's entry is deleted while it is the live model.
        set_model("s10s-scout-high")
        assert model_info.delete_custom_model("s10s-scout-high") is None
        set_model("gpt-5.5-mini")

        assert agent.model.base_url == "https://api.openai.com/v1"
        assert agent.model.api_key != "router-token"
        assert "extra_headers" not in agent.model.model_config
        # Nor is the default effort its adapter was built with, which the registry no longer names.
        assert "reasoning_effort" not in agent.model.model_config

    def test_an_explicit_null_effort_is_not_taken_for_a_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``reasoning_effort: None`` given for a model that has a default level is the caller's
        choice: the next model keeps it instead of getting a default of its own."""
        monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
        entry = {
            "context_length": 262144, "input_price_per_1M": 0.07, "output_price_per_1M": 0.15,
            "endpoint": "http://localhost:65534/v1", "thinking": "high",
        }
        (tmp_path / "MY_MODELS.json").write_text(json.dumps({"scout-a": entry, "scout-b": entry}))
        agent, tools = _make_agent()
        executor = executor_of(agent)
        executor.model = model_factory(
            "scout-a", model_config={"base_url": entry["endpoint"], "reasoning_effort": None},
        )
        executor.model_name = agent.model_name = "scout-a"

        _find_tool(tools, "set_model")("scout-b")

        assert executor.model.model_name == "scout-b"
        assert executor.model.model_config["reasoning_effort"] is None

    def test_extra_headers_stay_with_their_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``extra_headers`` (credentials among them) are the endpoint's they were given
        for: a switch on that endpoint keeps them, a switch to another provider drops
        them, so its SDK client authenticates with its own key and is not handed the
        old endpoint's token (audit 2026-11-21, S14 F1)."""
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "openrouter-key")
        headers = {"Authorization": "Bearer provider-a-token", "X-Team": "s10s"}
        agent, tools = _make_agent()
        executor = executor_of(agent)
        executor.model = model_factory("gpt-4o", model_config={
            "base_url": "https://api.openai.com/v1", "api_key": "task-key",
            "extra_headers": headers,
        })
        executor.model_name = agent.model_name = "gpt-4o"
        set_model = _find_tool(tools, "set_model")

        set_model("gpt-4o-mini")

        assert _openai_model(executor).api_key == "task-key"
        assert executor.model.model_config["extra_headers"] == headers
        assert agent.model_config["extra_headers"] == headers

        set_model("openrouter/openai/gpt-4o-mini")

        assert _openai_model(executor).base_url == "https://openrouter.ai/api/v1"
        assert "extra_headers" not in executor.model.model_config
        assert "extra_headers" not in agent.model_config
        client = _openai_model(executor)._build_client()
        sent = {**client.auth_headers, **client.default_headers}  # the SDK's own precedence
        assert sent["Authorization"] == "Bearer openrouter-key" and "X-Team" not in sent

    def test_a_native_models_headers_stay_within_its_provider(self) -> None:
        """A native adapter has no ``base_url``; its endpoint is its provider's: Claude to
        Claude keeps the run's ``extra_headers``, Claude to an OpenAI model drops them."""
        pytest.importorskip("anthropic")
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        agent, tools = _make_agent()
        executor = executor_of(agent)
        executor.model = model_factory("claude-sonnet-4-5", model_config={"extra_headers": headers})
        executor.model_name = agent.model_name = "claude-sonnet-4-5"
        set_model = _find_tool(tools, "set_model")

        set_model("claude-haiku-4-5")

        assert executor.model.model_config["extra_headers"] == headers

        set_model("gpt-4o-mini")

        assert _openai_model(executor).base_url == "https://api.openai.com/v1"
        assert "extra_headers" not in executor.model.model_config

    def test_explicit_responses_transport_stays_on_the_same_endpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit ``use_responses_api`` is the endpoint's it was given for (a gateway
        may serve nothing else): a switch on that endpoint, to a catalog name or to a
        ``MY_MODELS.json`` model registered at the same URL, keeps the Responses
        transport, for the continuation's executor too; a switch to another endpoint
        drops it (audit 2026-11-21, S14 F2)."""
        monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
        fields = {"context_length": 262144, "input_price_per_1M": 0.07, "output_price_per_1M": 0.15}
        gateway = {"base_url": "http://localhost:65535/v1", "api_key": "fake-key"}
        (tmp_path / "MY_MODELS.json").write_text(json.dumps({
            "same-gateway": {**fields, "endpoint": gateway["base_url"] + "/", "api_key": "own-key"},
            "other-gateway": {**fields, "endpoint": "http://localhost:65534/v1"},
        }))
        agent, tools = _make_agent()
        executor = executor_of(agent)
        launch = {**gateway, "use_responses_api": True}
        executor.model = model_factory("gpt-4o", model_config=launch)
        executor.model_name = agent.model_name = "gpt-4o"
        set_model = _find_tool(tools, "set_model")

        set_model("gpt-4o-mini")

        assert type(executor.model) is OpenAICompatibleModel2
        assert agent.model_config == launch
        continuation = model_factory(agent.model_name, model_config=agent._executor_model_config())
        assert type(continuation) is OpenAICompatibleModel2
        assert (continuation.base_url, continuation.api_key) == tuple(gateway.values())

        set_model("same-gateway")

        assert type(executor.model) is OpenAICompatibleModel2
        assert executor.model.api_key == "own-key"

        set_model("other-gateway")

        assert type(executor.model) is OpenAICompatibleModel
        assert "use_responses_api" not in agent.model_config

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

    @pytest.mark.parametrize(
        ("name", "classified", "configured"),
        [
            (OWN_ENDPOINT_MODEL, False, True),
            (MODEL, True, True),
            (OWN_ENDPOINT_MODEL, False, False),
        ],
        ids=["first-request", "classifier", "launched-by-name"],
    )
    def test_a_custom_model_deleted_mid_run_leaves_its_endpoint_to_no_other_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        name: str, classified: bool, configured: bool,
    ) -> None:
        """A run launched on a ``MY_MODELS.json`` model as the task runner launches it (on the
        entry's endpoint, key and headers) or by its name alone (the run then takes the three
        from the entry itself), whose entry is deleted while a request is in
        flight (the run's first; or the one of the classifier that goes ahead of the run, on
        a name the catalog lists too, which a run outlives): the model the run then switches
        to, and the next session's executor built from the agent's ``model_config``, get none
        of the three, as with the entry in place."""
        monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "0" if classified else "1")
        monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
        user_config.save_config({"classify_with_decisions": False})  # the run's model classifies
        clear_classification_cache()
        switch_and_finish = tool_calls_body([
            ("set_model", {"model_name": "gpt-5.5-mini"}),
            ("finish", {"success": True, "is_continue": False, "summary_in_html": "<p>Done.</p>"}),
        ], 500)
        script = [text_body('{"is_simple": true}')] * classified + [switch_and_finish]
        with serve(script, _DeletedInFlight) as (custom, requests):
            assert model_info.save_custom_model(name, custom, "custom-key", "X-Team: s10s") is None
            launch_config = model_info.custom_model_config(name)
            assert launch_config == {
                "base_url": custom, "api_key": "custom-key", "extra_headers": {"X-Team": "s10s"},
            }
            agent = SorcarAgent("test-deleted-custom-model")
            agent.run(
                model_name=name, model_config=launch_config if configured else None,
                prompt_template="Call set_model, then finish.",
                max_sub_sessions=1, max_budget=1, max_steps=5,
                # Only a run that can edit files is classified.
                tool_profile="" if classified else "control", web_tools=False, use_memory=False,
                work_dir=str(tmp_path), verbose=False,
            )
        assert [body["model"] for body in requests] == [name] * len(script)
        assert model_info.custom_model_config(name) is None
        assert agent.model_name == "gpt-5.5-mini"
        assert agent.model_config == {}

    def test_a_gateway_run_keeps_its_gateway_though_its_model_is_registered_elsewhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run sent to a gateway while a ``MY_MODELS.json`` entry gives its model an endpoint
        of its own has the gateway as the run's: the model it switches to is asked there.
        Whose endpoint a run has is said for that run alone: an earlier run of the agent on
        the entry's endpoint, which failed before its first session, has no say in the next."""
        monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
        monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
        fields = {"context_length": 262144, "input_price_per_1M": 0.1, "output_price_per_1M": 0.4}
        (tmp_path / "MY_MODELS.json").write_text(json.dumps({
            OWN_ENDPOINT_MODEL: {**fields, "endpoint": "http://127.0.0.1:9/v1"},
            "model-without-an-endpoint": fields,
        }))
        switch_body = tool_call_body("set_model", {"model_name": "model-without-an-endpoint"}, 500)
        done_body = tool_call_body("finish", {
            "success": True, "is_continue": False, "summary_in_html": "<p>Done.</p>",
        }, 500)
        agent = SorcarAgent("test-gateway-run")
        run: dict[str, Any] = {
            "model_name": OWN_ENDPOINT_MODEL, "prompt_template": "Call set_model, then finish.",
            "max_sub_sessions": 1, "max_budget": 1, "max_steps": 5, "tool_profile": "control",
            "web_tools": False, "use_memory": False, "work_dir": str(tmp_path), "verbose": False,
        }
        assert "no system prompt for this run" in agent.run(**run, system_prompt_hook=_refused)
        with serve([switch_body, done_body]) as (gateway, requests):
            agent.run(**run, model_config={"base_url": gateway, "api_key": "gateway-key"})
        assert [body["model"] for body in requests] == [
            OWN_ENDPOINT_MODEL, "model-without-an-endpoint",
        ]

    def test_a_fallback_after_a_switch_is_built_from_the_switched_models_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that switches to a ``MY_MODELS.json`` model whose endpoint then refuses it for
        good fails over to the fallback that entry declares where the entry points (its
        endpoint, as a run launched on that model does), not at the launch endpoint the run
        left with the switch."""
        monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
        monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
        fields = {"context_length": 262144, "input_price_per_1M": 0.1, "output_price_per_1M": 0.4}
        switch_body = tool_call_body("set_model", {"model_name": "switched-model"}, 500)
        done_body = tool_call_body("finish", {
            "success": True, "is_continue": False, "summary_in_html": "<p>Done.</p>",
        }, 500)
        with serve([switch_body, done_body]) as (launch, launch_requests):
            with serve([done_body], _FirstRequestRefused) as (custom, custom_requests):
                (tmp_path / "MY_MODELS.json").write_text(json.dumps({
                    "switched-model": {
                        **fields, "endpoint": custom, "api_key": "custom-key",
                        "fallback": "its-fallback",
                    },
                    "its-fallback": fields,
                }))
                SorcarAgent("test-switch-then-fallback").run(
                    model_name=MODEL,
                    model_config={"base_url": launch, "api_key": "launch-key"},
                    prompt_template="Call set_model, then finish.",
                    max_sub_sessions=1, max_budget=1, max_steps=5,
                    tool_profile="control", web_tools=False, use_memory=False,
                    work_dir=str(tmp_path), verbose=False,
                )
        assert [body["model"] for body in launch_requests] == [MODEL]
        assert [body["model"] for body in custom_requests] == ["switched-model", "its-fallback"]


def _inherit_until(agent: Any, done: threading.Event, seen: set[tuple[Any, ...]]) -> None:
    """Record what a sub-task dispatched now would inherit from *agent*, until *done*."""
    while not done.is_set():
        got = agent_dispatch.inherit_from_parent(agent, "", 1.0, agent_dispatch.RunOptions())
        config = got.options.model_config or {}
        seen.add((got.model_name, config.get("base_url"), got.model_owns_endpoint))


def test_a_dispatch_racing_a_switch_inherits_one_models_name_endpoint_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A background ``run_agent`` job reads its caller's model on its own thread while the
    caller's ``set_model`` replaces it: the sub-task gets one model's name, endpoint and
    record, never a name with the other model's endpoint or with none."""
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    fields = {"context_length": 262144, "input_price_per_1M": 0.07, "output_price_per_1M": 0.15}
    endpoints = {"model-a": "http://localhost:65531/v1", "model-b": "http://localhost:65532/v1"}
    (tmp_path / "MY_MODELS.json").write_text(json.dumps({
        name: {**fields, "endpoint": endpoint, "api_key": "its-key"}
        for name, endpoint in endpoints.items()
    }))
    agent, tools = _make_agent()
    _bootstrap_live_model(agent, "model-b")
    set_model = _find_tool(tools, "set_model")
    set_model("model-a")
    one_model = {(name, endpoint, True) for name, endpoint in endpoints.items()}
    seen: set[tuple[Any, ...]] = set()
    done = threading.Event()
    reader = threading.Thread(target=_inherit_until, args=(agent, done, seen))
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # the switch's stores are a few bytecodes apart
    reader.start()
    try:
        for _ in range(300):
            set_model("model-b")
            set_model("model-a")
            if not seen <= one_model:
                break
    finally:
        done.set()
        reader.join()
        sys.setswitchinterval(interval)
    assert seen <= one_model, seen - one_model
