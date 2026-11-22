# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the pre-run task classifier calls the run's own endpoint.

A settings-panel custom model (a ``~/.kiss/MY_MODELS.json`` entry with
an ``endpoint``) runs against ITS OWN endpoint / key: the task runner
resolves ``custom_model_config(model)`` for the run.  The classifier
that decides the run's worktree mode BEFORE the run used to be handed
a different configuration — the global config's, resolved a second
time, without the custom endpoint — so a custom-endpoint model was
classified by name against the default provider, which fails soft (no
worktree demotion) or waits out the classifier's stall timeout.

``_run_task_inner`` now resolves the model configuration once, and the
classifier and the run share it.

Harness: an isolated ``KISS_HOME`` whose ``MY_MODELS.json`` points a
bundled model name at a local stand-in OpenAI-compatible endpoint
(``StandInModelServer``) that records every request; a real
``AgentServer`` runs the task through ``_cmd_run`` with the real
``ChatSorcarAgent`` the server creates.  The stand-in answers the
classifier's request (no ``tools``) with a verdict and the agent's
first request (``tools`` attached) with a ``finish`` call, so the
test sees BOTH model calls of the run arrive at the custom endpoint.
No mocks or patches.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.headless import run_sea
from kiss.agents.sorcar.merge_conflict_resolver import run_merge_sea
from kiss.agents.sorcar.persistence import _add_task
from kiss.core.models import model_info
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.server.task_update import run_task_update_sea
from kiss.tests.conftest import clear_classification_cache, executor_of
from kiss.tests.server.conftest import clear_tab_states
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    tool_call_response,
)

_TAB = "tab-a1005-classifier-endpoint"
_WAIT_S = 90.0
_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"
_SUMMARY = "<p>audit1005 run finished at the custom endpoint</p>"
# A registry entry without an endpoint: the model runs only where a run's endpoint is handed to it.
_NO_ENDPOINT_MODEL = "audit1005-model-without-an-endpoint"
# A model only the registry knows: it runs nowhere but at its entry's endpoint.
_OWN_ENDPOINT_MODEL = "audit1005-model-with-its-own-endpoint"
_SWITCH_TASK = "switch the model, then finish"


class _CapturePrinter(JsonPrinter):
    """Real printer recording every broadcast."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))
        super().broadcast(event)


def _verdict(request: dict[str, Any]) -> dict[str, Any]:
    """Answer the classifier's chat-completions request with a verdict."""
    return {
        "id": "chatcmpl-audit1005",
        "object": "chat.completion",
        "created": 0,
        "model": request.get("model", STANDIN_MODEL),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps({"is_simple": True, "is_development": False}),
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
    }


class _Endpoint:
    """A stand-in endpoint that records the requests it receives.

    The classifier is a plain structured-output call (no ``tools``);
    the agent's tool loop attaches its toolset to every request.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.server = StandInModelServer(self._respond)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.requests.append(request)
        if request.get("tools"):
            return finish_response(_SUMMARY)
        return _verdict(request)


class _EditedWhileClassifying(_Endpoint):
    """The settings panel edits the registry while the classifier's request is in flight.

    The agent's first reply to a task that asks for the switch then calls ``set_model``
    for :data:`_NO_ENDPOINT_MODEL`; every other reply finishes.
    """

    def __init__(self, edit: Callable[[], object]) -> None:
        self.edit = edit
        super().__init__()

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        reply = super()._respond(request)
        last = request["messages"][-1]
        if not request.get("tools"):
            self.edit()
        elif last["role"] == "user" and "switch the model" in json.dumps(last):
            return tool_call_response("set_model", {"model_name": _NO_ENDPOINT_MODEL})
        return reply


def _delete_entry() -> None:
    """Delete the run's model from the registry, as the settings panel does."""
    model_info.delete_custom_model(STANDIN_MODEL)


def _add_entry() -> None:
    """Register the run's model with an endpoint of its own, as the settings panel does."""
    model_info.save_custom_model(STANDIN_MODEL, endpoint="http://127.0.0.1:9/v1", api_key="own-key")


def _no_edit() -> None:
    """Leave the registry as it is."""


# A SEA that sends its run to a gateway of its choice.
_GATEWAY_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {{'model_config': {{'base_url': '{url}', 'api_key': 'kiss-test-key'}}}}
"""


# A SEA that names its run's model (and a gateway run's endpoint) and asks for the classifier.
_DIRECT_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {settings!r}
"""


def _launch_chat(home: IsolatedKissHome, model: str, model_config: dict[str, Any] | None) -> None:
    """Run :data:`_SWITCH_TASK` through a ``ChatSorcarAgent.run`` no task runner prepared."""
    ChatSorcarAgent("direct launch").run(
        prompt_template=_SWITCH_TASK, model_name=model, model_config=model_config,
        work_dir=str(home.repo), web_tools=False, use_memory=False, verbose=False,
    )


def _launch_headless(
    home: IsolatedKissHome, model: str, model_config: dict[str, Any] | None,
) -> None:
    """Run :data:`_SWITCH_TASK` through ``headless.run_sea``, as a SEA naming the model."""
    settings = {"model": model, "auto_classify": True, "use_memory": False}
    if model_config is not None:
        settings["model_config"] = model_config
    sea = home.tmpdir / "direct" / "direct_sea.py"
    sea.parent.mkdir()
    sea.write_text(_DIRECT_SEA.format(settings=settings))
    outcome = run_sea(sea, _SWITCH_TASK, work_dir=str(home.repo), save_trajectories=False)
    assert outcome.error == "" and outcome.success, outcome


def _register_the_model_without_an_endpoint(home: IsolatedKissHome) -> None:
    """Make :data:`_NO_ENDPOINT_MODEL` the registry's only entry."""
    (home.kiss_home / "MY_MODELS.json").write_text(json.dumps({_NO_ENDPOINT_MODEL: {
        "context_length": 262144, "input_price_per_1M": 0.1, "output_price_per_1M": 0.4,
    }}))


@pytest.fixture
def home() -> Iterator[IsolatedKissHome]:
    """Isolated ``KISS_HOME`` with the LLM classifier enabled."""
    if STANDIN_MODEL not in get_available_models():
        pytest.skip(f"{STANDIN_MODEL} is not configured in this environment")
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "0"
    isolated = IsolatedKissHome("kiss-audit1005-classifier-")
    isolated.write_config(classify_with_decisions=False)
    # ``USER_MY_MODELS_PATH`` is bound at import time, so the isolated
    # home's file must be pointed at explicitly (as
    # ``test_settings_custom_models.py`` does) or the entry would land
    # in the developer's real ``~/.kiss/MY_MODELS.json``.
    saved_my_models = model_info.USER_MY_MODELS_PATH
    model_info.USER_MY_MODELS_PATH = isolated.kiss_home / "MY_MODELS.json"
    clear_classification_cache()
    clear_tab_states(_TAB)
    try:
        yield isolated
    finally:
        clear_tab_states(_TAB)
        clear_classification_cache()
        model_info.USER_MY_MODELS_PATH = saved_my_models
        if saved is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved
        isolated.cleanup()


def _wait_idle() -> None:
    deadline = time.time() + _WAIT_S
    while time.time() < deadline:
        with agent_state.STATE_LOCK:
            st = agent_state.find_by_tab(_TAB)
            thread = st.task_thread if st is not None else None
        if thread is None:
            return
        thread.join(timeout=max(0.0, deadline - time.time()))
    raise AssertionError("tab never went idle")


def _run_task(
    home: IsolatedKissHome, prompt: str, model_config: dict[str, Any] | None = None,
    **fields: Any,
) -> _CapturePrinter:
    """Run *prompt* on :data:`STANDIN_MODEL` through a real ``AgentServer``; return its printer.

    *fields* are further wire fields of the ``run`` command.
    """
    printer = _CapturePrinter()
    AgentServer(printer=printer)._cmd_run({
        "type": "run",
        "tabId": _TAB,
        "prompt": prompt,
        "workDir": str(home.repo),
        "model": STANDIN_MODEL,
        "modelConfig": model_config,
        "useWorktree": False,
        "autoCommit": False,
        "useWebTools": False,
        "classifyTasks": True,
        **fields,
    })
    _wait_idle()
    return printer


def test_classifier_uses_the_custom_models_endpoint(home: IsolatedKissHome) -> None:
    """The classifier's request reaches the MY_MODELS endpoint of the run's model."""
    endpoint = _Endpoint()
    try:
        # The bundled name now runs against the stand-in endpoint.
        error = model_info.save_custom_model(
            STANDIN_MODEL, endpoint=endpoint.server.url, api_key="kiss-test-key",
        )
        assert error is None, error
        assert model_info.custom_model_config(STANDIN_MODEL) == {
            "base_url": endpoint.server.url, "api_key": "kiss-test-key",
        }
        printer = _run_task(home, "classify me against my own endpoint")
        # The real agent ran and finished with the stand-in's summary.
        results = [e for e in printer.events if e.get("type") == "result"]
        assert len(results) == 1 and _SUMMARY in str(results[0].get("text")), results
        assert any(
            e.get("type") == "status" and e.get("running") is False
            for e in printer.events
        ), printer.events
        # Exactly two requests reached the custom endpoint, both for the
        # run's model: the classifier's structured-output call first,
        # then the agent's tool-loop call.
        assert [r.get("model") for r in endpoint.requests] == [STANDIN_MODEL] * 2, (
            endpoint.requests
        )
        classifier, run = endpoint.requests
        assert not classifier.get("tools") and classifier.get("response_format"), classifier
        assert run.get("tools") and not run.get("response_format"), run
        assert "classify me against my own endpoint" in json.dumps(run["messages"])
    finally:
        endpoint.server.stop()


@pytest.mark.parametrize(
    ("edit", "switched"), [(_delete_entry, False), (_add_entry, True)], ids=["deleted", "added"],
)
def test_a_registry_edit_while_classifying_leaves_the_run_its_launch_endpoint(
    home: IsolatedKissHome, edit: Callable[[], object], switched: bool,
) -> None:
    """The read that configures the run also says whose endpoint the run has.

    ``deleted``: the run starts on its model's own registry endpoint and the
    entry is deleted while the classifier runs; a later ``set_model`` hands
    that endpoint to no other model, so the switch to a model without one is
    refused and the run goes on as launched.  ``added``: the run starts on
    the settings panel's custom endpoint and an entry is saved under its
    model's name meanwhile; the endpoint stays the run's, so the other model
    is asked there.
    """
    endpoint = _EditedWhileClassifying(edit)
    try:
        _register_the_model_without_an_endpoint(home)
        if switched:
            home.write_config(custom_endpoint=endpoint.server.url, custom_api_key="kiss-test-key")
        else:
            assert model_info.save_custom_model(
                STANDIN_MODEL, endpoint=endpoint.server.url, api_key="kiss-test-key",
            ) is None
        printer = _run_task(home, "switch the model, then finish")
        results = [e for e in printer.events if e.get("type") == "result"]
        assert len(results) == 1 and _SUMMARY in str(results[0].get("text")), results
        # The edit happened inside the launch: after the classifier's request, before the run's.
        assert (model_info.custom_model_config(STANDIN_MODEL) is not None) is switched
        assert [r.get("model") for r in endpoint.requests] == [
            STANDIN_MODEL, STANDIN_MODEL, _NO_ENDPOINT_MODEL if switched else STANDIN_MODEL,
        ]
    finally:
        endpoint.server.stop()


def test_a_later_block_of_the_run_goes_by_the_launch_read_too(home: IsolatedKissHome) -> None:
    """Every ``<task>`` block of a run has the endpoint the one launch read said it has.

    The run starts on its model's own registry endpoint and the entry is deleted
    while the classifier runs.  The second block runs on the same configuration as
    the first: its ``set_model`` hands that endpoint to no other model either.
    """
    endpoint = _EditedWhileClassifying(_delete_entry)
    try:
        _register_the_model_without_an_endpoint(home)
        assert model_info.save_custom_model(
            STANDIN_MODEL, endpoint=endpoint.server.url, api_key="kiss-test-key",
        ) is None
        _run_task(home, "<task>just finish</task><task>switch the model, then finish</task>")
        assert model_info.custom_model_config(STANDIN_MODEL) is None
        # The second block called ``set_model``, and its next request came as launched.
        assert [
            call["function"]["name"]
            for message in endpoint.requests[-1]["messages"]
            for call in message.get("tool_calls") or []
        ] == ["set_model"]
        assert {r.get("model") for r in endpoint.requests} == {STANDIN_MODEL}
    finally:
        endpoint.server.stop()


def test_a_sent_gateway_is_not_taken_for_the_models_own_endpoint(home: IsolatedKissHome) -> None:
    """A run sent to a gateway keeps it across ``set_model``, wherever its model is registered."""
    endpoint = _EditedWhileClassifying(_no_edit)
    try:
        _register_the_model_without_an_endpoint(home)
        _add_entry()
        _run_task(
            home, "switch the model, then finish",
            {"base_url": endpoint.server.url, "api_key": "kiss-test-key"},
        )
        assert [r.get("model") for r in endpoint.requests] == [
            STANDIN_MODEL, STANDIN_MODEL, _NO_ENDPOINT_MODEL,
        ]
    finally:
        endpoint.server.stop()


@pytest.mark.parametrize(
    ("sent", "stated", "registered", "switched"),
    [
        ("endpoint", True, False, False),
        ("endpoint", False, True, True),
        ("flags", False, True, False),
        ("nothing", False, True, False),
        ("replaced", True, False, True),
    ],
    ids=["deleted", "added", "flags-only", "no-configuration", "replaced-by-the-sea"],
)
def test_a_sub_task_takes_its_callers_word_for_the_endpoint_it_inherited(
    home: IsolatedKissHome, sent: str, stated: bool, registered: bool, switched: bool,
) -> None:
    """A sub-task's inherited configuration comes with its caller's record of whose endpoint it has.

    The sub-task's daemon reads the registry after the read that configured the
    caller.  ``deleted``: the caller runs on its model's own registry endpoint and
    the entry is gone by now; ``set_model`` still hands that endpoint to no other
    model.  ``added``: the caller runs on a gateway and an entry naming that
    endpoint was saved under its model's name since; the endpoint stays the run's.
    The caller spoke about the configuration it sent: the daemon's own read answers
    for one it builds itself (the sent one holds framework flags alone, or nothing)
    and for the one a SEA's ``model_config`` setting puts in its place.
    """
    endpoint = _EditedWhileClassifying(_no_edit)
    url = endpoint.server.url
    fields: dict[str, Any] = {"modelOwnsEndpoint": stated}
    configs: dict[str, dict[str, Any] | None] = {
        "endpoint": {"base_url": url, "api_key": "kiss-test-key"},
        "flags": {"stream_stall_timeout": 120.0},
        "nothing": None,
        "replaced": {"base_url": "http://127.0.0.1:9/v1", "api_key": "the-callers-key"},
    }
    try:
        _register_the_model_without_an_endpoint(home)
        if registered:
            assert model_info.save_custom_model(STANDIN_MODEL, url, "kiss-test-key") is None
        if sent == "replaced":
            sea = home.repo / "gateway_sea.py"
            sea.write_text(_GATEWAY_SEA.format(url=url))
            fields["seaPath"] = str(sea)
        _run_task(home, "switch the model, then finish", configs[sent], **fields)
        assert [r.get("model") for r in endpoint.requests] == [
            STANDIN_MODEL, STANDIN_MODEL, _NO_ENDPOINT_MODEL if switched else STANDIN_MODEL,
        ]
    finally:
        endpoint.server.stop()


@pytest.mark.parametrize("launch", [_launch_chat, _launch_headless], ids=["chat", "headless"])
@pytest.mark.parametrize("by_name", [True, False], ids=["by-name", "gateway"])
def test_a_launch_without_the_daemon_reads_the_registry_once_ahead_of_its_classifier(
    home: IsolatedKissHome, launch: Callable[..., None], by_name: bool,
) -> None:
    """A run started by ``ChatSorcarAgent.run`` or ``headless.run_sea`` is configured as the
    daemon configures one: by one registry read, made before its classifier is asked.

    ``by-name``: the model has a registry endpoint and the launch names no other; the
    classifier is asked there as the run is, and the endpoint is the model's own, which
    ``set_model`` hands to no other model.  ``gateway``: the launch names an endpoint, and
    an entry naming it is saved under the model's name while the classifier runs; the
    endpoint stays the run's, so the other model is asked there.
    """
    endpoint = _EditedWhileClassifying(_no_edit)
    url = endpoint.server.url
    try:
        home.write_config(classify_tasks=True)
        _register_the_model_without_an_endpoint(home)
        if by_name:
            assert model_info.save_custom_model(_OWN_ENDPOINT_MODEL, url, "kiss-test-key") is None
            launch(home, _OWN_ENDPOINT_MODEL, None)
            models = [_OWN_ENDPOINT_MODEL] * 3
        else:
            endpoint.edit = functools.partial(
                model_info.save_custom_model, STANDIN_MODEL, url, "kiss-test-key",
            )
            launch(home, STANDIN_MODEL, {"base_url": url, "api_key": "kiss-test-key"})
            models = [STANDIN_MODEL, STANDIN_MODEL, _NO_ENDPOINT_MODEL]
        # The classifier's request, then the run's two (``set_model``, ``finish``).
        assert [(r.get("model"), bool(r.get("tools"))) for r in endpoint.requests] == list(
            zip(models, [False, True, True], strict=True)
        )
    finally:
        endpoint.server.stop()


def _parent_on(home: IsolatedKissHome, model: str, url: str, owns_endpoint: bool) -> Any:
    """A parent agent as its launch left it mid-run: on *model* at *url*, with its record."""
    parent = ChatSorcarAgent("side-channel-parent")
    parent.work_dir = str(home.repo)
    parent.model_name = model
    parent.model_config = {"base_url": url, "api_key": "kiss-test-key"}
    parent._model_owns_endpoint = owns_endpoint
    return parent


@pytest.mark.parametrize(
    ("stated", "registered", "switched"),
    [(True, False, False), (False, True, True)],
    ids=["deleted", "added"],
)
def test_the_merge_resolver_takes_its_parents_word_for_the_endpoint_it_inherited(
    home: IsolatedKissHome, stated: bool, registered: bool, switched: bool,
) -> None:
    """The resolver runs on its parent's configuration with the parent's record of whose it is.

    It is started after the read that configured its parent.  ``deleted``: the parent runs
    on its model's own registry endpoint and the entry is gone by now; the resolver's
    ``set_model`` still hands that endpoint to no other model.  ``added``: the parent runs
    on a gateway and an entry naming it was saved under its model's name since; the
    endpoint stays the run's.
    """
    endpoint = _EditedWhileClassifying(_no_edit)
    url = endpoint.server.url
    try:
        _register_the_model_without_an_endpoint(home)
        if registered:
            assert model_info.save_custom_model(STANDIN_MODEL, url, "kiss-test-key") is None
        run_merge_sea(_parent_on(home, STANDIN_MODEL, url, stated), _SWITCH_TASK, home.repo)
        # The resolver's classifier, its ``set_model`` call, its ``finish``.
        assert [r.get("model") for r in endpoint.requests] == [
            STANDIN_MODEL, STANDIN_MODEL, _NO_ENDPOINT_MODEL if switched else STANDIN_MODEL,
        ]
    finally:
        endpoint.server.stop()


def _switch_as_the_update_child_is_built(
    set_model: Callable[[str], str], frame: FrameType, event: str, arg: Any,
) -> None:
    """Profile hook: the parent's ``set_model`` runs as ``run_task_update_sea`` builds its child."""
    if (
        event == "call" and frame.f_code is ChatSorcarAgent.__init__.__code__
        and frame.f_back is not None and frame.f_back.f_code is run_task_update_sea.__code__
    ):
        sys.setprofile(None)
        set_model(STANDIN_MODEL)


def test_a_task_update_racing_a_switch_runs_on_one_models_name_and_configuration(
    home: IsolatedKissHome,
) -> None:
    """The update child takes its parent's model and configuration as one snapshot.

    The parent runs a model that is reachable only through the gateway its run was sent
    to, and switches to another model while the child is being set up: the child still
    gets a model with the configuration that model runs on, and answers.
    """
    endpoint = _EditedWhileClassifying(_no_edit)
    url = endpoint.server.url
    try:
        _register_the_model_without_an_endpoint(home)
        parent = _parent_on(home, _NO_ENDPOINT_MODEL, url, False)
        executor = executor_of(parent)
        executor.model = model_info.model(_NO_ENDPOINT_MODEL, model_config=parent.model_config)
        executor.model_name = _NO_ENDPOINT_MODEL
        parent._use_web_tools = False
        (set_model,) = (t for t in parent._get_tools() if t.__name__ == "set_model")
        task_id, chat_id = _add_task(
            "Parent task prompt", chat_id="",
            extra={"model": _NO_ENDPOINT_MODEL, "work_dir": str(home.repo)},
        )
        parent.resume_chat_by_id(chat_id)
        with parent._task_id_lock:
            parent._last_task_id = task_id
        sys.setprofile(functools.partial(_switch_as_the_update_child_is_built, set_model))
        try:
            text, _cost = run_task_update_sea(parent, task_id)
        finally:
            sys.setprofile(None)
        assert parent.model_name == STANDIN_MODEL, "the switch happened during the set-up"
        assert text == _SUMMARY
        assert len(endpoint.requests) == 1, endpoint.requests
    finally:
        endpoint.server.stop()
