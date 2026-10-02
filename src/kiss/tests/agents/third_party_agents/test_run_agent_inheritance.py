# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for what a path-mode ``run_agent`` sub-task inherits.

A ``run_agent`` call that names an agent-script path works on the calling
task's project, so the arguments the call leaves empty are filled from the
calling agent the way a ``run_parallel`` child's are
(:func:`kiss.agents.sorcar.agent_dispatch.inherit_from_parent`): model,
same-model model configuration, half of the remaining budget, chat id,
web-tools and memory settings, the live Docker container, and the
effective worktree / auto-commit choices.  Channel and cron sub-tasks and
explicit programmatic callers (``inherit=False``) inherit nothing.

Every test runs the real dispatch path up to the ``daemon_client.run``
boundary, where the kwargs are captured; the full-run tests drive a real
``WorktreeSorcarAgent.run`` against a scripted local model endpoint whose
first answer is the ``run_agent`` tool call.  The module lives in this
directory for the same reason as ``test_agent_dispatch.py``: the tool
scans the third-party channel directory to tell channel names from paths.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import agent_dispatch, daemon_client
from kiss.agents.sorcar.agent_dispatch import RunOptions, dispatch_result, make_run_agent_tool
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.kiss_error import BudgetExceededError
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    tool_call_response,
)

DUMMY_SEA = str(
    Path(agent_dispatch.__file__).resolve().parents[1] / "seas" / "dummy" / "dummy_sea.py"
)
PARENT_MODEL = "gpt-4o-mini"
PARENT_CONFIG = {"base_url": "http://127.0.0.1:1/v1", "api_key": "kiss-test-key"}


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[IsolatedKissHome]:
    """Isolated KISS_HOME / history DB / scratch repo with no reachable daemon."""
    isolated = IsolatedKissHome("kiss-run-agent-inherit-")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    monkeypatch.delenv("KISS_CHANNEL_WORKSPACE", raising=False)
    try:
        yield isolated
    finally:
        isolated.cleanup()


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every ``daemon_client.run`` call's kwargs; answer with a success."""
    calls: list[dict[str, Any]] = []

    def capture_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        calls.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(
            text="ok", success=True, cost=0.25, tokens=10, steps=1, chat_id="chat-child",
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)
    return calls


def _parent_after_a_run(repo: Path, auto_commit: bool, use_worktree: bool) -> WorktreeSorcarAgent:
    """A ``WorktreeSorcarAgent`` carrying the state a real ``run`` leaves behind.

    The values are the public per-run attributes ``run`` assigns (the
    full-run tests below check that assignment end to end); setting them
    directly keeps these tests independent of a model endpoint.
    """
    parent = WorktreeSorcarAgent("inherit-parent")
    parent.model_name = PARENT_MODEL
    parent._launch_model_name = PARENT_MODEL
    parent.model_config = dict(PARENT_CONFIG)
    parent.max_budget = 4.0
    parent.budget_used = 1.0
    parent._chat_id = "chat-parent"
    parent._use_web_tools = False
    parent._use_memory_override = True
    parent.auto_commit_enabled = auto_commit
    parent.use_worktree_enabled = use_worktree
    parent.work_dir = str(repo)
    return parent


def _dispatch(
    parent: Any,
    inherit: bool = True,
    model_name: str = "",
    budget: float | None = None,
    options: RunOptions = RunOptions(),
) -> daemon_client.TaskResult:
    """Run ``dispatch_result`` for ``dummy_sea.py`` on *parent*'s repo."""
    result = dispatch_result(
        "dummy_sea", "say hi", DUMMY_SEA, parent.work_dir, model_name, budget, 30.0,
        parent_agent=parent, inherit=inherit, options=options,
    )
    assert isinstance(result, daemon_client.TaskResult), result
    return result


class TestDispatchResultInheritance:
    """``dispatch_result(inherit=True)`` fills the empty arguments from the caller."""

    def test_empty_arguments_come_from_the_parent(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """Model, same-model config, budget share, chat, web, memory, worktree, auto-commit."""
        env.write_config(is_worktree=True, auto_commit_mode=True)
        parent = _parent_after_a_run(env.repo, auto_commit=False, use_worktree=False)
        _dispatch(parent)
        (call,) = captured
        assert call["model"] == PARENT_MODEL
        assert call["model_config"] == PARENT_CONFIG
        # (4.0 - 1.0 used) / 2: half of the remaining budget.
        assert call["max_budget"] == pytest.approx(1.5)
        assert call["chat_id"] == "chat-parent"
        assert call["use_web_tools"] is False
        assert call["use_memory"] is True
        assert call["docker_image"] == ""
        # The parent's EFFECTIVE choices beat the persisted settings.
        assert call["use_worktree"] is False
        assert call["auto_commit"] is False

    def test_parent_worktree_and_auto_commit_on_are_inherited_over_config_off(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """A parent running in a worktree with auto-commit passes both on."""
        env.write_config(is_worktree=False, auto_commit_mode=False)
        parent = _parent_after_a_run(env.repo, auto_commit=True, use_worktree=True)
        _dispatch(parent)
        assert captured[0]["use_worktree"] is True
        assert captured[0]["auto_commit"] is True

    def test_explicit_arguments_win(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """Every explicitly passed value is kept, nothing is overwritten."""
        parent = _parent_after_a_run(env.repo, auto_commit=True, use_worktree=True)
        options = RunOptions(
            chat_id="chat-explicit", model_config={"base_url": "http://x/v1"},
            use_worktree=False, auto_commit=False, use_web_tools=True, use_memory=False,
        )
        _dispatch(parent, model_name=PARENT_MODEL, budget=0.5, options=options)
        (call,) = captured
        assert call["model"] == PARENT_MODEL
        assert call["model_config"] == {"base_url": "http://x/v1"}
        assert call["max_budget"] == 0.5
        assert call["chat_id"] == "chat-explicit"
        assert call["use_web_tools"] is True
        assert call["use_memory"] is False
        assert call["use_worktree"] is False
        assert call["auto_commit"] is False

    def test_a_different_model_drops_the_parent_model_config(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """The parent's endpoint and key belong to its model only."""
        parent = _parent_after_a_run(env.repo, auto_commit=True, use_worktree=True)
        _dispatch(parent, model_name="claude-sonnet-4-5")
        assert captured[0]["model"] == "claude-sonnet-4-5"
        assert captured[0]["model_config"] is None

    def test_a_script_that_picks_its_model_gets_no_parent_model_config(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """The daemon applies the script's ``model()`` without touching ``modelConfig``.

        The parent's endpoint must therefore not be inherited when the
        script chooses the model — whichever way it binds ``model`` at
        module level.  Any other top-level name leaves the inheritance.
        """
        parent = _parent_after_a_run(env.repo, auto_commit=True, use_worktree=True)
        scripts = env.repo / "scripts"
        scripts.mkdir()
        bodies = {
            "def": "def model() -> str:\n    return 'claude-sonnet-4-5'\n",
            "async": "async def model():\n    return 'x'\n",
            "class": "class model:\n    pass\n",
            "assign": "model = lambda: 'x'\n",
            "annotated": "model: object = None\n",
            "import": "from os.path import basename as model\n",
            "import_as": "import os as model\n",
            "conditional": (
                "import os\nif os.name == 'posix':\n"
                "    def model():\n        return 'claude-sonnet-4-5'\n"
            ),
            "tuple": "def pick():\n    return 'x'\n\nmodel, other = pick, None\n",
            "try": "try:\n    from missing import model\nexcept ImportError:\n    model = None\n",
        }
        for label, body in bodies.items():
            script = scripts / f"{label}_sea.py"
            script.write_text(body)
            captured.clear()
            result = dispatch_result(
                label, "say hi", str(script), str(env.repo), "", None, 30.0,
                parent_agent=parent, inherit=True,
            )
            assert isinstance(result, daemon_client.TaskResult), result
            assert captured[0]["model_config"] is None, label
            # The wire ``model`` is still the parent's; the daemon's
            # override replaces it with the script's choice.
            assert captured[0]["model"] == PARENT_MODEL, label
        for label, body in {
            "other": "def model_name() -> str:\n    return 'x'\n\nmodels = []\n",
            "unparsable": "def model(:\n",
        }.items():
            script = scripts / f"{label}_sea.py"
            script.write_text(body)
            captured.clear()
            dispatch_result(
                label, "say hi", str(script), str(env.repo), "", None, 30.0,
                parent_agent=parent, inherit=True,
            )
            assert captured[0]["model_config"] == PARENT_CONFIG, label
        assert agent_dispatch.script_defines(str(scripts / "missing.py"), "model") is False

    def test_an_empty_parent_model_config_is_not_forwarded(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """``{}`` means "no configuration": the daemon default applies."""
        parent = _parent_after_a_run(env.repo, auto_commit=True, use_worktree=True)
        parent.model_config = {}
        _dispatch(parent)
        assert captured[0]["model_config"] is None

    def test_model_config_follows_the_launch_model_after_set_model(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """The run's config belongs to the launch model, not to a model switched to later.

        After ``set_model`` the parent's ``model_name`` is the new model
        while ``model_config`` still names the launch model's endpoint:
        an empty ``model_name`` inherits the new model WITHOUT that
        endpoint, and asking for the launch model by name gets it back.
        """
        parent = _parent_after_a_run(env.repo, auto_commit=True, use_worktree=True)
        parent.model_name = "claude-sonnet-4-5"  # what set_model leaves behind
        _dispatch(parent)
        assert captured[0]["model"] == "claude-sonnet-4-5"
        assert captured[0]["model_config"] is None
        captured.clear()
        _dispatch(parent, model_name=PARENT_MODEL)
        assert captured[0]["model"] == PARENT_MODEL
        assert captured[0]["model_config"] == PARENT_CONFIG

    def test_a_parent_without_a_run_inherits_only_its_defaults(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """A never-run ``ChatSorcarAgent`` has no budget, chat, or worktree choice yet."""
        env.write_config(is_worktree=False, auto_commit_mode=True)
        parent = ChatSorcarAgent("fresh-parent")
        parent.work_dir = str(env.repo)
        _dispatch(parent)
        (call,) = captured
        assert call["model"] == ""  # daemon default
        assert call["model_config"] is None
        assert call["max_budget"] is None  # daemon default
        assert call["chat_id"] == ""
        assert call["use_web_tools"] is True  # the agent's initial per-run value
        assert call["use_memory"] is None
        # No ``use_worktree_enabled`` / ``auto_commit_enabled`` on a
        # ChatSorcarAgent: the persisted settings decide, as before.
        assert call["use_worktree"] is False
        assert call["auto_commit"] is True

    def test_inherit_false_keeps_the_old_defaults(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """Programmatic callers that pass values explicitly are unaffected."""
        env.write_config(is_worktree=True, auto_commit_mode=True)
        parent = _parent_after_a_run(env.repo, auto_commit=False, use_worktree=False)
        _dispatch(parent, inherit=False)
        (call,) = captured
        assert call["model"] == ""
        assert call["model_config"] is None
        assert call["max_budget"] is None
        assert call["chat_id"] == ""
        assert call["use_web_tools"] is None
        assert call["use_memory"] is None
        assert call["use_worktree"] is True
        assert call["auto_commit"] is True

    def test_exhausted_parent_budget_raises_like_run_parallel(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """Nothing left to split: the fan-out signal propagates, no dispatch happens."""
        parent = _parent_after_a_run(env.repo, auto_commit=False, use_worktree=False)
        parent.budget_used = 4.0
        with pytest.raises(BudgetExceededError):
            _dispatch(parent)
        assert captured == []

    def test_channel_mode_inherits_nothing(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """A channel sub-task acts on an external service: no chat, model, or budget."""
        parent = _parent_after_a_run(env.repo, auto_commit=True, use_worktree=True)
        run_agent = make_run_agent_tool(str(env.repo), parent)
        text = run_agent(task="list channels", agent="slack", timeout="30")
        assert yaml.safe_load(text)["success"] is True, text
        (call,) = captured
        assert call["model"] == ""
        assert call["model_config"] is None
        assert call["max_budget"] is None
        assert call["chat_id"] == ""
        assert call["use_web_tools"] is None
        assert call["use_memory"] is None
        assert call["docker_image"] == ""
        assert call["use_worktree"] is False
        assert call["auto_commit"] is False


def _run_parent(
    env: IsolatedKissHome, server: StandInModelServer, *, use_worktree: bool, **kwargs: Any,
) -> WorktreeSorcarAgent:
    """Run a ``WorktreeSorcarAgent`` whose scripted model calls ``run_agent`` once."""
    parent = WorktreeSorcarAgent("inherit-e2e-parent")
    result = parent.run(
        prompt_template="delegate to the dummy agent",
        model_name=STANDIN_MODEL,
        model_config=server.model_config,
        work_dir=str(env.repo),
        max_budget=4.0,
        web_tools=False,
        use_memory=True,
        is_parallel=False,
        verbose=False,
        use_worktree=use_worktree,
        auto_commit=False,
        **kwargs,
    )
    assert yaml.safe_load(result)["success"] is True, result
    return parent


class _DelegatingModel:
    """Scripted model: first ``run_agent(dummy_sea.py)``, then ``finish``."""

    def __init__(self) -> None:
        self.turns = 0

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.turns += 1
        if self.turns == 1:
            return tool_call_response(
                "run_agent", {"task": "say hi", "agent": DUMMY_SEA, "timeout": "30"},
            )
        return finish_response("delegated")


class TestFullRunInheritance:
    """A real ``WorktreeSorcarAgent.run`` → ``run_agent`` → captured dispatch."""

    def test_direct_run_passes_its_live_state_to_the_sub_task(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """Model, config, budget share, chat, web/memory, worktree=False, auto-commit=False."""
        env.write_config(classify_tasks=False, is_worktree=True, auto_commit_mode=True)
        server = StandInModelServer(_DelegatingModel())
        try:
            parent = _run_parent(env, server, use_worktree=False)
        finally:
            server.stop()
        (call,) = captured
        assert call["model"] == STANDIN_MODEL
        assert call["model_config"]["base_url"] == server.url
        assert call["model_config"]["api_key"] == server.model_config["api_key"]
        assert 0.0 < call["max_budget"] <= 2.0  # half of what was left of 4.0
        assert call["chat_id"] == parent._chat_id != ""
        assert call["use_web_tools"] is False
        assert call["use_memory"] is True
        assert call["docker_image"] == ""
        assert call["work_dir"] == str(env.repo)
        # The run's effective choices, not the (opposite) persisted settings.
        assert parent.use_worktree_enabled is False
        assert call["use_worktree"] is False
        assert call["auto_commit"] is False
        # The sub-task's spend folded into the parent's accounting.
        assert parent.budget_used >= 0.25

    def test_worktree_run_passes_its_worktree_and_its_directory(
        self, env: IsolatedKissHome, captured: list[dict[str, Any]],
    ) -> None:
        """A parent that got a worktree hands the sub-task that worktree and ``use_worktree``."""
        env.write_config(classify_tasks=False, is_worktree=False, auto_commit_mode=False)
        server = StandInModelServer(_DelegatingModel())
        try:
            parent = _run_parent(env, server, use_worktree=True)
        finally:
            server.stop()
        (call,) = captured
        assert parent.use_worktree_enabled is True
        assert call["use_worktree"] is True
        assert call["auto_commit"] is False
        worktree_dir = Path(call["work_dir"])
        assert worktree_dir != env.repo.resolve() and worktree_dir != env.repo
        assert ".kiss-worktrees" in worktree_dir.parts


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _docker_available(), reason="Docker daemon is not running")
def test_parent_container_is_attached_by_the_sub_task(
    env: IsolatedKissHome, captured: list[dict[str, Any]],
) -> None:
    """A parent inside a Docker container sends the sub-task into the same container.

    An attached ``DockerManager`` works in the container's working
    directory — the parent's mounted tree — so the sub-task gets no
    worktree and no auto-commit of its own even though the parent runs
    with both: it works in the parent's tree like a ``run_parallel``
    child.  An explicit ``use_worktree`` still wins, as everywhere.
    """
    import docker

    from kiss.agents.sorcar.docker_manager import ATTACH_PREFIX, DockerManager

    client = docker.from_env()
    container = client.containers.run(
        "python:3.11-slim", command="sleep infinity", detach=True,
        name=f"kiss-inherit-{uuid.uuid4().hex[:8]}",
    )
    container_id = str(container.id)
    try:
        env.write_config(is_worktree=True, auto_commit_mode=True)
        parent = _parent_after_a_run(env.repo, auto_commit=True, use_worktree=True)
        with DockerManager(ATTACH_PREFIX + container_id) as manager:
            parent.docker_manager = manager
            assert manager.container is not None
            _dispatch(parent)
            assert captured[0]["docker_image"] == ATTACH_PREFIX + container_id
            assert captured[0]["use_worktree"] is False
            assert captured[0]["auto_commit"] is False
            captured.clear()
            _dispatch(parent, options=RunOptions(use_worktree=True))
            assert captured[0]["use_worktree"] is True
            assert captured[0]["auto_commit"] is False
        # Explicit ``docker_image`` is not a tool argument: the only way a
        # sub-task reaches a container is through its parent; outside one
        # the parent's own worktree / auto-commit choices apply again.
        captured.clear()
        parent.docker_manager = None
        _dispatch(parent)
        assert captured[0]["docker_image"] == ""
        assert captured[0]["use_worktree"] is True
        assert captured[0]["auto_commit"] is True
    finally:
        container.remove(force=True)
