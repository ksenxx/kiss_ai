# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for tool profiles (WP1b),
``run_parallel(model=..., tool_profile=...)`` (WP3), and the cost-lever
config toggles (WP0).

The fan-out tests run the parent's ``run_parallel`` tool against a
real daemon on a loopback endpoint; the children it spawns talk to the
scripted local model server and their state is read back from the
daemon's history database.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.fanout_guard import is_implementation_task
from kiss.agents.sorcar.sorcar_agent import (
    BROWSER_TOOL_NAMES,
    TOOL_GROUPS,
    TOOL_PROFILES,
    SorcarAgent,
    resolve_tool_profile,
)
from kiss.core.config import DEFAULT_CONFIG, Config
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)


def _bare_agent(tmp_path: Path, **attrs: Any) -> ChatSorcarAgent:
    """A ChatSorcarAgent with the attributes ``run()`` would set before ``_get_tools``."""
    agent = ChatSorcarAgent("profile-test")
    agent._use_web_tools = False
    agent._is_parallel = True
    agent._use_memory_override = None
    agent._append_basic_tools = True
    agent.web_use_tool = None
    agent._memory_tools = None
    agent.work_dir = str(tmp_path)
    agent.docker_manager = None
    agent.printer = None
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


def _names(tools: list[Callable[..., Any]]) -> set[str]:
    return {t.__name__ for t in tools}


def _tool(agent: SorcarAgent, name: str) -> Callable[..., Any]:
    tool: Callable[..., Any] = next(t for t in agent._get_tools() if t.__name__ == name)
    return tool


class TestConfigToggles:
    def test_env_toggles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KISS_READ_DEDUPE", "0")
        monkeypatch.setenv("KISS_TOOL_OUTPUT_COMPACTION", "off")
        monkeypatch.setenv("KISS_TOOL_PROFILES", "yes")
        monkeypatch.setenv("KISS_CONTEXT_LIMIT_FRACTION", "0.85")
        monkeypatch.setenv("KISS_READ_OUTLINE_LINES", "1500")
        monkeypatch.setenv("KISS_CHAT_HISTORY_DIGEST", "")
        cfg = Config()
        assert cfg.read_dedupe is False
        assert cfg.tool_output_compaction is False
        assert cfg.tool_profiles is True
        assert cfg.context_limit_fraction == 0.85
        assert cfg.read_outline_lines == 1500
        assert cfg.chat_history_digest is True  # empty = default
        assert cfg.dispatch_path_rewrite is True
        monkeypatch.setenv("KISS_READ_OUTLINE_LINES", "x")
        assert Config().read_outline_lines == 2000
        monkeypatch.setenv("KISS_TOOL_OUTPUT_MAX_CHARS", "1234")
        monkeypatch.setenv("KISS_COMPACTION_START_TOKENS", "40000")
        monkeypatch.setenv("KISS_COMPACTION_STEP_TOKENS", "20000")
        cfg = Config()
        assert cfg.tool_output_max_chars == 1234
        assert (cfg.compaction_start_tokens, cfg.compaction_step_tokens) == (40000, 20000)

    def test_defaults_are_on(self) -> None:
        for name in (
            "read_dedupe",
            "tool_output_compaction",
            "tool_profiles",
            "chat_history_digest",
            "dispatch_path_rewrite",
        ):
            assert getattr(DEFAULT_CONFIG, name) is True, name
        assert DEFAULT_CONFIG.context_limit_fraction == 0.7
        assert DEFAULT_CONFIG.tool_output_max_chars == 50000
        assert DEFAULT_CONFIG.compaction_start_tokens == 100_000
        assert DEFAULT_CONFIG.compaction_step_tokens == 100_000

    def test_tool_output_cap_env_reaches_bash_and_docker_defaults(self, tmp_path: Path) -> None:
        """KISS_TOOL_OUTPUT_MAX_CHARS sets the Bash default cap in a fresh process.

        Covers both the host and the Docker mode defaults.
        """
        code = (
            "import inspect\n"
            "from kiss.agents.sorcar import docker_manager, useful_tools\n"
            "print(docker_manager.MAX_OUTPUT_CHARS, "
            "inspect.signature(useful_tools.UsefulTools.Bash)"
            ".parameters['max_output_chars'].default)\n"
            "print(useful_tools.UsefulTools(work_dir=str(work))"
            ".Bash('yes | head -c 5000', 'long output'))\n".replace(
                "str(work)", repr(str(tmp_path))
            )
        )
        env = {**os.environ, "KISS_TOOL_OUTPUT_MAX_CHARS": "1200"}
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        ).stdout
        first, rest = out.split("\n", 1)
        assert first == "1200 1200"
        assert "[truncated" in rest and len(rest) < 1500


class TestToolProfiles:
    def test_full_profile_has_everything(self, tmp_path: Path) -> None:
        names = _names(_bare_agent(tmp_path)._get_tools())
        assert {"Bash", "Read", "Edit", "Write", "run_commands_parallel", "run_agent",
                "ask_user_question", "talk", "set_model", "summary", "run_parallel",
                "number_of_cores"} <= names

    def test_review_profile_is_read_only(self, tmp_path: Path) -> None:
        """``review`` reads, runs, browses and talks; it never edits or dispatches."""
        agent = _bare_agent(tmp_path, _tool_profile_name="review", _use_web_tools=True)
        names = _names(agent._get_tools())
        assert names <= set(TOOL_PROFILES["review"])  # type: ignore[arg-type]
        assert {"Bash", "Read", "run_commands_parallel", "summary", "talk"} <= names
        assert BROWSER_TOOL_NAMES <= names
        assert agent.web_use_tool is not None
        assert not names & {"Edit", "Write", "run_agent", "run_parallel",
                            "set_model", "ask_user_question"}

    def test_review_profile_honours_web_tools_off(self, tmp_path: Path) -> None:
        """With "Use web tools" off the review profile builds no browser."""
        agent = _bare_agent(tmp_path, _tool_profile_name="review", _use_web_tools=False)
        names = _names(agent._get_tools())
        assert not names & BROWSER_TOOL_NAMES
        assert agent.web_use_tool is None
        assert "talk" in names

    def test_shell_profile(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, _tool_profile_name="shell")
        assert _names(agent._get_tools()) == {
            "Bash", "bash_job", "Read", "run_commands_parallel",
        }

    def test_assistant_profile_adds_user_interaction(self, tmp_path: Path) -> None:
        """``assistant`` is the shell set plus the user-facing tools; still no editing."""
        agent = _bare_agent(tmp_path, _tool_profile_name="assistant")
        names = _names(agent._get_tools())
        expected = {
            "Bash", "bash_job", "Read", "run_commands_parallel",
            "ask_user_question", "talk", "summary", "set_model",
        }
        # ``decide`` needs an OpenRouter key; it is offered only when it can run.
        assert names - {"decide"} == expected
        assert names <= set(TOOL_PROFILES["assistant"])  # type: ignore[arg-type]
        assert not names & {"Edit", "Write", "run_agent", "run_parallel", "memory_search"}

    def test_reviewer_subagent_defaults_to_review(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent = _bare_agent(tmp_path, _subagent_info={"reviewer": True})
        assert agent._tool_profile() == "review"
        assert "Edit" not in _names(agent._get_tools())
        monkeypatch.setattr(DEFAULT_CONFIG, "tool_profiles", False)
        assert agent._tool_profile() == "full"
        assert "Edit" in _names(agent._get_tools())
        # An unknown explicit name falls through to the default rule.
        agent._tool_profile_name = "bogus"
        assert agent._tool_profile() == "full"

    def test_non_reviewer_subagent_keeps_full(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, _subagent_info={"reviewer": False})
        assert agent._tool_profile() == "full"

    def test_run_parallel_rejects_unknown_profile(self, tmp_path: Path) -> None:
        run_parallel = _tool(_bare_agent(tmp_path), "run_parallel")
        out = run_parallel('["do x"]', tool_profile="admin")
        assert out.startswith("Error: tool_profile must be one of full, review, shell")
        # A composite is rejected as a whole when one part is unknown.
        out = run_parallel('["do x"]', tool_profile="shell+admin")
        assert out.startswith("Error: tool_profile must be one of")
        assert "'shell+admin'" in out


class TestComposableProfiles:
    """Group profiles and ``+``-joined composites (:func:`resolve_tool_profile`)."""

    def test_every_group_is_a_profile(self) -> None:
        assert set(TOOL_GROUPS) <= set(TOOL_PROFILES)
        assert set(TOOL_GROUPS) == {
            "shell", "edit", "browser", "memory", "agents", "mcp", "skills",
            "user", "decide", "control",
        }
        # The groups partition the built-in tool names: no tool in two groups.
        names = [name for group in TOOL_GROUPS.values() for name in group]
        assert len(names) == len(set(names))
        assert TOOL_PROFILES["assistant"] == (
            TOOL_GROUPS["shell"] | TOOL_GROUPS["user"] | TOOL_GROUPS["decide"]
            | TOOL_GROUPS["control"]
        )

    def test_resolve_unions_parts_and_full_absorbs(self) -> None:
        assert resolve_tool_profile("") is None
        assert resolve_tool_profile("full") is None
        assert resolve_tool_profile("shell+full") is None
        assert resolve_tool_profile(" shell + edit ") == (
            TOOL_GROUPS["shell"] | TOOL_GROUPS["edit"]
        )
        assert resolve_tool_profile("review+edit") == (
            TOOL_PROFILES["review"] | TOOL_GROUPS["edit"]  # type: ignore[operator]
        )
        for bad in ("admin", "shell+admin", "+", "shell+", "shell++edit"):
            with pytest.raises(ValueError, match="tool_profile must be one of"):
                resolve_tool_profile(bad)

    def test_shell_plus_edit_is_the_file_toolset(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, _tool_profile_name="shell+edit")
        assert agent._tool_profile() == "shell+edit"
        assert _names(agent._get_tools()) == {
            "Bash", "bash_job", "Read", "run_commands_parallel", "Edit", "Write",
        }

    def test_agents_group_builds_dispatch_and_fanout(self, tmp_path: Path) -> None:
        """``agents`` builds ``run_agent`` and, in parallel mode, the fan-out."""
        agent = _bare_agent(tmp_path, _tool_profile_name="agents")
        assert _names(agent._get_tools()) == {
            "run_agent", "agent_job", "run_parallel", "number_of_cores",
        }
        serial = _bare_agent(tmp_path, _tool_profile_name="agents", _is_parallel=False)
        assert _names(serial._get_tools()) == {"run_agent", "agent_job"}

    def test_mcp_group_builds_the_sign_in_pair(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, _tool_profile_name="mcp")
        assert _names(agent._get_tools()) == {
            "connect_mcp_server", "finish_mcp_server_connect",
        }

    def test_skills_group_builds_the_skill_tool(self, tmp_path: Path) -> None:
        (tmp_path / ".kiss" / "skills" / "demo").mkdir(parents=True)
        (tmp_path / ".kiss" / "skills" / "demo" / "SKILL.md").write_text(
            "---\nname: demo\ndescription: A demo skill.\n---\nDo the demo.\n"
        )
        agent = _bare_agent(tmp_path, _tool_profile_name="skills")
        assert _names(agent._get_tools()) == {"skill"}
        # The same work dir with a profile lacking the group builds no skill tool.
        assert "skill" not in _names(
            _bare_agent(tmp_path, _tool_profile_name="shell")._get_tools()
        )

    def test_user_control_and_browser_groups(self, tmp_path: Path) -> None:
        agent = _bare_agent(
            tmp_path, _tool_profile_name="user+control+browser", _use_web_tools=True,
        )
        names = _names(agent._get_tools())
        assert names == {"ask_user_question", "talk", "summary", "set_model"} | BROWSER_TOOL_NAMES
        assert agent.web_use_tool is not None

    def test_full_in_a_composite_means_everything(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, _tool_profile_name="shell+full")
        assert {"Bash", "Edit", "Write", "run_agent", "run_parallel"} <= _names(agent._get_tools())


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    home = IsolatedKissHome(prefix="kiss-tool-profiles-")
    home.write_config(is_worktree=False, auto_commit_mode=False, classify_tasks=False)
    try:
        yield home
    finally:
        home.cleanup()


@pytest.fixture
def daemon(env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A real daemon on a loopback endpoint that ``run_parallel`` dispatches to."""
    endpoint_file = str(env.tmpdir / "sorcar-local.json")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", endpoint_file)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server = RemoteAccessServer(local_endpoint_file=endpoint_file, work_dir=str(env.repo))
    asyncio.run_coroutine_threadsafe(server.start_private_async(), loop).result(timeout=30)
    try:
        yield endpoint_file
    finally:
        asyncio.run_coroutine_threadsafe(server.stop_async(), loop).result(timeout=15)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


class _FinishingModel:
    """A stand-in model that answers every request with ``finish`` and records the requests.

    Daemon-run children stream their completions, so the harness's
    :class:`StandInModelServer` (which speaks the SSE wire format) is
    used rather than the non-streaming scripted server.
    """

    def __init__(self, summary: str) -> None:
        self.requests: list[dict[str, Any]] = []
        self.summary = summary
        self.server = StandInModelServer(self)

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Record *request* and answer it with a ``finish`` call.

        The usage is large enough that the child's spend survives the
        daemon's four-decimal cost reporting and shows up in the parent.
        """
        self.requests.append(request)
        body = finish_response(self.summary)
        body["usage"] = {"prompt_tokens": 20000, "completion_tokens": 5, "total_tokens": 20005}
        return body


@pytest.fixture
def model() -> Iterator[_FinishingModel]:
    finishing = _FinishingModel("<p>done</p>")
    try:
        yield finishing
    finally:
        finishing.server.stop()


def _parent(env: IsolatedKissHome, model: _FinishingModel, max_budget: float) -> ChatSorcarAgent:
    """A parent whose ``run_parallel`` children inherit *model*'s endpoint."""
    agent = _bare_agent(env.repo, max_budget=max_budget, budget_used=0.0)
    agent.model_name = STANDIN_MODEL
    agent.model_config = model.server.model_config
    agent._chat_id = ""
    agent._last_task_id = uuid.uuid4().hex
    return agent


def _child_rows(parent_agent: ChatSorcarAgent) -> list[tuple[str, float, str]]:
    """Return ``(model, max_budget, task)`` of the persisted children of *parent_agent*."""
    parent_id = str(getattr(parent_agent, "_last_task_id", "") or "")
    conn = th._get_db()
    rows: list[tuple[str, float, str]] = []
    for model, max_budget, task, start_ts, end_ts in conn.execute(
        "SELECT model, max_budget, task, start_ts, end_ts FROM task_history "
        "WHERE parent_task_id = ?",
        (parent_id,),
    ):
        # WP6: every finished child row carries an end timestamp.
        assert int(end_ts) >= int(start_ts) > 0, (start_ts, end_ts)
        rows.append((str(model), float(max_budget), str(task)))
    return rows


def _child_results(out: str) -> list[dict[str, Any]]:
    """``run_parallel`` returns a YAML list of per-child YAML result strings."""
    return [yaml.safe_load(item) for item in yaml.safe_load(out)]


class TestFanoutPropagation:
    """Real fan-outs: the children are daemon sub-tasks that run against
    the scripted server and are observed through the server's requests
    and their persisted rows."""

    def test_children_get_profile_model_and_plain_budget_share(
        self, env: IsolatedKissHome, daemon: str, model: _FinishingModel,
    ) -> None:
        model.summary = "<p>reviewed</p>"
        agent = _parent(env, model, max_budget=4.0)
        run_parallel = _tool(agent, "run_parallel")
        out = run_parallel(
            '["Review module A for bugs"]', model=STANDIN_MODEL, tool_profile="review",
        )
        result = _child_results(out)[0]
        assert result["success"] is True and "reviewed" in result["summary"]
        # The child reached OUR server: the parent's model_config was
        # forwarded (same model), and its request carried the review
        # toolset only.
        requests = model.requests
        assert len(requests) == 1
        sent_tools = {t["function"]["name"] for t in requests[0]["tools"]}
        assert "Edit" not in sent_tools and "run_parallel" not in sent_tools
        assert {"Bash", "Read", "finish"} <= sent_tools
        # Persisted child row: the model named at dispatch and the plain
        # remaining-budget share (4.0 / (1 + 1) = 2.0); a review task is
        # not clipped below it.
        rows = _child_rows(agent)
        assert len(rows) == 1
        child_model, max_budget, task = rows[0]
        assert child_model == STANDIN_MODEL and task == "Review module A for bugs"
        assert max_budget == pytest.approx(2.0)
        assert float(getattr(agent, "budget_used", 0.0)) > 0

    def test_different_model_is_dispatched_with_default_routing(
        self, env: IsolatedKissHome, daemon: str, model: _FinishingModel,
    ) -> None:
        # A different model gets default provider routing, not the parent's
        # endpoint: the child never reaches the parent's local server and,
        # having no key for the real provider, the daemon refuses it
        # before it starts, with a result the parent can read.
        agent = _parent(env, model, max_budget=4.0)
        run_parallel = _tool(agent, "run_parallel")
        out = run_parallel('["summarize a"]', model="no-such-model-cost-levers")
        assert model.requests == []
        result = _child_results(out)[0]
        assert result["success"] is False
        assert "No model available" in result["summary"], result
        assert _child_rows(agent) == []

    def test_reviewer_and_plain_children_get_the_same_share(
        self, env: IsolatedKissHome, daemon: str, model: _FinishingModel,
    ) -> None:
        """A mixed fan-out hands every child the plain share: the review
        child is neither clipped nor charged against a separate allowance."""
        agent = _parent(env, model, max_budget=6.0)
        run_parallel = _tool(agent, "run_parallel")
        out = run_parallel('["Review module A for bugs", "Summarize module B"]')
        assert len(_child_results(out)) == 2 and len(model.requests) == 2
        rows = {task: budget for _model, budget, task in _child_rows(agent)}
        # Plain share is 6.0 / (2 + 1) = 2.0 for both children.
        assert rows["Review module A for bugs"] == pytest.approx(2.0)
        assert rows["Summarize module B"] == pytest.approx(2.0)


def test_child_profile_is_stamped_by_daemon(
    env: IsolatedKissHome, daemon: str, model: _FinishingModel,
) -> None:
    """``run_parallel(tool_profile=...)`` reaches every child's toolset and system prompt."""
    agent = _parent(env, model, max_budget=1.0)
    run_parallel = _tool(agent, "run_parallel")
    out = run_parallel('["Summarize this", "Summarize that"]', tool_profile="shell")
    assert all(r["success"] for r in _child_results(out))
    assert len(model.requests) == 2
    for request in model.requests:
        names = {t["function"]["name"] for t in request["tools"]}
        assert names == {"Bash", "bash_job", "Read", "run_commands_parallel", "finish"}
        system = next(m for m in request["messages"] if m["role"] == "system")["content"]
        assert "# Restricted tool profile: shell" in system
        assert "Bash, Read, bash_job, run_commands_parallel" in system


class TestImplementationTasksKeepFullToolset:
    def test_is_implementation_task(self) -> None:
        assert is_implementation_task("Implement a regression test and fix the bug")
        assert is_implementation_task("Patch the vulnerability in auth.py")
        assert is_implementation_task("Run adversarial training on this model")
        assert is_implementation_task("Review X and fix any bugs you find")
        assert not is_implementation_task("Review the diff for bugs")
        assert not is_implementation_task("Verify the listed changes: a.py, b.py")

    def test_reviewer_with_implementation_task_gets_full(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, _subagent_info={"reviewer": True})
        assert agent._tool_profile("Patch the vulnerability") == "full"
        assert agent._tool_profile("Audit the vulnerability") == "review"

    def test_daemon_stamps_full_for_implementation_review_words(
        self, env: IsolatedKissHome, daemon: str, model: _FinishingModel,
    ) -> None:
        agent = _parent(env, model, max_budget=1.0)
        run_parallel = _tool(agent, "run_parallel")
        out = run_parallel(
            '["Implement a regression test for the parser", "Review the parser for bugs"]',
        )
        assert all(r["success"] for r in _child_results(out))
        assert len(model.requests) == 2
        by_task = {}
        for request in model.requests:
            prompt = request["messages"][-1]["content"]
            names = {t["function"]["name"] for t in request["tools"]}
            key = "impl" if "Implement a regression" in prompt else "review"
            by_task[key] = names
        assert "Edit" in by_task["impl"] and "run_parallel" in by_task["impl"]
        assert "Edit" not in by_task["review"] and "run_parallel" not in by_task["review"]


def test_env_float_rejects_non_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KISS_CONTEXT_LIMIT_FRACTION", "nan")
    assert Config().context_limit_fraction == 0.7
    monkeypatch.setenv("KISS_CONTEXT_LIMIT_FRACTION", "inf")
    assert Config().context_limit_fraction == 0.7


def test_review_share_in_prompt_leaves_reviewers_uncapped(
    env: IsolatedKissHome, daemon: str, model: _FinishingModel,
) -> None:
    """A real run whose prompt names a review share ("at most 10% of the
    budget for reviewing") creates no reviewer allowance: a review fan-out
    issued by that agent afterwards hands its child the plain share of the
    remaining budget, not 10% of the task budget."""
    agent = ChatSorcarAgent("share-in-prompt")
    agent.run(
        prompt_template="Say done. Use at most 10% of the budget for reviewing.",
        model_name=STANDIN_MODEL, work_dir=str(env.repo), max_steps=3, max_budget=8.0,
        model_config=model.server.model_config,
        web_tools=False, use_memory=False, verbose=False,
    )
    assert not hasattr(agent, "_review_quota")
    spent_before_fanout = float(agent.budget_used)
    run_parallel = _tool(agent, "run_parallel")
    out = run_parallel('["Review module A for bugs"]')
    assert _child_results(out)[0]["success"] is True
    assert len(model.requests) == 2
    rows = _child_rows(agent)
    assert len(rows) == 1
    # The removed allowance would have been 0.8; the plain share of the
    # remainder is (8.0 - parent spend so far) / 2.
    assert rows[0][1] > 0.8
    assert rows[0][1] == pytest.approx((8.0 - spent_before_fanout) / 2)
