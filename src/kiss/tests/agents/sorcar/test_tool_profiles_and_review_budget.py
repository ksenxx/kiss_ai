# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for tool profiles (WP1b), the review budget cap and
``run_parallel(model_name=..., tool_profile=...)`` (WP3), and the cost-lever
config toggles (WP0).

The fan-out tests spawn real ``ChatSorcarAgent`` children against the
scripted local model server and read the children's state back.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import sorcar_agent as sa
from kiss.agents.sorcar.agent_dispatch import _dispatch
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.fanout_guard import (
    REVIEW_BUDGET_REFUSAL,
    REVIEW_CAP_REFUSAL,
    ReviewQuota,
    is_implementation_task,
    review_budget_for,
    review_budget_from_prompt,
)
from kiss.agents.sorcar.sorcar_agent import TOOL_PROFILES, SorcarAgent
from kiss.core.config import DEFAULT_CONFIG, Config
from kiss.core.kiss_error import BudgetExceededError
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve


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
        monkeypatch.setenv("KISS_REVIEW_BUDGET_FRACTION", "junk")
        monkeypatch.setenv("KISS_READ_OUTLINE_LINES", "1500")
        monkeypatch.setenv("KISS_CHAT_HISTORY_DIGEST", "")
        cfg = Config()
        assert cfg.read_dedupe is False
        assert cfg.tool_output_compaction is False
        assert cfg.tool_profiles is True
        assert cfg.context_limit_fraction == 0.85
        assert cfg.review_budget_fraction == 0.0  # junk falls back to the default
        assert cfg.read_outline_lines == 1500
        assert cfg.chat_history_digest is True  # empty = default
        assert cfg.dispatch_path_rewrite is True
        monkeypatch.setenv("KISS_READ_OUTLINE_LINES", "x")
        assert Config().read_outline_lines == 2000

    def test_defaults_are_on(self) -> None:
        for name in ("read_dedupe", "tool_output_compaction", "tool_profiles",
                     "chat_history_digest", "dispatch_path_rewrite"):
            assert getattr(DEFAULT_CONFIG, name) is True, name
        assert DEFAULT_CONFIG.context_limit_fraction == 0.7
        assert DEFAULT_CONFIG.review_budget_fraction == 0.0  # prompt-derived by default


class TestToolProfiles:
    def test_full_profile_has_everything(self, tmp_path: Path) -> None:
        names = _names(_bare_agent(tmp_path)._get_tools())
        assert {"Bash", "Read", "Edit", "Write", "run_commands_parallel", "run_agent",
                "ask_user_question", "talk", "set_model", "summary", "run_parallel",
                "number_of_cores"} <= names

    def test_review_profile_is_read_only(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, _tool_profile_name="review")
        names = _names(agent._get_tools())
        assert names <= set(TOOL_PROFILES["review"])  # type: ignore[arg-type]
        assert {"Bash", "Read", "run_commands_parallel", "summary"} <= names
        assert not names & {"Edit", "Write", "run_agent", "run_parallel", "talk",
                            "set_model", "ask_user_question"}

    def test_shell_profile(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, _tool_profile_name="shell")
        assert _names(agent._get_tools()) == {
            "Bash", "bash_job", "Read", "run_commands_parallel",
        }

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


class TestReviewQuotaBudget:
    def test_reserve_and_release(self) -> None:
        quota = ReviewQuota(budget=1.0)
        assert quota.budget_left == 1.0
        assert quota.reserve_budget(0.6) == 0.6
        assert quota.reserve_budget(0.6) == pytest.approx(0.4)
        assert quota.reserve_budget(0.1) == 0.0
        quota.release(0.5)
        assert quota.budget_left == pytest.approx(0.5)
        quota.release(-3)  # negative unspent is ignored
        assert quota.budget_left == pytest.approx(0.5)
        quota.release(99)  # cannot go above the cap
        assert quota.budget_left == 1.0

    def test_unlimited_quota(self) -> None:
        quota = ReviewQuota()
        assert quota.budget_left is None
        assert quota.reserve_budget(123.0) == 123.0
        quota.release(123.0)
        assert quota.budget_left is None

    def test_review_budget_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No allowance in the prompt and no fallback fraction: no cap.
        assert review_budget_for(40.0) is None
        assert review_budget_for(40.0, "Implement the feature and test it.") is None
        monkeypatch.setattr(DEFAULT_CONFIG, "review_budget_fraction", 0.5)
        assert review_budget_for(40.0) == 20.0
        # The prompt wins over the fallback.
        assert review_budget_for(40.0, "Use at most 25% of the budget for reviewing.") == 10.0
        monkeypatch.setattr(DEFAULT_CONFIG, "review_budget_fraction", 1.0)
        assert review_budget_for(40.0) is None

    @pytest.mark.parametrize(
        ("prompt", "expected"),
        [
            ("Use at most 50% of the task budget in gpt-5.6-sol for reviewing.", 500.0),
            ("Keep review spend to 30 percent of the budget.", 300.0),
            ("Spend no more than $40 on the review.", 40.0),
            ("Reviewers may cost 25 dollars in total.", 25.0),
            ("The review may cost 5000 USD.", 1000.0),  # clipped to the task budget
            ("Keep the reviewers under a third of the budget.", 1000.0 / 3),
            ("Give the audit half the budget.", 500.0),
            ("Implement the feature; the whole budget is $1000.", None),  # no review
            ("Review the code for 100% coverage of the budget.", None),  # not a share
            (
                "## Previous tasks\nUse at most 20% of the budget for reviewing.\n"
                "# Task (work on it now)\n\nJust implement it.",
                None,  # earlier tasks' instructions are not the current task's
            ),
            (
                "# Task (work on it now)\n\nImplement X. Use 'gpt-5.6-sol' for review; "
                "keep its spend under $12.",
                12.0,
            ),
        ],
    )
    def test_review_budget_from_prompt(self, prompt: str, expected: float | None) -> None:
        got = review_budget_from_prompt(prompt, 1000.0)
        assert got == (pytest.approx(expected) if expected is not None else None)

    def test_run_parallel_refuses_exhausted_review_budget(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, max_budget=10.0, budget_used=0.0)
        agent._review_quota = ReviewQuota(budget=0.9)
        agent._review_quota.reserve_budget(0.5)  # an earlier round took most of it
        run_parallel = _tool(agent, "run_parallel")
        out = run_parallel('["Review the diff for bugs"]')
        assert out == f"Error: {REVIEW_BUDGET_REFUSAL}"
        # The failed attempt did not keep a reservation or burn a round.
        assert agent._review_quota.budget_left == pytest.approx(0.4)
        assert agent._review_quota.used == 0

    def test_run_parallel_releases_budget_when_round_cap_refuses(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, max_budget=10.0, budget_used=0.0)
        agent._review_quota = ReviewQuota(limit=0, budget=5.0)
        run_parallel = _tool(agent, "run_parallel")
        out = run_parallel('["Audit the module"]')
        assert out == f"Error: {REVIEW_CAP_REFUSAL}"
        assert agent._review_quota.budget_left == 5.0


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


class TestFanoutPropagation:
    """Real fan-outs: the children run against the scripted server and are
    observed through the server's requests and their persisted rows."""

    def test_children_get_profile_model_and_clipped_budget(self, tmp_path: Path) -> None:
        script = [finish_body("<p>reviewed</p>", prompt_tokens=1000)]
        with serve(script) as (url, requests):
            agent = _bare_agent(tmp_path, max_budget=4.0, budget_used=0.0)
            agent.model_name = MODEL
            agent.model_config = {"base_url": url, "api_key": "local"}
            agent._review_quota = ReviewQuota(budget=0.75)
            agent._chat_id = ""
            agent._last_task_id = uuid.uuid4().hex
            run_parallel = _tool(agent, "run_parallel")
            out = run_parallel(
                '["Review module A for bugs"]', model_name=MODEL, tool_profile="review",
            )
        # run_parallel returns a YAML list of per-child YAML result strings.
        result = yaml.safe_load(yaml.safe_load(out)[0])
        assert result["success"] is True and "reviewed" in result["summary"]
        # The child reached OUR server: the parent's model_config was
        # forwarded (same model), and its request carried the review
        # toolset only.
        assert len(requests) == 1
        sent_tools = {t["function"]["name"] for t in requests[0]["tools"]}
        assert "Edit" not in sent_tools and "run_parallel" not in sent_tools
        assert {"Bash", "Read", "finish"} <= sent_tools
        # Persisted child row: the model named at dispatch and a budget
        # clipped from the plain share (4.0 / 2 = 2.0) to the 0.75 review
        # allowance.
        rows = _child_rows(agent)
        assert len(rows) == 1
        model, max_budget, task = rows[0]
        assert model == MODEL and task == "Review module A for bugs"
        assert max_budget == pytest.approx(0.75)
        # The unspent reservation went back to the quota.
        spent = float(getattr(agent, "budget_used", 0.0))
        assert spent > 0
        assert agent._review_quota.budget_left == pytest.approx(0.75 - spent, abs=1e-6)
        assert agent._review_quota.used == 1

    def test_different_model_is_dispatched_with_default_routing(self, tmp_path: Path) -> None:
        # A different model gets default provider routing, not the parent's
        # endpoint: the child never reaches the parent's local server and,
        # having no key for the real provider, fails fast with a result the
        # parent can read.
        script = [finish_body("<p>never</p>", prompt_tokens=500)]
        with serve(script) as (url, requests):
            agent = _bare_agent(tmp_path, max_budget=4.0, budget_used=0.0)
            agent.model_name = MODEL
            agent.model_config = {"base_url": url, "api_key": "local"}
            agent._chat_id = ""
            agent._last_task_id = uuid.uuid4().hex
            run_parallel = _tool(agent, "run_parallel")
            out = run_parallel('["summarize a"]', model_name="no-such-model-cost-levers")
        assert requests == []
        assert yaml.safe_load(yaml.safe_load(out)[0])["success"] is False
        rows = _child_rows(agent)
        assert rows and rows[0][0] == "no-such-model-cost-levers"


def test_child_profile_is_stamped_by_engine(tmp_path: Path) -> None:
    """``run_tasks_parallel`` stamps ``_tool_profile_name`` on every child."""
    script = [finish_body("<p>ok</p>", prompt_tokens=500)]
    with serve(script) as (url, requests):
        results = sa.run_tasks_parallel(
            ["Summarize this", "Summarize that"],
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            web_tools=False,
            use_memory=False,
            tool_profile="shell",
        )
    assert all(yaml.safe_load(r)["success"] for r in results)
    assert len(requests) == 2
    for request in requests:
        names = {t["function"]["name"] for t in request["tools"]}
        assert names == {"Bash", "bash_job", "Read", "run_commands_parallel", "finish"}
        system = next(m for m in request["messages"] if m["role"] == "system")["content"]
        assert "# Restricted tool profile: shell" in system
        assert "Bash, Read, bash_job, run_commands_parallel" in system
    assert os.environ.get("KISS_HOME")  # tests run against an isolated KISS_HOME


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

    def test_engine_stamps_full_for_implementation_review_words(self, tmp_path: Path) -> None:
        script = [finish_body("<p>ok</p>", prompt_tokens=500)]
        with serve(script) as (url, requests):
            sa.run_tasks_parallel(
                ["Implement a regression test for the parser", "Review the parser for bugs"],
                model_name=MODEL, work_dir=str(tmp_path), max_budget=1.0,
                model_config={"base_url": url, "api_key": "local"},
                web_tools=False, use_memory=False,
            )
        assert len(requests) == 2
        by_task = {}
        for request in requests:
            prompt = request["messages"][-1]["content"]
            names = {t["function"]["name"] for t in request["tools"]}
            key = "impl" if "Implement a regression" in prompt else "review"
            by_task[key] = names
        assert "Edit" in by_task["impl"] and "run_parallel" in by_task["impl"]
        assert "Edit" not in by_task["review"] and "run_parallel" not in by_task["review"]


class TestMixedFanoutBudgets:
    def test_only_reviewer_children_are_clipped_and_charged(self, tmp_path: Path) -> None:
        script = [finish_body("<p>done</p>", prompt_tokens=1000)]
        with serve(script) as (url, requests):
            agent = _bare_agent(tmp_path, max_budget=6.0, budget_used=0.0)
            agent.model_name = MODEL
            agent.model_config = {"base_url": url, "api_key": "local"}
            agent._review_quota = ReviewQuota(budget=0.8)
            agent._chat_id = ""
            agent._last_task_id = uuid.uuid4().hex
            run_parallel = _tool(agent, "run_parallel")
            out = run_parallel('["Review module A for bugs", "Summarize module B"]')
        assert len(yaml.safe_load(out)) == 2 and len(requests) == 2
        rows = {task: budget for _model, budget, task in _child_rows(agent)}
        # Plain share is 6.0 / 3 = 2.0; the reviewer is clipped to the 0.8
        # allowance, its non-review sibling keeps the plain share.
        assert rows["Review module A for bugs"] == pytest.approx(0.8)
        assert rows["Summarize module B"] == pytest.approx(2.0)
        # Only the reviewer's spend stays charged against the quota.
        spent_total = float(agent.budget_used)
        left = agent._review_quota.budget_left
        assert left is not None and 0 < 0.8 - left < spent_total

    def test_reservation_released_when_fanout_raises(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, max_budget=6.0, budget_used=0.0)
        agent._review_quota = ReviewQuota(budget=1.0)
        agent._review_quota.reserve_budget(0.5)
        # The budget is gone between the reservation and the fan-out: the
        # inner budget-share computation raises, the reservation comes back.
        agent.budget_used = 6.5
        with pytest.raises(BudgetExceededError):
            agent._run_tasks_parallel(
                ["Review x"], review_budget=0.5, review_flags=[True],
            )
        assert agent._review_quota.budget_left == pytest.approx(1.0)

    def test_explicit_review_profile_counts_as_review(self, tmp_path: Path) -> None:
        agent = _bare_agent(tmp_path, max_budget=10.0, budget_used=0.0)
        agent._review_quota = ReviewQuota(limit=0, budget=5.0)
        run_parallel = _tool(agent, "run_parallel")
        out = run_parallel('["Summarize module B"]', tool_profile="review")
        assert out == f"Error: {REVIEW_CAP_REFUSAL}"
        assert agent._review_quota.budget_left == 5.0


class TestRunAgentReviewBudget:
    def test_dispatch_reserves_and_releases_review_budget(self, tmp_path: Path) -> None:
        parent = _bare_agent(tmp_path, max_budget=10.0, budget_used=0.0)
        parent._review_quota = ReviewQuota(budget=0.3)
        # Below the minimum a child can use: refused, nothing kept.
        out = _dispatch(
            name="helper", prompt="Review the diff for regressions",
            agent_path="/nonexistent/agent.py", work_dir=str(tmp_path),
            model_name="", budget=None, timeout=1.0, parent_agent=parent,
        )
        assert out == f"Error: {REVIEW_BUDGET_REFUSAL}"
        assert parent._review_quota.budget_left == pytest.approx(0.3)
        assert parent._review_quota.used == 0
        # Enough budget: the round is reserved, the (daemonless) dispatch
        # fails, and the whole reservation comes back.
        parent._review_quota = ReviewQuota(budget=2.0)
        out = _dispatch(
            name="helper", prompt="Review the diff for regressions",
            agent_path="/nonexistent/agent.py", work_dir=str(tmp_path),
            model_name="", budget=5.0, timeout=1.0, parent_agent=parent,
        )
        assert out.startswith("Error:") and "Review-" not in out
        assert parent._review_quota.used == 1
        assert parent._review_quota.budget_left == pytest.approx(2.0)


def test_env_float_rejects_non_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KISS_REVIEW_BUDGET_FRACTION", "nan")
    assert Config().review_budget_fraction == 0.0
    monkeypatch.setenv("KISS_REVIEW_BUDGET_FRACTION", "inf")
    assert Config().review_budget_fraction == 0.0


def test_run_creates_quota_from_the_prompt_allowance(tmp_path: Path) -> None:
    """A real run: the quota's budget is what the user's prompt allows reviewers."""
    script = [finish_body("<p>done</p>", prompt_tokens=500)]
    with serve(script) as (url, _requests):
        agent = ChatSorcarAgent("quota-from-prompt")
        agent.run(
            prompt_template="Say done. Use at most 10% of the budget for reviewing.",
            model_name=MODEL, work_dir=str(tmp_path), max_steps=3, max_budget=8.0,
            model_config={"base_url": url, "api_key": "local"},
            web_tools=False, use_memory=False, verbose=False,
        )
    assert agent._review_quota is not None
    assert agent._review_quota.budget_left == pytest.approx(0.8)
    with serve(script) as (url, _requests):
        plain = ChatSorcarAgent("quota-uncapped")
        plain.run(
            prompt_template="Say done.", model_name=MODEL, work_dir=str(tmp_path),
            max_steps=3, max_budget=8.0, model_config={"base_url": url, "api_key": "local"},
            web_tools=False, use_memory=False, verbose=False,
        )
    assert plain._review_quota is not None and plain._review_quota.budget_left is None
