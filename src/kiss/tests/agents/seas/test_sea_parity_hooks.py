# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: one SEA runs the same in a tab and headless.

A fixture SEA declares a tool's JSON schema (``@tool_schema``), ends the
run from ``tool_call_hook`` (a domain rule) and from ``llm_response_hook``
(a token cap), and asks for plain model I/O through its ``model_config``
(``plain_io``) next to the ``base_url`` that routes the run's calls to the
scripted local model.  The tab path runs it through ``ChatSorcarAgent``
configured as the daemon configures a SEA run; the headless path runs it
through :func:`kiss.agents.sorcar.headless.run_sea`.  What the model
reads is asserted on the wire, never in memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.base.base_sea import BaseSea, LLMReply, Verdict, refuse
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.headless import run_sea
from kiss.agents.sorcar.sea_commands import evaluate_sea, load_sea, sea_check
from kiss.agents.sorcar.sea_settings import SeaError
from kiss.core import config as config_module
from kiss.tests.agents.seas.sea_contract import system_message, tool_results
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    text_body,
    tool_call_body,
)

URL_ENV = "PARITY_SEA_URL"
MODE_ENV = "PARITY_SEA_MODE"
PROFILE_ENV = "PARITY_SEA_PROFILE"
TOKEN_CAP = 150

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["low", "high"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "line": {"type": "integer"}},
                "required": ["file", "line"],
            },
        },
    },
    "required": ["severity", "findings"],
}

SEA_SOURCE = f'''\
import os
from typing import Any

from kiss.agents.seas.base.base_sea import ALLOW, LLMReply, Verdict, WorkerSea, end_run, tool_schema

REPORT_SCHEMA = {REPORT_SCHEMA!r}
TOKEN_CAP = {TOKEN_CAP}


class ParitySea(WorkerSea):
    """Fixture SEA: a declared tool schema, run-ending hooks, plain model I/O."""

    def __init__(self) -> None:
        super().__init__()
        self.filed: list[dict[str, Any]] = []
        self.output_tokens = 0

    def description(self) -> str:
        return "Review a patch and file the findings."

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {{
            "model": {MODEL!r},
            "tool_profile": "none",
            "model_config": {{
                "base_url": os.environ[{URL_ENV!r}], "api_key": "local", "plain_io": True,
            }},
        }}

    def tools(self, tools: list) -> list:
        if os.environ.get({MODE_ENV!r}) == "broken-tools":
            return [42]
        return tools + [self.file_report]

    @tool_schema(parameters=REPORT_SCHEMA, description="File the review's findings.")
    def file_report(self, severity: str, findings: list[dict[str, Any]]) -> str:
        """Derived description, which the declared one replaces."""
        self.filed.append({{"severity": severity, "findings": findings}})
        return f"filed {{len(findings)}} {{severity}} finding(s)"

    def tool_call_hook(self, name: str, args: dict[str, Any]) -> Verdict:
        if name == "file_report" and args.get("severity") == "high":
            return end_run("<p>Report accepted: the episode is over.</p>")
        return ALLOW

    def llm_response_hook(self, reply: LLMReply) -> Verdict:
        self.output_tokens += reply.usage["output_tokens"]
        if os.environ.get({MODE_ENV!r}) == "token-cap" and self.output_tokens > TOKEN_CAP:
            cap = f"<p>Token cap: {{self.output_tokens}} output tokens.</p>"
            return end_run(cap, success=False)
        return ALLOW
'''


PROFILE_SEA_SOURCE = f'''\
import os
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea


class ProfileSea(WorkerSea):
    """Fixture SEA: the tool profile of the environment, the classifier asked for."""

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {{
            "model": {MODEL!r},
            "tool_profile": os.environ[{PROFILE_ENV!r}],
            "auto_classify": True,
            "model_config": {{"base_url": os.environ[{URL_ENV!r}], "api_key": "local"}},
        }}
'''


@pytest.fixture
def sea_path(tmp_path: Path) -> Path:
    """Write the fixture SEA as ``parity/parity_sea.py`` and return its path."""
    folder = tmp_path / "parity"
    folder.mkdir()
    path = folder / "parity_sea.py"
    path.write_text(SEA_SOURCE, encoding="utf-8")
    return path


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A clean mode for every test; the URL is set per server."""
    monkeypatch.delenv(MODE_ENV, raising=False)
    return monkeypatch


def _report_call(severity: str, prompt_tokens: int) -> bytes:
    return tool_call_body(
        "file_report",
        {"severity": severity, "findings": [{"file": "a.py", "line": 3}]},
        prompt_tokens,
    )


def _tool_schema_on_the_wire(request: dict[str, Any], name: str) -> dict[str, Any]:
    return next(t["function"] for t in request["tools"] if t["function"]["name"] == name)


def _trajectory_files() -> set[Path]:
    folder = Path(config_module.get_artifact_dir()) / "trajectories"
    return set(folder.glob("*.yaml")) if folder.is_dir() else set()


class TestTabPath:
    """The SEA through ``ChatSorcarAgent``, configured as the daemon configures a run."""

    def test_schema_plain_io_and_end_run_from_the_tool_hook(
        self, sea_path: Path, env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        sea = load_sea(sea_path)
        script = [_report_call("low", 1000), _report_call("high", 1200)]
        with serve(script) as (url, requests):
            env.setenv(URL_ENV, url)
            run = evaluate_sea([sea], "Review the patch.")
            settings = run.settings
            result = ChatSorcarAgent("parity-tab").run(
                prompt_template=run.prompt,
                model_name=settings["model"],
                work_dir=str(tmp_path),
                max_steps=6,
                max_budget=1.0,
                model_config=settings["model_config"],
                system_prompt_hook=run.system_prompt_hook,
                tools_hook=run.tools_hook,
                tool_call_hook=run.tool_call_hook,
                llm_call_hook=run.llm_call_hook,
                llm_response_hook=run.llm_response_hook,
                tool_profile=settings["tool_profile"],
                web_tools=False,
                use_memory=False,
                verbose=False,
            )
        agentic = [r for r in requests if r.get("tools")]
        assert len(agentic) == 2, "The run ended at the second reply's call: no third model call"
        # R1: the declared schema reaches the model verbatim, under the function's name.
        declared = _tool_schema_on_the_wire(agentic[0], "file_report")
        assert declared["parameters"] == REPORT_SCHEMA
        assert declared["description"] == "File the review's findings."
        assert "finish" in {t["function"]["name"] for t in agentic[0]["tools"]}
        # R3: no framework trailer after the base prompt (this SEA keeps it:
        # no ``system_prompt``), no usage line on the tool result.
        system = system_message(agentic[0])
        assert "# Task Settings" not in system
        assert "# MOST IMPORTANT INSTRUCTIONS" not in system
        assert "# Restricted tool profile" not in system
        (first_result,) = tool_results(agentic[1])
        assert first_result == "filed 1 low finding(s)"
        # R2: the domain rule ended the run; the high-severity call did not run.
        assert vars(sea)["filed"] == [
            {"severity": "low", "findings": [{"file": "a.py", "line": 3}]}
        ]
        assert "Report accepted: the episode is over." in result
        assert "success: true" in result

    def test_default_run_keeps_task_settings_and_usage_line(self, tmp_path: Path) -> None:
        """Without ``plain_io`` the model still reads both additions."""
        run = evaluate_sea([BaseSea()], "Say hello.")
        script = [
            tool_call_body("Bash", {"command": "echo hi", "description": "greet"}, 1000),
            tool_call_body(
                "finish", {"success": True, "is_continue": False, "summary_in_html": "<p>hi</p>"},
                1200,
            ),
        ]
        with serve(script) as (url, requests):
            ChatSorcarAgent("parity-default").run(
                prompt_template=run.prompt,
                model_name=MODEL,
                work_dir=str(tmp_path),
                max_steps=4,
                max_budget=1.0,
                model_config={"base_url": url, "api_key": "local"},
                system_prompt_hook=run.system_prompt_hook,
                tools_hook=run.tools_hook,
                tool_call_hook=run.tool_call_hook,
                llm_call_hook=run.llm_call_hook,
                llm_response_hook=run.llm_response_hook,
                tool_profile="bash",
                web_tools=False,
                use_memory=False,
                verbose=False,
            )
        agentic = [r for r in requests if r.get("tools")]
        system = system_message(agentic[0])
        assert "# Task Settings" in system
        assert "# MOST IMPORTANT INSTRUCTIONS" in system
        assert "# Restricted tool profile: bash" in system
        (bash_result,) = tool_results(agentic[1])
        assert "Steps: 1/" in bash_result and "Budget:" in bash_result


class TestHeadlessPath:
    """The SEA through ``run_sea``: outcome, counters, trajectory files."""

    def test_token_cap_ends_the_run_and_every_trajectory_file_is_reported(
        self, sea_path: Path, env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env.setenv(MODE_ENV, "token-cap")
        script = [text_body("Still thinking.", 1000, 100), text_body("More thinking.", 1100, 100)]
        before = _trajectory_files()
        with serve(script) as (url, requests):
            env.setenv(URL_ENV, url)
            outcome = run_sea(sea_path, "Review the patch.", work_dir=str(tmp_path))
        assert outcome.error == ""
        assert outcome.success is False
        assert "Token cap: 200 output tokens." in outcome.result
        assert "is_continue: false" in outcome.result
        assert len([r for r in requests if r.get("tools")]) == 2
        assert outcome.total_steps == 2
        assert outcome.total_tokens_used == 1000 + 100 + 1100 + 100
        assert outcome.budget_used > 0
        # R4: every file the run wrote (here the session executor's trajectory;
        # a RelentlessAgent saves none of its own), exactly the new files on disk.
        new_files = _trajectory_files() - before
        assert set(outcome.trajectory_paths) == new_files
        names = [path.name for path in outcome.trajectory_paths]
        assert names and all(name.startswith("trajectory_parity_sea_") for name in names), names
        assert any("Session-0" in name for name in names), names
        # The SEA's plain I/O held headless too.
        assert "# Task Settings" not in system_message(requests[0])
        assert "Steps:" not in json.dumps(requests[1]["messages"])
        for path in outcome.trajectory_paths:
            path.unlink()

    def test_save_trajectories_false_writes_nothing(
        self, sea_path: Path, env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env.setenv(MODE_ENV, "token-cap")
        script = [text_body("Thinking.", 1000, 200)]
        before = _trajectory_files()
        with serve(script) as (url, _requests):
            env.setenv(URL_ENV, url)
            outcome = run_sea(
                sea_path, "Review the patch.", work_dir=str(tmp_path), save_trajectories=False
            )
        assert "Token cap: 200 output tokens." in outcome.result
        assert outcome.trajectory_paths == ()
        assert _trajectory_files() == before

    def test_a_run_that_raises_is_returned_as_an_error(
        self, sea_path: Path, env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env.setenv(MODE_ENV, "broken-tools")
        with serve([text_body("unused")]) as (url, requests):
            env.setenv(URL_ENV, url)
            outcome = run_sea(sea_path, "Review the patch.", work_dir=str(tmp_path))
        assert outcome.result == ""
        assert outcome.success is False
        assert outcome.error.startswith("SeaError: tools() of SEA ")
        assert "must return a list of tool callables" in outcome.error
        assert requests == [], "The run failed before its first model call"

    @pytest.mark.parametrize("profile", ["none", "bash"])
    def test_a_profile_without_edit_tools_runs_as_in_a_tab(
        self, profile: str, env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Its prompt is fixed, so no classifier call; ``none`` has no built-in toolset to note."""
        home = tmp_path / "home"
        home.mkdir()
        # The LLM classifier (on the SEA's model: the local server) is the only one.
        (home / "config.json").write_text(json.dumps({"classify_with_decisions": False}))
        env.setenv("KISS_HOME", str(home))
        env.delenv("KISS_DISABLE_TASK_CLASSIFIER", raising=False)
        env.setenv(PROFILE_ENV, profile)
        path = tmp_path / "profile" / "profile_sea.py"
        path.parent.mkdir()
        path.write_text(PROFILE_SEA_SOURCE, encoding="utf-8")
        with serve([finish_body("<p>done</p>")]) as (url, requests):
            env.setenv(URL_ENV, url)
            outcome = run_sea(
                path, "Fix the bug in parser.py.", work_dir=str(tmp_path), save_trajectories=False
            )
        assert outcome.error == "" and outcome.success, outcome
        assert len(requests) == 1, "the run's own call, no classifier call ahead of it"
        noted = "# Restricted tool profile" in system_message(requests[0])
        assert noted == (profile == "bash")

    def test_unknown_sea_name_lists_the_known_ones(self) -> None:
        with pytest.raises(SeaError, match=r"unknown SEA 'no-such-sea'; known names: .*sh"):
            run_sea("no-such-sea", "task")


class TestContract:
    """The SEA layer's rules for the new hook."""

    def test_refuse_from_the_response_hook_is_a_broken_sea(self) -> None:
        class RefusingSea(BaseSea):
            def llm_response_hook(self, reply: LLMReply) -> Verdict:
                return refuse("no")

        run = evaluate_sea([RefusingSea()], "task")
        broken = r"llm_response_hook\(\) .* must return ALLOW or end_run"
        with pytest.raises(SeaError, match=broken):
            run.llm_response_hook(LLMReply("hello"))

    def test_sea_check_reports_the_response_hook_and_the_declared_tool(
        self, sea_path: Path, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv(URL_ENV, "http://127.0.0.1:1/v1")
        report = sea_check("parity", sea_path)
        assert "llm_response_hook" in report.split("methods defined:")[1].split("\n")[0]
        assert "tool_call_hook" in report
        assert "tools added: file_report" in report
        assert os.environ[URL_ENV] in report
        assert sea_commands.defines([load_sea(sea_path)], "llm_response_hook")
