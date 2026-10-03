# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of rsi7d's permission-gated scope over KISS Sorcar itself.

The pseudo-SEA ``sorcar`` (:data:`rsi7d_sea.SORCAR`) covers the plain
top-level runs, the system prompt files ``src/kiss/SYSTEM.md`` /
``SYSTEM_LITE.md``, the user's ``~/.kiss/AGENTS.md`` and the code under
``src/kiss``.  ``request_sorcar_permission`` must grant a target — from a
permitting sentence of the task text, or by asking the user — before
``patch_sorcar`` changes it.  The tests run against a fake KISS checkout
under ``tmp_path`` (a git repository where the scope needs one) and the
temporary ``KISS_HOME`` of the test session (so ``AGENTS.md`` and the
task history are test-local).
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas import agents_md
from kiss.agents.seas.rsi7d import rsi7d_sea as sea
from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.persistence import _add_task, _flush_chat_events, _save_task_result
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve, tool_call_body
from kiss.tests.server.parallel_agent_harness import init_repo, run_git

_SEA_PATH = Path(sea.__file__).resolve()

_DEMO_SEA = '''"""Demo SEA."""

SYSTEM_PROMPT = "You run demo tasks. " * 20


def system_prompt() -> str:
    """Replace the default prompt."""
    return SYSTEM_PROMPT
'''

_SYSTEM_MD = (
    "{{IDENTITY}}\n\n## Rules\n- Read before you edit.\n- Batch independent commands.\n"
)
_CODE = 'def greet(name: str) -> str:\n    """Greet."""\n    return "hi " + name\n'
_PERMIT = "You may modify KISS Sorcar itself (SYSTEM.md, AGENTS.md and the code) without asking."


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake KISS checkout (a git repository) resolved through the cwd; returns its root."""
    pkg = tmp_path / "src" / "kiss"
    seas = pkg / "agents" / "seas"
    (seas / "rsi7d").mkdir(parents=True)
    (seas / "demo").mkdir()
    shutil.copy(_SEA_PATH, seas / "rsi7d" / "rsi7d_sea.py")
    (seas / "demo" / "demo_sea.py").write_text(_DEMO_SEA, encoding="utf-8")
    (pkg / "SYSTEM.md").write_text(_SYSTEM_MD, encoding="utf-8")
    (pkg / "SYSTEM_LITE.md").write_text("{{IDENTITY}}\n\nBe brief.\n", encoding="utf-8")
    (pkg / "greet.py").write_text(_CODE, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
    init_repo(tmp_path)
    run_git(tmp_path, "add", "-A")
    run_git(tmp_path, "commit", "-q", "-m", "fake checkout")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _fresh_permission_state() -> Any:
    """Every test starts without grants and without a stored AGENTS.md."""
    sea._granted.clear()
    agents_md.agents_md_path().unlink(missing_ok=True)
    yield
    sea._granted.clear()
    agents_md.agents_md_path().unlink(missing_ok=True)


def _persist(prompt: str, **extra: object) -> str:
    """Persist a finished task row with the given columns; return its id."""
    now_ms = int(time.time() * 1000)
    payload: dict[str, object] = {
        "model": "model-a", "work_dir": "/work/dir", "startTs": now_ms - 60_000,
        "endTs": now_ms, "cost": 0.5, "steps": 5, "tokens": 10_000,
    }
    payload.update(extra)
    task_id, _chat = _add_task(prompt, extra=payload)
    _flush_chat_events(task_id)
    _save_task_result("success: true\nsummary: done", task_id=task_id)
    return task_id


class _Registered:
    """Register a real agent as the running task of the calling thread."""

    def __init__(self, task: str, ask: Any = None, work_dir: str = "") -> None:
        self.agent = WorktreeSorcarAgent("rsi7d-sorcar-scope")
        self.agent.task_description = task
        self.agent.work_dir = work_dir
        self.agent._ask_user_question_callback = ask
        self.state = agent_state.AgentState(
            "rsi7d-scope-task", agent=self.agent, tab_id="scope_tab",
            task_thread=threading.current_thread(),
        )

    def __enter__(self) -> WorktreeSorcarAgent:
        agent_state.register(self.state)
        return self.agent

    def __exit__(self, *exc: object) -> None:
        agent_state.unregister(self.state.task_id, self.state)


def test_plain_top_level_runs_are_mined_as_the_sorcar_pseudo_sea(checkout: Path) -> None:
    """Runs without a SEA and a parent are KISS Sorcar's own; children and SEA runs are not."""

    # The task DB is shared by the whole pytest session, so earlier tests'
    # plain runs are KISS Sorcar's own as well: assert relative to them
    # (the stats count every run; the listing is capped and newest first, so
    # this test's run starts later than any row another test may have left).
    entry = json.loads(sea.sea_runs(name=sea.SORCAR))["seas"].get(sea.SORCAR)
    runs_before = entry["stats"]["runs"] if entry else 0
    plain = _persist("Refactor the parser", startTs=int(time.time() * 1000) + 3_600_000)
    sea_run = _persist("Review this paper", sea="demo_sea")
    child = _persist("Reviewer sub-task", parent_task_id=plain)
    runs = json.loads(sea.sea_runs(name=sea.SORCAR))["seas"][sea.SORCAR]
    mine = [r for r in runs["runs"] if r["task_id"] == plain]
    assert len(mine) == 1 and mine[0]["children"] == 1, runs["runs"]
    assert runs["agents"] == [sea.SORCAR_AGENT_LABEL]
    assert runs["stats"]["runs"] == runs_before + 1
    everything = json.loads(sea.sea_runs())["seas"]
    listed = {tid for entry in everything.values() for r in entry["runs"] for tid in [r["task_id"]]}
    assert plain in listed and sea_run not in listed and child not in listed
    scanned = json.loads(sea.sea_findings(sea.SORCAR, runs=1000))["runs_scanned"]
    assert plain in scanned and sea_run not in scanned and child not in scanned


def test_indexed_seas_describes_sorcar_with_its_targets_and_grants(checkout: Path) -> None:
    """The ``sorcar`` row names the editable SYSTEM.md, the targets and the grants so far."""
    rows = {r["name"]: r for r in json.loads(sea.indexed_seas())["seas"]}
    row = rows[sea.SORCAR]
    assert row["editable_path"] == str(checkout / "src" / "kiss" / "SYSTEM.md")
    assert row["prompt_constant"] == "" and row["prompt_chars"] == len(_SYSTEM_MD)
    assert row["targets"] == ["SYSTEM.md", "SYSTEM_LITE.md", "AGENTS.md", "src/kiss/**/*.py"]
    assert row["permission"].startswith("required") and row["granted"] == []
    with _Registered("Sweep. Additional instructions: " + _PERMIT):
        sea.request_sorcar_permission("SYSTEM.md", "evidence", prompt_quote=_PERMIT)
    rows = {r["name"]: r for r in json.loads(sea.indexed_seas())["seas"]}
    assert rows[sea.SORCAR]["granted"] == [str(checkout / "src" / "kiss" / "SYSTEM.md")]


def test_indexed_seas_marks_sorcar_not_editable_outside_a_git_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a git checkout only AGENTS.md is in scope; SYSTEM.md is reported, not editable."""
    pkg = tmp_path / "src" / "kiss"
    (pkg / "agents" / "seas").mkdir(parents=True)
    (pkg / "SYSTEM.md").write_text(_SYSTEM_MD, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    row = {r["name"]: r for r in json.loads(sea.indexed_seas())["seas"]}[sea.SORCAR]
    assert row["editable_path"] == "" and row["prompt_chars"] == len(_SYSTEM_MD)
    assert sea.sorcar_text("SYSTEM.md") == (
        f"Error: {pkg.resolve()} is not inside a git checkout; only AGENTS.md can be changed here"
    )
    assert sea.sorcar_text("AGENTS.md") == f"({agents_md.agents_md_path()} does not exist yet)"


def test_sorcar_text_resolves_prompt_files_code_and_rejects_the_rest(checkout: Path) -> None:
    """Targets are the prompt files, AGENTS.md and code under src/kiss; SEA files are refused."""
    pkg = checkout / "src" / "kiss"
    assert sea.sorcar_text() == _SYSTEM_MD
    assert sea.sorcar_text("SYSTEM_LITE.md") == "{{IDENTITY}}\n\nBe brief.\n"
    assert sea.sorcar_text("src/kiss/greet.py") == _CODE
    assert sea.sorcar_text(str(pkg / "greet.py")) == _CODE
    assert sea.sorcar_text("AGENTS.md") == f"({agents_md.agents_md_path()} does not exist yet)"
    agents_md.add_instruction("Always answer in French")
    assert "- Always answer in French" in sea.sorcar_text("AGENTS.md")
    assert sea.sorcar_text("pyproject.toml") == (
        f"Error: 'pyproject.toml' is not a file of {pkg.resolve()}"
    )
    assert sea.sorcar_text("src/kiss/../../pyproject.toml").startswith("Error: ")
    assert sea.sorcar_text("src/kiss/agents/seas/demo/demo_sea.py") == (
        "Error: 'src/kiss/agents/seas/demo/demo_sea.py' is a SEA file; SEA prompts change "
        "through patch_sea_prompt"
    )
    assert sea.sorcar_text("src/kiss/missing.py") == "Error: 'src/kiss/missing.py' does not exist"


def test_permission_from_the_task_text_needs_a_whole_permitting_sentence(
    checkout: Path,
) -> None:
    """A quote counts only as a whole sentence of the task text that permits changing the
    requested targets; tails of prohibitions, mentions and other files' grants do not."""
    task = (
        "Improve every SEA.\n\nAdditional instructions: all. " + _PERMIT
        + " Do not touch the vscode extension. Ask me before changing SYSTEM_LITE.md.\n"
        "Analyze KISS Sorcar performance. You can also change ~/.kiss/AGENTS.md\n"
        "Do not modify KISS Sorcar itself."
    )
    not_whole = (
        "the quote is not a whole sentence of the user's task text (quote the complete "
        "sentence, verbatim)"
    )
    forbids = "the quote forbids or conditions the change instead of permitting it"
    with _Registered(task):
        assert sea.request_sorcar_permission("SYSTEM.md", "why", prompt_quote=_PERMIT) == (
            f"Permission granted for SYSTEM.md by the user's task text: {_PERMIT!r}."
        )
        # Blank runs and the final period do not matter; the words must be the user's.
        squeezed = " You  may modify KISS Sorcar itself (SYSTEM.md, AGENTS.md and the code) "
        assert sea.request_sorcar_permission(
            "AGENTS.md, src/kiss/greet.py", "why", prompt_quote=squeezed + "without asking"
        ) == (
            "Permission granted for AGENTS.md, src/kiss/greet.py by the user's task text: "
            f"{_PERMIT.rstrip('.')!r}."
        )
        # A newline ends a sentence like a period does.
        assert sea.request_sorcar_permission(
            "AGENTS.md", "why", prompt_quote="You can also change ~/.kiss/AGENTS.md"
        ).startswith("Permission granted for AGENTS.md by")
        for quote, why in (
            ("Modify KISS Sorcar freely.", not_whole),
            ("You may", "the quote is too short to be a permission sentence"),
            # The tail of a prohibition, and the tail of the permitting sentence.
            ("modify KISS Sorcar itself.", not_whole),
            ("KISS Sorcar itself (SYSTEM.md, AGENTS.md and the code) without asking.", not_whole),
            ("Do not touch the vscode extension.", forbids),
            ("Ask me before changing SYSTEM_LITE.md.", forbids),
            ("Do not modify KISS Sorcar itself.", forbids),
            (
                "Analyze KISS Sorcar performance.",
                "the quote does not permit anything (no 'may', 'can', 'allowed', "
                "'without asking')",
            ),
            (
                "You can also change ~/.kiss/AGENTS.md",
                "the quote does not mention SYSTEM_LITE.md (nor KISS Sorcar as a whole)",
            ),
        ):
            assert sea.request_sorcar_permission("SYSTEM_LITE.md", "why", prompt_quote=quote) == (
                f"Error: {why}; ask the user instead (call again without prompt_quote)"
            ), quote
        assert sea.request_sorcar_permission("", "why", prompt_quote=_PERMIT) == (
            "Error: name at least one target"
        )
        assert sea.request_sorcar_permission(
            "SYSTEM.md, src/kiss/nope.py", "why", prompt_quote=_PERMIT
        ) == "Error: 'src/kiss/nope.py' does not exist"
    assert sorted(sea._granted) == sorted(
        str(p) for p in (
            checkout / "src" / "kiss" / "SYSTEM.md", checkout / "src" / "kiss" / "greet.py",
            agents_md.agents_md_path(),
        )
    )
    # Outside a task nothing can be quoted or asked.
    assert sea.request_sorcar_permission("SYSTEM_LITE.md", "why", prompt_quote=_PERMIT) == (
        f"Error: {not_whole}; ask the user instead (call again without prompt_quote)"
    )
    assert sea.request_sorcar_permission("SYSTEM_LITE.md", "why") == (
        "Denied: the user cannot be asked from here; report the change as a recommendation."
    )


def test_permission_is_asked_from_the_user_and_read_strictly(checkout: Path) -> None:
    """The user sees the targets and the reason; only a plain yes grants, and only those targets."""
    questions: list[str] = []
    answers = iter([
        "Yes, go ahead.", "yes but only AGENTS.md", "Yes, if I approve the diff first.",
        "Yes, never change SYSTEM.md.", "no", "ok",
    ])

    def ask(question: str) -> str:
        questions.append(question)
        return next(answers)

    with _Registered("Improve every SEA.\n\nAdditional instructions: all", ask=ask):
        assert sea.request_sorcar_permission(
            "SYSTEM.md", "bullets X and Y (task abc, entry 4)"
        ) == "Permission granted for SYSTEM.md by the user's answer 'Yes, go ahead.'."
        assert questions[-1] == (
            "rsi7d asks permission to change KISS Sorcar itself.\nFiles: SYSTEM.md\n"
            "Change and evidence: bullets X and Y (task abc, entry 4)\n"
            "Answer yes to allow exactly these changes; anything else (no, or what you allow "
            "instead) denies them."
        )
        conditional = sea.request_sorcar_permission("SYSTEM_LITE.md, AGENTS.md", "z")
        assert conditional == (
            "Denied by the user: 'yes but only AGENTS.md'. Do not make these changes; report "
            "them as recommendations, or ask again with only the targets the answer allows."
        )
        for qualified in ("Yes, if I approve the diff first.", "Yes, never change SYSTEM.md."):
            assert sea.request_sorcar_permission("SYSTEM_LITE.md", "z").startswith(
                f"Denied by the user: {qualified!r}."
            )
        assert sea.request_sorcar_permission("SYSTEM_LITE.md", "z").startswith(
            "Denied by the user: 'no'."
        )
        assert sea.request_sorcar_permission("AGENTS.md", "z") == (
            "Permission granted for AGENTS.md by the user's answer 'ok'."
        )
        assert len(questions) == 6
        assert sea.patch_sorcar("SYSTEM_LITE.md", "Be brief.", "Be terse.") == (
            "Error: no permission to change SYSTEM_LITE.md; call request_sorcar_permission first"
        )
    assert (checkout / "src" / "kiss" / "SYSTEM_LITE.md").read_text(encoding="utf-8").endswith(
        "Be brief.\n"
    )
    # An unattended (cron) sweep never asks: the callback is not even called.
    with _Registered(cron_agent.PROMPT_PREAMBLE + "Run /rsi7d all", ask=ask):
        assert sea.request_sorcar_permission("SYSTEM_LITE.md", "z") == (
            "Denied: this sweep runs unattended (scheduled automation) and nobody can grant "
            "permission. Do not ask again; report the change as a recommendation."
        )
    assert len(questions) == 6
    # A task with nobody to answer (no callback) is denied too.
    with _Registered("Improve every SEA."):
        assert sea.request_sorcar_permission("SYSTEM_LITE.md", "z") == (
            "Denied: the user cannot be asked from here; report the change as a recommendation."
        )


def test_patch_sorcar_edits_granted_prompt_code_and_agents_md_targets(checkout: Path) -> None:
    """Granted targets are edited in place (once, compiling, backed up); the rest is refused."""
    pkg = checkout / "src" / "kiss"
    assert sea.patch_sorcar("SYSTEM.md", "Read before you edit.", "Read first.") == (
        "Error: no permission to change SYSTEM.md; call request_sorcar_permission first"
    )
    assert sea.patch_sorcar("pyproject.toml", "", "x") == (
        f"Error: 'pyproject.toml' is not a file of {pkg.resolve()}"
    )
    with _Registered("Sweep. Additional instructions: " + _PERMIT, work_dir=str(checkout)):
        sea.request_sorcar_permission(
            "SYSTEM.md\nsrc/kiss/greet.py\nAGENTS.md", "why", prompt_quote=_PERMIT
        )
        # The system prompt: exactly-once replacement, append, and the two failure modes.
        system_md = pkg / "SYSTEM.md"
        assert sea.patch_sorcar("SYSTEM.md", "- Read before you edit.", "- Read first.") == (
            f"Patched {system_md} ({len(_SYSTEM_MD)} -> {len(_SYSTEM_MD) - 10} chars); "
            f"revert with `git checkout -- {system_md}`"
        )
        assert sea.patch_sorcar("SYSTEM.md", "- ", "* ") == (
            f"Error: `old` occurs 2 times in {system_md}; it must occur exactly once"
        )
        assert sea.patch_sorcar("SYSTEM.md", "gone", "x") == (
            f"Error: `old` occurs 0 times in {system_md}; it must occur exactly once"
        )
        section = "## Lessons from recent runs (rsi7d)\n- Cite the task id."
        assert sea.patch_sorcar("SYSTEM.md", "", section + "\n\n").startswith("Patched ")
        assert system_md.read_text(encoding="utf-8") == (
            "{{IDENTITY}}\n\n## Rules\n- Read first.\n- Batch independent commands.\n\n"
            + section + "\n"
        )
        assert sea._sorcar_system_prompt().startswith("You are KISS Sorcar")
        assert "{{IDENTITY}}" not in sea._sorcar_system_prompt()
        # Code: a change that does not compile is refused and leaves the file alone.
        code = pkg / "greet.py"
        broken = sea.patch_sorcar("src/kiss/greet.py", 'return "hi " + name', "return (")
        assert broken.startswith("Error: greet.py would not compile: ")
        assert code.read_text(encoding="utf-8") == _CODE
        assert sea.patch_sorcar(
            "src/kiss/greet.py", 'return "hi " + name', 'return f"hi {name}"'
        ).startswith(f"Patched {code} ")
        assert code.read_text(encoding="utf-8") == _CODE.replace(
            'return "hi " + name', 'return f"hi {name}"'
        )
        assert run_git(checkout, "status", "--porcelain").stdout.split() == [
            "M", "src/kiss/SYSTEM.md", "M", "src/kiss/greet.py",
        ]
        # AGENTS.md: bullets through the storage layer, backed up before the first change.
        assert sea.patch_sorcar("AGENTS.md", "", "") == (
            "Error: give `old` (the bullet to remove), `new` (the bullet to add) or both"
        )
        backup = checkout / "tmp" / "rsi7d" / "AGENTS.md.before"
        md = agents_md.agents_md_path()
        assert sea.patch_sorcar("AGENTS.md", "", "Always answer in French") == (
            f"Remembered in {md}: Always answer in French"
        )
        assert agents_md.read_instructions() == ["Always answer in French"]
        assert not backup.exists()  # there was no file to back up
        assert sea.patch_sorcar("AGENTS.md", "always answer in french", "Answer in French") == (
            f"Forgot from {md}: Always answer in French Remembered in {md}: Answer in French"
        )
        assert agents_md.read_instructions() == ["Answer in French"]
        assert backup.read_text(encoding="utf-8").rstrip().endswith("- Always answer in French")
        assert sea.patch_sorcar("AGENTS.md", "Answer in French", "") == (
            f"Forgot from {md}: Answer in French"
        )
        assert agents_md.read_instructions() == []
        assert backup.read_text(encoding="utf-8").rstrip().endswith("- Always answer in French")
    # Ungranted SYSTEM_LITE.md stays refused even after the other grants.
    assert sea.patch_sorcar("SYSTEM_LITE.md", "", "x").startswith("Error: no permission")


def test_agent_run_asks_the_user_through_the_tool_and_patches_only_what_was_granted(
    checkout: Path,
) -> None:
    """A real ReAct loop: the model asks for permission (the user says yes), patches SYSTEM.md,
    is refused on AGENTS.md (never granted) and finishes."""
    questions: list[str] = []

    def ask(question: str) -> str:
        questions.append(question)
        return "yes"

    script = [
        tool_call_body(
            "request_sorcar_permission",
            {"targets": "SYSTEM.md", "reason": "Runs abc and def re-read files (entries 3, 9)."},
            prompt_tokens=500,
        ),
        tool_call_body(
            "patch_sorcar",
            {"target": "SYSTEM.md", "old": "", "new": "## Lessons from recent runs (rsi7d)\n- X."},
            prompt_tokens=600,
        ),
        tool_call_body(
            "patch_sorcar", {"target": "AGENTS.md", "old": "", "new": "Prefer uv."},
            prompt_tokens=700,
        ),
        finish_body("<p>Patched SYSTEM.md.</p>", prompt_tokens=800),
    ]
    with serve(script) as (url, requests):
        agent = WorktreeSorcarAgent("rsi7d-sorcar-scope-run")
        state = agent_state.AgentState(
            "rsi7d-scope-run", agent=agent, tab_id="scope_run_tab",
            task_thread=threading.current_thread(),
        )
        agent_state.register(state)
        try:
            result = agent.run(
                prompt_template="all",
                model_name=MODEL,
                work_dir=str(checkout),
                max_steps=6,
                max_budget=sea.max_budget(),
                model_config={"base_url": url, "api_key": "local"},
                tools=sea.tools(),
                base_system_prompt=sea.system_prompt(),
                web_tools=False,
                use_memory=False,
                is_parallel=False,
                verbose=False,
                ask_user_question_callback=ask,
                use_worktree=False,
                auto_commit=False,
            )
        finally:
            agent_state.unregister(state.task_id, state)
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert questions == [
        "rsi7d asks permission to change KISS Sorcar itself.\nFiles: SYSTEM.md\n"
        "Change and evidence: Runs abc and def re-read files (entries 3, 9).\n"
        "Answer yes to allow exactly these changes; anything else (no, or what you allow "
        "instead) denies them."
    ]
    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 4, [list(r) for r in requests]
    tool_results = [
        str([m for m in r["messages"] if m["role"] == "tool"][-1]["content"]) for r in agentic[1:]
    ]
    assert tool_results[0].startswith(
        "Permission granted for SYSTEM.md by the user's answer 'yes'."
    )
    assert tool_results[1].startswith("Patched ")
    assert tool_results[2].startswith(
        "Error: no permission to change AGENTS.md; call request_sorcar_permission first"
    )
    assert (checkout / "src" / "kiss" / "SYSTEM.md").read_text(encoding="utf-8").endswith(
        "## Lessons from recent runs (rsi7d)\n- X.\n"
    )
    assert not agents_md.agents_md_path().exists()
