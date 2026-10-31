# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/discover`` agent (``kiss.agents.seas.discover.discover_sea``).

The SEA appends the AI discovery loop (once the ``## AI discovery, auto
research, and optimization`` section of ``SYSTEM.md``, now the SEA's
``DISCOVERY_INSTRUCTIONS``) to the system prompt and pins ``model:
bestrouter`` and ``tool_profile: full``, so the tests check: the default
prompts no longer carry the loop, a leftover copy in a prompt is cut out
cleanly and the loop appended once, the slash command resolves to this
file, the settings name the bestrouter picker (which the daemon resolves
to its primary model) and the full toolset, the daemon-side loader stages
the rewrite as a system-prompt hook, and a real :class:`ChatSorcarAgent`
run configured that way (against the scripted local chat-completions
server) sends the default prompt with the loop at the end and offers the
file tools.

A task that asks only for research still gets the loop, and one that
forbids changing its repository gets it in a lab copy under the scratch
dir: a real daemon sub-agent run of the SEA, scripted through the lab
procedure while its caller isolates the artifact's repository (the shape
of task 6360f17b, whose /discover sub-task was asked for research only,
with no repository changes, and skipped the loop), leaves the caller's
tree alone and hands back a patch that applies to it.

A task the user types that names ``/discover``, or follows one that did
(task 5150d948: "continue" after "You must use /discover ...", which never
called it and counted one earlier consult as enough), gets the SEA's
description after the task; the SEA's own run and a ``run_agent`` sub-task
in that chat do not, so ``/discover`` is never sent after itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.bestrouter import bestrouter_sea
from kiss.agents.seas.discover import discover_sea
from kiss.agents.seas.discover.discover_sea import DiscoverSea, split_discovery_section
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.agent_dispatch import resolve_agent
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.core.tool_verdict import ALLOW
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_no_removed_getters,
    system_message,
    tool_names,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body
from kiss.tests.server.conftest import init_repo, kiss_wt_branches, run_git
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)
from kiss.tests.server.test_autocommit_wire_toggle import _FakeCredentials

_SEA_PATH = Path(discover_sea.__file__).resolve()
_HEADING = discover_sea.SECTION_HEADING
_LAST_STEP = "delete only files you created."
"""The end of the "How to Work" list, which the discovery section used to follow."""
_NEXT_HEADING = next(
    line
    for line in SYSTEM_PROMPT[SYSTEM_PROMPT.index(_LAST_STEP) :].splitlines()
    if line.startswith("## ")
)
"""The default prompt's heading right after :data:`_LAST_STEP`, read off the prompt."""


def test_the_instructions_are_the_discovery_loop_and_left_the_default_prompt() -> None:
    """``DISCOVERY_INSTRUCTIONS`` is the loop, heading to last step; ``SYSTEM.md`` lost it."""
    instructions = discover_sea.DISCOVERY_INSTRUCTIONS
    assert instructions.startswith(_HEADING + "\n\n")
    assert instructions.endswith(
        "go to step 5 with the frontier\n   point it builds on as the parent."
    )
    assert instructions.count("\n## ") == 0
    assert [line[:3] for line in instructions.splitlines() if line[:1].isdigit()] == [
        f"{n}. " for n in range(10)
    ]
    for phrase in (
        # A research-only task still gets the loop; a no-change task gets it in a lab.
        "A task that asks only for research, ideas or a ranking still gets this loop, "
        "ranked by\nmeasured dev metrics",
        "When the task forbids changing its repository, work in a lab so\n"
        "that no `git reset` touches that tree",
        "copy the artifact's current files (not its data)\nto <scratch dir>/lab, `git init` "
        "and commit them there, and call create_worktree on it;\nrun every round in the "
        "worktree it returns (in the lab when it makes none)",
        "in a lab, also write\n"
        "   `git diff --binary <B commit> <its commit>` to <scratch dir>/best.patch",
        # Every point is a commit with a parent, held by a ref that outlives the
        # resets and the squash merge; the files carry both.
        "so that every evaluated idea is a commit,\n"
        "kept reachable through resets and squash merges by its ref "
        "refs/discover/<task id>/<id>",
        # The held-out ban is up front and concrete: two real runs counted the held-out
        # rows in their very first "inspect the repo" command.  It starts when the part
        # exists, so the split of step 1 may still write it.
        "The held-out part (named by the task, else written by the split of step 1) is off "
        "limits\nfrom the moment it exists, the first command included: nothing but the "
        "held-out check of\nsteps 5 and 6 opens it, no cat, head, wc, cut, count, profile or "
        "test.",
        "one line per frontier point: id, parent id, commit, idea, aspect (the part of the\n"
        "   system the idea changes",
        "<scratch dir>/pareto.md,\n   one line per frontier point",
        # explored-ideas.md is the one log: appended in order, with the draw and the
        # status, which pareto.md no longer duplicates.
        "one row appended per idea in the order evaluated, with the same fields plus draw (the\n"
        "   step 6 draw that led to it, with a fresh draw's second draw), status and "
        "observations",
        "A point's status lives there alone: a point that leaves the frontier leaves "
        "pareto.md.",
        # Dev metrics only; the split and the noise floor come first, cover deterministic
        # commands, and are not amended mid-run (a real run relaxed its floor after round 2).
        "1. Fix the dev / held-out split (the task's when it names one)",
        "<scratch dir>/split.md",
        "the exact dev evaluation command, and its noise floor (the spread\n   of two or three "
        "baseline runs, varied by seed or fold assignment when the command is\n   deterministic; "
        "metric values closer than that are equal); none of it changes later in\n   the run.",
        "Every metric in pareto.md and explored-ideas.md is a dev metric from that\n   command",
        "Then read + profile the dev data, the tests and the baseline, commit the\n   baseline, "
        "run `git update-ref refs/discover/<task id>/B HEAD`",
        "insert it as the\n   first frontier point (id B, no parent)",
        "Web-search for SOTA approaches",
        "<scratch dir>/ideas.md",
        "Pairwise judge the ideas not yet in <scratch dir>/explored-ideas.md",
        # One round: parent snapshot, implement, evaluate, commit, log observations.
        "`git reset --hard <parent commit>`",
        "`git add -A && git commit && git update-ref\n   refs/discover/<task id>/<id> HEAD`",
        "append id, parent, commit, idea, aspect, draw,\n   metrics, status and observations "
        "(what the profile, errors and outputs showed",
        "to <scratch dir>/explored-ideas.md",
        "KEPT when no\n   frontier point dominates it",
        "better beyond the noise\n   floor on one, in each metric's own direction",
        "insert it in pareto.md, drop the points\n   it dominates from pareto.md and set their "
        "explored-ideas.md status to DOMINATED",
        "TIE\n   when it equals a frontier point on every metric (not inserted); FAILED otherwise",
        "TIE\n   and FAILED ideas are never retried",
        # Passing the check means meeting the goal on the held-out metrics as well (a real
        # run invented a looser threshold); a baseline brought back is KEPT again.
        "generalization check on it once to prove it is not\n   overfit: stop (step 6) if it "
        "passes (the held-out metrics meet the goal too); otherwise\n   drop it from pareto.md, "
        "set its status to OVERFIT (re-insert the baseline, KEPT again,\n   if the frontier is "
        "now empty) and continue",
        # Step 6: stop rule with a default round limit, else one exclusive draw by a tool.
        "A round is one pass through step 5",
        "the round limit (the task's; 100 rounds when the\n   task gives none)",
        "`git reset --hard` its commit so the working\n   tree holds it",
        "counts (KEPT, TIE, FAILED, OVERFIT, DOMINATED) and the\n"
        "   refs/discover/<task id>/ refs that hold every evaluated commit",
        "one Bash call, `python3 -c 'import random; print(random.random())'`",
        "logged by step 5 as the draw of the idea it leads to",
        "r < 0.33 -> step 7 (variant);\n"
        "   0.33 <= r < 0.66 -> step 8 (compose); r >= 0.66 -> step 9 (fresh)",
        "A compose draw with\n   fewer than two eligible frontier points is a fresh draw",
        # The three operators, each ending in step 5 with a named parent.
        "7. Variant: pick a frontier point and, from the observations logged for it",
        "Go to step 5 with that point as the parent",
        "8. Compose: pick two frontier points of different aspects, neither the baseline",
        "Go to step 5 with the first as the parent",
        "9. Fresh: extensive internet search or gpt-6-astra or gemini-4-argon (chosen by a second\n"
        "   draw)",
    ):
        assert phrase in instructions, phrase
    # The ambiguous "with probability" wording and the unnumbered composition cap are gone.
    # The lab keys on the task, not on being a sub-agent: /revise_and_review_paper's
    # writer is a sub-agent that carries this loop and edits the paper in place.
    # The status is not kept in two files any more (the two drifted apart in a real
    # run), the draw is a row field rather than a section inserted anywhere, and
    # profiling no longer comes before the split is fixed.
    for gone in (
        "With probability", "probability 0.5", "up to three", "got-6-astra", "step 10",
        "Is subagent", "in both files", "record the draw", "dev metrics,\n   status",
        "Read + profile the data",
    ):
        assert gone not in instructions, gone
    assert instructions.count("Go to step 5") + instructions.count("go to step 5") == 3
    # The loop is the SEA's alone: neither default prompt carries it, and
    # the rest of the default prompt reads on where the section used to be.
    for prompt in (SYSTEM_PROMPT, SYSTEM_PROMPT_LITE):
        assert _HEADING not in prompt
        assert "explored-ideas.md" not in prompt
    assert _LAST_STEP + "\n\n" + _NEXT_HEADING in SYSTEM_PROMPT


def test_split_cuts_a_copy_of_the_section_out_cleanly() -> None:
    """The split removes a section and its trailing blank lines; the rest reads on."""
    # A base prompt that still carries the section where SYSTEM.md used to.
    marker = _NEXT_HEADING + "\n"
    old = SYSTEM_PROMPT.replace(marker, discover_sea.DISCOVERY_INSTRUCTIONS + "\n\n" + marker, 1)
    assert old != SYSTEM_PROMPT
    rest, section = split_discovery_section(old)
    assert section == discover_sea.DISCOVERY_INSTRUCTIONS + "\n\n"
    assert rest == SYSTEM_PROMPT
    # A prompt without the section is returned as is; a section at the
    # very end of a prompt runs to the end.
    assert split_discovery_section(SYSTEM_PROMPT) == (SYSTEM_PROMPT, "")
    assert split_discovery_section("plain") == ("plain", "")
    tail = "BASE\n\n" + _HEADING + "\n\n1. profile\n"
    assert split_discovery_section(tail) == ("BASE\n\n", _HEADING + "\n\n1. profile\n")


def test_system_prompt_appends_the_instructions_once_at_the_end() -> None:
    """The default prompt is kept whole, in order, and ends with the loop."""
    out = DiscoverSea().system_prompt(SYSTEM_PROMPT)
    assert out == SYSTEM_PROMPT.rstrip("\n") + "\n\n" + discover_sea.DISCOVERY_INSTRUCTIONS + "\n"
    assert out.count(_HEADING) == 1
    assert out.lstrip().startswith("<identity>")
    # The loop is appended after the caller's additions too.
    out = DiscoverSea().system_prompt(SYSTEM_PROMPT + "\n\nCALLER")
    assert out.endswith("CALLER\n\n" + discover_sea.DISCOVERY_INSTRUCTIONS + "\n")
    assert out.count(_HEADING) == 1
    # A base prompt that still carries a copy of the section has it moved, not doubled.
    marker = _NEXT_HEADING + "\n"
    old = SYSTEM_PROMPT.replace(marker, discover_sea.DISCOVERY_INSTRUCTIONS + "\n\n" + marker, 1)
    assert DiscoverSea().system_prompt(old) == DiscoverSea().system_prompt(SYSTEM_PROMPT)
    assert DiscoverSea().system_prompt(out) == out  # idempotent
    # The lite prompt and a replacement prompt get it appended as well.
    out = DiscoverSea().system_prompt(SYSTEM_PROMPT_LITE)
    assert out == (
        SYSTEM_PROMPT_LITE.rstrip("\n") + "\n\n" + discover_sea.DISCOVERY_INSTRUCTIONS + "\n"
    )
    assert DiscoverSea().system_prompt("BASE") == (
        "BASE\n\n" + discover_sea.DISCOVERY_INSTRUCTIONS + "\n"
    )


@pytest.mark.usefixtures("reset_registry")
def test_description_is_one_sentence_naming_the_command() -> None:
    """``/discover help`` returns the description without running the SEA."""
    text = DiscoverSea().description()
    assert text.count(". ") == 0 and text.endswith(".")
    assert "/discover" in text
    assert sea_commands.help_text_if_command("/discover help") == text.strip()
    assert sea_commands.get_command("discover") == _SEA_PATH
    assert sea_commands.slash_command("/discover make it faster") == ("make it faster", _SEA_PATH)
    assert resolve_agent("discover", "") == (str(_SEA_PATH), "discover")


def test_settings_name_the_bestrouter_and_the_full_toolset() -> None:
    """The run's model is the ``bestrouter`` picker entry and its toolset is the full one."""
    assert DiscoverSea().settings({}) == {"model": "bestrouter", "tool_profile": "full"}
    assert DiscoverSea().settings({"max_budget": 2.0}) == {
        "max_budget": 2.0, "model": "bestrouter", "tool_profile": "full",
    }
    assert sea_commands.sea_settings(_SEA_PATH) == {"model": "bestrouter", "tool_profile": "full"}
    # ``bestrouter`` is a model-picker SEA, so the daemon resolves the
    # model to the picker's primary model instead of forwarding the name.
    picker = Path(bestrouter_sea.__file__).resolve()
    assert sea_commands.model_sea("bestrouter") == picker
    assert sea_commands.sea_settings(picker)["model"] == bestrouter_sea.PRIMARY_MODEL
    assert DiscoverSea().register_as_model() is False
    assert DiscoverSea().tools([print]) == [print]
    assert DiscoverSea().prompt("the task") == "the task"
    assert_no_removed_getters(discover_sea)


def test_loader_stages_the_rewrite_as_a_system_prompt_hook() -> None:
    """The daemon-side loader pins the model and profile and stages the prompt hook."""
    cmd: dict[str, Any] = {"seaPath": str(_SEA_PATH), "appendToSystemPrompt": "CALLER"}
    assert apply_sea(cmd) == {"model", "toolProfile"}
    assert cmd["model"] == "bestrouter"
    assert cmd["toolProfile"] == "full"
    assert cmd["appendToSystemPrompt"] == "CALLER"
    hook = cmd["systemPromptHook"]
    assert hook(SYSTEM_PROMPT + "\n\nCALLER") == DiscoverSea().system_prompt(
        SYSTEM_PROMPT + "\n\nCALLER"
    )
    assert hook("BASE") == "BASE\n\n" + discover_sea.DISCOVERY_INSTRUCTIONS + "\n"
    assert "prompt" not in cmd
    assert cmd["toolsHook"]([print]) == [print]
    assert cmd["llmCallHook"]([{"role": "user", "content": "x"}]) == [
        {"role": "user", "content": "x"}
    ]
    assert cmd["toolCallHook"]("Bash", {"command": "ls"}) == ALLOW


def test_agent_run_ends_the_default_prompt_with_the_section(tmp_path: Path) -> None:
    """A run configured as the daemon configures it sends the default prompt plus the loop.

    The scripted model finishes at once; the test checks the system
    message of the real request: the default Sorcar prompt (``<identity>``)
    comes first, the AI discovery loop follows the last of its sections
    (``## Sorcar repo specific``) and appears nowhere else, only the agent's
    own run-time blocks (``# MOST IMPORTANT INSTRUCTIONS``, ``# Task
    Settings``) come after it, and the full toolset is offered.
    """
    run = sea_commands.evaluate_sea([DiscoverSea()], "Make the parser twice as fast.")
    assert run.settings == {"model": "bestrouter", "tool_profile": "full"}
    parsed, agentic = run_sea(
        run,
        [finish_body("<p>done</p>", prompt_tokens=500)],
        tmp_path,
        max_steps=3,
        is_parallel=False,
    )
    assert parsed["success"] is True
    assert parsed["summary"] == "<p>done</p>"

    assert len(agentic) == 1, agentic
    system = system_message(agentic[0])
    assert system.lstrip().startswith("<identity>"), system[:200]
    assert system.count(_HEADING) == 1
    assert discover_sea.DISCOVERY_INSTRUCTIONS in system
    assert system.index(_NEXT_HEADING) < system.index("## Sorcar repo specific")
    assert system.index("## Sorcar repo specific") < system.index(_HEADING)
    after = system[system.index(discover_sea.DISCOVERY_INSTRUCTIONS) :]
    after = after[len(discover_sea.DISCOVERY_INSTRUCTIONS) :]
    assert "## " not in after, after
    assert "# MOST IMPORTANT INSTRUCTIONS" in after and "# Task Settings" in after
    assert {"Bash", "Read", "Edit", "Write", "finish"} <= tool_names(agentic[0])
    assert "# Restricted tool profile" not in system


_LAB_TAB = "discover-lab-tab"
"""The tab of the ``/discover`` sub-agent in the lab test."""

_PARENT_ID = "0123456789abcdef0123456789abcdef"
"""The task id of the calling task in the lab test."""

_COMMIT = "git -c user.email=t@example.com -c user.name=T -c commit.gpgsign=false commit -qm"
"""A commit command that needs no git identity or signer in the environment."""


class _LabScript:
    """The stand-in model's answers: a ``/discover`` sub-agent working in a lab.

    Step 0 copies the caller's current files to ``<scratch dir>/lab`` and
    commits them (the baseline), step 1 isolates the lab with
    ``create_worktree``, step 2 writes an idea into the lab's worktree,
    step 3 commits it and writes ``best.patch``, and every later step is
    ``finish`` (the first one is refused once: the lab's worktree is unmerged).
    """

    def __init__(self, caller_tree: Path) -> None:
        self.caller_tree = caller_tree
        self.requests: list[dict[str, Any]] = []
        self.child: ChatSorcarAgent | None = None

    def lab(self) -> Path:
        """Return the lab directory inside the sub-agent's scratch dir."""
        assert self.child is not None
        return Path(self.child.work_dir or "") / "tmp" / self.child.last_task_id / "lab"

    def lab_tree(self) -> Path:
        """Return the worktree ``create_worktree`` made for the lab."""
        assert self.child is not None
        return self.child.worktrees[0].wt_dir

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Record *request* and answer the sub-agent's next model call."""
        self.requests.append(request)
        state = agent_state.find_by_tab(_LAB_TAB)
        assert state is not None and state.agent is not None
        self.child = state.agent
        lab = self.lab()
        step = len(self.requests) - 1
        # Paths in a Bash command are spelt for bash (Git Bash on
        # Windows), where the backslashes of a native path are escapes.
        lab_sh, caller_sh = lab.as_posix(), self.caller_tree.as_posix()
        if step == 0:
            return tool_call_response("Bash", {
                "description": "copy the caller's current files to the lab",
                "command": (
                    f"mkdir -p {lab_sh} && cp -a {caller_sh}/. {lab_sh}/ && rm -f {lab_sh}/.git"
                    f" && cd {lab_sh} && git init -q && git add -A && {_COMMIT} baseline"
                ),
            })
        if step == 1:
            return tool_call_response("create_worktree", {"path": str(lab)})
        if step == 2:
            return tool_call_response(
                "Write", {"file_path": str(self.lab_tree() / "model.py"), "content": "SCORE = 3\n"},
            )
        if step == 3:
            base = run_git(lab, "rev-parse", "HEAD").stdout.strip()
            return tool_call_response("Bash", {
                "description": "commit the idea and write the patch",
                "command": (
                    f"cd {self.lab_tree().as_posix()} && git add -A && {_COMMIT} idea"
                    f" && git diff --binary {base} HEAD > {(lab.parent / 'best.patch').as_posix()}"
                ),
            })
        return finish_response("The best point is in best.patch.")


def _last_message(request: dict[str, Any]) -> str:
    """Return the text of the last message of *request* (the latest tool result)."""
    return request_text({"messages": request["messages"][-1:]})


def test_a_sub_agent_lab_leaves_the_callers_tree_alone_and_hands_back_a_patch() -> None:
    """A real daemon sub-agent run of the SEA can follow the lab procedure end to end.

    As in task 6360f17b, the task forbids repository changes, the calling
    task isolates the artifact's repository in its worktree (here with an
    uncommitted change) and the sub-agent's work dir is another
    repository (there ``~/kiss``, the artifact in ``~/bright``).  Through
    the lab procedure, ``create_worktree`` isolates the lab, the write
    guard lets the edit in the lab's worktree through, the caller's
    commit, branch and uncommitted change stay as they were, and
    ``best.patch`` applies to the caller's tree and carries the idea.
    """
    home = IsolatedKissHome(prefix="kiss-discover-lab-")
    credentials = _FakeCredentials()
    parent = ChatSorcarAgent("parent")
    script: _LabScript | None = None
    standin: StandInModelServer | None = None
    try:
        (home.repo / ".gitignore").write_text("tmp/\n", encoding="utf-8")
        run_git(home.repo, "add", ".gitignore")
        run_git(home.repo, "commit", "-qm", "ignore tmp")
        artifact = home.tmpdir / "artifact"
        init_repo(artifact)
        (artifact / "model.py").write_text("SCORE = 1\n", encoding="utf-8")
        run_git(artifact, "add", "model.py")
        run_git(artifact, "commit", "-qm", "model")
        home.write_config(max_budget=5.0, use_web_browser=False)
        parent.work_dir = str(home.repo)
        parent._run_tool_names = frozenset({"create_worktree"})
        assert "Created worktree" in parent.create_worktree(str(artifact))
        caller_tree = parent.worktrees[0].wt_dir
        (caller_tree / "model.py").write_text("SCORE = 2\n", encoding="utf-8")
        caller_head = run_git(caller_tree, "rev-parse", "HEAD", "--abbrev-ref", "HEAD").stdout
        agent_state.register(AgentState(
            _PARENT_ID, agent=parent, tab_id="parent-tab", is_task_active=True,
        ))
        script = _LabScript(caller_tree)
        standin = StandInModelServer(script)
        printer = CapturePrinter()
        server = AgentServer(printer=printer)
        server.work_dir = str(home.repo)
        server._run_task({
            "type": "run",
            "tabId": _LAB_TAB,
            "prompt": "Rank model designs for the scorer (research only; do not modify any "
                      "repository).",
            "seaPath": str(_SEA_PATH),
            "model": STANDIN_MODEL,
            "workDir": str(home.repo),
            "isParallel": False,
            "useWebTools": False,
            "classifyTasks": False,
            "maxBudget": 5.0,
            "modelConfig": standin.model_config,
            "parentTaskId": _PARENT_ID,
        })

        reqs = script.requests
        assert len(reqs) == 6, [_last_message(r)[-300:] for r in reqs] or printer.captured
        # The SEA ran, as a sub-agent, with the loop in its prompt.
        assert discover_sea.DISCOVERY_INSTRUCTIONS in request_text(reqs[0])
        assert "- Is subagent: yes\n" in request_text(reqs[0])
        assert "Created worktree" in _last_message(reqs[2])
        assert "Successfully wrote" in _last_message(reqs[3])
        assert "has not been merged" in _last_message(reqs[5])
        lab, lab_tree = script.lab(), script.lab_tree()
        assert lab_tree.parent == lab / ".kiss-worktrees"
        assert (lab_tree / "model.py").read_text(encoding="utf-8") == "SCORE = 3\n"
        assert (lab / "model.py").read_text(encoding="utf-8") == "SCORE = 2\n"
        # The caller's tree is as the caller left it, and its repository gained no branch.
        assert run_git(caller_tree, "rev-parse", "HEAD", "--abbrev-ref", "HEAD").stdout == (
            caller_head
        )
        assert run_git(caller_tree, "diff", "--name-only").stdout == "model.py\n"
        assert (caller_tree / "model.py").read_text(encoding="utf-8") == "SCORE = 2\n"
        assert kiss_wt_branches(artifact) == [parent.worktrees[0].branch]
        # The patch carries the idea onto the caller's current tree.
        patch = str(lab.parent / "best.patch")
        assert run_git(caller_tree, "apply", "--check", patch).returncode == 0
        assert run_git(caller_tree, "apply", patch).returncode == 0
        assert (caller_tree / "model.py").read_text(encoding="utf-8") == "SCORE = 3\n"
    finally:
        agent_state.unregister(_PARENT_ID)
        for agent in (script.child if script else None, parent):
            if agent is not None and agent._wt_pending:
                agent.discard()
        if standin is not None:
            standin.stop()
        credentials.restore()
        home.cleanup()


class _Recorder:
    """The stand-in model: records every request and finishes at once."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Record *request* and answer with ``finish``."""
        self.requests.append(request)
        return finish_response("done")


def _first_agent_request(
    server: AgentServer, standin: StandInModelServer, recorder: _Recorder, cmd: dict[str, Any],
) -> str:
    """Run the daemon task *cmd* and return the text of its agent's first model request."""
    start = len(recorder.requests)
    server._run_task({
        "type": "run",
        "model": STANDIN_MODEL,
        "workDir": server.work_dir,
        "isParallel": False,
        "useWebTools": False,
        "classifyTasks": False,
        "maxBudget": 5.0,
        "modelConfig": standin.model_config,
    } | cmd)
    texts = [request_text(r) for r in recorder.requests[start:]]
    agent_texts = [text for text in texts if "# Task Settings" in text]
    assert agent_texts, texts
    return agent_texts[0]


def test_a_typed_task_naming_discover_sees_its_help_but_its_runs_do_not() -> None:
    """Only a task the user typed gets the help of the slash commands it names.

    The chat's first task says "You must use /discover" and the second is
    "continue": both prompts end with the SEA's description, the second
    through the chat history.  A ``/discover`` run in that chat (whose
    history names the command) and a ``run_agent`` sub-task whose own task
    names it get none, so the SEA is never told to call itself.
    """
    home = IsolatedKissHome(prefix="kiss-discover-named-")
    credentials = _FakeCredentials()
    recorder = _Recorder()
    standin = StandInModelServer(recorder)
    help_line = f"- `/discover`: {DiscoverSea().description()}"
    try:
        home.write_config(max_budget=5.0, use_web_browser=False)
        server = AgentServer(printer=CapturePrinter())
        server.work_dir = str(home.repo)

        typed = _first_agent_request(server, standin, recorder, {
            "tabId": "named-tab", "prompt": "You must use /discover to make model.py faster.",
        })
        assert help_line in typed
        assert typed.index("You must use /discover") < typed.index(
            sea_commands.NAMED_COMMANDS_HEADING
        )

        follow_up = _first_agent_request(server, standin, recorder, {
            "tabId": "named-tab", "prompt": "continue",
        })
        assert "## Previous tasks and results" in follow_up
        assert help_line in follow_up

        sea_run = _first_agent_request(server, standin, recorder, {
            "tabId": "named-tab", "prompt": "Make model.py faster.", "seaPath": str(_SEA_PATH),
        })
        assert discover_sea.DISCOVERY_INSTRUCTIONS in sea_run
        assert "You must use /discover" in sea_run
        assert sea_commands.NAMED_COMMANDS_HEADING not in sea_run

        agent_state.register(AgentState(
            _PARENT_ID, agent=ChatSorcarAgent("parent"), tab_id="parent-tab", is_task_active=True,
        ))
        sub_agent = _first_agent_request(server, standin, recorder, {
            "tabId": "named-sub-tab", "prompt": "Use /discover on model.py.",
            "parentTaskId": _PARENT_ID,
        })
        assert "- Is subagent: yes\n" in sub_agent
        assert sea_commands.NAMED_COMMANDS_HEADING not in sub_agent
    finally:
        standin.stop()
        credentials.restore()
        home.cleanup()
