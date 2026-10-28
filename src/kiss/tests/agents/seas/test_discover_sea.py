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
from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.core.tool_verdict import ALLOW
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_no_removed_getters,
    system_message,
    tool_names,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body

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
        # Every point is a commit with a parent, and the files carry both.
        "so that every evaluated idea is a commit",
        "one line per point: id, parent id, commit, idea, aspect (the part of the system",
        "<scratch dir>/pareto.md,\n   one line per point",
        "the same fields plus observations",
        # Dev metrics only; the held-out part is read in the step 5 check alone.
        "Fix the dev / held-out split now, before\n   any idea is evaluated",
        "<scratch dir>/split.md",
        "the exact dev evaluation\n   command, and its noise floor",
        "Every metric in pareto.md and explored-ideas.md is a dev\n   metric from that command",
        "the held-out part is read only by the held-out check of\n   steps 5 and 6",
        "insert it as the first frontier point (id B,\n   no parent)",
        "Web-search for SOTA approaches",
        "<scratch dir>/ideas.md",
        "Pairwise judge the ideas not yet in <scratch dir>/explored-ideas.md",
        # One round: parent snapshot, implement, evaluate, commit, log observations.
        "`git reset --hard <parent commit>`",
        "`git add -A && git commit`",
        "status and observations (what the profile, errors and outputs showed",
        "KEPT when\n   no frontier point dominates it",
        "better beyond the noise\n   floor on one, in each metric's own direction",
        "mark the points it\n   dominates DOMINATED in both files",
        "mark it OVERFIT in both files",
        "TIE when it equals a frontier point on\n   every metric (not inserted); FAILED otherwise",
        "TIE and FAILED ideas are never retried",
        "generalization check on it once to prove it is not overfit: stop (step 6) if it passes",
        "mark it OVERFIT in both files, evict it (re-insert the baseline if the\n"
        "   frontier is now empty) and continue",
        # Step 6: stop rule with a default round limit, else one exclusive draw by a tool.
        "A round is one pass through step 5",
        "the round limit (the task's; 20 rounds when the\n   task gives none)",
        "`git reset --hard` its commit so the working\n   tree holds it",
        "counts (KEPT, TIE, FAILED, OVERFIT, DOMINATED)",
        "one Bash call, `python3 -c 'import random; print(random.random())'`",
        "record the draw r in <scratch dir>/explored-ideas.md",
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
    for gone in ("With probability", "probability 0.5", "up to three", "got-6-astra", "step 10"):
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
