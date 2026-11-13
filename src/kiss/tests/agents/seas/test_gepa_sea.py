# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``/gepa`` SEA (``kiss/agents/seas/gepa/gepa_sea.py``).

The SEA is the plain Sorcar agent with the GEPA loop, pseudocode written
from the official implementation (gepa-ai/gepa), appended to the system
prompt, on the ``bestrouter`` with the full toolset plus ``reflect``, the
reflection step of the official package as a tool.  The tests check the
text the agent is given, the way the daemon loads the SEA (command
registry, settings, system-prompt hook, tools), the system message a real
run sends to the model, and the tool: its prompt and its reading of a
reply against answers computed with the official package
(``reports/gepa-official-vs-sea-2026-11-11/reflect_check.py``), and the
request a run sends when the model calls it.  The model is the scripted
local server of the SEA test suite, nothing is mocked.

How a ``/gepa`` run behaves on real tasks (three benchmarks of the GEPA
paper) is measured outside the test suite, with real models:
``reports/gepa-sea-2026-11-10.html``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.discover import discover_sea
from kiss.agents.seas.gepa import gepa_sea
from kiss.agents.seas.gepa.gepa_sea import GepaSea
from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.agent_dispatch import resolve_agent
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.core import config as config_module
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.core.models.model_info import calculate_cost
from kiss.core.tool_verdict import ALLOW
from kiss.tests.agents.seas.conftest import registered, run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_no_removed_getters,
    system_message,
    tool_names,
    tool_result_text,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    text_body,
    tool_call_body,
)
from kiss.tests.agents.sorcar.test_cost_audit_20261023 import (
    _llm_call,
    _model_costs,
    history_db,  # noqa: F401  (the fixture)
)
from kiss.tests.core.models import gemini_sse_harness
from kiss.tests.core.models.gemini_sse_harness import GeminiScript, chunk, text_part

_SEA_PATH = Path(gepa_sea.__file__).resolve()
_HEADING = gepa_sea.SECTION_HEADING
_SETTINGS = {"model": "bestrouter", "tool_profile": "full"}

_RECORDS: list[dict[str, Any]] = [
    {"Inputs": {"problem": " p ", "n": 3}, "Generated Outputs": ["a", {"b": [1, 2.5, None]}],
     "Feedback": "f\n"},
    {"Inputs": {}, "Generated Outputs": [], "Feedback": ""},
]
"""The records of a minibatch of two: nested values, an empty dict, an empty list."""

_EXAMPLES = (
    "# Example 1\n## Inputs\n### problem\np\n\n### n\n3\n\n## Generated Outputs\n### Item 1\na\n\n"
    "### Item 2\n#### b\n##### Item 1\n1\n\n##### Item 2\n2.5\n\n##### Item 3\nNone\n\n"
    "## Feedback\nf\n\n\n\n# Example 2\n## Inputs\n\n## Generated Outputs\n\n## Feedback\n\n\n"
)
"""How the official package renders :data:`_RECORDS` in its reflection prompt."""

_REPLIES = [
    ("```\nnew text\n```", "new text"),
    ("Here it is:\n```\nnew text\n```\nDone.", "new text"),
    ("```markdown\nnew text\n```", "new text"),
    ('```text title="x"\nnew text\n```', "new text"),
    ("```{.md}\nnew text\n```", "new text"),
    ("````\nuse ```python\ncode\n``` here\n````", "use ```python\ncode\n``` here"),
    ("````md\nouter ```inner``` text\n````", "outer ```inner``` text"),
    ("first\n```\na\n```\nmiddle\n```\nb\n```", "a\n```\nmiddle\n```\nb"),
    ("no fence at all", "no fence at all"),
    ("  padded text \n", "padded text"),
    ("```\nonly an opening fence", "only an opening fence"),
    ("```python\nonly an opening fence with a tag", "only an opening fence with a tag"),
    ("only a closing fence\n```", "only a closing fence"),
    ("~~~\ntilde text\n~~~", "tilde text"),
    ("~~~~md\ntilde text\n~~~~", "tilde text"),
    ("~~~\n~~~", ""),
    ("<think>cut off while thinking", ""),
    ("<think>done</think>\nplain answer", "<think>done</think>\nplain answer"),
    ("<think>done</think>\n```\nfenced answer\n```", "fenced answer"),
    ("", ""),
    ("   \n ", ""),
    ("``````", ""),
    ("```\n```", ""),
    ("``` ```", ""),
    ("text with one ` backtick and ``two``", "text with one ` backtick and ``two``"),
]
"""Replies of a reflection model and the text the official parser reads from each."""


def _pseudocode(instructions: str) -> list[str]:
    """Return the lines inside the one fenced block of *instructions*."""
    lines = instructions.splitlines()
    first = lines.index("```")
    return lines[first + 1 : lines.index("```", first + 1)]


def _flat(text: str) -> str:
    """Return *text* on one line: comment marks dropped, each run of white space one space."""
    return " ".join(re.sub(r"^#", "", text, flags=re.MULTILINE).split())


def test_the_instructions_are_the_official_gepa_loop_as_short_pseudocode() -> None:
    """``GEPA_INSTRUCTIONS`` is the official loop as pseudocode, no longer than /discover's."""
    instructions = gepa_sea.GEPA_INSTRUCTIONS
    assert instructions.startswith(_HEADING + "\n\n")
    assert instructions.endswith("```")
    assert instructions.count("\n## ") == 0
    assert instructions.count("```") == 2
    # It names the implementation it was written from and the agent's part in it.
    flat = _flat(instructions)
    assert f"(gepa-ai/gepa, main at {gepa_sea.OFFICIAL_COMMIT}, 2026-11-09)" in flat
    assert "Run GEPA(task) below exactly" in flat
    assert (
        "You run the loop; its reflection LM is your tool reflect, a fresh call of a model on "
        "the official reflection prompt: you never write, edit or shorten a candidate text "
        "yourself."
    ) in flat
    # The engine counts its own evaluations and nothing else (two runs of the 2026-11-11
    # benchmark counted a trial call made by hand and ended one metric call above it).
    assert (
        "Only gepa.py evaluates, so that metric_calls is what the official engine counts: a "
        "trial of the evaluation command before the run, when you need one to learn its output, "
        "is one train example, is named in your report and is not counted."
    ) in flat
    # The loop is the procedures and their helpers, each defined once.
    for name in ("procedure GEPA(task):", "procedure MUTATE():", "procedure MERGE():",
                 "procedure ADD(k, parents, scores):", "DOMINATORS() :=",
                 "SELECT_CANDIDATE() :=", "MINIBATCH() :=", "COMPONENT(p) :=", "TRIPLET(D) :=",
                 "SELFTEST() :=", "REFLECT(text, records) :="):
        assert instructions.count(name) == 1, name
    for phrase in (
        # The official defaults (gepa.api.optimize) and the one default the SEA adds.
        "seed 0, minibatch_size 3, perfect_score 1, skip_perfect_score true, "
        "use_merge false, max_merge_invocations 5",
        "a task without a budget gets 10 * |valset|",
        # GEPAEngine.run: the seed's validation scores, the budget, a merge only after an
        # accepted mutation, the best validation average.
        "ADD(0, [], EVAL(seed, all val ids, val))",
        "while metric_calls < max_metric_calls and the task's budget is not spent:",
        "the budget is tested here only: the last iteration may end above it, as in the engine",
        "if use_merge and merges_due > 0 and found_new:",
        "best := the candidate with the highest avg, the lowest index on a tie",
        # ReflectiveMutationProposer: perfect minibatches are skipped, strict improvement.
        "if skip_perfect_score and every score of before >= perfect_score: log skipped; return",
        "if sum(after) > sum(before):",
        "if merges_tested < max_merge_invocations: merges_due += 1",
        # MergeProposer: no draw without two dominators and three candidates, both parents
        # changed a text: the better one's, the floor of five validation examples inside
        # the attempts, five validation examples, no worse than the better parent.
        "if |D| < 2 or there are fewer than 3 candidates: return none",
        "when both changed it, the one of the higher avg, rng.choice([p, q]) on a tie",
        "if (p, q, from) is in merged or the valset has fewer than 5 examples: next",
        "S := 5 val ids: rng.sample(B, min(|B|, 2, 5 - |S|))",
        "if sum(after) >= max(sum of val[p] on S, sum of val[q] on S):",
        # Pareto selection, the minibatch order and the component order are seeded arithmetic.
        "n := {k: the number of DOMINATORS() fronts that hold k}",
        "next_component[p] := (n + 1) mod the number of components",
        "The bookkeeping is exact arithmetic and seeded draws: never do it by eye",
        # The draws are the official code's calls on the official orders, so that a seed
        # gives the run the official engine gives (run A of the 2026-11-10 benchmark drew
        # its parents with rng.choices and left the engine's path at iteration 8).
        "rng random.Random(seed): every draw below comes from it, by the call written there",
        "a Python set of ints filled in ascending index, then copied by set(..); like every "
        "set below it is read in Python's order",
        "for each y of sorted(Y, key=avg): # stable on equal avg",
        "return rng.choice([k for k in n for _ in range(n[k])])",
        "order := list(range(|trainset|)); rng.shuffle(order)",
        "p, q := sorted(rng.sample(D, 2))",
        "a := rng.choices(E, weights)[0]",
        # The self-check of gepa.py: the official engine's draws on a small state
        # (reports/gepa-official-vs-sea-2026-11-11/known_answers.py computes them).
        "write gepa.py; SELFTEST() holds before the first EVAL",
        "candidates 0..16 added in order, val[k][v] := 1 if (3k + 5v) mod 11 < 3 else 0 on "
        "val ids 0..9",
        "the candidates [5, 5, 10, 5, 15, 7] and the minibatches [[4, 1, 5], [2, 0, 3], "
        "[6, 6, 3], [6, 0, 3], [5, 1, 4], [2, 2, 4]]",
        # The reflection step is the official one: a fresh call on the official prompt with
        # the records whole, not the agent's own rewrite (in the 2026-11-11 benchmark at most
        # 27% of the characters of the records reached the agent, and its rewrites raised the
        # minibatch total less often than a fresh call on the same text and records).
        "records/<i>.json the records of iteration i's minibatch for the component: a JSON "
        "list of {Inputs, Generated Outputs, Feedback} by the order of ids, whole, uncut",
        "REFLECT is given the records whole (every character of each example's Inputs, "
        "Generated Outputs and Feedback, the examples scored perfect too) and nothing else of "
        "the run; you do not read them for it. Train records only: val examples are scored, "
        "never read; a test set the task names is opened by nothing here.",
        "text := REFLECT(p's text of c, the records of c); none: log skipped; return",
        "your tool reflect(text_file, records_file, new_text_file[, model]): one fresh call of "
        "the reflection model (the task's; without one, the model you run on; the same for "
        "the whole run) on the official reflection prompt and nothing else (no tools, no "
        "system prompt, no memory of the run).",
        # No records, an empty reply, a refused prompt: the iteration ends, as in the engine.
        "The tool writes the proposed text to new_text_file, or says that there is no proposal "
        "(no records, a reply without text, a prompt the model refused): none.",
        "That file is the proposal as it stands: one call per iteration, repeated only when "
        "the call failed",
        # The files a run can be audited from have one layout.
        "parents (ints), component, ids, before, after (lists of scores in the order of ids)",
    ):
        assert phrase in flat, phrase
    # As short as the /discover pseudocode: no more lines, no wider.
    discover = discover_sea.DISCOVERY_INSTRUCTIONS
    assert len(_pseudocode(instructions)) <= len(_pseudocode(discover))
    assert max(map(len, instructions.splitlines())) <= max(map(len, discover.splitlines()))


def test_system_prompt_appends_the_instructions_once_at_the_end() -> None:
    """The default prompt is kept whole, in order, and ends with the loop."""
    out = GepaSea().system_prompt(SYSTEM_PROMPT)
    assert out == SYSTEM_PROMPT.rstrip("\n") + "\n\n" + gepa_sea.GEPA_INSTRUCTIONS + "\n"
    assert out.count(_HEADING) == 1
    assert out.lstrip().startswith("<identity>")
    assert GepaSea().system_prompt(out) == out  # a prompt that carries the section is kept
    # The loop is appended after the caller's additions too.
    out = GepaSea().system_prompt(SYSTEM_PROMPT + "\n\nCALLER")
    assert out.endswith("CALLER\n\n" + gepa_sea.GEPA_INSTRUCTIONS + "\n")
    # The lite prompt and a replacement prompt get it appended as well.
    assert GepaSea().system_prompt(SYSTEM_PROMPT_LITE) == (
        SYSTEM_PROMPT_LITE.rstrip("\n") + "\n\n" + gepa_sea.GEPA_INSTRUCTIONS + "\n"
    )
    assert GepaSea().system_prompt("BASE") == "BASE\n\n" + gepa_sea.GEPA_INSTRUCTIONS + "\n"


@pytest.mark.usefixtures("reset_registry")
def test_description_is_one_sentence_naming_the_command() -> None:
    """``/gepa help`` returns the description without running the SEA."""
    text = GepaSea().description()
    assert text.count(". ") == 0 and text.endswith(".")
    assert "/gepa" in text and 'run_agent(agent="gepa"' in text
    assert sea_commands.help_text_if_command("/gepa help") == text.strip()
    assert sea_commands.get_command("gepa") == _SEA_PATH
    assert sea_commands.slash_command("/gepa optimize prompt.txt") == (
        "optimize prompt.txt", _SEA_PATH,
    )
    assert resolve_agent("gepa", "") == (str(_SEA_PATH), "gepa")


def test_settings_name_the_bestrouter_and_the_full_toolset() -> None:
    """The run's model is the ``bestrouter`` picker entry and its toolset is the full one."""
    assert GepaSea().settings({}) == _SETTINGS
    assert GepaSea().settings({"max_budget": 2.0}) == {"max_budget": 2.0, **_SETTINGS}
    assert sea_commands.sea_settings(_SEA_PATH) == _SETTINGS
    assert GepaSea().register_as_model() is False
    assert GepaSea().tools([print]) == [print, gepa_sea.reflect]
    assert GepaSea().prompt("the task") == "the task"
    assert_no_removed_getters(gepa_sea)


def test_loader_stages_the_rewrite_as_a_system_prompt_hook() -> None:
    """The daemon-side loader pins the model and profile and stages the prompt hook."""
    cmd: dict[str, Any] = {"seaPath": str(_SEA_PATH), "appendToSystemPrompt": "CALLER"}
    assert apply_sea(cmd) == {"model", "toolProfile"}
    assert cmd["model"] == "bestrouter"
    assert cmd["toolProfile"] == "full"
    assert cmd["appendToSystemPrompt"] == "CALLER"
    hook = cmd["systemPromptHook"]
    assert hook(SYSTEM_PROMPT + "\n\nCALLER") == GepaSea().system_prompt(
        SYSTEM_PROMPT + "\n\nCALLER"
    )
    assert hook("BASE") == "BASE\n\n" + gepa_sea.GEPA_INSTRUCTIONS + "\n"
    assert "prompt" not in cmd
    # The loader imports the file as a module of its own: the tool is that module's function.
    assert [tool.__name__ for tool in cmd["toolsHook"]([print])] == ["print", "reflect"]
    assert cmd["toolCallHook"]("Bash", {"command": "ls"}) == ALLOW


def test_agent_run_ends_the_default_prompt_with_the_section(tmp_path: Path) -> None:
    """A run configured as the daemon configures it sends the default prompt plus the loop.

    The scripted model finishes at once; the test checks the system
    message of the real request: the default Sorcar prompt (``<identity>``)
    comes first, the GEPA loop follows the last of its sections and appears
    nowhere else, only the agent's own run-time blocks (``# MOST IMPORTANT
    INSTRUCTIONS``, ``# Task Settings``) come after it, and the full
    toolset is offered.
    """
    run = sea_commands.evaluate_sea([GepaSea()], "Optimize prompt.txt with eval.py.")
    assert run.settings == _SETTINGS
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
    assert gepa_sea.GEPA_INSTRUCTIONS in system
    assert system.index("## Sorcar repo specific") < system.index(_HEADING)
    after = system[system.index(gepa_sea.GEPA_INSTRUCTIONS) :]
    after = after[len(gepa_sea.GEPA_INSTRUCTIONS) :]
    assert "## " not in after, after
    assert "# MOST IMPORTANT INSTRUCTIONS" in after and "# Task Settings" in after
    assert {"Bash", "Read", "Edit", "Write", "finish", "reflect"} <= tool_names(agentic[0])
    assert "# Restricted tool profile" not in system


def test_reflection_prompt_is_the_official_prompt_around_the_records_whole() -> None:
    """The prompt is the official template with the text and the official record layout."""
    prompt = gepa_sea.reflection_prompt("Solve it.", _RECORDS)
    assert prompt.startswith(
        "I provided an assistant with the following instructions to perform a task for me:\n"
        "```\nSolve it.\n```\n\nThe following are examples of different task inputs provided to "
    )
    assert f"could be better:\n```\n{_EXAMPLES}\n```\n\nYour task is to write a new" in prompt
    assert (
        "Identify all niche and domain specific factual information about the task and include "
        "it in the instruction, as a lot of it may not be available to the assistant in the "
        "future."
    ) in prompt
    assert prompt.endswith("\n\nProvide the new instructions within ``` blocks.")
    assert len(prompt) == len(gepa_sea.REFLECTION_PROMPT) - len("<curr_param><side_info>") + len(
        "Solve it." + _EXAMPLES
    )
    # Headers stop at level six; an empty container is an empty line; other values their text.
    deep = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
    assert gepa_sea.markdown(deep) == "### a\n#### b\n##### c\n###### d\n###### e\ndeep\n\n"
    values: tuple[object, ...] = ({}, [], None, " x ")
    assert [gepa_sea.markdown(value) for value in values] == ["\n", "\n", "None\n\n", "x\n\n"]
    assert gepa_sea.markdown(("a", "b")) == "### Item 1\na\n\n### Item 2\nb\n\n"
    assert "```\n\n```\n\nYour task" in gepa_sea.reflection_prompt("t", [])  # no records


def test_proposal_reads_a_reply_as_the_official_parser_does() -> None:
    """Every rule of the official parser: fences, tags, tildes, lone fences, refused replies.

    A reply the output limit cut proposes only a complete pair of fences.
    """
    for reply, text in _REPLIES:
        assert gepa_sea.proposal(reply) == text, reply
    assert gepa_sea.proposal("Sure:\n```\nSolve it.\n```\nI changed", cut=True) == "Solve it."
    assert gepa_sea.proposal("```\nSolve", cut=True) == ""
    assert gepa_sea.proposal("Solve it with", cut=True) == ""


def test_reflect_outside_a_running_task_is_an_error_line(tmp_path: Path) -> None:
    """Called off a task's thread, the tool says so, asks no model and writes nothing."""
    target = tmp_path / "new"
    result = gepa_sea.reflect("text", "records.json", str(target))
    assert result == "Error: reflect is a tool of a running /gepa task, called on the task's thread"
    assert not target.exists()


def test_a_run_reflects_with_one_fresh_call_on_the_official_prompt(tmp_path: Path) -> None:
    """The model calls ``reflect``; the tool asks the model once, afresh, and writes the text.

    The scripted run calls the tool six times, on file names relative to
    the work directory and absolute.  Each of the first five calls sends
    one request of its own: no tools, the official prompt built from the
    two files as its only message (the system instruction of the caller's
    model configuration stays out), the run's model.  The first reply
    holds a fenced text, which is written to the file named; the second is
    cut off inside a reasoning block, which the official parser refuses.
    The next three ended early, and the tool reads each as the official
    parser does: the third was stopped by the provider's content filter
    and the fourth by the output limit, both after a complete fenced text,
    which is the proposal; the fifth was cut inside its fenced text: no
    proposal and no file.  The sixth call is on a component without
    records: no request at all, as the official reflection step skips it.
    The tokens of every reflection call, the cut ones too, are added to
    the task's.

    Not reached here: a reply the output limit cut that the provider hands
    over without an error (Anthropic, for a call without tools), whose stop
    reason the tool reads from the call's usage record.  The scripted
    Anthropic stand-in of the tests serves tool calls only.
    """
    gepa = tmp_path / "gepa"
    (gepa / "candidates" / "0").mkdir(parents=True)
    (gepa / "candidates" / "0" / "instruction").write_text("Solve it.")
    (gepa / "records").mkdir()
    (gepa / "records" / "1.json").write_text(json.dumps(_RECORDS))
    (gepa / "records" / "2.json").write_text("[]")
    files = {
        "text_file": "gepa/candidates/0/instruction",
        "records_file": str(gepa / "records" / "1.json"),
    }
    refused = str(gepa / "proposals" / "2" / "instruction")
    script = [
        tool_call_body(
            "reflect", {**files, "new_text_file": "gepa/proposals/1/instruction"}, prompt_tokens=500
        ),
        text_body("Here it is:\n```\nSolve it with care.\n```"),
        tool_call_body(
            "reflect",
            {**files, "new_text_file": refused, "model": MODEL},
            prompt_tokens=600,
        ),
        text_body("<think>cut off"),
        tool_call_body(
            "reflect", {**files, "new_text_file": "gepa/proposals/3/instruction"}, prompt_tokens=650
        ),
        text_body("```\nSolve it.\n```\nMy reasons were").replace(
            b'"finish_reason": "stop"', b'"finish_reason": "content_filter"'
        ),
        tool_call_body(
            "reflect", {**files, "new_text_file": "gepa/proposals/4/instruction"}, prompt_tokens=655
        ),
        text_body("```\nSolve it well.\n```\nI also").replace(
            b'"finish_reason": "stop"', b'"finish_reason": "length"'
        ),
        tool_call_body("reflect", {**files, "new_text_file": refused}, prompt_tokens=660),
        text_body("```\nSolve").replace(b'"finish_reason": "stop"', b'"finish_reason": "length"'),
        tool_call_body(
            "reflect",
            {**files, "records_file": "gepa/records/2.json", "new_text_file": refused},
            prompt_tokens=680,
        ),
        finish_body("<p>done</p>", prompt_tokens=700),
    ]
    run = sea_commands.evaluate_sea([GepaSea()], "Optimize prompt.txt with eval.py.")
    agent = ChatSorcarAgent("gepa-sea-test")
    with registered(agent, "gepa-sea-test"), serve(script) as (url, requests):
        agent.run(
            prompt_template=run.prompt,
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=8,
            max_budget=1.0,
            system_prompt_hook=run.system_prompt_hook,
            tools_hook=run.tools_hook,
            tool_call_hook=run.tool_call_hook,
            llm_call_hook=run.llm_call_hook,
            llm_response_hook=run.llm_response_hook,
            tool_profile=run.settings["tool_profile"],
            web_tools=False,
            use_memory=False,
            verbose=False,
            is_parallel=False,
            model_config={
                "base_url": url,
                "api_key": "local",
                "system_instruction": "The caller's system instruction.",
            },
        )

    prompt = gepa_sea.reflection_prompt("Solve it.", _RECORDS)
    reflections = [request for request in requests if not request.get("tools")]
    assert [request["messages"] for request in reflections] == [
        [{"role": "user", "content": prompt}]
    ] * 5
    assert {request["model"] for request in reflections} == {MODEL}
    proposal = gepa / "proposals" / "1" / "instruction"
    assert proposal.read_text() == "Solve it with care."
    assert not (gepa / "proposals" / "2").exists()
    agentic = [request for request in requests if request.get("tools")]
    assert len(agentic) == 7
    first, second, third, fourth, fifth, sixth = [
        tool_result_text(result) for result in tool_results(agentic[6])
    ]
    assert first.startswith(
        f"proposal written to {proposal}: 19 characters, from a prompt of {len(prompt)} "
        "characters, $"
    )
    assert second == "no proposal: the reply holds no text"
    assert third.startswith(f"proposal written to {gepa / 'proposals' / '3' / 'instruction'}: 9 ")
    assert (gepa / "proposals" / "3" / "instruction").read_text() == "Solve it."
    assert fourth.startswith(f"proposal written to {gepa / 'proposals' / '4' / 'instruction'}: 14 ")
    assert (gepa / "proposals" / "4" / "instruction").read_text() == "Solve it well."
    assert fifth == "no proposal: the reply was refused or cut short (length)"
    assert sixth == "no proposal: the component has no records"
    assert agent.total_tokens_used == sum(
        json.loads(body)["usage"]["total_tokens"] for body in script
    )


@pytest.fixture
def gemini_endpoint() -> Iterator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from gemini_sse_harness.serve()


def test_reflect_reads_a_gemini_reply_cut_at_the_output_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gemini_endpoint: tuple[str, GeminiScript],
) -> None:
    """Gemini hands a tool-less caller the text the cap cut; its ``MAX_TOKENS`` is read.

    The real ``google-genai`` SDK streams from the local Gemini endpoint.
    A reply cut inside its fenced text, with the usage and then the finish
    reason in chunks of their own, gives no proposal; the same text
    ended by ``STOP`` is read like any reply; a complete fenced text the
    cap cut after is the proposal.
    """
    (tmp_path / "text").write_text("Solve it.")
    (tmp_path / "records.json").write_text(json.dumps(_RECORDS))
    usage = {"promptTokenCount": 100, "candidatesTokenCount": 7}
    cut = [
        chunk([text_part("```\nSolve")]),
        {"usageMetadata": usage},
        chunk([], finish_reason="MAX_TOKENS"),
    ]
    stopped = [chunk([text_part("```\nSolve")], usage=usage, finish_reason="STOP")]
    capped = [chunk([text_part("```\nSolve it.\n```\nI")], usage=usage, finish_reason="MAX_TOKENS")]
    caller = ChatSorcarAgent("gepa-gemini-test")
    caller.work_dir = str(tmp_path)
    caller.model_name = "gemini-2.5-flash"
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", gemini_endpoint[0])
    results = []
    with registered(caller, "gepa-gemini-test"):
        for n, chunks in enumerate([cut, stopped, capped]):
            gemini_endpoint[1].play(chunks)
            results.append(gepa_sea.reflect("text", "records.json", f"new/{n}"))
    assert results[0] == "no proposal: the reply was refused or cut short (MAX_TOKENS)"
    assert not (tmp_path / "new" / "0").exists()
    assert results[1].startswith(f"proposal written to {tmp_path / 'new' / '1'}: 5 ")
    assert (tmp_path / "new" / "1").read_text() == "Solve"
    assert (tmp_path / "new" / "2").read_text() == "Solve it."


@pytest.mark.usefixtures("history_db")
def test_reflect_spend_is_charged_to_the_task_under_the_reflection_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gemini_endpoint: tuple[str, GeminiScript],
) -> None:
    """A reflection call is a side call of the task: banked and filed under its model.

    It emits no ``llm_call``, so its spend must also be recorded under the
    row (as a ``decide`` call is), or the Spend panel files it under the
    task's own model and the row's total is not explained by its records.
    """
    fable = "claude-fable-5-1"
    task_id, _ = th._add_task("gepa task", "", {"model": fable})
    th._queue_chat_event(_llm_call(fable, 1.0, 1_000), task_id)
    th._flush_chat_events()
    th._save_task_extra({"cost": 1.0, "tokens": 1_000, "steps": 1}, task_id=task_id)
    (tmp_path / "text").write_text("Solve it.")
    (tmp_path / "records.json").write_text(json.dumps(_RECORDS))
    usage = {"promptTokenCount": 100, "candidatesTokenCount": 7, "totalTokenCount": 107}
    caller = ChatSorcarAgent("gepa-spend-test")
    caller.work_dir = str(tmp_path)
    caller.model_name = "gemini-2.5-flash"
    caller._last_task_id = task_id
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", gemini_endpoint[0])
    with registered(caller, "gepa-spend-test"):
        gemini_endpoint[1].play(
            [chunk([text_part("```\nSolve it.\n```")], usage=usage, finish_reason="STOP")]
        )
        assert gepa_sea.reflect("text", "records.json", "new").startswith("proposal written")
    cost = calculate_cost("gemini-2.5-flash", 100, 7)
    assert cost > 0
    assert caller.budget_used == pytest.approx(cost)  # banked on the running task's ledger
    assert caller.total_steps == 0  # not an agent step
    th._save_task_extra(
        {"cost": 1.0 + cost, "tokens": 1_000 + caller.total_tokens_used, "steps": 1, "endTs": 1},
        task_id=task_id,
    )
    split = json.loads(_model_costs(task_id))
    assert split[fable] == [1.0, 1_000]
    assert split["gemini-2.5-flash"][0] == pytest.approx(cost)
    assert split["gemini-2.5-flash"][1] == caller.total_tokens_used
