# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""GEPA agent: reflective text evolution on a Pareto frontier, on the bestrouter.

The agent is the plain Sorcar agent with :data:`GEPA_INSTRUCTIONS`, the
GEPA loop written as pseudocode (``GEPA``, ``MUTATE``, ``MERGE``, ``ADD``
and their helpers), appended to the system prompt by the SEA's
``system_prompt`` method.  GEPA (Genetic-Pareto; Agrawal et al., "GEPA:
Reflective Prompt Evolution Can Outperform Reinforcement Learning",
arXiv 2507.19457) optimizes the text components of a system (prompts,
instructions, code, configuration) against a metric: it scores the seed
on the validation set, then per iteration draws a candidate from the
per-example Pareto front, runs it on a minibatch of training examples,
reads the outputs and the feedback, rewrites one component, keeps the
rewrite only when the minibatch total improves, and then scores it on
the whole validation set.

The pseudocode follows the official implementation,
https://github.com/gepa-ai/gepa at commit ``ce5b0d6`` (main, 2026-11-09,
139 commits after the release v0.1.4), which ships no pseudocode of its
own; each procedure is one piece of that code:

- ``GEPA``: the defaults of ``gepa.api.optimize`` and the loop of
  ``GEPAEngine.run`` (the seed's validation scores, a merge only after
  an accepted mutation, the metric-call budget checked between
  iterations, the best validation average returned).
- ``MUTATE``: ``ReflectiveMutationProposer.propose`` and the engine's
  acceptance test (a minibatch of perfect scores is skipped; a strict
  improvement of the minibatch sum is accepted).
- ``MERGE`` and ``TRIPLET``: ``MergeProposer.propose`` and
  ``sample_and_attempt_merge_programs_by_common_predictors``.
- ``ADD``: ``GEPAState.update_state_with_new_program``.
- ``DOMINATORS`` and ``SELECT_CANDIDATE``:
  ``gepa_utils.remove_dominated_programs`` and
  ``select_program_candidate_from_pareto_front``.
- ``MINIBATCH``: ``EpochShuffledBatchSampler``; ``COMPONENT``:
  ``RoundRobinReflectionComponentSelector``.
- ``REFLECT``: ``StatelessReflectionLM.reflect``, the default reflection
  step: one call of the reflection model on the prompt of
  ``InstructionProposalSignature``, its reply parsed by ``ProposalAdapter``.

The reflection LM is :func:`reflect`, a tool the SEA adds to the run: one
fresh call of the reflection model (the task's; without one, the model
the run is on), with no tools, no system prompt and no memory of the
run, on the official prompt (:data:`REFLECTION_PROMPT`, the records
rendered whole by :func:`reflection_prompt` as the official code renders
them), its reply read by the official parser's rules (:func:`proposal`).
A component without records, a reply without text, a prompt the model
refuses and a reply cut at the output limit before its fenced block was
complete give no proposal, and the iteration ends there, as in the
official engine; like the engine, the tool still takes the complete
fenced block of a cut reply and the text a content filter let through.
The agent runs the loop and writes no candidate text.
Earlier versions had the agent reflect itself, and that was not the
official step in two measured ways (the 2026-11-11 benchmark,
``reports/gepa-official-vs-sea-2026-11-11``).  The agent did not get the
records whole: its drivers printed the head and tail of each output and
the first lines of the feedback, and over a run its tools returned at
most 12% to 27% of the characters of the records in six runs; told to
read them whole, an agent was still given less than the records hold in
43 of 65 and in 66 of 78 reflections of two runs (the sizes decide
nothing for the third).  And its rewrites raised the minibatch total
less often than a fresh call on the official prompt given the same
parent text and the same records: 60 against 94 times in 137 iterations
(``reflector_ab.py``).
The task's evaluation command is the adapter (``GEPAAdapter.evaluate``
and ``make_reflective_dataset``).  Only the driver calls it, so that the
run's count of metric calls is the engine's, which counts its own
evaluations and no trial made before it starts (two runs of that
benchmark counted such a trial and ended one call above the engine).  All
the rest, which the official code computes, the agent writes once as
``<scratch dir>/gepa/gepa.py`` from the pseudocode and calls, so that
the Pareto fronts, the seeded draws and the budget are arithmetic and
not judgement; the run's state lives in files with fixed names
(``state.json``, ``log.jsonl``, ``candidates/``, ``records/``), which
lets a run be audited against the official rules and resumed.

The draws are written as the official code's own calls on its own
orders (``rng.choice`` over the per-front list, ``rng.shuffle``, Python's
set order of the candidate indices), and ``SELFTEST`` states what the
official engine draws on a small state, which ``gepa.py`` must reproduce
before the first evaluation.  A seed then gives the run the official
engine gives: ``reports/gepa-official-vs-sea-2026-11-11/replay_official.py``
feeds a finished run's scores and rewrites to ``gepa.optimize`` and
requires the same parents, minibatches, decisions and best candidate.
A pseudocode that named only the distributions let a driver draw its
parents with ``rng.choices`` and leave the engine's path (run A of the
2026-11-10 benchmark, at iteration 8).  This holds for the default,
merge off; a merge makes the official calls too, but its orders are not
checked and the official merge itself iterates a set of strings.

Only the official defaults are spelled out (Pareto candidate selection
on per-example fronts, epoch-shuffled minibatches of 3, one component
per iteration in round robin, strict improvement, merge off); a task
may override the values the pseudocode names.  The official
``optimize`` has no default budget and requires a stopping condition; a
task that names none gets ten validation passes of metric calls.

The run's model is ``bestrouter`` and the toolset is the full one, as
for ``/discover``: the daemon resolves the name to the model the
bestrouter's own settings name
(:data:`kiss.agents.seas.bestrouter.bestrouter_sea.SIMPLE_MODEL`, the
model a bestrouter tab starts on).  The reflection model is the model
the run is on unless the task names another.

Three ways to run it::

    /gepa optimize prompts/answer.txt for accuracy with `python eval.py` on data/train.jsonl
        and data/val.jsonl, 500 metric calls

    /gepa improve the two prompts of src/pipeline.py; the metric and its feedback are in
        tests/judge.py; use merge

    run_agent(agent="src/kiss/agents/seas/gepa/gepa_sea.py", task="...")

The task names the texts to optimize, the training and validation data,
the evaluation command (or the metric and its feedback) and the budget.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.sorcar.agent_dispatch import _parent_model_config
from kiss.agents.sorcar.sorcar_agent import _attribute_side_call_usage, side_call_binding
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import ModelRefusalError, OutputCapError
from kiss.server.agent_state import current_agent

SECTION_HEADING = "## GEPA: reflective text evolution on a Pareto frontier"
"""The heading of the GEPA section (:data:`GEPA_INSTRUCTIONS`)."""

OFFICIAL_COMMIT = "ce5b0d6"
"""The commit of https://github.com/gepa-ai/gepa the pseudocode was written from."""

GEPA_INSTRUCTIONS = f"""\
{SECTION_HEADING}

Mandatory Instructions (MUST FOLLOW): You will be optimizing the text components of a system
(prompts, instructions, code, configuration) against a metric with GEPA (Genetic-Pareto).
Run GEPA(task) below exactly: it is the loop of the official implementation (gepa-ai/gepa,
main at {OFFICIAL_COMMIT}, 2026-11-09).  You run the loop; its reflection LM is your tool
reflect, a fresh call of a model on the official reflection prompt: you never write, edit or
shorten a candidate text yourself.  Only gepa.py evaluates, so that metric_calls is what the
official engine counts: a trial of the evaluation command before the run, when you need one
to learn its output, is one train example, is named in your report and is not counted.

```
# candidate   component name -> text; candidate 0 is the seed, the task's current texts
# Files in <scratch dir>/gepa (a continued run resumes from them):
#   candidates/<k>/<component>  candidate k's texts, written once; a proposal waits in
#                proposals/<i>/ and is copied to candidates/<k>/ when it is accepted
#   state.json   {{i, metric_calls, merges_due, merges_tested, found_new, rng, order,
#                next_component: {{k: n}}, parents: {{k: [..]}}, val: {{k: {{val id: score}}}},
#                merged: [..]}}
#   records/<i>.json  the records of iteration i's minibatch for the component: a JSON list
#                of {{Inputs, Generated Outputs, Feedback}} by the order of ids, whole, uncut
#   log.jsonl    one object per iteration: i, op (reflect | merge), parents (ints), component,
#                ids, before, after (lists of scores in the order of ids), accepted, skipped
#                (why), new, avg, metric_calls
#   gepa.py      a CLI you write once from this pseudocode: every procedure below except
#                REFLECT, over state.json.  It runs EVAL itself, so the loop is one command
#                and one reflect call per REFLECT: the command runs on, through merges and
#                skipped iterations, to the next records and prints the files reflect takes;
#                called again, it goes on from the proposal.  The bookkeeping is exact
#                arithmetic and seeded draws: never do it by eye
# EVAL(c, ids, set)  the task's evaluation command, built first when the task gives only a
#     metric: runs candidate c on the examples ids of set (train | val) and returns per
#     example a score (higher is better) and per component a record {{Inputs, Generated
#     Outputs, Feedback}}; Feedback says what went wrong and what was expected (errors, the
#     reference answer, sub-scores).  One example evaluated is one metric call.
# rng        random.Random(seed): every draw below comes from it, by the call written there
# avg(k)     the mean of candidate k's val scores
# front(v)   the candidates with the highest score on val example v, held as the official
#            code holds them: a Python set of ints filled in ascending index, then copied by
#            set(..); like every set below it is read in Python's order
# REFLECT is given the records whole (every character of each example's Inputs, Generated
# Outputs and Feedback, the examples scored perfect too) and nothing else of the run; you
# do not read them for it.  Train records only: val examples are scored, never read; a test
# set the task names is opened by nothing here.

procedure GEPA(task):
    seed candidate, trainset, valset (the trainset when there is none), EVAL and
        max_metric_calls := the task's; a task without a budget gets 10 * |valset|
    config := the official defaults unless the task overrides one: seed 0, minibatch_size 3,
        perfect_score 1, skip_perfect_score true, use_merge false, max_merge_invocations 5
    write gepa.py; SELFTEST() holds before the first EVAL
    ADD(0, [], EVAL(seed, all val ids, val))                         # every front is {{0}}
    i := 0; merges_due := 0; merges_tested := 0; found_new := false
    # the budget is tested here only: the last iteration may end above it, as in the engine
    while metric_calls < max_metric_calls and the task's budget is not spent:
        i += 1
        if use_merge and merges_due > 0 and found_new:
            found_new := false
            if MERGE() == evaluated: continue    # accepted or not, no mutation this iteration
        found_new := false
        MUTATE()
    best := the candidate with the highest avg, the lowest index on a tie
    copy candidates/<best> to candidates/best, and into the task's artifact when the task
        asks for the texts in place
    report best's texts, avg(0) and avg(best), the iterations, the accepted mutations and
        merges, metric_calls and the files above

procedure MUTATE():                                              # reflective mutation
    p := SELECT_CANDIDATE(); ids := MINIBATCH()
    before := EVAL(p, ids, train)                                # scores and records
    if skip_perfect_score and every score of before >= perfect_score: log skipped; return
    c := COMPONENT(p)
    text := REFLECT(p's text of c, the records of c); none: log skipped; return
    child := p with c's text := text
    after := EVAL(child, ids, train)
    if sum(after) > sum(before):                         # strict improvement, else rejected
        ADD(the next index, [p], EVAL(child, all val ids, val))
        found_new := true
        if merges_tested < max_merge_invocations: merges_due += 1
    log the iteration

procedure MERGE():                                       # merge two frontier lineages
    D := list(the set of the candidates DOMINATORS() leaves in some front)
    if |D| < 2 or there are fewer than 3 candidates: return none
    repeat up to 10 times until an m is kept (none kept: return none):
        (p, q, a) := TRIPLET(D); none: next
        m, from := per component the text and the one of p, q it comes from: the one that
             changed a's text when the other kept it; when both changed it, the one of the
             higher avg, rng.choice([p, q]) on a tie; when both kept it, p
        if (p, q, from) is in merged or the valset has fewer than 5 examples: next
        add (p, q, from) to merged
        if m equals a candidate: next
    add (p, q, a) to merged
    S := 5 val ids: rng.sample(B, min(|B|, 2, 5 - |S|)) for B := the ids where p beats q,
         then where q beats p, then the ties; then rng.sample(the ids not in S, 5 - |S|)
    after := EVAL(m, S, val)
    if sum(after) >= max(sum of val[p] on S, sum of val[q] on S):
        ADD(the next index, [p, q], EVAL(m, all val ids, val))
        merges_due -= 1; merges_tested += 1
    log the iteration; return evaluated

procedure ADD(k, parents, scores):
    val[k] := scores; parents[k] := parents; write candidates/<k>/
    next_component[k] := the largest next_component of its parents, 0 for the seed

DOMINATORS() := fronts := {{v: front(v)}}; Y := the candidates in the order first met reading
        the fronts by ascending v; for each y of sorted(Y, key=avg):    # stable on equal avg
        if every front that holds y still holds another candidate: remove y from all fronts
    return {{v: {{k for k in front(v) if k was not removed}}}}        # no front is left empty
SELECT_CANDIDATE() := n := {{k: the number of DOMINATORS() fronts that hold k}}, its keys in
    the order first met reading those fronts by ascending v
    return rng.choice([k for k in n for _ in range(n[k])])
MINIBATCH() := s := (i - 1) * minibatch_size; at the first call and at each new epoch
    (s div |order| grew): order := list(range(|trainset|)); rng.shuffle(order); while |order|
    is no multiple of minibatch_size: append the id that is least frequent in order, of
    several the one first met last
    return the minibatch_size ids of order from position s mod |order|
COMPONENT(p) := n := next_component[p]; next_component[p] := (n + 1) mod the number of
    components; return the n-th component name                  # round robin per lineage
TRIPLET(D) := p, q := sorted(rng.sample(D, 2)), neither an ancestor of the other, and
    a := rng.choices(E, weights)[0], E := the common ancestors of p and q (the intersection
    of their ancestor sets) with avg(a) <= avg(p), avg(a) <= avg(q), (p, q, a) not in merged
    and a component whose text in a one of p, q keeps and the other changed, weights := the
    avg of each (none when every avg is 0 or one is negative); none when 10 draws find no
    such p, q, a
SELFTEST() := gepa.py, on a scratch state: rng := random.Random(0), 7 train ids,
    minibatch_size 3, candidates 0..16 added in order, val[k][v] := 1 if (3k + 5v) mod 11 < 3
    else 0 on val ids 0..9; for i := 1..6: SELECT_CANDIDATE() then MINIBATCH() gives the
    candidates [5, 5, 10, 5, 15, 7] and the minibatches [[4, 1, 5], [2, 0, 3], [6, 6, 3],
    [6, 0, 3], [5, 1, 4], [2, 2, 4]], as the official engine; any other answer: gepa.py
    does not make the official draws, correct it
REFLECT(text, records) := your tool reflect(text_file, records_file, new_text_file[, model]):
    one fresh call of the reflection model (the task's; without one, the model you run on;
    the same for the whole run) on the official reflection prompt and nothing else (no tools,
    no system prompt, no memory of the run).  The prompt gives it the text and the records
    and asks it to infer the task from the Inputs, to read every Generated Outputs with its
    Feedback, and to write a new instruction with all niche and domain-specific facts of the
    task in it (the system will not have them later) and every strategy that generalizes.
    The tool writes the proposed text to new_text_file, or says that there is no proposal
    (no records, a reply without text, a prompt the model refused): none.  That file is
    the proposal as it stands: one call per iteration, repeated only when the call failed
```"""
"""The GEPA loop appended to the system prompt of every ``/gepa`` run."""

MODEL = "bestrouter"
"""The run's model: the ``bestrouter`` model-picker SEA, resolved to the model it starts on."""

TOOL_PROFILE = "full"
"""The run's toolset: every built-in tool."""

REFLECTION_PROMPT = (
    "I provided an assistant with the following instructions to perform a task for me:\n"
    "```\n<curr_param>\n```\n\n"
    "The following are examples of different task inputs provided to the assistant along with "
    "the assistant's response for each of them, and some feedback on how the assistant's "
    "response could be better:\n"
    "```\n<side_info>\n```\n\n"
    "Your task is to write a new instruction for the assistant.\n\n"
    "Read the inputs carefully and identify the input format and infer detailed task "
    "description about the task I wish to solve with the assistant.\n\n"
    "Read all the assistant responses and the corresponding feedback. Identify all niche and "
    "domain specific factual information about the task and include it in the instruction, as "
    "a lot of it may not be available to the assistant in the future. The assistant may have "
    "utilized a generalizable strategy to solve the task, if so, include that in the "
    "instruction as well.\n\n"
    "Provide the new instructions within ``` blocks."
)
"""The reflection prompt of the official package (``InstructionProposalSignature``), verbatim."""

FENCE_INFO_LINE = re.compile(
    r"\S*(?:[ \t]+(?:[\w-]+=(?:\"[^\"\n]*\"|'[^'\n]*')|\{[^}\n]*\}))*[ \t]*\r?\n"
)
"""The rest of an opening fence line: a language tag, attributes and the line end."""

OUTPUT_LIMIT_STOPS = frozenset({"length", "max_tokens", "max_output_tokens", "MAX_TOKENS"})
"""The stop reasons of a reply cut at the output limit, as KISS records them
(Chat Completions, Anthropic, the Responses API, Gemini); the official package
reads them as the finish reasons ``length`` and ``max_tokens``."""

TILDE_BLOCK = re.compile(r"(~{3,})[^\n]*\n(?:(.*)\n)?\1~*", re.DOTALL)
"""A reply that is one ``~~~`` block."""


def markdown(value: Any, level: int = 3) -> str:
    """Return a record *value* as the official reflection prompt renders it.

    A dict becomes one header per key and a list one ``Item n`` header per
    item, a level deeper each time (six at most); anything else its text.
    """
    if isinstance(value, dict):
        parts = list(value.items())
    elif isinstance(value, list | tuple):
        parts = [(f"Item {n}", item) for n, item in enumerate(value, 1)]
    else:
        return f"{str(value).strip()}\n\n"
    return "".join(
        f"{'#' * level} {key}\n{markdown(item, min(level + 1, 6))}" for key, item in parts
    ) or "\n"


def reflection_prompt(text: str, records: list[dict[str, Any]]) -> str:
    """Return the official reflection prompt for *text* and the *records* of a minibatch."""
    examples = "\n\n".join(
        f"# Example {n}\n"
        + "".join(f"## {key}\n{markdown(value)}" for key, value in record.items())
        for n, record in enumerate(records, 1)
    )
    return REFLECTION_PROMPT.replace("<curr_param>", text).replace("<side_info>", examples)


def proposal(reply: str, cut: bool = False) -> str:
    """Return the text a reflection *reply* proposes, read as the official parser reads it.

    The text between the first and the last fence when the reply holds
    both (a longer opening fence closes with a run of its own length; a
    language tag on the opening line is dropped); otherwise the reply
    itself, without a ``~~~`` wrapper or a lone fence at either end.
    Empty when the reply proposes nothing: no text, a ``<think>`` block
    that never closes (the reply was cut off), or, when *cut* says that the
    reply was stopped at the output limit, anything but a complete pair of
    fences.
    """
    reply = reply.strip()
    start, end = reply.find("```"), reply.rfind("```")
    if start + 3 < end:
        fence = len(reply) - start - len(reply[start:].lstrip("`"))
        if fence > 3 and reply.rfind("`" * fence) > start + fence:
            end = reply.rfind("`" * fence)
        else:
            fence = 3
        text = reply[start + fence : end]
        info = FENCE_INFO_LINE.match(text)
        return text[info.end() if info else 0 :].strip()
    if cut:
        return ""
    if reply.startswith("<think>") and reply.count("<think>") > reply.count("</think>"):
        return ""
    tilde = TILDE_BLOCK.fullmatch(reply)
    if tilde:
        return (tilde.group(2) or "").strip()
    if reply.startswith("```"):
        return re.sub(r"^```\S*\n?", "", reply).strip()
    return reply.removesuffix("```").strip()


def reflect(text_file: str, records_file: str, new_text_file: str, model: str = "") -> str:
    """Write the official GEPA reflection of a text on its minibatch records to a file.

    The reflection LM of a ``/gepa`` run: one fresh call of a model, with no
    tools and no memory of the run, on the reflection prompt of the official
    package (the current text, every record whole, the instruction to write a
    new text that holds the task's niche and domain-specific facts and the
    strategies that generalize).  The reply is read as the official parser
    reads it, with its rule for a reply that ended early: a reply stopped at
    the output limit proposes its fenced block when that is complete and
    nothing otherwise, and the text a content filter let through is read
    like any reply.  A component without records is not sent, and a prompt
    the model refuses gives no proposal.  The cost of the call is charged to
    the task as a side call, filed under the reflection model in the Spend
    panel.  File names are absolute or relative to the task's work
    directory.

    Args:
        text_file: The file holding the component's current text.
        records_file: The JSON file holding the list of the minibatch's records
            of that component, each an object with ``Inputs``, ``Generated
            Outputs`` and ``Feedback``, whole.
        new_text_file: The file the proposed text is written to.
        model: The reflection model; empty for the model the run is on.

    Returns:
        ``proposal written to <file>: ...`` with the sizes and the cost of the
        call, ``no proposal: <why>`` when the component has no records, the
        reply holds no text, or the reply was refused or cut short before it
        held a proposal (nothing is written), or an ``Error:`` line when the
        caller is not a running task.
    """
    caller = current_agent()
    if caller is None:
        return "Error: reflect is a tool of a running /gepa task, called on the task's thread"
    work_dir = Path(caller.work_dir)
    records = json.loads((work_dir / records_file).read_text())
    if not records:
        return "no proposal: the component has no records"
    prompt = reflection_prompt((work_dir / text_file).read_text(), records)
    name = model or caller.model_name
    config = _parent_model_config(caller, name, False) or {}
    config.pop("system_instruction", None)  # the reflection call has no system prompt
    reflector = KISSAgent("gepa reflection")
    # Bound now, like a ``decide`` call: the spend is charged to this task.
    binding = side_call_binding(caller)
    ended = ""  # the stop reason of a reply that did not end by itself
    try:
        reply = reflector.run(
            model_name=name,
            prompt_template="{prompt}",
            arguments={"prompt": prompt},
            is_agentic=False,
            verbose=False,
            model_config=config or None,
        )
        # Anthropic and Gemini hand a tool-less caller the text of a reply the cap cut.
        stop = (reflector.last_call_usage or {}).get("stop_reason")
        if stop in OUTPUT_LIMIT_STOPS:
            ended = stop
    except (ModelRefusalError, OutputCapError) as error:
        reply, ended = error.text, error.stop_reason
    finally:
        _attribute_side_call_usage(
            caller, name, reflector.budget_used, reflector.total_tokens_used, binding
        )
    text = proposal(reply, ended in OUTPUT_LIMIT_STOPS)
    if not text and ended:
        return f"no proposal: the reply was refused or cut short ({ended})"
    if not text:
        return "no proposal: the reply holds no text"
    target = work_dir / new_text_file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return (
        f"proposal written to {target}: {len(text)} characters, from a prompt of "
        f"{len(prompt)} characters, ${reflector.budget_used:.4f}"
    )


class GepaSea(BaseSea):
    """The ``/gepa`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/gepa help``."""
        return (
            "Runs GEPA (Genetic-Pareto), the reflective optimizer of the text components of a "
            "system (prompts, instructions, code, configuration), on the bestrouter with the "
            "full toolset, as the loop of the official implementation (gepa-ai/gepa) with a "
            "fresh call of its model on the official reflection prompt as the reflection LM: it "
            "scores the seed texts on the validation set, then per iteration draws a candidate "
            "from the per-example Pareto front, runs it on a minibatch of 3 training examples, "
            "has the reflection LM rewrite one component from the outputs and their feedback, "
            "keeps the rewrite only when the minibatch total improves and then "
            "scores it on the whole validation set, optionally merges two frontier candidates, "
            "and stops at the metric-call budget (10 validation passes unless the task names "
            "one), reporting the candidate with the best validation average; give it the work "
            "with `/gepa <the texts to optimize, the training and validation data, the "
            "evaluation command or the metric with its feedback, and the budget>` in the chat "
            'or run_agent(agent="gepa", task="...").'
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Append :data:`GEPA_INSTRUCTIONS` to *system_prompt*, once.

        A prompt that already carries the section is returned as is, so
        the loop is stated once.
        """
        if SECTION_HEADING in system_prompt:
            return system_prompt
        return system_prompt.rstrip("\n") + "\n\n" + GEPA_INSTRUCTIONS + "\n"

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Run on :data:`MODEL` with the :data:`TOOL_PROFILE` toolset."""
        return settings | {"model": MODEL, "tool_profile": TOOL_PROFILE}

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Return *tools* with :func:`reflect`, the run's reflection LM, added."""
        return tools + [reflect]
