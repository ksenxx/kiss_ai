# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Discover agent: AI discovery, auto research and optimization on the bestrouter.

The agent is the plain Sorcar agent with :data:`DISCOVERY_INSTRUCTIONS`,
the AI discovery loop written as pseudocode (``DISCOVER``, ``ROUND``,
``FINISH`` and their helpers), appended to the system prompt by the SEA's
``system_prompt`` method.  The loop keeps a Pareto frontier of ideas
against the user's metrics (``<scratch dir>/pareto.md``, seeded with the
baseline), every point a git commit with a parent point, each commit
held by a ref ``refs/discover/<task id>/<id>`` so that the resets and
the squash merge of the task branch do not orphan it: fix the dev /
held-out split, the dev evaluation command and its noise floor
(``split.md``, fixed for the whole run; every frontier metric is a dev
metric, the noise floor of a deterministic command comes from bootstrap
resamples of one run's per-item dev results, or is 0 when it yields one
number, and nothing opens the held-out part but the held-out check and,
when the task names no held-out part, the split command, run before
anything else reads the data), profile the dev data and the baseline, search for
the state of the art, write ideas
(``ideas.md``), judge the unexplored ones pairwise, implement the winner
on its parent's commit, evaluate and commit it, log it with the draw
that led to it, its status and observations (``explored-ideas.md``, one
row per idea in the order evaluated, the baseline first, whose status
alone changes later, in place: KEPT when no frontier point
dominates it, TIE when it equals one, else FAILED; TIE and FAILED are
never retried; a status lives there alone, and an evicted point leaves
``pareto.md`` and becomes DOMINATED), and draw the next operator with one
random number from a Bash call: a variant of a frontier point reasoned
from its observations (r < 0.33), a composition of two frontier points
(0.33 <= r < 0.66; an eligible pair has neither the baseline nor a
shared aspect, neither commit is an ancestor of the other, a compose
being a ``git merge`` commit, and no logged idea already merged them;
when no pair is eligible the draw forks the oldest frontier point that
is an ancestor of another one with an idea on an aspect that other
point lacks, so that rounds built one on another still yield a pair to
compose, and it is a fresh draw when there is no such point), or fresh
ideas from a web search, ``gpt-6-astra`` or ``gemini-4-argon``, picked
by a second draw at the same thresholds (r >= 0.66); variant, fork and
fresh ideas go to ``ideas.md`` too.  A frontier or TIE point that meets
the goal on the dev metrics is checked once on the held-out split: the
loop stops when it passes and evicts it as OVERFIT when it fails.  The
loop also stops at the budget or the round limit (the task's, else 100
rounds); then it held-out checks the frontier and TIE points in the
task's metric priority and takes the first that passes, else the
baseline.  A point passes the held-out check only when it is no worse
than the baseline there on the task's first-priority metric, by any
amount, and the baseline neither dominates nor equals it there; one that
fails is evicted as OVERFIT.  It resets the working tree to the chosen
point's commit and reports it with its dev and held-out metrics.

A task that asks only for research, ideas or a ranking still gets the
loop, ranked by measured dev metrics.  The loop resets and commits the
tree it works in, so a task that forbids changing its repository gets
it in a lab: a copy of the artifact's current files under ``<scratch
dir>/lab`` with its own git history and worktree, whose best point is
handed back as ``<scratch dir>/best.patch``.  A caller that keeps
working in the repository while a ``/discover`` sub-agent runs forbids
it to change the repository; a sub-agent allowed to change it works in
the caller's worktree, as every sub-agent does.

A task the user types that names ``/discover`` gets :meth:`DiscoverSea.description`
after its text (``sea_commands.named_commands_help``), so the description
tells such a caller to run its improvement work through the SEA, again
while the goal is unmet, instead of using it once as a consultant.

The loop used to be a section of the default ``SYSTEM.md``; it lives
here now, so an ordinary Sorcar run no longer carries it and a
``/discover`` run gets it as the last and freshest instruction.  A
prompt that still holds a copy of the section (a custom base prompt) has
that copy cut out first, so the loop is stated once.

The run's model is ``bestrouter``: the daemon resolves the name to the
bestrouter's primary model
(:data:`kiss.agents.seas.bestrouter.bestrouter_sea.PRIMARY_MODEL`); the bestrouter's own
review protocol is added only when ``bestrouter`` is also the tab's
picked model.  The toolset is the full one, so the discovery loop can
edit, run, browse and remember.

Three ways to run it::

    /discover raise the accuracy of the classifier in src/model.py above 0.9 on data/test.csv

    /discover make the ingest script in scripts/ingest.py at least twice as fast

    run_agent(agent="src/kiss/agents/seas/discover/discover_sea.py", task="...")

The task names the artifact to improve, the metric and the goal.
"""

from __future__ import annotations

import re
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea

SECTION_HEADING = "## AI discovery, auto research, and optimization"
"""The heading of the AI discovery section (:data:`DISCOVERY_INSTRUCTIONS`)."""

DISCOVERY_INSTRUCTIONS = f"""\
{SECTION_HEADING}

Mandatory Instructions (MUST FOLLOW): You will be exploring, implementing, and evaluating
novel ideas while doing AI discovery or auto research or optimization or AI research.
Run DISCOVER(task) below exactly.  A task that asks only for research, ideas or a ranking
still gets it, ranked by measured dev metrics.

```
# Files in <scratch dir>:
#   split.md           what is held out (files, rows, seed, or a second workload when there
#                      is no dataset), the split command, the exact dev evaluation command
#                      and its noise floor; written once, never changed in the run
#   pareto.md          the frontier, one line per point: id, parent id, commit, idea,
#                      aspect, dev metrics; a point that leaves the frontier leaves the file
#   explored-ideas.md  the log, one row per evaluated idea in the order evaluated: the
#                      pareto.md fields + draw, status, observations; rows are never deleted
#                      or reordered, only a row's status changes, in place; a point's
#                      status lives here alone
#   ideas.md           ideas, each with its aspect, parent frontier point and rationale
# aspect    the part of the system an idea changes: data, features, model, loss, training,
#           algorithm, I/O, ...; a point's aspect is its own idea's, not its ancestors';
#           a compose has both of its points' aspects
# metrics   dev metrics from the split.md evaluation command, the only metrics in the files
# status    KEPT | TIE | FAILED | DOMINATED | OVERFIT
# RANDOM()  one Bash call: python3 -c 'import random; print(random.random())'
# RESET(c)  git reset --hard c && git clean -fd
# COMMIT(id) git add -A && git commit && git update-ref refs/discover/<task id>/<id> HEAD
#           (the ref keeps the commit reachable through resets and squash merges)
# HELD-OUT RULE: when the task names the held-out part, it is off limits from the first
#   command on; else no command reads the data before the split command, which alone reads
#   all of it, once.  After that only HELDOUT_CHECK opens the held-out part: no cat, head,
#   wc, cut, count, profile or test.
# Comparisons use the noise floor and each metric's own direction:
#   x DOMINATES y := x is no worse than y on every metric and better beyond noise on one
#   x EQUALS y    := every metric of x is within noise of y's

procedure DISCOVER(task):
    round_limit := the task's round limit, else 100
    if the task forbids changing its repository:          # a lab: no reset touches its tree
        copy the artifact's current files (not its data) to <scratch dir>/lab, `git init`
        and commit them there, call create_worktree on it, and work in the worktree it
        returns (in the lab when it makes none)
    else: work in the task's git repo, else `git init` one around the artifact
    in the repo you work in, git-ignore data, large outputs and <scratch dir> (or
        `git clean -fd` deletes them)

    heldout := the task's held-out part, else the part one split command holds out
    eval    := the exact dev evaluation command
    noise   := NOISE_FLOOR(eval)                   # dev values closer than noise are equal
    write split.md
    read + profile the dev data, the tests and the baseline
    COMMIT(B); B := point(id B, parent none, commit HEAD, metrics eval())
    pareto.md := [B]; log B with draw none, status KEPT

    web-search for SOTA approaches, papers, repos, issues; append ideas to ideas.md
    idea, parent := PAIRWISE_WINNER(); draw := none
    rounds := 0
    loop:
        winner := ROUND(idea, parent, draw); rounds += 1
        if winner != none: return FINISH(winner)
        if the budget is spent or rounds == round_limit: return FINISH(none)
        r := RANDOM()
        if r < 0.33:                                                    # variant
            p := a frontier point
            idea := one change, reasoned from p's logged observations, that should move
                    the frontier; append it to ideas.md; parent := p
        elif r < 0.66 and ELIGIBLE_PAIRS() is not empty:                # compose
            (p, q) := a pair from ELIGIBLE_PAIRS()
            idea := "merge q into p"; parent := p
        elif r < 0.66 and FORK_POINT() != none:          # fork, so that a pair can compose
            (o, d) := FORK_POINT()
            idea := a change on an aspect d does not have: the pairwise winner of such ideas
                    in ideas.md, else one reasoned from o's logged observations, appended
                    to ideas.md; parent := o
        else:                             # fresh, also a compose draw with no pair or fork
            s := RANDOM()
            source := extensive internet search if s < 0.33, gpt-6-astra if s < 0.66,
                      else gemini-4-argon
            with deep thinking, append fresh ideas from source that are not in
                explored-ideas.md to ideas.md
            idea, parent := PAIRWISE_WINNER()
        draw := r, plus s for a fresh idea

procedure ROUND(idea, parent, draw):
    RESET(parent.commit)
    implement idea                # a compose: `git merge --no-commit <q commit>`, resolve
    m := eval()
    COMMIT(id); x := point(id, parent, HEAD, idea, aspect, m)
    if some frontier point DOMINATES x:  status := FAILED
    elif some frontier point EQUALS x:   status := TIE                  # not inserted
    else:
        status := KEPT; insert x in pareto.md
        for each frontier point f that x DOMINATES:
            remove f from pareto.md; f.status := DOMINATED
    append x, draw, status and observations (what the profile, errors and outputs showed,
        what the next change should be) to explored-ideas.md    # TIE, FAILED never retried
    for each frontier or TIE point f not yet held-out checked whose dev metrics meet the goal:
        if HELDOUT_CHECK(f) meets the goal and HELDOUT_PASS(f): return f  # each f once
        remove f from pareto.md if it is there; f.status := OVERFIT
        if pareto.md is empty: re-insert B; B.status := KEPT
    return none

procedure FINISH(winner):
    best := winner
    if best == none:     # budget or round limit: check frontier and TIE points in turn
        for each frontier or TIE point f != B, in the task's metric priority on the dev
                metrics (its first-named metric when it gives none):
            if HELDOUT_PASS(f): best := f; break
            remove f from pareto.md if it is there; f.status := OVERFIT
        if best == none:                # report that no point beat the baseline held out
            best := B; re-insert B in pareto.md if absent; B.status := KEPT
    if best.status == TIE: insert best in pareto.md; best.status := KEPT
    RESET(best.commit)
    in a lab: write `git diff --binary <B commit> <best commit>` to <scratch dir>/best.patch
    report best with its dev and held-out metrics (HELDOUT_CHECK(best)), whether it meets
        the goal on both, the explored-ideas.md counts (KEPT, TIE, FAILED, OVERFIT,
        DOMINATED), the refs/discover/<task id>/ refs that hold every evaluated commit, and
        best.patch in a lab

ELIGIBLE_PAIRS() := every pair (p, q) of frontier points where neither is B, they share
    no aspect, neither commit is an ancestor of the other (`git merge-base --is-ancestor`;
    a compose is a merge commit, so this also rules out a point composed from the other),
    no explored-ideas.md row already merged q into p, and p is the better of the two on
    the task's first metric
FORK_POINT() := (o, d) for the oldest frontier point o that is an ancestor of another
    frontier point d (`git merge-base --is-ancestor`), else none; a fork of o and d are
    no ancestors of each other, so they make an eligible pair while both are on the frontier
HELDOUT_PASS(f) := f is B, or with hB := HELDOUT_CHECK(B) and hf := HELDOUT_CHECK(f): hf is
    no worse than hB on the task's first-priority metric (by any amount, no noise
    allowance), and hB neither DOMINATES nor EQUALS hf
NOISE_FLOOR(eval) := the spread of two or three baseline runs, with different seeds when
    eval takes one; for a deterministic eval, the spread of the metric over bootstrap
    resamples of one baseline run's per-item dev results (the dev split stays as it is),
    or 0 when it yields one number
PAIRWISE_WINNER() := judge the ideas in ideas.md not yet in explored-ideas.md pairwise;
    return the winner and the frontier point it builds on
HELDOUT_CHECK(f) := f's metrics on the held-out part, run once per point
```"""
"""The AI discovery loop appended to the system prompt of every ``/discover`` run."""

MODEL = "bestrouter"
"""The run's model: the ``bestrouter`` model-picker SEA, resolved to its primary model."""

TOOL_PROFILE = "full"
"""The run's toolset: every built-in tool."""


def split_discovery_section(system_prompt: str) -> tuple[str, str]:
    """Cut an AI discovery section out of *system_prompt*.

    The section is :data:`SECTION_HEADING` on a line of its own and
    everything up to the next ``## `` heading (or the end of the text),
    with the blank lines that separate it from that heading.

    Args:
        system_prompt: A system prompt.

    Returns:
        ``(rest, section)``: the prompt without the section and the
        section itself; ``(system_prompt, "")`` when the prompt has no
        such section.
    """
    match = re.search(rf"^{re.escape(SECTION_HEADING)}$", system_prompt, re.MULTILINE)
    if match is None:
        return system_prompt, ""
    start = match.start()
    end = system_prompt.find("\n## ", match.end())
    end = len(system_prompt) if end < 0 else end + 1
    return system_prompt[:start] + system_prompt[end:], system_prompt[start:end]


class DiscoverSea(BaseSea):
    """The ``/discover`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/discover help``."""
        return (
            "Runs an AI discovery loop on the bestrouter with the full toolset: it fixes a "
            "dev / held-out split, searches the state of the art, implements and evaluates one "
            "idea per round as a git commit, keeps a Pareto frontier of the winners, and stops "
            "when a frontier point meets the task's goal and passes the held-out check, or at "
            "the budget or round limit (100 rounds by default); give it the work, not a "
            "question, with `/discover <what to improve, the metric and the goal>` in the chat "
            'or run_agent(agent="discover", task="..."), and a task told to use /discover runs '
            "its improvement work through it, calling it again while the goal is unmet (a "
            "research-only task still gets measured rounds; one that forbids changing its "
            "repository gets them in a lab copy under the scratch dir and the best point back "
            "as a patch)."
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Append :data:`DISCOVERY_INSTRUCTIONS` to *system_prompt*.

        A copy of the section already in the prompt is cut out first, so
        the loop is stated once, at the end.
        """
        rest, _section = split_discovery_section(system_prompt)
        return rest.rstrip("\n") + "\n\n" + DISCOVERY_INSTRUCTIONS + "\n"

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Run on :data:`MODEL` with the :data:`TOOL_PROFILE` toolset."""
        return settings | {"model": MODEL, "tool_profile": TOOL_PROFILE}
