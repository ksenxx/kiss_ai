# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Discover agent: AI discovery, auto research and optimization on the bestrouter.

The agent is the plain Sorcar agent with :data:`DISCOVERY_INSTRUCTIONS`,
the AI discovery loop, appended to the system prompt by the SEA's
``system_prompt`` method.  The loop keeps a Pareto frontier of ideas
against the user's metrics (``<scratch dir>/pareto.md``, seeded with the
baseline), every point a git commit with a parent point, each commit
held by a ref ``refs/discover/<task id>/<id>`` so that the resets and
the squash merge of the task branch do not orphan it: fix the dev /
held-out split, the dev evaluation command and its noise floor
(``split.md``, fixed for the whole run; every frontier metric is a dev
metric, and nothing but the held-out check opens the held-out part, from
the first command on), profile the dev data and the baseline, search for
the state of the art, write ideas
(``ideas.md``), judge the unexplored ones pairwise, implement the winner
on its parent's commit, evaluate and commit it, log it with the draw
that led to it, its status and observations (``explored-ideas.md``, one
row per idea in the order evaluated: KEPT when no frontier point
dominates it, TIE when it equals one, else FAILED; TIE and FAILED are
never retried; a status lives there alone, and an evicted point leaves
``pareto.md`` and becomes DOMINATED), and draw the next operator with one
random number from a Bash call: a variant of a frontier point reasoned
from its observations (r < 0.33), a composition of two frontier points
of different aspects (0.33 <= r < 0.66; a fresh draw when fewer than
two are eligible), or fresh ideas from a web search or ``gpt-6-astra``
or ``gemini-4-argon`` (r >= 0.66).  A frontier point that meets the
goal on the dev metrics is checked once on the held-out split: the loop
stops when it passes and evicts it as OVERFIT when it fails.  The loop
also stops at the budget or the round limit (the task's, else 100
rounds), resets the working tree to the best frontier point's commit
and reports it with its dev and held-out metrics.

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

from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea

SECTION_HEADING = "## AI discovery, auto research, and optimization"
"""The heading of the AI discovery section (:data:`DISCOVERY_INSTRUCTIONS`)."""

DISCOVERY_INSTRUCTIONS = f"""\
{SECTION_HEADING}

Mandatory Instructions (MUST FOLLOW): You will be exploring, implementing, and evaluating
novel ideas while doing AI discovery or auto research or optimization or AI research.
Work in a git repository (the task's; `git init` one around the artifact when it has
none, and git-ignore data and large outputs) so that every evaluated idea is a commit,
kept reachable through resets and squash merges by its ref refs/discover/<task id>/<id>.
The held-out part (named by the task, else written by the split of step 1) is off limits
from the moment it exists, the first command included: nothing but the held-out check of
steps 5 and 6 opens it, no cat, head, wc, cut, count, profile or test.
A task that asks only for research, ideas or a ranking still gets this loop, ranked by
measured dev metrics.  When the task forbids changing its repository, work in a lab so
that no `git reset` touches that tree: copy the artifact's current files (not its data)
to <scratch dir>/lab, `git init` and commit them there, and call create_worktree on it;
run every round in the worktree it returns (in the lab when it makes none).

0. Keep the Pareto frontier of ideas against the user's metrics in <scratch dir>/pareto.md,
   one line per frontier point: id, parent id, commit, idea, aspect (the part of the
   system the idea changes: data, features, model, loss, training, algorithm, I/O, ...),
   dev metrics; start empty.  Log every evaluated idea in <scratch dir>/explored-ideas.md,
   one row appended per idea in the order evaluated, with the same fields plus draw (the
   step 6 draw that led to it, with a fresh draw's second draw), status and observations.
   A point's status lives there alone: a point that leaves the frontier leaves pareto.md.
1. Fix the dev / held-out split (the task's when it names one) and write it to
   <scratch dir>/split.md: what is held out (files, rows, seed, or a second workload when
   there is no dataset), the exact dev evaluation command, and its noise floor (the spread
   of two or three baseline runs, varied by seed or fold assignment when the command is
   deterministic; metric values closer than that are equal); none of it changes later in
   the run.  Every metric in pareto.md and explored-ideas.md is a dev metric from that
   command.  Then read + profile the dev data, the tests and the baseline, commit the
   baseline, run `git update-ref refs/discover/<task id>/B HEAD`, and insert it as the
   first frontier point (id B, no parent).
2. Web-search for SOTA approaches, papers, repos, issues.
3. Append ideas with their aspect, parent frontier point and rationale to
   <scratch dir>/ideas.md.
4. Pairwise judge the ideas not yet in <scratch dir>/explored-ideas.md to find a winner
   idea.
5. One round: `git reset --hard <parent commit>`, implement the idea, run the dev
   evaluation command, `git add -A && git commit && git update-ref
   refs/discover/<task id>/<id> HEAD`, and append id, parent, commit, idea, aspect, draw,
   metrics, status and observations (what the profile, errors and outputs showed, what
   the next change should be) to <scratch dir>/explored-ideas.md.  Status: KEPT when no
   frontier point dominates it (no worse on every metric and better beyond the noise
   floor on one, in each metric's own direction): insert it in pareto.md, drop the points
   it dominates from pareto.md and set their explored-ideas.md status to DOMINATED; TIE
   when it equals a frontier point on every metric (not inserted); FAILED otherwise.  TIE
   and FAILED ideas are never retried.  When a frontier point meets the user's goal on the
   dev metrics, run the held-out / generalization check on it once to prove it is not
   overfit: stop (step 6) if it passes (the held-out metrics meet the goal too); otherwise
   drop it from pareto.md, set its status to OVERFIT (re-insert the baseline, KEPT again,
   if the frontier is now empty) and continue.
6. Stop or draw.  A round is one pass through step 5.  Stop when step 5 passed the
   held-out check or when the budget or the round limit (the task's; 100 rounds when the
   task gives none) is hit: pick the best frontier point (the one that passed the check,
   else the frontier point the task's metric priority prefers, its first-named metric
   when it gives none, held-out checked now), `git reset --hard` its commit so the working
   tree holds it, and report it with its dev and held-out metrics, the
   explored-ideas.md counts (KEPT, TIE, FAILED, OVERFIT, DOMINATED) and the
   refs/discover/<task id>/ refs that hold every evaluated commit; in a lab, also write
   `git diff --binary <B commit> <its commit>` to <scratch dir>/best.patch and name it in
   the report.  Otherwise draw the
   next operator with one Bash call, `python3 -c 'import random; print(random.random())'`,
   logged by step 5 as the draw of the idea it leads to: r < 0.33 -> step 7 (variant);
   0.33 <= r < 0.66 -> step 8 (compose); r >= 0.66 -> step 9 (fresh).  A compose draw with
   fewer than two eligible frontier points is a fresh draw.
7. Variant: pick a frontier point and, from the observations logged for it, reason out one
   change that should move the frontier.  Go to step 5 with that point as the parent.
8. Compose: pick two frontier points of different aspects, neither the baseline nor a
   composition of the other, and merge the changes of the second into the first (the
   better one on the task's first metric).  Go to step 5 with the first as the parent.
9. Fresh: extensive internet search or gpt-6-astra or gemini-4-argon (chosen by a second
   draw), with deep thinking, for fresh ideas that are not in
   <scratch dir>/explored-ideas.md; pairwise judge the ideas not yet in
   <scratch dir>/explored-ideas.md to find a winner idea; go to step 5 with the frontier
   point it builds on as the parent."""
"""The AI discovery loop appended to the system prompt of every ``/discover`` run."""

MODEL = "bestrouter"
"""The run's model: the ``bestrouter`` model-picker SEA, resolved to its primary model."""

TOOL_PROFILE = "full"
"""The run's toolset: every built-in tool."""


def split_discovery_section(system_prompt: str) -> tuple[str, str]:
    """Cut an AI discovery section out of *system_prompt*.

    The section is :data:`SECTION_HEADING` and everything up to the next
    ``## `` heading (or the end of the text), with the blank lines that
    separate it from that heading.

    Args:
        system_prompt: A system prompt.

    Returns:
        ``(rest, section)``: the prompt without the section and the
        section itself; ``(system_prompt, "")`` when the prompt has no
        such section.
    """
    start = system_prompt.find(SECTION_HEADING)
    if start < 0:
        return system_prompt, ""
    end = system_prompt.find("\n## ", start + len(SECTION_HEADING))
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
