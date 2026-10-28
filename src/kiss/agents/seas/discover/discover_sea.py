# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Discover agent: AI discovery, auto research and optimization on the bestrouter.

The agent is the plain Sorcar agent with :data:`DISCOVERY_INSTRUCTIONS`,
the AI discovery loop, appended to the system prompt by the SEA's
``system_prompt`` method.  The loop keeps a Pareto frontier of ideas
against the user's metrics (``<scratch dir>/pareto.md``, seeded with the
baseline), every point a git commit with a parent point: profile the
baseline, fix the dev / held-out split, the dev evaluation command and
its noise floor (``split.md``; every frontier metric is a dev metric),
search for the state of the art, write ideas (``ideas.md``), judge the
unexplored ones pairwise, implement the winner on its parent's commit,
evaluate and commit it, log it with its observations and status
(``explored-ideas.md``: KEPT when no frontier point dominates it, TIE
when it equals one, else FAILED; TIE and FAILED are never retried;
evicted points are DOMINATED), and draw the next operator with one
random number from a Bash call: a variant of a frontier point reasoned
from its observations (r < 0.33), a composition of two frontier points
of different aspects (0.33 <= r < 0.66; a fresh draw when fewer than
two are eligible), or fresh ideas from a web search or ``gpt-6-astra``
or ``gemini-4-argon`` (r >= 0.66).  A frontier point that meets the
goal on the dev metrics is checked once on the held-out split: the loop
stops when it passes and evicts it as OVERFIT when it fails.  The loop
also stops at the budget or the round limit (the task's, else 20
rounds), resets the working tree to the best frontier point's commit
and reports it with its dev and held-out metrics.

The loop used to be a section of the default ``SYSTEM.md``; it lives
here now, so an ordinary Sorcar run no longer carries it and a
``/discover`` run gets it as the last and freshest instruction.  A
prompt that still holds a copy of the section (a custom base prompt) has
that copy cut out first, so the loop is stated once.

The run's model is ``bestrouter``: the daemon resolves the name to the
bestrouter's primary model (``claude-fable-5-1``,
:mod:`kiss.agents.seas.bestrouter.bestrouter_sea`); the bestrouter's own
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
none, and git-ignore data and large outputs) so that every evaluated idea is a commit.

0. Keep a Pareto frontier of ideas against the user's metrics in <scratch dir>/pareto.md,
   one line per point: id, parent id, commit, idea, aspect (the part of the system the
   idea changes: data, features, model, loss, training, algorithm, I/O, ...), dev metrics,
   status; start empty.  Log every evaluated idea in <scratch dir>/explored-ideas.md with
   the same fields plus observations.
1. Read + profile the data / tests / baseline.  Fix the dev / held-out split now, before
   any idea is evaluated, and write it to <scratch dir>/split.md: what is held out (files,
   rows, seed, or a second workload when there is no dataset), the exact dev evaluation
   command, and its noise floor (the spread of two or three baseline runs; metric values
   closer than that are equal).  Every metric in pareto.md and explored-ideas.md is a dev
   metric from that command; the held-out part is read only by the held-out check of
   steps 5 and 6.  Commit the baseline and insert it as the first frontier point (id B,
   no parent).
2. Web-search for SOTA approaches, papers, repos, issues.
3. Append ideas with their aspect, parent frontier point and rationale to
   <scratch dir>/ideas.md.
4. Pairwise judge the ideas not yet in <scratch dir>/explored-ideas.md to find a winner
   idea.
5. One round: `git reset --hard <parent commit>`, implement the idea, run the dev
   evaluation command, `git add -A && git commit`, and log id, parent, commit, idea,
   aspect, metrics, status and observations (what the profile, errors and outputs showed,
   what the next change should be) in <scratch dir>/explored-ideas.md.  Status: KEPT when
   no frontier point dominates it (no worse on every metric and better beyond the noise
   floor on one, in each metric's own direction): insert it and mark the points it
   dominates DOMINATED in both files; TIE when it equals a frontier point on
   every metric (not inserted); FAILED otherwise.  TIE and FAILED ideas are never retried.
   When a frontier point meets the user's goal on the dev metrics, run the held-out /
   generalization check on it once to prove it is not overfit: stop (step 6) if it passes;
   otherwise mark it OVERFIT in both files, evict it (re-insert the baseline if the
   frontier is now empty) and continue.
6. Stop or draw.  A round is one pass through step 5.  Stop when step 5 passed the
   held-out check or when the budget or the round limit (the task's; 100 rounds when the
   task gives none) is hit: pick the best frontier point (the one that passed the check,
   else the frontier point the task's metric priority prefers, its first-named metric
   when it gives none, held-out checked now), `git reset --hard` its commit so the working
   tree holds it, and report it with its dev and held-out metrics and the
   explored-ideas.md counts (KEPT, TIE, FAILED, OVERFIT, DOMINATED).  Otherwise draw the
   next operator with one Bash call, `python3 -c 'import random; print(random.random())'`,
   and record the draw r in <scratch dir>/explored-ideas.md: r < 0.33 -> step 7 (variant);
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
            "Runs an AI discovery loop (profile the baseline and fix a dev / held-out split, "
            "search for the state of the art, write and judge ideas, implement each one on its "
            "parent's git commit and evaluate it on the dev split, keep a Pareto frontier of "
            "the winners, draw the next step at random among a variant of a frontier point, a "
            "composition of two and fresh ideas from the web or gpt-6-astra or gemini-4-argon, "
            "repeat until a frontier idea meets the metric goal and passes the held-out check "
            "or the budget or round limit is hit, 20 rounds by default) on the "
            "bestrouter with the full toolset, with the "
            "discovery instructions appended to the system prompt; use `/discover <what "
            'to improve, the metric and the goal>` in the chat or run_agent(agent="discover", '
            'task="...").'
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
