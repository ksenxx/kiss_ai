# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Discover-viz agent: an HTML animation of the Pareto frontier of a ``/discover`` run.

``/discover_viz <task id>`` reads the persisted trajectory of a
``/discover`` run and writes one self-contained HTML page: the
evaluated points appear one by one, each under its parent, a tooltip
on a point shows its ``pareto.md`` details, and after every round the
page says how the idea was picked and whether the point entered the
frontier or was discarded.

A run's files differ in layout from run to run, so the model reads
them and two tools keep it honest:

* :func:`discover_trajectory` writes the evidence of the run
  (:func:`kiss.agents.seas.discover_viz.trajectory.evidence`);
* the model writes the story of the run as JSON
  (:class:`kiss.agents.seas.discover_viz.story.Story`);
* :func:`render_discover_viz` reads the trajectory again, refuses a
  story the loop could not have produced or that quotes a value the
  run did not record, and writes the page
  (:func:`kiss.agents.seas.discover_viz.animation.render`).

Three ways to run it::

    /discover_viz                      the newest /discover run
    /discover_viz 9de623ce             a run by the first characters of its task id
    run_agent(agent="discover_viz", task="<task id>, write it to docs/run.html")
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.discover_viz import animation, story, trajectory
from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.task_digest import child_tasks, load_task

ADD_TO_PROMPT = "Animate the /discover run named above; the newest one when none is named."
"""Ends every run's prompt, and is all of it when the user typed the command alone."""

MAX_ERRORS = 25
"""How many story errors one refusal of :func:`render_discover_viz` lists."""

SYSTEM_PROMPT = """\
You turn the trajectory of a /discover run into an HTML animation of its Pareto frontier.

A /discover run evaluates one idea per round. Every evaluated idea is a point with a parent
point. `pareto.md` holds the frontier, one line per point. `explored-ideas.md` logs every
point in the order evaluated: the pareto.md fields plus the random draw that picked the
idea, the status (KEPT, TIE, FAILED, DOMINATED, OVERFIT) and observations. A draw r picks
the operator: variant of a frontier point (r < 0.33); compose of two frontier points, else
a fork, else fresh ideas when the frontier allows neither (0.33 <= r < 0.66); fresh ideas
(r >= 0.66). The first idea needs no draw.

Procedure:
1. Call `discover_trajectory(task, evidence_file)`. `task` is the task id the user gave or
   its first characters, empty for the newest run, `list` to see the runs; a task that
   dispatched /discover runs stands for the newest of them. `evidence_file`
   is `evidence.md` in your scratch dir. Read all of that file.
2. Write `story.json` in your scratch dir, one JSON object:
   - `task_id`: the run's full id. `title`: what the run improved, at most 80 characters.
     `goal`: its goal in one sentence. `metric`: the first metric's name and direction, at
     most 60 characters, as in `dev accuracy, higher is better`.
   - `rounds`: one object per evaluated point, in the order evaluated, the baseline first:
     - `id`, and `parents`: [] for the baseline, [parent] otherwise, [parent, merged point]
       for a compose.
     - `idea`: the idea in one sentence.
     - `operator`: baseline, initial (the first idea, picked without a draw), variant,
       compose, fork or fresh. `draw`: the draw behind it as logged, "" when none.
     - `picked`: why this idea and this parent, in one or two sentences.
     - `score`: the point's value of the first metric as logged.
     - `status`: KEPT (it entered the frontier), TIE or FAILED, as at the end of its own
       round. DOMINATED and OVERFIT come later, as a change of the round that caused them.
     - `outcome`: what the evaluation showed and why the point was kept or discarded, in
       one or two sentences.
     - `details`: the point's pareto.md line as {field: value}, in file order under the
       file's field names; for a point that never had a pareto.md line, the same fields of
       its explored-ideas.md row. `source`: the file they come from, `pareto.md` or
       `explored-ideas.md`.
     - `changes`: [{`id`, `status`, `why`}], one per point this round moved: DOMINATED (it
       left the frontier), OVERFIT (it failed the held-out check), KEPT (it joined the
       frontier: the baseline put back, a TIE point promoted). [] when none.
   - `finish`: {`best`, `reason`, `changes`}: the point the run reports, why the run
     stopped, the changes made while finishing; null when the run did not finish.
3. Call `render_discover_viz(story_file, html_file)`. `html_file` is the path the user
   named, else `reports/discover-viz-<first 8 characters of the task id>.html` in the work
   dir. Fix every error it lists in story.json and call it again until it writes the page.
4. Open the URL it returns with `go_to_url`. `evaluate_script` of
   `() => window.discoverViz.selfCheck()` must return `ok: true`; it leaves the page on the
   last frame. Take one `screenshot` and look at it: the graph must be legible. Then call
   `show_browser()` so the user watches the animation; if that fails, go on.
5. Call `finish` with the path of the page, the number of rounds and the final frontier.

Rules:
- Copy `details` values, `score` and `draw` character for character from the evidence. In
  your own sentences quote numbers as the run recorded them and compute none.
- Report only what the evidence shows. When it shows no evaluated point, finish with
  `success=false` and say so.
- The page prints `idea`, `picked`, `outcome`, `why` and `reason` as you write them, beside
  its own sentence for the operator and the status. Write them for a reader who has not
  seen the run: plain words, no filler.
- Pass absolute paths. Never read history.db yourself.
"""


def _not_absolute(*paths: str) -> str:
    """Return the ``Error:`` line that names the *paths* that are not absolute, ``""`` for none."""
    relative = [repr(path) for path in paths if not Path(path).is_absolute()]
    return f"Error: pass absolute paths, not {', '.join(relative)}." if relative else ""


def discover_trajectory(task: str, evidence_file: str) -> str:
    """Write the evidence of a /discover run, read from its trajectory, to a file.

    Args:
        task: The run's task id, the first six or more characters of
            it, empty or ``latest`` for the newest /discover run, or
            ``list`` to list the newest runs instead of writing a file.
            A task that is not a /discover run but dispatched some
            stands for the newest of them.
        evidence_file: The absolute path of the Markdown file to write.

    Returns:
        The run found, the file written and its size; the list of runs
        for ``list``; or an ``Error:`` line, followed by the runs to
        choose from when *task* names no /discover run.
    """
    if task.strip().lower() == "list":
        return trajectory.recent_runs()
    if error := _not_absolute(evidence_file):
        return error
    found = trajectory.find_task(task)
    if isinstance(found, str):
        return found
    note = ""
    if found.get("sea") != trajectory.DISCOVER_SEA:
        runs = [
            child for child in child_tasks(found["id"])
            if child.get("sea") == trajectory.DISCOVER_SEA
        ]
        if runs:
            note = (
                f"Task {found['id']} is not a /discover run; it dispatched "
                + "\n".join(["these /discover runs:", *map(trajectory.run_line, runs)])
                + f"\nAnimating the newest, {runs[-1]['id']}.\n"
            )
            found = load_task(str(runs[-1]["id"])) or found
    text = trajectory.evidence(found)
    if not text:
        return (
            f"Error: task {found['id']} wrote no pareto.md or explored-ideas.md, so it ran no "
            f"discovery loop.\n{trajectory.recent_runs()}"
        )
    path = Path(evidence_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return (
        f"{note}/discover run {found['id']}: evidence written to {path}, {len(text)} characters "
        f"in {text.count(chr(10))} lines. Read all of it before you write the story."
    )


def render_discover_viz(story_file: str, html_file: str) -> str:
    """Check the story of a /discover run against its trajectory and write the HTML animation.

    Args:
        story_file: The absolute path of the JSON story.
        html_file: The absolute path of the HTML page to write.

    Returns:
        The page written, its ``file://`` URL, its frame count and the
        final frontier; or ``Error:`` and what to fix in the story, one
        line per problem (nothing is written then).
    """
    if error := _not_absolute(story_file, html_file):
        return error
    try:
        data = json.loads(Path(story_file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"Error: cannot read the story {story_file}: {error}"
    task_id = data.get("task_id") if isinstance(data, dict) else None
    task = load_task(task_id) if persistence.is_task_history_id(task_id) else None
    evidence = trajectory.evidence(task) if task is not None else ""
    if not evidence:
        return "Error: task_id must be the full id of the /discover run, as the evidence gives it."
    parsed = story.parse(data)
    if isinstance(parsed, list):
        return _refusal(parsed)
    frames, errors = story.replay(parsed)
    errors += story.quote_errors(parsed, evidence)
    if errors:
        return _refusal(errors)
    path = Path(html_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(animation.render(parsed, frames), encoding="utf-8")
    final = ", ".join(story.frontier(frames[-1]["states"]))
    return (
        f"Animation written to {path}: {len(parsed.rounds)} points, {len(frames)} frames, final "
        f"frontier {final}. Compare that frontier with the last pareto.md of the evidence, then "
        f"open {path.as_uri()}"
    )


def _refusal(errors: list[str]) -> str:
    """Return the refusal of a story with *errors*, at most :data:`MAX_ERRORS` of them listed."""
    more = len(errors) - MAX_ERRORS
    listed = errors[:MAX_ERRORS] + ([f"... and {more} more"] if more > 0 else [])
    return "Error: the story is refused, nothing was written. Fix:\n- " + "\n- ".join(listed)


class DiscoverVizSea(WorkerSea):
    """The ``/discover_viz`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/discover_viz help``."""
        return (
            "Animates the Pareto frontier of a /discover run from its persisted trajectory: one "
            "self-contained HTML page in which the evaluated points appear one by one under "
            "their parents, a tooltip shows a point's pareto.md details, and every round says "
            "how its idea was picked and whether the point entered the frontier or was "
            "discarded; use it as `/discover_viz <task id>` (no id: the newest /discover run) "
            'or run_agent(agent="discover_viz", task="<task id>").'
        )

    def prompt(self, task: str) -> str:
        """Return *task* followed by :data:`ADD_TO_PROMPT`, which a bare command runs on alone."""
        return f"{task.strip()}\n\n{ADD_TO_PROMPT}".strip()

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's own system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Run with the shell, edit and browser tools: the agent opens the page it wrote."""
        return settings | {"tool_profile": "shell+edit+browser", "use_web_tools": True}

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Add :func:`discover_trajectory` and :func:`render_discover_viz` to the run's tools."""
        return [*tools, discover_trajectory, render_discover_viz]
