# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The restricted-profile note names exactly the tools the run is given.

``SorcarAgent.run`` appends a "Restricted tool profile" note listing the
profile's tools.  ``_get_tools`` builds ``checkpoint`` / ``rollback`` only
for a top-level host run and the worktree tools only for a host run, so a
sub-agent (``_subagent_info`` set, as the daemon does for every
``run_agent`` / ``run_parallel`` child) and a Docker run must not be
promised them.  Each test runs a real agent against the scripted local
model server and compares the note with the tool list the model received.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import docker
import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.sorcar.docker_test_containers import (
    IMAGE,
    image_container_ids,
    is_docker_available,
    remove_new_image_containers,
)
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve

PROFILE = "shell+edit"


def _run_and_capture(agent: SorcarAgent, work_dir: Path, **run_kwargs: Any) -> dict[str, Any]:
    """Run *agent* with the ``shell+edit`` profile; return the first model request."""
    with serve([finish_body("<p>ok</p>", prompt_tokens=500)]) as (url, requests):
        agent.run(
            prompt_template="Say ok.",
            model_name=MODEL,
            max_steps=2,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            work_dir=str(work_dir),
            use_memory=False,
            verbose=False,
            tool_profile=PROFILE,
            **run_kwargs,
        )
    return requests[0]


def _noted_and_sent(request: dict[str, Any]) -> tuple[set[str], set[str]]:
    """The tool names the note lists and the tool names the model was sent."""
    system = str(next(m for m in request["messages"] if m["role"] == "system")["content"])
    note = system.split(f"# Restricted tool profile: {PROFILE}", 1)[1]
    listed = note.split("plus finish: ", 1)[1].split(". Rules above", 1)[0]
    noted = {name.strip() for name in listed.replace("\n", " ").split(",")}
    sent = {t["function"]["name"] for t in request["tools"]} - {"finish"}
    return noted, sent


def test_sub_agent_note_omits_checkpoint_and_rollback(tmp_path: Path) -> None:
    """A sub-agent gets no checkpoint/rollback, and its note does not list them."""
    agent = SorcarAgent("sub-note")
    agent._subagent_info = {
        "parent_task_id": "parent-task", "parent_tab_id": "", "reviewer": False,
        "side_channel": False,
    }
    noted, sent = _noted_and_sent(_run_and_capture(agent, tmp_path))
    assert "checkpoint" not in sent and "rollback" not in sent
    assert noted == sent


def test_top_level_note_lists_checkpoint_and_rollback(tmp_path: Path) -> None:
    """A top-level host run gets both tools and is told about them."""
    noted, sent = _noted_and_sent(_run_and_capture(SorcarAgent("top-note"), tmp_path))
    assert {"checkpoint", "rollback", "create_worktree", "commit_and_merge"} <= sent
    assert noted == sent


@pytest.mark.skipif(not is_docker_available(), reason="needs a Docker daemon")
def test_docker_note_omits_worktree_and_checkpoint_tools(tmp_path: Path) -> None:
    """A container run edits in place: no worktree or checkpoint tools, none noted."""
    client = docker.from_env()
    before = image_container_ids(client)
    try:
        request = _run_and_capture(SorcarAgent("docker-note"), tmp_path, docker_image=IMAGE)
    finally:
        remove_new_image_containers(client, before)
    noted, sent = _noted_and_sent(request)
    assert not {"checkpoint", "rollback", "create_worktree", "commit_and_merge"} & sent
    assert noted == sent
