# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-19 (F4, S2-3): ``SorcarAgent.run`` resolves ``work_dir`` once.

With ``work_dir=None`` the run executes in the current directory
(``_reset`` passes ``work_dir or "."``), but the restricted-profile
note's skill probe and the repository memory domain used
``resolve_work_dir(None)``, which is ``artifact_dir/kiss_workdir``: a
``.kiss/skills/`` folder in the current directory gave the run a
``skill`` tool the note said did not exist.  ``run`` now resolves the
directory once, before anything looks at it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve

_SKILL = """---
name: demo
description: a demo skill
---
Do the demo.
"""


def test_default_work_dir_skill_is_offered_and_promised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir = tmp_path / ".kiss" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL)
    # The developer's own ~/.claude/skills and ~/.agents/skills must not
    # count as "a skill exists": an empty home stands in for theirs.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # what Path.home() reads on Windows
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with serve([finish_body("<p>ok</p>", prompt_tokens=500)]) as (url, requests):
        agent = SorcarAgent("f4-work-dir")
        result = agent.run(
            prompt_template="Say ok.",
            model_name=MODEL,
            max_steps=2,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            use_memory=False,
            verbose=False,
            tool_profile="skills",
        )
    assert yaml.safe_load(result)["success"] is True
    assert agent.work_dir == str(tmp_path.resolve())
    request = requests[0]
    names = {t["function"]["name"] for t in request["tools"]}
    assert "skill" in names
    system = str(next(m for m in request["messages"] if m["role"] == "system")["content"])
    note = system.split("# Restricted tool profile: skills", 1)[1]
    assert "skill" in note.split("plus finish: ", 1)[1].split("\n", 1)[0]
