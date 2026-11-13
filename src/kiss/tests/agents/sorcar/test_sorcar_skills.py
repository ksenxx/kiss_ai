# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for sorcar Agent Skills (SKILL.md directories).

These exercise the real behaviour end to end: skill directories are
written to real user (``KISS_HOME``/``CLAUDE_CONFIG_DIR``/``HOME``) and
project directories, discovery and permission filtering run against the
real filesystem and the real ``~/.kiss/config.json``, and the ``skill``
tool is built and invoked for real.  No model calls are made.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import skills, useful_tools
from kiss.agents.sorcar.skills import (
    Skill,
    bundled_skills_dir,
    discover_skills,
    load_permission_rules,
    load_skill_content,
    make_skill_tool,
    parse_frontmatter,
    skill_permission,
)
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve
from kiss.tests.conftest import posix_only


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every user-level skills location into *tmp_path*.

    Points ``KISS_HOME``, ``CLAUDE_CONFIG_DIR``, and ``HOME`` at
    isolated directories so discovery never picks up the developer's
    real ``~/.kiss/skills``, ``~/.claude/skills``, or
    ``~/.agents/skills`` during tests.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claudehome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))  # Path.home() on Windows
    (tmp_path / "project").mkdir()
    return tmp_path


def _write_skill(root: Path, name: str, description: str = "", body: str = "") -> Path:
    """Create ``<root>/<name>/SKILL.md`` with the given frontmatter/body."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = f"---\nname: {name}\n"
    if description:
        frontmatter += f"description: {description}\n"
    frontmatter += "---\n"
    (skill_dir / "SKILL.md").write_text(
        frontmatter + (body or f"# {name}\n\nInstructions for {name}."),
        encoding="utf-8",
    )
    return skill_dir



def test_discovers_project_kiss_skills(isolated_homes: Path) -> None:
    """Skills in ``<project>/.kiss/skills`` are discovered."""
    project = isolated_homes / "project"
    _write_skill(project / ".kiss" / "skills", "deploy", "Deploy the app")
    skills = discover_skills(str(project))
    assert "deploy" in skills
    assert skills["deploy"].description == "Deploy the app"
    assert skills["deploy"].source == "project"


def test_discovers_claude_project_skills(isolated_homes: Path) -> None:
    """Skills in ``<project>/.claude/skills`` work unchanged."""
    project = isolated_homes / "project"
    _write_skill(project / ".claude" / "skills", "review", "Review code")
    skills = discover_skills(str(project))
    assert skills["review"].source == "claude-project"


def test_discovers_agents_project_skills(isolated_homes: Path) -> None:
    """Skills in the cross-client ``<project>/.agents/skills`` are found."""
    project = isolated_homes / "project"
    _write_skill(project / ".agents" / "skills", "roll-dice", "Roll dice")
    skills = discover_skills(str(project))
    assert skills["roll-dice"].source == "agents-project"


def test_discovers_user_skills_via_kiss_home(isolated_homes: Path) -> None:
    """Skills in ``$KISS_HOME/skills`` are discovered as user skills."""
    _write_skill(
        isolated_homes / ".kisshome" / "skills", "git-release", "Cut releases"
    )
    skills = discover_skills(str(isolated_homes / "project"))
    assert skills["git-release"].source == "user"


def test_discovers_claude_user_skills(isolated_homes: Path) -> None:
    """Skills in ``$CLAUDE_CONFIG_DIR/skills`` (~/.claude/skills) are found."""
    _write_skill(
        isolated_homes / ".claudehome" / "skills", "summarize", "Summarize diffs"
    )
    skills = discover_skills(str(isolated_homes / "project"))
    assert skills["summarize"].source == "claude-user"


def test_discovers_agents_user_skills(isolated_homes: Path) -> None:
    """Skills in ``~/.agents/skills`` are found (HOME redirected)."""
    _write_skill(
        isolated_homes / "home" / ".agents" / "skills", "lint", "Run linters"
    )
    skills = discover_skills(str(isolated_homes / "project"))
    assert skills["lint"].source == "agents-user"


def test_project_overrides_user_skill(isolated_homes: Path) -> None:
    """A project skill shadows a user skill with the same name."""
    project = isolated_homes / "project"
    _write_skill(isolated_homes / ".kisshome" / "skills", "deploy", "User deploy")
    _write_skill(project / ".kiss" / "skills", "deploy", "Project deploy")
    skills = discover_skills(str(project))
    assert skills["deploy"].description == "Project deploy"
    assert skills["deploy"].source == "project"


def test_home_work_dir_does_not_load_the_stock_brand_global_skills(
    isolated_homes: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task working in ``~`` never reads ``~/.kiss/skills`` as project skills.

    ``~/.kiss`` is the stock brand's global home, stale once a brand
    with a home of its own (``~/.s10s``) is installed; it is not a
    project directory, so its skills must not shadow the live user
    skills of ``kiss_home()`` (audit 2026-11-07, A13_fable-4).
    """
    home = isolated_homes / "home"
    live = home / ".s10s"
    monkeypatch.setenv("KISS_HOME", str(live))
    _write_skill(live / "skills", "example", "Live", "LIVE BRAND INSTRUCTIONS")
    _write_skill(home / ".kiss" / "skills", "example", "Stale", "STALE STOCK INSTRUCTIONS")
    found = discover_skills(str(home))["example"]
    assert found.source == "user", found
    assert Path(found.path) == live / "skills" / "example" / "SKILL.md"
    assert "LIVE BRAND INSTRUCTIONS" in load_skill_content(found)


def test_kiss_overrides_claude_at_same_level(isolated_homes: Path) -> None:
    """A native ``.kiss`` skill wins over a ``.claude`` skill at the same level."""
    project = isolated_homes / "project"
    _write_skill(project / ".claude" / "skills", "fmt", "Claude fmt")
    _write_skill(project / ".kiss" / "skills", "fmt", "Kiss fmt")
    skills = discover_skills(str(project))
    assert skills["fmt"].description == "Kiss fmt"


def test_bundled_plugin_skills_namespaced(isolated_homes: Path) -> None:
    """Bundled ``<plugin>/skills/<name>/SKILL.md`` are namespaced plugin:name."""
    bundled = bundled_skills_dir()
    plugin_skill = bundled / "test-plugin-xyz" / "skills" / "demo-skill"
    plugin_skill.mkdir(parents=True)
    try:
        (plugin_skill / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: Bundled demo\n---\nBody.",
            encoding="utf-8",
        )
        skills = discover_skills(str(isolated_homes / "project"))
        assert "test-plugin-xyz:demo-skill" in skills
        assert skills["test-plugin-xyz:demo-skill"].source == "bundled"
    finally:
        import shutil

        shutil.rmtree(bundled / "test-plugin-xyz")
        try:
            bundled.rmdir()
        except OSError:
            pass



def test_description_falls_back_to_first_paragraph(isolated_homes: Path) -> None:
    """Without a frontmatter description, the first body paragraph is used."""
    project = isolated_homes / "project"
    root = project / ".kiss" / "skills" / "nodesc"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: nodesc\n---\nUse this when testing\nfallbacks.\n\nMore.",
        encoding="utf-8",
    )
    skills = discover_skills(str(project))
    assert skills["nodesc"].description == "Use this when testing fallbacks."


def test_skill_without_any_description_skipped(isolated_homes: Path) -> None:
    """A skill with no description and an empty body is skipped entirely."""
    project = isolated_homes / "project"
    root = project / ".kiss" / "skills" / "empty"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text("---\nname: empty\n---\n", encoding="utf-8")
    assert "empty" not in discover_skills(str(project))


def test_bad_yaml_frontmatter_tolerated(isolated_homes: Path) -> None:
    """Unparseable YAML frontmatter falls back to the body description."""
    project = isolated_homes / "project"
    root = project / ".kiss" / "skills" / "badyaml"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\ndescription: broken: yaml: here\n---\nBody description here.",
        encoding="utf-8",
    )
    skills = discover_skills(str(project))
    assert skills["badyaml"].description == "Body description here."


def test_name_derived_from_directory(isolated_homes: Path) -> None:
    """The directory name is the lookup key even if frontmatter disagrees."""
    project = isolated_homes / "project"
    root = project / ".kiss" / "skills" / "dirname"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: other-name\ndescription: D\n---\nBody.", encoding="utf-8"
    )
    assert "dirname" in discover_skills(str(project))


@posix_only("os.mkfifo: a FIFO is the non-regular file whose open blocks")
def test_fifo_skill_file_skipped_without_blocking(isolated_homes: Path) -> None:
    """A FIFO named ``SKILL.md`` is skipped at once, by discovery and by activation.

    Opening a FIFO with no writer blocks for ever, with no deadline and
    no bytecode boundary where the runner's Stop could land, and every
    agent's tool setup runs discovery.  The regular skill beside it must
    still be found.
    """
    project = isolated_homes / "project"
    _write_skill(project / ".agents" / "skills", "good", description="Good one.")
    fifo = project / ".agents" / "skills" / "blocked" / "SKILL.md"
    fifo.parent.mkdir(parents=True)
    os.mkfifo(fifo)
    outcome: list[object] = []

    def run() -> None:
        outcome.append(set(discover_skills(str(project))))
        outcome.append(load_skill_content(
            Skill(name="blocked", description="d", path=str(fifo), source="project"),
        ))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=5)
    try:
        assert not worker.is_alive(), f"blocked opening the FIFO after {outcome}"
    finally:
        deadline = time.monotonic() + 5  # a writer releases each blocked open
        while worker.is_alive() and time.monotonic() < deadline:
            with contextlib.suppress(OSError):
                os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            worker.join(timeout=0.05)
    assert outcome[0] == {"good"}
    assert "no longer readable" in str(outcome[1])


def _stop_at(
    seen: list[tuple[CodeType, int]], point: tuple[CodeType, int] | None,
    frame: Any, event: str, arg: object,
) -> Any:
    """Trace function: note in *seen* the instructions of the tool modules, or stop at *point*.

    Python delivers a Stop (an exception injected into the task thread)
    at a check point of the thread's bytecode; raising from the opcode
    hook right before an instruction is the same, at a chosen one.
    """
    if frame.f_code.co_filename not in (skills.__file__, useful_tools.__file__):
        return None
    frame.f_trace_opcodes = True
    here = (frame.f_code, frame.f_lasti)
    if event == "opcode" and point is None:
        seen.append(here)
    elif event == "opcode" and here == point and not seen:
        seen.append(here)
        raise KeyboardInterrupt
    return functools.partial(_stop_at, seen, point)


def _descriptors_on(path: Path) -> list[int]:
    """Return this process's open descriptors of the file *path*."""
    file = path.stat()
    found = []
    for name in os.listdir("/dev/fd"):
        with contextlib.suppress(OSError):  # the listing's own descriptor is closed by now
            if os.path.samestat(os.fstat(int(name)), file):
                found.append(int(name))
    return found


@posix_only("/dev/fd lists the open descriptors")
def test_stop_during_a_skill_read_leaves_no_descriptor_open(isolated_homes: Path) -> None:
    """Audit 2026-11-12 B5/A8: a Stop landing between the ``os.open`` of ``SKILL.md`` and the
    ``with`` that closes it left the descriptor open for the life of the daemon.  The read is
    stopped at every instruction the calling thread runs in the tool modules; none may leak."""
    path = _write_skill(isolated_homes / "project" / ".agents" / "skills", "one", "d") / "SKILL.md"
    points: list[tuple[CodeType, int]] = []
    sys.settrace(functools.partial(_stop_at, points, None))
    try:
        assert parse_frontmatter(path) is not None
    finally:
        sys.settrace(None)
    assert points
    for point in dict.fromkeys(points):
        stopped: list[tuple[CodeType, int]] = []
        sys.settrace(functools.partial(_stop_at, stopped, point))
        try:
            with contextlib.suppress(KeyboardInterrupt):
                parse_frontmatter(path)
        finally:
            sys.settrace(None)
        assert stopped == [point]
        deadline = time.monotonic() + 5  # a read still closing on its own thread is no leak
        while (leaked := _descriptors_on(path)) and time.monotonic() < deadline:
            time.sleep(0.01)
        for fd in leaked:
            os.close(fd)
        assert not leaked, f"descriptor left open by a Stop at {point[0].co_name}+{point[1]}"



def test_skill_permission_default_allow() -> None:
    """With no matching rule the skill is allowed."""
    assert skill_permission("anything", {}) == "allow"
    assert skill_permission("anything", {"other": "deny"}) == "allow"


def test_skill_permission_deny_pattern() -> None:
    """``internal-*: deny`` hides matching skills."""
    rules = {"*": "allow", "internal-*": "deny"}
    assert skill_permission("internal-docs", rules) == "deny"
    assert skill_permission("internal-tools", rules) == "deny"
    assert skill_permission("public-docs", rules) == "allow"


def test_skill_permission_last_rule_wins() -> None:
    """The last matching rule wins (OpenCode semantics)."""
    rules = {"internal-*": "deny", "internal-ok": "allow"}
    assert skill_permission("internal-ok", rules) == "allow"
    assert skill_permission("internal-bad", rules) == "deny"
    rules2 = {"internal-ok": "allow", "internal-*": "deny"}
    assert skill_permission("internal-ok", rules2) == "deny"


def test_load_skill_permissions_from_config(isolated_homes: Path) -> None:
    """``skill_permissions`` is read from ``$KISS_HOME/config.json``."""
    from kiss.core.user_config import CONFIG_PATH

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    original = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
    try:
        CONFIG_PATH.write_text(
            json.dumps({"skill_permissions": {"internal-*": "DENY "}}),
            encoding="utf-8",
        )
        assert load_permission_rules("skill_permissions") == {"internal-*": "deny"}
    finally:
        if original is None:
            CONFIG_PATH.unlink()
        else:
            CONFIG_PATH.write_text(original)


def test_denied_skills_hidden_from_discovery(isolated_homes: Path) -> None:
    """Denied skills are excluded from the catalog entirely."""
    from kiss.core.user_config import CONFIG_PATH

    project = isolated_homes / "project"
    _write_skill(project / ".kiss" / "skills", "internal-secret", "Hidden")
    _write_skill(project / ".kiss" / "skills", "public-tool", "Visible")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    original = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
    try:
        CONFIG_PATH.write_text(
            json.dumps(
                {"skill_permissions": {"*": "allow", "internal-*": "deny"}}
            ),
            encoding="utf-8",
        )
        skills = discover_skills(str(project))
        assert "internal-secret" not in skills
        assert "public-tool" in skills
    finally:
        if original is None:
            CONFIG_PATH.unlink()
        else:
            CONFIG_PATH.write_text(original)



def test_load_skill_content_strips_frontmatter(isolated_homes: Path) -> None:
    """Activation returns the body without frontmatter, in skill_content tags."""
    project = isolated_homes / "project"
    _write_skill(
        project / ".kiss" / "skills", "deploy", "Deploy", "Run the deploy steps."
    )
    skill = discover_skills(str(project))["deploy"]
    content = load_skill_content(skill)
    assert content.startswith('<skill_content name="deploy">')
    assert "Run the deploy steps." in content
    assert "---" not in content.split("\n", 1)[1].split("Skill directory")[0]
    assert f"Skill directory: {skill.directory}" in content
    assert content.rstrip().endswith("</skill_content>")


def test_load_skill_content_lists_resources(isolated_homes: Path) -> None:
    """Bundled resource files are listed (not read) on activation."""
    project = isolated_homes / "project"
    skill_dir = _write_skill(project / ".kiss" / "skills", "pdf", "PDFs")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "extract.py").write_text("print('x')")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "forms.md").write_text("forms")
    (skill_dir / "a&b.md").write_text("markup characters in a name")
    # Leave less recursion headroom than the real tree's depth.  This exercises the
    # iterative walk without exceeding macOS PATH_MAX or Windows MAX_PATH.
    tree_depth = 50
    deep = skill_dir
    for _ in range(tree_depth):
        deep = deep / "d"
        deep.mkdir()
    (deep / "bottom.md").write_text("bottom")
    locked = skill_dir / "locked"
    locked.mkdir()
    (locked / "hidden.md").write_text("hidden")
    skill = discover_skills(str(project))["pdf"]
    locked.chmod(0)  # cannot be listed: contributes nothing, the rest still lists
    old_recursion_limit = sys.getrecursionlimit()
    frame: FrameType | None = sys._getframe()
    stack_depth = 0
    while frame is not None:
        stack_depth += 1
        frame = frame.f_back
    try:
        sys.setrecursionlimit(stack_depth + tree_depth - 10)
        content = load_skill_content(skill)
    finally:
        sys.setrecursionlimit(old_recursion_limit)
        locked.chmod(0o755)
    can_list_locked = os.name == "nt" or os.geteuid() == 0
    assert ("<file>locked/hidden.md</file>" in content) == can_list_locked
    assert "<skill_resources>" in content
    deep_resource = f"<file>{'d/' * tree_depth}bottom.md</file>"
    references_resource = "<file>references/forms.md</file>"
    assert deep_resource in content
    assert content.index(deep_resource) < content.index(references_resource)
    assert "<file>scripts/extract.py</file>" in content
    assert references_resource in content
    assert "<file>SKILL.md</file>" not in content
    assert "<file>a&amp;b.md</file>" in content  # escaped as the catalog's fields are
    assert "print('x')" not in content


def test_load_skill_content_unreadable_file() -> None:
    """A vanished SKILL.md produces an error string, not an exception."""
    skill = Skill(
        name="gone", description="d", path="/nonexistent/gone/SKILL.md",
        source="user",
    )
    assert "no longer readable" in load_skill_content(skill)


def test_make_skill_tool_none_without_skills(isolated_homes: Path) -> None:
    """No tool is registered when no skills are discovered."""
    assert make_skill_tool(str(isolated_homes / "project")) is None


def test_skill_tool_docstring_is_catalog(isolated_homes: Path) -> None:
    """The tool docstring carries only names + descriptions (no bodies)."""
    project = isolated_homes / "project"
    _write_skill(
        project / ".kiss" / "skills", "git-release", "Cut releases",
        "SECRET-BODY-MARKER full instructions",
    )
    tool = make_skill_tool(str(project))
    assert tool is not None
    assert tool.__name__ == "skill"
    doc = tool.__doc__ or ""
    assert "<available_skills>" in doc
    assert "<name>git-release</name>" in doc
    assert "<description>Cut releases</description>" in doc
    assert "SECRET-BODY-MARKER" not in doc
    assert "Args:" in doc


def test_skill_tool_loads_content(isolated_homes: Path) -> None:
    """Calling the tool returns the full SKILL.md body."""
    project = isolated_homes / "project"
    _write_skill(
        project / ".kiss" / "skills", "git-release", "Cut releases",
        "Step 1: tag.\nStep 2: push.",
    )
    tool = make_skill_tool(str(project))
    assert tool is not None
    result = tool("git-release")
    assert "Step 1: tag." in result
    assert '<skill_content name="git-release">' in result


def test_skill_tool_unknown_name(isolated_homes: Path) -> None:
    """Unknown skill names return a helpful error listing valid names."""
    project = isolated_homes / "project"
    _write_skill(project / ".kiss" / "skills", "known", "K")
    tool = make_skill_tool(str(project))
    assert tool is not None
    result = tool("nope")
    assert "unknown or unavailable skill" in result
    assert "known" in result


def test_skill_tool_picks_up_new_skill_after_creation(
    isolated_homes: Path,
) -> None:
    """Skills created after the tool was built are still loadable."""
    project = isolated_homes / "project"
    _write_skill(project / ".kiss" / "skills", "first", "F")
    tool = make_skill_tool(str(project))
    assert tool is not None
    _write_skill(project / ".kiss" / "skills", "second", "S", "Second body.")
    assert "Second body." in tool("second")



def test_sorcar_agent_gets_skill_tool(isolated_homes: Path) -> None:
    """SorcarAgent._get_tools includes the skill tool when skills exist."""
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    project = isolated_homes / "project"
    _write_skill(project / ".kiss" / "skills", "deploy", "Deploy the app")
    agent = SorcarAgent("skills-test")
    agent.work_dir = str(project)
    agent._use_web_tools = False
    tools = agent._get_tools()
    names = [t.__name__ for t in tools]
    assert "skill" in names
    skill_tool = tools[names.index("skill")]
    assert "<name>deploy</name>" in (skill_tool.__doc__ or "")


def test_sorcar_agent_skill_tool_follows_the_work_dir_into_the_worktree(
    isolated_homes: Path,
) -> None:
    """The skill tool reads the agent's work dir at call time.

    ``create_worktree`` moves the task's work dir into its worktree; a
    tool bound to the directory it was built for kept loading the main
    checkout's project skills, so an edit of a project skill inside
    the worktree had no effect on the task (audit 2026-11-07,
    A13_fable-5).
    """
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent
    from kiss.tests.server.conftest import init_repo, run_git_checked

    project = isolated_homes / "project"
    init_repo(project)
    _write_skill(project / ".kiss" / "skills", "deploy", "Deploy the app", "Checkout body.")
    run_git_checked(project, "add", ".kiss")
    run_git_checked(project, "commit", "-q", "-m", "skill")
    agent = SorcarAgent("skills-worktree")
    agent.work_dir = str(project)
    agent._use_web_tools = False
    skill_tool = next(t for t in agent._get_tools() if t.__name__ == "skill")
    try:
        assert agent.create_worktree().startswith("Created worktree")
        wt_skill = Path(agent.work_dir) / ".kiss" / "skills" / "deploy" / "SKILL.md"
        assert wt_skill.is_file()
        wt_skill.write_text(wt_skill.read_text().replace("Checkout body.", "Worktree body."))
        result = skill_tool("deploy")
        assert "Worktree body." in result and "Checkout body." not in result, result
        assert f"Skill directory: {wt_skill.parent}" in result, result
    finally:
        agent.discard()


def test_sorcar_agent_no_skill_tool_without_skills(
    isolated_homes: Path,
) -> None:
    """No skill tool is registered when there are no skills."""
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    agent = SorcarAgent("skills-test")
    agent.work_dir = str(isolated_homes / "project")
    agent._use_web_tools = False
    names = [t.__name__ for t in agent._get_tools()]
    assert "skill" not in names


def _run_once(agent: Any, project: Path, **run_args: Any) -> dict[str, Any]:
    """Run *agent* in *project* against a one-answer local model server; its first request."""
    with serve([finish_body("<p>ok</p>", prompt_tokens=500)]) as (url, requests):
        result = agent.run(
            prompt_template="Say ok.",
            model_name=MODEL,
            max_steps=2,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            use_memory=False,
            web_tools=False,
            verbose=False,
            work_dir=str(project),
            **run_args,
        )
    assert yaml.safe_load(result)["success"] is True
    return requests[0]


def test_skills_profile_run_offers_the_skill_and_the_next_run_discovers_afresh(
    isolated_homes: Path,
) -> None:
    """A ``skills``-profile run offers the skill its note promises; the next run rediscovers.

    The restricted-profile note and the toolset share one discovery per
    run (A09-2), and that discovery does not outlive the run: once the
    project's only skill is gone, the agent's next run (the full
    profile, which does not probe for the note) offers no ``skill`` tool.
    """
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    project = isolated_homes / "project"
    skill_dir = _write_skill(project / ".kiss" / "skills", "deploy", "Deploy the app")
    agent = SorcarAgent("skills-run")
    request = _run_once(agent, project, tool_profile="skills")
    assert {t["function"]["name"] for t in request["tools"]} == {"skill", "finish"}
    system = str(next(m for m in request["messages"] if m["role"] == "system")["content"])
    note = system.split("# Restricted tool profile: skills", 1)[1]
    assert "skill" in note.split("plus finish: ", 1)[1].split("\n", 1)[0]

    shutil.rmtree(skill_dir)
    request = _run_once(agent, project)
    names = {t["function"]["name"] for t in request["tools"]}
    assert "Bash" in names
    assert "skill" not in names
