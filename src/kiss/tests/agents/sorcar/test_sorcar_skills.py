# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for sorcar Agent Skills (SKILL.md directories).

These exercise the real behaviour end to end: skill directories are
written to real user (``KISS_HOME``/``CLAUDE_CONFIG_DIR``/``HOME``) and
project directories, discovery and permission filtering run against the
real filesystem and the real ``~/.kiss/config.json``, the ``skill``
tool is built and invoked for real, and the REPL is driven through a
real subprocess reading piped stdin.  No model calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiss.agents.sorcar.skills import (
    Skill,
    bundled_skills_dir,
    discover_skills,
    format_skill_listing,
    load_skill_content,
    load_skill_permissions,
    make_skill_tool,
    skill_permission,
)


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


# ---------------------------------------------------------------------------
# Discovery


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
        # Remove claude_skills only if this test created it and it is empty.
        try:
            bundled.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Parsing


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


# ---------------------------------------------------------------------------
# Permissions


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
    # Reversed order: the catch-all deny at the end wins.
    rules2 = {"internal-ok": "allow", "internal-*": "deny"}
    assert skill_permission("internal-ok", rules2) == "deny"


def test_load_skill_permissions_from_config(isolated_homes: Path) -> None:
    """``skill_permissions`` is read from ``$KISS_HOME/config.json``."""
    # NOTE: vscode_config.CONFIG_PATH is bound at import time to the
    # session-level KISS_HOME set by conftest.py, so write there.
    from kiss.server.vscode_config import CONFIG_PATH

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    original = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
    try:
        CONFIG_PATH.write_text(
            json.dumps({"skill_permissions": {"internal-*": "DENY "}}),
            encoding="utf-8",
        )
        assert load_skill_permissions() == {"internal-*": "deny"}
    finally:
        if original is None:
            CONFIG_PATH.unlink()
        else:
            CONFIG_PATH.write_text(original)


def test_denied_skills_hidden_from_discovery(isolated_homes: Path) -> None:
    """Denied skills are excluded from the catalog entirely."""
    from kiss.server.vscode_config import CONFIG_PATH

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


# ---------------------------------------------------------------------------
# Activation (load_skill_content) and the skill tool


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
    skill = discover_skills(str(project))["pdf"]
    content = load_skill_content(skill)
    assert "<skill_resources>" in content
    assert "<file>scripts/extract.py</file>" in content
    assert "<file>references/forms.md</file>" in content
    assert "<file>SKILL.md</file>" not in content
    # Resource contents are never inlined.
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
    # Token efficiency: the body is NOT in the catalog.
    assert "SECRET-BODY-MARKER" not in doc
    assert "Args:" in doc  # schema builder needs the Args section


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


# ---------------------------------------------------------------------------
# Listing (used by the /skills REPL command)


def test_format_skill_listing_empty_hint(isolated_homes: Path) -> None:
    """The empty listing explains where to create skills."""
    text = format_skill_listing({})
    assert "No skills found" in text
    assert ".claude/skills" in text
    assert ".agents/skills" in text


def test_format_skill_listing_alignment(isolated_homes: Path) -> None:
    """The listing shows name, source, and description."""
    project = isolated_homes / "project"
    _write_skill(project / ".kiss" / "skills", "deploy", "Deploy the app")
    _write_skill(
        isolated_homes / ".claudehome" / "skills", "review", "Review code"
    )
    text = format_skill_listing(discover_skills(str(project)))
    assert "deploy" in text
    assert "(project)" in text
    assert "(claude-user)" in text
    assert "Deploy the app" in text


# ---------------------------------------------------------------------------
# Agent integration: the tool is in the agent's toolset


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



