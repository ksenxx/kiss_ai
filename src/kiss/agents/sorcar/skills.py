# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent Skills support for the ``sorcar`` CLI and agent.

Implements the `Agent Skills <https://agentskills.io>`_ open standard
with Claude Code compatibility: a *skill* is a directory containing a
``SKILL.md`` file whose YAML frontmatter provides ``name`` and
``description``.  Skills are disclosed progressively for token
efficiency:

1. **Catalog** — only each skill's name and description are embedded in
   the ``skill`` tool's description (~50-100 tokens per skill).
2. **Instructions** — the full ``SKILL.md`` body is loaded into context
   only when the agent activates the skill by calling the tool.
3. **Resources** — bundled files (``scripts/``, ``references/``,
   ``assets/``) are listed (not read) in the activation result so the
   agent can load them on demand with its file tools.

Discovery paths (low → high precedence; later wins on a name clash):

* **Bundled** — plugins shipped with Sorcar at
  ``kiss/agents/claude_skills/<plugin>/skills/<name>/SKILL.md``,
  namespaced as ``<plugin>:<name>`` (mirrors Claude Code plugin skills).
* ``~/.claude/skills/<name>/SKILL.md`` (respecting
  ``CLAUDE_CONFIG_DIR``) — Claude Code user skills work unchanged.
* ``~/.agents/skills/<name>/SKILL.md`` — the cross-client convention.
* ``~/.kiss/skills/<name>/SKILL.md`` (respecting ``KISS_HOME``) — native
  user skills.
* ``<work_dir>/.claude/skills/``, ``<work_dir>/.agents/skills/``, and
  ``<work_dir>/.kiss/skills/`` — project skills (override user skills).

Pattern-based permissions control which skills the agent may load.  The
``skill_permissions`` key in ``~/.kiss/config.json`` maps wildcard
patterns to ``"allow"`` or ``"deny"`` (e.g. ``{"*": "allow",
"internal-*": "deny"}``).  Rules are evaluated in order with the *last*
matching rule winning (OpenCode semantics); denied skills are hidden
from the catalog entirely.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import yaml

from kiss.agents.sorcar.persistence import _default_kiss_dir

logger = logging.getLogger(__name__)

# ``---\n<yaml>\n---`` frontmatter block at the very start of the file.
# Shared with custom_commands.py via :func:`parse_frontmatter`.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def claude_config_dir() -> Path:
    """Return Claude Code's config directory (``~/.claude``).

    Honours the ``CLAUDE_CONFIG_DIR`` environment variable, the same
    override Claude Code itself uses for its ``~/.claude`` directory.
    Shared by the skill and custom-command discovery paths.

    Returns:
        The resolved config directory path.
    """
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(config_dir) if config_dir else Path.home() / ".claude"


def parse_frontmatter(path: Path) -> tuple[dict[str, Any], str] | None:
    """Read *path* and split its YAML frontmatter from the Markdown body.

    Shared parsing pipeline for skill (``SKILL.md``) and custom-command
    (``*.md``) definition files, so the two discovery paths can never
    drift.

    Args:
        path: The Markdown file to read and split.

    Returns:
        A ``(meta, body)`` pair — *meta* is the frontmatter mapping
        (empty when the block is absent, malformed YAML, or not a
        mapping) and *body* is the text after the frontmatter block
        (the whole text when no block is present) — or ``None`` when
        the file is unreadable.
    """
    try:
        # utf-8-sig drops a leading BOM (common in Windows-authored
        # files) that would otherwise defeat the \A frontmatter match.
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        logger.debug("unreadable definition file: %s", path, exc_info=True)
        return None
    meta: dict[str, Any] = {}
    match = _FRONTMATTER_RE.match(text)
    body = text
    if match:
        body = text[match.end():]
        try:
            loaded = yaml.safe_load(match.group(1))
            if isinstance(loaded, dict):
                meta = loaded
        except yaml.YAMLError:
            logger.debug("bad frontmatter in %s", path, exc_info=True)
    return meta, body


def collapse_whitespace(value: object) -> str:
    """Collapse all whitespace in *value* into single spaces.

    A YAML block scalar may span lines; the one-line listings (skills
    catalog, ``/commands``, ``/help``) require single-line strings.

    Args:
        value: Any value; ``None``/falsy becomes ``""``.

    Returns:
        The whitespace-collapsed single-line string.
    """
    return " ".join(str(value or "").split())

# Maximum number of bundled resource files listed on activation.
_MAX_RESOURCE_LISTING = 50

# Directory names never listed as skill resources.
_RESOURCE_SKIP_DIRS = {".git", "node_modules", "__pycache__"}


@dataclass(frozen=True)
class Skill:
    """A skill discovered from a ``SKILL.md`` file.

    Attributes:
        name: Unique skill name used to activate it (bundled plugin
            skills are namespaced ``<plugin>:<name>``).
        description: One-line summary from the frontmatter, shown in
            the catalog so the agent can decide when to load the skill.
        path: Absolute path of the defining ``SKILL.md`` file.
        source: Where the skill was found — ``"bundled"``,
            ``"claude-user"``, ``"agents-user"``, ``"user"``,
            ``"claude-project"``, ``"agents-project"``, or
            ``"project"``.
    """

    name: str
    description: str
    path: str
    source: str

    @property
    def directory(self) -> Path:
        """Return the skill's directory (the parent of ``SKILL.md``)."""
        return Path(self.path).parent


def bundled_skills_dir() -> Path:
    """Return the directory of skills bundled with Sorcar.

    These are the official Claude Code plugins downloaded into
    ``kiss/agents/claude_skills`` by ``scripts/fetch-claude-skills.sh``
    (each plugin holds skills at ``<plugin>/skills/<name>/SKILL.md``).
    The directory may be absent in development checkouts.
    """
    return Path(__file__).resolve().parent.parent / "claude_skills"


def claude_user_skills_dir() -> Path:
    """Return Claude Code's user skills directory (``~/.claude/skills``).

    Honours the ``CLAUDE_CONFIG_DIR`` environment variable via
    :func:`claude_config_dir`.
    """
    return claude_config_dir() / "skills"


def agents_user_skills_dir() -> Path:
    """Return the cross-client user skills directory (``~/.agents/skills``)."""
    return Path.home() / ".agents" / "skills"


def user_skills_dir() -> Path:
    """Return the native user skills directory (``~/.kiss/skills``)."""
    return _default_kiss_dir() / "skills"


def _parse_skill_file(path: Path, name: str, source: str) -> Skill | None:
    """Parse one ``SKILL.md`` file into a :class:`Skill`.

    Validation is deliberately lenient (per the Agent Skills
    implementor guide): a frontmatter ``name`` that disagrees with the
    directory name is tolerated (the directory-derived *name* wins for
    lookup stability), but a skill with no usable description is
    skipped because the description is essential for disclosure.

    Args:
        path: The ``SKILL.md`` file to parse.
        name: The skill name derived from the directory layout.
        source: Discovery source label (e.g. ``"user"``).

    Returns:
        The parsed skill, or ``None`` when the file is unreadable or
        has no description.
    """
    parsed = parse_frontmatter(path)
    if parsed is None:
        return None
    meta, body = parsed
    # Collapse all whitespace (a YAML block scalar may span lines) so
    # the description is the one-line summary the catalog and the
    # /skills listing require.
    description = collapse_whitespace(meta.get("description", ""))
    if not description:
        # Claude Code falls back to the first paragraph of the body.
        first_paragraph = body.strip().split("\n\n", 1)[0]
        description = collapse_whitespace(first_paragraph)
    if not description:
        logger.debug("skill %s has no description; skipping", path)
        return None
    return Skill(
        name=name,
        description=description,
        path=str(path),
        source=source,
    )


def _load_skills_dir(root: Path, source: str) -> dict[str, Skill]:
    """Load every ``<name>/SKILL.md`` directly under *root*."""
    skills: dict[str, Skill] = {}
    if not root.is_dir():
        return skills
    for skill_md in sorted(root.glob("*/SKILL.md")):
        skill = _parse_skill_file(skill_md, skill_md.parent.name, source)
        if skill is not None:
            skills[skill.name] = skill
    return skills


def _load_bundled_skills(root: Path) -> dict[str, Skill]:
    """Load plugin-shipped skills under *root* (``<plugin>/skills/<name>/``).

    Skill names are namespaced ``<plugin>:<name>`` (Claude Code's plugin
    convention) so bundled skills can never clash with user or project
    skills.
    """
    skills: dict[str, Skill] = {}
    if not root.is_dir():
        return skills
    for skill_md in sorted(root.glob("*/skills/*/SKILL.md")):
        plugin = skill_md.parent.parent.parent.name
        name = f"{plugin}:{skill_md.parent.name}"
        skill = _parse_skill_file(skill_md, name, "bundled")
        if skill is not None:
            skills[skill.name] = skill
    return skills


def load_permission_rules(key: str) -> dict[str, str]:
    """Load one wildcard permission mapping from ``~/.kiss/config.json``.

    Shared loader for ``skill_permissions`` and ``mcp_permissions``,
    which use identical shapes and normalization.

    Args:
        key: The top-level config key holding the rules.

    Returns:
        Mapping of wildcard pattern → ``"allow"``/``"deny"``, in file
        order.  Empty when the config or key is missing/malformed.
    """
    try:
        from kiss.core.vscode_config import load_config

        raw = load_config().get(key)
    except Exception:
        logger.debug("could not load %s", key, exc_info=True)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v).strip().lower() for k, v in raw.items()}


def load_skill_permissions() -> dict[str, str]:
    """Load the ``skill_permissions`` rules from ``~/.kiss/config.json``.

    Returns:
        Mapping of wildcard pattern → ``"allow"``/``"deny"``, in file
        order.  Empty when the config or key is missing/malformed.
    """
    return load_permission_rules("skill_permissions")


def skill_permission(name: str, rules: dict[str, str]) -> str:
    """Resolve the permission for skill *name* against *rules*.

    Rules are evaluated in insertion order; the **last** matching
    pattern wins (OpenCode semantics).  Patterns use shell-style
    wildcards (``*`` matches any run of characters, ``?`` one
    character).  When no rule matches, the skill is allowed.

    Args:
        name: The skill name (e.g. ``"internal-docs"``).
        rules: Mapping of pattern → ``"allow"``/``"deny"``.

    Returns:
        ``"allow"`` or ``"deny"``.
    """
    decision = "allow"
    for pattern, action in rules.items():
        if fnmatch.fnmatchcase(name, pattern):
            decision = "deny" if action == "deny" else "allow"
    return decision


def discover_skills(work_dir: str) -> dict[str, Skill]:
    """Discover all skills visible from *work_dir*, honouring permissions.

    Claude Code skill directories (``~/.claude/skills`` and
    ``<work_dir>/.claude/skills``) and the cross-client
    ``.agents/skills`` convention are scanned alongside the native
    ``.kiss`` directories and the skills bundled with Sorcar.  Later
    directories override earlier ones on a name clash; the load order
    (low → high precedence) is: bundled, claude-user, agents-user,
    user, claude-project, agents-project, project — so project skills
    win over user skills, and a native ``.kiss`` skill wins over a
    Claude Code skill at the same level.

    Skills denied by the ``skill_permissions`` config rules are
    excluded entirely (hidden from the catalog rather than blocked at
    activation time).

    Args:
        work_dir: The project directory whose skills to include.

    Returns:
        Mapping of skill name → :class:`Skill`.
    """
    wd = Path(work_dir)
    skills = _load_bundled_skills(bundled_skills_dir())
    skills.update(_load_skills_dir(claude_user_skills_dir(), "claude-user"))
    skills.update(_load_skills_dir(agents_user_skills_dir(), "agents-user"))
    skills.update(_load_skills_dir(user_skills_dir(), "user"))
    skills.update(
        _load_skills_dir(wd / ".claude" / "skills", "claude-project")
    )
    skills.update(
        _load_skills_dir(wd / ".agents" / "skills", "agents-project")
    )
    skills.update(_load_skills_dir(wd / ".kiss" / "skills", "project"))
    rules = load_skill_permissions()
    if rules:
        skills = {
            name: skill
            for name, skill in skills.items()
            if skill_permission(name, rules) == "allow"
        }
    return skills


def _catalog_xml(skills: dict[str, Skill]) -> str:
    """Render the token-efficient ``<available_skills>`` catalog."""
    lines = ["<available_skills>"]
    for skill in sorted(skills.values(), key=lambda s: s.name):
        lines.append("<skill>")
        lines.append(f"<name>{xml_escape(skill.name)}</name>")
        lines.append(f"<description>{xml_escape(skill.description)}</description>")
        lines.append("</skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _list_resources(skill_dir: Path) -> tuple[list[str], bool]:
    """List the skill's bundled resource files (relative paths, capped).

    Files are listed — never read — so the agent can load specific
    resources on demand with its file tools.  ``SKILL.md`` itself and
    bookkeeping directories are excluded.

    Returns:
        ``(resources, truncated)`` where *truncated* is ``True`` only
        when at least one further resource file exists beyond the
        ``_MAX_RESOURCE_LISTING`` cap.
    """
    resources: list[str] = []
    truncated = False
    try:
        for path in sorted(skill_dir.rglob("*")):
            if not path.is_file() or path.name == "SKILL.md":
                continue
            rel = path.relative_to(skill_dir)
            if any(part in _RESOURCE_SKIP_DIRS for part in rel.parts):
                continue
            if len(resources) >= _MAX_RESOURCE_LISTING:
                truncated = True
                break
            resources.append(str(rel))
    except OSError:
        logger.debug("could not list resources in %s", skill_dir, exc_info=True)
    return resources, truncated


def load_skill_content(skill: Skill) -> str:
    """Load *skill*'s full instructions for injection into the context.

    Reads ``SKILL.md`` at activation time (so edits between activations
    are picked up), strips the YAML frontmatter, and wraps the body in
    ``<skill_content>`` tags together with the skill directory and a
    capped listing of bundled resource files.

    Args:
        skill: The skill to load.

    Returns:
        The structured skill content string, or an error message when
        the file has become unreadable.
    """
    try:
        text = Path(skill.path).read_text(encoding="utf-8-sig")
    except OSError:
        return f"Error: skill file is no longer readable: {skill.path}"
    match = _FRONTMATTER_RE.match(text)
    body = text[match.end():].strip() if match else text.strip()
    skill_dir = skill.directory
    # Escape the name exactly like the catalog does (attribute context
    # additionally needs the double quote escaped).
    escaped_name = xml_escape(skill.name, {'"': "&quot;"})
    parts = [f'<skill_content name="{escaped_name}">', body, ""]
    parts.append(f"Skill directory: {skill_dir}")
    parts.append(
        "Relative paths in this skill are relative to the skill directory."
    )
    resources, truncated = _list_resources(skill_dir)
    if resources:
        parts.append("<skill_resources>")
        parts.extend(f"<file>{r}</file>" for r in resources)
        if truncated:
            parts.append("<note>listing truncated</note>")
        parts.append("</skill_resources>")
    parts.append("</skill_content>")
    return "\n".join(parts)


def make_skill_tool(work_dir: str) -> Callable[[str], str] | None:
    """Build the ``skill`` tool for the agent running in *work_dir*.

    The available skills are embedded — name and description only — in
    the tool's docstring, so the agent can choose a relevant skill
    without paying the token cost of every skill's full instructions
    (progressive disclosure).  Calling the tool loads the named skill's
    complete ``SKILL.md`` body.

    Args:
        work_dir: The project directory whose skills to expose.

    Returns:
        The tool callable, or ``None`` when no user or project skills
        are configured (per the Agent Skills implementor guide, no tool
        should be registered when there is nothing to activate).
        Bundled plugin skills alone do not cause the tool to be
        registered: they are infrastructural defaults that the user
        opts into by creating a skill in their user or project
        directories.
    """
    skills = discover_skills(work_dir)
    if not any(s.source != "bundled" for s in skills.values()):
        return None

    def skill(name: str) -> str:
        # The real docstring is assigned dynamically below so it can
        # embed the discovered skill catalog.
        current = discover_skills(work_dir)
        found = current.get(name)
        if found is None:
            available = ", ".join(sorted(current)) or "(none)"
            return (
                f"Error: unknown or unavailable skill {name!r}. "
                f"Available skills: {available}"
            )
        return load_skill_content(found)

    skill.__name__ = "skill"
    skill.__doc__ = (
        "Load a skill's full instructions into the conversation.\n\n"
        "Skills provide specialized instructions for specific tasks. "
        "When the current task matches a skill's description below, "
        "call this tool with the skill's name to load its complete "
        "SKILL.md instructions before proceeding. Bundled resource "
        "files listed in the result can be read or executed on demand "
        "with the file and bash tools.\n\n"
        f"{_catalog_xml(skills)}\n\n"
        "Args:\n"
        "    name: The name of the skill to load, exactly as listed in "
        "available_skills.\n\n"
        "Returns:\n"
        "    The skill's full instructions, its directory, and a "
        "listing of its bundled resource files."
    )
    return skill


def truncate_listing_description(desc: str) -> str:
    """Cap *desc* at 100 characters for aligned catalog listings.

    Shared by :func:`format_skill_listing` and
    :func:`~kiss.agents.sorcar.custom_commands.format_command_listing`
    so the two listings can never drift.

    Args:
        desc: The raw one-line description.

    Returns:
        *desc* unchanged when at most 100 characters, otherwise its
        first 97 characters followed by ``"..."``.
    """
    if len(desc) > 100:
        return desc[:97] + "..."
    return desc


def format_skill_listing(skills: dict[str, Skill]) -> str:
    """Format *skills* as the aligned listing printed by ``/skills``.

    Args:
        skills: Mapping of skill name → skill (from
            :func:`discover_skills`).

    Returns:
        A printable multi-line listing, or a hint about where to create
        skill directories when *skills* is empty.
    """
    if not skills:
        return (
            "No skills found.\n"
            f"Create <name>/SKILL.md directories in {user_skills_dir()} "
            "(user) or <project>/.kiss/skills (project) to define them.\n"
            "Claude Code skills in ~/.claude/skills and "
            "<project>/.claude/skills, and cross-client skills in "
            "~/.agents/skills and <project>/.agents/skills, are picked "
            "up too."
        )
    entries = sorted(skills.values(), key=lambda s: s.name)
    width = max(len(s.name) for s in entries)
    lines = []
    for skill in entries:
        desc = truncate_listing_description(skill.description)
        lines.append(f"  {skill.name:<{width}}  ({skill.source}) {desc}")
    return "\n".join(lines)
