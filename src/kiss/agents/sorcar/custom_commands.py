# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Custom slash commands for the ``sorcar`` CLI, defined as Markdown files.

Reusable prompts become slash commands, following the conventions shared
by Claude Code, OpenCode, and Gemini CLI:

* **User commands** live in ``~/.kiss/commands/*.md`` (respecting
  ``KISS_HOME``) and are available in every project.
* **Project commands** live in ``<work_dir>/.kiss/commands/*.md`` and can
  be checked into version control; they override user commands with the
  same name.
* **Claude Code commands** are picked up too, so existing
  ``~/.claude/commands/*.md`` (respecting ``CLAUDE_CONFIG_DIR``) and
  ``<work_dir>/.claude/commands/*.md`` files work unchanged.  On a name
  conflict the native ``.kiss`` command wins at the same level, and any
  project command wins over any user command — precedence (low → high):
  claude-user → user → claude-project → project.
* The file name (without ``.md``) becomes the command name:
  ``test.md`` → ``/test``.  Subdirectories namespace the name with ``:``
  — ``git/commit.md`` → ``/git:commit``.
* An optional YAML frontmatter block may set ``description`` (shown in
  ``/help`` and ``/commands``) and ``argument-hint`` (shown next to the
  name, e.g. ``[issue-number]``).  The Markdown body is the prompt
  template.

The template supports the placeholder syntax common to those tools:

* ``$ARGUMENTS`` — everything typed after the command name.
* ``$1`` … ``$9`` — positional arguments (shell-style quoting, so
  ``"two words"`` is one argument).
* ``@{path}`` — replaced with the contents of *path* (relative paths
  resolve against the working directory).
* !\\`command\\` — the backtick-quoted command is executed in the
  working directory and replaced with its output before the prompt is
  sent.
* If the template uses no ``$ARGUMENTS``/``$N`` placeholder and
  arguments were given, they are appended after two newlines (Gemini
  CLI's default behaviour) so they are never silently dropped.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from kiss.agents.sorcar.persistence import _default_kiss_dir

logger = logging.getLogger(__name__)

# ``---\n<yaml>\n---`` frontmatter block at the very start of the file.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
# ``$ARGUMENTS`` placeholder (all arguments, verbatim).
_ARGUMENTS_RE = re.compile(r"\$ARGUMENTS\b")
# ``$1`` … ``$9`` positional-argument placeholders.
_POSITIONAL_RE = re.compile(r"\$([1-9])\b")
# Combined pattern for the single-pass substitution in
# :func:`expand_command`, matching every injection construct the
# template supports: ``@{path}`` (group ``file``), ``!`command```
# (group ``shell``), ``$ARGUMENTS``, and ``$1`` … ``$9`` (group
# ``pos``).  One pass over the ORIGINAL template is essential:
# replacement text — a user-supplied argument value, an injected
# file's contents, or a shell command's output — must never be
# re-scanned by a later substitution.  Otherwise a data file merely
# referenced with ``@{path}`` could get an embedded ``!`cmd```
# EXECUTED, and literal ``$1`` / ``$ARGUMENTS`` text inside file
# contents or shell output would be rewritten with the user's
# arguments.
_INJECT_RE = re.compile(
    r"@\{(?P<file>[^{}]+)\}"
    r"|!`(?P<shell>[^`]+)`"
    r"|\$ARGUMENTS\b"
    r"|\$(?P<pos>[1-9])\b"
)
# The non-argument injection constructs alone (``@{path}`` and
# ``!`command```).  Used to blank them out of the template before
# deciding whether it contains a *real* argument placeholder: a ``$1``
# or ``$ARGUMENTS`` inside a file path or shell command is consumed by
# the earlier alternatives of ``_INJECT_RE`` and never substituted, so
# it must not suppress the append-args fallback either.
_NON_ARG_INJECT_RE = re.compile(r"@\{[^{}]+\}|!`[^`]+`")

_SHELL_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class CustomCommand:
    """A user- or project-defined slash command loaded from a ``.md`` file.

    Attributes:
        name: Command name as typed after ``/`` (e.g. ``git:commit``).
        description: One-line description from the frontmatter (may be
            empty), shown in ``/help`` and ``/commands``.
        argument_hint: Optional hint for expected arguments (e.g.
            ``[issue-number]``), shown next to the name in listings.
        template: The prompt template (Markdown body after frontmatter).
        source: ``"project"`` or ``"user"`` — where the file was found.
        path: Absolute path of the defining ``.md`` file.
    """

    name: str
    description: str
    argument_hint: str
    template: str
    source: str
    path: str


def user_commands_dir() -> Path:
    """Return the user-level commands directory (``~/.kiss/commands``)."""
    return _default_kiss_dir() / "commands"


def project_commands_dir(work_dir: str) -> Path:
    """Return the project-level commands directory for *work_dir*."""
    return Path(work_dir) / ".kiss" / "commands"


def claude_user_commands_dir() -> Path:
    """Return Claude Code's user commands directory (``~/.claude/commands``).

    Honours the ``CLAUDE_CONFIG_DIR`` environment variable, the same
    override Claude Code itself uses for its ``~/.claude`` directory.
    """
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(config_dir) if config_dir else Path.home() / ".claude"
    return base / "commands"


def claude_project_commands_dir(work_dir: str) -> Path:
    """Return Claude Code's project commands directory for *work_dir*."""
    return Path(work_dir) / ".claude" / "commands"


def _parse_command_file(path: Path, root: Path, source: str) -> CustomCommand | None:
    """Parse one command ``.md`` file into a :class:`CustomCommand`.

    Args:
        path: The ``.md`` file to parse.
        root: The commands directory *path* lives under (for naming).
        source: ``"user"`` or ``"project"``.

    Returns:
        The parsed command, or ``None`` when the file is unreadable or
        its template body is empty.
    """
    try:
        # utf-8-sig drops a leading BOM (common in Windows-authored
        # files) that would otherwise defeat the \A frontmatter match.
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        logger.debug("unreadable command file: %s", path, exc_info=True)
        return None
    meta: dict[str, object] = {}
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
    body = body.strip()
    if not body:
        return None
    rel = path.relative_to(root)
    name = ":".join((*rel.parts[:-1], rel.stem))
    # Collapse all whitespace (a YAML block scalar may span lines) so
    # the one-line ``/commands`` and ``/help`` listings never break —
    # same normalization as skills.py applies to skill descriptions.
    return CustomCommand(
        name=name,
        description=" ".join(str(meta.get("description", "") or "").split()),
        argument_hint=" ".join(str(meta.get("argument-hint", "") or "").split()),
        template=body,
        source=source,
        path=str(path),
    )


def _load_commands_dir(root: Path, source: str) -> dict[str, CustomCommand]:
    """Load every ``*.md`` command file under *root* (recursively)."""
    commands: dict[str, CustomCommand] = {}
    if not root.is_dir():
        return commands
    for path in sorted(root.rglob("*.md")):
        cmd = _parse_command_file(path, root, source)
        if cmd is not None:
            commands[cmd.name] = cmd
    return commands


def discover_commands(work_dir: str) -> dict[str, CustomCommand]:
    """Discover all custom commands visible from *work_dir*.

    Claude Code command files (``~/.claude/commands`` and
    ``<work_dir>/.claude/commands``) are included alongside the native
    ``.kiss`` directories.  Later directories override earlier ones on a
    name conflict; the load order (low → high precedence) is:
    claude-user, user, claude-project, project — so project commands win
    over user commands, and a native ``.kiss`` command wins over a
    Claude Code command at the same level.

    Args:
        work_dir: The project directory whose commands to include.

    Returns:
        Mapping of command name → :class:`CustomCommand`.
    """
    commands = _load_commands_dir(claude_user_commands_dir(), "claude-user")
    commands.update(_load_commands_dir(user_commands_dir(), "user"))
    commands.update(
        _load_commands_dir(claude_project_commands_dir(work_dir), "claude-project")
    )
    commands.update(_load_commands_dir(project_commands_dir(work_dir), "project"))
    return commands


def _read_injected_file(raw: str, work_dir: str) -> str:
    """Return the contents of the ``@{path}`` file named by *raw*.

    Relative paths resolve against *work_dir*.  Unreadable files inject
    a readable error marker instead of raising.

    Args:
        raw: The path text between ``@{`` and ``}``.
        work_dir: Directory against which relative paths resolve.

    Returns:
        The file contents, or an error marker when unreadable.
    """
    raw = raw.strip()
    path = Path(raw)
    if not path.is_absolute():
        path = Path(work_dir) / raw
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return f"[could not read file: {raw}]"


def _run_injected_shell(command: str, work_dir: str) -> str:
    """Return the output of the ``!`command``` shell injection *command*.

    Commands run in *work_dir*.  On failure the stderr output and an
    exit-status note are injected instead, so the model can see what
    went wrong (mirrors Gemini CLI's behaviour).

    Args:
        command: The command text between the backticks.
        work_dir: Working directory for the command.

    Returns:
        The command's output (stdout, plus stderr and an exit-status
        note on failure), stripped of surrounding newlines.
    """
    command = command.strip()
    try:
        proc = subprocess.run(
            command, shell=True, cwd=work_dir, capture_output=True,
            text=True, timeout=_SHELL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"[shell command timed out: {command}]"
    output = proc.stdout
    if proc.returncode != 0:
        output += proc.stderr
        output += f"\n[Shell command exited with code {proc.returncode}]"
    return output.strip("\n")


def expand_command(command: CustomCommand, args_text: str, work_dir: str) -> str:
    """Expand *command*'s template into the prompt to send to the agent.

    All injections — ``@{path}`` file contents, ``!`command``` shell
    output, and the ``$ARGUMENTS`` / ``$1`` … ``$9`` argument
    placeholders — are substituted in one pass over the original
    template, evaluated left to right.  One pass is essential:
    replacement text (argument values, file contents, shell output) is
    never rescanned, so a data file containing ``!`cmd``` is injected
    verbatim instead of having its command executed, and literal
    ``$ARGUMENTS`` / ``$N`` text inside any injected content is never
    rewritten with the user's arguments.  When the template has no
    argument placeholder but arguments were provided, they are appended
    after two newlines.

    Args:
        command: The custom command to expand.
        args_text: Everything the user typed after the command name.
        work_dir: Directory for relative paths and shell commands.

    Returns:
        The fully expanded prompt text.
    """
    args_text = args_text.strip()
    # Computed on the ORIGINAL template (so placeholder-looking text
    # inside injected file contents / shell output cannot suppress the
    # append-args fallback below) with the ``@{path}`` / ``!`command```
    # constructs blanked out first: a ``$1`` or ``$ARGUMENTS`` inside
    # them is consumed by those alternatives of ``_INJECT_RE`` and never
    # substituted, so it is not a real placeholder either.
    scannable = _NON_ARG_INJECT_RE.sub("", command.template)
    has_placeholder = bool(
        _ARGUMENTS_RE.search(scannable) or _POSITIONAL_RE.search(scannable)
    )
    try:
        positional = shlex.split(args_text)
    except ValueError:
        positional = args_text.split()

    def inject_repl(match: re.Match[str]) -> str:
        file_path = match.group("file")
        if file_path is not None:
            return _read_injected_file(file_path, work_dir)
        shell_cmd = match.group("shell")
        if shell_cmd is not None:
            return _run_injected_shell(shell_cmd, work_dir)
        pos = match.group("pos")
        if pos is not None:
            index = int(pos) - 1
            return positional[index] if index < len(positional) else ""
        return args_text  # bare $ARGUMENTS

    text = _INJECT_RE.sub(inject_repl, command.template)
    if args_text and not has_placeholder:
        text = f"{text}\n\n{args_text}"
    return text


def format_command_listing(commands: dict[str, CustomCommand]) -> str:
    """Format *commands* as the aligned listing printed by ``/commands``.

    Args:
        commands: Mapping of command name → command (from
            :func:`discover_commands`).

    Returns:
        A printable multi-line listing, or a hint about where to create
        command files when *commands* is empty.
    """
    if not commands:
        return (
            "No custom commands found.\n"
            f"Create Markdown files in {user_commands_dir()} (user) or "
            "<project>/.kiss/commands (project) to define them.\n"
            "Claude Code commands in ~/.claude/commands and "
            "<project>/.claude/commands are picked up too."
        )
    entries = sorted(commands.values(), key=lambda c: c.name)
    width = max(len(f"/{c.name} {c.argument_hint}".rstrip()) for c in entries)
    lines = []
    for cmd in entries:
        invocation = f"/{cmd.name} {cmd.argument_hint}".rstrip()
        desc = cmd.description or Path(cmd.path).name
        if len(desc) > 100:
            # format_skill_listing parity: keep the listing aligned.
            desc = desc[:97] + "..."
        lines.append(f"  {invocation:<{width}}  ({cmd.source}) {desc}")
    return "\n".join(lines)
