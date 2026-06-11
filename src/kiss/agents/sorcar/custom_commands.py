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
# ``@{path}`` file-content injection.
_FILE_INJECT_RE = re.compile(r"@\{([^{}]+)\}")
# ``!`command``` shell-output injection.
_SHELL_INJECT_RE = re.compile(r"!`([^`]+)`")

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
        text = path.read_text(encoding="utf-8")
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
    return CustomCommand(
        name=name,
        description=str(meta.get("description", "") or ""),
        argument_hint=str(meta.get("argument-hint", "") or ""),
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

    User commands (``~/.kiss/commands``) load first; project commands
    (``<work_dir>/.kiss/commands``) load second and override user
    commands with the same name, mirroring Gemini CLI's precedence.

    Args:
        work_dir: The project directory whose commands to include.

    Returns:
        Mapping of command name → :class:`CustomCommand`.
    """
    commands = _load_commands_dir(user_commands_dir(), "user")
    commands.update(_load_commands_dir(project_commands_dir(work_dir), "project"))
    return commands


def _inject_files(template: str, work_dir: str) -> str:
    """Replace every ``@{path}`` with the named file's contents."""

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        path = Path(raw)
        if not path.is_absolute():
            path = Path(work_dir) / raw
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return f"[could not read file: {raw}]"

    return _FILE_INJECT_RE.sub(repl, template)


def _inject_shell(template: str, work_dir: str) -> str:
    """Replace every ``!`command``` with the command's output.

    Commands run in *work_dir*.  On failure the stderr output and an
    exit-status note are injected instead, so the model can see what
    went wrong (mirrors Gemini CLI's behaviour).
    """

    def repl(match: re.Match[str]) -> str:
        command = match.group(1).strip()
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

    return _SHELL_INJECT_RE.sub(repl, template)


def expand_command(command: CustomCommand, args_text: str, work_dir: str) -> str:
    """Expand *command*'s template into the prompt to send to the agent.

    Substitution order follows Gemini CLI: file injection first, then
    shell injection, then argument placeholders.  When the template has
    no argument placeholder but arguments were provided, they are
    appended after two newlines.

    Args:
        command: The custom command to expand.
        args_text: Everything the user typed after the command name.
        work_dir: Directory for relative paths and shell commands.

    Returns:
        The fully expanded prompt text.
    """
    text = _inject_files(command.template, work_dir)
    text = _inject_shell(text, work_dir)
    args_text = args_text.strip()
    has_placeholder = bool(
        _ARGUMENTS_RE.search(text) or _POSITIONAL_RE.search(text)
    )
    try:
        positional = shlex.split(args_text)
    except ValueError:
        positional = args_text.split()

    def positional_repl(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return positional[index] if index < len(positional) else ""

    text = _POSITIONAL_RE.sub(positional_repl, text)
    text = _ARGUMENTS_RE.sub(lambda _m: args_text, text)
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
            "<project>/.kiss/commands (project) to define them."
        )
    entries = sorted(commands.values(), key=lambda c: c.name)
    width = max(len(f"/{c.name} {c.argument_hint}".rstrip()) for c in entries)
    lines = []
    for cmd in entries:
        invocation = f"/{cmd.name} {cmd.argument_hint}".rstrip()
        desc = cmd.description or Path(cmd.path).name
        lines.append(f"  {invocation:<{width}}  ({cmd.source}) {desc}")
    return "\n".join(lines)
