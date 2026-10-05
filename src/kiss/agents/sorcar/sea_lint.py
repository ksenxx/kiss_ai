# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``sea lint``: a deterministic checker (and codemod) over every agent script (SEA).

The SEA contract lives in :mod:`kiss.agents.sorcar.sea_settings`
(``settings()`` keys, kinds and their defaults) and
:mod:`kiss.agents.sorcar.sea_commands` (getters, ``extends`` layers,
the ``/command`` registry).  This module checks every script against
it, so a contract change is enforced by ``uv run check`` instead of by
hand-grepping scripts and docstrings
(``reports/sea-run-agent-semantics-and-automation-2026-10-04.md``, A1).

Rules, each a :class:`Finding` code:

``broken``
    The script does not load or its settings / getters violate the
    contract (import error, unknown, renamed or removed key, ill-typed
    value, unknown kind, ``extends`` cycle, a getter returning the
    wrong type).  A renamed key is reported as ``renamed-key`` instead.
``renamed-key``
    ``settings()`` uses a former key name
    (:data:`~kiss.agents.sorcar.sea_settings.RENAMED_SETTINGS`);
    ``--fix`` rewrites it.
``redundant-key``
    A declared key merely repeats the default of the script's ``kind``.
``no-description``
    A registered ``/command`` whose script defines no ``description()``.
``unknown-model``
    ``settings()["model"]`` names neither a catalogued model nor a
    model-picker SEA.
``lock-without-value``
    A ``locked`` key the merged settings give no value, so the lock
    protects nothing.
``hidden-not-literal``
    ``settings()["hidden"]`` evaluates to ``True`` but is not written
    as the literal ``True``, so the command registry (which reads the
    source, never runs it) still lists the script.
``stale-docstring``
    The module docstring mentions a getter the contract no longer has
    (``model()``, ``tool_profile()``, ``is_parallel()``,
    ``dispatch_timeout()``, ``append_to_system_prompt()``, ...).
``home-literal``
    A ``~/.kiss`` path in a code string (not prose): the home directory
    is the brand's (``~/.s10s`` for Seamless Loop) and ``$KISS_HOME``
    may move it; use :func:`kiss.core.config.kiss_home`.
``stale-prose``
    A sentence about SEA semantics that the code no longer backs, in
    the documentation pages ``sea docs`` generates into and in the
    dispatcher modules (:data:`PROSE_FILES`): a ``~/.kiss/`` home path
    (write ``$KISS_HOME/``), the pre-2026-10-04 claim that a SEA's
    settings "still win" / "win over" a call (the one rule is
    :data:`~kiss.agents.sorcar.sea_settings.PRECEDENCE_RULE`), or a
    "``/name`` declares ``{...}``" claim whose dict is not what the
    command's ``settings()`` returns.  Text inside a generated
    ``<!-- sea-docs: ... -->`` block is skipped.

Cost and duration rules (a ``max_budget`` below the script's observed
cost, a ``timeout`` below its observed duration) need the task history
and live in the ``rsi7d`` settings tuner, not here: this checker is
environment-free so that ``uv run check`` is deterministic.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from kiss.agents.sorcar.sea_commands import (
    bundled_commands,
    check_sea,
    get_command,
    list_commands,
    model_sea,
    sea_getter_value,
)
from kiss.agents.sorcar.sea_settings import (
    META_SETTINGS,
    RENAMED_SETTINGS,
    SeaError,
    declares_hidden,
    kind_defaults,
)

REMOVED_GETTERS = (
    "model",
    "tool_profile",
    "is_parallel",
    "dispatch_timeout",
    "work_dir",
    "max_budget",
    "use_worktree",
    "auto_commit",
    "append_to_system_prompt",
    "append_to_prompt",
    "docker_image",
)
"""Former getter names: a docstring mentioning ``<name>()`` describes a contract that no longer
exists."""

BUNDLED_SEA_DIRS = ("agents/seas", "agents/sorcar", "agents/third_party_agents")
"""Where the bundled scripts live, relative to the ``kiss`` package."""


@dataclass(frozen=True)
class Finding:
    """One lint finding.

    Attributes:
        path: The script.
        code: The rule (see the module docstring).
        message: What is wrong, in one sentence.
        fixable: Whether ``--fix`` rewrites it.
    """

    path: Path
    code: str
    message: str
    fixable: bool = False

    def __str__(self) -> str:
        """Return ``path: code: message`` (``[fixable]`` when ``--fix`` handles it)."""
        tag = " [fixable]" if self.fixable else ""
        return f"{self.path}: {self.code}: {self.message}{tag}"


def bundled_seas() -> list[Path]:
    """Return every bundled script, sorted: the ``*_sea.py`` files and the built-in commands."""
    package = Path(__file__).resolve().parents[2]
    scripts = {path for sub in BUNDLED_SEA_DIRS for path in (package / sub).rglob("*_sea.py")}
    return sorted(scripts | set(bundled_commands().values()))


def registered_seas() -> list[Path]:
    """Return the scripts of every registered ``/command`` (bundled and user folders)."""
    paths = [get_command(name) for name in list_commands()]
    return sorted({path for path in paths if path is not None})


def default_targets(registered: bool) -> list[Path]:
    """Return the scripts ``sea lint`` checks when no path is given.

    The bundled scripts always; the user's registered ``SEAS.md``
    scripts only when *registered* is true (``--registered``), so
    ``uv run check`` never fails on, and ``--fix`` never rewrites, a
    file outside this checkout.
    """
    scripts = set(bundled_seas())
    if registered:
        scripts |= set(registered_seas())
    return sorted(scripts)


def lint_all(paths: Iterable[Path] | None = None, registered: bool = False) -> list[Finding]:
    """Lint *paths*, or the default targets (then also the prose of :data:`PROSE_FILES`).

    Args:
        paths: Scripts (or SEA folders) to check; ``None`` checks
            :func:`default_targets` and the prose files.
        registered: With ``paths=None``, also check the user's
            registered scripts.

    Returns:
        The findings, in path order.
    """
    scripts = default_targets(registered) if paths is None else [_script_of(Path(p)) for p in paths]
    commands = {path: name for name in list_commands() if (path := get_command(name))}
    findings: list[Finding] = []
    for script in scripts:
        findings.extend(lint_sea(script, commands.get(script)))
    if paths is None:
        findings.extend(lint_prose())
    return findings


PROSE_FILES = (
    "website/kisssorcar.github.io/docs/sea-commands.md",
    "website/kisssorcar.github.io/docs/cli.md",
    "src/kiss/server/README.md",
    "src/kiss/agents/sorcar/sea_settings.py",
    "src/kiss/agents/sorcar/agent_dispatch.py",
    "src/kiss/agents/sorcar/sorcar_agent.py",
    "src/kiss/agents/sorcar/run_config.py",
    "src/kiss/server/agent_file.py",
)
"""Files (relative to the checkout) whose prose the ``stale-prose`` rule reads."""

STALE_PROSE = (
    (re.compile(r"~/\.kiss/"), "a `~/.kiss/` home path; write `$KISS_HOME/`"),
    (
        re.compile(r"(?i)\b(?:settings|script|SEA)(?:'s)?(?: \w+){0,3} (?:still )?wins? over\b"
                   r"|\bstill wins?\b"),
        "claims a SEA's settings win over a call; state sea_settings.PRECEDENCE_RULE instead",
    ),
    (
        re.compile(r"(?i)\b(?:then|before) stop(?:s|ping) (?:it|the sub-task)\b"
                   r"|\bstops it when the wait runs out\b"
                   r"|\bon expiry the sub-task is stopped\b"),
        "claims a run_agent timeout stops the sub-task; since U1 the call returns the "
        "agent_job id and the sub-task keeps running",
    ),
)
"""``(pattern, message)`` pairs of the ``stale-prose`` rule."""

_DECLARES_CLAIM = re.compile(
    r"`/([A-Za-z0-9_-]+)` declares `(\{[^`]*\})`"
    r"|`(\{[^`]*\})` \(what `/([A-Za-z0-9_-]+)` declares\)"
)
"""A prose claim about a command's ``settings()`` literal, in either word order."""

_GENERATED_BLOCK = re.compile(r"<!-- sea-docs: \w+ -->.*?<!-- /sea-docs -->", re.DOTALL)


def _stale_declares_claim(match: re.Match[str]) -> str:
    """Return why a "`/name` declares `{...}`" claim is stale, or ``""`` when the SEA agrees.

    The claim is compared with what the command's script's
    ``settings()`` returns as written (before its kind's defaults are
    laid under it), so a settings change of a bundled SEA is caught in
    every page that quotes it.
    """
    name = match.group(1) or match.group(4)
    claim = match.group(2) or match.group(3)
    path = get_command(name)
    if path is None:
        return f"`/{name}` is not a registered command"
    try:
        claimed = json.loads(claim)
    except ValueError:
        try:
            claimed = ast.literal_eval(claim)
        except (ValueError, SyntaxError):
            return f"the quoted settings of `/{name}` are not a dict literal"
    try:
        declared = sea_getter_value(path, "settings") or {}
    except SeaError as exc:
        return f"`/{name}` does not load: {exc}"
    if claimed != declared:
        return f"`/{name}` declares `{json.dumps(declared)}`"
    return ""


def _blank_lines(match: re.Match[str]) -> str:
    """Return as many newlines as *match* spans, so line numbers after it stay right."""
    return "\n" * match.group(0).count("\n")


def lint_prose(root: Path | None = None) -> list[Finding]:
    """Return the ``stale-prose`` findings of :data:`PROSE_FILES` under *root* (the checkout).

    A file that does not exist (an installed package without the
    website) is skipped.
    """
    root = root or Path(__file__).resolve().parents[4]
    findings: list[Finding] = []
    for rel in PROSE_FILES:
        path = root / rel
        if not path.is_file():
            continue
        text = _GENERATED_BLOCK.sub(_blank_lines, path.read_text("utf-8"))
        for number, line in enumerate(text.splitlines(), 1):
            for pattern, message in STALE_PROSE:
                if pattern.search(line):
                    findings.append(Finding(path, "stale-prose", f"line {number}: {message}"))
            for match in _DECLARES_CLAIM.finditer(line):
                why = _stale_declares_claim(match)
                if why:
                    findings.append(Finding(path, "stale-prose", f"line {number}: {why}"))
    return findings


def _script_of(path: Path) -> Path:
    """Return the ``*_sea.py`` of a SEA folder, or *path* itself."""
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*_sea.py"))
        if candidates:
            return candidates[0]
    return path


def lint_sea(path: Path, command: str | None = None) -> list[Finding]:
    """Return the findings for one script.

    Args:
        path: The ``*_sea.py`` file.
        command: The ``/command`` name the script is registered under,
            or ``None`` when it is not a command.
    """
    findings: list[Finding] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [Finding(path, "broken", f"cannot parse: {exc}")]
    findings.extend(_lint_source(path, source, tree))
    if any(f.code == "renamed-key" for f in findings):
        # The loader refuses renamed keys; the rest needs the loaded script.
        return findings
    try:
        # Exactly what the daemon does for a run of the script, plus the
        # getter checks ``/<name> check`` makes.
        layers, _cmd, _description = check_sea(path, require_description=False)
    except SeaError as exc:
        findings.append(Finding(path, "broken", str(exc)))
        return findings
    own = layers[-1]
    namespace = own.namespace
    settings_fn = namespace.get("settings")
    try:
        declared = settings_fn() if callable(settings_fn) else {}
    except Exception as exc:  # noqa: BLE001 - a second call may fail where the loader's passed
        return [*findings, Finding(path, "broken", f"settings() raised on a second call: {exc}")]
    declared = declared if isinstance(declared, dict) else {}
    kind = own.settings["kind"]
    defaults = kind_defaults()[kind]
    for key, value in declared.items():
        if key not in META_SETTINGS and key in defaults and defaults[key] == value:
            findings.append(
                Finding(
                    path,
                    "redundant-key",
                    f"settings()[{key!r}] = {value!r} repeats the default of kind {kind!r}",
                )
            )
    if command is not None and not callable(namespace.get("description")):
        findings.append(
            Finding(
                path,
                "no-description",
                f"/{command} has no description() for its help text",
            )
        )
    model = own.settings.get("model")
    if isinstance(model, str) and model and not _known_model(model):
        findings.append(
            Finding(
                path,
                "unknown-model",
                f"settings()['model'] = {model!r} is no catalogued model or model-picker SEA",
            )
        )
    merged = _merged(layers)
    for key in merged.get("locked") or ():
        if key not in merged:
            findings.append(
                Finding(
                    path,
                    "lock-without-value",
                    f"locked key {key!r} has no value to protect",
                )
            )
    if declared.get("hidden") is True and not declares_hidden(path):
        findings.append(
            Finding(
                path,
                "hidden-not-literal",
                "settings()['hidden'] is computed; only a literal `\"hidden\": True` hides the "
                "script",
            )
        )
    return findings


def _merged(layers: list) -> dict:
    """Return the merged settings of *layers*."""
    from kiss.agents.sorcar.sea_settings import merge_settings

    return merge_settings([layer.settings for layer in layers])


def _known_model(name: str) -> bool:
    """Return whether *name* is in the model catalogue or names a model-picker SEA."""
    from kiss.core.models.model_info import MODEL_INFO

    return name in MODEL_INFO or model_sea(name) is not None


def _lint_source(path: Path, source: str, tree: ast.Module) -> list[Finding]:
    """Return the findings that need only the script's text and AST."""
    findings: list[Finding] = []
    for node in _settings_dict_keys(tree):
        if node.value in RENAMED_SETTINGS:
            findings.append(
                Finding(
                    path,
                    "renamed-key",
                    f"settings() key {node.value!r} is now {RENAMED_SETTINGS[node.value]!r}",
                    fixable=True,
                )
            )
    doc = ast.get_docstring(tree) or ""
    stale = sorted(
        {name for name in REMOVED_GETTERS if re.search(rf"``{name}\(\)``|\b{name}\(\)", doc)}
    )
    if stale:
        findings.append(
            Finding(
                path,
                "stale-docstring",
                "module docstring mentions removed getters: " + ", ".join(f"{n}()" for n in stale),
            )
        )
    for node in _code_strings(tree):
        if isinstance(node.value, str) and "~/.kiss" in node.value:
            findings.append(
                Finding(
                    path,
                    "home-literal",
                    f"line {node.lineno}: '~/.kiss' in a code string; use kiss_home()",
                )
            )
    return findings


def _code_strings(tree: ast.Module) -> list[ast.Constant]:
    """Return the string constants of *tree* that are not docstrings (expression statements)."""
    prose = {
        id(stmt.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for stmt in node.body
        if isinstance(stmt, ast.Expr)
    }
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose
    ]


def _settings_dict_keys(tree: ast.Module) -> list[ast.Constant]:
    """Return the string constants naming settings keys inside the module's ``settings`` function.

    The keys of every dict literal in the function except dicts that
    are the *value* of another dict's key (``model_config``'s contents
    are not settings), plus the string elements of a ``locked`` list,
    which name keys too.
    """
    keys: list[ast.Constant] = []
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == "settings"):
            continue
        nested = {
            id(value)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Dict)
            for value in sub.values
            if isinstance(value, ast.Dict)
        }
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Dict) or id(sub) in nested:
                continue
            for key, value in zip(sub.keys, sub.values, strict=True):
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                keys.append(key)
                if key.value == "locked" and isinstance(value, ast.List | ast.Tuple):
                    keys.extend(
                        e
                        for e in value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
    return keys


def fix_sea(path: Path) -> list[str]:
    """Rewrite the renamed ``settings()`` keys of the script at *path* in place.

    Args:
        path: The ``*_sea.py`` file.

    Returns:
        One line per rewritten key (empty when nothing changed).
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    # ``ast`` columns are UTF-8 byte offsets: slice the encoded source.
    data = source.encode("utf-8")
    lines = data.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = [
        (offsets[k.lineno - 1] + k.col_offset, offsets[k.end_lineno - 1] + k.end_col_offset, k)
        for k in _settings_dict_keys(tree)
        if k.value in RENAMED_SETTINGS and k.end_lineno is not None and k.end_col_offset is not None
    ]
    changed: list[str] = []
    for start, end, key in sorted(edits, reverse=True):
        old = str(key.value)
        new = RENAMED_SETTINGS[old]
        # Replace the name inside the literal, keeping its prefix and quotes.
        literal = data[start:end].decode("utf-8").replace(old, new, 1)
        data = data[:start] + literal.encode("utf-8") + data[end:]
        changed.append(f"{path}:{key.lineno}: {old!r} -> {new!r}")
    if changed:
        rewritten = data.decode("utf-8")
        ast.parse(rewritten, filename=str(path))  # never save a script that no longer parses
        path.write_text(rewritten, encoding="utf-8")
    return list(reversed(changed))


def main(argv: list[str] | None = None) -> int:
    """``sea lint [--fix] [PATH ...]``: check (and rewrite) agent scripts.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        ``0`` when there are no findings (after ``--fix``), else ``1``.
    """
    parser = argparse.ArgumentParser(prog="sea lint", description=(__doc__ or "").split("\n\n")[0])
    add_arguments(parser)
    return run(parser.parse_args(argv))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``sea lint`` arguments to *parser*."""
    parser.add_argument(
        "paths", nargs="*", help="scripts or SEA folders; default: every bundled script"
    )
    parser.add_argument("--fix", action="store_true", help="rewrite renamed settings() keys")
    parser.add_argument(
        "--registered", action="store_true", help="also check the scripts your SEAS.md registers"
    )


def run(args: argparse.Namespace) -> int:
    """Execute ``sea lint`` with parsed *args* (see :func:`main`)."""
    paths = [Path(p) for p in args.paths] or None
    if args.fix:
        targets = [_script_of(p) for p in paths] if paths else default_targets(args.registered)
        for script in targets:
            for line in fix_sea(script):
                print(f"fixed {line}")
    findings = lint_all(paths, args.registered)
    for finding in findings:
        print(finding)
    print(f"sea lint: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
