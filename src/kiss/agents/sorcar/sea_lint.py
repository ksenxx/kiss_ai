# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``sea lint``: a deterministic checker (and codemod) over every SEA.

The SEA contract lives in :class:`kiss.agents.seas.base.base_sea.BaseSea`
(the methods), :mod:`kiss.agents.sorcar.sea_settings` (``settings``
keys, kinds and their defaults) and :mod:`kiss.agents.sorcar.sea_commands`
(the launcher, the ``/command`` registry).  This module checks every script against
it, so a contract change is enforced by ``uv run check`` instead of by
hand-grepping scripts and docstrings
(``reports/sea-run-agent-semantics-and-automation-2026-10-04.md``, A1).

Rules, each a :class:`Finding` code:

``broken``
    The script does not load or its settings / methods violate the
    contract (import error, no or several ``BaseSea`` subclasses,
    unknown, renamed or removed key, ill-typed value, unknown kind, a
    method returning the wrong type).  A renamed key is reported as
    ``renamed-key`` instead.
``renamed-key``
    ``settings()`` uses a former key name
    (:data:`~kiss.agents.sorcar.sea_settings.RENAMED_SETTINGS`);
    ``--fix`` rewrites it.
``verdict``
    A ``tool_call_hook`` returns a literal ``None`` or string, the
    allow / refuse spellings of older hooks.  The contract is a
    :class:`~kiss.core.tool_verdict.Verdict`: ``ALLOW`` allows,
    ``refuse(text)`` refuses.  ``--fix`` rewrites ``None`` to ``ALLOW``
    and ``"text"`` to ``refuse("text")`` and imports both names from
    ``kiss.agents.seas.base.base_sea``.
``channel-kind``
    ``settings()`` writes ``"kind": "channel"``, which became the flag
    ``"channel": True`` (a kind is defaults only; a channel is a
    worker).  ``--fix`` rewrites the entry.
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
    A docstring of the script (module, class, function or constant)
    mentions a getter the contract no longer has (``model()``,
    ``tool_profile()``, ``add_to_tools()``, ``add_to_system_prompt()``,
    ``add_to_prompt()``, ... as ``name()`` or ``:func:`name```), or
    makes one of the ``stale-prose`` claims
    below (a ``~/.kiss/`` path, settings that "win over" a call, a
    timeout that "stops" the sub-task).
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
    BASE_FOLDER,
    base_settings,
    bundled_commands,
    check_sea,
    defines,
    get_command,
    list_commands,
    load_sea,
    model_sea,
    own_settings,
)
from kiss.agents.sorcar.sea_settings import (
    META_SETTINGS,
    RENAMED_SETTINGS,
    SeaError,
    declares_hidden,
    kind_defaults,
    settings_functions,
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
    "add_to_system_prompt",
    "add_to_prompt",
    "add_to_tools",
    "docker_image",
)
"""Former getter names: a docstring mentioning ``<name>()`` describes a contract that no longer
exists (the ``add_to_*`` getters became the ``prompt``, ``system_prompt`` and ``tools``
methods, which receive the current value and return the new one)."""

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
    scripts = {
        path for sub in BUNDLED_SEA_DIRS for path in (package / sub).rglob("*_sea.py")
        if path.parent.name != BASE_FOLDER  # base_sea.py is the contract, not a SEA
    }
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
    "src/kiss/agents/sorcar/sea_apply.py",
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
        declared = own_settings(load_sea(path))
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
        # method checks ``/<name> check`` makes.
        seas, _cmd, _description = check_sea(path, require_description=False)
        merged = base_settings(seas)
        declared = own_settings(seas[-1])
    except SeaError as exc:
        findings.append(Finding(path, "broken", str(exc)))
        return findings
    kind = merged["kind"]
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
    if command is not None and not defines(seas, "description"):
        findings.append(
            Finding(
                path,
                "no-description",
                f"/{command} has no description() for its help text",
            )
        )
    model = merged.get("model")
    if isinstance(model, str) and model and not _known_model(model):
        findings.append(
            Finding(
                path,
                "unknown-model",
                f"settings()['model'] = {model!r} is no catalogued model or model-picker SEA",
            )
        )
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
    for node in _literal_verdicts(tree):
        findings.append(
            Finding(
                path,
                "verdict",
                f"line {node.lineno}: tool_call_hook returns {node.value!r}; return ALLOW or "
                f"refuse(text) (a Verdict, from kiss.agents.seas.base.base_sea)",
                fixable=True,
            )
        )
    for key, _value in _channel_kinds(tree):
        findings.append(
            Finding(
                path,
                "channel-kind",
                f'line {key.lineno}: settings() writes "kind": "channel", which became the '
                f'flag "channel": True',
                fixable=True,
            )
        )
    for lineno, doc in _docstrings(tree, source):
        stale = sorted(
            name for name in REMOVED_GETTERS
            if re.search(rf"\b{name}\(\)|:func:`~?(?:[\w.]+\.)?{name}`", doc)
        )
        if stale:
            findings.append(
                Finding(
                    path,
                    "stale-docstring",
                    f"line {lineno}: docstring mentions removed getters: "
                    + ", ".join(f"{n}()" for n in stale),
                )
            )
        for pattern, message in STALE_PROSE:
            match = pattern.search(doc)
            if match:
                line = lineno + doc.count("\n", 0, match.start())
                findings.append(Finding(path, "stale-docstring", f"line {line}: {message}"))
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


def _prose_nodes(tree: ast.Module) -> list[ast.Constant]:
    """Return the docstrings of *tree*: every string that is a bare expression statement.

    That is the module, class and function docstrings plus the strings
    documenting a module-level constant (the line after its assignment).
    """
    return [
        stmt.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for stmt in node.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    ]


def _docstrings(tree: ast.Module, source: str) -> list[tuple[int, str]]:
    """Return ``(line, text)`` of every docstring of *tree* (see :func:`_prose_nodes`).

    *text* is the docstring as written in *source*, quotes included, so
    a newline in it is a physical line and ``line + text.count("\\n",
    0, offset)`` is the source line of a match; an escaped newline in
    the decoded value would make that count drift.
    """
    return [
        (node.lineno, ast.get_source_segment(source, node) or str(node.value))
        for node in _prose_nodes(tree)
    ]


def _code_strings(tree: ast.Module) -> list[ast.Constant]:
    """Return the string constants of *tree* that are not docstrings (expression statements)."""
    prose = {id(node) for node in _prose_nodes(tree)}
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose
    ]


def _settings_dict_keys(tree: ast.Module) -> list[ast.Constant]:
    """Return the string constants naming settings keys inside the SEA's ``settings`` method.

    The keys of every dict literal in the function except dicts that
    are the *value* of another dict's key (``model_config``'s contents
    are not settings), plus the string elements of a ``locked`` list,
    which name keys too.
    """
    keys: list[ast.Constant] = []
    for sub in _settings_dicts(tree):
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


def _settings_dicts(tree: ast.Module) -> list[ast.Dict]:
    """Return the dict literals of the SEA's ``settings`` method that hold settings keys.

    Every dict literal in the function except dicts anywhere inside
    the *value* of another dict's key (``model_config``'s contents,
    however deep, are not settings).
    """
    dicts: list[ast.Dict] = []
    for node in settings_functions(tree):
        nested = {
            id(inner)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Dict)
            for value in sub.values
            for inner in ast.walk(value)
            if isinstance(inner, ast.Dict)
        }
        dicts.extend(
            sub for sub in ast.walk(node) if isinstance(sub, ast.Dict) and id(sub) not in nested
        )
    return dicts


def _own_returns(node: ast.AST) -> list[ast.Return]:
    """Return the ``return`` statements of *node*'s own body, not of nested functions or classes."""
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    returns: list[ast.Return] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Return):
            returns.append(child)
        elif not isinstance(child, scopes):
            returns.extend(_own_returns(child))
    return returns


def _literal_verdicts(tree: ast.Module) -> list[ast.Constant]:
    """Return every ``None`` or string literal a ``tool_call_hook`` of the script returns itself."""
    return [
        ret.value
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "tool_call_hook"
        for ret in _own_returns(node)
        if isinstance(ret.value, ast.Constant)
        and (ret.value.value is None or isinstance(ret.value.value, str))
    ]


def _channel_kinds(tree: ast.Module) -> list[tuple[ast.Constant, ast.Constant]]:
    """Return every ``("kind", "channel")`` key/value literal pair of a ``settings()`` dict.

    Nested dicts (``model_config``'s contents) are not settings and are
    left alone, as in :func:`_settings_dict_keys`.
    """
    return [
        (key, value)
        for sub in _settings_dicts(tree)
        for key, value in zip(sub.keys, sub.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "kind"
        and isinstance(value, ast.Constant) and value.value == "channel"
    ]


def _span(node: ast.expr, offsets: list[int]) -> tuple[int, int]:
    """Return the ``(start, end)`` byte offsets of *node* in the source *offsets* index."""
    assert node.end_lineno is not None and node.end_col_offset is not None
    return (
        offsets[node.lineno - 1] + node.col_offset,
        offsets[node.end_lineno - 1] + node.end_col_offset,
    )


def _rewrites(tree: ast.Module, data: bytes) -> list[tuple[int, int, str, str]]:
    """Return the byte-span edits ``--fix`` makes to the source *data*, as
    ``(start, end, new text, note)``.

    A renamed ``settings()`` key gets its current name inside its own
    quotes; ``"kind": "channel"`` becomes ``"channel": True``; a
    ``tool_call_hook``'s literal ``None`` becomes ``ALLOW`` and its
    literal string ``refuse(<the literal>)``, with ``from
    kiss.agents.seas.base.base_sea import ...`` of the names the
    script does not import yet inserted after its last top-level
    import.
    """
    lines = data.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def text(node: ast.expr) -> str:
        start, end = _span(node, offsets)
        return data[start:end].decode("utf-8")

    edits: list[tuple[int, int, str, str]] = []
    for key in _settings_dict_keys(tree):
        if key.value in RENAMED_SETTINGS:
            new = RENAMED_SETTINGS[key.value]
            edits.append((*_span(key, offsets), text(key).replace(key.value, new, 1),
                          f"{key.value!r} -> {new!r}"))
    for key, value in _channel_kinds(tree):
        edits.append((*_span(key, offsets), text(key).replace("kind", "channel", 1),
                      "'kind' -> 'channel'"))
        edits.append((*_span(value, offsets), "True", "'channel' -> True"))
    needed: set[str] = set()
    for node in _literal_verdicts(tree):
        if node.value is None:
            edits.append((*_span(node, offsets), "ALLOW", "None -> ALLOW"))
            needed.add("ALLOW")
        else:
            edits.append((*_span(node, offsets), f"refuse({text(node)})",
                          f"{node.value!r} -> refuse({node.value!r})"))
            needed.add("refuse")
    imported = {
        alias.asname or alias.name
        for node in tree.body if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    missing = sorted(needed - imported)
    if missing:
        last_import = max(
            (node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)),
            key=lambda node: node.end_lineno or 0, default=None,
        )
        at = offsets[last_import.end_lineno or 0] if last_import is not None else 0
        import_line = f"from kiss.agents.seas.base.base_sea import {', '.join(missing)}\n"
        edits.append((at, at, import_line, f"import {', '.join(missing)}"))
    return edits


def fix_sea(path: Path) -> list[str]:
    """Rewrite the fixable findings of the script at *path* in place (:func:`_rewrites`).

    Args:
        path: The ``*_sea.py`` file.

    Returns:
        One line per rewrite (empty when nothing changed).
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    # ``ast`` columns are UTF-8 byte offsets: slice the encoded source.
    data = source.encode("utf-8")
    changed: list[str] = []
    # Applied last span first, so earlier offsets stay valid; the spans are distinct.
    for start, end, new, note in sorted(_rewrites(tree, data), reverse=True):
        lineno = data[:start].count(b"\n") + 1
        data = data[:start] + new.encode("utf-8") + data[end:]
        changed.append(f"{path}:{lineno}: {note}")
    if changed:
        rewritten = data.decode("utf-8")
        ast.parse(rewritten, filename=str(path))  # never save a script that no longer parses
        path.write_text(rewritten, encoding="utf-8")
    return list(reversed(changed))


def main(argv: list[str] | None = None) -> int:
    """``sea lint [--fix] [PATH ...]``: check (and rewrite) SEAs.

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
    parser.add_argument(
        "--fix", action="store_true",
        help="rewrite the fixable findings (renamed-key, channel-kind, verdict)",
    )
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
