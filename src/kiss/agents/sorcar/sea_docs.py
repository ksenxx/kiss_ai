# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``sea docs``: render the SEA vocabulary tables from the code that defines them.

The settings keys, the kinds, the ``run_agent`` options and the bundled
commands each have one source of truth in the code
(:data:`~kiss.agents.sorcar.sea_settings.SETTING_TYPES`,
:func:`~kiss.agents.sorcar.sea_settings.kind_defaults`,
:data:`~kiss.agents.sorcar.agent_dispatch.OPTION_TYPES`,
:func:`~kiss.agents.sorcar.sea_commands.bundled_commands`).  The
Markdown pages that describe them carry marked blocks::

    <!-- sea-docs: settings -->
    ...generated table...
    <!-- /sea-docs -->

``sea docs`` rewrites every block in :data:`GENERATED_FILES` (``uv run
check`` runs it, like ``generate-api-docs``); ``sea docs --check`` only
reports the blocks that are out of date and exits ``1``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from kiss.agents.sorcar.agent_dispatch import OPTION_DOCS, OPTION_TYPES
from kiss.agents.sorcar.sea_commands import bundled_commands, sea_description
from kiss.agents.sorcar.sea_settings import (
    DISPATCHER_SETTINGS,
    KIND_DOCS,
    SETTING_DOCS,
    SETTING_TYPES,
    kind_defaults,
    wire_field,
)
from kiss.core.config import kiss_home

REPO_ROOT = Path(__file__).resolve().parents[4]
"""The checkout that holds the generated pages (``src/kiss/agents/sorcar`` is four levels down)."""

GENERATED_FILES = (
    "website/kisssorcar.github.io/docs/sea-commands.md",
    "src/kiss/server/README.md",
)
"""Pages (relative to :data:`REPO_ROOT`) whose ``sea-docs`` blocks are regenerated."""

_BLOCK = re.compile(r"(<!-- sea-docs: ([\w-]+) -->\n)(.*?)(<!-- /sea-docs -->)", re.S)


def type_name(expected: type | tuple[type, ...]) -> str:
    """Render a :data:`SETTING_TYPES` value (``int`` or ``(int, float)``) as ``int \\| float``.

    The pipe is escaped because the name sits in a Markdown table cell.
    """
    return " \\| ".join(
        t.__name__ for t in (expected if isinstance(expected, tuple) else (expected,))
    )


def settings_table() -> str:
    """The ``settings()`` keys: key, type, wire field, meaning."""
    rows = ["| Key | Type | `run` wire field | Meaning |", "|---|---|---|---|"]
    for key, expected in SETTING_TYPES.items():
        field = "—" if key in DISPATCHER_SETTINGS else f"`{wire_field(key)}`"
        rows.append(f"| `{key}` | `{type_name(expected)}` | {field} | {SETTING_DOCS[key]} |")
    return "\n".join(rows)


def kinds_table() -> str:
    """The kinds: name, the defaults each lays under the explicit keys, when to use it.

    A path under the Sorcar home is rendered as ``<home>/...`` so the
    generated page does not depend on the machine it was built on.
    """
    home = str(kiss_home())
    rows = ["| Kind | Defaults | Use |", "|---|---|---|"]
    for name, values in kind_defaults().items():
        sets = (
            ", ".join(
                f"`{key}={value.replace(home, '<home>') if isinstance(value, str) else value!r}`"
                for key, value in values.items()
            )
            or "nothing"
        )
        rows.append(f"| `{name}` | {sets} | {KIND_DOCS[name]} |")
    return "\n".join(rows)


def options_table() -> str:
    """The ``run_agent`` / ``run_parallel`` ``options`` keys: key, type, meaning."""
    rows = ["| Option | Type | Meaning |", "|---|---|---|"]
    for key, expected in OPTION_TYPES.items():
        doc = OPTION_DOCS.get(key) or SETTING_DOCS[key]
        rows.append(f"| `{key}` | `{type_name(expected)}` | {doc} |")
    return "\n".join(rows)


def commands_table() -> str:
    """The bundled commands: name, script (relative to ``src/kiss``) and the first sentence of
    ``description()``."""
    package = (REPO_ROOT / "src" / "kiss").resolve()
    rows = ["| Command | Script | Description |", "|---|---|---|"]
    for name, script in sorted(bundled_commands().items()):
        try:
            text = sea_description(script) or ""
        except Exception as exc:  # a broken bundled script is a lint finding, not a docs crash
            text = f"(broken: {exc})"
        first = re.split(r"(?<=[.!?])\s", text.strip().split("\n")[0], maxsplit=1)[0].replace(
            "|", "\\|"
        )
        rel = script.resolve()
        if rel.is_relative_to(package):
            rel = rel.relative_to(package)
        rows.append(f"| `/{name}` | `{rel}` | {first} |")
    return "\n".join(rows)


TABLES = {
    "settings": settings_table,
    "kinds": kinds_table,
    "options": options_table,
    "commands": commands_table,
}
"""Block name -> renderer."""


def render(text: str) -> str:
    """Return *text* with every ``sea-docs`` block regenerated.

    Raises:
        KeyError: a block names an unknown table.
    """

    def replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}{TABLES[match.group(2)]()}\n{match.group(4)}"

    return _BLOCK.sub(replace, text)


def update_file(path: Path, check: bool) -> bool:
    """Regenerate the blocks of *path*; return whether it was (or would be) changed.

    Args:
        path: A Markdown page with ``sea-docs`` blocks.
        check: ``True`` only compares, ``False`` writes the page.
    """
    old = path.read_text(encoding="utf-8")
    new = render(old)
    if new != old and not check:
        path.write_text(new, encoding="utf-8")
    return new != old


def main(argv: list[str] | None = None) -> int:
    """``sea docs [--check] [FILE ...]``: regenerate (or verify) the vocabulary tables.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        ``0`` when every page is up to date (after writing), ``1`` when
        ``--check`` found stale blocks.
    """
    parser = argparse.ArgumentParser(prog="sea docs", description=(__doc__ or "").split("\n\n")[0])
    add_arguments(parser)
    return run(parser.parse_args(argv))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``sea docs`` arguments to *parser*."""
    parser.add_argument("files", nargs="*", help="pages to regenerate; default: the bundled docs")
    parser.add_argument(
        "--check", action="store_true", help="report stale blocks instead of rewriting them"
    )


def run(args: argparse.Namespace) -> int:
    """Execute ``sea docs`` with parsed *args* (see :func:`main`)."""
    files = [Path(f) for f in args.files] or [REPO_ROOT / rel for rel in GENERATED_FILES]
    stale = [path for path in files if path.exists() and update_file(path, args.check)]
    for path in stale:
        print(("stale: " if args.check else "updated: ") + str(path))
    return 1 if stale and args.check else 0
