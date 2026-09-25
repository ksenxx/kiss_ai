# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Keep ``docs/sea-commands.md`` truthful against the real registry.

The public page ``website/kisssorcar.github.io/docs/sea-commands.md``
tells external contributors how ``~/.kiss/SEAS.md`` is parsed and how
``/xxx text`` reaches a SEA.  These tests pull the examples straight
out of that Markdown and run them through
:mod:`kiss.agents.sorcar.sea_commands`, so a behaviour change that
invalidates the page fails here instead of silently going stale.  They
also check that the page is wired into the site index files.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.core.config import kiss_home
from kiss.tests.conftest import IS_WINDOWS

_REPO = Path(__file__).resolve().parents[5]
_SITE = _REPO / "website" / "kisssorcar.github.io"
_DOC = _SITE / "docs" / "sea-commands.md"


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Drop the module's in-memory state before and after each test."""
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make *tmp_path* the user's home directory.

    ``Path.home()`` reads ``HOME`` on POSIX and ``USERPROFILE`` on
    Windows; both are set so the fake home works on either.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def _fenced_block(text: str, first_line: str) -> str:
    """Return the dedented body of the fenced code block whose first line is *first_line*.

    Blocks nested inside a Markdown list are indented; dedenting makes
    them usable verbatim as files.
    """
    for match in re.finditer(r"```[a-z]*\n(.*?)\n *```", text, re.DOTALL):
        body = textwrap.dedent(match.group(1)) + "\n"
        if body.splitlines()[0].strip() == first_line:
            return body
    raise AssertionError(f"no fenced block starting with {first_line!r} in {_DOC}")


def test_page_is_listed_in_every_site_index() -> None:
    """The page must be reachable from the docs index, llms.txt, sitemap, and llms-full."""
    assert _DOC.is_file()
    url = "https://kisssorcar.github.io/docs/sea-commands.md"
    assert "](sea-commands.md)" in (_SITE / "docs" / "index.md").read_text()
    assert url in (_SITE / "llms.txt").read_text()
    assert f"<loc>{url}</loc>" in (_SITE / "sitemap.xml").read_text()
    assert f"<!-- Source: {url} -->" in (_SITE / "llms-full.txt").read_text()


def test_seas_md_example_parses_as_documented(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The "complete example" SEAS.md block yields exactly the three documented folders.

    The block uses ``~``, ``$WORK``, inline comments, and a leading
    ``#`` comment line; the page promises each of those is handled.
    """
    monkeypatch.setenv("WORK", str(home / "work"))
    block = _fenced_block(_DOC.read_text(), "# ~/.kiss/SEAS.md")
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(block, encoding="utf-8")

    folders = sea_commands._read_seas_md_folders()

    # ``/opt/...`` is absolute on POSIX.  On Windows a path without a
    # drive is relative, so the parser anchors it in the home directory
    # (which only contributes its drive: ``C:/opt/agents/experimental``).
    opt = Path("/opt/agents/experimental")
    if IS_WINDOWS:
        opt = home / opt
    assert folders == [home / "my-seas", home / "work" / "agents", opt]


def test_three_step_walkthrough_registers_standup_command(home: Path) -> None:
    """Following the page's three steps yields ``/standup`` bound to the new file.

    Step 1 writes the documented ``standup_sea.py`` (and the snippet
    must be a working SEA: its ``system_prompt()`` returns text).  Step 2
    appends ``~/my-seas`` to SEAS.md.  Step 3 relies on substring
    autocomplete (``st`` matches ``standup``) and the ``/standup ...``
    prompt being rewritten into a run_agent directive on that file.
    """
    doc = _DOC.read_text()
    sea_src = _fenced_block(doc, "# ~/my-seas/standup_sea.py")
    folder = home / "my-seas"
    folder.mkdir()
    sea_file = folder / "standup_sea.py"
    sea_file.write_text(sea_src, encoding="utf-8")
    namespace: dict[str, Any] = {}
    exec(compile(sea_src, str(sea_file), "exec"), namespace)  # noqa: S102
    system_prompt = namespace["system_prompt"]
    assert callable(system_prompt)
    text = system_prompt()
    assert isinstance(text, str) and "stand-up" in text

    kiss_home().mkdir(parents=True, exist_ok=True)
    with (kiss_home() / "SEAS.md").open("a", encoding="utf-8") as fh:
        fh.write("~/my-seas\n")

    commands = sea_commands.refresh_registry()
    assert "standup" in commands
    assert [c for c in commands if "st" in c.lower()].count("standup") == 1

    task = "finished the docs page, next is the release, blocked on review"
    hit = sea_commands.rewrite_prompt_if_command(f"/standup {task}")
    assert hit is not None
    rewritten, path = hit
    assert path.resolve() == sea_file.resolve()
    assert f'agent = "{path}"' in rewritten
    assert rewritten.endswith(task)


def test_documented_edge_cases_hold(home: Path) -> None:
    """The bullet list under "The slash-command flow" describes real parser behaviour.

    Checks: dotted and spaced stems are skipped, a leading underscore is
    kept, a bare ``/name`` is not rewritten, an unknown command is not
    rewritten, a leading space disables the command, and ``/deployx``
    does not match ``/deploy``.
    """
    folder = home / "seas"
    folder.mkdir()
    for stem in ("deploy", "_scratch", "release.notes", "my agent"):
        (folder / f"{stem}_sea.py").write_text("# stub\n", encoding="utf-8")
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text("seas\n", encoding="utf-8")

    commands = sea_commands.refresh_registry()
    assert "deploy" in commands
    assert "_scratch" in commands
    assert "release.notes" not in commands
    assert "my agent" not in commands

    assert sea_commands.rewrite_prompt_if_command("/deploy") is None
    assert sea_commands.rewrite_prompt_if_command("/nosuch ship") is None
    assert sea_commands.rewrite_prompt_if_command(" /deploy ship") is None
    assert sea_commands.rewrite_prompt_if_command("/deployx ship") is None
    hit = sea_commands.rewrite_prompt_if_command("/deploy <task>a</task><task>b</task>")
    assert hit is not None
    assert hit[0].endswith("<task>a</task><task>b</task>")
