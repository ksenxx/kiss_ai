"""Every bundled ``seas/<name>/<name>_sea.py`` is a slash command resolving to its file.

One parametrised test over the folder on disk: a new bundled SEA is
covered the moment it lands, and a SEA whose ``settings`` declares
``"hidden": True`` (``coding``, ``sorcar``) is asserted to stay out of
the registry.  The hidden set is pinned literally rather than read
through ``declares_hidden()`` (which the registry itself uses): a public
SEA misclassified as hidden would otherwise pass this test while its
slash command silently disappeared.  The SEA-specific extras (``/sync``
with no text, ``/rsi7d`` needing task text, the ``write`` timeout, the
model-picker registration of the routers) stay in each SEA's own test file.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands

pytestmark = pytest.mark.usefixtures("reset_registry")

_SEAS_DIR = Path(sea_commands.__file__).resolve().parents[1] / "seas"
_NAMES = sorted(
    folder.name
    for folder in _SEAS_DIR.iterdir()
    if folder.name != sea_commands.BASE_FOLDER and (folder / f"{folder.name}_sea.py").is_file()
)
_HIDDEN = {"coding", "sorcar"}
"""The bundled SEAs that are intentionally not slash commands; every other one is public."""


@pytest.mark.parametrize("name", _NAMES)
def test_bundled_sea_is_a_slash_command(name: str) -> None:
    """``/<name> text`` resolves to the text and the bundled file; ``/<name> help`` describes it.

    The settings the registry reports for the file equal those of the
    class the package module defines, so the loaded script and the
    importable module are the same SEA; the ``help`` text is its
    ``description()``.  A hidden SEA is not a command at all.
    """
    path = _SEAS_DIR / name / f"{name}_sea.py"
    if name in _HIDDEN:
        assert sea_commands.get_command(name) is None
        assert sea_commands.slash_command_task(f"/{name} some text") is None
        return
    assert sea_commands.get_command(name) == path
    assert sea_commands.slash_command_task(f"/{name} some text") == ("some text", path)
    assert sea_commands.slash_command_task(f"/{name} help") is None
    module = importlib.import_module(f"kiss.agents.seas.{name}.{name}_sea")
    sea = getattr(module, type(sea_commands.load_sea(path)).__name__)()
    assert sea_commands.sea_settings(path) == sea_commands.base_settings([sea])
    assert sea_commands.help_text_if_command(f"/{name} help") == sea.description()
