# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the installer must NOT copy ``MODEL_INFO.json`` into ``~/.kiss/``.

The ``install.sh`` (bash bootstrap, runs before VS Code launches)
entry point must skip the copy.  The ``DependencyInstaller.ts`` (VS
Code extension finalization step) tests moved to
``kiss.tests.agents.vscode.test_install_no_model_info_copy``.

The bundled ``src/kiss/core/models/MODEL_INFO.json`` is the runtime
source of truth for pricing/context tables and is read directly from
the installed package by ``kiss.core.models.model_info``.  Only
``MY_MODELS.json`` (a purely user-curated overrides/extensions file)
is lazily written into ``~/.kiss/`` at runtime, by
``ensure_user_asset_from_default``.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def test_install_sh_does_not_copy_model_info_json() -> None:
    """``install.sh`` must not ``cp ... MODEL_INFO.json`` into ``~/.kiss/``."""
    install_sh = _REPO / "install.sh"
    assert install_sh.exists(), f"install.sh not found at {install_sh}"
    text = install_sh.read_text()
    forbidden_patterns = [
        r'cp\s+["\']?\$MODEL_INFO_SRC["\']?\s+["\']?\$MODEL_INFO_DST["\']?',
        r'cp\s+\S+\s+\S*\.kiss/MODEL_INFO\.json',
        r'MODEL_INFO_DST\s*=\s*["\']?\$HOME/\.kiss/MODEL_INFO\.json',
        r'MODEL_INFO_DST\s*=\s*["\']?\$KISS_HOME_DIR/MODEL_INFO\.json',
    ]
    for pattern in forbidden_patterns:
        m = re.search(pattern, text)
        assert m is None, (
            f"install.sh still contains a MODEL_INFO.json copy matching "
            f"{pattern!r}: {m.group(0) if m else ''!r}"
        )


def test_install_sh_explains_no_model_info_copy() -> None:
    """install.sh keeps a comment explaining why MODEL_INFO.json is not copied."""
    install_sh = _REPO / "install.sh"
    text = install_sh.read_text()
    assert "MODEL_INFO.json" in text, (
        "install.sh should retain a comment explaining that "
        "MODEL_INFO.json is intentionally NOT copied (bundled file is "
        "read directly at runtime)."
    )
