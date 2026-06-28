# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the installer must NOT copy ``MODEL_INFO.json`` into ``~/.kiss/``.

Two install entry points must skip the copy:

* ``install.sh``       (bash bootstrap, runs before VS Code launches)
* ``DependencyInstaller.ts`` (VS Code extension finalization step)

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


def test_dependency_installer_does_not_copy_model_info_json() -> None:
    """``DependencyInstaller.ts`` must not actively copy MODEL_INFO.json."""
    di = (
        _REPO / "src" / "kiss" / "agents" / "vscode" / "src"
        / "DependencyInstaller.ts"
    )
    assert di.exists(), f"DependencyInstaller.ts not found at {di}"
    text = di.read_text()
    for m in re.finditer(r"copyFileSync\(([^)]*)\)", text):
        args = m.group(1)
        assert "MODEL_INFO.json" not in args, (
            f"DependencyInstaller.ts still calls copyFileSync with "
            f"MODEL_INFO.json: {m.group(0)!r}"
        )
    assert (
        re.search(
            r"path\.join\(\s*LOG_DIR\s*,\s*['\"]MODEL_INFO\.json['\"]\s*\)",
            text,
        )
        is None
    ), "DependencyInstaller.ts still writes to LOG_DIR/MODEL_INFO.json"


def test_install_sh_explains_no_model_info_copy() -> None:
    """install.sh keeps a comment explaining why MODEL_INFO.json is not copied."""
    install_sh = _REPO / "install.sh"
    text = install_sh.read_text()
    assert "MODEL_INFO.json" in text, (
        "install.sh should retain a comment explaining that "
        "MODEL_INFO.json is intentionally NOT copied (bundled file is "
        "read directly at runtime)."
    )


def test_dependency_installer_explains_no_model_info_copy() -> None:
    """DependencyInstaller.ts keeps a comment explaining why no copy happens."""
    di = (
        _REPO / "src" / "kiss" / "agents" / "vscode" / "src"
        / "DependencyInstaller.ts"
    )
    text = di.read_text()
    assert "MODEL_INFO.json" in text, (
        "DependencyInstaller.ts should retain a comment explaining "
        "that MODEL_INFO.json is intentionally NOT copied."
    )
