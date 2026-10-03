# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""install.sh derives the default state directory from the brand being built.

``kiss_home()`` (Python) and ``kissHomeDir()`` (extension) default to
``~/<home_dir>`` with ``home_dir`` from ``media/brand.json``; install.sh
must write its reload marker, progress file and MODEL_INFO.json into the
same directory, so its ``brand_home_dir_name`` reads the key from the
``.brand/`` overlay when there is one, else from the checkout's own
brand.json, and falls back to ``.kiss`` for anything but a plain name.
The function is run by bash straight out of install.sh.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_INSTALL = _REPO / "install.sh"
_FUNCTION = re.compile(r"^brand_home_dir_name\(\) \{\n.*?^\}\n", re.M | re.S)


def _home_dir_name(project_dir: Path, kiss_home: str | None = None) -> tuple[str, str]:
    """Return ``(brand_home_dir_name, KISS_HOME_DIR)`` as install.sh computes them."""
    match = _FUNCTION.search(_INSTALL.read_text(encoding="utf-8"))
    assert match, "brand_home_dir_name() not found in install.sh"
    script = (
        f"PROJECT_DIR={str(project_dir)!r}\nHOME=/home/u\n"
        + (f"KISS_HOME={kiss_home!r}\n" if kiss_home is not None else "unset KISS_HOME\n")
        + match.group(0)
        + 'BRAND_HOME_DIR_NAME="$(brand_home_dir_name)"\n'
        'KISS_HOME_DIR="${KISS_HOME:-$HOME/$BRAND_HOME_DIR_NAME}"\n'
        'printf "%s\\n%s\\n" "$BRAND_HOME_DIR_NAME" "$KISS_HOME_DIR"\n'
    )
    out = subprocess.run(
        ["bash", "-c", script], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return out[0], out[1]


def _write_brand(path: Path, home_dir: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"product_name": "X", "home_dir": home_dir}), encoding="utf-8")


def test_checkout_brand_json_drives_the_default(tmp_path: Path) -> None:
    """The checkout's media/brand.json names the directory; $KISS_HOME still wins."""
    media = tmp_path / "src" / "kiss" / "agents" / "vscode" / "media" / "brand.json"
    _write_brand(media, ".s10s")
    assert _home_dir_name(tmp_path) == (".s10s", "/home/u/.s10s")
    assert _home_dir_name(tmp_path, "/tmp/private") == (".s10s", "/tmp/private")
    assert _home_dir_name(tmp_path, "") == (".s10s", "/home/u/.s10s")


def test_brand_overlay_wins_over_the_checkout(tmp_path: Path) -> None:
    """A .brand/brand.json overlay (the VSIX is built with it) decides, not the checkout."""
    _write_brand(tmp_path / "src" / "kiss" / "agents" / "vscode" / "media" / "brand.json", ".kiss")
    _write_brand(tmp_path / ".brand" / "brand.json", ".loop")
    assert _home_dir_name(tmp_path) == (".loop", "/home/u/.loop")


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", 3])
def test_anything_but_a_plain_name_falls_back_to_kiss(tmp_path: Path, bad: object) -> None:
    """An empty, relative or nested value, or no file at all, keeps ``.kiss``."""
    _write_brand(tmp_path / "src" / "kiss" / "agents" / "vscode" / "media" / "brand.json", bad)
    assert _home_dir_name(tmp_path) == (".kiss", "/home/u/.kiss")


def test_missing_brand_json_and_the_real_checkout(tmp_path: Path) -> None:
    """No brand.json at all is stock; the real checkout agrees with kiss.core.brand."""
    assert _home_dir_name(tmp_path) == (".kiss", "/home/u/.kiss")
    from kiss.core.brand import HOME_DIR

    assert _home_dir_name(_REPO)[0] == HOME_DIR
