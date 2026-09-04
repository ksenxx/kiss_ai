# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration repro: installer lookup diverges between the twins.

Commit 1c720158 introduced ``web_server._find_install_script`` as the
"Python twin of ``findInstallScript()`` in the extension's
``installerPath.js`` so the remote webapp's Update button probes the
exact same location as the VS Code extension".

The two implementations are NOT equivalent:

* JS:     ``fs.existsSync(candidate)`` — true for **any** filesystem
          entry, including a directory named ``install.sh``.
* Python: ``candidate.is_file()`` — true only for regular files.

So when ``~/.kiss/kiss_ai/install.sh`` is a directory (e.g. a botched
checkout or extraction), the VS Code extension "finds" the installer
and tries to ``bash`` a directory in a terminal, while the remote
webapp reports "install.sh not found".  The two frontends that are
documented to behave identically disagree.

These tests drive the real ``node`` implementation and the real Python
implementation against the same on-disk fixtures (no mocks) and assert
they agree.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kiss.server.web_server import _KISS_AI_ROOT, _find_install_script

_INSTALLER_PATH_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "src"
    / "installerPath.js"
)

_NODE = shutil.which("node")


def _js_kiss_ai_root() -> str:
    """Run the extension's ``kissAiRoot()`` under real node."""
    assert _NODE is not None
    script = (
        f"const {{kissAiRoot}} = require({json.dumps(str(_INSTALLER_PATH_JS))});"
        "console.log(JSON.stringify(kissAiRoot()));"
    )
    out = subprocess.run(
        [_NODE, "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result = json.loads(out.stdout.strip())
    assert isinstance(result, str)
    return result


def _js_find_install_script(root: Path) -> str | None:
    """Run the extension's ``findInstallScript(root)`` under real node."""
    assert _NODE is not None
    script = (
        f"const {{findInstallScript}} = require({json.dumps(str(_INSTALLER_PATH_JS))});"
        f"console.log(JSON.stringify(findInstallScript({json.dumps(str(root))})));"
    )
    out = subprocess.run(
        [_NODE, "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result = json.loads(out.stdout.strip())
    assert result is None or isinstance(result, str)
    return result


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
class TestInstallerPathParity:
    """Extension JS and web-server Python must resolve identically."""

    def test_regular_file_both_find_it(self, tmp_path: Path) -> None:
        """Control: a real install.sh file is found by both twins."""
        script = tmp_path / "install.sh"
        script.write_text("#!/bin/bash\necho hi\n")
        py = _find_install_script(tmp_path)
        js = _js_find_install_script(tmp_path)
        assert py is not None and str(py) == str(script)
        assert js == str(script)

    def test_missing_file_both_report_none(self, tmp_path: Path) -> None:
        """Control: with nothing on disk both twins report None."""
        assert _find_install_script(tmp_path) is None
        assert _js_find_install_script(tmp_path) is None

    def test_directory_named_install_sh_parity(self, tmp_path: Path) -> None:
        """A directory named install.sh must be treated identically.

        Currently the JS twin (``fs.existsSync``) returns the directory
        path — the extension then tries to ``bash`` a directory — while
        the Python twin (``is_file()``) returns ``None`` — the webapp
        shows "install.sh not found".  The documented contract is that
        both probe "the exact same location ... identically".
        """
        (tmp_path / "install.sh").mkdir()
        py = _find_install_script(tmp_path)
        js = _js_find_install_script(tmp_path)
        py_found = py is not None
        js_found = js is not None
        assert py_found == js_found, (
            f"installer lookup diverged for a directory named install.sh: "
            f"python={py!r} js={js!r}"
        )

    def test_default_root_is_kiss_home_clone_in_both_twins(self) -> None:
        """Both twins default to ``~/.kiss/kiss_ai``, the curl installer's clone.

        ``scripts/install.sh`` (the ``curl | bash`` entry point) clones the
        public repo into ``~/.kiss/kiss_ai``; an Update button probing the
        legacy ``~/kiss_ai`` would never find that clone.
        """
        expected = Path.home() / ".kiss" / "kiss_ai"
        assert _KISS_AI_ROOT == expected
        assert _js_kiss_ai_root() == str(expected)
        assert _KISS_AI_ROOT != Path.home() / "kiss_ai"
