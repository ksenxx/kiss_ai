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

So when ``~/kiss_ai/install.sh`` is a directory (e.g. a botched
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

from kiss.server.web_server import _find_install_script

_INSTALLER_PATH_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "src"
    / "installerPath.js"
)

_NODE = shutil.which("node")


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
