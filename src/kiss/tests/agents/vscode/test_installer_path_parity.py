# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration repro: installer lookup diverges between the twins.

Commit 1c720158 introduced ``web_server._find_install_script`` as the
"Python twin of ``findInstallScript()`` in the extension's
``installerPath.js`` so the remote webapp's Update button probes the
exact same location as the VS Code extension".

The two implementations were NOT equivalent at the time:

* JS:     ``fs.existsSync(candidate)`` — true for **any** filesystem
          entry, including a directory named ``install.sh``.
* Python: ``candidate.is_file()`` — true only for regular files.

So when ``~/.kiss/kiss_ai/install.sh`` was a directory (e.g. a botched
checkout or extraction), the VS Code extension "found" the installer
and tried to ``bash`` a directory in a terminal, while the remote
webapp took the missing-installer path.  The two frontends that are
documented to behave identically disagreed.

These tests drive the real ``node`` implementation and the real Python
implementation against the same on-disk fixtures (no mocks) and assert
they agree — including on the curl-bootstrap URL both fall back to
when the installer is missing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kiss.server.web_server import (
    _KISS_AI_ROOT,
    _bootstrap_install_url,
    _find_install_script,
)

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


def _with_bootstrap_env(value: str | None, fn: object) -> str:
    """Call *fn* with ``$KISS_UPDATE_BOOTSTRAP_URL`` set to *value*.

    ``None`` removes the variable.  The prior value is restored
    afterwards so tests never leak environment into each other.

    Args:
        value: Env value to set, or ``None`` to unset.
        fn: Zero-argument callable returning a string.

    Returns:
        The string returned by *fn*.
    """
    saved = os.environ.get("KISS_UPDATE_BOOTSTRAP_URL")
    try:
        if value is None:
            os.environ.pop("KISS_UPDATE_BOOTSTRAP_URL", None)
        else:
            os.environ["KISS_UPDATE_BOOTSTRAP_URL"] = value
        result = fn()  # type: ignore[operator]
        assert isinstance(result, str)
        return result
    finally:
        if saved is None:
            os.environ.pop("KISS_UPDATE_BOOTSTRAP_URL", None)
        else:
            os.environ["KISS_UPDATE_BOOTSTRAP_URL"] = saved


def _js_bootstrap_install_url(env_value: str | None) -> str:
    """Run the extension's ``bootstrapInstallUrl()`` under real node.

    Args:
        env_value: Value for ``$KISS_UPDATE_BOOTSTRAP_URL`` in the node
            process, or ``None`` to run without it.

    Returns:
        The URL the extension would bootstrap from.
    """
    assert _NODE is not None
    script = (
        "const {bootstrapInstallUrl} = "
        f"require({json.dumps(str(_INSTALLER_PATH_JS))});"
        "console.log(JSON.stringify(bootstrapInstallUrl()));"
    )
    env = dict(os.environ)
    env.pop("KISS_UPDATE_BOOTSTRAP_URL", None)
    if env_value is not None:
        env["KISS_UPDATE_BOOTSTRAP_URL"] = env_value
    out = subprocess.run(
        [_NODE, "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        env=env,
    )
    result = json.loads(out.stdout.strip())
    assert isinstance(result, str)
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

        Historically the JS twin (``fs.existsSync``) returned the
        directory path — the extension then tried to ``bash`` a
        directory — while the Python twin (``is_file()``) returned
        ``None`` — the webapp fell back to the curl bootstrap.  The
        documented contract is that both probe "the exact same
        location ... identically".
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

    def test_bootstrap_url_default_parity(self) -> None:
        """Both twins fall back to the same public curl bootstrap URL.

        When ``~/.kiss/kiss_ai/install.sh`` is missing, both frontends
        run ``curl -fsSL <url> | bash``; the URL must be the README's
        install one-liner in both, or the two Update buttons would
        bootstrap from different sources.
        """
        py = _with_bootstrap_env(None, _bootstrap_install_url)
        js = _js_bootstrap_install_url(None)
        assert py == js
        assert py == (
            "https://raw.githubusercontent.com/ksenxx/kiss_ai/"
            "main/scripts/install.sh"
        )

    def test_bootstrap_url_env_override_parity(self) -> None:
        """``$KISS_UPDATE_BOOTSTRAP_URL`` overrides both twins alike."""
        override = "file:///tmp/kiss-fake-bootstrap.sh"
        py = _with_bootstrap_env(override, _bootstrap_install_url)
        js = _js_bootstrap_install_url(override)
        assert py == override
        assert js == override
