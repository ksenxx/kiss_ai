# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Static integration tests for installation ownership.

KISS Sorcar can be installed either from a cloned repository via the top-level
``install.sh`` or directly from a VSIX.  In both cases the VS Code extension is
installed and activated, so runtime dependency setup must live in the extension
installer instead of being duplicated in source-install shell scripts.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
TOP_LEVEL_INSTALL = REPO_ROOT / "install.sh"
VSCODE_DIR = REPO_ROOT / "src" / "kiss" / "agents" / "vscode"
DEPENDENCY_INSTALLER = VSCODE_DIR / "src" / "DependencyInstaller.ts"
KISS_PATHS = VSCODE_DIR / "src" / "kissPaths.ts"


class TestInstallationOwnership(unittest.TestCase):
    """Verify runtime setup is centralized in the extension installer."""

    def test_stale_vscode_install_script_removed(self) -> None:
        """The old standalone VS Code install script must not come back.

        It duplicated top-level source install logic and extension activation
        setup, but was not packaged in the VSIX and had no tracked caller.
        """
        self.assertFalse(
            (VSCODE_DIR / "install.sh").exists(),
            "src/kiss/agents/vscode/install.sh is obsolete; keep runtime "
            "setup in DependencyInstaller.ts and source bootstrapping in "
            "the repository install.sh.",
        )

    def test_source_install_delegates_runtime_setup_to_extension(self) -> None:
        """Top-level install.sh should only bootstrap/build/install the extension."""
        text = TOP_LEVEL_INSTALL.read_text()
        self.assertIn("Runtime setup is owned by the extension", text)
        self.assertIn("--install-extension", text)
        self.assertIn(".extension-updated", text)

        forbidden_snippets = [
            "uv sync",
            "playwright install chromium",
            "remote_password",
            "systemctl --user enable --now kiss-web",
            "launchctl bootstrap",
            "cloudflared/releases/latest/download",
            "Persist PATH in shell rc",
        ]
        for snippet in forbidden_snippets:
            self.assertNotIn(
                snippet,
                text,
                f"source install.sh must not duplicate extension runtime setup: {snippet}",
            )

    def test_extension_installer_owns_runtime_setup_for_vsix_and_source(self) -> None:
        """Direct VSIX installs must receive the same runtime setup."""
        text = DEPENDENCY_INSTALLER.read_text()
        for snippet in [
            "installUv()",
            "installGit()",
            "installNode()",
            "playwright",
            "installCloudflaredIfNeeded()",
            "installCliScript",
            "restartKissWebDaemon",
            "ensurePathInShellRc",
            "ensureApiKeys",
            "ensureRemotePassword",
        ]:
            self.assertIn(snippet, text)

    def test_install_dir_marker_is_not_used(self) -> None:
        """Direct VSIX install model: kissProjectPath is the VSIX-bundled copy.

        ``findKissProject()`` must NOT consult ``~/.kiss/install_dir`` and
        the extension must NOT write that marker.  Source installs run
        against the VSIX-shipped ``kiss_project`` just like direct VSIX
        installs, so the marker would only create a divergent path.
        """
        paths_text = KISS_PATHS.read_text()
        self.assertNotIn(
            "install_dir",
            paths_text,
            "findKissProject() must not read ~/.kiss/install_dir; the "
            "extension always uses the VSIX-bundled kiss_project.",
        )

        installer_text = DEPENDENCY_INSTALLER.read_text()
        self.assertNotIn(
            "writeInstallDirMarker",
            installer_text,
            "DependencyInstaller must not write ~/.kiss/install_dir; "
            "source installs converge on the embedded VSIX bundle.",
        )
        self.assertNotIn(
            "'install_dir'",
            installer_text,
            "DependencyInstaller must not reference the install_dir marker.",
        )

        install_sh = TOP_LEVEL_INSTALL.read_text()
        self.assertNotIn(
            'printf \'%s\\n\' "$PROJECT_DIR" > "$HOME/.kiss/install_dir"',
            install_sh,
            "install.sh must not write the install_dir marker; the "
            "embedded VSIX copy is the authoritative kiss_project.",
        )


if __name__ == "__main__":
    unittest.main()
