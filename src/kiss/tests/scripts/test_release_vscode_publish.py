"""Integration checks for VS Code extension release publishing."""

import json
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RELEASE_SCRIPT = REPO_ROOT / "scripts" / "release.sh"
VSCODE_PACKAGE = REPO_ROOT / "src" / "kiss" / "agents" / "vscode" / "package.json"


def _release_script_publish_command_tokens() -> list[str]:
    lines = RELEASE_SCRIPT.read_text().splitlines()
    for index, line in enumerate(lines):
        if "npx @vscode/vsce publish" not in line:
            continue
        command_lines = [line]
        next_index = index + 1
        while command_lines[-1].rstrip().endswith("\\") and next_index < len(lines):
            command_lines.append(lines[next_index])
            next_index += 1
        command = " ".join(part.rstrip().removesuffix("\\") for part in command_lines)
        return shlex.split(command)
    raise AssertionError("release.sh must publish the VS Code extension with vsce")


def test_release_publish_allows_manifest_api_proposals() -> None:
    """Every proposed API declared in package.json must be allowed during publish."""
    manifest = json.loads(VSCODE_PACKAGE.read_text())
    proposals = manifest.get("enabledApiProposals", [])
    assert proposals, "package.json should declare the proposed APIs that the extension uses"

    tokens = _release_script_publish_command_tokens()
    option_index = tokens.index("--allow-proposed-apis")
    allowed = tokens[option_index + 1 :]

    assert "--allow-all-proposed-apis" not in tokens
    assert set(proposals) <= set(allowed)


def test_release_publish_uses_existing_vsix_package() -> None:
    """Release publishing must use the VSIX built earlier in the release process."""
    tokens = _release_script_publish_command_tokens()
    package_index = tokens.index("--packagePath")
    assert tokens[package_index + 1] == "kiss-sorcar.vsix"
