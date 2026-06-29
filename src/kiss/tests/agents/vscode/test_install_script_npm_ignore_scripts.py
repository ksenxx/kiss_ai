# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: the Update button's ``install.sh`` must run ``npm ci`` with
``--ignore-scripts`` so the extension build cannot hang on dependency
install scripts.

Background — the bug
====================
Clicking the settings Update button runs ``~/kiss_ai/install.sh`` in an
integrated terminal.  Step ``[5/6] Building VS Code extension...`` ran a
plain ``npm ci``, which executes every dependency lifecycle (install /
postinstall) script.  The extension's lockfile contains exactly two packages
with install scripts:

* ``keytar`` — an *optional*, lazily-imported dependency of ``@vscode/vsce``
  used only for publish credential storage, never by ``vsce package``.  Its
  install script is ``prebuild-install || node-gyp rebuild``:
  ``prebuild-install`` downloads a prebuilt binary from the **archived**
  atom/node-keytar GitHub releases and the fallback compiles natively with
  node-gyp.  Either path can block forever (no timeout, no output) on
  network or toolchain trouble, freezing the update at::

      >>> [5/6] Building VS Code extension...
      npm warn deprecated prebuild-install@7.1.3: No longer maintained. ...
      ^C

* ``@vscode/vsce-sign`` — postinstall used only for VSIX *signing*, which
  ``vsce package`` never does.

Neither script is needed to compile and package the VSIX, so ``install.sh``
(and the release scripts, which build the same extension) now pass
``--ignore-scripts --no-audit --no-fund`` to ``npm ci``.

The behavioural test below executes ``npm ci`` with the exact flags parsed
out of ``install.sh`` against a throwaway project whose dependency carries a
marker-writing install script, proving lifecycle scripts cannot run (and
therefore cannot hang) during the update build.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"
RELEASE_SCRIPTS = [
    REPO / "scripts" / "release.sh",
    REPO / "scripts" / "release_exp.sh",
]


def _npm_ci_invocation(script: Path) -> tuple[str, int]:
    """Return ``(line, byte_offset)`` of the executable ``npm ci`` invocation.

    ``install.sh`` wraps ``npm ci`` in ``run_with_heartbeat`` so the user
    sees elapsed-time output during the otherwise-silent dependency
    install; the release scripts call it directly.  Either way exactly one
    line invokes ``npm ci`` with its flags — find it and return both the
    line and its byte offset so ordering checks can compare against later
    ``npm run …`` invocations.
    """
    src = script.read_text(encoding="utf-8")
    lines: list[tuple[str, int]] = []
    offset = 0
    for raw in src.splitlines(keepends=True):
        stripped = raw.lstrip()
        # Drop bash comments and lines whose only ``npm ci`` mention is the
        # heartbeat *label* string (preceding the real invocation tokens).
        if not stripped.startswith("#"):
            # Match either ``npm ci <flags>`` at the start of the line or
            # ``… npm ci "${NPM_CI_FLAGS[@]}"`` after run_with_heartbeat.
            if re.search(r"(^|\s)npm ci(\s|$)", raw) and "echo " not in raw:
                lines.append((raw.rstrip("\n"), offset))
        offset += len(raw)
    # Pick the last executable invocation — the retry branch in
    # ``install.sh`` re-runs the same command, but the *first* one is the
    # one whose ordering we care about (it runs before compile/package).
    assert lines, f"no 'npm ci' invocation found in {script}"
    return lines[0]


def _flag_source(script: Path) -> str:
    """Return text containing the npm-ci flags in *script*.

    ``install.sh`` defines them in a bash array (``NPM_CI_FLAGS=(…)``); the
    release scripts inline them on the ``npm ci`` line.  Concatenate the
    invocation line with the lines defining ``NPM_CI_FLAGS`` so a single
    substring check covers both layouts.
    """
    line, _ = _npm_ci_invocation(script)
    src = script.read_text(encoding="utf-8")
    flag_array = "\n".join(
        raw for raw in src.splitlines() if "NPM_CI_FLAGS=" in raw
    )
    return line + "\n" + flag_array


def _npm_ci_flags(script: Path) -> list[str]:
    """Return the list of flags passed to ``npm ci`` in *script*.

    ``install.sh`` defines them in a bash array (``NPM_CI_FLAGS=(…)``); the
    release scripts inline them on the ``npm ci`` line.  Either way the
    returned list contains the tokens following ``npm ci``.
    """
    src = script.read_text(encoding="utf-8")
    m = re.search(r"NPM_CI_FLAGS=\(([^)]*)\)", src)
    if m:
        return m.group(1).split()
    line, _ = _npm_ci_invocation(script)
    # Drop the ``npm ci`` prefix; any preceding ``run_with_heartbeat "label"``
    # tokens never appear on the release-script side (which uses the inline
    # form), so split-after-``ci`` is sufficient here.
    tokens = line.split()
    idx = tokens.index("ci")
    return tokens[idx + 1 :]


def _line_offset(script: Path, needle: str) -> int:
    """Return the byte offset of the first line containing *needle*.

    The heartbeat wrapper means ``npm run compile`` no longer starts a
    line — it follows ``run_with_heartbeat "tsc" `` — so a plain
    ``str.index("\\n    npm run compile")`` no longer works.  Locate the
    first non-comment line mentioning the needle instead.
    """
    src = script.read_text(encoding="utf-8")
    offset = 0
    for raw in src.splitlines(keepends=True):
        if needle in raw and not raw.lstrip().startswith("#"):
            return offset
        offset += len(raw)
    raise AssertionError(f"{needle!r} not found in {script}")


def test_install_sh_npm_ci_ignores_lifecycle_scripts() -> None:
    """``install.sh`` must build the extension with ``npm ci --ignore-scripts``.

    A plain ``npm ci`` runs keytar's ``prebuild-install || node-gyp rebuild``
    install script, which can hang the Update button forever at step [5/6].
    """
    flag_src = _flag_source(INSTALL_SCRIPT)
    assert "--ignore-scripts" in flag_src, (
        "install.sh must pass --ignore-scripts to npm ci; without it the "
        "keytar/prebuild-install lifecycle script can hang the update at "
        "'[5/6] Building VS Code extension...'"
    )
    _, npm_ci_offset = _npm_ci_invocation(INSTALL_SCRIPT)
    compile_offset = _line_offset(INSTALL_SCRIPT, "npm run compile")
    copy_offset = _line_offset(INSTALL_SCRIPT, "npm run copy-kiss")
    package_offset = _line_offset(INSTALL_SCRIPT, "npm run package")
    assert npm_ci_offset < compile_offset, (
        "npm ci --ignore-scripts must run before compiling the extension"
    )
    assert compile_offset < copy_offset, (
        "install.sh must compile before copying the bundled runtime"
    )
    assert copy_offset < package_offset, (
        "install.sh must copy the bundled runtime before packaging the VSIX"
    )


def test_release_scripts_npm_ci_ignore_lifecycle_scripts() -> None:
    """The release scripts build the same extension and need the same guard."""
    for script in RELEASE_SCRIPTS:
        flag_src = _flag_source(script)
        assert "--ignore-scripts" in flag_src, (
            f"{script.name} must pass --ignore-scripts to npm ci "
            "(parity with install.sh)"
        )
        _, npm_ci_offset = _npm_ci_invocation(script)
        compile_offset = _line_offset(script, "npm run compile")
        copy_offset = _line_offset(script, "npm run copy-kiss")
        package_offset = _line_offset(script, "npm run package")
        assert npm_ci_offset < compile_offset
        assert compile_offset < copy_offset
        assert copy_offset < package_offset


def test_install_sh_npm_ci_omits_optional_and_prefers_offline() -> None:
    """``install.sh`` must skip keytar entirely and prefer the npm cache.

    ``--omit=optional`` drops keytar (the optional, hang-prone dep that
    triggered the "[5/6] Building VS Code extension..." freeze), and
    ``--prefer-offline`` makes re-runs after an interrupted attempt
    finish in seconds by reusing ~/.npm tarballs instead of refetching.
    Both flags appeared on the release-script side first; install.sh has
    to match so the Update button benefits from the same hardening.
    """
    flag_src = _flag_source(INSTALL_SCRIPT)
    assert "--omit=optional" in flag_src, (
        "install.sh must pass --omit=optional so keytar is never installed; "
        "its prebuild-install deprecation warning was the last log line "
        "users saw before the script appeared to hang."
    )
    assert "--prefer-offline" in flag_src, (
        "install.sh must pass --prefer-offline so retry runs use the npm "
        "cache populated by the first (interrupted) attempt."
    )


def test_install_sh_has_heartbeat_for_silent_steps() -> None:
    """install.sh must surface progress during long silent npm steps.

    Without a heartbeat the user sees ``npm warn deprecated …`` and then
    nothing for 30-60 s while tarballs are fetched — long enough to assume
    the script hung and abort it.  The fix wraps ``npm ci`` and the build
    scripts in ``run_with_heartbeat`` which prints elapsed-time output
    every ``HEARTBEAT_INTERVAL`` seconds.
    """
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "run_with_heartbeat()" in src, (
        "install.sh must define a run_with_heartbeat helper to keep users "
        "informed during silent npm/git steps."
    )
    assert "run_with_heartbeat \"npm ci\"" in src, (
        "install.sh must wrap the npm ci invocation in run_with_heartbeat "
        "so the user sees progress output every ~15 s."
    )


def test_install_sh_traps_sigint_for_double_confirm() -> None:
    """install.sh must require two SIGINTs to abort.

    The regression report — '^C' appearing right after the deprecation
    warning despite the user not pressing Ctrl+C — is consistent with a
    stray signal from a backgrounded shell / sleeping laptop / closed
    terminal.  A single-signal trap that ignores the first interrupt and
    only honors a second within a short window prevents accidental aborts
    while still letting a determined user abort the install.
    """
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "trap handle_interrupt INT TERM" in src, (
        "install.sh must trap SIGINT and SIGTERM so a stray signal does "
        "not kill the install during a silent npm ci stretch."
    )


def _write_dep_tarball(dest: Path) -> None:
    """Create ``evil-dep-1.0.0.tgz`` in *dest* with a marker install script.

    The dependency's install script writes ``$MARKER`` when executed — the
    stand-in for keytar's hang-capable ``prebuild-install`` script.
    """
    manifest = {
        "name": "evil-dep",
        "version": "1.0.0",
        "scripts": {
            "install": (
                "node -e \"require('fs')"
                ".writeFileSync(process.env.MARKER, 'ran')\""
            ),
        },
    }
    data = json.dumps(manifest).encode()
    with tarfile.open(dest / "evil-dep-1.0.0.tgz", "w:gz") as tar:
        info = tarfile.TarInfo("package/package.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def _run_npm_ci(flags: list[str], tmp_path: Path, name: str) -> Path:
    """Run ``npm ci`` with *flags* in a fresh project; return the marker path.

    The project depends on the local ``evil-dep`` tarball whose install
    script writes the returned marker file when lifecycle scripts execute.
    Everything resolves offline (``file:`` dependency only).
    """
    proj = tmp_path / name
    proj.mkdir()
    _write_dep_tarball(proj)
    (proj / "package.json").write_text(
        json.dumps(
            {
                "name": "update-build-fixture",
                "version": "1.0.0",
                "dependencies": {"evil-dep": "file:./evil-dep-1.0.0.tgz"},
            }
        ),
        encoding="utf-8",
    )
    marker = proj / "lifecycle-script-ran"
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path)}
    npm = shutil.which("npm")
    assert npm is not None
    env["PATH"] = f"{Path(npm).parent}:{env['PATH']}"
    # Build the lockfile without installing anything (runs no scripts).
    subprocess.run(
        [npm, "install", "--package-lock-only", "--no-audit", "--no-fund"],
        cwd=proj,
        env={**env, "MARKER": str(marker)},
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        [npm, "ci", *flags],
        cwd=proj,
        env={**env, "MARKER": str(marker)},
        check=True,
        capture_output=True,
        timeout=120,
    )
    assert (proj / "node_modules" / "evil-dep" / "package.json").is_file(), (
        "npm ci must still install the dependency tree"
    )
    return marker


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm not installed")
def test_update_build_npm_ci_flags_block_install_scripts(
    tmp_path: Path,
) -> None:
    """``npm ci`` with install.sh's exact flags must not run install scripts.

    First reproduces the original bug — a plain ``npm ci`` *does* execute the
    dependency's install script (the keytar hang vector) — then proves the
    flags parsed from ``install.sh`` prevent it.
    """
    # Reproduce the bug: without install.sh's flags the lifecycle script runs.
    marker = _run_npm_ci([], tmp_path, "buggy")
    assert marker.is_file(), (
        "fixture self-check: plain npm ci should run the install script"
    )

    # The fix: install.sh's flags must prevent the script from running.
    flags = _npm_ci_flags(INSTALL_SCRIPT)
    marker = _run_npm_ci(flags, tmp_path, "fixed")
    assert not marker.exists(), (
        "npm ci with install.sh's flags executed a dependency install "
        "script — the Update button can hang again on keytar's "
        "prebuild-install/node-gyp"
    )
