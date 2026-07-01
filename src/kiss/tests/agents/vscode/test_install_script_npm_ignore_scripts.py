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
import os
import re
import shutil
import signal
import subprocess
import tarfile
import textwrap
import time
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


def _extract_signal_helpers(script: Path) -> str:
    """Return the bash source defining the signal-handling helpers.

    The chunk covers ``LAST_SIGNAL_TS``, ``CURRENT_CMD_PID``,
    ``handle_interrupt``, the ``trap`` installation and
    ``run_with_heartbeat`` — everything the regression test needs to
    exercise without sourcing all of ``install.sh`` (which would run the
    whole installer).
    """
    src = script.read_text(encoding="utf-8")
    start = src.index("LAST_SIGNAL_TS=0")
    end = src.index('OS="$(uname -s)"')
    chunk = src[start:end]
    # ``return $rc`` inside ``run_with_heartbeat`` requires the function
    # body — included in the slice — so the chunk is self-contained.
    return chunk


def _wait_for_log_text(log: Path, needle: str, timeout: float = 10.0) -> str:
    """Re-read *log* until *needle* appears or *timeout* elapses.

    The harness writes its log through an inherited file descriptor; under
    heavy parallel test load the final trap diagnostics can land a moment
    after the harness process itself has been reaped.  Polling instead of
    a single post-``wait`` read removes that flake window.  Returns the
    last-read log text either way so assertion messages stay informative.
    """
    deadline = time.time() + timeout
    text = ""
    while time.time() < deadline:
        text = (
            log.read_text(encoding="utf-8", errors="replace")
            if log.exists()
            else ""
        )
        if needle in text:
            return text
        time.sleep(0.1)
    return text


def test_run_with_heartbeat_survives_stray_sigint(tmp_path: Path) -> None:
    """A single stray SIGINT must NOT kill the wrapped command.

    Regression: the user reported '^C' appearing during
    "Copying source files..." even though they did not press Ctrl+C, and
    install.sh aborted right there.  Root cause: SIGINT delivered to
    install.sh's terminal process group killed ``npm``/``copy-kiss.sh``
    children even though install.sh's own SIGINT trap ignored it.  The
    wrapped child's death made ``wait`` in ``run_with_heartbeat`` return
    non-zero and ``set -e`` aborted the install.

    Fix: ``run_with_heartbeat`` now spawns the wrapped command inside a
    subshell that sets ``trap '' INT TERM`` and ``exec``s the binary, so
    SIG_IGN is inherited and a stray signal can no longer kill it.

    This end-to-end test extracts the signal helpers from install.sh,
    runs ``run_with_heartbeat`` against a 2-second ``sleep``, sends a
    stray SIGINT to the harness's process group ~0.5 s in, and verifies
    that the sleep completed normally and the harness exited 0.
    """
    helpers = _extract_signal_helpers(INSTALL_SCRIPT)
    marker = tmp_path / "sleep_finished"
    ready = tmp_path / "wrapped_started"
    log = tmp_path / "harness.log"
    # The harness:
    #   * pulls in the extracted helpers verbatim,
    #   * runs ``sleep`` under ``run_with_heartbeat`` with a wide
    #     heartbeat interval so its output doesn't race the test,
    #   * touches a marker file if and only if the wrapped command
    #     exited 0 (i.e. wasn't killed by the stray signal).
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "set -eo pipefail\n"
        'export KISS_HEARTBEAT_INTERVAL=60\n'
        + helpers
        + "\n"
        + textwrap.dedent(
            f"""\
            if run_with_heartbeat "sleep" bash -c ': > "{ready}"; exec sleep 3'; then
                : > "{marker}"
            fi
            """
        ),
        encoding="utf-8",
    )
    harness.chmod(0o755)

    # Redirect harness output to a file rather than a pipe: the heartbeat
    # subshell spawns a ``sleep $HEARTBEAT_INTERVAL`` child that, after a
    # stray SIGINT kills its parent shell, is reparented to init and keeps
    # the inherited stdout fd open until it finishes — which would hold
    # ``proc.communicate()`` open until the heartbeat interval elapses.
    # A file fd does not have that EOF dependency.
    with open(log, "wb") as out:
        proc = subprocess.Popen(
            ["bash", str(harness)],
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(tmp_path),
        )
        try:
            # Wait until the wrapped command is actually running (it writes
            # the ready marker) before delivering SIGINT to the harness's
            # entire process group — the exact condition that previously
            # killed npm/copy-kiss.sh.  A fixed sleep raced bash's trap
            # installation under heavy parallel test load.
            deadline = time.time() + 15.0
            while time.time() < deadline and not ready.exists():
                time.sleep(0.05)
            assert ready.exists(), "wrapped command never started"
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                raise
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    stdout = _wait_for_log_text(log, "Interrupt received but ignored")
    assert proc.returncode == 0, (
        "run_with_heartbeat must NOT abort on a single stray SIGINT — the "
        "harness exited "
        f"{proc.returncode}.  Output:\n{stdout}"
    )
    assert marker.exists(), (
        "sleep was killed by the stray SIGINT — the wrapped command must "
        "inherit SIG_IGN for INT/TERM via the subshell `trap '' INT TERM; "
        f"exec ...` wrapper.  Output:\n{stdout}"
    )
    # And the install.sh-level trap must have printed the "Interrupt
    # received but ignored" diagnostic so the user knows what happened.
    assert "Interrupt received but ignored" in stdout, (
        "the install.sh-level SIGINT trap did not fire — handle_interrupt "
        f"is wired wrong.  Output:\n{stdout}"
    )


def test_run_with_heartbeat_double_sigint_aborts(tmp_path: Path) -> None:
    """A confirmed double-Ctrl+C must still abort the install.

    The protective subshell makes the wrapped command ignore SIGINT, so a
    determined abort needs ``handle_interrupt`` to forcibly kill the
    tracked ``CURRENT_CMD_PID`` on the second signal.  Without this kill
    a user could no longer stop a runaway build at all.
    """
    helpers = _extract_signal_helpers(INSTALL_SCRIPT)
    marker = tmp_path / "should_not_exist"
    ready = tmp_path / "wrapped_started"
    log = tmp_path / "harness.log"
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "set -eo pipefail\n"
        'export KISS_HEARTBEAT_INTERVAL=60\n'
        + helpers
        + "\n"
        + textwrap.dedent(
            f"""\
            run_with_heartbeat "sleep" bash -c ': > "{ready}"; exec sleep 30'
            : > "{marker}"
            """
        ),
        encoding="utf-8",
    )
    harness.chmod(0o755)

    with open(log, "wb") as out:
        proc = subprocess.Popen(
            ["bash", str(harness)],
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(tmp_path),
        )
        try:
            deadline = time.time() + 15.0
            while time.time() < deadline and not ready.exists():
                time.sleep(0.05)
            assert ready.exists(), "wrapped command never started"
            os.killpg(proc.pid, signal.SIGINT)
            # Wait until the FIRST trap has actually executed before
            # sending the second signal: bash runs traps only between
            # commands, so under heavy load two quick SIGINTs can coalesce
            # into a single ``handle_interrupt`` invocation — the
            # second-signal abort branch would then never run and the
            # wrapped ``sleep 30`` would outlive the wait below.  The
            # 3-second double-interrupt window starts at the first trap's
            # execution, so detect-then-send stays safely inside it.
            first = _wait_for_log_text(log, "Interrupt received but ignored")
            assert "Interrupt received but ignored" in first, (
                f"first SIGINT trap never fired.  Output:\n{first}"
            )
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                raise
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    stdout = _wait_for_log_text(log, "Second interrupt received")
    assert proc.returncode == 130, (
        "double-Ctrl+C must abort with exit 130; got "
        f"{proc.returncode}.  Output:\n{stdout}"
    )
    assert not marker.exists(), (
        "the line after run_with_heartbeat ran — the abort did not stop "
        f"the script.  Output:\n{stdout}"
    )
    assert "Second interrupt received" in stdout, (
        "the second-signal branch of handle_interrupt did not run — its "
        f"diagnostic is missing.  Output:\n{stdout}"
    )


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
