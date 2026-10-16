# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``install.sh``'s post-install hooks (step [5/5]).

Every executable in ``$KISS_HOME/post-install.d/`` runs once the app is
on disk and before the daemon restart, in name order, with ``KISS_HOME``
and ``KISS_APP_DIR`` set and no stdin.  A white-label layer
(SeamlessLabs' s10s) patches the installed files from such a hook, so
the daemon serves the brand from its first start.  A hook that fails is
reported as a warning and the install still completes; a file without
the executable bit is skipped; a missing hook directory is fine.

Each test extracts the hook loop verbatim from ``install.sh`` and runs
it under real bash in a sandboxed ``$KISS_HOME`` -- no mocks, no patches.
"""

from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

pytestmark = posix_only("install.sh is a bash script run under real bash")

_INSTALL_SH = Path(__file__).resolve().parents[4] / "install.sh"


def _extract_hook_loop() -> str:
    """The ``for hook in "$KISS_HOME_DIR"/post-install.d/* ... done`` loop."""
    text = _INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(
        r'^( *)for hook in "\$KISS_HOME_DIR"/post-install\.d/\*; do\n.*?^\1done\n',
        text,
        re.S | re.M,
    )
    assert match, "install.sh lost its post-install hook loop"
    return textwrap.dedent(match.group(0))


def _run_hooks(kiss_home: Path, app_dir: Path) -> subprocess.CompletedProcess[str]:
    script = (
        "set -u\n"
        f"KISS_HOME_DIR={kiss_home.as_posix()!r}\n"
        f"APP_DIR={app_dir.as_posix()!r}\n"
        + _extract_hook_loop()
        + "echo HOOKS-DONE\n"
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "HOME": kiss_home.parent.as_posix()},
        check=False,
    )


def test_hooks_run_in_order_with_env_no_stdin_and_tolerate_failure(
    tmp_path: Path,
) -> None:
    kiss_home = tmp_path / "custom-kiss-home"
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    hooks = kiss_home / "post-install.d"
    hooks.mkdir(parents=True)
    record = tmp_path / "hook-record.txt"
    for name in ("10-brand", "30-second"):
        (hooks / name).write_text(
            textwrap.dedent(
                f"""\
                #!/bin/bash
                stdin=closed
                read -r -t 1 _line && stdin=open
                printf '%s KISS_HOME=%s KISS_APP_DIR=%s stdin=%s\\n' \\
                    {name} "$KISS_HOME" "$KISS_APP_DIR" "$stdin" >> {record.as_posix()!r}
                """
            ),
            encoding="utf-8",
        )
        (hooks / name).chmod(0o755)
    (hooks / "20-broken").write_text("#!/bin/bash\nexit 3\n", encoding="utf-8")
    (hooks / "20-broken").chmod(0o755)
    (hooks / "README").write_text("#!/bin/bash\nexit 7\n", encoding="utf-8")

    result = _run_hooks(kiss_home, app_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith("HOOKS-DONE\n"), "a failing hook must not stop the install"
    assert record.read_text(encoding="utf-8") == (
        f"10-brand KISS_HOME={kiss_home.as_posix()} "
        f"KISS_APP_DIR={app_dir.as_posix()} stdin=closed\n"
        f"30-second KISS_HOME={kiss_home.as_posix()} "
        f"KISS_APP_DIR={app_dir.as_posix()} stdin=closed\n"
    ), "hooks run in name order, see $KISS_HOME / $KISS_APP_DIR and get no stdin"
    assert (
        f"WARNING: post-install hook {hooks.as_posix()}/20-broken exited with status 3"
        in result.stdout
    )
    assert result.stdout.index("Running post-install hook") < result.stdout.index(
        "20-broken exited"
    )
    assert "README" not in result.stdout, "a file without the executable bit is not a hook"


def test_missing_hook_directory_is_not_an_error(tmp_path: Path) -> None:
    kiss_home = tmp_path / "home-without-hooks"
    kiss_home.mkdir()
    result = _run_hooks(kiss_home, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "HOOKS-DONE\n"


def test_install_sh_documents_the_hooks_in_step_5_5() -> None:
    text = _INSTALL_SH.read_text(encoding="utf-8")
    assert 'report_step "[5/5]' in text
    assert text.index('report_step "[5/5]') < text.index(
        'for hook in "$KISS_HOME_DIR"/post-install.d/*'
    ), "the hooks belong to the finishing step"
    assert text.index('for hook in "$KISS_HOME_DIR"/post-install.d/*') < text.index(
        "start_kiss_web_daemon \"$APP_DIR\""
    ), "hooks run before the daemon restart"


if __name__ == "__main__":
    pytest.main([__file__])
