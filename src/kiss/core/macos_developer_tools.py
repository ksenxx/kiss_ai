# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Keep macOS from offering to install Apple's developer tools.

A Mac that has neither Xcode nor the Command Line Tools still has
``/usr/bin/git``, ``/usr/bin/python3``, ``make``, ``cc`` and some seventy
other commands.  They are stubs: running one opens the system dialog
'The "git" command requires the command line developer tools. Would you
like to install the tools now?' and exits 1.  KISS Sorcar runs git at
the start of every task, and an agent's shell commands run git, python3
and make all the time, so on such a Mac the dialog came up again and
again.

The stubs look their tool up in the active developer directory, which
the ``DEVELOPER_DIR`` environment variable overrides (xcode-select(1)).
Pointed at a directory that does not exist, a stub prints "xcrun: error:
missing DEVELOPER_DIR path" and exits 1 at once, with no dialog.  Every
caller already handles a git that fails.

:func:`silence_developer_tools_prompt` exports that variable when the
tools are missing.  The directory it names is the one the Command Line
Tools install themselves into, so the stubs start working, in a daemon
that is already running too, as soon as the user installs them.  (After
installing Xcode instead, restart the daemon.)  ``install.sh`` and
``scripts/install.sh`` do the same for their own git calls.
"""

import os
import subprocess
import sys

COMMAND_LINE_TOOLS_DIR = "/Library/Developer/CommandLineTools"


def developer_tools_missing() -> bool:
    """Tell whether this is a Mac with neither Xcode nor the Command Line Tools.

    ``xcode-select -p`` prints the active developer directory and fails
    when there is none.  It repeats a ``DEVELOPER_DIR`` it inherits
    without looking at it, hence the check that the directory exists.

    Returns:
        True on macOS when no developer directory is selected or the
        selected one does not exist; False on other systems and when
        ``xcode-select`` cannot be run.
    """
    if sys.platform != "darwin":
        return False
    try:
        selected = subprocess.run(
            ["xcode-select", "-p"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return selected.returncode != 0 or not os.path.isdir(selected.stdout.strip())


def silence_developer_tools_prompt() -> None:
    """Make the developer-tool stubs of a Mac without the tools fail quietly.

    Exports ``DEVELOPER_DIR`` for this process and every process it
    starts when :func:`developer_tools_missing`; does nothing otherwise,
    and nothing in a process that inherited the variable from a parent
    that already did this.
    """
    if os.environ.get("DEVELOPER_DIR") == COMMAND_LINE_TOOLS_DIR:
        return
    if developer_tools_missing():
        os.environ["DEVELOPER_DIR"] = COMMAND_LINE_TOOLS_DIR
