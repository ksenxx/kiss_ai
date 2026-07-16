# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.cli_daemon_bridge`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module.
"""

import sys

from kiss.ui.cli import cli_daemon_bridge as _impl
from kiss.ui.cli.cli_daemon_bridge import (  # noqa: F401
    _LOCK,
    _WRITER,
    _WRITER_PATH,
    _connect,
    _send_envelope,
    _sock_path,
    send_cli_task_end,
    send_cli_task_start,
    send_event,
)

sys.modules[__name__] = _impl
