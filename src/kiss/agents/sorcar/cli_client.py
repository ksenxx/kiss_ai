# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.cli_client`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module (same module object, so
monkeypatching through either name is equivalent).
"""

import sys

from kiss.ui.cli import cli_client as _impl
from kiss.ui.cli.cli_client import (  # noqa: F401
    CliClient,
    _drain_queue,
    _EventDispatcher,
    _handle_client_slash,
    _print_elapsed,
    _request_cli_info,
    _request_models,
    _run_anchored_client,
    _run_repl_loop,
    _sock_path,
    _submit_task,
    _submit_task_anchored,
    _wait_for_socket,
    run_client,
)

sys.modules[__name__] = _impl
