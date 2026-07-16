# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.mcp_cli`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module.
"""

import sys

from kiss.ui.cli import mcp_cli as _impl
from kiss.ui.cli.mcp_cli import (  # noqa: F401
    _OAuthCallbackServer,
    run_mcp_cli,
)

sys.modules[__name__] = _impl
