# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.cli_steering`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module.
"""

import sys

from kiss.ui.cli import cli_steering as _impl
from kiss.ui.cli.cli_steering import (  # noqa: F401
    _BOX_H,
    _NEWLINE_AFTER_ESC,
    AnchoredRepl,
    SteeringSession,
    _box_body_h,
    _box_h_for,
    _box_top_row,
    _InputBox,
    _normalize_candidates,
    _partial_suffix_len,
    _StdoutProxy,
    run_with_steering,
    supports_steering,
)

sys.modules[__name__] = _impl
