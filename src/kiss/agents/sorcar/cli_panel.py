# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compat alias: implementation moved to :mod:`kiss.ui.cli.cli_panel`.

The ``from`` imports below are static re-exports so type checkers see
the public surface; at runtime the ``sys.modules`` assignment makes
this module name an alias of the real module.
"""

import sys

from kiss.ui.cli import cli_panel as _impl
from kiss.ui.cli.cli_panel import (  # noqa: F401
    ASK_TITLE,
    BOLD,
    CSI_U_ENTER,
    CYAN,
    DIM,
    IDLE_TITLE,
    KEYBOARD_PROTO_DISABLE,
    KEYBOARD_PROTO_ENABLE,
    MODIFY_OTHER_KEYS_ENTER,
    ORANGE,
    PLACEHOLDER,
    PROMPT_MARKER,
    QUESTION_FMT,
    QUEUED_FMT,
    QUEUED_STATUS_FMT,
    RESET,
    STEER_TITLE,
    _clip_pad,
    _term_size,
    body_cursor_col,
    clip_buf,
    display_width,
    menu_row,
    panel_body,
    panel_bottom,
    panel_cols,
    panel_top,
    visible_line_window,
)

sys.modules[__name__] = _impl
