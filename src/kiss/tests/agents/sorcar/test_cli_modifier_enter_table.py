# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the shared modifier+Enter escape tables.

The canonical modifier+Enter sequence tables (xterm modifyOtherKeys and
kitty/CSI-u forms) and the extended-keyboard-protocol enable/disable
pairs live in :mod:`kiss.agents.sorcar.cli_panel` and are consumed by
both terminal input paths — the idle prompt's key bindings
(:mod:`kiss.agents.sorcar.cli_prompt`) and the mid-task steering box
(:mod:`kiss.agents.sorcar.cli_steering`).  These tests pin the exact
byte values and assert that every consumer sees the identical tables,
so the two input paths can never drift apart.
"""

from __future__ import annotations

import random

from kiss.agents.sorcar import cli_panel, cli_prompt, cli_steering
from kiss.agents.sorcar.cli_panel import (
    CSI_U_ENTER,
    KEYBOARD_PROTO_DISABLE,
    KEYBOARD_PROTO_ENABLE,
    MODIFY_OTHER_KEYS_ENTER,
    PROMPT_MARKER,
    _clip_pad,
    clip_buf,
    display_width,
    visible_line_window,
)


def test_modify_other_keys_enter_exact_bytes() -> None:
    """The xterm modifyOtherKeys table covers modifiers 2..16 exactly."""
    assert MODIFY_OTHER_KEYS_ENTER == tuple(
        f"\x1b[27;{m};13~" for m in range(2, 17)
    )


def test_csi_u_enter_exact_bytes() -> None:
    """The kitty/CSI-u table covers modifiers 2..16 exactly."""
    assert CSI_U_ENTER == tuple(f"\x1b[13;{m}u" for m in range(2, 17))


def test_keyboard_protocol_pairs_exact_bytes() -> None:
    """The enable/disable pairs match the documented escape sequences."""
    assert KEYBOARD_PROTO_ENABLE == "\x1b[>4;2m\x1b[>1u"
    assert KEYBOARD_PROTO_DISABLE == "\x1b[>4;0m\x1b[<u"


def test_cli_prompt_consumes_canonical_table() -> None:
    """cli_prompt's unmap table IS the canonical cli_panel table."""
    assert cli_prompt._MODIFY_OTHER_KEYS_ENTER is MODIFY_OTHER_KEYS_ENTER


def test_cli_steering_after_esc_table_derives_from_canonical() -> None:
    """cli_steering's after-ESC table matches the canonical sequences.

    ``_NEWLINE_AFTER_ESC`` holds the byte suffixes that follow the ESC
    introducer — the canonical full sequences minus their first byte —
    followed by the bare CR / LF Alt+Enter forms, in that order (the
    prefix-startswith match requires long sequences first).
    """
    assert cli_steering._NEWLINE_AFTER_ESC == (
        *(seq[1:] for seq in MODIFY_OTHER_KEYS_ENTER),
        *(seq[1:] for seq in CSI_U_ENTER),
        "\r",
        "\n",
    )


def test_both_input_paths_see_identical_tables() -> None:
    """The prompt and steering consumers agree on every sequence."""
    steering_csis = {
        "\x1b" + suffix
        for suffix in cli_steering._NEWLINE_AFTER_ESC
        if suffix not in ("\r", "\n")
    }
    prompt_csis = set(cli_prompt._MODIFY_OTHER_KEYS_ENTER) | set(CSI_U_ENTER)
    assert steering_csis == prompt_csis


def _random_text(rng: random.Random, length: int) -> str:
    """Return a random buffer mixing narrow, wide, and control chars."""
    pool = (
        "abcdefghijklmnopqrstuvwxyz0123456789 "
        "漢字テスト🙂🎉"
        "\n\t"
        "éüñ"
    )
    return "".join(rng.choice(pool) for _ in range(length))


def test_clip_buf_equals_visible_line_window_at_end() -> None:
    """clip_buf(buf, cols) == visible_line_window(buf, cols, len(buf))[0].

    Property test over random buffers (narrow / wide CJK / emoji /
    newline / tab characters) and panel widths: ``clip_buf`` is defined
    in terms of ``visible_line_window`` with the cursor at the end, so
    the tail-clip contract must hold for every input.
    """
    rng = random.Random(20260721)
    for _ in range(300):
        buf = _random_text(rng, rng.randint(0, 60))
        cols = rng.randint(10, 120)
        shown = clip_buf(buf, cols)
        expected, _ = visible_line_window(buf, cols, len(buf))
        assert shown == expected
        sanitized = buf.replace("\n", "⏎").replace("\t", " ")
        avail = (cols - 4) - display_width(PROMPT_MARKER)
        # The visible slice is always a suffix of the sanitised text
        # and never exceeds the available room.
        assert sanitized.endswith(shown)
        assert display_width(shown) <= max(avail, 0)
        # When the whole text fits it is shown in full.
        if display_width(sanitized) <= avail:
            assert shown == sanitized


def test_clip_pad_pads_to_exact_width() -> None:
    """_clip_pad returns exactly *width* display columns for any input."""
    rng = random.Random(20260722)
    for _ in range(300):
        text = _random_text(rng, rng.randint(0, 40))
        width = rng.randint(0, 60)
        padded = _clip_pad(text, width)
        assert display_width(padded) == width


def test_shared_ask_and_queued_literals() -> None:
    """The ask/queued literals render the exact historical bytes."""
    assert cli_panel.ASK_TITLE == " answer the question above, then Enter "
    assert cli_panel.QUESTION_FMT.format(question="Q?") == (
        "\n\x1b[33m? Q?\x1b[0m\n"
    )
    assert cli_panel.QUEUED_FMT.format(text="do it") == (
        "\x1b[2m▸ queued: do it\x1b[0m\n"
    )
    assert cli_panel.QUEUED_STATUS_FMT.format(n=3) == " queued: 3 "


def test_steer_title_matches_idle_title_style() -> None:
    """STEER_TITLE uses IDLE_TITLE's middle-dot separators and wording."""
    assert "·" in cli_panel.STEER_TITLE
    assert " . " not in cli_panel.STEER_TITLE
    assert "type a task" in cli_panel.STEER_TITLE
    assert not cli_panel.STEER_TITLE.startswith(" ")
    # Both titles advertise the same completion / newline hints.
    for hint in ("Tab to autocomplete", "Alt+Enter/Shift+Enter for newline"):
        assert hint in cli_panel.IDLE_TITLE
        assert hint in cli_panel.STEER_TITLE
