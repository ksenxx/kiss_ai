# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Both HTML converters drop a ``<head>`` subtree whole (audit 2026-10-21, D-F5).

The terminal renderer used to skip only the *text* inside ``<head>``,
``<title>``, ``<script>`` and ``<style>``; a block tag inside them still
emitted its newlines, bullets or rule, so ``a<head><h1>x</h1></head>b``
rendered with a blank line the Markdown converter did not produce.  The
two parsers now share the subtree-skipping base class and agree.
"""

from __future__ import annotations

import pytest

from kiss.core.html_render import html_to_rich
from kiss.core.html_to_markdown import html_to_markdown

_SKIPPED = [
    "<head><h1>x</h1></head>",
    "<head><title>t</title><h1>x</h1></head>",
    "<head><p>x</p><ul><li>y</li></ul><hr><br></head>",
    "<script><p>x</p></script><style><h2>y</h2></style><title><p>z</p></title>",
]


@pytest.mark.parametrize("html", _SKIPPED)
def test_a_skipped_subtree_renders_nothing(html: str) -> None:
    assert html_to_rich(html).plain == ""
    assert html_to_markdown(html) == ""


@pytest.mark.parametrize("html", _SKIPPED)
def test_a_skipped_subtree_leaves_no_trace_between_text(html: str) -> None:
    for before, after in (
        ("<b>a</b>", "b"),
        ("<p>a</p>", "<p>b</p>"),
        ("a", "<ul><li>b</li></ul>"),
    ):
        with_subtree = f"{before}{html}{after}"
        without = f"{before}{after}"
        assert html_to_rich(with_subtree).plain == html_to_rich(without).plain, with_subtree
        assert html_to_markdown(with_subtree) == html_to_markdown(without), with_subtree


def test_unbalanced_end_tags_never_skip_the_rest() -> None:
    html = "</head></script><p>kept</p>"
    assert html_to_rich(html).plain == "kept"
    assert html_to_markdown(html) == "kept"
