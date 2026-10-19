# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``<meta>`` / ``<link>`` are void elements: HTMLParser fires no end tag for
them, so counting them as skip-depth openers swallowed every text node that
followed (finding conc1018 A2-F3)."""

from kiss.core.html_render import html_to_rich


def test_meta_in_head_does_not_swallow_body_text() -> None:
    html = (
        '<html><head><meta charset="utf-8"><title>t</title></head><body><p>hello</p></body></html>'
    )
    assert html_to_rich(html).plain.strip() == "hello"


def test_link_between_paragraphs_does_not_swallow_following_text() -> None:
    assert html_to_rich('<p>a</p><link rel="x"><p>b</p>').plain.split() == ["a", "b"]


def test_self_closing_meta_still_renders_following_text() -> None:
    assert html_to_rich('<meta charset="utf-8"/><p>hello</p>').plain.strip() == "hello"


def test_head_title_and_script_are_still_skipped() -> None:
    html = (
        "<html><head><title>skip me</title><script>var x = 1;</script>"
        "<style>p {}</style></head><body><p>kept</p></body></html>"
    )
    assert html_to_rich(html).plain.strip() == "kept"
