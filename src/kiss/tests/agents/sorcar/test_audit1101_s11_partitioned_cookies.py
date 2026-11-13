# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A switch of browser tells partitioned (CHIPS) cookies apart.

Chromium keeps one cookie per name, domain, path *and* partition (the
top-level site a third-party cookie was set under).  The session carried
between the headless browser and the Browser tab identified a cookie by
name, domain and path only, so with the same ``session`` cookie of an
identity provider alive under site B and logged out under site A, B
counted as proof that A was still alive: the dead A cookie was never
deleted on the other side and the logout was undone when browsing
returned there.  The deletion itself (``BrowserContext.clear_cookies``)
knew no partition either, so once the key told them apart it would have
taken every partition's cookie with it, including one the destination
had on its own.

The identity now includes the partition, and a dead cookie is deleted
through CDP with its exact partition key.  Real headless Chromium, two
real contexts, the real capture/restore of ``WebUseTool``; no doubles.
"""

from __future__ import annotations

import atexit
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar.web_use_tool import WebUseTool


def _cookie(
    value: str, partition: str | None = None, cross_site_ancestor: bool | None = None,
) -> dict[str, Any]:
    cookie: dict[str, Any] = {
        "name": "session", "value": value, "domain": "idp.example", "path": "/",
        "secure": True, "sameSite": "None",
    }
    if partition is not None:
        cookie["partitionKey"] = partition
    if cross_site_ancestor is not None:
        cookie["_crHasCrossSiteAncestor"] = cross_site_ancestor
    return cookie


def _values(context: Any) -> set[tuple[str, str | None]]:
    return {(c["value"], c.get("partitionKey")) for c in context.cookies()}


@pytest.fixture
def browser() -> Iterator[Any]:
    with web_stealth.playwright_api().sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        try:
            yield b
        finally:
            b.close()


@pytest.fixture
def tool() -> Iterator[WebUseTool]:
    t = WebUseTool(ephemeral=True)
    try:
        yield t
    finally:
        t._context = t._page = None  # the contexts are the test's, closed with its browser
        atexit.unregister(t.close)


def _use(tool: WebUseTool, context: Any) -> None:
    """Point *tool* at *context* as if it were the browser it just launched."""
    tool._context = context
    tool._page = context.pages[0]


def test_a_logout_in_one_partition_is_carried_over_and_the_rest_is_kept(
    browser: Any, tool: WebUseTool
) -> None:
    """Two same-name cookies in partitions A and B: A dies, B survives, strangers stay."""
    source, destination = browser.new_context(), browser.new_context()
    source.new_page()
    destination.new_page()
    source.add_cookies([_cookie("A", "https://site-a.example"), _cookie("B", "https://site-b.example")])
    # The destination also holds a cookie of a partition the source never
    # had, and an unpartitioned one; neither is the switch's business.
    destination.add_cookies([
        _cookie("C", "https://site-c.example"), _cookie("U"),
    ])
    # First switch: both partitions are carried into the destination.
    _use(tool, source)
    session = tool._capture_session()
    _use(tool, destination)
    tool._restore_session(session)
    assert _values(destination) == {
        ("A", "https://site-a.example"), ("B", "https://site-b.example"),
        ("C", "https://site-c.example"), ("U", None),
    }
    # Logged out under site A in the source: partition A's cookie is gone there.
    source.clear_cookies()
    source.add_cookies([_cookie("B", "https://site-b.example")])
    # Switch back: only A dies in the destination.
    _use(tool, source)
    session = tool._capture_session()
    assert [c["value"] for c in session.cookies or []] == ["B"]
    _use(tool, destination)
    tool._restore_session(session)
    assert _values(destination) == {
        ("B", "https://site-b.example"), ("C", "https://site-c.example"), ("U", None),
    }


def test_an_unpartitioned_logout_leaves_the_partitioned_cookies_alone(
    browser: Any, tool: WebUseTool
) -> None:
    """The unpartitioned ``session`` cookie dies; the partitioned ones of the same name stay."""
    source, destination = browser.new_context(), browser.new_context()
    source.new_page()
    destination.new_page()
    source.add_cookies([_cookie("U"), _cookie("A", "https://site-a.example")])
    _use(tool, source)
    session = tool._capture_session()
    _use(tool, destination)
    tool._restore_session(session)
    assert _values(destination) == {("U", None), ("A", "https://site-a.example")}
    # Logged out of the first-party session in the source; A lives on.
    source.clear_cookies()
    source.add_cookies([_cookie("A", "https://site-a.example")])
    _use(tool, source)
    session = tool._capture_session()
    _use(tool, destination)
    tool._restore_session(session)
    assert _values(destination) == {("A", "https://site-a.example")}


def _ancestor_values(context: Any) -> set[tuple[str, bool]]:
    return {(c["value"], c["_crHasCrossSiteAncestor"]) for c in context.cookies()}


def test_a_logout_in_one_ancestor_variant_of_a_partition_is_carried_over(
    browser: Any, tool: WebUseTool
) -> None:
    """One top-level site, two cookies: with and without a cross-site ancestor; one logs out."""
    source, destination = browser.new_context(), browser.new_context()
    source.new_page()
    destination.new_page()
    site = "https://site.example"
    source.add_cookies([_cookie("same-site", site, False), _cookie("cross-site", site, True)])
    _use(tool, source)
    session = tool._capture_session()
    _use(tool, destination)
    tool._restore_session(session)
    assert _ancestor_values(destination) == {("same-site", False), ("cross-site", True)}
    # Logged out where the site embeds the provider directly: only the
    # cookie without a cross-site ancestor dies; the other lives on.
    source.clear_cookies()
    source.add_cookies([_cookie("cross-site", site, True)])
    _use(tool, source)
    session = tool._capture_session()
    assert [c["value"] for c in session.cookies or []] == ["cross-site"]
    _use(tool, destination)
    tool._restore_session(session)
    assert _ancestor_values(destination) == {("cross-site", True)}


def test_a_jar_that_could_not_be_read_deletes_nothing(browser: Any, tool: WebUseTool) -> None:
    """A source whose cookies cannot be read (it closed under the switch here)
    carries no jar at all, which is not an empty one: what the previous
    switch carried into the destination stays there."""
    source, destination = browser.new_context(), browser.new_context()
    source.new_page()
    destination.new_page()
    source.add_cookies([_cookie("A", "https://site-a.example")])
    _use(tool, source)
    session = tool._capture_session()
    _use(tool, destination)
    tool._restore_session(session)
    assert _values(destination) == {("A", "https://site-a.example")}
    # Switch back, the source gone by the time its cookies are read.
    _use(tool, source)
    source.close()
    session = tool._capture_session()
    assert session.cookies is None
    _use(tool, destination)
    tool._restore_session(session)
    assert _values(destination) == {("A", "https://site-a.example")}
    assert tool._carried_cookies == []


def test_a_failed_import_carries_nothing(browser: Any, tool: WebUseTool) -> None:
    """Cookies the destination never took are not carried: the next switch deletes none.

    The destination closed under the first switch, so the import failed;
    the browser taken over next has an empty jar of its own, which read
    as a logout and deleted the original login on the way back."""
    source, closing, reopened = (browser.new_context() for _ in range(3))
    for context in (source, closing, reopened):
        context.new_page()
    source.add_cookies([_cookie("A")])
    _use(tool, source)
    session = tool._capture_session()
    _use(tool, closing)
    closing.close()
    tool._restore_session(session)
    _use(tool, reopened)
    session = tool._capture_session()
    assert session.cookies == []
    _use(tool, source)
    tool._restore_session(session)
    assert _values(source) == {("A", None)}


if __name__ == "__main__":
    pytest.main([__file__])
