#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Measure which bot-protected sites the KISS browser tool gets through.

Drives the real :class:`kiss.agents.sorcar.web_use_tool.WebUseTool` (the
engine, display and input helpers of :mod:`kiss.agents.sorcar.web_stealth`)
at one page per host named in ``web_stealth.py`` and prints, per URL, the
verdict (``OK``, ``BLOCKED(<vendor>)`` or ``ERROR``), the seconds the
load took, the final URL and the page title.  The last line is the score.
Exit status 1 when any page was blocked, so a scheduled run can tell.

Usage::

    uv run python scripts/web_stealth_bench.py            # the built-in list
    uv run python scripts/web_stealth_bench.py URL [URL …] # your own pages
    uv run python scripts/web_stealth_bench.py --headless  # real headless mode

``--headless`` skips the Xvfb display so Chromium runs in its real
headless mode, as it does on macOS and Windows where there is no Xvfb:
that is how the blocks a Mac daemon reports are reproduced on a Linux box.

A fresh (ephemeral) browser profile is used, so cookies from earlier
cleared challenges do not flatter the result.
"""

from __future__ import annotations

import sys
import time

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar.web_use_tool import WebUseTool

# One page per host named in web_stealth.py's analysis, deep links where
# the shallow page is not protected (walmart product pages are, the home
# page is not).
URLS = [
    "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python",
    "https://dl.acm.org/doi/10.1145/3290605.3300233",
    "https://www.npmjs.com/package/express",
    "https://requests.readthedocs.io/en/latest/",
    "https://www.congress.gov/bill/118th-congress/house-bill/1",
    "https://medium.com/@netflixtechblog",
    "https://www.carmax.com/cars/all",
    "https://www.costco.com/laptops.html",
    "https://dblp.org/search?q=koushik+sen",
    "https://wiki.archlinux.org/title/Xorg",
    "https://www.cato.org/",
    "https://www.walmart.com/ip/Apple-AirPods-Pro-2/5689919121",
    "https://www.fiverr.com/categories/programming-tech",
    "https://www.google.com/search?q=koushik+sen+berkeley",
    # Cloudflare managed challenges (and one Akamai block) that stopped
    # the macOS daemon's real-headless browser in Oct 2026.
    "https://www.sciencedirect.com/science/article/pii/S0379711201000571",
    "https://patents.justia.com/patent/6064064",
    "https://worldwide.espacenet.com/publicationDetails/biblio?CC=US&NR=6064064",
    "https://www.mdpi.com/1999-4893/7/4/523",
]


def verdict(tool: WebUseTool, result: str) -> str:
    """Classify what ``go_to_url`` returned for the page now showing.

    Args:
        tool: The tool whose page was just navigated.
        result: The text ``go_to_url`` returned.

    Returns:
        ``"OK"``, ``"BLOCKED(<vendor>) <note>"`` or ``"ERROR <message>"``.
    """
    first = result.splitlines()[0] if result else ""
    if result.startswith("Error"):
        return "ERROR " + first[:120]
    try:
        # Bounded read: a renderer stuck in a busy loop raises instead of
        # hanging the whole run.
        title, body = tool._read_page(
            tool._page,
            "() => [document.title, document.body ? document.body.innerText.slice(0, 1500) : '']",
        )
    except Exception as exc:  # noqa: BLE001 — any read failure is an ERROR row
        return f"ERROR page read failed: {type(exc).__name__}"
    vendor = web_stealth.challenge_vendor(str(title), str(body))
    if vendor or first.startswith("Note:"):
        return f"BLOCKED({vendor}) {first[:160]}"
    return "OK"


def main() -> int:
    """Load every URL, print the table and return the exit status."""
    args = sys.argv[1:]
    if "--headless" in args:
        args.remove("--headless")
        # The tool asks for a display through this one hook; "none" is
        # what it gets on a machine without Xvfb.
        web_stealth.virtual_display = lambda: None  # type: ignore[assignment]
        print("real headless mode (no Xvfb display)")
    urls = args or URLS
    tool = WebUseTool(ephemeral=True, headless=True)
    loaded = 0
    try:
        for url in urls:
            started = time.monotonic()
            result = tool.go_to_url(url)
            status = verdict(tool, result)
            loaded += status == "OK"
            # ``page.url`` is local state; the title comes from the tree's
            # first line, so a hung renderer cannot stall the report.
            print(
                f"{status:40s} {time.monotonic() - started:5.1f}s {url}\n"
                f"    -> {tool._page.url[:100]} {result.splitlines()[0][:100]!r}",
                flush=True,
            )
    finally:
        tool.close()
        web_stealth.stop_virtual_display()
    print(f"\n{loaded}/{len(urls)} loaded")
    return 0 if loaded == len(urls) else 1


if __name__ == "__main__":
    sys.exit(main())
