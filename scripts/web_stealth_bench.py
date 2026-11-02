#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Measure which bot-protected sites the KISS browser tool gets through.

Drives the real :class:`kiss.agents.sorcar.web_use_tool.WebUseTool` (the
engine, display and input helpers of :mod:`kiss.agents.sorcar.web_stealth`)
at one page per host named in ``web_stealth.py`` and prints, per URL, the
verdict (``OK``, ``FALLBACK(<vendor>)``, ``BLOCKED(<vendor>)`` or
``ERROR``), the seconds the load took, the final URL and the page title.
The last line is the score.  Exit status 1 when any page was blocked or
errored, so a scheduled run can tell.  A page in :data:`KNOWN_PUZZLES` is
expected to answer with a particular vendor's CAPTCHA puzzle that no
browser setting clears (the site rates the network's IP address); the
tool then recognises the puzzle and opens the same paper or search on
Crossref (``FALLBACK(<vendor>)``), which counts as loaded since the agent
gets the information it asked for -- or the content page itself when the
site's policy has relaxed (``OK``).  Anything else on such a page (another
vendor's block, an AWS WAF page -- the token challenge that never cleared
or the puzzle -- without a Crossref answer that names the paper, the
puzzle page unrecognised and returned as content) is a block like
elsewhere.

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
    # AWS WAF: a silent token challenge on every page, and since 30 Oct
    # 2026 the CAPTCHA puzzle on paper and search pages (KNOWN_PUZZLES).
    "https://www.semanticscholar.org/paper/Enhancing-the-performance-of-kNN-for-glass-dataset"
    "-Bhowmick-Saha/f5d7aaf00f43a18cf20fba144cdf88039a502262",
]

# URL -> (the vendor label of the expected puzzle, or a tuple of the labels
# the site may answer with; a fragment of the content page's title, also
# expected in the Crossref page's title).  Pages that show a CAPTCHA puzzle
# to this network's IP address even in an un-automated Chrome with a fresh
# profile (checked 31 Oct and 1 Nov 2026 for semanticscholar.org): the pass
# is that puzzle being recognised and the same paper opened on Crossref, or
# the content page when the site's policy has relaxed.  AWS WAF's blank
# token challenge comes first and, on a loaded machine, can outlast its
# 12 s wait before handing over to the puzzle; Crossref answers either.
KNOWN_PUZZLES: dict[str, tuple[str | tuple[str, ...], str]] = {
    URLS[-1]: (
        (web_stealth.AWS_WAF_CAPTCHA, web_stealth.AWS_WAF_CHALLENGE),
        "Enhancing the performance of kNN",
    ),
}


def verdict(tool: WebUseTool, url: str, result: str) -> str:
    """Classify what ``go_to_url`` returned for the page now showing.

    Args:
        tool: The tool whose page was just navigated.
        url: The URL that was loaded.
        result: The text ``go_to_url`` returned.

    Returns:
        ``"OK"``, ``"FALLBACK(<vendor>) <note>"`` (a :data:`KNOWN_PUZZLES`
        page's puzzle recognised and the paper or search opened on
        Crossref), ``"BLOCKED(<vendor>) <note>"`` or ``"ERROR <message>"``.
    """
    first = result.splitlines()[0] if result else ""
    if result.startswith("Error"):
        return "ERROR " + first[:120]
    try:
        # Bounded read: a renderer stuck in a busy loop raises instead of
        # hanging the whole run.
        title, body, scripts = tool._read_page(
            tool._page,
            "() => [document.title, document.body ? document.body.innerText.slice(0, 1500) : '',"
            " Array.from(document.scripts, s => s.src).filter(Boolean).join(' ').slice(0, 4000)]",
        )
    except Exception as exc:  # noqa: BLE001 — any read failure is an ERROR row
        return f"ERROR page read failed: {type(exc).__name__}"
    vendor = web_stealth.challenge_vendor(str(title), str(body), str(scripts))
    if url in KNOWN_PUZZLES:
        expected, content_title = KNOWN_PUZZLES[url]
        labels = (expected,) if isinstance(expected, str) else expected
        named = next((label for label in labels if f"({label})" in first), None)
        has_content = content_title.lower() in str(title).lower()
        if vendor is None and first.startswith("Note:") and named and "Crossref" in first:
            # The Crossref page is titled after the words searched for, so
            # the paper is looked for among the works it lists.
            try:
                works = tool._read_page(
                    tool._page,
                    "() => Array.from(document.querySelectorAll('article h2'),"
                    " h => h.textContent).slice(0, 20)",
                )
            except Exception as exc:  # noqa: BLE001 — any read failure is an ERROR row
                return f"ERROR page read failed: {type(exc).__name__}"
            if any(content_title.lower() in str(work).lower() for work in works):
                return f"FALLBACK({named}) {first[:160]}"
            return f"BLOCKED({named}) Crossref lists no work titled {content_title!r}"
        if vendor is None and not first.startswith("Note:") and not has_content:
            return f"BLOCKED({vendor}) unrecognised page titled {str(title)[:60]!r}"
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
    fallbacks = 0
    try:
        for url in urls:
            started = time.monotonic()
            result = tool.go_to_url(url)
            status = verdict(tool, url, result)
            loaded += status == "OK"
            fallbacks += status.startswith("FALLBACK(")
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
    print(f"\n{loaded}/{len(urls)} loaded, {fallbacks} answered by Crossref for a known puzzle")
    return 0 if loaded + fallbacks == len(urls) else 1


if __name__ == "__main__":
    sys.exit(main())
