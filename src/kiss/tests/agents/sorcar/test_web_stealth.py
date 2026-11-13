# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the browser tool looks and behaves like a person's Chrome.

Bot-protection vendors blocked 95 hosts in the task history.  These tests
drive a REAL Chromium (no mocks) against local HTTP servers and check the
properties :mod:`kiss.agents.sorcar.web_stealth` provides:

1. Engine: Patchright drives the browser; the bundled Chromium (or Chrome)
   channel is chosen; on Linux the "no window" default is a headed
   Chromium on a process-wide Xvfb display, and the display is shared,
   restartable and torn down cleanly.
2. Fingerprint: no ``HeadlessChrome`` token, ``navigator.webdriver`` off,
   real window geometry, no emulated timezone / scale factor.
3. Human input: a click arrives after a curved pointer path and a held
   button; typing has an uneven cadence; scrolling wheels in uneven
   notches.  Elements without a box or with an unhittable point still
   get clicked (plain-click fallbacks).
4. Challenge pages: a Cloudflare-style ``cf-mitigated: challenge``
   interstitial that clears is waited out and the real page returned; one
   that stays is reported with a ``Note:``; Turnstile's checkbox is
   ticked; HUMAN / PerimeterX's press-and-hold bar is held for seconds
   (twice for the two-step variant), on its button when the widget's
   frame is wider than the button (fiverr), and a page without the bar is
   reported; an Anubis proof-of-work page gets its longer wait; a Google
   "unusual traffic" page is answered by re-running the query on Bing;
   AWS WAF's blank token challenge (known by its script alone) is waited
   out and its CAPTCHA puzzle is reported without pressing "Begin".

Unreachable without doubles (documented, not mocked): the ``chrome``
channel branch of ``chrome_channel`` (needs Google Chrome installed), the
``playwright`` fallback import (needs Patchright uninstalled), the
non-Linux branch of ``virtual_display``, an Xvfb that starts but never
reports its display (15 s timeout), descriptor exhaustion at the selector
of ``_read_display_number`` (``Popen`` has just released more descriptors
than it takes, so only another thread can cause it), and ``_glide_onto`` returning
``None`` for an element without a rendered box — ``scroll_into_view_if_needed``
already waits for visibility, so that guard only fires when the element
is removed between that call and ``bounding_box()``.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import gc
import importlib.util
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from urllib.parse import parse_qs, quote, urlparse

import pytest

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.tool_interrupt import ToolCallInterrupted, inject_async_exception
from kiss.tests.agents.sorcar.test_webuse_unresponsive_page_timeout import _stop_in
from kiss.tests.server.pause import Pause

_REPO_ROOT = Path(__file__).resolve().parents[5]

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("Xvfb") is None,
    reason="virtual display tests need Linux with Xvfb",
)
# ``resource`` (rlimits) does not exist on Windows; the module is Linux-only
# anyway, so skip at import instead of failing collection there.
resource = pytest.importorskip("resource")

_EVENTS_PAGE = """<!doctype html><html><head><title>events</title></head>
<body style="margin:0">
<div style="height:1400px"></div>
<button id="target" style="width:180px;height:50px">Press me</button>
<input id="box" aria-label="Name">
<a id="wrapped" href="#" style="display:inline-block;width:200px">wrapped link</a>
<pre id="log"></pre>
<script>
window.ev = {moves: [], down: 0, up: 0, keys: [], wheels: [], clicked: 0};
document.addEventListener('mousemove', e => ev.moves.push([e.clientX, e.clientY]));
document.addEventListener('mousedown', e => ev.down = performance.now());
document.addEventListener('mouseup', e => ev.up = performance.now());
document.addEventListener('wheel', e => ev.wheels.push(e.deltaY));
document.getElementById('target').addEventListener('click', () => {
  ev.clicked++; document.getElementById('target').textContent = 'Pressed';
});
document.getElementById('box').addEventListener('keydown', e => ev.keys.push(performance.now()));
</script></body></html>"""


# HUMAN / PerimeterX's block page as walmart.com/blocked shows it: the
# heading and sentence the detector keys on, and the ``#px-captcha``
# element whose iframe holds the "Press & Hold" bar.  The bar counts a
# hold of at least 5 s (the real bar fills over several seconds) and the
# page loads the content once ``NEED`` holds were counted, carrying the
# last hold's length and the pointer moves seen while held.  ``WIDTH`` x
# ``HEIGHT`` is the element's size (the real bar is about 300 x 100).
_PRESS_HOLD_PAGE = """<!doctype html><html><head><title>Robot or human?</title></head>
<body style="margin:0;padding:40px;font:16px sans-serif">
<h1>Robot or human?</h1>
<p>Activate and hold the button to confirm that you\u2019re human.</p>
<div id="px-captcha" style="width:WIDTHpx;height:HEIGHTpx"><iframe
 style="width:100%;height:100%;border:0" srcdoc="<body style='margin:0'>
<div id=b style='width:100%;height:100px;background:#ddd;user-select:none'>Press &amp; Hold</div>
<script>
let down = 0, moves = 0, holds = 0;
const b = document.getElementById('b');
b.addEventListener('mousedown', () => { down = performance.now(); moves = 0; });
b.addEventListener('mousemove', () => { if (down) moves++; });
b.addEventListener('mouseup', () => {
  const ms = performance.now() - down; down = 0;
  if (ms >= 5000) { holds++; parent.postMessage({holds, ms, moves}, '*'); }
});
</script></body>"></iframe></div>
<script>
addEventListener('message', e => {
  if (e.data.holds >= NEED)
    location.href = '/press-hold-done?ms=' + Math.round(e.data.ms) + '&moves=' + e.data.moves;
});
</script></body></html>"""


# fiverr.com's "It needs a human touch" page as served on 30 Oct 2026:
# ``#px-captcha`` is a block as wide as the column; its closed shadow root
# holds three ``display:none`` copies of the widget iframe (HUMAN's
# honeypots) and then the visible one, as wide as the element, whose
# "Press & Hold" button is 200 px at its left edge, so no part of it lies
# in the middle 35-65 % of the 900 px element.  Ahead of the button the
# frame carries an off-frame decoy with the same role and name.  The page
# loads the content after ``NEED`` holds of 5 s or more on the button;
# ``SHOWN`` 0 keeps every copy hidden (the widget still rendering).
_WIDE_PRESS_HOLD_PAGE = """<!doctype html><html><head><title>It needs a human touch</title></head>
<body style="margin:0;padding:40px;font:16px sans-serif">
<h1>It needs a human touch</h1>
<p>Complete the task and we\u2019ll get you right back into Fiverr.</p>
<div id="px-captcha" style="display:block;width:900px;min-height:102px"></div>
<script>
const root = document.getElementById('px-captcha').attachShadow({mode: 'closed'});
const bar = "<body style='margin:0'><div role=button aria-label='Press &amp; Hold'"
  + " style='position:absolute;left:-400px;top:0;width:200px;height:100px'>Press</div>"
  + "<div id=b role=button aria-label='Press &amp; Hold' style='width:200px;height:100px;"
  + "background:#ddd;user-select:none'>Press &amp; Hold</div></body>";
let holds = 0;
for (let i = 0; i < 4; i++) {
  const f = document.createElement('iframe');
  f.title = 'Human verification challenge';
  f.style.cssText = 'width:100%;height:102px;border:0;display:'
    + (i === 3 && SHOWN ? 'block' : 'none');
  f.addEventListener('load', () => {
    const b = f.contentDocument.getElementById('b');
    let down = 0, moves = 0;
    b.addEventListener('mousedown', () => { down = performance.now(); moves = 0; });
    b.addEventListener('mousemove', () => { if (down) moves++; });
    b.addEventListener('mouseup', () => {
      if (!down) return;
      const ms = performance.now() - down;
      down = 0;
      if (ms >= 5000 && ++holds >= NEED)
        location.href = '/press-hold-done?ms=' + Math.round(ms) + '&moves=' + moves;
    });
  });
  f.srcdoc = bar;
  root.appendChild(f);
}
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    """Serve the probe pages.

    ``/events`` records input events in ``window.ev``; ``/challenge``
    answers the first *N* requests per path with a Cloudflare-style
    interstitial (``cf-mitigated: challenge`` header, "Just a moment..."
    title, a script reloading after 1 s) and content afterwards;
    ``/blocked`` never clears; ``/press-hold?need=N&w=W&h=H`` is a
    press-and-hold page cleared by *N* holds of its *W* x *H* bar and
    ``/press-hold-bare`` the same page without the bar;
    ``/press-hold-wide?need=N&shown=S`` is fiverr's layout of the widget
    (a narrow button in a wide frame among hidden copies); ``/anubis``
    answers the first request per query with an Anubis page that reloads
    after 16 s (more than the ordinary 12 s wait); ``/sorry/index`` is
    Google's "unusual traffic" page; ``/aws-challenge`` answers the first
    request per query with AWS WAF's blank token interstitial (HTTP 202,
    empty title, no text, the ``*.awswaf.com`` challenge script -- which
    the test's route makes reload the page after 1.5 s) and content
    afterwards; ``/aws-captcha`` is AWS WAF's puzzle page (HTTP 405, title
    "Human Verification", the "Begin" button counting its clicks in
    ``window.begins``) and ``/aws-challenge-captcha`` the token
    interstitial first and the puzzle afterwards, as a fresh profile gets
    them; ``/malformed-script`` is content with a script ``src`` that is
    not a URL; ``/headers`` echoes the request's ``User-Agent`` and
    ``Sec-CH-UA`` headers as JSON; ``/reddit-block`` and
    ``/reddit-old-block`` are Reddit's two network-policy pages (HTTP
    403); ``/so-nocaptcha?pass=1|0`` is Stack Overflow's "Human
    verification" page with reCAPTCHA's two frames (served by the test's
    route on www.google.com), whose tick either submits the page to
    ``/so-nocaptcha-done`` or opens the puzzle, and with ``back=N`` leads
    back to the page that led there on all but the *N*-th request;
    ``/links?to=P`` is a page with a link to *P* and a search form
    submitting to *P*.
    """

    challenge_hits: dict[str, int] = {}

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Answer the request according to its path."""
        path = urlparse(self.path).path
        headers = {}
        status = 200
        if path == "/events":
            body = _EVENTS_PAGE
        elif path.startswith("/challenge"):
            n = self.challenge_hits.get(path, 0) + 1
            self.challenge_hits[path] = n
            if n <= 2:
                headers["cf-mitigated"] = "challenge"
                body = (
                    "<title>Just a moment...</title><body>Performing security "
                    "verification<script>setTimeout(()=>location.reload(),1000)</script></body>"
                )
            else:
                body = "<title>Real content</title><body><h1>Welcome</h1></body>"
        elif path == "/blocked":
            headers["cf-mitigated"] = "challenge"
            body = "<title>Just a moment...</title><body>Verifying you are human.</body>"
        elif path == "/press-hold":
            query = parse_qs(urlparse(self.path).query)
            body = (
                _PRESS_HOLD_PAGE.replace("NEED", query.get("need", ["1"])[0])
                .replace("WIDTH", query.get("w", ["300"])[0])
                .replace("HEIGHT", query.get("h", ["100"])[0])
            )
        elif path == "/press-hold-bare":
            body = _PRESS_HOLD_PAGE.split('<div id="px-captcha"')[0] + "</body></html>"
        elif path == "/press-hold-wide":
            query = parse_qs(urlparse(self.path).query)
            body = _WIDE_PRESS_HOLD_PAGE.replace("NEED", query.get("need", ["1"])[0]).replace(
                "SHOWN", query.get("shown", ["1"])[0]
            )
        elif path == "/press-hold-done":
            body = "<title>Real content</title><body><h1>Welcome</h1></body>"
        elif path == "/anubis":
            n = self.challenge_hits.get(self.path, 0) + 1
            self.challenge_hits[self.path] = n
            if n == 1:
                body = (
                    "<title>Making sure you're not a bot!</title><body>Calculating..."
                    "<script>setTimeout(()=>location.reload(),16000)</script></body>"
                )
            else:
                body = "<title>Real content</title><body><h1>Welcome</h1></body>"
        elif path == "/turnstile":
            # Cloudflare's interactive managed challenge: the widget iframe
            # (served by the test's route on challenges.cloudflare.com)
            # tells the page when its box is ticked; the page then loads
            # the content (``?then=`` another page), as the real challenge
            # reloads it.
            then = parse_qs(urlparse(self.path).query).get("then", ["/turnstile-done"])[0]
            headers["cf-mitigated"] = "challenge"
            body = (
                "<title>Just a moment...</title><body><h2>Performing security "
                "verification</h2><iframe src='https://challenges.cloudflare.com/"
                "cdn-cgi/challenge-platform/turnstile/widget' style='width:300px;"
                "height:65px;border:0'></iframe><script>addEventListener('message',"
                f" e => {{ if (e.data === 'ticked') location.href = {json.dumps(then)}; }})"
                "</script></body>"
            )
        elif path == "/turnstile-done":
            body = "<title>Real content</title><body><h1>Welcome</h1></body>"
        elif path == "/sorry/index":
            body = (
                "<title>https://www.google.com/search</title><body>Our systems have "
                "detected unusual traffic from your computer network. IP address: 10.0.0.1"
                "</body>"
            )
        elif path == "/aws-challenge":
            n = self.challenge_hits.get(self.path, 0) + 1
            self.challenge_hits[self.path] = n
            if n == 1:
                status = 202
                body = _AWS_CHALLENGE_PAGE
            else:
                body = "<title>Real content</title><body><h1>Welcome</h1></body>"
        elif path == "/aws-captcha":
            status = 405
            body = _AWS_CAPTCHA_PAGE
        elif path == "/aws-challenge-captcha":
            n = self.challenge_hits.get(self.path, 0) + 1
            self.challenge_hits[self.path] = n
            status = 202 if n == 1 else 405
            body = _AWS_CHALLENGE_PAGE if n == 1 else _AWS_CAPTCHA_PAGE
        elif path == "/malformed-script":
            # Chromium keeps a script ``src`` that is not a URL as written.
            body = (
                "<title>Real content</title><script src='https://[broken/challenge.js'>"
                "</script><body><h1>Welcome</h1></body>"
            )
        elif path == "/reddit-block":
            status = 403
            body = _REDDIT_BLOCK_PAGE
        elif path == "/reddit-old-block":
            status = 403
            body = _REDDIT_OLD_BLOCK_PAGE
        elif path == "/so-nocaptcha":
            query = parse_qs(urlparse(self.path).query)
            body = _SO_NOCAPTCHA_PAGE.replace("PASS", query.get("pass", ["0"])[0])
            n = self.challenge_hits[self.path] = self.challenge_hits.get(self.path, 0) + 1
            if n < int(query.get("back", ["0"])[0]):
                body = body.replace("'/so-nocaptcha-done'", "document.referrer")
        elif path == "/so-nocaptcha-done":
            body = "<title>Real content</title><body><h1>Welcome</h1></body>"
        elif path == "/links":
            to = parse_qs(urlparse(self.path).query).get("to", ["/inert"])[0]
            body = (
                f"<title>links</title><body><a id=go href='{to}'>go</a>"
                f"<form action='{to}'><input name=q aria-label='query'></form></body>"
            )
        elif path == "/headers":
            echo = {
                "user_agent": self.headers.get("User-Agent", ""),
                "sec_ch_ua": self.headers.get("Sec-CH-UA", ""),
            }
            body = f"<title>headers</title><body><pre id=h>{json.dumps(echo)}</pre></body>"
        else:
            body = "<title>inert</title><body>a page</body>"
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default stderr request log."""


# AWS WAF's two stock interstitials as semanticscholar.org served them on
# 30-31 Oct 2026 (markup trimmed): the Challenge action's blank page, whose
# token script reloads the page once a token is set, and the CAPTCHA
# action's puzzle page, whose "Begin" button opens the picture puzzle.
_AWS_WAF_HOST = "https://0123456789ab.35039563.us-east-2.token.awswaf.com/0123456789ab/4996efde3854"
_AWS_CHALLENGE_PAGE = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title></title>
<script type="text/javascript">window.gokuProps = {{"key": "AQID", "iv": "A6xF", "context": "hob"}};
</script><script src="{_AWS_WAF_HOST}/fba16e1c8e93/challenge.js"></script></head>
<body><div id="challenge-container"></div></body></html>"""
_AWS_CAPTCHA_PAGE = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Human Verification</title>
<script type="text/javascript">window.gokuProps = {{"key": "AQID", "iv": "A6xF", "context": "hob"}};
</script><script src="{_AWS_WAF_HOST}/fba16e1c8e93/challenge.js"></script>
<script src="{_AWS_WAF_HOST.replace(".token.", ".captcha.")}/fba16e1c8e93/captcha.js"></script>
</head><body><div id="captcha-container"><div id="root" class="amzn-captcha-modal">
<h1>Let's confirm you are human</h1>
<p id="root-instructions">Complete the security check before continuing. This step verifies
that you are not a bot, which helps to protect your account and prevent spam.</p>
<button class="amzn-captcha-verify-button btn-primary" onclick="window.begins++">Begin</button>
<div class="amzn-captcha-lang-selector"><label for="lang">Select language</label>
<select id="lang"><option>English</option><option>Deutsch</option></select>
<button class="amzn-captcha-lang-apply">Apply</button></div></div></div>
<script>window.begins = 0</script></body></html>"""


# Reddit's two block pages as www.reddit.com and old.reddit.com served
# them from this network on 3 Nov 2026 (HTTP 403, markup trimmed).
_REDDIT_BLOCK_PAGE = """<!DOCTYPE html><html><head><title></title></head><body>
<img alt="" src="data:,"><p>You've been blocked by network security.</p>
<p>To continue, log in to your Reddit account or use your developer token</p>
<p>If you think you've been blocked by mistake, file a ticket below and we'll look into it.</p>
<a href="https://www.reddit.com/login/">Log in</a>
<a href="https://support.reddithelp.com/hc/en-us/requests/new">File a ticket</a></body></html>"""
_REDDIT_OLD_BLOCK_PAGE = """<!DOCTYPE html><html><head><title>Blocked</title></head><body>
<h1>whoa there, pardner!</h1><p>Your request has been blocked due to a network policy.</p>
<p>Try logging in or creating an account <a href="https://www.reddit.com/login/">here</a>
to get back to browsing.</p><p>If you're running a script or application, please register
or sign in with your developer credentials here.</p>
<p>When contacting us, please include your Reddit account along with the following code:</p>
<p>01a1a14b-773e-77dc-8fc5-cfeabf8ec3e3</p></body></html>"""

# Stack Overflow's "Human verification" page (``/nocaptcha?s=<id>``, 3 Nov
# 2026, markup trimmed): its wording, reCAPTCHA v2's anchor frame (the
# box) and challenge frame (the puzzle, in a wrapper kept hidden until the
# box's verdict calls for it), and the page's own reaction to the box:
# ``#nocaptcha-form`` submits itself (PASS=1) or the puzzle opens (PASS=0).
_SO_NOCAPTCHA_PAGE = """<!DOCTYPE html><html><head>
<title>Human verification - Stack Overflow</title></head>
<body><header>Skip to main content About Products Log in Sign up</header>
<div id="mainbar"><h1>Human verification</h1>
<p>Are you a human being? We apologize for the confusion, but we can't quite tell if you're a
person or a script. Please don't take this personally. Bots and scripts can be remarkably
lifelike these days!</p><p>Check the CAPTCHA box, and we'll be out of your way.</p>
<form method="post" action="/so-nocaptcha" id="nocaptcha-form"><div id="no-captcha-here">
<div style="width: 304px; height: 78px"><iframe title="reCAPTCHA" width="304" height="78"
role="presentation" src="https://www.google.com/recaptcha/api2/anchor?k=6Lc5test&hl=en"></iframe>
</div></div></form></div>
<div id="puzzle" style="visibility: hidden; position: absolute; top: 0; left: 0">
<div style="width: 400px; height: 580px"><iframe title="recaptcha challenge expires in two
minutes" src="https://www.google.com/recaptcha/api2/bframe?hl=en&k=6Lc5test" width="400"
height="580"></iframe></div></div>
<script>window.ticks = 0; addEventListener('message', e => { if (e.data !== 'ticked') return;
window.ticks++; if (PASS) location.href = '/so-nocaptcha-done';
else document.getElementById('puzzle').style.visibility = 'visible'; })</script>
</body></html>"""
# reCAPTCHA's anchor frame: the "I'm not a robot" box (a 28 px span with
# the checkbox role, as Google renders it); its tick is reported to the
# page, as the real widget's callback is.
_RECAPTCHA_ANCHOR = (
    "<body style='margin:0;padding:12px;font:14px sans-serif'><span id='recaptcha-anchor'"
    " role='checkbox' aria-checked='false' tabindex='0' style='display:inline-block;"
    "width:28px;height:28px;border:2px solid #c1c1c1;vertical-align:middle'"
    " onclick=\"this.setAttribute('aria-checked', 'true'); parent.postMessage('ticked', '*')\">"
    "</span> I'm not a robot</body>"
)


def _serve_recaptcha_frames(route) -> None:
    """Answer www.google.com/recaptcha requests: the box in the anchor frame, an empty puzzle."""
    body = _RECAPTCHA_ANCHOR if "/anchor" in route.request.url else "<body></body>"
    route.fulfill(status=200, content_type="text/html", body=body)


_TURNSTILE_WIDGET = (
    "<body style='margin:0;padding:20px;font:14px sans-serif'>"
    "<input type=checkbox id=cb aria-label='Verify you are human'"
    " style='width:24px;height:24px;vertical-align:middle'"
    " onclick=\"parent.postMessage('ticked', '*')\"> Verify you are human</body>"
)


# The same box with a slow verdict: it stays visible and checked for 5 s
# and only then reports the tick — unless it was clicked again meanwhile,
# which unticks it (as a real checkbox does) and loses the verdict.
_SLOW_TURNSTILE_WIDGET = _TURNSTILE_WIDGET.replace(
    "onclick=\"parent.postMessage('ticked', '*')\"",
    "onclick=\"setTimeout(() => { if (cb.checked) parent.postMessage('ticked', '*'); }, 5000)\"",
)


def _serve_turnstile_widget(route) -> None:
    """Answer the challenges.cloudflare.com iframe request with a checkbox widget."""
    route.fulfill(status=200, content_type="text/html", body=_TURNSTILE_WIDGET)


def _serve_slow_turnstile_widget(route) -> None:
    """Answer the iframe request with the checkbox whose verdict takes 5 s."""
    route.fulfill(status=200, content_type="text/html", body=_SLOW_TURNSTILE_WIDGET)


def _serve_aws_waf_script(route) -> None:
    """Answer a ``*.awswaf.com`` script request like the token challenge.

    On the blank interstitial (no title) the page reloads after 1.5 s,
    as the real script does once it holds a token; on the puzzle page,
    which carries the same script, it does nothing.
    """
    route.fulfill(
        status=200,
        content_type="application/javascript",
        body="if (!document.title) setTimeout(() => location.reload(), 1500)",
    )


def _serve_inert_script(route) -> None:
    """Answer a script request with a script that does nothing."""
    route.fulfill(status=200, content_type="application/javascript", body="")


# A semanticscholar.org paper URL and the Crossref answer for its slug's
# words (shape of api.crossref.org/works?query.bibliographic=... on
# 1 Nov 2026, trimmed to the selected fields).
_S2_PAPER_URL = (
    "https://www.semanticscholar.org/paper/Real-content-paper-Doe-Roe/"
    "0123456789abcdef0123456789abcdef01234567"
)
_CROSSREF_ANSWER = {
    "status": "ok",
    "message-type": "work-list",
    "message": {
        "total-results": 2,
        "items": [
            {
                "DOI": "10.1000/real.1",
                "title": ["Real content paper <with> markup"],
                "author": [
                    {"given": "Jane", "family": "Doe"},
                    {"given": "Rob", "family": "Roe"},
                    {"name": "The Consortium"},
                ],
                "container-title": ["Journal of Tests"],
                "issued": {"date-parts": [[2024, 5]]},
                "is-referenced-by-count": 7,
                "abstract": (
                    "<jats:p>An <jats:italic>abstract</jats:italic>  of the paper.</jats:p>"
                ),
                "URL": "https://doi.org/10.1000/real.1",
            },
            {"DOI": "10.1000/other.2", "title": ["Another paper"], "issued": {}},
        ],
    },
}


def _serve_aws_captcha_page(route) -> None:
    """Answer like semanticscholar.org does from this network: the AWS WAF puzzle."""
    route.fulfill(status=405, content_type="text/html", body=_AWS_CAPTCHA_PAGE)


def _serve_crossref_answer(route) -> None:
    route.fulfill(status=200, content_type="application/json", body=json.dumps(_CROSSREF_ANSWER))


@pytest.fixture(scope="module")
def server():
    """Yield the base URL of a local server serving the probe pages."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    """Yield one default-configured WebUseTool shared by the module's tests."""
    base = tmp_path_factory.mktemp("stealth")
    web = WebUseTool(user_data_dir=str(base / "profile"), work_dir=str(base))
    try:
        yield web
    finally:
        web.close()


def _main(tool: WebUseTool, expression: str) -> object:
    """Evaluate in the page's main world.

    Patchright evaluates in an isolated world by default (so page scripts
    cannot see the automation); the recorder object ``window.ev`` and
    the masked ``navigator.userAgent`` live in the main world, where site
    scripts run.
    """
    return tool._page.evaluate(expression, isolated_context=False)


def _ev(tool: WebUseTool) -> dict:
    ev = _main(tool, "window.ev")
    assert isinstance(ev, dict)
    return ev


# ---------------------------------------------------------------------------
# 1. Engine and virtual display
# ---------------------------------------------------------------------------


def test_patchright_drives_the_browser_headed_on_a_virtual_display(tool, server):
    """Default tool: Patchright API, headed Chromium on the shared Xvfb display."""
    assert web_stealth.playwright_api().__name__ == "patchright.sync_api"
    assert web_stealth.playwright_package() == "patchright"
    assert web_stealth.chrome_channel() in ("chrome", "chromium")

    tool.go_to_url(f"{server}/inert")
    display = web_stealth.virtual_display()
    assert display is not None and display.startswith(":")
    assert web_stealth.virtual_display() == display, "one Xvfb per process"
    assert tool._headless is True, "still 'no window' from the caller's view"
    assert tool._chromium_headless is False, "but Chromium itself runs headed"
    assert tool._context_args() == {"no_viewport": True}
    # The browser really is on that display.  (Chromium >= 153 zeroes its
    # own /proc/<pid>/environ block, so the DISPLAY variable cannot be
    # read back from the process.)  A headed window reports the X screen
    # it sits on, and Xvfb was started with ``-screen 0 1920x1080x24``.
    assert tool._page.evaluate("[screen.width, screen.height, screen.colorDepth]") == [
        1920, 1080, 24,
    ]
    if shutil.which("xlsclients"):
        clients = subprocess.run(
            ["xlsclients", "-display", display, "-l"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        assert "Command:  chrome" in clients, clients


def test_fingerprint_has_no_automation_tells(tool, server):
    """UA, webdriver flag, window geometry and emulation look like a desktop Chrome."""
    tool.go_to_url(f"{server}/inert")
    probe = tool._page.evaluate(
        "() => ({ua: navigator.userAgent, wd: navigator.webdriver,"
        " inner: [innerWidth, innerHeight], outer: [outerWidth, outerHeight],"
        " dpr: devicePixelRatio,"
        " tz: Intl.DateTimeFormat().resolvedOptions().timeZone})"
    )
    assert "HeadlessChrome" not in probe["ua"] and "Chrome/" in probe["ua"]
    assert probe["wd"] is False
    # A headed window has browser chrome: the outer height exceeds the
    # inner one, unlike headless where the two coincide.
    assert probe["outer"][1] > probe["inner"][1]
    assert probe["outer"] == list(tool.viewport)
    assert tuple(probe["inner"]) == tool._viewport_size()
    assert probe["dpr"] == 1  # the machine's real scale, no Retina emulation
    # No timezone override: whatever the machine reports is what a
    # person's browser here would report too.
    assert probe["tz"]
    if time.tzname[0] == "UTC":
        assert probe["tz"] in ("UTC", "Etc/UTC")


def test_unattended_browser_has_no_webauthn(tool, server):
    """A security-key request would open a native dialog nobody can see or satisfy.

    Chrome's modal "Use your security key" prompt is invisible in a
    browser without a human-visible window and blocks every click on
    the page beneath it, so such a browser exposes no WebAuthn API and
    sign-in pages (Duo, Google) offer their other methods instead.  A
    browser with a visible window keeps the API: its owner can see the
    dialog and plug in the key.
    """
    tool.go_to_url(f"{server}/inert")
    assert tool._page.evaluate("typeof window.PublicKeyCredential") == "undefined"
    assert web_stealth.disabled_blink_features(unattended=True) == (
        "--disable-blink-features=AutomationControlled,WebAuth"
    )
    assert web_stealth.disabled_blink_features(unattended=False) == (
        "--disable-blink-features=AutomationControlled"
    )


def _open_fds() -> set[tuple[str, str]]:
    """The process's open descriptors as ``(number, target)`` pairs.

    The target (``pipe:[inode]``, ``socket:[inode]``, a path) tells a
    leaked pipe from an unrelated descriptor that was closed meanwhile
    and whose number it took.  ``os.listdir`` names its own directory
    handle too, the lowest free number at the time; it is closed again
    by the time the names are read, so it (and anything else closed by
    then) is left out.
    """
    fds: set[tuple[str, str]] = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            fds.add((fd, os.readlink(f"/proc/self/fd/{fd}")))
        except OSError:
            pass
    return fds


def test_virtual_display_restarts_after_stop_and_falls_back_without_xvfb(tool, server):
    """Stopping the display kills Xvfb; without the binary the tool goes headless."""
    assert web_stealth.virtual_display() is not None
    proc = web_stealth._XVFB_PROC
    assert proc is not None and proc.poll() is None
    xvfb_pid = int(subprocess.run(
        ["pgrep", "-P", str(proc.pid), "-x", "Xvfb"], capture_output=True, text=True
    ).stdout.split()[0])
    web_stealth.stop_virtual_display()
    assert proc.poll() is not None, "wrapper shell exited"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and os.path.exists(f"/proc/{xvfb_pid}"):
        time.sleep(0.1)
    assert not os.path.exists(f"/proc/{xvfb_pid}"), "Xvfb itself was killed"
    assert web_stealth._XVFB_PROC is None
    web_stealth.stop_virtual_display()  # idempotent when nothing runs
    # Restart cycles neither leak descriptors nor pile up atexit callbacks.
    # A leak is a descriptor that is open afterwards and was not before;
    # the count alone also drops when the cyclic collector closes an
    # earlier test's unreachable socket meanwhile (full-suite load,
    # 2026-11-05), so collect first and compare the sets.
    gc.collect()
    fds_before = _open_fds()
    callbacks_before = atexit._ncallbacks()
    for _ in range(2):
        assert web_stealth.virtual_display() is not None
        web_stealth.stop_virtual_display()
    assert not _open_fds() - fds_before
    assert atexit._ncallbacks() == callbacks_before
    # A daemon holding more than FD_SETSIZE descriptors gets a displayfd
    # pipe numbered above 1023, which ``select.select`` refused (C/C03-F1).
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))
    held = [os.open(os.devnull, os.O_RDONLY)]
    try:
        while held[-1] < 1024:
            held.append(os.open(os.devnull, os.O_RDONLY))
        assert web_stealth.virtual_display() is not None
        web_stealth.stop_virtual_display()
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert not _open_fds() - fds_before
    # Descriptor exhaustion: with two descriptors of headroom the pipe
    # still fits but Popen's own pipe does not; with none, not even the
    # pipe does (where a descriptor another thread opened meanwhile lands
    # the two-headroom case under load).  Either way the tool falls back
    # to headless and closes whatever pipe ends it opened.
    # Earlier tests leave holes in the descriptor table (closed browser
    # sockets), and the kernel hands out the lowest free number, so first
    # plug every hole: the loop ends once a plug lands above the old top.
    for headroom in (2, 0):
        highest = max(int(fd) for fd in os.listdir("/proc/self/fd"))
        plugs: list[int] = []
        while not plugs or plugs[-1] < highest:
            plugs.append(os.open(os.devnull, os.O_RDONLY))
        highest = plugs[-1]
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (highest + 1 + headroom, hard))
        try:
            assert web_stealth.virtual_display() is None
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
            for fd in plugs:
                os.close(fd)
        assert not _open_fds() - fds_before

    # No Xvfb binary on PATH -> no display -> the launch falls back to
    # real headless mode (with its headed user agent, see
    # test_headless_fallback_uses_a_native_user_agent).  Only one sync
    # Playwright driver may live per thread, so the shared tool is closed
    # first (it relaunches on its next call).
    tool.close()
    saved_path = os.environ["PATH"]
    os.environ["PATH"] = str(os.devnull)
    try:
        assert web_stealth.virtual_display() is None
        headless_tool = WebUseTool(user_data_dir=None)
        try:
            headless_tool.go_to_url(f"{server}/inert")
            assert headless_tool._chromium_headless is True
            assert headless_tool._context_args() == {
                "viewport": {"width": 1280, "height": 900}
            }
            assert headless_tool._viewport_size() == (1280, 900)
            # The browser itself reports a headed Chrome: both worlds
            # agree, nothing was patched from inside the page.
            main_ua = _main(headless_tool, "navigator.userAgent")
            assert isinstance(main_ua, str) and "HeadlessChrome" not in main_ua
            assert headless_tool._page.evaluate("navigator.userAgent") == main_ua
        finally:
            headless_tool.close()
    finally:
        os.environ["PATH"] = saved_path

    # A fresh display comes up on demand and the shared tool relaunches on it.
    fresh = web_stealth.virtual_display()
    assert fresh is not None
    tool.go_to_url(f"{server}/inert")
    assert tool._chromium_headless is False


def test_broken_xvfb_binary_means_no_virtual_display(tool, tmp_path):
    """An Xvfb that exits at once yields no display (and no leaked wrapper)."""
    tool.close()
    web_stealth.stop_virtual_display()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "Xvfb"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    saved_path = os.environ["PATH"]
    os.environ["PATH"] = f"{fake_bin}:{saved_path}"
    try:
        assert web_stealth.virtual_display() is None
        assert web_stealth._XVFB_PROC is None
        # Bytes that are no display number fail the call: the error reaches the
        # caller from the start-up thread, which first terminated the wrapper.
        fake.write_text('#!/bin/sh\nprintf "\\377\\n" > "/proc/self/fd/$2"\nexec sleep 300\n')
        with pytest.raises(UnicodeDecodeError):
            web_stealth.virtual_display()
        wrappers = subprocess.run(
            ["pgrep", "-P", str(os.getpid()), "-f", "^sh -c Xvfb -displayfd "], capture_output=True
        ).stdout
        assert not wrappers and web_stealth._XVFB_PROC is None
    finally:
        os.environ["PATH"] = saved_path
    assert web_stealth.virtual_display() is not None


def test_stop_during_xvfb_startup_leaves_no_wrapper_behind(tool, tmp_path):
    """A Stop landing while Xvfb starts up kills the wrapper instead of leaking it.

    ``virtual_display`` runs on the task thread, where a Stop arrives as
    an exception injected with ``PyThreadState_SetAsyncExc`` (the
    runner's mechanism, used here directly).  Before the fix the
    interrupted call closed its pipe ends but left the wrapper shell and
    its Xvfb running for the rest of the process, with ``_XVFB_PROC``
    still ``None`` so nothing could ever reap them.
    """
    tool.close()
    web_stealth.stop_virtual_display()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pid_file = tmp_path / "xvfb.pid"
    # Starts, reports its pid, and never writes the display number.
    fake = fake_bin / "Xvfb"
    fake.write_text(f"#!/bin/sh\necho $$ > {pid_file}\nexec sleep 300\n")
    fake.chmod(0o755)
    outcome: list[object] = []

    def start_display() -> None:
        try:
            outcome.append(web_stealth.virtual_display())
        except KeyboardInterrupt:
            outcome.append(KeyboardInterrupt)

    saved_path = os.environ["PATH"]
    os.environ["PATH"] = f"{fake_bin}:{saved_path}"
    xvfb_pid = -1
    try:
        thread = threading.Thread(target=start_display, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (
            pid_file.exists() and pid_file.read_text().strip()
        ):
            time.sleep(0.05)
        xvfb_pid = int(pid_file.read_text())
        assert os.path.exists(f"/proc/{xvfb_pid}")
        time.sleep(0.2)  # its start-up thread is now polling the displayfd pipe
        assert inject_async_exception(thread.ident or -1, KeyboardInterrupt) == 1
        thread.join(timeout=15)
        assert not thread.is_alive()
        assert outcome == [KeyboardInterrupt]
        assert web_stealth._XVFB_PROC is None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and os.path.exists(f"/proc/{xvfb_pid}"):
            time.sleep(0.1)
        assert not os.path.exists(f"/proc/{xvfb_pid}"), "fake Xvfb leaked past the Stop"
    finally:
        os.environ["PATH"] = saved_path
        if xvfb_pid > 0:  # on failure: do not leave the fake behind for real
            with contextlib.suppress(ProcessLookupError):
                os.kill(xvfb_pid, signal.SIGTERM)
    # The lock was released and a real display comes up afterwards.
    assert web_stealth.virtual_display() is not None


def _start_display(outcome: list[object]) -> None:
    """Thread body: record what ``virtual_display`` returned, or the exception that ended it."""
    try:
        outcome.append(web_stealth.virtual_display())
    except BaseException as error:  # the Stop under test, or what replaced it
        outcome.append(type(error))


def _group_alive(pgid: int) -> bool:
    """Whether process group *pgid* still has a member."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


class _StopCallerAt:
    """Profile hook: inject the task's Stop into the calling thread at one moment of start-up.

    Set with ``threading.setprofile``, it runs on whichever thread
    executes the start-up; the Stop is the runner's own, raised in
    ``caller`` at its next check point.  The moment is the first time,
    once the wrapper is spawned, that the Python function with code
    *site* returns (*event* ``"return"``) or the builtin *site* does
    (``"c_return"``); with no such *event* the hook only watches.  The
    wrapper's ``Popen`` is kept for the test.
    """

    def __init__(self, event: str, site: object) -> None:
        self.event, self.site = event, site
        self.outcome: list[object] = []
        self.caller = threading.Thread(target=_start_display, args=(self.outcome,), daemon=True)
        self.wrapper: subprocess.Popen[bytes] | None = None
        self.fired = False

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "return" and frame.f_code is subprocess.Popen.__init__.__code__:
            self.wrapper = self.wrapper or frame.f_locals["self"]
        if self.wrapper is None or self.fired or event != self.event:
            return
        if (arg if event == "c_return" else frame.f_code) is self.site:
            self.fired = True
            inject_async_exception(self.caller.ident or -1, KeyboardInterrupt)


@pytest.mark.parametrize(
    ("event", "site"),
    [
        pytest.param("return", subprocess.Popen.__init__.__code__, id="wrapper spawned"),
        pytest.param("c_return", os.close, id="write end closed"),
        pytest.param("return", web_stealth._read_display_number.__code__, id="display number read"),
    ],
)
def test_stop_at_any_moment_of_xvfb_startup_leaves_it_owned_or_dead(tool, event, site):
    """A Stop landing on the caller at any moment of the start-up must not orphan Xvfb.

    The start-up ran on the calling task thread, where a Stop is raised
    as a call returns, before its result is bound.  As ``Popen``
    returned, the wrapper and its Xvfb lived on with no owner
    (B/S13-F1).  As the pipe's write end was closed, the ``finally``
    closed its number again: the ``OSError`` replaced the Stop and
    skipped the wrapper's termination (A/S13-F8).  Once the display
    number was read, the owner was not published yet (A14-4, astra).
    """
    tool.close()
    web_stealth.stop_virtual_display()
    hook = _StopCallerAt(event, site)
    threading.setprofile(hook)
    try:
        hook.caller.start()
        hook.caller.join(timeout=30)
        assert not hook.caller.is_alive()
        assert hook.outcome == [KeyboardInterrupt]
        # Owned (then reaped here) or already dead, never a running orphan.
        web_stealth.stop_virtual_display()
        assert hook.wrapper is not None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _group_alive(hook.wrapper.pid):
            time.sleep(0.1)
        assert not _group_alive(hook.wrapper.pid), "the wrapper or its Xvfb leaked past the Stop"
    finally:
        threading.setprofile(None)
        if hook.wrapper is not None and hook.wrapper.poll() is None:  # on failure: no orphan
            os.killpg(hook.wrapper.pid, signal.SIGKILL)
            hook.wrapper.wait(timeout=10)
    assert web_stealth.virtual_display() is not None


def _threads_since(others: set[threading.Thread]) -> set[threading.Thread]:
    """The running threads that are not in *others*."""
    return {thread for thread in threading.enumerate() if thread.ident and thread not in others}


def test_caller_stopped_while_another_start_up_runs_starts_no_xvfb(tool):
    """A request whose caller left while it waited for the lock is dropped, not started.

    The start-up thread of a stopped caller outlives it, queued behind
    the start-up that holds the lock: once that one ends without a
    display, it must not start (and kill) an Xvfb for nobody.
    """
    tool.close()
    web_stealth.stop_virtual_display()
    hook = _StopCallerAt("", None)
    others = set(threading.enumerate())
    threading.setprofile(hook)
    try:
        with web_stealth._XVFB_LOCK:  # held as another caller's start-up holds it
            hook.caller.start()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and len(_threads_since(others)) < 2:
                time.sleep(0.01)  # until the caller's start-up thread runs, queued
            assert inject_async_exception(hook.caller.ident or -1, KeyboardInterrupt) == 1
            hook.caller.join(timeout=30)
            assert hook.outcome == [KeyboardInterrupt]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and _threads_since(others):
            time.sleep(0.01)  # until the queued start-up had its turn
        assert not _threads_since(others)
        assert hook.wrapper is None, "an Xvfb was started for a caller that had left"
    finally:
        threading.setprofile(None)
    assert web_stealth.virtual_display() is not None


# ---------------------------------------------------------------------------
# 2. Human-like input
# ---------------------------------------------------------------------------


def test_click_travels_a_curved_path_and_holds_the_button(tool, server):
    """The pointer glides in, pauses, presses for a while, and the click lands."""
    tree = tool.go_to_url(f"{server}/events")
    button_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'button "Press me"' in line
    )
    tool._page.mouse.move(5, 5)  # far from the (scrolled-away) button
    tool._mouse_xy = (5.0, 5.0)
    _main(tool, "ev.moves = []")
    result = tool.click(button_id)
    assert 'button "Pressed"' in result
    ev = _ev(tool)
    assert ev["clicked"] == 1
    assert len(ev["moves"]) >= 6, "a glide, not a teleport"
    # Not a straight line: the recorded points bow away from the chord.  The
    # chord starts at the pointer's known start, not at ev.moves[0]: a busy
    # renderer coalesces the leading mousemove events into one, so the
    # first recorded point can already sit on the nearly straight tail of
    # the curve.
    (x0, y0), (x1, y1) = (5.0, 5.0), ev["moves"][-1]
    chord = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    max_dev = max(
        abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / chord for x, y in ev["moves"][:-1]
    )
    assert max_dev > 2.0, ev["moves"]
    assert 40 <= ev["up"] - ev["down"] <= 400, "button held like a finger press"
    assert tool._mouse_xy is not None


def test_hover_glides_without_pressing(tool, server):
    """action="hover" moves onto the element and presses nothing."""
    tree = tool.go_to_url(f"{server}/events")
    button_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'button "Press me"' in line
    )
    tool.click(button_id, action="hover")
    ev = _ev(tool)
    assert ev["clicked"] == 0 and ev["down"] == 0
    assert len(ev["moves"]) >= 6


def test_typing_has_uneven_cadence_and_correct_text(tool, server):
    """Keystroke intervals vary and pause at spaces; the value is exact."""
    tree = tool.go_to_url(f"{server}/events")
    box_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'textbox "Name"' in line
    )
    text = "hello human world"
    tool.type_text(box_id, text)
    assert tool._page.evaluate("document.getElementById('box').value") == text
    keys = _ev(tool)["keys"]
    # Control+a and Backspace precede the text.
    typed = keys[-len(text):]
    gaps = [b - a for a, b in zip(typed, typed[1:])]
    assert statistics.pstdev(gaps) > 5, "not a metronome"
    # Playwright waits *after* each key, so the inter-word pause is the
    # gap following the space: 90-260 ms versus 35-115 ms per letter.
    after_space = [gaps[i] for i, ch in enumerate(text[:-1]) if ch == " "]
    after_letter = [gaps[i] for i, ch in enumerate(text[:-1]) if ch != " "]
    assert min(after_space) >= 80
    # The median, not the max: on a busy host a single keystroke's CDP
    # round trip can add 80 ms (195 ms seen once under a full parallel
    # test run) without changing the cadence being checked.
    assert statistics.median(after_letter) <= 150


def test_type_text_with_enter_submits_after_a_pause(tool, server):
    """press_enter waits a moment, then submits."""
    tree = tool.go_to_url(f"{server}/events")
    box_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'textbox "Name"' in line
    )
    _main(
        tool,
        "document.getElementById('box').addEventListener('keydown',"
        " e => { if (e.key === 'Enter') ev.enter = performance.now(); })",
    )
    tool.type_text(box_id, "go", press_enter=True)
    ev = _ev(tool)
    # keys[-1] is the Enter keydown itself; keys[-2] is the last letter.
    assert abs(ev["enter"] - ev["keys"][-1]) < 5
    assert ev["enter"] - ev["keys"][-2] >= 150


def test_scroll_wheels_in_uneven_notches(tool, server):
    """Wheel deltas vary around the nominal step and the page really scrolls."""
    tool.go_to_url(f"{server}/events")
    tool.scroll("down", 4)
    ev = _ev(tool)
    assert len(ev["wheels"]) == 4
    assert all(240 <= d <= 360 for d in ev["wheels"])
    assert len(set(ev["wheels"])) > 1
    assert tool._page.evaluate("scrollY") > 0
    tool.scroll("up", 2)
    assert _ev(tool)["wheels"][-1] < 0


def test_click_falls_back_when_the_glided_point_is_covered(tool, server):
    """A point that receives no events within 5 s yields to the plain click.

    An overlay covers the button for 6 s: the positioned click times out
    (5 s), the plain click's own actionability wait then sees the overlay
    go away and lands.
    """
    tool.go_to_url(
        "data:text/html,<button id=b style='width:300px;height:60px'"
        " onclick=\"this.textContent='done'\">covered</button>"
        "<div id=o style='position:absolute;left:0;top:0;width:400px;height:100px'></div>"
        "<script>setTimeout(() => document.getElementById('o').remove(), 6000)</script>"
    )
    locator = tool._page.get_by_role("button", name="covered")
    start = time.monotonic()
    tool._human_click(locator)
    assert tool._page.evaluate("document.getElementById('b').textContent") == "done"
    assert 5 <= time.monotonic() - start < 30
    # Same for a hover: cover the button again, the positioned hover
    # times out and the plain hover lands once the overlay is gone.
    tool._page.evaluate(
        "() => { const o = document.createElement('div'); o.id = 'o2';"
        " o.style.cssText = 'position:absolute;left:0;top:0;width:400px;height:100px';"
        " document.body.append(o); setTimeout(() => o.remove(), 6000);"
        " document.getElementById('b').textContent = 'covered';"
        " document.getElementById('b').onmouseover = () =>"
        " document.getElementById('b').textContent = 'hovered'; }"
    )
    start = time.monotonic()
    tool._human_click(locator, hover_only=True)
    assert tool._page.evaluate("document.getElementById('b').textContent") == "hovered"
    assert 5 <= time.monotonic() - start < 30


def test_glide_failure_falls_back_to_plain_click(tool, server):
    """A Playwright error while gliding still ends in a click.

    The button is hidden for 12 s: ``scroll_into_view_if_needed`` gives up
    after the 10 s page-read timeout (a Playwright error), and the plain
    click's 30 s wait then catches the button becoming visible.
    """
    tool.go_to_url(
        "data:text/html,<button id=b hidden onclick=\"this.textContent='done'\">go</button>"
        "<script>setTimeout(() => document.getElementById('b').hidden = false, 12000)</script>"
    )
    locator = tool._page.get_by_role("button", name="go", include_hidden=True)
    start = time.monotonic()
    tool._human_click(locator)
    assert tool._page.evaluate("document.getElementById('b').textContent") == "done"
    assert 10 <= time.monotonic() - start < 40


def test_mouse_path_shape_and_typing_chunks():
    """Pure helpers: Bezier waypoints end on target; chunks rebuild the text."""
    path = web_stealth.mouse_path((0.0, 0.0), (400.0, 100.0))
    assert path[-1] == (400.0, 100.0)
    assert 6 <= len(path) <= 24
    assert web_stealth.mouse_path((10.0, 10.0), (10.4, 10.2)) == [(10.4, 10.2)]
    chunks = web_stealth.typing_chunks("a  b c")
    assert "".join(c for c, _ in chunks) == "a  b c"
    assert all(90 <= d <= 260 for c, d in chunks if c == " ")
    assert all(35 <= d <= 115 for c, d in chunks if c != " ")
    assert web_stealth.typing_chunks("") == []
    assert web_stealth.typing_duration_secs([("ab", 100), (" ", 200)]) == 0.4
    assert all(
        8 <= web_stealth.press_hold_secs(web_stealth.PERIMETERX_CHECK) <= 10 for _ in range(20)
    )
    assert all(
        0.06 <= web_stealth.press_hold_secs(web_stealth.CLOUDFLARE_CHALLENGE) <= 0.14
        for _ in range(20)
    )
    tx, ty = web_stealth.hold_tremor(100.0, 50.0)
    assert abs(tx - 100.0) <= 1.5 and abs(ty - 50.0) <= 1.5


@pytest.mark.timeout(120)
def test_sync_api_fails_fast_once_the_driver_has_exited():
    """Every sync API call after the driver process exits raises at once.

    Unguarded, only the first call after the exit fails (its request dies
    with the closed pipe) and every later one busy-spins forever on the
    finished dispatcher greenlet (``BrowserContext.close`` hung 900 s in
    the 2026-10-25 full-suite run).  Its own thread: a sync Playwright is
    one per thread and the module's tool already owns this one.
    """
    api = web_stealth.playwright_api()
    errors: list[BaseException] = []
    elapsed: list[float] = []

    def body() -> None:
        pw = api.sync_playwright().start()
        try:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("data:text/html,<h1>x</h1>")
            assert page.evaluate("1 + 1") == 2
            os.kill(pw._impl_obj._connection._transport._proc.pid, signal.SIGKILL)
            time.sleep(0.3)
            with pytest.raises(Exception, match="closed|exited"):
                page.evaluate("1")
            started = time.monotonic()
            for call in (browser.close, page.title, page.context.close):
                with pytest.raises(api.Error, match=web_stealth.DRIVER_EXITED_MESSAGE):
                    call()
            elapsed.append(time.monotonic() - started)
        finally:
            pw.stop()

    def run() -> None:
        try:
            body()
        except BaseException as exc:  # noqa: BLE001 - reported in the test thread
            errors.append(exc)

    thread = threading.Thread(target=run, name="dead-driver-sync-api", daemon=True)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "a sync API call hung on the dead driver"
    if errors:
        raise errors[0]
    assert elapsed and elapsed[0] < 5


# ---------------------------------------------------------------------------
# 3. Challenge pages
# ---------------------------------------------------------------------------


def test_cloudflare_style_challenge_is_waited_out(tool, server):
    """An interstitial that clears on its own yields the real page, no note."""
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/challenge/{time.time_ns()}")
    assert tree.startswith("Page: Real content")
    assert "Note:" not in tree
    assert time.monotonic() - start < 12


def test_persistent_block_is_reported_with_a_note(tool, server):
    """An interstitial that never clears is described plainly to the agent."""
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/blocked")
    elapsed = time.monotonic() - start
    first = tree.splitlines()[0]
    assert first.startswith("Note: 127.0.0.1:") and "(Cloudflare challenge)" in first
    assert "show_browser()" in first
    assert "Page: Just a moment..." in tree
    assert 12 <= elapsed < 40


def test_page_read_survives_a_navigation_landing_under_it(tool, server):
    """A page that navigates itself during a read is read again on the new document.

    Cloudflare's challenge reloads the real page on its own once its checks
    pass.  When that landed between ``wait_for_function`` resolving and
    ``json_value`` fetching the result, ``go_to_url`` answered ``Error
    navigating to https://journals.sagepub.com/: JSHandle.json_value:
    Execution context was destroyed``.  Each attempt here schedules a
    navigation 1 ms ahead from the page, which lands in that gap about
    half the time on the development machine (the raw read failed 27-30
    of 60 attempts); the read must answer every time.
    """
    tool.go_to_url(f"{server}/hop-0")
    for i in range(1, 41):
        tool._page.evaluate(f"() => setTimeout(() => location.replace('/hop-{i}'), 1)")
        assert tool._read_page(tool._page, "() => [document.title]") == ["inert"]
        tool._page.wait_for_url(f"**/hop-{i}")
        tool._page.wait_for_load_state("domcontentloaded")


_CHAIN_READ = """
() => {
  const hops = Number(location.pathname.split('/').pop());
  if (hops > 0) {
    setTimeout(() => location.replace('/chain/' + (hops - 1) + location.search), 0);
  }
  return [document.title];
}
"""


def test_page_read_survives_a_chain_of_navigations(tool, server):
    """A page that navigates twice in a row under a read is still read.

    The live journals.sagepub.com run of 2026-09-30 destroyed the context
    of the re-read too (the challenge reloaded into the real page, which
    redirected once more), so a single repeat was not enough: ``go_to_url``
    answered the same ``Execution context was destroyed`` error.  Here the
    read expression itself schedules the next hop of ``/chain/2`` ->
    ``/chain/1`` -> ``/chain/0`` the moment it resolves, so every hop but
    the last lands in the gap before ``json_value``; with a single repeat
    the read fails on the first attempt.
    """
    tool.go_to_url(f"{server}/chain/0")
    for i in range(1, 41):
        tool._page.goto(f"{server}/chain/2?a={i}")
        assert tool._read_page(tool._page, _CHAIN_READ) == ["inert"]
        tool._page.wait_for_url(f"**/chain/0?a={i}")
        tool._page.wait_for_load_state("domcontentloaded")


def test_turnstile_checkbox_is_ticked_like_a_person(tool, server):
    """The interactive "Verify you are human" box is found in the widget frame and pressed.

    The widget origin is Cloudflare's, so the test answers that request
    itself with a checkbox page; everything else — frame discovery, the
    curved approach, the held press, the page's reaction — is real.
    """
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://challenges.cloudflare.com/**", _serve_turnstile_widget)
    try:
        start = time.monotonic()
        tree = tool.go_to_url(f"{server}/turnstile")
    finally:
        tool._context.unroute("https://challenges.cloudflare.com/**")
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree
    assert time.monotonic() - start < 2 * 12 + 10


def test_turnstile_checkbox_is_ticked_once(tool, server):
    """A box whose verdict takes 5 s is not ticked again while it waits.

    A second click on a checked box unticks it and throws the verdict
    away; the tool ticks Turnstile once and waits.
    """
    tool.go_to_url(f"{server}/inert")  # launches the browser: ``_context`` is None before
    tool._context.route("https://challenges.cloudflare.com/**", _serve_slow_turnstile_widget)
    start = time.monotonic()
    try:
        tree = tool.go_to_url(f"{server}/turnstile")
    finally:
        tool._context.unroute("https://challenges.cloudflare.com/**")
    elapsed = time.monotonic() - start
    assert tree.startswith("Page: Real content"), tree[:300]
    assert 5 <= elapsed < 2 * 12 + 10, elapsed


def test_turnstile_is_not_ticked_when_the_widget_shows_no_box(tool, server):
    """Without a checkbox in the widget frame nothing is pressed and the block is reported."""
    tool.go_to_url(f"{server}/inert")

    def spinner(route) -> None:
        route.fulfill(status=200, content_type="text/html", body="<body>Verifying...</body>")

    tool._context.route("https://challenges.cloudflare.com/**", spinner)
    try:
        tree = tool.go_to_url(f"{server}/turnstile")
    finally:
        tool._context.unroute("https://challenges.cloudflare.com/**")
    assert tree.splitlines()[0].startswith("Note:") and "(Cloudflare challenge)" in tree
    assert tool._frame_checkbox("challenges.cloudflare.com") is None


def test_press_and_hold_bar_is_held_like_a_person(tool, server):
    """HUMAN's "Press & Hold" bar is pressed in its iframe and held for seconds.

    The page is the walmart.com/blocked layout; its bar counts a hold of
    5 s or more and the page then loads the content, reporting how long
    the button was down and how often the pointer moved meanwhile (a
    resting hand drifts; a pointer frozen on one pixel is a script).
    """
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/press-hold?need=1")
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree
    result = parse_qs(urlparse(tool._page.url).query)
    held_ms = int(result["ms"][0])
    assert 8000 <= held_ms < 11000, held_ms
    assert int(result["moves"][0]) >= 10, result
    assert time.monotonic() - start < 12 + 10 + 10


def test_stop_during_a_held_press_releases_the_button(tool, server):
    """A Stop landing while the challenge widget is held lets the button go.

    The release was reached only when the hold ran to its end: the page
    kept the button down for whatever tool call came next.
    """
    tool.go_to_url(f"{server}/events")
    stopping = functools.partial(_stop_in, threading.get_ident())
    with Pause(web_stealth.hold_tremor.__code__, stopping), pytest.raises(ToolCallInterrupted):
        tool._press_at({"x": 300.0, "y": 300.0, "width": 40.0, "height": 40.0}, 5.0)
    ev = _ev(tool)
    assert ev["down"] > 0, "the press never went down"
    assert ev["up"] >= ev["down"], "the mouse button was left pressed"


def _close(page, _entry) -> bool:
    """A :class:`Pause` hook: close *page* at the entry of the hooked function."""
    page.close()
    return False


def test_a_press_on_a_page_that_closes_raises_what_cut_it_short(tool, server):
    """The release of a press whose page closed fails quietly: the hold's own error is raised."""
    tool.go_to_url(f"{server}/events")
    page = tool._page
    tool._page = closing = tool._context.new_page()
    error = web_stealth.playwright_api().Error
    try:
        with Pause(web_stealth.hold_tremor.__code__, functools.partial(_close, closing)):
            with pytest.raises(error, match=r"^Mouse\.move: "):
                tool._press_at({"x": 300.0, "y": 300.0, "width": 40.0, "height": 40.0}, 5.0)
    finally:
        tool._page = page


def test_press_and_hold_two_step_variant_is_pressed_again(tool, server):
    """A bar that asks to "press again when prompted" gets a second hold, not a note.

    The second hold comes only after the bar has stayed showing for 5 s
    after the first (a cleared bar reloads the page within ~2 s; pressing
    inside that window threw away a real pass).
    """
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/press-hold?need=2")
    assert tree.startswith("Page: Real content"), tree[:300]
    elapsed = time.monotonic() - start
    assert 2 * 8 + 5 <= elapsed < 2 * 12 + 2 * 10 + 10, elapsed


def test_press_and_hold_gives_up_after_two_holds(tool, server):
    """A bar that keeps asking is held twice at most, then the page is reported."""
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/press-hold?need=3")
    elapsed = time.monotonic() - start
    first = tree.splitlines()[0]
    assert first.startswith("Note:") and "(PerimeterX press-and-hold check)" in first, first
    assert "show_browser()" in first
    # Two holds of 8-10 s, 5 s apart, then the 12 s verdict wait; the note
    # states the time actually spent, not the nominal 12 s.
    assert 2 * 8 + 5 + 12 <= elapsed < 2 * 10 + 5 + 12 + 2 * 12, elapsed
    match = re.search(r"did not clear within (\d+)s", first)
    assert match is not None, first
    assert elapsed - 6 <= int(match.group(1)) <= elapsed, (first, elapsed)
    assert "Page: Robot or human?" in tree


def test_press_and_hold_page_without_a_bar_is_reported(tool, server):
    """The block page without its widget (nothing to press) is waited out and reported."""
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/press-hold-bare")
    elapsed = time.monotonic() - start
    first = tree.splitlines()[0]
    assert first.startswith("Note:") and "(PerimeterX press-and-hold check)" in first, first
    assert tool._challenge_widget(web_stealth.PERIMETERX_CHECK) is None
    assert 12 <= elapsed < 40, elapsed
    # The empty ``#px-captcha`` placeholder that content pages carry (too
    # narrow or too short to be the bar) is not a widget either; other
    # vendors' pages have nothing to press at all.
    for size in ("w=30&h=100", "w=300&h=10"):
        tool._page.goto(f"{server}/press-hold?{size}", wait_until="domcontentloaded")
        assert tool._challenge_widget(web_stealth.PERIMETERX_CHECK) is None, size
    tool._page.goto(f"{server}/press-hold", wait_until="domcontentloaded")
    box = _widget_box(tool)
    assert box is not None and box["width"] == 300 and box["height"] == 100, box
    assert tool._challenge_widget("Akamai block") is None
    assert web_stealth.widget_presses(web_stealth.PERIMETERX_CHECK) == 2
    assert web_stealth.widget_presses(web_stealth.CLOUDFLARE_CHALLENGE) == 1
    assert web_stealth.widget_presses(web_stealth.ANUBIS_CHECK) == 0


def _widget_box(tool: WebUseTool) -> dict[str, float] | None:
    """Poll the press-and-hold target for up to 10 s (its frame renders late)."""
    deadline = time.monotonic() + 10
    while True:
        box = tool._challenge_widget(web_stealth.PERIMETERX_CHECK)
        if box is not None or time.monotonic() >= deadline:
            return box
        time.sleep(0.2)


def test_press_and_hold_button_narrower_than_its_widget_is_held(tool, server):
    """fiverr's layout: the hold lands on the button, not the middle of ``#px-captcha``.

    The widget's visible frame (in a closed shadow root, after hidden
    honeypot copies) is 900 px wide and its button 200 px at the left
    edge.  Holding the middle of the element, as the tool did until
    30 Oct 2026, lands on empty frame and the page never clears.
    """
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/press-hold-wide?need=1")
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree
    result = parse_qs(urlparse(tool._page.url).query)
    assert 8000 <= int(result["ms"][0]) < 11000, result
    assert int(result["moves"][0]) >= 10, result
    assert time.monotonic() - start < 12 + 10 + 10


def _live_frame_handles(tool: WebUseTool) -> int:
    """Count the element handles the driver keeps alive for the tool's browser."""
    objects = tool._page._impl_obj._connection._objects.values()
    return sum(getattr(obj, "_type", None) == "ElementHandle" for obj in objects)


def test_press_and_hold_target_is_the_visible_frames_button(tool, server):
    """The target is the visible copy's button: not the element, a decoy or a hidden copy."""
    tool.go_to_url(f"{server}/inert")  # launches the browser when this test runs alone
    tool._page.goto(f"{server}/press-hold-wide", wait_until="domcontentloaded")
    widget = tool._read_page(tool._page, web_stealth.PRESS_AND_HOLD_BOX_JS)[0]
    assert widget["width"] == 900, widget
    box = _widget_box(tool)
    assert box is not None, widget
    # A poll keeps none of the frame handles it read the boxes through
    # (five frames a poll here, the widget polled for as long as it shows).
    handles = _live_frame_handles(tool)
    assert tool._challenge_widget(web_stealth.PERIMETERX_CHECK) == box
    assert _live_frame_handles(tool) == handles
    assert abs(box["x"] - widget["x"]) < 1 and abs(box["y"] - widget["y"]) < 1, (box, widget)
    assert (box["width"], box["height"]) == (200, 100), box
    # Every point the old press drew from (35-65 % of the element) misses it.
    assert box["x"] + box["width"] < widget["x"] + 0.35 * widget["width"]
    # While every copy is hidden nothing is pressed, though the element
    # itself is bar-sized.
    tool._page.goto(f"{server}/press-hold-wide?shown=0", wait_until="load")
    assert tool._read_page(tool._page, web_stealth.PRESS_AND_HOLD_BOX_JS)
    copies: list = []
    deadline = time.monotonic() + 10
    while len(copies) < 4 and time.monotonic() < deadline:
        time.sleep(0.1)
        copies = [f for f in tool._page.frames if f.parent_frame is not None]
    assert len(copies) == 4, copies
    assert tool._challenge_widget(web_stealth.PERIMETERX_CHECK) is None


def test_anubis_proof_of_work_gets_its_longer_wait(tool, server):
    """An Anubis page that takes 16 s (more than the 12 s most vendors get) still clears."""
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/anubis?t={int(start)}")
    elapsed = time.monotonic() - start
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree
    assert 16 <= elapsed < 45 + 10, elapsed
    assert web_stealth.challenge_wait_secs(web_stealth.ANUBIS_CHECK) == 45
    assert web_stealth.challenge_wait_secs(web_stealth.CLOUDFLARE_CHALLENGE) == 12


def test_aws_waf_token_challenge_is_waited_out(tool, server):
    """AWS WAF's blank Challenge interstitial yields the real page, not "(empty page)".

    The 202 page has an empty title and no text at all, only the
    ``*.awswaf.com`` token script (answered here by the test's route),
    which reloads the page once its token is set; semanticscholar.org
    came back as an empty page three times in the task history when it
    was read before that.
    """
    tool.go_to_url(f"{server}/inert")  # launches the browser: ``_context`` is None before
    tool._context.route("https://*.awswaf.com/**", _serve_aws_waf_script)
    path = f"/aws-challenge?t={time.time_ns()}"
    try:
        start = time.monotonic()
        tree = tool.go_to_url(server + path)
        elapsed = time.monotonic() - start
    finally:
        tool._context.unroute("https://*.awswaf.com/**")
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree and "(empty page)" not in tree
    # The interstitial was served and the content only after its 1.5 s
    # reload: the tool waited through the blank page rather than
    # returning it.
    assert _Handler.challenge_hits[path] == 2
    assert 1.5 <= elapsed < 12, elapsed
    assert web_stealth.widget_presses(web_stealth.AWS_WAF_CHALLENGE) == 0
    assert web_stealth.challenge_wait_secs(web_stealth.AWS_WAF_CHALLENGE) == 12


def test_aws_waf_captcha_puzzle_is_reported_without_pressing_begin(tool, server):
    """The puzzle page is reported after its short wait; its "Begin" button is never pressed.

    "Begin" opens a picture puzzle only a person solves, so pressing it
    would help nobody; the note tells the agent to show the browser or
    use another source, and the page's own tree follows it.
    """
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://*.awswaf.com/**", _serve_inert_script)
    try:
        start = time.monotonic()
        tree = tool.go_to_url(f"{server}/aws-captcha")
        elapsed = time.monotonic() - start
        begins = _main(tool, "window.begins")  # the page's own world, as its script sees it
    finally:
        tool._context.unroute("https://*.awswaf.com/**")
    first = tree.splitlines()[0]
    assert first.startswith("Note: 127.0.0.1:") and "(AWS WAF CAPTCHA puzzle)" in first, first
    assert "show_browser()" in first
    assert "Page: Human Verification" in tree and "Begin" in tree
    assert begins == 0
    assert 3 <= elapsed < 12, elapsed
    assert tool._challenge_widget(web_stealth.AWS_WAF_CAPTCHA) is None
    assert web_stealth.widget_presses(web_stealth.AWS_WAF_CAPTCHA) == 0
    assert web_stealth.challenge_wait_secs(web_stealth.AWS_WAF_CAPTCHA) == 3


def test_aws_waf_puzzle_after_token_challenge_gets_the_puzzles_short_wait(tool, server):
    """The token challenge handing over to the puzzle is reported ~3 s after the puzzle shows.

    A fresh profile gets the blank 202 page first (a 12 s wait), then the
    405 puzzle; the wait is re-derived from the page now showing rather
    than running the token challenge's 12 s down on a puzzle no wait
    clears.
    """
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://*.awswaf.com/**", _serve_aws_waf_script)
    path = f"/aws-challenge-captcha?t={time.time_ns()}"
    try:
        start = time.monotonic()
        tree = tool.go_to_url(server + path)
        elapsed = time.monotonic() - start
    finally:
        tool._context.unroute("https://*.awswaf.com/**")
    first = tree.splitlines()[0]
    assert first.startswith("Note:") and "(AWS WAF CAPTCHA puzzle)" in first, first
    assert "Page: Human Verification" in tree
    assert _Handler.challenge_hits[path] == 2
    # 1.5 s on the blank page, 3 s on the puzzle, some overhead -- not 12 s.
    assert 4.5 <= elapsed < 10, elapsed


def test_stack_overflow_captcha_box_is_ticked_and_the_page_passes(tool, server):
    """Stack Overflow's "Human verification" page: its reCAPTCHA box is ticked once, like Turnstile.

    The search endpoint redirected a research task to this page on
    3 Nov 2026 and the tool handed its tree over as content.  On a
    network Google rates well the tick is the whole check: the page
    submits itself and the content follows, with no note.
    """
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://www.google.com/recaptcha/**", _serve_recaptcha_frames)
    try:
        start = time.monotonic()
        tree = tool.go_to_url(f"{server}/so-nocaptcha?pass=1&t={time.time_ns()}")
        elapsed = time.monotonic() - start
    finally:
        tool._context.unroute("https://www.google.com/recaptcha/**")
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree
    assert elapsed < 12, elapsed
    assert web_stealth.widget_presses(web_stealth.STACK_OVERFLOW_CAPTCHA) == 1


def test_recaptcha_box_is_ticked_after_a_turnstile_box_cleared_into_it(tool, server):
    """Each page's widget gets its own tick: Turnstile's press is not charged to reCAPTCHA's.

    Stack Overflow sits behind Cloudflare, so a search can meet the
    managed challenge first and its "Human verification" page next.
    Both boxes allow one press; the count restarts when the page gives
    way to another vendor's, or the second box would never be ticked.
    """
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://challenges.cloudflare.com/**", _serve_turnstile_widget)
    tool._context.route("https://www.google.com/recaptcha/**", _serve_recaptcha_frames)
    then = quote(f"/so-nocaptcha?pass=1&t={time.time_ns()}", safe="")
    try:
        start = time.monotonic()
        tree = tool.go_to_url(f"{server}/turnstile?then={then}")
        elapsed = time.monotonic() - start
    finally:
        tool._context.unroute("https://www.google.com/recaptcha/**")
        tool._context.unroute("https://challenges.cloudflare.com/**")
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree
    assert elapsed < 2 * 12 + 10, elapsed


def test_pages_handing_over_to_each_other_run_out_of_presses(tool, server):
    """Each vendor's box is pressed once in one wait, however often the pages alternate.

    The count restarted at every hand-over and each press renewed the
    wait: a Turnstile box clearing into a reCAPTCHA box that leads back
    to it was ticked for as long as the two alternated (four rounds
    here, then content).  Turnstile's second showing is now waited out
    and reported.
    """
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://challenges.cloudflare.com/**", _serve_turnstile_widget)
    tool._context.route("https://www.google.com/recaptcha/**", _serve_recaptcha_frames)
    then = quote(f"/so-nocaptcha?pass=1&back=4&t={time.time_ns()}", safe="")
    try:
        start = time.monotonic()
        tree = tool.go_to_url(f"{server}/turnstile?then={then}")
        elapsed = time.monotonic() - start
    finally:
        tool._context.unroute("https://www.google.com/recaptcha/**")
        tool._context.unroute("https://challenges.cloudflare.com/**")
    assert tree.startswith("Note: 127.0.0.1:") and "/turnstile" in tool._page.url, tree[:300]
    assert elapsed < 3 * 12 + 10, elapsed


def test_stack_overflow_captcha_puzzle_ends_the_wait(tool, server):
    """A tick answered with the picture puzzle is reported at once, with the way out.

    From this machine's address the real box opens the puzzle within
    2 s of the tick and nothing clears it (checked 3 Nov 2026), so the
    wait ends when the puzzle shows rather than running the 12 s down;
    the box is not ticked again and the note names the puzzle and the
    ``site:stackoverflow.com`` search that finds the same questions.
    """
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://www.google.com/recaptcha/**", _serve_recaptcha_frames)
    try:
        start = time.monotonic()
        tree = tool.go_to_url(f"{server}/so-nocaptcha?pass=0&t={time.time_ns()}")
        elapsed = time.monotonic() - start
        ticks = _main(tool, "window.ticks")
        puzzle = tool._puzzle_open(web_stealth.STACK_OVERFLOW_CAPTCHA)
    finally:
        tool._context.unroute("https://www.google.com/recaptcha/**")
    first = tree.splitlines()[0]
    assert first.startswith("Note: 127.0.0.1:"), first
    assert "(Stack Overflow 'Human verification' CAPTCHA) that did not clear within" in first
    assert "picture puzzle" in first and "site:stackoverflow.com" in first
    assert "show_browser()" in first
    assert "Page: Human verification - Stack Overflow" in tree
    assert ticks == 1 and puzzle is True
    # The 1.2 s a person takes to read the box, the tick, one poll -- not 12 s.
    assert 1 <= elapsed < 8, elapsed
    assert tool._puzzle_open(web_stealth.CLOUDFLARE_CHALLENGE) is False


@pytest.mark.parametrize(
    ("path", "wording"),
    [
        ("/reddit-block", "You've been blocked by network security"),
        ("/reddit-old-block", "pardner"),
    ],
)
def test_reddit_block_page_is_reported_without_a_wait(tool, server, path, wording):
    """Reddit's 403 pages (www and old.reddit.com) are reported at once: no wait clears them.

    Reddit refuses this network's address for visitors who are not
    logged in, on every host (checked 3 Nov 2026: www, old, the
    ``.json`` API); the note says so, names the login way out, and the
    page's own tree follows -- not "did not clear within 0s".
    """
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}{path}")
    elapsed = time.monotonic() - start
    first = tree.splitlines()[0]
    assert first.startswith("Note: 127.0.0.1:") and "(Reddit network-policy block)" in first, first
    assert "did not clear" not in first
    assert "log in to Reddit" in first and "old.reddit.com" in first and "show_browser()" in first
    assert wording in tree
    assert elapsed < 4, elapsed
    assert web_stealth.challenge_wait_secs(web_stealth.REDDIT_BLOCK) == 0
    assert web_stealth.widget_presses(web_stealth.REDDIT_BLOCK) == 0


def test_click_waits_out_a_challenge_like_go_to_url(tool, server):
    """A click that lands on an interstitial waits it out instead of returning it.

    On 3 Nov 2026 a click on a search result returned Cloudflare's
    "Just a moment..." page after 3 s: only ``go_to_url`` settled
    challenges.  Here the link leads to a challenge that clears on its
    third request.
    """
    tree = tool.go_to_url(f"{server}/links?to=/challenge/{time.time_ns()}")
    match = re.search(r"\[(\d+)\] link \"go\"", tree)
    assert match, tree[:300]
    start = time.monotonic()
    tree = tool.click(int(match.group(1)))
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree
    assert time.monotonic() - start < 12


def test_typed_submission_and_enter_report_a_block_like_go_to_url(tool, server):
    """A search typed with Enter, or Enter on a focused link, reports the block it lands on."""
    tree = tool.go_to_url(f"{server}/links?to=/reddit-block")
    match = re.search(r"\[(\d+)\] textbox \"query\"", tree)
    assert match, tree[:300]
    tree = tool.type_text(int(match.group(1)), "mobileconfig", press_enter=True)
    first = tree.splitlines()[0]
    assert first.startswith("Note:") and "(Reddit network-policy block)" in first, first
    assert "You've been blocked by network security" in tree

    tool.go_to_url(f"{server}/links?to=/reddit-block")
    tool._page.focus("#go")
    tree = tool.press_key("Enter")
    first = tree.splitlines()[0]
    assert first.startswith("Note:") and "(Reddit network-policy block)" in first, first
    # A key that navigates nowhere leaves the block page showing: it is
    # reported again, at once (Reddit's page gets no wait).
    start = time.monotonic()
    tree = tool.press_key("ArrowDown")
    assert time.monotonic() - start < 3
    assert tree.splitlines()[0].startswith("Note:") and "Reddit" in tree.splitlines()[0]


def test_semanticscholar_puzzle_is_answered_from_crossref(tool, server):
    """A semanticscholar page kept behind the AWS WAF puzzle is looked up on Crossref.

    The site answers paper and search pages from this network with the
    puzzle even for an un-automated Chrome, so after the puzzle's short
    wait the tool opens Crossref's works API for the slug's words and
    renders its JSON as a page: the note names the vendor, the terms and
    the count; the tree shows title, authors, year, venue, DOI link,
    citation count and the abstract without its JATS markup.
    """
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://*.awswaf.com/**", _serve_inert_script)
    tool._context.route("https://www.semanticscholar.org/**", _serve_aws_captcha_page)
    tool._context.route("https://api.crossref.org/**", _serve_crossref_answer)
    try:
        start = time.monotonic()
        tree = tool.go_to_url(_S2_PAPER_URL)
        elapsed = time.monotonic() - start
        title = _main(tool, "document.title")
    finally:
        tool._context.unroute("https://api.crossref.org/**")
        tool._context.unroute("https://www.semanticscholar.org/**")
        tool._context.unroute("https://*.awswaf.com/**")
    first = tree.splitlines()[0]
    assert first.startswith("Note: www.semanticscholar.org answered with a bot-protection"), first
    assert "(AWS WAF CAPTCHA puzzle)" in first and "show_browser()" in first
    assert "'Real content paper Doe Roe' was looked up on Crossref" in first, first
    assert "2 work(s) found" in first
    assert f"show_browser() and then go_to_url('{_S2_PAPER_URL}')" in first
    assert title == "Crossref: Real content paper Doe Roe"
    assert "Page: Crossref: Real content paper Doe Roe" in tree
    assert tool._page.url.startswith(
        "https://api.crossref.org/works?query.bibliographic=Real+content+paper+Doe+Roe&rows=5&"
    )
    assert 'heading "Real content paper <with> markup" [level=2]' in tree
    assert "Jane Doe, Rob Roe, The Consortium (2024). Journal of Tests" in tree
    assert 'link "https://doi.org/10.1000/real.1"' in tree
    assert "Cited by 7 (Crossref)" in tree
    assert "An abstract of the paper." in tree and "jats" not in tree
    assert 'heading "Another paper" [level=2]' in tree and "Cited by 0 (Crossref)" in tree
    assert "Begin" not in tree
    assert 3 <= elapsed < 15, elapsed


def test_crossref_lookup_survives_a_network_change(tool, server, monkeypatch):
    """A Crossref lookup that a host network change aborts is issued again.

    On 12 Nov 2026 the bench's semanticscholar row and two of four reruns
    fell back to the bare puzzle note because ``page.goto`` to
    api.crossref.org died with ``net::ERR_NETWORK_CHANGED`` (other jobs'
    Docker veth links come and go on this box).  Chromium raises that
    error only for a real interface change, so the first Crossref
    navigation fails with it here through the tool's own ``_goto``; the
    real navigation then runs against the routed Crossref answer.
    """
    real_goto = tool._goto
    crossref_calls: list[str] = []

    def goto(url: str, wait_until: str = "domcontentloaded"):
        if url.startswith(web_stealth.CROSSREF_WORKS_URL):
            crossref_calls.append(url)
            if len(crossref_calls) == 1:
                raise web_stealth.playwright_api().Error(
                    f"Page.goto: net::ERR_NETWORK_CHANGED at {url}"
                )
        return real_goto(url, wait_until)

    tool.go_to_url(f"{server}/inert")
    monkeypatch.setattr(tool, "_goto", goto)
    tool._context.route("https://*.awswaf.com/**", _serve_inert_script)
    tool._context.route("https://www.semanticscholar.org/**", _serve_aws_captcha_page)
    tool._context.route("https://api.crossref.org/**", _serve_crossref_answer)
    try:
        tree = tool.go_to_url(_S2_PAPER_URL)
    finally:
        tool._context.unroute("https://api.crossref.org/**")
        tool._context.unroute("https://www.semanticscholar.org/**")
        tool._context.unroute("https://*.awswaf.com/**")
    first = tree.splitlines()[0]
    assert len(crossref_calls) == 2, crossref_calls
    assert "'Real content paper Doe Roe' was looked up on Crossref" in first, first
    assert "2 work(s) found" in first
    assert 'heading "Real content paper <with> markup" [level=2]' in tree
    assert tool._page.url.startswith(web_stealth.CROSSREF_WORKS_URL)


@pytest.mark.parametrize(
    "crossref",
    [
        lambda route: route.abort("failed"),
        lambda route: route.fulfill(
            status=503, content_type="text/html", body="<title>503</title><p>Service unavailable"
        ),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status": "failed", "message-type": "validation-failure", "message": []}',
        ),
    ],
    ids=["aborted", "html-503", "not-a-works-list"],
)
def test_semanticscholar_puzzle_is_reported_when_crossref_fails(tool, server, crossref):
    """Without a Crossref works list the puzzle is reported as any other block, its page showing."""
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://*.awswaf.com/**", _serve_inert_script)
    tool._context.route("https://www.semanticscholar.org/**", _serve_aws_captcha_page)
    tool._context.route("https://api.crossref.org/**", crossref)
    try:
        tree = tool.go_to_url(_S2_PAPER_URL)
    finally:
        tool._context.unroute("https://api.crossref.org/**")
        tool._context.unroute("https://www.semanticscholar.org/**")
        tool._context.unroute("https://*.awswaf.com/**")
    first = tree.splitlines()[0]
    assert first.startswith("Note: www.semanticscholar.org answered with a bot-protection"), first
    assert "(AWS WAF CAPTCHA puzzle)" in first and "Crossref" not in first
    assert "use another source for the same information" in first
    assert "Page: Human Verification" in tree and "Begin" in tree
    assert tool._page.url == _S2_PAPER_URL


def test_semanticscholar_search_is_answered_from_crossref(tool, server):
    """A search page's query goes to Crossref as a bibliographic query for ten works."""
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://*.awswaf.com/**", _serve_inert_script)
    tool._context.route("https://www.semanticscholar.org/**", _serve_aws_captcha_page)
    tool._context.route("https://api.crossref.org/**", _serve_crossref_answer)
    try:
        tree = tool.go_to_url("https://www.semanticscholar.org/search?q=Real%20content&sort=x")
    finally:
        tool._context.unroute("https://api.crossref.org/**")
        tool._context.unroute("https://www.semanticscholar.org/**")
        tool._context.unroute("https://*.awswaf.com/**")
    first = tree.splitlines()[0]
    assert "'Real content' was looked up on Crossref" in first, first
    assert tool._page.url.startswith(
        "https://api.crossref.org/works?query.bibliographic=Real+content&rows=10&"
    )
    assert 'heading "Real content paper <with> markup" [level=2]' in tree


def test_semanticscholar_puzzle_stays_when_the_user_can_see_it(tool, server, monkeypatch):
    """A page shown to the user is not swapped for Crossref: they may be solving the puzzle."""
    tool.go_to_url(f"{server}/inert")
    monkeypatch.setattr(tool, "_user_sees_page", lambda: True)
    tool._context.route("https://*.awswaf.com/**", _serve_inert_script)
    tool._context.route("https://www.semanticscholar.org/**", _serve_aws_captcha_page)
    tool._context.route("https://api.crossref.org/**", _serve_crossref_answer)
    try:
        tree = tool.go_to_url(_S2_PAPER_URL)
    finally:
        tool._context.unroute("https://api.crossref.org/**")
        tool._context.unroute("https://www.semanticscholar.org/**")
        tool._context.unroute("https://*.awswaf.com/**")
    first = tree.splitlines()[0]
    assert "(AWS WAF CAPTCHA puzzle)" in first and "Crossref" not in first, first
    assert "Page: Human Verification" in tree and tool._page.url == _S2_PAPER_URL


def test_user_sees_page_in_the_browser_tab_a_window_or_their_own_chrome(tool, server):
    """The predicate behind the puzzle's stay: headed Chromium on Xvfb is not visible."""
    tool.go_to_url(f"{server}/inert")
    assert tool._user_sees_page() is False
    for attribute, value in (("_live", True), ("_headless", False), ("_attached_url", "http://x")):
        before = getattr(tool, attribute)
        setattr(tool, attribute, value)
        try:
            assert tool._user_sees_page() is True, attribute
        finally:
            setattr(tool, attribute, before)
    assert tool._user_sees_page() is False


@pytest.mark.parametrize(
    ("afterwards", "reason"),
    [
        (lambda route: route.abort("failed"), "Error"),
        (
            lambda route: route.fulfill(
                status=503, content_type="text/html", body="<title>503</title>Service unavailable"
            ),
            "HTTP 503",
        ),
    ],
    ids=["aborted", "http-503"],
)
def test_semanticscholar_puzzle_not_reopened_is_said_so(tool, server, afterwards, reason):
    """When Crossref fails and the puzzle page cannot be reopened, the note says both."""
    tool.go_to_url(f"{server}/inert")
    hits = {"n": 0}

    def puzzle_then_nothing(route) -> None:
        hits["n"] += 1
        if hits["n"] == 1:
            _serve_aws_captcha_page(route)
        else:
            afterwards(route)

    tool._context.route("https://*.awswaf.com/**", _serve_inert_script)
    tool._context.route("https://www.semanticscholar.org/**", puzzle_then_nothing)
    tool._context.route("https://api.crossref.org/**", lambda route: route.abort("failed"))
    try:
        tree = tool.go_to_url(_S2_PAPER_URL)
    finally:
        tool._context.unroute("https://api.crossref.org/**")
        tool._context.unroute("https://www.semanticscholar.org/**")
        tool._context.unroute("https://*.awswaf.com/**")
    first = tree.splitlines()[0]
    assert first.startswith("Note: www.semanticscholar.org answered with a bot-protection"), first
    assert "Crossref (api.crossref.org) gave no answer" in first
    assert f"could not be reopened afterwards ({reason})" in first, first
    assert f"Call go_to_url('{_S2_PAPER_URL}') again" in first
    assert "complete the check" not in first
    assert hits["n"] == 3  # the puzzle, then the two attempts to reopen it


def test_content_fallback_names_crossref_for_semanticscholar_pages():
    """Paper pages with a title slug and searches with a query map to Crossref, nothing else."""
    fallback = web_stealth.content_fallback
    fields = "DOI,title,author,container-title,issued,abstract,is-referenced-by-count,URL"
    assert fallback(_S2_PAPER_URL) == (
        "https://api.crossref.org/works?query.bibliographic=Real+content+paper+Doe+Roe"
        f"&rows=5&select={fields}",
        "Real content paper Doe Roe",
    )
    assert fallback(
        "https://semanticscholar.org/paper/Meta-Learning%3A-a-survey-Lee/"
        "0123456789ABCDEF0123456789abcdef01234567/"
    ) == (
        "https://api.crossref.org/works?query.bibliographic=Meta+Learning%3A+a+survey+Lee"
        f"&rows=5&select={fields}",
        "Meta Learning: a survey Lee",
    )
    assert fallback("https://www.semanticscholar.org/search?q=Differential%20Testing&sort=x") == (
        "https://api.crossref.org/works?query.bibliographic=Differential+Testing"
        f"&rows=10&select={fields}",
        "Differential Testing",
    )
    # The host is what counts: case, a port and userinfo do not change it.
    for authority in (
        "WWW.SemanticScholar.org",
        "www.semanticscholar.org:443",
        "u:p@semanticscholar.org",
    ):
        assert fallback(f"https://{authority}/search?q=Differential%20Testing") == (
            "https://api.crossref.org/works?query.bibliographic=Differential+Testing"
            f"&rows=10&select={fields}",
            "Differential Testing",
        ), authority
    assert fallback("https://www.semanticscholar.org@evil.example/search?q=x") is None
    # No words to look up: the id alone, an empty query, other pages, other hosts.
    paper_id = "0123456789abcdef0123456789abcdef01234567"
    assert fallback(f"https://www.semanticscholar.org/paper/{paper_id}") is None
    assert fallback("https://www.semanticscholar.org/search?q=%20") is None
    assert fallback("https://www.semanticscholar.org/search") is None
    assert fallback("https://www.semanticscholar.org/author/Jane-Doe/1234") is None
    assert fallback("https://www.semanticscholar.org/") is None
    assert fallback("https://api.semanticscholar.org/graph/v1/paper/search?query=x") is None
    assert fallback("https://www.google.com/search?q=semanticscholar") is None


def test_bench_verdict_tells_expected_puzzles_from_other_blocks(tool, server):
    """The bench passes a known puzzle only when Crossref answered for its vendor's puzzle."""
    spec = importlib.util.spec_from_file_location(
        "web_stealth_bench", _REPO_ROOT / "scripts" / "web_stealth_bench.py"
    )
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://*.awswaf.com/**", _serve_inert_script)
    try:
        puzzle = f"{server}/aws-captcha"
        result = tool.go_to_url(puzzle)
        # A page that is not in the list is simply blocked, and so is a
        # listed page whose puzzle got no Crossref answer.
        assert bench.verdict(tool, puzzle, result).startswith("BLOCKED(AWS WAF CAPTCHA puzzle)")
        bench.KNOWN_PUZZLES[puzzle] = (web_stealth.AWS_WAF_CAPTCHA, "Real content")
        assert bench.verdict(tool, puzzle, result).startswith("BLOCKED(AWS WAF CAPTCHA puzzle)")
        tool._context.route("https://www.semanticscholar.org/**", _serve_aws_captcha_page)
        tool._context.route("https://api.crossref.org/**", _serve_crossref_answer)
        try:
            result = tool.go_to_url(_S2_PAPER_URL)
        finally:
            tool._context.unroute("https://api.crossref.org/**")
            tool._context.unroute("https://www.semanticscholar.org/**")
        # The Crossref page for the expected vendor's puzzle counts as loaded ...
        bench.KNOWN_PUZZLES[_S2_PAPER_URL] = (web_stealth.AWS_WAF_CAPTCHA, "real content paper")
        assert bench.verdict(tool, _S2_PAPER_URL, result).startswith(
            "FALLBACK(AWS WAF CAPTCHA puzzle) Note: www.semanticscholar.org"
        )
        # ... also when listed among several labels the site answers with ...
        bench.KNOWN_PUZZLES[_S2_PAPER_URL] = (
            (web_stealth.CLOUDFLARE_CHALLENGE, web_stealth.AWS_WAF_CAPTCHA),
            "Real content",
        )
        assert bench.verdict(tool, _S2_PAPER_URL, result).startswith(
            "FALLBACK(AWS WAF CAPTCHA puzzle) Note:"
        )
        # ... not for another vendor's, nor unlisted, nor when Crossref's
        # works (not the page title, which names the words searched for)
        # lack the paper.
        bench.KNOWN_PUZZLES[_S2_PAPER_URL] = (web_stealth.CLOUDFLARE_CHALLENGE, "Real content")
        assert bench.verdict(tool, _S2_PAPER_URL, result).startswith("BLOCKED(None) Note:")
        del bench.KNOWN_PUZZLES[_S2_PAPER_URL]
        assert bench.verdict(tool, _S2_PAPER_URL, result).startswith("BLOCKED(None) Note:")
        bench.KNOWN_PUZZLES[_S2_PAPER_URL] = (web_stealth.AWS_WAF_CAPTCHA, "Doe Roe")
        assert bench.verdict(tool, _S2_PAPER_URL, result) == (
            "BLOCKED(AWS WAF CAPTCHA puzzle) Crossref lists no work titled 'Doe Roe'"
        )
        empty = {"status": "ok", "message": {"items": []}}
        tool._context.route("https://www.semanticscholar.org/**", _serve_aws_captcha_page)
        tool._context.route(
            "https://api.crossref.org/**",
            lambda route: route.fulfill(
                status=200, content_type="application/json", body=json.dumps(empty)
            ),
        )
        try:
            result = tool.go_to_url(_S2_PAPER_URL)
        finally:
            tool._context.unroute("https://api.crossref.org/**")
            tool._context.unroute("https://www.semanticscholar.org/**")
        assert "0 work(s) found" in result.splitlines()[0]
        bench.KNOWN_PUZZLES[_S2_PAPER_URL] = (web_stealth.AWS_WAF_CAPTCHA, "real content paper")
        assert bench.verdict(tool, _S2_PAPER_URL, result).startswith(
            "BLOCKED(AWS WAF CAPTCHA puzzle) Crossref lists no work"
        )
    finally:
        tool._context.unroute("https://*.awswaf.com/**")
    # The content page (the site's policy relaxed) is OK; an unrecognised
    # page that is not the content either is the failure the list exists for.
    content = f"{server}/press-hold-done"
    result = tool.go_to_url(content)
    bench.KNOWN_PUZZLES[content] = (web_stealth.AWS_WAF_CAPTCHA, "Real content")
    assert bench.verdict(tool, content, result) == "OK"
    bench.KNOWN_PUZZLES[content] = (web_stealth.AWS_WAF_CAPTCHA, "Semantic Scholar")
    assert bench.verdict(tool, content, result).startswith("BLOCKED(None) unrecognised page")
    assert bench.verdict(tool, f"{server}/other", result) == "OK"
    assert bench.verdict(tool, content, "Error navigating: boom").startswith("ERROR ")
    # A recognised block that no fallback answers (Reddit's page) is a
    # block, recognised or not: no page of that kind is in the list.
    blocked = f"{server}/reddit-block"
    result = tool.go_to_url(blocked)
    assert bench.verdict(tool, blocked, result).startswith(
        "BLOCKED(Reddit network-policy block) Note: 127.0.0.1:"
    )
    assert not any("reddit.com" in url or "stackoverflow.com/search" in url for url in bench.URLS)


# What the live Cloudflare attempts give the challenge to clear.  The
# product's 12 s (``web_stealth._CHALLENGE_WAIT_SECS``) is the agent's
# budget for a page that keeps its interstitial; the test below checks
# the rating, not the budget.  Traced under the full parallel run
# (2026-11-02, 3 of 35 attempts; never alone, under CPU load alone or
# under Docker container churn alone): the challenge page's first
# request to Cloudflare stalls and the page shows no Turnstile frame at
# all until Cloudflare's own retry reloads it (``?__cf_chl_rt_tk=``)
# after 45 s; the checkbox then shows, is ticked, and the content came
# at 53-56 s.  Every other attempt cleared in 6-8 s, as alone.  One such
# retry plus the usual clear fits in 75 s.
_LIVE_CHALLENGE_WAIT_SECS = 75.0


def test_live_cloudflare_protected_site_loads(tool, monkeypatch):
    """A real Cloudflare-protected publisher that challenged this network loads.

    journals.sagepub.com shows the Turnstile checkbox to every
    configuration from this machine; with a fresh profile (no clearance
    cookie) the box must be ticked and the page come back as content.
    Skipped offline.
    """
    try:
        import socket

        socket.create_connection(("journals.sagepub.com", 443), timeout=5).close()
    except OSError:
        pytest.skip("no Internet access")
    monkeypatch.setattr(web_stealth, "_CHALLENGE_WAIT_SECS", _LIVE_CHALLENGE_WAIT_SECS)
    # Two fresh browsers at most: a profile that is always challenged
    # (the browser rated a bot, its tick refused) still fails on the
    # second try.
    tree = ""
    for _attempt in range(2):
        tool.close()
        # Fresh profile: no clearance cookie from earlier tests nor from a
        # first attempt whose challenge cleared late.
        assert tool.user_data_dir
        shutil.rmtree(tool.user_data_dir, ignore_errors=True)
        tree = tool.go_to_url("https://journals.sagepub.com/")
        if tree.startswith("Page: Sage Journals"):
            break
    assert tree.startswith("Page: Sage Journals"), tree[:200]


def test_challenge_vendor_recognises_each_vendor():
    """Titles/bodies seen in the task history map to their vendor labels."""
    cases = {
        ("Just a moment...", "Performing security verification"): "Cloudflare challenge",
        ("stackoverflow.com", "Verifying you are human. Ray ID: 9a"): "Cloudflare challenge",
        ("Attention Required! | Cloudflare", "Sorry, you have been blocked"): "Cloudflare block",
        ("medium.com", "Sorry, you have been blocked ... Cloudflare Ray ID"): "Cloudflare block",
        ("Access Denied", "Reference #18.x errors.edgesuite.net"): "Akamai block",
        ("", "You don't have permission to access ... Reference #18.2"): "Akamai block",
        ("Oh noes!", "Sad Anubis"): "Anubis proof-of-work check",
        ("Egyptian gods", "Anubis was the god of the dead"): None,
        ("Making sure you're not a bot!", "Loading..."): "Anubis proof-of-work check",
        ("dblp", "Oh noes! Sad Anubis: the check failed"): "Anubis proof-of-work check",
        ("", "Request unsuccessful. Incapsula incident ID: 12"): "Imperva Incapsula block",
        ("Robot or human?", "Press & Hold"): "PerimeterX press-and-hold check",
        ("Walmart", "Hold the button to confirm that you're human. Press & Hold"):
            "PerimeterX press-and-hold check",
        # walmart.com/blocked, fiverr and PerimeterX's stock page, as seen live.
        ("", "Robot or human?\n\nActivate and hold the button to confirm that you\u2019re human"):
            "PerimeterX press-and-hold check",
        ("It needs a human touch", "Complete the task and we\u2019ll get you right back"):
            "PerimeterX press-and-hold check",
        ("Access to this page has been denied.", "Please verify you are a human"):
            "PerimeterX press-and-hold check",
        ("zillow.com", "Press & Hold ... Access to this page has been denied because we believe "
                       "you are using automation tools to browse the website. Reference ID"):
            "PerimeterX press-and-hold check",
        ("Are you a robot?", "ScienceDirect"): "CAPTCHA",
        ("Security check required", ""): "CAPTCHA",
        ("https://www.google.com/search",
         "Our systems have detected unusual traffic from your computer network. IP address: 1.2"):
            "Google 'unusual traffic' page",
        # Ordinary content that merely talks about these things.
        ("Newest 'python' Questions - Stack Overflow", "Ask Question"): None,
        ("CAPTCHA accessibility guide", "A captcha asks: are you a robot? Security check required"):
            None,
        ("ACL tutorial", "The message 'You don't have permission to access' means ..."): None,
        ("Blog: unusual traffic", "Google shows 'unusual traffic from your computer network'"):
            None,
        ("Docs", "Just a moment... while the demo loads; verify you are human later"): None,
        # Two articles about the widget that the old 'press and hold' + 'human'
        # pair mistook for it (docs.humansecurity.com, scrapfly.io).
        ("HUMAN Challenge | HUMAN Documentation",
         "HUMAN Challenge is a simple, no-hassle \u201cpress and hold\u201d challenge ... "
         "detect if a user is a bot or a human"): None,
        ("How to scrape Perimeter X: Please verify you are human?",
         "we might encounter messages like \"Please verify you are Human: Press & Hold\""): None,
        # AWS WAF's CAPTCHA page (semanticscholar.org, 30 Oct 2026) by its
        # wording; its title alone is too generic.
        ("Human Verification",
         "Let's confirm you are human\n\nComplete the security check before continuing. This "
         "step verifies that you are not a bot, which helps to protect your account and "
         "prevent spam.\n\nBegin\nSelect language"): "AWS WAF CAPTCHA puzzle",
        ("Human Verification", "Our identity verification product"): None,
        ("AWS WAF CAPTCHA docs", "the page says 'Let's confirm you are human' and shows"): None,
        # Stack Overflow's "Human verification" page (3 Nov 2026) by its title
        # and its instruction, on any Stack Exchange site; not a question with
        # that title, nor a meta post quoting the page.
        ("Human verification - Stack Overflow",
         "Skip to main content\nAbout\n...\nHuman verification\n\nAre you a human being? We "
         "apologize for the confusion, but we can't quite tell if you're a person or a "
         "script.\n\nCheck the CAPTCHA box, and we'll be out of your way."):
            "Stack Overflow 'Human verification' CAPTCHA",
        ("Human verification - Software Engineering Stack Exchange",
         "Check the CAPTCHA box, and we'll be out of your way."):
            "Stack Overflow 'Human verification' CAPTCHA",
        ("Human verification - Super User", "Check the CAPTCHA box"):
            "Stack Overflow 'Human verification' CAPTCHA",
        ("Human verification - Stack Overflow", "How do I verify a human in my app? Ask Question"):
            None,
        ("Why does the site show 'Human verification'? - Meta Stack Exchange",
         "The page says: Are you a human being? ... Check the CAPTCHA box, and we'll be out "
         "of your way."): None,
        # Reddit's two block pages (3 Nov 2026): the sentence opens the page,
        # which articles about the error never do.
        ("", "You've been blocked by network security.\nTo continue, log in to your Reddit "
             "account or use your developer token\n\nIf you think you've been blocked by "
             "mistake, file a ticket below and we'll look into it.\nLog in\nFile a ticket"):
            "Reddit network-policy block",
        ("", "You\u2019ve been blocked by network security. To continue, log in"):
            "Reddit network-policy block",
        ("Blocked", "whoa there, pardner!\n\nYour request has been blocked due to a network "
                    "policy.\n\nTry logging in or creating an account here"):
            "Reddit network-policy block",
        ("Blocked", "A short story about a goalkeeper"): None,
        ("Reddit blocked by network security? Here's why - VoidMob",
         "Reddit's \"You've been blocked by network security\" page means Reddit has blocked "
         "the request coming from your connection. To continue, log in to your Reddit account"):
            None,
        ("", "Menu\nYou've been blocked by network security"): None,
    }
    for (title, body), expected in cases.items():
        assert web_stealth.challenge_vendor(title, body) == expected, (title, body)


def test_block_advice_names_the_way_out():
    """The note's advice is the vendor's own for pages that rate the address, generic otherwise."""
    reddit = web_stealth.block_advice(web_stealth.REDDIT_BLOCK)
    assert "log in to Reddit" in reddit and "old.reddit.com" in reddit and ".json" in reddit
    so = web_stealth.block_advice(web_stealth.STACK_OVERFLOW_CAPTCHA)
    assert "picture puzzle" in so and "site:stackoverflow.com" in so
    for vendor in ("Akamai block", web_stealth.CLOUDFLARE_CHALLENGE, web_stealth.AWS_WAF_CAPTCHA):
        assert web_stealth.block_advice(vendor) == (
            "Call show_browser() so the user can complete the check, or use another source "
            "for the same information."
        )
    for advice in (reddit, so):
        assert "show_browser()" in advice


def test_challenge_vendor_knows_aws_waf_pages_by_their_scripts():
    """AWS WAF's interstitials are told by their scripts: the blank one needs a blank page."""
    waf = "https://c09bc9c04079.35039563.us-east-2.token.awswaf.com/c09bc9c04079/4996efde3854"
    challenge = f"{waf}/fba16e1c8e93/challenge.js"
    captcha = f"{waf.replace('.token.', '.captcha.')}/fba16e1c8e93/captcha.js"
    vendor = web_stealth.challenge_vendor
    # The Challenge action's page: empty title, no text, the token script.
    assert vendor("", "", challenge) == "AWS WAF challenge"
    assert vendor("", "  \n", f"https://cdn.example.com/app.js {challenge}") == "AWS WAF challenge"
    # A site that embeds the same SDK script on its own pages (which have text).
    assert vendor("Semantic Scholar", "Search 200M papers", challenge) is None
    assert vendor("", "Loading...", challenge) is None
    # The CAPTCHA page in another language is known by its puzzle script.
    assert vendor("Human Verification", "Best\u00e4tigen Sie, dass Sie ein Mensch sind",
                  f"{challenge} {captcha}") == "AWS WAF CAPTCHA puzzle"
    # The puzzle script on a site's own page (AWS WAF's CAPTCHA JavaScript
    # API renders the puzzle inside a login form) is content, not a block.
    assert vendor("Store checkout", "Your shopping cart", f"{challenge} {captcha}") is None
    assert vendor("", "", captcha) is None
    # Other scripts, and unrelated hosts or paths that merely mention the names.
    assert vendor("", "", "https://cdn.example.com/challenge.js") is None
    assert vendor("", "", "https://awswaf.com.evil.example/x/captcha.js") is None
    assert vendor("", "", "https://cdn.example.com/.awswaf.com/a/challenge.js") is None
    mention = "https://cdn.example.com/sdk.js?url=https://x.awswaf.com/a/challenge.js"
    assert vendor("", "", mention) is None
    assert vendor("Human Verification", "", f"{captcha}.map") is None
    assert vendor("Human Verification", "", f"{captcha};not-js") is None
    assert vendor("", "", f"http://{challenge[8:]}") is None
    assert vendor("", "", f"{challenge}?v=2") == "AWS WAF challenge"
    assert vendor("", "", "https://x.awswaf.com:443/a/b/challenge.js") == "AWS WAF challenge"
    assert vendor("", "", "HTTPS://X.AWSWAF.COM/a/b/challenge.js") == "AWS WAF challenge"
    # Script sources that are not URLs (Chromium keeps them as written)
    # are skipped, not raised on, and do not hide a vendor script after them.
    for bad in ("https://[broken/challenge.js", "https://x.awswaf.com:bad/challenge.js",
                "https://x.awswaf.com:65536/challenge.js"):
        assert vendor("", "", bad) is None, bad
        assert vendor("", "", f"{bad} {challenge}") == "AWS WAF challenge", bad
    assert vendor("", "", "") is None
    assert vendor("inert", "a page") is None


def test_content_with_a_malformed_script_source_is_still_content(tool, server):
    """A page whose script ``src`` is not a URL is returned as content, not as an error."""
    tree = tool.go_to_url(f"{server}/malformed-script")
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree
    assert tool._challenge_vendor() is None


def test_search_fallback_url_maps_google_queries_to_bing():
    """Google search / sorry URLs become the same query on Bing; others do not."""
    assert web_stealth.search_fallback_url(
        "https://www.google.com/search?q=playwright+stealth&hl=en"
    ) == "https://www.bing.com/search?q=playwright+stealth"
    sorry = (
        "https://www.google.com/sorry/index?continue=https://www.google.com/search"
        "%3Fq%3Dpatchright%2Bpython&q=EhAmAGAlD2ecTQ"
    )
    assert web_stealth.search_fallback_url(sorry) == (
        "https://www.bing.com/search?q=patchright+python"
    )
    assert web_stealth.search_fallback_url("https://www.google.com/sorry/index") is None
    assert web_stealth.search_fallback_url("https://www.google.com/maps") is None
    assert web_stealth.search_fallback_url("https://www.google.com/search") is None
    assert web_stealth.search_fallback_url("https://www.bing.com/search?q=x") is None
    assert web_stealth.search_fallback_url("https://google.co.uk/search?q=tea") == (
        "https://www.bing.com/search?q=tea"
    )


def test_google_unusual_traffic_page_is_reported_when_no_query_is_known(tool, server):
    """A Google-style block without a recoverable query gets the generic note."""
    tree = tool.go_to_url(f"{server}/sorry/index")
    first = tree.splitlines()[0]
    assert first.startswith("Note:") and "(Google 'unusual traffic' page)" in first


def test_google_unusual_traffic_page_falls_back_to_bing(tool):
    """A real Google search from this network is answered on Bing when blocked.

    Talks to google.com and bing.com; skipped when offline.
    """
    try:
        import socket

        socket.create_connection(("www.google.com", 443), timeout=5).close()
    except OSError:
        pytest.skip("no Internet access")
    tree = tool.go_to_url("https://www.google.com/search?q=patchright+python")
    if "unusual traffic" not in tree.splitlines()[0]:
        pytest.skip("Google did not block this network right now")
    assert "opened on Bing instead" in tree.splitlines()[0]
    assert "URL: https://www.bing.com/search?q=patchright+python" in tree


def test_headless_fallback_still_waits_out_challenges(tool, server):
    """The challenge logic does not depend on the display mode."""
    tool.close()  # one sync driver per thread
    saved_path = os.environ["PATH"]
    web_stealth.stop_virtual_display()
    os.environ["PATH"] = str(os.devnull)
    try:
        web = WebUseTool(user_data_dir=None)
        try:
            tree = web.go_to_url(f"{server}/challenge/{time.time_ns()}")
            assert tree.startswith("Page: Real content")
            assert web._chromium_headless is True
        finally:
            web.close()
    finally:
        os.environ["PATH"] = saved_path


def test_headed_user_agent_helpers():
    """The token is rewritten natively and the headed string is remembered per channel."""
    raw = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 HeadlessChrome/154.0.0.0 Safari/537"
    headed = web_stealth.headed_user_agent(raw)
    assert headed == raw.replace("HeadlessChrome", "Chrome")
    assert headed is not None and web_stealth.headed_user_agent(headed) is None
    assert web_stealth.headed_user_agent("") is None
    channel = f"test-{time.time_ns()}"
    assert web_stealth.user_agent_switch(channel) is None
    web_stealth.remember_user_agent(channel, headed)
    assert web_stealth.user_agent_switch(channel) == f"--user-agent={headed}"
    web_stealth.forget_user_agent(channel)
    assert web_stealth.user_agent_switch(channel) is None
    web_stealth.forget_user_agent(channel)  # idempotent
    assert web_stealth.user_agent_matches_version(headed, "154.0.8037.57")
    assert web_stealth.user_agent_matches_version(f"--user-agent={headed}", "154.0.8037.57")
    assert not web_stealth.user_agent_matches_version(headed, "155.0.8100.1")
    assert not web_stealth.user_agent_matches_version("Mozilla/5.0 Firefox/154.0", "154.0.1")


def test_headless_fallback_uses_a_native_user_agent(tool, server):
    """Real headless mode reports a headed Chrome everywhere, with nothing patched.

    The blocks of Oct 2026 came from the macOS daemon, where there is no
    Xvfb: Cloudflare's challenge rejected every headless browser whose
    ``navigator.userAgent`` had been rewritten by an init script while
    ``navigator.appVersion`` still said ``HeadlessChrome``.  With Chrome's
    own ``--user-agent`` switch the header, ``userAgent`` and
    ``appVersion`` agree, the property stays on the prototype, and the
    client-hint brands are still sent.  The headed string is learnt from
    the first launch (one relaunch) and remembered for the next; one
    remembered for another Chrome build is forgotten and relearnt.
    """
    tool.close()  # one sync driver per thread
    saved_path = os.environ["PATH"]
    web_stealth.stop_virtual_display()
    os.environ["PATH"] = str(os.devnull)
    channel = web_stealth.chrome_channel()
    remembered = web_stealth._HEADED_USER_AGENTS.pop(channel, None)
    launches: list[list[str]] = []
    original_launch = WebUseTool._launch_browser

    def counting_launch(self, launcher, kwargs):
        launches.append(list(kwargs["args"]))
        original_launch(self, launcher, kwargs)

    WebUseTool._launch_browser = counting_launch  # type: ignore[method-assign]
    try:
        for expected_launches in (2, 3):
            web = WebUseTool(user_data_dir=None)
            try:
                web.go_to_url(f"{server}/headers")
                assert web._chromium_headless is True
                probe = _main(
                    web,
                    "() => ({ua: navigator.userAgent, av: navigator.appVersion,"
                    " own: Object.getOwnPropertyDescriptor(navigator, 'userAgent') !== undefined,"
                    " brands: navigator.userAgentData.brands.map(b => b.brand),"
                    " echo: JSON.parse(document.getElementById('h').textContent)})",
                )
                assert isinstance(probe, dict)
                assert "HeadlessChrome" not in probe["ua"] and "Chrome/" in probe["ua"]
                assert "HeadlessChrome" not in probe["av"] and probe["ua"].endswith(probe["av"])
                assert probe["own"] is False, "userAgent must stay on Navigator.prototype"
                assert probe["echo"]["user_agent"] == probe["ua"]
                assert "HeadlessChrome" not in probe["echo"]["sec_ch_ua"]
                assert probe["echo"]["sec_ch_ua"], "client-hint brands are still sent"
                assert any(b in ("Chromium", "Google Chrome") for b in probe["brands"])
                assert web._page.evaluate("navigator.appVersion") == probe["av"]
            finally:
                web.close()
            # First tool: a plain launch, then the relaunch with the switch.
            # Second tool: the remembered switch from the start, no relaunch.
            assert len(launches) == expected_launches, launches
            assert any(a.startswith("--user-agent=") for a in launches[-1])
            assert web_stealth.user_agent_switch(channel) == launches[-1][-1]
        good = launches[-1][-1]
        # Chrome updated under the daemon: the remembered string names
        # the old major version.  That launch is discarded, the build is
        # asked again and the relaunch carries the current string.
        stale = re.sub(r"Chrome/\d+\.", "Chrome/1.", good)
        web_stealth.remember_user_agent(channel, stale[len("--user-agent=") :])
        web = WebUseTool(user_data_dir=None)
        try:
            web.go_to_url(f"{server}/headers")
            assert _main(web, "navigator.userAgent") == good[len("--user-agent=") :]
        finally:
            web.close()
        assert [a[-1] for a in launches[3:]] == [stale, launches[0][-1], good], launches[3:]
        assert web_stealth.user_agent_switch(channel) == good
    finally:
        WebUseTool._launch_browser = original_launch  # type: ignore[method-assign]
        os.environ["PATH"] = saved_path
        if remembered is not None:
            web_stealth.remember_user_agent(channel, remembered)
    assert not any(a.startswith("--user-agent=") for a in launches[0])
    assert len(launches) == 6


def test_tab_list_uses_the_selected_api_timeout_error(tool, server):
    """tab:list survives an unresponsive tab under the Patchright API."""
    tool.go_to_url(f"{server}/inert")
    listing = tool.go_to_url("tab:list")
    assert listing.startswith("Open tabs (") and "(active)" in listing


def test_typing_deadline_scales_with_text(tool, server):
    """A long text is typed completely: the watchdog does not fire early."""
    tree = tool.go_to_url(f"{server}/events")
    box_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'textbox "Name"' in line
    )
    text = "word " * 40
    tool.type_text(box_id, text.strip())
    assert tool._page.evaluate("document.getElementById('box').value") == text.strip()
    assert tool._is_alive()


_CONCURRENT_API_SCRIPT = r"""
import json, sys, threading
from importlib import import_module
from kiss.agents.sorcar import web_stealth

workers = 16
barrier = threading.Barrier(workers)
unguarded = []

def body():
    barrier.wait()
    api = web_stealth.playwright_api()
    sync_base = import_module(api.__name__.split(".")[0] + "._impl._sync_base").SyncBase
    if not getattr(sync_base, "_kiss_dead_driver_guard", False):
        unguarded.append(1)

threads = [threading.Thread(target=body) for _ in range(workers)]
for t in threads:
    t.start()
for t in threads:
    t.join()

api = web_stealth.playwright_api()
sync_base = import_module(api.__name__.split(".")[0] + "._impl._sync_base").SyncBase
layers = 0
fn = sync_base._sync
while fn.__qualname__.endswith("_guard_sync_api_against_dead_driver.<locals>._sync"):
    layers += 1
    fn = next(c.cell_contents for c in fn.__closure__ if callable(c.cell_contents))
print(json.dumps({"layers": layers, "unguarded": len(unguarded)}))
"""


@pytest.mark.timeout(120)
def test_concurrent_first_callers_get_one_guarded_api():
    """Task threads racing the first ``playwright_api`` call share one guarded module.

    Unlocked, two threads could both import and both wrap
    ``SyncBase._sync`` (a guard around a guard), and a thread that saw
    the module published before its guard was installed drove an
    unguarded API.  A fresh interpreter makes every thread a first
    caller; the result must be exactly one wrapper layer and no caller
    returning before the guard is in place.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _CONCURRENT_API_SCRIPT],
        capture_output=True,
        text=True,
        timeout=100,
        check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {"layers": 1, "unguarded": 0}
