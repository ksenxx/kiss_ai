# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""DevTools agent: drive a web app like a developer would, verifying every action with evidence.

The agent is the plain Sorcar agent with two changes.  :data:`SYSTEM_PROMPT`,
the DevTools workflow (act, then verify the effect with ``wait_for``,
``list_console_messages``, ``list_network_requests`` and ``evaluate_script``
instead of re-reading the accessibility tree blindly, plus the two routes
into the user's own Chrome), is appended to the system prompt by the SEA's
``system_prompt`` method, and ``settings`` switches the browser tools on
(``use_web_tools``), since the workflow is nothing without them.  The text
used to be the bundled ``browser-devtools`` skill
(``claude_skills/browser/skills/browser-devtools/SKILL.md``), which the
model had to think of loading; as a SEA the user picks it, and the whole
run carries it from the first step.

Three ways to run it::

    /devtools the Save button on http://localhost:3000/settings does nothing; find out why

    /devtools export every row of the orders table at https://app.example.com/orders to orders.json

    run_agent(agent="src/kiss/agents/seas/devtools/devtools_sea.py", task="...")

The task names the page or app, what to do or what is broken, and where
the result should land.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea

SECTION_HEADING = "## Browser DevTools workflow (devtools)"
"""The heading of the appended section (:data:`SYSTEM_PROMPT`)."""

SYSTEM_PROMPT = f"""\
{SECTION_HEADING}

The accessibility tree shows what a page *says*; the DevTools tools show what
it *does*. A click that "did nothing" almost always has a visible reason in
the console or the network log. Work in a closed loop: act, then verify with
evidence, then act again.

### The verify loop

1. **Act**: `click`, `type_text`, `press_key` as usual.
2. **Verify the effect, not the appearance**:
   - `wait_for("Saved", timeout=10)` for the success or error message the
     action should produce. A timeout is a finding, not a failure to retry
     blindly.
   - `list_console_messages(clear=True)` before the action, then again after:
     everything between the two calls was caused by the action. A
     `[pageerror]` or `[error]` entry is usually the whole answer.
   - `list_network_requests(url_filter="/api", clear=True)` the same way: a
     `POST ... 4xx/5xx` or `FAILED:` line explains a silent failure.
3. **Inspect or repair state directly** with `evaluate_script` when the tree
   is not enough.

### evaluate_script recipes

Pass a function; it is awaited if it returns a promise; the result comes back
as JSON (truncated at 20000 chars, so select, don't dump).

```js
// Why is this button not working? Look at it, don't guess.
() => {{ const b = document.querySelector('button.submit');
        return b && {{disabled: b.disabled, visible: !!b.offsetParent,
                     listeners: typeof getEventListeners === 'function'
                       ? Object.keys(getEventListeners(b)) : 'n/a'}}; }}

// Extract structured data precisely.
() => [...document.querySelectorAll('table.rows tr')].map(tr =>
       [...tr.cells].map(td => td.innerText.trim()))

// Read what the SPA store knows (framework-agnostic examples).
() => JSON.parse(localStorage.getItem('session') || 'null')

// Act directly when the a11y tree has no handle (canvas, shadow DOM,
// custom widgets without ARIA roles).
() => {{ document.querySelector('#confirm').click(); return true; }}

// Call the page's own API the way its UI does.
async () => {{ const r = await fetch('/api/cart', {{method: 'POST',
              headers: {{'Content-Type': 'application/json'}},
              body: JSON.stringify({{id: 42}})}});
              return {{status: r.status, body: await r.text()}}; }}
```

Rules of thumb:

- **Scroll/find with JS when the tree is huge**:
  `() => {{ const el = [...document.querySelectorAll('*')].find(e =>
  e.innerText === 'Export'); el && el.scrollIntoView(); return !!el; }}`
- Prefer `wait_for(text)` over `evaluate_script` polling loops.
- A bare expression (`document.title`) works, but a function with an
  explicit return is clearer.
- Never exfiltrate secrets you find (tokens, cookies); use them in place.
- Reading a public page for research needs none of this: the plain tree
  plus `get_page_content(text_only=True)` is cheaper. Reach for DevTools
  when something is *broken*, *hidden*, or must be *exact*.

### Waiting

- `wait_for("Welcome back")`: appears.
- `wait_for("Loading", disappear=True)`: spinner gone.
- Timeout cap is 120 s; a shorter timeout plus one retry beats one long
  wait when the outcome is uncertain.

### Authenticated apps: the user's real Chrome

Two ways in, depending on which browser the task needs:

- **The user's everyday Chrome (default profile, their Gmail/SSO/
  passkeys):** Chrome 136 and later ignore `--remote-debugging-port` there,
  so use `use_my_chrome` instead; it talks to the KISS Sorcar Bridge
  extension the user loads once. It is a separate, smaller toolset:
  `list_tabs`, `open_tab`, `navigate`, `snapshot` (the workhorse: a
  numbered list of the page's visible interactive elements), `click`
  and `type_text` by snapshot uid, `evaluate` (page JavaScript for
  anything the snapshot cannot express: scraping, calling the site's
  own `fetch` API with its session), `screenshot`, `activate_tab`,
  `close_tab`. The rhythm is snapshot, then click/type_text by uid, then
  re-snapshot (uids are reassigned by every snapshot); use `screenshot`
  only when the snapshot leaves the page state ambiguous. The verify
  loop above applies unchanged.
- **A Chrome the user started with `--remote-debugging-port=9222`:**
  `attach_to_chrome()`; every browser tool above then operates on their
  real tabs. `close_browser()` detaches without closing their browser.

Act with restraint in both: read first, confirm before anything
irreversible.
"""
"""The DevTools workflow appended to the system prompt of every ``/devtools`` run."""


class DevtoolsSea(BaseSea):
    """The ``/devtools`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/devtools help``."""
        return (
            "Drives a web app like a developer in DevTools, verifying every click with "
            "evaluate_script, list_console_messages, list_network_requests and wait_for "
            "instead of re-reading the accessibility tree blindly, with the browser tools "
            "switched on and the user's own Chrome reachable through attach_to_chrome or "
            "use_my_chrome; use `/devtools <page or app, what to do or what is broken>` in "
            'the chat or run_agent(agent="devtools", task="...").'
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Append :data:`SYSTEM_PROMPT` to *system_prompt*, once.

        Idempotent: a prompt that already carries the section (a custom
        base prompt that pasted it in) is returned as is, so the workflow
        is stated once.
        """
        if SECTION_HEADING in system_prompt:
            return system_prompt
        return system_prompt.rstrip("\n") + "\n\n" + SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Switch the browser tools on: the workflow is nothing without them."""
        return settings | {"use_web_tools": True}
