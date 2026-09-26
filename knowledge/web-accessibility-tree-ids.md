---
title: Browser accessibility tree and [N] element ids
uuid: 7a8e78ee-9022-4fa2-8a29-73438c30a2c6
summary: How WebUseTool numbers interactive elements in the ARIA snapshot ([N] ids,
  INTERACTIVE_ROLES) and resolves an id to a locator via get_by_role and occurrence
  index.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Browser accessibility tree and [N] element ids

## Rendering the tree

`WebUseTool._get_ax_tree` calls `page.locator("body").aria_snapshot()` (10 s timeout,
`_PAGE_READ_TIMEOUT_MS`) and prefixes `Page: <title>` and `URL: <url>`. Output longer than
`max_chars` (50,000) is cut with `... [truncated]`. An empty snapshot returns `(empty page)`.

`_number_interactive_elements` walks the snapshot line by line (regex `_ROLE_LINE_RE` matches
`- role "name" ...`). A line whose role is in `INTERACTIVE_ROLES` gets a running id:
`- [7] button "Sign in"`. The roles are: link, button, textbox, searchbox, combobox, checkbox,
radio, switch, slider, spinbutton, tab, menuitem, menuitemcheckbox, menuitemradio, option,
treeitem. Headings, text and images are left unnumbered.

For each numbered element it stores a dict in `self._elements`:
`role`, `name` (unescaped from the quoted snapshot text), `occurrence` (index among elements
with the same role+name, in document order) and `role_occurrence` (index among all elements of
that role). The tallies use `Counter`s so the pass is linear on large pages.

## Resolving an id

`_resolve_locator(element_id)`:
1. If the id is out of range of the cached list, the snapshot is re-taken once and renumbered;
   still out of range raises `ValueError("Element with ID N not found.")`.
2. Named element: `page.get_by_role(role, name=name, exact=True)` and pick `.nth(occurrence)`.
   Unnamed element: `page.get_by_role(role)` and pick `.nth(role_occurrence)` (because
   `get_by_role(role)` matches named and unnamed elements alike).
3. If only one match, use it; if the recorded occurrence is past the match count (page
   changed), fall back to the first visible match.

Why occurrence matters: both the ARIA snapshot and `get_by_role` enumerate in document order,
so the recorded index targets the exact element the id was assigned to, not the first
visible element that shares role and name (pages with several identical "Edit" or "Reply"
links would otherwise always hit the first one).

## Gotchas

- Ids are only valid for the most recent tree. Any tool that returns a tree replaces
  `self._elements`; after navigation, re-read before clicking.
- `locator.count()` has no timeout in Playwright, so `_resolve_locator` first calls
  `_require_responsive_renderer` and runs the count under `_input_hang_watchdog`
  (see `web-browser-profile-and-launch`).
- `get_page_content(text_only=True)` returns `inner_text("body")` with no ids; use it for
  reading, the default mode for acting.
- `click(action="hover")` only moves the pointer. After a click, `_check_for_new_tab` adopts a
  newly opened tab so the returned tree is for the new page.

## Sources
- `src/kiss/agents/sorcar/web_use_tool.py` (`INTERACTIVE_ROLES`, `_number_interactive_elements`, `WebUseTool._get_ax_tree`, `WebUseTool._resolve_locator`, `WebUseTool.get_page_content`, `WebUseTool._check_for_new_tab`)
