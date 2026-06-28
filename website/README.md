# kisssorcar.github.io update — § 02 "All you need is a short prompt"

This directory contains a drop-in updated `index.html` for the
[kisssorcar.github.io](https://github.com/kisssorcar/kisssorcar.github.io)
website (the live site is hosted at <https://kisssorcar.github.io/>).

## What changed

A new section `§ 02 — All you need is a short prompt` was inserted between
the existing `§ 01 — Compare` and `§ 02 — What is in the Name` sections.
All subsequent section numbers (§ 02 → § 03, …, § 09 → § 10) were shifted
by one, and a `Prompts` link was added to the top nav. No other content
was changed.

The new section shows a deck of panels, one panel per entry from:

- `./src/kiss/SAMPLE_TASKS.md` &mdash; **12 sample tasks** (rendered with a
  `Task` badge), and
- `./src/kiss/INJECTIONS.md` &mdash; **15 promptlet tricks** (rendered with
  a `Trick` badge).

Each panel has:

- A meaningful title.
- A 1&ndash;2 sentence description.
- The actual prompt rendered inside a fixed-height (180&nbsp;px) scrollable
  area with a fade-out at the bottom.
- A `more …` button that expands the panel to show the entire prompt
  (button toggles to `less …`).

Panels are laid out one after another (single column) so the page reads
top-to-bottom on every screen size.

## Files

- `kisssorcar.github.io/index.html` &mdash; the updated page (drop-in
  replacement for the file in the website repo).
- `screenshots/` &mdash; screenshots captured while testing the page in a
  local browser served from `python3 -m http.server`:
  - `01-prompts-top.png` &mdash; top of the page after `#prompts` jump.
  - `02-prompts-mid.png` &mdash; middle of the prompt deck.
  - `03-prompts-expanded.png` &mdash; one panel expanded via `more …`
    (button shows `less ▾` with `aria-expanded="true"`).
  - `04-tricks-section.png` &mdash; further down the deck showing the
    `Promptlet tricks — INJECTIONS.md` group.

## How to push

```bash
# Clone the website repo somewhere convenient
git clone https://github.com/kisssorcar/kisssorcar.github.io.git
cd kisssorcar.github.io

# Copy in the updated index.html from this artifact directory
cp /path/to/this/repo/website/kisssorcar.github.io/index.html ./index.html

# Review and commit
git diff
git add index.html
git commit -m "Add § 02 'All you need is a short prompt' deck"
git push origin main
```

GitHub Pages will redeploy automatically.
