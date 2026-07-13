# kisssorcar.github.io — website artifact mirror

This directory contains a full mirror of the
[kisssorcar.github.io](https://github.com/kisssorcar/kisssorcar.github.io)
website repo (the live site is hosted at <https://kisssorcar.github.io/>).

## llms.txt + pure-Markdown docs (shipped & live)

The site now ships an [llms.txt](https://llmstxt.org/) content index plus
pure-Markdown documentation so LLMs and coding assistants can index and
recommend KISS Sorcar. These files are **already committed and pushed to the
live site repo** (commit "Ship llms.txt + pure-Markdown docs for
LLM/coding-assistant indexing") and verified live:

- `llms.txt` — spec-compliant index (H1 + blockquote summary + `## Docs`,
  `## Papers`, `## Source & Install`, `## Optional` link sections), live at
  <https://kisssorcar.github.io/llms.txt>.
- `.well-known/llms.txt` — mirror of the same file at the `.well-known` path.
- `llms-full.txt` — all Markdown docs concatenated for single-fetch ingestion,
  live at <https://kisssorcar.github.io/llms-full.txt>.
- `docs/*.md` — 10 pure-Markdown pages: `index.md`, `overview.md`,
  `installation.md`, `cli.md`, `api.md`, `models.md`, `messaging-agents.md`,
  `sample-tasks.md`, `prompt-tricks.md`, `tips.md` (content sourced from
  `README.md`, `API.md`, `src/kiss/SAMPLE_TASKS.md`, `src/kiss/INJECTIONS.md`,
  and `src/kiss/TIPS.md`).
- `index.html.md` — plain-Markdown twin of the homepage.
- `robots.txt` — allows all crawlers, references llms.txt and the sitemap.
- `sitemap.xml` — lists the HTML homepage plus every Markdown/txt page.
- `.nojekyll` — ensures GitHub Pages serves all files verbatim.
- `index.html` — gained `<link rel="alternate" type="text/markdown">` and
  `<link rel="llms-txt">` tags in `<head>` plus footer links to `Docs` and
  `llms.txt`.

To regenerate `llms-full.txt` after editing any `docs/*.md`, re-concatenate
the docs in the order listed in `docs/index.md`.

### Directory submissions (done, pending approval)

`https://kisssorcar.github.io/llms.txt` was submitted to both canonical
llms.txt directories listed on <https://llmstxt.org/#directories>:

- **llmstxt.site** — submitted via <https://llmstxt.site/submit>
  (confirmed via the thank-you page); the listing appears after their
  moderation / site refresh.
- **directory.llmstxt.cloud** — submitted via their Tally form
  (<https://tally.so/r/wAydjB>), Category "AI"; pending curation-team
  approval, notification goes to ksen@berkeley.edu.

## Earlier update — § 02 "All you need is a short prompt"

A new section `§ 02 — All you need is a short prompt` was inserted between
the existing `§ 01 — Compare` and `§ 02 — What is in the Name` sections.
All subsequent section numbers (§ 02 → § 03, …, § 09 → § 10) were shifted
by one, and a `Prompts` link was added to the top nav. No other content
was changed.

The new section shows a deck of panels, one panel per entry from:

- `./src/kiss/SAMPLE_TASKS.md` — **12 sample tasks** (rendered with a
  `Task` badge), and
- `./src/kiss/INJECTIONS.md` — **15 promptlet tricks** (rendered with
  a `Trick` badge).

Each panel has:

- A meaningful title.
- A 1–2 sentence description.
- The actual prompt rendered inside a fixed-height (180 px) scrollable
  area with a fade-out at the bottom.
- A `more …` button that expands the panel to show the entire prompt
  (button toggles to `less …`).

Panels are laid out one after another (single column) so the page reads
top-to-bottom on every screen size.

## Files

- `kisssorcar.github.io/index.html` — the updated page (drop-in
  replacement for the file in the website repo).
- `screenshots/` — screenshots captured while testing the page in a
  local browser served from `python3 -m http.server`:
  - `01-prompts-top.png` — top of the page after `#prompts` jump.
  - `02-prompts-mid.png` — middle of the prompt deck.
  - `03-prompts-expanded.png` — one panel expanded via `more …`
    (button shows `less ▾` with `aria-expanded="true"`).
  - `04-tricks-section.png` — further down the deck showing the
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
