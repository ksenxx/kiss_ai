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
  `## Papers`, `## Blog`, `## Source & Install`, `## Optional` link
  sections), live at <https://kisssorcar.github.io/llms.txt>.
- `.well-known/llms.txt` — mirror of the same file at the `.well-known` path
  (update both when adding a docs page).
- `llms-full.txt` — all Markdown docs concatenated for single-fetch ingestion,
  live at <https://kisssorcar.github.io/llms-full.txt>.
- `docs/*.md` — 11 pure-Markdown pages: `index.md`, `overview.md`,
  `installation.md`, `cli.md`, `api.md`, `models.md`, `messaging-agents.md`,
  `sea-commands.md`, `sample-tasks.md`, `prompt-tricks.md`, `tips.md`
  (content sourced from `README.md`, `API.md`, `src/kiss/SAMPLE_TASKS.md`,
  `src/kiss/INJECTIONS.md`, and `src/kiss/TIPS.md`; `sea-commands.md`
  documents `src/kiss/agents/sorcar/sea_commands.py`).
- `index.html.md` — plain-Markdown twin of the homepage.
- `robots.txt` — allows all crawlers, references llms.txt and the sitemap.
- `sitemap.xml` — lists the HTML homepage, `index.html.md`, `llms.txt`,
  `llms-full.txt`, the 11 `docs/*.md` pages, and the five `blog/*.html`
  posts (not `robots.txt` or `.well-known/llms.txt`).
- `.nojekyll` — ensures GitHub Pages serves all files verbatim.
- `index.html` — gained `<link rel="alternate" type="text/markdown">` and
  `<link rel="llms-txt">` tags in `<head>` plus footer links to `Docs` and
  `llms.txt`.

To regenerate `llms-full.txt` after editing any `docs/*.md`: keep the two
header lines (H1 + blockquote), then append, for `docs/index.md` followed by
each page in the order listed in `docs/index.md`,

```
---

<!-- Source: https://kisssorcar.github.io/docs/<page>.md -->

<page text>
```

with relative links such as `](overview.md)` rewritten to
`](https://kisssorcar.github.io/docs/overview.md)`. When adding a page, also
list it in `docs/index.md`, `llms.txt`, `.well-known/llms.txt`, and
`sitemap.xml`. `src/kiss/tests/agents/sorcar/test_sea_commands_docs.py`
checks that `docs/sea-commands.md` is wired into `docs/index.md`, `llms.txt`,
`llms-full.txt`, and `sitemap.xml` (it does not check `.well-known/llms.txt`)
and runs that page's examples against the real `sea_commands` module.

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

The section shows a 20-panel carousel, one panel per entry from:

- `./src/kiss/SAMPLE_TASKS.md` — **12 sample tasks** (rendered with a
  `Task` badge), and
- `./src/kiss/INJECTIONS.md` — **8 promptlet tricks** (rendered with a
  `Trick` badge).

Each panel has:

- A meaningful title.
- A 1–2 sentence description.
- The actual prompt rendered inside a fixed-height (eight lines)
  scrollable area with a copy button.

The carousel shuffles the panels on page load and shows one at a time with
previous/next buttons, dot navigation, a counter, and autoplay. The
`.prompt-fade` overlay and `more …`/`less …` button from the first version
remain in the markup but are hidden (`display: none`).

## Files

- `kisssorcar.github.io/index.html` — the updated page (drop-in
  replacement for the file in the website repo).
- Under `kisssorcar.github.io/`: `docs/`, `llms.txt`, `.well-known/llms.txt`,
  `llms-full.txt`, `index.html.md`, `robots.txt`, `sitemap.xml`, and
  `.nojekyll` — the LLM-indexing files described above.
- `kisssorcar.github.io/assets/` — images and paper PDFs linked from the
  homepage and `llms.txt`; `kisssorcar.github.io/blog/` — the five blog
  posts listed in `llms.txt` and `sitemap.xml`.

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
