# KISS domain knowledge base

Short Markdown pages about how this repo works: where things live, how the pieces fit,
why they are built that way, and which bugs have bitten before. Coding agents read it
through the `memory_search` / `memory_read` tools before they work on the repo.

## Use it as the agent's memory

Set `memory_dir` in `~/.kiss/config.json` to this directory (absolute path) and keep
`use_memory` on:

```json
{"use_memory": true, "memory_dir": "/path/to/kiss/knowledge"}
```

`_memory_settings()` in `src/kiss/agents/sorcar/sorcar_agent.py` reads both keys when a
task starts. Agents then search these pages and write new lessons back here, so new
knowledge shows up in `git diff` and is reviewed like code.

## Layout

- `<area>-<topic>.md`: one topic per page, YAML frontmatter (`title`, `summary`,
  `uuid`, `created`, `updated`), body under 8 KB, ending with a `## Sources` list
  of files and functions. Areas: `core`, `models`, `config`, `memory`, `sorcar`,
  `git`, `db`, `server`, `vscode`, `channels`, `cron`, `muse`, `web`, `docker`,
  `seas`, `dev`. Each area except `cron` and `muse` (three and two pages) has an
  `<area>-overview` page that maps its files and pages.
- `eval/questions.yaml`: search test questions with the pages that answer them.
- `eval/search_eval.py`: runs those questions against the index.
- `*.sqlite3`: the embedding index. It is rebuilt from the pages on demand, so it is
  gitignored, as are `.<page>.md-*` staging files a killed write can leave behind.

This file is not a page: `MemoryDir` only lists lowercase-hyphen file names.

## Maintenance

- Test search after large edits:
  `uv run python knowledge/eval/search_eval.py`. It uses the same embedder as
  `memory_search` (`text-embedding-3-small` when `OPENAI_API_KEY` is set, otherwise
  the offline `hashed-bow-v1`); `--hashed` forces the offline one. When you add pages,
  add questions for them.
- Find near-duplicate and stale pages with the `memory_refresh` tool, then merge
  duplicates with `memory_write` + `memory_delete` and fix references to the deleted name.
- Pages cite code paths. When a page and the code disagree, trust the code and fix the page.
