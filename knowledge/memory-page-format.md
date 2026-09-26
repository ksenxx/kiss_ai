---
title: Memory page format, names and MemoryDir writes
uuid: dd53287d-8b82-426f-8666-2e1bbdf380ab
summary: 'memoryfield page format: flat dir of Markdown pages with YAML frontmatter
  (title uuid summary created updated), PAGE_NAME_RE names, MAX_PAGE_BYTES, debris,
  MemoryDir.write merge rules'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Memory page format, names and MemoryDir writes

Sorcar's persistent memory follows the memoryfield spec (https://github.com/calpaterson/memoryfield-spec).
A memory is a **flat directory of Markdown pages**. The pages are the canonical data; the SQLite
vector index (see `memory-vector-index`) is a regenerable cache. `pages.py` uses only the stdlib
plus PyYAML.

## Page file
- Path: `<root>/<name>.md`. No sub-directories: `MemoryDir.page_stats` skips directories,
  symlinks, non-`.md` files, debris and invalid names in one `os.scandir` pass.
- Name rule `PAGE_NAME_RE = ^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$`: ASCII lowercase letters, digits and
  hyphens, starting and ending with a letter or digit. `slugify` turns free text into a valid
  name (falls back to `"page"`, truncates at a hyphen boundary, default 60 chars).
- Debris ignored even if it ends in `.md` (`is_debris`): `.DS_Store`, `desktop.ini`, `Thumbs.db`,
  names ending in `~`, and names containing `.sync-conflict-` (Syncthing).
- `MAX_PAGE_BYTES = 8192`: pages SHOULD stay under this; indexers embed only this prefix.

## Frontmatter
Optional block: a leading `---` line, YAML mapping, closing `---` line
(`split_frontmatter`, CRLF tolerated). Malformed YAML or a non-mapping means "no frontmatter"
and the whole text becomes the body; pages without frontmatter are valid.

Known keys, emitted first in this order by `render_page`: `title`, `uuid`, `summary`, `created`,
`updated` (`FRONTMATTER_KEYS`), then any extra keys. All values are **stringified** so timestamps
stay quoted strings (YAML 1.1 would otherwise coerce them to `datetime`). Empty/None values are
dropped. Timestamps come from `now_iso()`: UTC `YYYY-MM-DDTHH:MM:SSZ`.

`Page.title` falls back to the page name; `Page.summary` falls back to `""`.

## MemoryDir.write semantics
- Refuses an empty body (`ValueError`).
- On create: generates `uuid` (uuid4), `created`, and a default `title` of the stem with hyphens
  replaced by spaces.
- On update: stored `uuid` and `created` are always preserved, even if the incoming body carries
  its own frontmatter with different values. `title`/`summary` are kept unless overridden;
  `updated` is always refreshed.
- A frontmatter block at the top of *body* is merged in; `extra=` adds more keys.
- Published with `atomic_write_text(path, raw, create_mode=0o666)` (stage + `os.replace`), because
  the directory is shared by parallel sub-agents and daemon processes; `0o666` because pages are
  plain documents, unlike the helper's secret-safe default `0o600` (see `config-atomic-writes-and-locks`).

## Path safety
`MemoryDir.page_path` strips a trailing `.md`, validates the name, rejects symlinked pages, and
rejects any path whose resolved parent is not the root. `MemoryDir.__init__` resolves the root with
`expanduser().resolve()`; the directory is created lazily on first write.

Reads go through `read_page_text`: UTF-8 with `errors="replace"`-style decoding via
`read_bytes_waiting_for_writer`, so a damaged page stays usable and a concurrent Windows
`os.replace` is waited out instead of failing.

## Sources
- `src/kiss/core/memoryfield/pages.py` (`PAGE_NAME_RE`, `DEBRIS_NAMES`, `MAX_PAGE_BYTES`, `FRONTMATTER_KEYS`, `split_frontmatter`, `render_page`, `slugify`, `MemoryDir`)
- `src/kiss/core/utils.py` (`atomic_write_text`, `read_bytes_waiting_for_writer`)
