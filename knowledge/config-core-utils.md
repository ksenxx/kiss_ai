---
title: kiss.core.utils helpers (finish, ensure_html, substitute_prompt_args, is_root_dir,
  config_to_dict)
uuid: dd60cd64-584f-4ba4-946a-6e939c1acdc1
summary: 'core/utils.py helpers: finish() YAML with HTML summary, ensure_html Markdown
  to HTML, single-pass substitute_prompt_args, is_root_dir guard, config_to_dict drops
  API keys, dump_yaml'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# kiss.core.utils helpers

Besides the atomic-write primitives (see `config-atomic-writes-and-locks`), `src/kiss/core/utils.py`
holds small helpers used across agents.

## finish(success, is_continue=False, summary_in_html="", suggested_next_task="")
The default `finish` tool. Returns a YAML string with `success`, `is_continue` and `summary`, plus
`suggested_next_task` when non-empty. Booleans go through `_coerce_bool` (strings `true`/`1`/`yes`
are True), because models sometimes pass strings. The summary always goes through `ensure_html`.

## ensure_html(text)
Guarantees the HTML summary:
- non-string input is `str()`-ed; empty stays empty;
- text starting with `<!doctype` or containing a known HTML tag (`_HTML_TAG_RE`) passes through;
- fully entity-escaped HTML (`&lt;h3&gt;`, a known LLM mistake) is unescaped
  (`_unescape_escaped_html`);
- anything else is rendered as CommonMark with tables via `markdown_it`. If `markdown_it` is
  momentarily unimportable (e.g. during `uv sync`), it falls back to HTML-escaped text in `<p>` with
  `<br/>`: `finish` runs on error paths and must never raise.

## substitute_prompt_args(template, arguments)
Replaces `{key}` placeholders in **one regex pass**. Unlike `str.format`, literal braces (JSON, code,
`${VAR}`) are untouched. A single pass also prevents re-expansion: sequential `str.replace` per key
would expand a placeholder that appears inside another argument's value (e.g. a task quoting
`{result}`), depending on dict order.

## is_root_dir(path)
True for `/`, `//`, and Windows drive roots (`C:`, `C:\`, `C:/`) after normalization (so `/./`,
`/..` count). Roots reach the daemon when a GUI-launched VS Code with no folder open inherits `/` as
cwd; callers treat a root as "no work dir". UNC roots are deliberately not detected (a `//` prefix is
a legal POSIX path). Mirrors the extension's `SorcarSidebarView._getWorkDir` guard.

## config_to_dict()
Serializes `DEFAULT_CONFIG` for trajectories, dropping any field whose name contains `API_KEY` or
`WORKSPACE_ID` (`_is_secret_config_field`).

## dump_yaml(data, stream=None, **kwargs)
`yaml.dump` with `_KissDumper`, a `yaml.Dumper` subclass whose string presenter writes multi-line
strings as `|` literal blocks (single-line values keep the default style). Used for trajectories
(`kiss.core.base`) and `finish` results. The representer is registered on the subclass, not on
`yaml.Dumper`, because a global `yaml.add_representer` would change every `yaml.dump` in the
embedding process.

## Sources
- `src/kiss/core/utils.py` (`finish`, `ensure_html`, `_unescape_escaped_html`, `substitute_prompt_args`, `is_root_dir`, `config_to_dict`, `_is_secret_config_field`, `_coerce_bool`, `dump_yaml`)
