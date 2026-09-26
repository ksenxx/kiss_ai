---
title: 'Version number: _version.py single source, YYYY.M.N scheme, release bump'
uuid: ce3eaef6-017c-4845-a543-fcf81a8f2bd7
summary: __version__ in src/kiss/core/_version.py is the single source (hatch dynamic
  version); calendar scheme YYYY.M.N bumped by scripts/release.sh; uv run check syncs
  VS Code package.json version
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Version number

## Single source
`src/kiss/core/_version.py` holds one literal, `__version__ = "YYYY.M.N"`. Everything else reads it:
- `pyproject.toml`: `dynamic = ["version"]` with `[tool.hatch.version] path = "src/kiss/core/_version.py"`.
- `kiss/__init__.py` re-exports `__version__`.
- `kiss.server.task_runner` imports it; `kiss.server.web_server` parses `__version__` out of installed
  extension copies with a regex to pick the newest.
- `src/kiss/scripts/check.py` (run by `uv run check`) execs `_version.py` and rewrites the `version` field
  of `src/kiss/agents/vscode/package.json` when it differs ("Synced extension version to ...").

## Scheme
Calendar versioning `YEAR.MONTH.MINOR` with no zero padding on the month (e.g. `2026.9.24`).
`bump_version` in `scripts/release.sh`: same year and month as today -> `MINOR + 1`; otherwise
`<year>.<month>.0`.

## Release flow (scripts/release.sh)
Only when origin is ahead of the public `kiss_ai` repo does it bump the version in `_version.py`,
`README.md` (version badge), `SYSTEM.md`, `package.json` and `package-lock.json`, commit
"Version bumped", push, publish the history (with `scripts/exclude.json` paths purged) to
`https://github.com/ksenxx/kiss_ai`, tag, create the GitHub release with the `.vsix`, publish to PyPI
(`kiss-agent-framework`) and the VS Code marketplace, then run `./install.sh`.

Do not edit the version by hand in several places; change `_version.py` (or let `release.sh` do it)
and run `uv run check` to sync `package.json`.

## Sources
- `src/kiss/core/_version.py` (`__version__`)
- `pyproject.toml` (`[tool.hatch.version]`)
- `scripts/release.sh` (`bump_version`, `update_version_file`, `get_version`)
- `src/kiss/scripts/check.py` (extension version sync)
