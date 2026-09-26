---
title: 'pyproject.toml: package name, version source, wheel/sdist contents, entry
  points'
uuid: 80ea8a72-14f8-4b84-be11-2ae72d603ac2
summary: How kiss-agent-framework is packaged with hatchling - version from _version.py,
  sdist only-include, node_modules exclude, console scripts (sorcar, kiss-web, check,
  kiss-* channel agents), dev deps.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# pyproject.toml: packaging and entry points

## Identity
- PyPI name: `kiss-agent-framework`; `requires-python = ">=3.13"`; license Apache-2.0.
- Build backend: hatchling. `dynamic = ["version"]` with `[tool.hatch.version] path =
  "src/kiss/core/_version.py"`, which is the single source of the version literal.
  `release.sh` bumps it there and copies it into README, `src/kiss/SYSTEM.md` and the extension's
  `package.json` / `package-lock.json`. `check.py` also syncs `package.json`.
- Versions are calendar-based, `YEAR.MONTH.N` (`bump_version` in `scripts/release.sh`: same year
  and month increments N, otherwise `YEAR.MONTH.0`).

## What goes into the distributions
- Wheel: `packages = ["src/kiss", "projects/swedefend"]`, so `swedefend` is an importable
  top-level package that ships with KISS.
- sdist: `only-include = ["src/kiss", "projects/swedefend"]` (hatchling also adds
  `pyproject.toml`, README and LICENSE). An exclude list used to be used here, but it fell
  behind as benchmark results, media, papers and reports grew the sdist to 350 MB, and PyPI
  rejects any file over 100 MiB (`dev-release-pypi-size-guard`).
- `[tool.hatch.build] exclude = ["src/kiss/agents/vscode/node_modules"]`: in an agent worktree
  `node_modules` is a symlink, which the directory-only `node_modules/` gitignore rule does not
  match. Without this rule a build from a worktree shipped 147 MB of npm packages.

## Console scripts (`[project.scripts]`)
- `check` -> `kiss.scripts.check:main` (see `dev-check-full-stages`)
- `generate-api-docs` -> `kiss.scripts.generate_api_docs:main`
- `sorcar` -> `kiss.agents.sorcar.sorcar_agent:main`
- `kiss-web` -> `kiss.server.web_server:main` (the daemon plus web app)
- One `kiss-<channel>` script per messaging agent, e.g. `kiss-slack`, `kiss-telegram`,
  `kiss-gmail`, `kiss-discord`, `kiss-email`, `kiss-sms`, `kiss-phone`, `kiss-ntfy`, `kiss-ha`
  (Home Assistant), `kiss-a2a`, all pointing at `kiss.agents.third_party_agents.<name>_sea:main`.

The repo-root `./sorcar` shell script is a different thing: it runs
`uv run python -m kiss.agents.sorcar.worktree_sorcar_agent "$@"` from a checkout.

## Dev dependency group
`[dependency-groups] dev` holds mypy, ruff, pytest, pytest-cov, pytest-timeout, mdformat,
requests, types-requests, pyright and others. `uv sync` (the first stage of `uv run check`)
installs them.

## Other config in the same file
- `[tool.pytest.ini_options]`: see `dev-pytest-config-and-markers`.
- `[tool.coverage.run]`: `branch = true`, `parallel = true`, `source = ["src/kiss"]`, omits
  tests, `concurrency = ["greenlet", "thread"]`. Coverage is opt-in: pass `--cov`.
- `[tool.uv.workspace] members = ["kiss_ai"]`.

## Sources
- `pyproject.toml` (`[project]`, `[tool.hatch.*]`, `[project.scripts]`, `[dependency-groups]`, `[tool.coverage.*]`)
- `src/kiss/core/_version.py`
- `scripts/release.sh` (`bump_version`, `update_*_version`)
- `sorcar` (repo-root launcher)
- `scripts/test_release_pypi_size.sh`
