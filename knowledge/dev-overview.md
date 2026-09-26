---
title: 'Build, install, release, testing and dev conventions: area overview'
uuid: e76a5f58-a394-44df-a52d-8d59e6356399
summary: 'Map of dev tooling in KISS Sorcar: pyproject.toml, check.py, conftest.py,
  test layout, run-all.js, release.sh, exclude.json, install.sh, scripts/install.sh,
  rsorcar, CI; links to dev-* pages.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Build, install, release, testing: area overview

## Daily commands
| Task | Command | Page |
|---|---|---|
| lint and typecheck everything | `uv run check --full` (once, at the end of a change) | `dev-check-full-stages` |
| run impacted tests | `uv run pytest src/kiss/tests/<dir>/test_x.py -q` | `dev-pytest-config-and-markers` |
| run the whole suite | parallel splits via `run_commands_parallel` | `dev-parallel-test-splits` |
| JS extension tests | `cd src/kiss/agents/vscode && npm test` | `vscode-js-tests` |
| regenerate API.md | `uv run generate-api-docs` | `dev-api-docs-generation` |
| install from the checkout | `./install.sh [--non-interactive]` | `dev-install-sh-flow` |
| release | `scripts/release.sh` (needs `UV_PUBLISH_TOKEN`, `VSCE_PAT`, `gh`) | `dev-release-workflow` |
| deploy to a remote machine | `./rsorcar user@host` | `server-rsorcar-deploy` |

## File map
- Packaging and config: `pyproject.toml` (hatch build, console scripts, pytest, ruff, mypy,
  coverage), `pyrightconfig.json`, `src/kiss/core/_version.py`, the only place the version
  literal is edited. See `dev-pyproject-packaging`.
- Quality gate: `src/kiss/scripts/check.py`. See `dev-check-full-stages`.
- Tests:
  - `conftest.py` (root) and `src/kiss/tests/conftest.py`: `dev-test-conftest-isolation`
  - `src/kiss/tests/subprocess_reaper.py`: `dev-subprocess-reaper`
  - `src/kiss/tests/cli_locator_stub.py` and the layout rules: `dev-test-conventions-and-layout`
  - `src/kiss/agents/vscode/test/run-all.js`: `vscode-js-tests`
- Release: `scripts/release.sh`, `scripts/exclude.json`, `scripts/test_release_exclude.sh`,
  `scripts/test_release_pypi_size.sh`. See `dev-release-workflow`,
  `dev-release-exclude-and-history-filter`, `dev-release-pypi-size-guard`.
- Install:
  - `scripts/install.sh` (curl one-liner and Update button) and the shared update lock:
    `dev-install-bootstrap-and-update-lock`
  - `install.sh`: `dev-install-sh-flow`
  - the `.brand/` overlay: `dev-brand-overlay`
- Remote and Docker: `rsorcar`, `sorcar-docker`, `Dockerfile`, `scripts/docker-startup.sh`,
  `scripts/sync-*.sh`, `scripts/install-*.sh`, `scripts/collect-github-auth.sh`,
  `scripts/check-remote-disk-space.sh`, `scripts/move-home-to-disk.sh`,
  `scripts/wait-for-public-url.sh`, `scripts/count-api-keys.sh`. See
  `server-rsorcar-deploy`.
- CI: `.github/workflows/*.yml`. See `dev-ci-workflows` (it records two known defects).
- Everything else: `dev-repo-directory-map`.

## Conventions that apply to every change
- Tests are end-to-end only: no mocks, patches or fakes; 100% branch coverage of changed code
  where reachable (`src/kiss/SYSTEM.md`).
- `core/` imports only `core/`, and `agents/sorcar/` imports only itself and `core/`. Tests are
  placed by the same layering.
- Run `uv run check --full` once at the end and fix every stage it reports before the single
  re-run. There is no formatter stage, so do not reformat files.
- Temporary files go in `./tmp/`, which is gitignored and excluded from releases.
- Each source file starts with the author/contributors header comment
  (`# Author: Koushik Sen ...`, `# Contributors:`, `# add your name here`). The same header is on
  Python, bash and JS files.
- The two-branch model: develop in the private origin `ksenxx/kiss`. `release.sh` publishes a
  filtered history to the public `ksenxx/kiss_ai`, which is what users clone.

## Sources
- `pyproject.toml`, `src/kiss/scripts/check.py`, `conftest.py`, `src/kiss/tests/conftest.py`
- `scripts/release.sh`, `scripts/exclude.json`, `install.sh`, `scripts/install.sh`, `rsorcar`
- `src/kiss/SYSTEM.md` (Testing, Code Style)
