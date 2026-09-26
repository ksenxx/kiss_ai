---
title: 'Test conventions and layout: end-to-end only, no mocks, where a test file
  belongs'
uuid: 141f2d42-e9d2-4d7d-a973-3b337af2953c
summary: KISS test rules (end-to-end, no mocks/patches/fakes, 100% branch coverage
  of changed code) and the src/kiss/tests directory layout mirroring layers, packaging-invariant
  placement, bash-suite wrappers.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Test conventions and layout

## Rules
The project's system prompt (`src/kiss/SYSTEM.md`, "Testing") sets these rules as
required conventions for new tests (some existing tests have exceptions, noted below):
- Write end-to-end tests only. No unit tests, mocks, patches, fakes or test doubles. Each test
  is independent and checks real behavior.
- Cover 100% of the branches in new or modified code wherever the branch is reachable without
  test doubles. When a branch is unreachable (network failure, disk full), explain why in the
  test file instead of mocking.
- No structural tests that assert on source text.
- Reproduce a bug with a failing end-to-end test first, then fix it.

In practice, "end-to-end" means real subprocesses, real sockets, real SQLite databases and
real git repos in temp dirs. Shell code is tested by extracting functions from the real
script and running them under bash (for example `src/kiss/tests/test_install_brand_overlay.py`,
which puts a stub `npm` on PATH, an exception to the no-test-doubles rule). Real-model tests are gated by API-key helpers or the
`live_api` marker (see `dev-pytest-config-and-markers`).

## Layout (`src/kiss/tests/`, the only `testpaths`)
| Directory | Covers |
|---|---|
| `core/` (incl. `core/models/`) | `src/kiss/core` |
| `agents/sorcar/`, `agents/vscode/`, `agents/third_party_agents/`, `agents/seas/`, `agents/obsolete/` | agent layers, the extension (Python-side and Playwright tests), channel agents |
| `server/` | `src/kiss/server` (daemon, web app) |
| `scripts/` | `src/kiss/scripts/*` and the bash scripts in `scripts/` |
| `benchmarkings/`, `swedefend/`, `viz_trajectory/` | those packages |
| top level | install.sh helpers (`test_install_brand_overlay.py`, `test_install_model_info_copy.py`), the reaper, rlimit, nproc helpers |

`agents/` and `server/` hold about 530 and 510 test files, `core/` about 210, `scripts/` about 50.

## Where a new test goes: packaging invariants
Layering tests enforce these invariants:
- `src/kiss/tests/core/test_layering_invariants.py`: code in `src/kiss/core/` imports no
  first-party module outside `src/kiss/core/`. This counts lazy, conditional and relative imports,
  and even literal `importlib.import_module("kiss....")` strings.
- `src/kiss/tests/agents/sorcar/test_layering_invariants.py`: `src/kiss/agents/sorcar/`
  imports only itself and `kiss.core`.

Tests are placed by what they depend on. A test that imports only `kiss.core` lives under
`tests/core/`, and one that also needs sorcar lives under `tests/agents/sorcar/`. When a test
moves out of `tests/core/models/`, whose `conftest.py` applies the CLI-locator stub
automatically, the module opts back in with
`from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401`.

## Bash suites
`scripts/test_release_exclude.sh`, `scripts/test_release_pypi_size.sh` and the purge suite are
standalone bash tests (`bash scripts/test_*.sh`). Pytest wrappers in `src/kiss/tests/scripts/`
run them, e.g. `test_release_pypi_size.py` and `test_release_exclude_and_vsix.py`. The wrappers
are marked `posix_only(...)` and call `subprocess.run(["bash", script], timeout=600)`.

## Shared helpers worth reusing
- `install_cli_script`, `install_fake_cloudflared` in `src/kiss/tests/conftest.py` put
  executable scripts on PATH for tests that drive real CLIs.
- `uds_tmp_path` fixture: a short temp path for Unix-domain sockets, which have a path length
  limit.
- `nproc_limit_lowered_to_one`, `thread_start_can_be_starved`: helpers for resource-limit
  tests.

## Sources
- `src/kiss/SYSTEM.md` (Testing section)
- `src/kiss/tests/core/test_layering_invariants.py`, `src/kiss/tests/agents/sorcar/test_layering_invariants.py`
- `src/kiss/tests/cli_locator_stub.py` (`stub_cli_locators`), `src/kiss/tests/core/models/conftest.py`
- `src/kiss/tests/conftest.py` (`install_cli_script`, `uds_tmp_path`)
- `src/kiss/tests/scripts/test_release_pypi_size.py`, `src/kiss/tests/test_install_brand_overlay.py`
