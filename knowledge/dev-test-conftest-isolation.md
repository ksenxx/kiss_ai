---
title: 'Test isolation in conftest.py: KISS_HOME temp dir, env gates, autouse fixtures,
  orphan-sweep join'
uuid: c8fe3082-c8a1-41b8-b3ff-7c025bcf978f
summary: What the root and src/kiss/tests conftest.py set up for every test - temp
  KISS_HOME and sorcar.db, KISS_WORKDIR, KISS_MUSE_AUTH=0, worktree-pool/classifier
  gates, RLIMIT_NOFILE, config/tab isolation.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Test isolation set up by the two conftest.py files

## Root `conftest.py` (runs first, at import)
- `pytest_plugins = ["kiss.tests.subprocess_reaper"]`: plugins can only be registered from the
  root conftest (see `dev-subprocess-reaper`).
- `KISS_DISABLE_WORKTREE_POOL=1`: background spare-worktree refills
  (`kiss.agents.sorcar.worktree_pool`) would write worktrees into temp repos that tests inspect.
- `KISS_DISABLE_TASK_CLASSIFIER=1`: the pre-run classifier would make an extra real LLM call
  before every Sorcar run.
- Both are set unconditionally, so an inherited `0` or empty value cannot defeat them. The
  pool's and classifier's own tests turn them back on.
- `_raise_nofile_soft_limit` raises the `RLIMIT_NOFILE` soft limit to at least 4096. macOS
  defaults to 256, which surfaced as `OSError: [Errno 24] Too many open files` inside asyncio
  `accept()`.
- Inert `--cov`/`--cov-branch` options when pytest-cov is disabled.

## `src/kiss/tests/conftest.py` (session-wide environment)
- `KISS_HOME` is set to a fresh `tempfile.mkdtemp(prefix="kiss_test_")`, and the persistence
  module is re-pointed at it (`_th._KISS_DIR`, `_th._DB_PATH = .../sorcar.db`,
  `_th._db_conn = None`). Tests never touch the developer's `~/.kiss/sorcar.db`. The temp dir is
  shared by every test in the process, which is why the per-test fixtures below exist.
- `BROWSER=true`, so nothing opens a real browser.
- `KISS_MUSE_AUTH=0` pins the legacy connector transport. Muse suites opt back in with
  `monkeypatch.setenv("KISS_MUSE_AUTH", "1")`.
- `COVERAGE_PROCESS_START` is set when `.coveragerc.subprocess` exists at the repo root.

## Autouse fixtures (per test)
- `_isolated_default_workdir`: sets `KISS_WORKDIR` to a temp dir. `VSCodeServer` falls back to
  `os.getcwd()`, which is the developer repo, and a task run auto-commits a dirty tree at the
  end. Without this fixture a test could commit into the repo.
- `_isolated_shared_config`: undoes per-test damage to the shared `config.json`. A leaked
  non-empty `remote_password` turned every later websocket handshake into `auth_required`,
  and Playwright tests hung behind `#auth-modal`. `vscode_config.CONFIG_DIR`/`CONFIG_PATH` are
  lazy module attributes that follow `$KISS_HOME`, so pinning them in a test leaks.
- `_isolated_tab_registry`: each test starts with an empty `KISS_HOME/tabs.json`.

## `pytest_runtest_call` hook wrapper
- For `unittest.TestCase` items it wraps `tearDown` (and `asyncTearDown` for
  `IsolatedAsyncioTestCase`, which runs first) so that every live `orphan-task-sweep` thread
  is joined, with a 30 s timeout, before teardown closes `persistence._db_conn`. This fixes a
  race where a sweep touched a closed connection.
- Afterwards it unbinds the runner thread's stop event (`kiss.core.stop_signal`). A test that
  left a set event behind made the next test's first `print()` raise `KeyboardInterrupt`.

## Gotchas
- A test that needs its own home must redirect `persistence._KISS_DIR` itself or use
  `monkeypatch`. Do not delete the session `KISS_HOME`.
- Playwright and service-worker probes only get this environment if they live under
  `src/kiss/tests/`.

## Sources
- `conftest.py` (`_raise_nofile_soft_limit`, `pytest_addoption`, env gates)
- `src/kiss/tests/conftest.py` (`_isolated_default_workdir`, `_isolated_shared_config`, `_isolated_tab_registry`, `pytest_runtest_call`, `join_orphan_sweeps`)
- `src/kiss/tests/test_rlimit_nofile_bump.py`
