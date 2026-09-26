---
title: 'Subprocess reaper pytest plugin: killing processes tests leave running'
uuid: 99d81083-6030-4516-aa47-0ed562d8db05
summary: kiss.tests.subprocess_reaper wraps subprocess.Popen.__init__, attributes
  children to test/fixture/process buckets, SIGTERM then SIGKILL of process groups,
  LeakedSubprocessWarning.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Subprocess reaper pytest plugin

`src/kiss/tests/subprocess_reaper.py` is registered by the root `conftest.py`
(`pytest_plugins = ["kiss.tests.subprocess_reaper"]`).

## Why it exists
One test run left 529 detached `muse_auth.daemon` processes behind. Tests started daemons
through the product's `subprocess.Popen(..., start_new_session=True)` and never stopped them,
so each daemon outlived its test, the pytest process and eventually the worktree. The plugin
makes this impossible to repeat without anyone writing a finalizer.

## How it works
- `_tracking_popen_init` wraps `subprocess.Popen.__init__`. `subprocess.run`, `check_output` and
  asyncio subprocess transports all construct a `Popen`, so they are covered too.
- Each process is attributed to the innermost open bucket:
  - the test being set up, run or torn down, swept in `pytest_runtest_teardown` after the test's
    own fixtures and `tearDown`. Only processes that nothing references any more are killed
    there (`_sweep_test`, `_owned`). A process that a shared fixture's object still holds is
    left for that fixture;
  - a class, module, package or session-scoped fixture while it is set up (`pytest_fixture_setup`),
    swept by a finalizer right after that fixture's teardown (`_sweep_fixture`);
  - the process, for anything else (conftest import, collection), swept in
    `pytest_sessionfinish`. An `atexit` sweep (`_sweep_at_exit`) is the last resort for
    `pytest.exit`/Ctrl-C.
- A sweep sends SIGTERM, waits a grace period, then SIGKILLs. A child that leads its own process
  group (`start_new_session=True` or `process_group=0`, detected by `_leads_own_group`) is
  signalled as a group. It counts as alive while any member of the group exists, so the
  descendants of a double-forked daemon die too. On Windows,
  `kiss.core.processes.kill_process_group` terminates the Job Object or process tree.
- A sweep that had to kill something emits a `LeakedSubprocessWarning` naming the test or
  fixture.
- `terminate_leftovers` is the public helper behind the sweeps.

## Implications for test authors
- A leaked process is not silent: look for `LeakedSubprocessWarning` in the output and stop the
  process in the test's own teardown.
- A server that must survive across tests has to be owned by a class, module or session fixture
  that keeps a reference to it. Otherwise the per-test sweep kills it.
- Tests that deliberately `killpg` whole groups are marked `process_killer`, because they can
  hit sibling pytest processes when splits run in parallel (see `dev-parallel-test-splits`).

## Sources
- `src/kiss/tests/subprocess_reaper.py` (`_tracking_popen_init`, `_sweep_test`, `_sweep_fixture`, `terminate_leftovers`, `LeakedSubprocessWarning`)
- `src/kiss/tests/test_subprocess_reaper.py`
- `conftest.py` (`pytest_plugins`)
