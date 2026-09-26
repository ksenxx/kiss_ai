---
title: Atomic file writes and cross-platform file locks (atomic_write_text, exclusive_file_lock)
uuid: be4298d6-8c90-4ef3-8991-3b44664a9ca1
summary: atomic_write_text stage + os.replace with mode/create_mode rules, Windows
  sharing-violation retries, read_bytes_waiting_for_writer, exclusive_file_lock via
  fcntl/msvcrt
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Atomic file writes and cross-platform file locks

Shared state under `~/.kiss` (config.json, api_keys.env, memory pages, MY_MODELS.json, trajectories,
tabs.json) is read and written by several processes at once: the daemon, sub-agents, the VS Code
extension, the trajectory viewer. Two primitives make that safe.

## atomic_write_text(target, content, mode=None, create_mode=0o600)
Stages the text in a sibling temp file, then publishes it with `os.replace`, so readers see the old
or the new file, never an empty or half-written one (a plain `open(path, "w")` truncates first).
- Parent directories are created.
- Content goes through a buffered file object, not a bare `os.write`, whose legal short write would
  otherwise publish a truncated file.
- Permission rules, in priority order:
  1. explicit `mode` always wins, for new and existing targets (secret files force `0o600`);
  2. an existing target keeps its current bits (copied to the staged file, since `os.replace`
     publishes the staged inode's mode);
  3. a new target gets `create_mode` filtered by the umask. Default `0o600` because many callers
     hold secrets; memory pages pass `create_mode=0o666` for normal `write_text` semantics.
- On any exception the temp file is removed.

Callers: `save_config` (`mode=0o600`), `_atomic_write_text_secure` for the key store and shell RCs,
`MemoryDir.write` (`create_mode=0o666`).

## Windows sharing violations
Windows refuses `os.replace` while a reader has the target open (Python never sets
`FILE_SHARE_DELETE`), and refuses opens during a replace. `replace_waiting_for_readers` and
`read_bytes_waiting_for_writer` retry on `PermissionError` every 5 ms for up to
`SHARING_RETRY_SECONDS = 10.0`, restoring POSIX semantics; a directory target raises immediately and a
holder that never lets go still raises. Always read shared files with
`read_bytes_waiting_for_writer` (config.json, memory pages and the index sync all do).

## file_lock.py
`fcntl` does not exist on Windows, and one top-level `import fcntl` once made the whole package
unimportable there (`kiss.core.base` imports `model_info`). All locking goes through
`kiss.core.file_lock`:
- `lock_exclusive(fd_or_file, blocking=True) -> bool`: `fcntl.flock(LOCK_EX[|LOCK_NB])` on POSIX;
  on Windows `msvcrt.locking(LK_NBLCK)` on byte 0, polled every 50 ms (`LK_LOCK` sleeps a full second
  between attempts, which made two cron jobs finishing together take over 2 s). Returns `False` only
  for a non-blocking attempt that lost.
- `unlock(fd_or_file)`.
- `exclusive_file_lock(path)`: context manager that creates the file (`a+b`), locks, yields, unlocks.
  **The lock file is never deleted**: unlinking would let a later opener lock a different inode.

Semantics: advisory, per open file description, released on close or process death, **not
reentrant** across separate opens. Opening and locking the same lock file twice in one process
deadlocks, which is why `vscode_config` pairs every flock with the in-process `_config_lock`.

Lock the sidecar, not a file you replace atomically: after `os.replace` a lock on the old inode no
longer excludes anyone (e.g. `.api_keys.env.kiss.lock`, `.config.lock`).

Other users: `kiss.core.models.model_info`, `git_worktree`, `web_server` (daemon single-instance
locks via `blocking=False`), `voice_wake`, muse_auth client.

## Sources
- `src/kiss/core/utils.py` (`atomic_write_text`, `replace_waiting_for_readers`, `read_bytes_waiting_for_writer`, `SHARING_RETRY_SECONDS`, `_open_staging_file`)
- `src/kiss/core/file_lock.py` (`lock_exclusive`, `unlock`, `exclusive_file_lock`)
- `src/kiss/core/vscode_config.py` (`save_config`, `_api_keys_store_flock`)
