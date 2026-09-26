---
title: Remote deploy with ./rsorcar, its scripts/ helpers, and the sorcar launcher
uuid: 8f02bff7-7f69-43e3-a650-97588343bf6f
summary: './rsorcar user@host step by step with its scripts/ helpers: prereqs, idle
  and disk checks, ssh identity, git sync, API keys, sorcar.db and memory merge, kiss-web
  service, tunnel, gh auth, env overrides.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Remote deploy with `./rsorcar` and the `sorcar` launcher

## `sorcar` (repo root)
Runs `uv run python -m kiss.agents.sorcar.worktree_sorcar_agent "$@"`: the worktree Sorcar agent CLI in the repo's environment. The `sorcar` console script in `pyproject.toml` (`kiss.agents.sorcar.sorcar_agent:main`) is a different entry point.

## `rsorcar` (repo root, bash)
Usage: `./rsorcar user@ip-address`. Deploys *this* checkout to a remote Linux machine and publishes its `kiss-web` worldwide. `install.sh` installs a `~/.local/bin/rsorcar` launcher pointing at the main repo. Most steps ship a helper over ssh (`ssh host 'bash -s' < script`), so both machines run the same code even on a first deploy with no remote checkout.

| Step | What | Helper |
|---|---|---|
| 1 | ssh connectivity | |
| 1a | install git, curl, tar, python3, ssh client via the package manager with `sudo -n` (never prompts; fresh cloud images lack git) | `scripts/install-remote-prereqs.sh` |
| 1b | refuse if a task is running remotely, since the kiss-web restart would kill it; `SORCAR_FORCE_RESTART=1` overrides | `src/kiss/scripts/running_tasks.py` |
| 1c | room under `$HOME` = task DB size + `SORCAR_DISK_HEADROOM_GB` (3); otherwise names what fills the disk, which filesystem has room, and suggests a bind-mount move of home | `scripts/check-remote-disk-space.sh`, `scripts/move-home-to-disk.sh` |
| 2 | copy `~/.ssh` as a tar stream; never `authorized_keys`; replaced remote files go to `~/.kiss/ssh-replaced-<time>/` | `scripts/install-ssh-identity.sh` |
| 3 | total git sync through `origin`: push every branch, create/update the remote checkout, pull back what it gained. Uncommitted work, including non-ignored untracked files, is committed (`git add -A`); ignored files (`.venv`, `tmp/`) do not travel. Diverged branches are merged, no force-push or branch deletion; an unmergeable branch stops the deploy. `kiss/wt-*` branches are never deployed | `scripts/sync-repo.sh` |
| 4 | API keys only to a remote with none (an unreadable remote key file stops the deploy), distilled into `~/.kiss/api_keys.env`, the single store the daemon parses at startup; one delimited `~/.bashrc` block sources it | `scripts/count-api-keys.sh`, `scripts/install-api-keys.sh` |
| 4b | `~/.kiss/sorcar.db` both ways: insert missing rows only, never delete. Full copy only on first deploy or schema refusal, keeping `sorcar.db.replaced` | `scripts/sync-task-db.sh`, `src/kiss/scripts/sync_db.py`, `carry_over_tables.py`, `relocate_work_dir.py` |
| 4c | memory pages (`~/.kiss/memories` or config `memory_dir`) both ways, newest `updated` wins, never delete; vector index not copied, it is rebuilt | `scripts/sync-memory.sh`, `src/kiss/scripts/merge_memory_pages.py` |
| 5 | run `install.sh` remotely; installs code-server when no `code` CLI exists | `install.sh` |
| 6 | `uv sync`; `kiss-web` as a systemd user service with lingering (survives logout and reboot; the supervisor `Restart=always` server resets rely on) | |
| 7 | password required (kiss-web opens a Cloudflare tunnel only when `remote_password` is set, see `server-remote-access-auth`); install `cloudflared` if missing; drop a copied `~/.kiss/ntfy_topic` equal to the local one so each machine gets its own topic (see `server-cloudflare-tunnel`) | `src/kiss/scripts/remote_config.py`, `scripts/wait-for-public-url.sh` |
| 8 | copy GitHub credentials for every `gh` account (tokens on stdin, never argv; remote's existing accounts backed up in `~/.kiss/`); no credentials is a warning. Skip with `SORCAR_SKIP_GITHUB_AUTH=1` | `scripts/collect-github-auth.sh`, `scripts/install-github-auth.sh` |
| 9 | verify the public URL from the local machine (this verdict counts) and open the browser. The remote's own check is advisory, because DNS may negatively cache a fresh quick-tunnel name | |
| 10 | run `install.sh` locally from the main repo, not a worktree, because the launchers point at its directory | `install.sh` |

## Environment overrides
`SORCAR_REMOTE_DIR` (default: project name), `SORCAR_WEB_PORT` (default 8787), `SORCAR_PASSWORD` (default: this machine's `remote_password` in `~/.kiss/config.json`, else the remote's existing one, else random), `SORCAR_GITHUB_USER` (default `ksenxx`), `SORCAR_GIT_BRANCH` (default: the main repo's branch), `SORCAR_GIT_NAME`, `SORCAR_GIT_EMAIL`, `SORCAR_NO_BROWSER`, `SORCAR_SKIP_GITHUB_AUTH`, `SORCAR_DISK_HEADROOM_GB`, `SORCAR_SKIP_DISK_CHECK`, `SORCAR_FORCE_RESTART`.

## Design rules across the helpers
- Nothing is destroyed: replaced ssh files, rc files, DBs and gh accounts are backed up under `~/.kiss/` first.
- Secrets travel on stdin, never in process arguments (readable in `/proc/<pid>/cmdline`).
- Sync helpers work standalone, e.g. `scripts/sync-task-db.sh user@host` refreshes history without redeploying.
- Some helper comments (`scripts/sync-repo.sh`, `sync-task-db.sh`, `install-api-keys.sh`) still name `./sorcar-cloud`, the deploy script's former name; no such file exists.

## Related
- `./sorcar-docker [PORT] [--rebuild]`: reuses or builds the `kiss-sorcar` image from `Dockerfile`; `scripts/docker-startup.sh` clones the repo, runs `install.sh`, and serves code-server on 8080.
- Tests: `src/kiss/tests/scripts/test_sync_repo.py`, `test_sync_task_db*.py`, `test_sync_db.py`, `test_sync_memory.py`, `test_github_auth.py`, `test_deploy_no_data_loss.py`.

## Sources
- `rsorcar` (header "Flow" and "Environment overrides")
- `sorcar`; `pyproject.toml` (`[project.scripts]`); `install.sh` (`install_repo_script_launcher`)
- The `scripts/` and `src/kiss/scripts/` helpers named in the table
- `sorcar-docker`, `scripts/docker-startup.sh`
