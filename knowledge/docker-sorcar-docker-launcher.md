---
title: sorcar-docker launcher and Dockerfile (Sorcar in code-server)
uuid: e8d5b637-b897-48f9-96b6-d824b08e4fa2
summary: ./sorcar-docker builds the kiss-sorcar code-server image from Dockerfile,
  passes GH_TOKEN and API keys, clones the repo, runs install.sh, serves code-server
  on port 8080.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# sorcar-docker launcher and Dockerfile

This is unrelated to the per-task `docker_image` sandbox (see `docker-overview`). It runs the
whole Sorcar product, VS Code extension included, inside a browser-based code-server.

## Usage

```
./sorcar-docker [PORT] [--rebuild]
```

Steps performed by the script:
1. Pre-flight checks; validates at most one PORT (1-65535).
2. Git credentials: uses `GH_TOKEN`, or `gh auth token` when the `gh` CLI is logged in; exits
   with "No GitHub token found" otherwise (the repo clone inside the container needs it).
3. Image `kiss-sorcar`: reused if `docker image inspect` finds it, rebuilt from `./Dockerfile`
   with `--rebuild`.
4. Removes the old container, collects env vars (`GH_TOKEN` plus `ANTHROPIC_API_KEY`,
   `ANTHROPIC_WORKSPACE_ID`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`,
   `OPENROUTER_API_KEY` when set).
5. `docker run -d -p <PORT>:8080 ...`, waits for code-server to answer, opens the browser.

## Image (Dockerfile)

- `FROM codercom/code-server:latest`; apt installs git, curl, build-essential, python3 and
  friends; `uv` is installed from a pinned release tarball (`UV_VERSION`) for x86_64 or aarch64.
- Passwordless sudo for user `coder` (needed by `playwright install-deps`).
- `/home/kiss` (primary clone target for kiss.git) and `/home/kiss_ai` (fallback for
  kiss_ai.git) are created and owned by `coder`.
- Entrypoint `scripts/docker-startup.sh`: configures git from `GH_TOKEN`, clones or pulls the
  repo (primary URL, then fallback), runs `install.sh` (Python env, Playwright, VS Code
  extension), then launches code-server with the cloned directory as workspace (it rewrites the
  placeholder workspace path in `CMD`).
- `CMD` binds `0.0.0.0:8080` with `--auth none`, `--disable-workspace-trust` (so the extension
  activates without the trust dialog) and `--enable-proposed-api ksenxx.kiss-sorcar`.

Security note: `--auth none` means anyone who can reach the published port gets an editor with
the API keys in its environment; publish the port only on a trusted interface.

## Sources
- `sorcar-docker`
- `Dockerfile`
- `scripts/docker-startup.sh`
