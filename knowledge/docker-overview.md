---
title: Docker mode area overview (DockerManager, DockerTools, sorcar-docker)
uuid: 1daafdd4-63cf-4e80-8fa8-1cf6a2376155
summary: 'Map of Docker files: docker_manager.py (container lifecycle, exec), docker_tools.py
  (file tools), docker_image parameter, sorcar-docker launcher and Dockerfile (code-server).'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Docker mode area overview

"Docker" means two unrelated things in this repo:

1. **Docker mode for a task** (`docker_image` parameter): the agent's shell and file tools run
   inside a container instead of on the host. Implemented by `DockerManager` and `DockerTools`.
2. **Running Sorcar itself in a container** (`./sorcar-docker` + `Dockerfile`): a code-server
   image with the VS Code extension, opened in a browser.

## File map

| File | Role |
|---|---|
| `src/kiss/agents/sorcar/docker_manager.py` | `DockerManager`: pull/start or attach, `Bash`, `run_commands_parallel`, timeouts and kill, `close`; `ATTACH_PREFIX`, `OWNER_LABEL` |
| `src/kiss/agents/sorcar/docker_tools.py` | `DockerTools`: `Read`, `Write`, `Edit` implemented as bash commands run in the container |
| `src/kiss/agents/sorcar/relentless_agent.py` | `RelentlessAgent.run` opens a `DockerManager` around `perform_task` when `docker_image` is set |
| `src/kiss/agents/sorcar/sorcar_agent.py` | `_get_tools` swaps in the Docker toolset; `run_parallel` children attach to the same container |
| `src/kiss/server/agent_file.py` | `docker_image()` SEA getter (wire field `dockerImage`) |
| `sorcar-docker`, `Dockerfile`, `scripts/docker-startup.sh` | Launch Sorcar in code-server |

## Setting docker_image

- `SorcarAgent.run(docker_image=...)`, `kiss.server.sorcar.run(docker_image=...)`, or an SEA's
  `docker_image()` getter.
- `"python:3.12"`: start a fresh container, removed at task end.
- `"container:<name-or-id>"`: attach to a running container and leave it running.
- Empty: host tools.

## Detail pages

- `docker-manager-lifecycle`: open/attach/close, work-dir bind mount, shared volume, labels.
- `docker-exec-timeouts-and-streaming`: token-tagged execs, kill on timeout, output handling.
- `docker-file-tools`: how Read/Write/Edit work through bash.
- `docker-mode-disabled-features`: what is unavailable or different in a Docker run.
- `docker-sorcar-docker-launcher`: the code-server image and launcher script.

## Sources
- `src/kiss/agents/sorcar/docker_manager.py` (`DockerManager`, `ATTACH_PREFIX`, `OWNER_LABEL`)
- `src/kiss/agents/sorcar/docker_tools.py` (`DockerTools`)
- `src/kiss/agents/sorcar/relentless_agent.py` (`RelentlessAgent.run`)
- `API.md` (`docker_image` parameter of `kiss.server.sorcar.run`)
