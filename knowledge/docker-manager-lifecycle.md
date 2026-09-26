---
title: DockerManager container lifecycle (open, attach, close, volumes, labels)
uuid: 6ca4e013-2cd7-4ded-b168-def84ce09d38
summary: 'DockerManager open/close: image pull, container run with work dir bind-mounted
  at host path, /testbed shared volume, kiss.owner label, container:<id> attach mode,
  lifecycle lock.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# DockerManager container lifecycle

## Construction

`DockerManager(image_name, tag="latest", workdir="/", mount_shared_volume=True, ports=None,
volumes=None)` creates a `docker.from_env()` client. `image_name` parsing:
- `container:<name-or-id>` (`ATTACH_PREFIX`) sets `attached_container`; attach mode.
- `name:tag` is split with `rsplit(":", 1)`; otherwise `tag` is used.

It is a context manager (`__enter__` opens, `__exit__` closes).

## open()

Serialized by `_lifecycle_lock`. Raises `KISSError` if a container is already open on this
manager (a second one would be orphaned) or if a previous shared-volume dir could not be removed.

Start mode:
- `client.images.get`; on `ImageNotFound`, `client.images.pull(image, tag=tag)`.
- `containers.run(detach=True, tty=True, stdin_open=True, command="/bin/bash",
  working_dir=workdir, labels={OWNER_LABEL: owner_label_value()})`.
- `OWNER_LABEL` is `kiss.owner`, value `host:pid`, so cleanup can filter this process's
  containers on a shared daemon.
- Mounts: each `volumes` entry read-write; if `mount_shared_volume`, a new `tempfile.mkdtemp()`
  host dir bound at `/testbed` (`client_shared_path`). Set `mount_shared_volume=False` for
  images whose workdir already has content (for example SWE-bench).
- If `containers.run` fails, the temp dir is removed before re-raising.

Attach mode: `containers.get(id)`; `workdir` becomes the container's `Config.WorkingDir` (or
`/`); `self.volumes` is filled from the container's own bind mounts; no shared volume.

## close()

Also under the lock. Attach mode only drops the handle (the caller owns the container).
Otherwise `container.stop()` then `container.remove()` (failures are logged, not raised), then
`_remove_shared_volume_dir()`. A dir that fails to delete is remembered and retried by the next
`open()`/`close()`.

## How a Sorcar task uses it

`RelentlessAgent.run`:
```python
with DockerManager(self.docker_image, workdir=self.work_dir,
                   volumes={self.work_dir: self.work_dir}) as docker_mgr:
    self.docker_manager = docker_mgr
    ...
    return self.perform_task(tools or [], attachments=attachments)
```
The task's work dir is bind-mounted at the same path and is the container's working
directory, so relative paths and the "Work dir" prompt line mean the same thing inside and out
(commit `8493b350b`). The prompt's work-dir line is only emitted when the work dir is among
`docker_manager.volumes` (`work_dir_visible`). With a printer, `stream_callback` forwards
output as `bash_stream` events.

`run_parallel` children receive `docker_image = "container:" + container.id`, so they attach to
the parent's container instead of starting their own, and do not remove it.

## Sources
- `src/kiss/agents/sorcar/docker_manager.py` (`DockerManager.__init__`, `open`, `_open_locked`, `close`, `_close_locked`, `_remove_shared_volume_dir`, `owner_label_value`, `get_host_port`)
- `src/kiss/agents/sorcar/relentless_agent.py` (`RelentlessAgent.run`, `perform_task`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`run_parallel` child `docker_image`)
