# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A second ``kiss-web`` daemon in its own process, for cross-machine tests.

The persistence module keeps the database path in module globals, so
two daemons with different databases cannot share a process.
:func:`start_remote_daemon` runs this module as a subprocess with its
own ``KISS_HOME``: the child writes the ``remote_password`` into that
home's ``config.json``, seeds its ``history.db`` with the given tasks,
then serves on ``127.0.0.1:<port>`` until it is killed.  It prints one
JSON line, ``{"tasks": {"<chat_id>/<task>": "<task_id>"}, "pid": <pid>}``,
once the rows are in place.

The child binds port 0 and the OS picks the port: the parent learns it
from the endpoint file the daemon publishes once the listener is bound
(``sorcar-local.json`` under the child's home, which also carries the
child's pid).  The pid to match is the one the child printed, not
``Popen.pid``: on Windows a virtual environment's ``python.exe`` is
CPython's venv launcher, which runs the real interpreter as a child
process, so the daemon's ``os.getpid()`` differs from the pid the
parent spawned.  Choosing a free port in the parent and probing it with a
TCP connect is a race under a parallel test run: another process (a
private daemon of a sibling test binds ``127.0.0.1:0``) can take the
port first, the probe then answers from that foreign listener and the
test talks to the wrong daemon.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from kiss.agents.sorcar import local_endpoint


class RemoteDaemon:
    """Handle of a daemon started by :func:`start_remote_daemon`."""

    def __init__(self, process: subprocess.Popen[str], home: Path, port: int,
                 password: str, task_ids: dict[str, str], pid: int) -> None:
        self.process = process
        self.home = home
        self.port = port
        self.password = password
        self.task_ids = task_ids
        """``"<chat_id>/<task>"`` → the id the child gave that seeded row."""
        self.pid = pid
        """The daemon's own ``os.getpid()`` (``process.pid`` on POSIX)."""

    @property
    def url(self) -> str:
        """The daemon's ``https://127.0.0.1:<port>`` address."""
        return f"https://127.0.0.1:{self.port}"

    def stop(self) -> None:
        """Kill the child, wait for it and close its pipe."""
        _kill(self.process)


def _kill(process: subprocess.Popen[str]) -> None:
    """Kill *process*, wait for it and close the pipe its seed line came through."""
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    assert process.stdout is not None
    process.stdout.close()


def start_remote_daemon(
    home: Path, *, password: str = "", seed: list[dict[str, Any]] | None = None,
    certfile: Path | None = None, keyfile: Path | None = None, timeout: float = 60.0,
    config: dict[str, Any] | None = None, port: int = 0, host: str = "127.0.0.1",
    env: dict[str, str] | None = None,
) -> RemoteDaemon:
    """Start a daemon under *home* on an OS-picked port and wait until it is bound.

    Args:
        home: The child's ``KISS_HOME`` (created when missing).
        env: Environment variables set for the child on top of this
            process's (a ``HOME`` of its own, so its start-up scan of the
            home directory for the ``@`` picker does not walk the
            developer's).
        password: Its remote password (stored hashed); empty means loopback-only.
        seed: Rows for its ``task_history``: dicts with ``chat_id``,
            ``task`` and optionally ``task_id`` (forced), ``timestamp``
            (forced), ``result`` and ``running`` (true: the daemon holds
            the task as running, for as long as it lives).
        certfile: Serve this certificate instead of a generated local CA.
        keyfile: The key of *certfile*.
        timeout: Seconds to wait for the child to publish its endpoint.
        config: More ``config.json`` keys of the child (a model
            endpoint for a daemon that runs tasks).
        port: Bind this port instead of an OS-picked one (to restart a
            daemon where an earlier one listened).
        host: The address to bind (``0.0.0.0`` makes the daemon
            reachable at the host's LAN address, as a non-loopback peer).

    Returns:
        The running daemon's handle.

    Raises:
        RuntimeError: When the child exits or never publishes its endpoint.
    """
    home.mkdir(parents=True, exist_ok=True)
    endpoint_file = home / local_endpoint.LOCAL_ENDPOINT_FILE
    child_env = dict(os.environ, **(env or {}), KISS_HOME=str(home))
    # The child runs in *home*: a ``PYTHONPATH`` naming the checkout
    # under test relative to the test's cwd (``src``) is resolved for
    # it, or the child would import the venv's installed copy instead
    # (or nothing, when that install points elsewhere).
    child_env["PYTHONPATH"] = os.pathsep.join(
        os.path.abspath(p) for p in child_env.get("PYTHONPATH", "").split(os.pathsep) if p
    )
    # The seed goes through a file: a row with a multi-megabyte task
    # text does not fit on the command line.
    seed_file = home / "seed.json"
    seed_file.write_text(json.dumps(seed or []), encoding="utf-8")
    cmd = [
        sys.executable, "-m", "kiss.tests.remote_daemon",
        "--password", password, "--seed-file", str(seed_file),
        "--config", json.dumps(config or {}), "--port", str(port), "--host", host,
    ]
    if certfile is not None and keyfile is not None:
        cmd += ["--certfile", str(certfile), "--keyfile", str(keyfile)]
    process = subprocess.Popen(
        cmd, env=child_env, cwd=str(home), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdout is not None
    try:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"remote daemon exited with {process.poll()} before seeding")
        seeded = json.loads(line)
        task_ids, pid = seeded["tasks"], seeded["pid"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"remote daemon exited with {process.returncode}")
            # A stale file from an earlier daemon of the same home names
            # another pid: wait for this child's record.
            endpoint = local_endpoint.read_endpoint(endpoint_file)
            if endpoint is not None and endpoint.pid == pid:
                bound = urlsplit(endpoint.url).port
                assert bound is not None, endpoint.url
                return RemoteDaemon(process, home, bound, password, task_ids, pid)
            time.sleep(0.1)
        raise RuntimeError(f"remote daemon did not publish {endpoint_file} within {timeout}s")
    except BaseException:
        # The caller gets no handle to stop: a child that did not start,
        # or whose start was interrupted, ends here.
        _kill(process)
        raise


def _watch_database_redirect(home: Path, original: Path) -> None:
    """Apply a test-requested database path and publish an acknowledgement."""
    from kiss.agents.sorcar import persistence as th

    control = home / "history-db-redirect"
    status = home / "history-db-redirect.status"
    current: Path | None = None
    while True:
        try:
            requested = control.read_text().strip() if control.exists() else ""
        except OSError:
            requested = ""
        desired = Path(requested) if requested else original
        if desired != current:
            th._DB_PATH = desired
            try:
                status.write_text(str(desired))
            except OSError:
                pass
            current = desired
        time.sleep(0.05)


def main() -> None:
    """Child entry point: write the config, seed the database, serve on ``--port`` (0: any)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--config", default="{}")
    parser.add_argument("--certfile", default=None)
    parser.add_argument("--keyfile", default=None)
    args = parser.parse_args()

    from kiss.agents.sorcar import persistence as th
    from kiss.core.config import kiss_home
    from kiss.core.user_config import save_config
    from kiss.server import agent_state
    from kiss.server.web_server import RemoteAccessServer

    home = kiss_home()
    save_config({"remote_password": args.password, **json.loads(args.config)})
    task_ids: dict[str, str] = {}
    seed = json.loads(Path(args.seed_file).read_text(encoding="utf-8")) if args.seed_file else []
    for row in seed:
        task_id, chat_id = th._add_task(row["task"], chat_id=row.get("chat_id", ""))
        if row.get("task_id"):
            db = th._get_db()
            db.execute("UPDATE task_history SET id = ? WHERE id = ?", (row["task_id"], task_id))
            db.commit()
            task_id = row["task_id"]
        if row.get("timestamp") is not None:
            db = th._get_db()
            db.execute(
                "UPDATE task_history SET timestamp = ? WHERE id = ?",
                (float(row["timestamp"]), task_id),
            )
            db.commit()
        if row.get("result"):
            th._save_task_result(row["result"], task_id=task_id)
        if row.get("running"):
            # What a run registers: its state, a parked thread as its worker.
            worker = threading.Thread(target=threading.Event().wait, daemon=True)
            worker.start()
            agent_state.register(agent_state.AgentState(
                task_id, tab_id="seeded-" + task_id, chat_id=chat_id, server_owned=True,
                task_thread=worker, is_task_active=True,
            ))
        task_ids[f"{chat_id}/{row['task']}"] = task_id
    th._close_thread_db()
    threading.Thread(
        target=_watch_database_redirect, args=(home, Path(th._DB_PATH)), daemon=True,
    ).start()
    print(json.dumps({"tasks": task_ids, "pid": os.getpid()}), flush=True)
    server = RemoteAccessServer(
        host=args.host, port=args.port, certfile=args.certfile, keyfile=args.keyfile,
        url_file=home / "remote-url.json",
        local_endpoint_file=home / local_endpoint.LOCAL_ENDPOINT_FILE,
    )
    server.start()


if __name__ == "__main__":
    main()
