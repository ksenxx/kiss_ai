# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Launch a private KISS Sorcar daemon for benchmark runs.

The daemon gets its own ``KISS_HOME`` (default ``~/.kiss-tb``) so benchmark
tasks never share chats, ledgers or settings with the user's interactive
daemon.  The API keys and the model catalog are copied from ``~/.kiss``;
the persisted settings switch off everything an unattended benchmark agent
must not have: the browser and web-search tools, persistent memory, task
classification, git worktrees and auto-commit.

Usage (from the repo root)::

    uv run python -m kiss.benchmarks.terminal_bench.daemon --home ~/.kiss-tb --port 8797

The Harbor agent (:mod:`kiss.benchmarks.terminal_bench.harbor_agent`) reaches
it through ``<home>/sorcar-local.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

DEFAULT_HOME = Path("~/.kiss-tb").expanduser()
DEFAULT_PORT = 8797
#: Name of the endpoint file the daemon writes under its home.
ENDPOINT_FILE = "sorcar-local.json"

#: Settings written to the private ``config.json``: nothing interactive,
#: no worktrees, no auto-commit, no task classification, no browser, no memory.
DAEMON_CONFIG = {
    "max_budget": 50,
    "use_web_browser": False,
    "auto_commit_mode": False,
    "is_worktree": False,
    "classify_tasks": False,
    "use_memory": False,
    "last_model": "claude-opus-5-5",
}


def prepare_home(home: Path) -> Path:
    """Create the private KISS home and return the scratch work directory.

    Args:
        home: The private ``KISS_HOME`` directory.

    Returns:
        A git-initialised scratch directory used as the daemon's work dir.
    """
    home.mkdir(parents=True, exist_ok=True)
    source = Path.home() / ".kiss"
    for name in ("api_keys.env", "MY_MODELS.json", "MODEL_INFO.json"):
        if (source / name).is_file() and not (home / name).exists():
            shutil.copy(source / name, home / name)
    work_dir = home / "scratch"
    work_dir.mkdir(exist_ok=True)
    if not (work_dir / ".git").exists():
        os.system(
            f"git -C {work_dir} init -q && git -C {work_dir} -c user.email=bench@kiss "
            f"-c user.name=bench commit -q --allow-empty -m init"
        )
    config = dict(DAEMON_CONFIG, work_dir=str(work_dir))
    (home / "config.json").write_text(json.dumps(config, indent=2))
    return work_dir


def build_server(home: Path, port: int) -> Any:
    """The private daemon's server, configured for unattended benchmark clients.

    The server runs no cron scheduler: the user's scheduled jobs (and the
    default ones every installation gets) would spend money on unrelated
    work in the middle of a run, and the hand-over of jobs between homes
    must not see the benchmark home as the user's last daemon home.

    Args:
        home: The private ``KISS_HOME`` (prepared by :func:`prepare_home`).
        port: Local HTTP port of the daemon.

    Returns:
        A ``RemoteAccessServer`` that has not been started.
    """
    work_dir = prepare_home(home)
    os.environ["KISS_HOME"] = str(home)
    os.environ["KISS_SORCAR_LOCAL"] = str(home / ENDPOINT_FILE)

    from kiss.server.web_server import RemoteAccessServer

    return RemoteAccessServer(
        host="127.0.0.1",
        port=port,
        use_tunnel=False,
        work_dir=str(work_dir),
        local_endpoint_file=home / ENDPOINT_FILE,
        scheduler=False,
    )


def main() -> None:
    """CLI entry point: prepare the private home and run the daemon (blocking)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    build_server(args.home.expanduser().resolve(), args.port).start()


if __name__ == "__main__":
    main()
