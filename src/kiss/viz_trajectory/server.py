#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)

"""Simple Flask server for visualizing agent trajectories."""

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template

logger = logging.getLogger(__name__)

# Name of the per-project artifact root directory.  Job directories live under
# ``<root>/.kiss.artifacts/jobs/job_*``.  Keep this in sync with
# ``kiss.core.config._ARTIFACTS_DIR_NAME``.
_ARTIFACTS_DIR_NAME = ".kiss.artifacts"

# Directories that never contain job artifacts; skipping them keeps the
# project-wide job discovery walk fast.
_WALK_SKIP_DIRS = frozenset(
    {".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages"}
)

template_dir = Path(__file__).parent / "templates"
app = Flask(__name__, template_folder=str(template_dir))
app.json.sort_keys = False  # type: ignore[attr-defined]  # Preserve key order in JSON
ARTIFACT_DIR: Path | None = None


def _parse_trajectory_yaml(file_path: Path) -> dict:
    """Parse a single saved agent trajectory YAML into the JSON shape used by the UI.

    This is intentionally minimal: only fields used by the browser UI are returned.

    Args:
        file_path: Path to the trajectory YAML file to parse.

    Returns:
        Dictionary containing trajectory metadata and messages formatted for the UI,
        including name, id, timestamps, model, command, step counts, budget info,
        and the actual messages.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    messages = data.get("messages")
    agent_cfg = (data.get("config") or {}).get("agent") or {}
    agent_max_budget = (
        agent_cfg.get("max_agent_budget")
        or data.get("max_agent_budget")
        or data.get("total_budget")
        or 0.0
    )

    return {
        "name": data.get("name", "Unknown"),
        "id": data.get("id", 0),
        "run_start_timestamp": data.get("run_start_timestamp", 0),
        "run_end_timestamp": data.get("run_end_timestamp", 0),
        "model": data.get("model", "Unknown"),
        "command": data.get("command", "Unknown"),
        "step_count": data.get("step_count", 0),
        "max_steps": data.get("max_steps", 0),
        "tokens_used": data.get("tokens_used", 0),
        "max_tokens": data.get("max_tokens", 0),
        "agent_budget_used": data.get("budget_used", 0.0),
        "agent_max_budget": agent_max_budget,
        "messages": messages,
    }


def _parse_state_dir_timestamp(state_dir: str) -> datetime:
    """Parse state directory name (job_YYYY_MM_DD_HH_MM_SS_random) to datetime.

    Args:
        state_dir: Name of the state directory to parse.

    Returns:
        Datetime object parsed from the directory name, or datetime.min if parsing fails.
    """
    try:
        parts = state_dir.split("_")
        if len(parts) >= 7 and parts[0] == "job":
            yyyy, mo, dd, hh, mm, ss = map(int, parts[1:7])
            return datetime(yyyy, mo, dd, hh, mm, ss)
    except (ValueError, IndexError):
        logger.debug("Exception caught", exc_info=True)
        pass
    return datetime.min


def _project_root_for(artifact_dir: Path) -> Path | None:
    """Return the project root for a standard jobs directory layout.

    Args:
        artifact_dir: Path to the jobs directory.

    Returns:
        The project root (the parent of ``.kiss.artifacts``) when
        ``artifact_dir`` follows the standard ``<root>/.kiss.artifacts/jobs``
        layout, or ``None`` otherwise.
    """
    resolved = artifact_dir.resolve()
    if resolved.name == "jobs" and resolved.parent.name == _ARTIFACTS_DIR_NAME:
        return resolved.parent.parent
    return None


def _add_job_dirs(jobs_dir: Path, found: dict[str, Path]) -> None:
    """Record every ``job_*`` subdirectory of ``jobs_dir`` into ``found``.

    When a job name already exists, the most recently modified directory wins
    so the freshest copy of a job is displayed.

    Args:
        jobs_dir: A ``jobs`` directory containing ``job_*`` subdirectories.
        found: Mapping of job name to its absolute path, updated in place.
    """
    if not jobs_dir.is_dir():
        return
    for job_dir in jobs_dir.glob("job_*"):
        if not job_dir.is_dir():
            continue
        existing = found.get(job_dir.name)
        if existing is None or job_dir.stat().st_mtime > existing.stat().st_mtime:
            found[job_dir.name] = job_dir


def discover_job_dirs(artifact_dir: Path) -> dict[str, Path]:
    """Find every job directory the viewer should display.

    Always includes ``job_*`` subdirectories of ``artifact_dir``.  When
    ``artifact_dir`` follows the standard ``<root>/.kiss.artifacts/jobs``
    layout, also discovers jobs stored in any other ``.kiss.artifacts/jobs``
    directory beneath the project root - including git worktrees under
    ``.kiss-worktrees/`` and nested artifact directories - so the viewer
    aggregates jobs from every run.

    Args:
        artifact_dir: The primary jobs directory.

    Returns:
        Mapping of job directory name to its absolute path.
    """
    found: dict[str, Path] = {}
    _add_job_dirs(artifact_dir, found)

    project_root = _project_root_for(artifact_dir)
    if project_root is not None:
        for dirpath, dirnames, _ in os.walk(project_root):
            current = Path(dirpath)
            if current.name == "jobs" and current.parent.name == _ARTIFACTS_DIR_NAME:
                _add_job_dirs(current, found)
                # Do not descend into individual job directories; only follow
                # nested ``.kiss.artifacts`` roots to find jobs stored within.
                dirnames[:] = [d for d in dirnames if d == _ARTIFACTS_DIR_NAME]
            else:
                dirnames[:] = [d for d in dirnames if d not in _WALK_SKIP_DIRS]
    return found


def find_job_dir(artifact_dir: Path, job_name: str) -> Path | None:
    """Return the directory for ``job_name`` across all known job roots.

    Args:
        artifact_dir: The primary jobs directory.
        job_name: Name of the job directory to locate.

    Returns:
        The absolute path to the job directory, or ``None`` if not found.
    """
    direct = artifact_dir / job_name
    if direct.is_dir():
        return direct
    return discover_job_dirs(artifact_dir).get(job_name)


def list_jobs(artifact_dir: Path) -> list[dict]:
    """List all job directories with basic metadata.

    Aggregates jobs from every ``.kiss.artifacts/jobs`` root beneath the
    project (main directory, git worktrees, and nested artifact dirs).

    Args:
        artifact_dir: Path to the artifact directory containing job folders.

    Returns:
        List of job info dictionaries sorted by creation time (newest first),
        each containing 'name' and 'trajectory_count' keys.
    """
    jobs = []
    for name, job_dir in discover_job_dirs(artifact_dir).items():
        trajectories_dir = job_dir / "trajectories"
        trajectory_count = (
            len(list(trajectories_dir.glob("trajectory_*.yaml")))
            if trajectories_dir.exists()
            else 0
        )
        jobs.append(
            {
                "name": name,
                "trajectory_count": trajectory_count,
            }
        )

    jobs.sort(key=lambda x: _parse_state_dir_timestamp(str(x["name"])), reverse=True)
    return jobs


def load_job_trajectories(artifact_dir: Path, job_name: str) -> list[dict]:
    """Load all trajectory files for a specific job.

    Resolves ``job_name`` across all known job roots and loads from
    ``<job_dir>/trajectories/trajectory_*.yaml``.

    Args:
        artifact_dir: Path to the artifact directory containing job folders.
        job_name: Name of the job directory to load trajectories from.

    Returns:
        List of trajectory dictionaries sorted by run_start_timestamp in ascending order.
    """
    trajectories: list[dict] = []
    job_dir = find_job_dir(artifact_dir, job_name)
    if job_dir is None:
        return trajectories
    trajectories_dir = job_dir / "trajectories"

    for file_path in sorted(trajectories_dir.glob("trajectory_*.yaml")):
        try:
            trajectories.append(_parse_trajectory_yaml(file_path))
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            print(f"Error loading {file_path}: {e}")

    trajectories.sort(key=lambda x: x.get("run_start_timestamp", 0))
    return trajectories


@app.route("/")
def index():
    """Serve the main HTML page.

    Returns:
        Rendered HTML template for the trajectory visualizer.
    """
    return render_template("index.html")


@app.route("/api/jobs")
def get_jobs():
    """API endpoint to list all job directories.

    Returns:
        JSON response containing a list of job info dictionaries, or an error
        response with status 500 if the artifact directory is not set.
    """
    if ARTIFACT_DIR is None:
        return jsonify({"error": "Artifact directory not set"}), 500

    return jsonify(list_jobs(ARTIFACT_DIR))


@app.route("/api/jobs/<job_name>/trajectories")
def get_job_trajectories(job_name: str):
    """API endpoint to get trajectories for a specific job.

    Args:
        job_name: Name of the job to retrieve trajectories for.

    Returns:
        JSON response containing a list of trajectory dictionaries, or an error
        response with status 400 for invalid job names, 404 if job not found,
        or 500 if the artifact directory is not set.
    """
    if ARTIFACT_DIR is None:
        return jsonify({"error": "Artifact directory not set"}), 500

    if "/" in job_name or "\\" in job_name or ".." in job_name:
        return jsonify({"error": "Invalid job name"}), 400

    if find_job_dir(ARTIFACT_DIR, job_name) is None:
        return jsonify({"error": f"Job '{job_name}' not found"}), 404

    return jsonify(load_job_trajectories(ARTIFACT_DIR, job_name))


def main():
    """Main entry point for the server.

    Parses command-line arguments, validates the artifact directory, and starts
    the Flask development server for trajectory visualization.

    Returns:
        None. Exits with status 1 if the artifact directory does not exist.
    """
    global ARTIFACT_DIR

    parser = argparse.ArgumentParser(description="Visualize agent trajectories")
    parser.add_argument(
        "artifact_dir",
        type=str,
        help="Path to the artifact directory containing trajectory files",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to bind to (default: 5050)",
    )

    args = parser.parse_args()
    ARTIFACT_DIR = Path(args.artifact_dir)

    if not ARTIFACT_DIR.exists():
        print(f"Error: Artifact directory '{ARTIFACT_DIR}' does not exist")
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    print("Starting trajectory visualizer server...")
    print(f"Artifact directory: {ARTIFACT_DIR}")
    print(f"Server running at {url}")

    def _open_browser() -> None:
        time.sleep(2)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
