#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Simple Flask server for visualizing agent trajectories."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template

# Set template folder relative to this file
template_dir = Path(__file__).parent / "templates"
app = Flask(__name__, template_folder=str(template_dir))
app.json.sort_keys = False  # type: ignore[attr-defined]  # Preserve key order in JSON
ARTIFACT_DIR: Path | None = None


def _parse_trajectory_yaml(file_path: Path) -> dict:
    """Parse a single saved agent trajectory YAML into the JSON shape used by the UI.

    This is intentionally minimal: only fields used by the browser UI are returned.
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
    global_max_budget = data.get("global_max_budget") or agent_cfg.get("global_max_budget") or 0.0

    return {
        # Basic metadata
        "name": data.get("name", "Unknown"),
        "id": data.get("id", 0),
        "run_start_timestamp": data.get("run_start_timestamp", 0),
        "run_end_timestamp": data.get("run_end_timestamp", 0),
        "model": data.get("model", "Unknown"),
        "command": data.get("command", "Unknown"),
        # Counters
        "step_count": data.get("step_count", 0),
        "max_steps": data.get("max_steps", 0),
        "tokens_used": data.get("tokens_used", 0),
        "max_tokens": data.get("max_tokens", 0),
        # Local (agent) budget
        "agent_budget_used": data.get("budget_used", 0.0),
        "agent_max_budget": agent_max_budget,
        # Global budget
        "global_budget_used": data.get("global_budget_used", 0.0),
        "global_max_budget": global_max_budget,
        # Actual messages
        "messages": messages,
    }


def _parse_state_dir_timestamp(state_dir: str) -> datetime:
    """Parse state directory name (job_YYYY_MM_DD_HH_MM_SS_random) to datetime."""
    try:
        parts = state_dir.split("_")
        # Expected format: job_YYYY_MM_DD_HH_MM_SS_random
        if len(parts) >= 7 and parts[0] == "job":
            yyyy, mo, dd, hh, mm, ss = map(int, parts[1:7])
            return datetime(yyyy, mo, dd, hh, mm, ss)
    except (ValueError, IndexError):
        pass
    return datetime.min  # fallback for unparseable names


def list_jobs(artifact_dir: Path) -> list[dict]:
    """List all job directories with basic metadata.

    Returns a list of job info dicts sorted by creation time (newest first).
    """
    jobs = []
    for job_dir in artifact_dir.glob("job_*"):
        if not job_dir.is_dir():
            continue
        trajectories_dir = job_dir / "trajectories"
        trajectory_count = (
            len(list(trajectories_dir.glob("trajectory_*.yaml")))
            if trajectories_dir.exists()
            else 0
        )
        jobs.append(
            {
                "name": job_dir.name,
                "trajectory_count": trajectory_count,
            }
        )

    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: _parse_state_dir_timestamp(str(x["name"])), reverse=True)
    return jobs


def load_job_trajectories(artifact_dir: Path, job_name: str) -> list[dict]:
    """Load all trajectory files for a specific job.

    <artifact_dir>/<job_name>/trajectories/trajectory_*.yaml
    """
    trajectories = []
    job_dir = artifact_dir / job_name / "trajectories"

    for file_path in sorted(job_dir.glob("trajectory_*.yaml")):
        try:
            trajectories.append(_parse_trajectory_yaml(file_path))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Sort by run_start_timestamp in ascending order
    trajectories.sort(key=lambda x: x.get("run_start_timestamp", 0))
    return trajectories


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/api/jobs")
def get_jobs():
    """API endpoint to list all job directories."""
    if ARTIFACT_DIR is None:
        return jsonify({"error": "Artifact directory not set"}), 500

    return jsonify(list_jobs(ARTIFACT_DIR))


@app.route("/api/jobs/<job_name>/trajectories")
def get_job_trajectories(job_name: str):
    """API endpoint to get trajectories for a specific job."""
    if ARTIFACT_DIR is None:
        return jsonify({"error": "Artifact directory not set"}), 500

    # Validate job_name to prevent path traversal
    if "/" in job_name or "\\" in job_name or ".." in job_name:
        return jsonify({"error": "Invalid job name"}), 400

    job_dir = ARTIFACT_DIR / job_name
    if not job_dir.exists():
        return jsonify({"error": f"Job '{job_name}' not found"}), 404

    return jsonify(load_job_trajectories(ARTIFACT_DIR, job_name))


def main():
    """Main entry point for the server."""
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

    print("Starting trajectory visualizer server...")
    print(f"Artifact directory: {ARTIFACT_DIR}")
    print(f"Server running at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
