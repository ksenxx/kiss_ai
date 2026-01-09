#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Simple Flask server for visualizing agent trajectories."""

import argparse
import json
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

    trajectory_json = data.get("trajectory", "[]")
    if isinstance(trajectory_json, str):
        trajectory_data = json.loads(trajectory_json)
    else:
        trajectory_data = trajectory_json

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
        "trajectory": trajectory_data,
    }


def _parse_state_dir_timestamp(state_dir: str) -> datetime:
    """Parse state directory name (SS_MM_HH_DD_MM_YYYY_random) to datetime."""
    try:
        parts = state_dir.split("_")
        if len(parts) >= 6:
            ss, mm, hh, dd, mo, yyyy = map(int, parts[:6])
            return datetime(yyyy, mo, dd, hh, mm, ss)
    except (ValueError, IndexError):
        pass
    return datetime.min  # fallback for unparseable names


def load_trajectories(artifact_dir: Path) -> dict[str, list[dict]]:
    """Load all trajectory files from the artifact directory.

    `KISSAgent._save()` writes YAML files under: <artifact_dir>/<run_subdir>/trajectory_*.yaml
    So we scan subdirectories and group results by that run subdir.
    """
    state_to_trajectories: dict[str, list[dict]] = {}

    for file_path in sorted(artifact_dir.glob("*/*trajectory_*.yaml")):
        try:
            state_dir = file_path.parent.name
            state_to_trajectories.setdefault(state_dir, []).append(
                _parse_trajectory_yaml(file_path)
            )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Sort by run_start_timestamp in ascending order
    for trajectories in state_to_trajectories.values():
        trajectories.sort(key=lambda x: x.get("run_start_timestamp", 0))

    # Sort state directories by creation time (descending - newest first)
    sorted_states = sorted(
        state_to_trajectories.keys(),
        key=_parse_state_dir_timestamp,
        reverse=True,
    )
    return {state: state_to_trajectories[state] for state in sorted_states}


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/api/trajectories")
def get_trajectories():
    """API endpoint to get all trajectories."""
    if ARTIFACT_DIR is None:
        return jsonify({"error": "Artifact directory not set"}), 500

    return jsonify(load_trajectories(ARTIFACT_DIR))


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
