# Agent Trajectory Visualizer

A modern browser-based visualizer for KISS agent trajectories.

## Installation

The visualizer is part of the KISS Agent Framework. Flask is included as a dependency. See the main [README.md](../../../README.md) for installation instructions.

## Usage

Run the visualizer server:

```bash
uv run python -m kiss.viz_trajectory.server <artifact_directory>
```

Or directly:

```bash
uv run python src/kiss/viz_trajectory/server.py <artifact_directory>
```

**Options:**

- `--host HOST`: Host to bind to (default: 127.0.0.1)
- `--port PORT`: Port to bind to (default: 5050)

**Example:**

```bash
uv run python -m kiss.viz_trajectory.server artifacts --port 5050
```

Then open your browser to `http://127.0.0.1:5050` to view the trajectories.

## Features

- **Modern UI**: Dark theme with smooth animations
- **Sidebar Navigation**: List of all trajectories sorted by start time
- **Markdown Rendering**: Full markdown support for message content
- **Code Highlighting**: Syntax highlighting for fenced code blocks (plus lightweight Python tool-call highlighting)
- **Message Display**: Clean, organized view of agent conversations
- **Metadata Display**: Shows agent ID, model, steps, tokens, and budget information

## Trajectory Format

Trajectories are automatically saved by KISSAgent instances to the artifacts directory. Each trajectory file contains:

- Complete message history with token usage and budget information
- Tool calls and results
- Configuration used
- Timestamps
- Budget and token usage statistics

The visualizer reads these YAML trajectory files and displays them in an interactive web interface.

## Authors

- Koushik Sen (ksen@berkeley.edu)
