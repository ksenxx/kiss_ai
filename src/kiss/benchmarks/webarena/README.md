# WebArena Benchmark for KISS Sorcar

Runs KISS Sorcar on [WebArena](https://github.com/web-arena-x/webarena) using
Sorcar's built-in Playwright browser tools.

## Setup

### 1. Clone WebArena and start the web application servers

WebArena requires five web apps running locally (Reddit, Shopping, CMS, GitLab,
Maps). Follow the [WebArena setup guide](https://github.com/web-arena-x/webarena#installation)
to start them via Docker:

```bash
git clone https://github.com/web-arena-x/webarena
cd webarena
docker compose up -d
```

### 2. Generate task configs with your server URLs

```bash
python scripts/generate_test_data.py
```

This produces config JSON files under `config_files/` with your local server
URLs injected as `start_url`.

### 3. Install kiss-agent-framework

```bash
uv pip install kiss-agent-framework
```

## Quick Run

```bash
uv run python -m kiss.benchmarks.webarena.run \
    --config-dir path/to/webarena/config_files \
    --model claude-opus-4-6
```

## Quick Test (first 5 tasks)

```bash
uv run python -m kiss.benchmarks.webarena.run \
    --config-dir path/to/webarena/config_files \
    --model claude-opus-4-6 \
    --max-tasks 5
```

## Notes

- Results are saved to `src/kiss/benchmarks/results/webarena_results.json`.
- Scoring currently covers `string_match` eval tasks. Tasks with `url_match`
  or `program_html` eval types are logged but not scored (marked `score: -1`).
- Sorcar runs with its browser tools enabled (Playwright), so no `--no-web` flag.

## References

- [WebArena Paper](https://arxiv.org/abs/2307.13854)
- [WebArena GitHub](https://github.com/web-arena-x/webarena)
- [WebArena Leaderboard](https://webarena.dev/#leaderboard)
