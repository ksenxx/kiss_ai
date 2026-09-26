---
title: Repository directory map (top level, src/kiss, benchmarkings, projects, website)
uuid: df85e137-5137-4cc8-9164-03855bb7608d
summary: 'Where things live in the KISS repo: src/kiss packages, scripts/, tests,
  benchmarkings (private), projects (swedefend ships; rest experiments), website mirror,
  connectors.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Repository directory map

## Top level
| Path | What |
|---|---|
| `src/kiss/` | the Python package (below) |
| `scripts/` | bash release, install and deploy scripts, plus `exclude.json` and bash tests (`dev-overview`) |
| `install.sh` | source installer (`dev-install-sh-flow`) |
| `rsorcar`, `sorcar-docker`, `sorcar` | remote deploy, Docker launcher, and a checkout-local `uv run python -m kiss.agents.sorcar.worktree_sorcar_agent` launcher |
| `Dockerfile`, `.dockerignore` | image used by `sorcar-docker` |
| `pyproject.toml`, `uv.lock`, `pyrightconfig.json`, `.pylintrc` | packaging and lint config |
| `conftest.py` | root pytest config (`dev-test-conftest-isolation`) |
| `API.md` | generated (`dev-api-docs-generation`) |
| `README.md` | user docs. Its version string is bumped by `release.sh` |
| `connectors/` | curated MCP connector catalog (`catalog.json`, `enable.py`, `verify.py`) |
| `knowledge/` | this knowledge base |
| `website/kisssorcar.github.io/` | mirror of the public site repo (index.html, `llms.txt`, `llms-full.txt`, `docs/*.md`, blog). Edits here must also be pushed to the site repo |
| `assets/` | logos, GIFs and screenshots used by the README |
| `benchmarkings/`, `papers/`, `reports/`, `marketing/`, `templates/`, `tmp/`, `SCRATCH.md`, `RECIPES.md` | listed in `scripts/exclude.json`, so never published to kiss_ai |
| `projects/` | experiments and one shipped package (below) |
| `test_data/` | small fixtures, e.g. a calculator package |
| `.github/workflows/` | CI (`dev-ci-workflows`) |

## `src/kiss/`
- `core/`: agent loop, models (`core/models/MODEL_INFO.json`), memoryfield, printers, config,
  `_version.py`. It must not import anything outside `core/`.
- `agents/sorcar/`: Sorcar agents, persistence (`sorcar.db`), worktrees. Imports only itself and
  `core`.
- `agents/vscode/`: VS Code extension (TypeScript `src/`, `media/`, `test/`, `copy-kiss.sh`,
  `scripts/apply-brand.js`, `scripts/package-vsix.js`).
- `agents/third_party_agents/`: channel agents (`*_sea.py`, the `kiss-*` console scripts).
- `agents/seas/`: bundled slash-command SEAs. `agents/obsolete/`: legacy code.
- `server/`: `kiss-web` daemon and web app.
- `scripts/`: Python maintenance tools: `check.py`, `generate_api_docs.py`, `update_models.py`
  (refresh the model catalog from vendor APIs), `update_responses_api_support.py`,
  `sync_db.py`, `carry_over_tables.py`, `relocate_work_dir.py`, `db_fingerprint.py`,
  `running_tasks.py`, `remote_config.py`, `merge_memory_pages.py`, `cost_report.py`,
  `cost_levers_experiment.py`, `redundancy_analyzer.py`.
- `tests/`: the whole pytest suite (`dev-test-conventions-and-layout`).
- `viz_trajectory/`: trajectory visualizer.
- `SYSTEM.md`, `SYSTEM_LITE.md`, `TIPS.md`, `INJECTIONS.md`, `SAMPLE_TASKS.md`: prompts and docs
  shipped in the package.

## `benchmarkings/` (overview; private)
- `harbor_sorcar_agent.py`: Harbor agent wrapper that runs SorcarAgent inside a Harbor trial
  container for data-eng-bench (`HarborEnvShim` mimics `DockerManager`). `compute_scores.py`
  and `make_report.py` produce leaderboard metrics and reports. `jobs/` holds results.
- `harnesstax/`: harness-overhead experiments (trials, runners for SWE-bench and Terminal-Bench 2,
  reports). `mypy src/` type-checks it because tests import it.
- `memoryfield/bench_public.py`: memory retrieval quality and latency on LoCoMo-style public
  benchmarks.
- `task_classifier/`: benchmark of the pre-run task classifiers (`tasks.jsonl`, `results.json`).
- Tests for these live in `src/kiss/tests/benchmarkings/`.

## `projects/` (overview)
- `swedefend/`: the only project that ships. It is in both wheel and sdist (`packages` /
  `only-include`) and ruff treats it as first-party. It is a defense harness against SWExploit
  attacks on SWE-agents. Tests are in `src/kiss/tests/swedefend/`.
- The rest are self-contained experiments with their own READMEs: `bespoke_tpch_x4` (TPC-H
  engine reproduction), `kv_adversarial` (HydraKV C++ store), `speed-audit-2026-09-19`,
  `cost-levers-followup-2026-09-20`, `cost-levers-implementation-plan.md`.

## Sources
- repository tree; `scripts/exclude.json`; `pyproject.toml` (`[tool.hatch.build.targets.*]`, `[tool.ruff]`)
- `benchmarkings/harbor_sorcar_agent.py`, `benchmarkings/compute_scores.py`, `benchmarkings/memoryfield/bench_public.py`, `benchmarkings/task_classifier/run_benchmark.py`
- `projects/swedefend/__init__.py`, `website/README.md`, `connectors/README.md`
