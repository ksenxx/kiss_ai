---
title: PyPI 100 MiB file-size guard and sdist contents
uuid: 913dd711-d5fa-4ea4-a3cf-05f25fbde501
summary: Why release.sh checks every dist file against PyPI's 100 MiB limit before
  uv publish, how the sdist was shrunk with hatch only-include, and the test_release_pypi_size.sh
  suite.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# PyPI 100 MiB file-size guard

## The failure it prevents
PyPI rejects any single file over 100 MiB. `uv publish` uploads the wheel before the sdist, so an
oversize sdist left a wheel-only release that could never be completed, because PyPI refuses to
re-upload a version. The sdist had grown to about 350 MB because it carried benchmark results,
demo media, papers and reports (commit "prevent oversized sdist from breaking PyPI publish").

## The two fixes
1. Packaging: `[tool.hatch.build.targets.sdist] only-include = ["src/kiss",
   "projects/swedefend"]`, which is exactly the wheel's packages. An exclude list was tried
   first and kept falling behind. `[tool.hatch.build] exclude` also drops
   `src/kiss/agents/vscode/node_modules`, a symlink in worktrees that gitignore does not match
   (147 MB).
2. Release guard: `publish_to_pypi` in `scripts/release.sh` deletes old `dist/*.tar.gz` and
   `dist/*.whl`, runs `uv build`, then calls `check_pypi_file_sizes dist/*.tar.gz dist/*.whl`
   (`PYPI_FILE_SIZE_LIMIT = 100 * 1024 * 1024`) before anything is uploaded. On failure it
   prints the size and "Nothing was uploaded. Trim [tool.hatch.build.targets.sdist]..." and
   returns 1. `UV_PUBLISH_TOKEN` must be set, or it stops before `uv publish`.

## Test
`scripts/test_release_pypi_size.sh`, run by
`src/kiss/tests/scripts/test_release_pypi_size.py` (POSIX only, 600 s timeout):
- Part 1 builds the real sdist and wheel with `uv build`. It checks that both are under 100 MiB,
  that the sdist contains only the wheel's packages (no benchmark results, papers, reports or
  node_modules), and that the wheel can be built from the sdist.
- Part 2 sources `release.sh` and runs `publish_to_pypi` with a stub `uv` on PATH. An oversize
  sdist must abort before `uv publish` runs, and a small build must reach it.

## When adding large files
Anything added under `src/kiss/` ships in both distributions. That includes media in
`src/kiss/agents/vscode/media/` and the bundled `kiss_project` the extension build creates,
which `release.sh` removes after packaging. Run the size suite after adding large assets.

## Sources
- `scripts/release.sh` (`publish_to_pypi`, `check_pypi_file_sizes`, `PYPI_FILE_SIZE_LIMIT`)
- `pyproject.toml` (`[tool.hatch.build]`, `[tool.hatch.build.targets.sdist]`, `[tool.hatch.build.targets.wheel]`)
- `scripts/test_release_pypi_size.sh`, `src/kiss/tests/scripts/test_release_pypi_size.py`
