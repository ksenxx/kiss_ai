---
title: GitHub Actions workflows (.github/workflows) and their known defects
uuid: 2a4e54b8-f254-4c2b-ad6f-70da53674bb3
summary: GitHub Actions check.yml and Windows workflows; `uv run check --clean` is
  parsed as --clean-only (argparse prefix), so CI runs no checks; Windows UI jobs
  expect a root VSIX.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# GitHub Actions workflows

`.github/workflows/` holds four workflows. None of them runs the pytest suite or the JS tests.
Tests run locally in parallel splits (see `dev-parallel-test-splits`).

| File | Trigger | What it does |
|---|---|---|
| `check.yml` ("Code Quality") | push / PR to `main` | ubuntu, setup-uv, Python 3.13, then `uv run check --clean` |
| `windows-test.yml` ("Windows Extension Test") | manual, or a push that changes the workflow file itself | installs VS Code and the extension, sets up uv/Python, verifies the package, tests Anthropic connectivity, runs an agent task, takes UI screenshots |
| `windows-ui.yml` | manual, or a push touching the workflow or `kiss-sorcar.vsix` | installs `${{ github.workspace }}\kiss-sorcar.vsix` and uploads UI screenshots |
| `windows-workflow.yml` | same as above | visual walk-through: opens the panel, types a task, uploads screenshots |

## Known defect: `check.yml` runs no checks
`check.py` defines `--no-clean`, `--clean-only` and `--full`, but no `--clean`. Python's
argparse accepts unambiguous prefixes by default (`allow_abbrev=True`, and `check.py` does not
turn it off), so `--clean` is parsed as `--clean-only`. `main()` then cleans build artifacts
and returns 0 without running any stage, and the workflow passes whatever the code state.
Reproduce with an argparse parser built from the same three options:
`parse_args(["--clean"])` gives `clean_only=True`. The fix is to change the workflow to
`uv run check` (or `uv run check --full` once node and npm are set up in the job), or to pass
`allow_abbrev=False` so the typo fails loudly.

## Known defect: VSIX path in the Windows UI workflows
`windows-ui.yml` and `windows-workflow.yml` install `kiss-sorcar.vsix` from the repo root, and
their `paths` filter watches the root `kiss-sorcar.vsix`. The build writes the file to
`src/kiss/agents/vscode/kiss-sorcar.vsix` (`VSIX_FILE` in `scripts/release.sh`, `VSIX` in
`install.sh`), and it is gitignored in origin. Only kiss_ai's release tip commit carries it,
at that nested path. `windows-test.yml` does use the nested path,
`src\kiss\agents\vscode\kiss-sorcar.vsix`. For the other two, the push trigger fires only when the workflow file itself changes (never
for a VSIX build), and every run finds no file unless someone places a VSIX at the root.

## Sources
- `.github/workflows/check.yml`, `.github/workflows/windows-test.yml`, `.github/workflows/windows-ui.yml`, `.github/workflows/windows-workflow.yml`
- `src/kiss/scripts/check.py` (`main`: argparse options)
- `scripts/release.sh` (`VSIX_FILE`, `tree_with_vsix`), `install.sh` (step `[4/5]`)
