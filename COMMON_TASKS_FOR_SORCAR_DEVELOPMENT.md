# How to use the file?

Select a task below in sorcar editor and press cmd/ctrl-L to run the task in the chat window.

## increase test coverage

can you write integration tests with no mocks or test doubles to achieve 100% branch coverage of the files under src/kiss/agents/sorcar/? Please check the branch coverage first for the existing tests with the coverage tool.  Then try to reach uncovered branches by crafting integration tests without any mocks, test doubles. You MUST repeat the task until you get 100% branch coverage or you cannot increase branch coverage after 10 tries.

## code review

find redundancy, duplication, AI slop, lack of abstractions, and inconsistencies in the code of the project, and fix them. Make sure that you test every change by writing and running integration tests with no mocks or test doubles to achieve 100% branch coverage. Do not change any functionality. Make that existing tests pass.

## check

run 'uv run check --full' and fix

## test

run 'uv run pytest -v' with 900 seconds timeout and fix tests

## race detection

can you please work hard and carefully to precisly detect all actual race conditions in src/kiss/agents/sorcar/sorcar.py? You can add random delays within 0.1 seconds before racing events to reliably trigger a race condition to confirm a race condition.

## test compaction

can you use src/kiss/scripts/redundancy_analyzer.py to get rid of redundant test methods in src/kiss/tests/?  Make sure that you don't decrease the overall branch coverage after removing the redundant test methods.

# Merge View Logic in Sorcar

## Overview

The merge view is an inline diff review system that shows the user what changes
the agent made during a task, allowing per-hunk accept/reject decisions directly
in the code-server editor. It uses red/blue line decorations (red = old lines,
blue = new lines) instead of a side-by-side diff viewer.

## How hunks are determined (what is shown to the user)

When an agent task finishes (in `run_agent_thread` in `sorcar.py`), the system
calls `_prepare_merge_view()` from `code_server.py`. This function computes
which hunks to show by comparing the state before and after the agent ran:

1. **Before the task starts**, the system captures:

   - `pre_hunks`: existing git diff hunks (`git diff -U0 HEAD`) — these are
     pre-existing uncommitted changes from previous accepted tasks.
   - `pre_untracked`: set of untracked files.
   - `pre_file_hashes`: MD5 hashes of all files that have diffs or are untracked.
   - Copies of untracked/modified files are saved to `untracked-base/` directory
     via `_save_untracked_base()` so we have a snapshot of the pre-task state.

1. **After the task finishes**, `_prepare_merge_view()`:

   - Runs `git diff -U0 HEAD` again to get `post_hunks`.
   - For each changed file, it determines which hunks are **new** (from this
     agent task) vs **pre-existing** (from before the task):
     - If a saved base copy exists (`untracked-base/{filename}`), it diffs
       that saved copy against the current file using `diff -U0` to get only
       the agent's changes.
     - If no saved base exists (tracked file without pre-existing changes),
       it filters out any hunks whose `(base_start, base_count)` match
       the pre-task hunks.
     - If the file's MD5 hash hasn't changed from pre-task, it's skipped entirely.
   - Newly created (untracked) files are shown in full as additions.
   - Modified pre-existing untracked files are diffed against the saved base.

1. **Hunk coordinates** in the merge manifest (`pending-merge.json`):

   - `bs`: 0-based line number in the base file where old content starts.
   - `bc`: number of old (deleted) lines from the base.
   - `cs`: 0-based line number in the current file where new content starts.
     For pure deletions (`cc=0`), `cs` is kept as the raw diff value (insertion
     point); for modifications/additions (`cc>0`), `cs = diff_cs - 1` (0-based).
   - `cc`: number of new (added) lines in the current file.

## How hunks are displayed in the editor

The merge manifest is written to `pending-merge.json`. The VS Code extension
(`_CS_EXTENSION_JS` in `code_server.py`) polls for this file every 800ms. When
found, `openMerge()` runs:

1. Saves all open files and clears previous decorations.
1. For each file in the manifest:
   - Opens the current file in the editor.
   - Reads the base file content.
   - For each hunk (processed top-to-bottom), inserts the old (deleted) lines
     into the current file above the new lines, keeping track of line offsets.
   - Records each hunk's position as `{os, oc, ns, nc}`:
     - `os`: line where old (red) lines start in the merged view.
     - `oc`: count of old (red) lines.
     - `ns`: line where new (blue) lines start (right after the red lines).
     - `nc`: count of new (blue) lines.
1. Applies red background decorations to old lines and blue to new lines.
1. Navigates to the first hunk and sets `curHunk`.

The user sees interleaved red/blue blocks in the editor, where red = what was
there before (to be removed on accept) and blue = what the agent wrote.

## User interactions

- **Accept Change** (`kiss.acceptChange`): Deletes the old (red) lines for the
  current hunk, keeping the new (blue) lines. Adjusts positions of subsequent hunks.
- **Reject Change** (`kiss.rejectChange`): Deletes the new (blue) lines, keeping
  the old (red) lines. Adjusts positions of subsequent hunks.
- **Next/Previous Change** (`kiss.nextChange`/`kiss.prevChange`): Navigates
  between hunks across all files, scrolling the editor to center on the hunk.
- **Accept All** (`kiss.acceptAll`): Accepts all hunks in all files by deleting
  all old (red) lines (processed in reverse order to preserve positions).
- **Reject All** (`kiss.rejectAll`): Rejects all hunks by deleting all new
  (blue) lines (processed in reverse order).

A merge toolbar appears in the chatbot UI with these buttons when a
`merge_started` event is broadcast. The input box is disabled with the
placeholder "Resolve all diffs in the merge view to continue…".

## What happens when the user closes Sorcar without accepting all hunks

When Sorcar is closed (browser tab closed or server shutdown) while hunks remain
unreviewed:

- **The merged file on disk is restored to only contain the new lines.** During
  `_prepare_merge_view()`, copies of each current file (containing only the
  agent's new lines) are saved to `merge-current/`. On shutdown, `_cleanup()`
  detects the `merging` flag and calls `_restore_merge_files()` which copies
  those saved versions back to the work directory, effectively "accepting all"
  unreviewed changes. This also runs at startup for crash recovery.
- **All merge support files are cleaned up.** `_restore_merge_files()` calls
  `_cleanup_merge_data()` which removes `merge-temp/`, `merge-current/`,
  `untracked-base/`, and `pending-merge.json`.
- **The `merging` flag** is reset to `False` in `_cleanup()`, so on restart
  new tasks can be submitted immediately.

## Completion flow

When all hunks across all files have been individually accepted/rejected:

1. `checkAllDone()` in the extension fires, saves all files, and POSTs
   `{action: "all-done"}` to `/merge-action`.
1. The server sets `merging = False`, broadcasts `merge_ended`, and calls
   `_cleanup_merge_data()` to remove `merge-temp/` and `untracked-base/`.
1. The chatbot UI hides the merge toolbar and re-enables the input box.

The same flow applies when the user clicks "Accept All" or "Reject All".

# documentation update

Can you read all \*.md files, except API.md, in the project carefully and check and precisely fix any inconsistencies with the code in the project?

# porting 'autresearch' to src/kiss/agents/autoresearch/

can you implement the 'autoresearch' agent at https://github.com/karpathy/autoresearch in the folder src/kiss/agents/autoresearch/ using src/kiss/agents/sorcar/sorcar_agent.py and src/kiss/agents/sorcar/sorcar.py ? Please write integration tests with no mocks or test doubles to achieve 100% branch coverage.  Please do it precisely and do the the most intuitive design for the ambiguous parts.  Simplify code.  Use the browser tool if necessary.

can you write documentation on 'autoresearch' and how to use it, how it works, and what are the advantages of KISS based 'autresesearch' over original 'autoresearch' in src/kiss/agents/autoresearch/README.md?
