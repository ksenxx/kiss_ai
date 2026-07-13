# PROGRESS — Issue #41: `sorcar --version` (current task)

## Task

Address https://github.com/ksenxx/kiss_ai/issues/41 — `sorcar --version`
should print `sorcar <version>` (from `kiss.__version__`) and exit 0
immediately, no task/model/keys needed. Implementation by claude-fable-5,
review/debugging by gpt-5.6-sol.

## What was done (chronological)

1. Web research (10 sites, ./tmp/information-issue41.md): consensus =
   argparse `action='version'`; `-V` spelling (pip precedent) since `-v`
   is taken by `--verbose`; version action exits 0 during `parse_args`
   before any app logic.
1. Implementation (claude-fable-5):
   - `src/kiss/agents/sorcar/cli_helpers.py`: added
     `from kiss import __version__` and, as the first argument in
     `_build_arg_parser()`:
     ```python
     parser.add_argument(
         "-V", "--version", action="version",
         version=f"sorcar {__version__}",
         help="Show the sorcar version and exit",
     )
     ```
     Literal `"sorcar ..."` (not `%(prog)s`) so output is stable
     regardless of argv[0] (pytest, python -m, console script).
   - New e2e test `src/kiss/tests/agents/sorcar/test_cli_version_flag.py`
     (5 tests, no mocks): `--version` & `-V` → stdout `sorcar <ver>\n`,
     exit 0, empty stderr; `--version -t x --worktree` still exits 0
     printing version (argparse first-flag-wins, before the
     `_reject_interactive_only_flags` guard); `kiss.__version__` ==
     `kiss._version.__version__`; `--vers` rejected (allow_abbrev=False).
   - Docs: README.md + website/kisssorcar.github.io/docs/cli.md — added
     `sorcar --version` example and `-V, --version` row to CLI options
     tables.
1. Review (gpt-5.6-sol) — findings & resolutions:
   - `sorcar mcp --version` → exit 2 (mcp subparser has its own parser,
     dispatched before main parser); acceptable, issue only asks for
     `sorcar --version`.
   - `get_default_model()` default evaluated at parser-build time is
     cheap env-var checks only — `--version` needs no API keys
     (verified with all key env vars unset: still exits 0).
   - Verified real console script: `uv run sorcar --version`,
     `-V`, `--version -t x --worktree`, `-t x --version` all print
     `sorcar 2026.7.19`, exit 0.
   - MISSED WIRING found: `website/kisssorcar.github.io/llms-full.txt`
     is the concatenation of docs/\*.md and contained the stale cli.md
     section → updated to match (verified with awk/diff extraction).
   - No completion tables/`_INTERACTIVE_ONLY_FLAGS`/import-cycle issues
     (kiss/__init__.py only imports \_version).
1. Verification: `uv run check --full` all green; 71 CLI-parser tests
   (version flag + non-interactive validation + default modes + only
   sorcar agent + wave2/wave3 CLI bugs) all pass.

______________________________________________________________________

# PROGRESS — Run all tests in parallel, diagnose & fix failures — Session 2 (previous task)

## Task

Run all tests split by test-method count into (cores − 2 = 8) parallel
splits via `run_parallel`, report causes of failing tests, classify
each failure as a project bug vs a test bug, and fix accordingly.
(Session 1's fixes — the persistence.py `_get_db()` TOCTOU project-bug
fix and the SIGINT `preexec_fn` test-bug fix in
test_install_script_npm_ignore_scripts.py — were already committed as
c64acc16.)

## Session 2

1. Collected 6114 tests, split round-robin into 8 files (765/764 each),
   ran all 8 splits concurrently via `run_parallel`
   (`uv run pytest --no-cov -q -p no:cacheprovider @./tmp/split_N.txt`).
1. Results: splits 1–7 fully green (761+757+756+755+759+757+762 passed,
   plus skips/subtests). Split 0: exactly ONE failure — a HANG at test
   #410, `test_bughunt9_c_sigterm_during_cleanup.py::TestSigtermDuringSigintCleanup::test_sigint_shutdown_arms_sigterm_guard`
   (`pytest-timeout` dump: main thread parked forever in
   `KqueueSelector.select()` inside `srv.start()` →
   `asyncio.run(self._serve_async())`; server had printed its
   listening banner; only idle helper threads alive). All 6113 other
   tests passed. Confirmed the previous 5 disk-I/O persistence
   failures are gone (fixed in c64acc16).
1. Root cause of the hang — **TEST BUG**, the same SIG_IGN inheritance
   family as Session 1's install-script flake but in a different test:
   `run_parallel` workers launch pytest as a background job of a
   non-job-control shell (`sh -c 'pytest … &'`), which per POSIX starts
   the job with SIGINT set to `SIG_IGN`. CPython then never installs
   its default `KeyboardInterrupt` handler (verified:
   `signal.getsignal(SIGINT) == SIG_IGN` inside a backgrounded pytest;
   a backgrounded python survives a self-delivered SIGINT). The test's
   interrupter thread `os.kill(pid, SIGINT)` is therefore a silent
   no-op, `srv.start()` never unwinds, and the test hangs. Not a
   web_server.py bug: `RemoteAccessServer` deliberately relies on
   Python's default SIGINT→KeyboardInterrupt path, which is correct in
   a real terminal (test passes instantly in the foreground and passed
   in the earlier serial re-run).
1. Fix (test code only): in
   `test_bughunt9_c_sigterm_during_cleanup.py`, after snapshotting the
   old handlers, install `signal.signal(signal.SIGINT, signal.default_int_handler)` for the duration of the test (restored
   in the existing `finally`), with a comment documenting the POSIX
   background-job SIG_IGN semantics. This gives the test the same
   signal environment as a real terminal regardless of how pytest was
   launched.
1. Verification: test passes in foreground and 4/4 iterations as a
   `sh -c '… &'` background job (previously hung/failed 100% when
   backgrounded); the first-410-tests-of-split-0 run that reproduced
   the hang now passes (406 passed, 4 skipped); full split_0 re-run
   green (759 passed, 6 skipped); `uv run check --full` all checks
   pass.

## FINAL RESULT

- 6114 collected tests (non-slow), 8 parallel splits: **6113 passed /
  skipped cleanly; 1 failure**.
- The 1 failure was an order-independent, environment-dependent hang:
  a **test bug** (SIGINT test assumed the pytest process never inherits
  SIGINT=SIG_IGN from a backgrounding shell). Fixed in the test.
- No project-code bugs found this session (Session 1's TOCTOU
  persistence bug was the only project bug, already fixed).

______________________________________________________________________

# PROGRESS — Run all tests in parallel, diagnose & fix failures — Session 1

## Task

Run all 6113 tests split by test-method count into (cores − 2 = 8)
parallel splits via `run_parallel`, report causes of failing tests,
classify each failure as a project bug vs a test bug, and fix
accordingly.

## What was done

1. Collected 6113 tests (`uv run pytest --collect-only -q`; addopts add
   `-m 'not slow'`, 73 slow tests deselected); split node IDs
   round-robin into 8 files (~764 each); ran all 8 splits concurrently
   with `run_parallel`, each worker running
   `uv run pytest --no-cov -q -p no:cacheprovider @./tmp/split_N.txt`.
1. Results: splits 0, 2–7 ALL PASSED. Split 1: 5 FAILED, all with
   `sqlite3.OperationalError: disk I/O error` raised from
   `src/kiss/agents/sorcar/persistence.py` against the shared
   per-process test DB:
   - test_bughunt_srv2_stale_subscriptions.py::TestReplayOtherChatDropsOldStream::test_replay_back_to_running_chat_restores_stream
   - test_non_git_workdir.py::TestNonGitCommandsDoNotCrash::test_resume_session_unknown_chat
   - test_orphan_task_recovery.py::TestOrphanTaskRecovery::test_sweep_ignores_rows_created_after_cutoff
   - test_simplify_server_cmds_regr.py::test_get_input_history_conn_id_stamping
   - test_wave3_runner_merge_bugs.py::test_b2_warm_agent_second_run_still_attributes_own_metrics
     All 5 passed when re-run serially → order/race-dependent poisoning,
     not a deterministic logic bug. One additional flake surfaced during
     subset reruns: test_install_script_npm_ignore_scripts.py::
     test_run_with_heartbeat_double_sigint_aborts ("first SIGINT trap
     never fired" — empty log after the 10 s `_wait_for_log_text`
     default timeout).
1. Root cause of the 5 disk-I/O failures (confirmed with targeted
   SQLite experiments + a full-stack race harness): **PROJECT BUG** — a
   TOCTOU race in `_get_db()` (persistence.py ~line 910). The stale
   side-file cleanup `if not _DB_PATH.exists(): unlink _DB_PATH+'-wal'/'-shm'` re-read the `_DB_PATH` **global** between
   the existence check and the unlink. Tests that redirect `_DB_PATH`
   to a scratch dir, delete the scratch DB, and restore the shared path
   (e.g. test_persistence_db_deleted_file.py) let a background thread
   (kiss-event-writer / orphan-sweep) observe "missing" for the SCRATCH
   file while computing unlink targets from the freshly RESTORED shared
   path — unlinking the live shared DB's `-shm`. SQLite then fails
   every NEW connection to that DB with a permanent `disk I/O error`
   while any old connection keeps the unlinked `-shm` mapped, poisoning
   later vscode-server tests in the same process.
1. Fix (project code): in `_get_db()` both the existence check and the
   unlink targets are now derived from the single `current_path`
   snapshot (`os.path.exists(current_path)` /
   `os.unlink(current_path + suffix)` in try/except OSError), never a
   re-read of `_DB_PATH`; detailed comment documents the TOCTOU.
1. Regression test added (e2e, no mocks):
   `src/kiss/tests/agents/sorcar/test_persistence_wal_unlink_toctou.py`
   — hammers the redirect/delete/restore pattern with background
   `_get_db()` threads for 5 s, watches the shared `-shm` inode, then
   health-checks a fresh thread. Verified to FAIL on pre-fix code
   ("shared -shm was unlinked") and PASS with the fix.
1. Sigint flake — TRUE root cause found (the earlier "timeout too
   tight" theory was wrong; the test failed again with a 30 s timeout
   and a completely empty harness log even in a solo run): the failure
   reproduces 100% whenever pytest is launched as a **background job of
   a non-job-control shell** (`sh -c 'pytest … &'` — exactly how
   `run_parallel` workers and `nohup … &` invoke it). POSIX requires
   such a shell to start background jobs with SIGINT set to `SIG_IGN`;
   the ignored disposition is inherited across fork/exec down to the
   harness bash, and bash cannot trap signals ignored on entry
   ("Signals ignored upon entry to the shell cannot be trapped or
   reset") — so install.sh's `trap handle_interrupt INT` silently
   became a no-op and `handle_interrupt` never ran (empty log,
   `sleep 30` unaffected). Confirmed empirically: identical harness
   loop fails 36/36 when backgrounded (`getsignal(SIGINT) == SIG_IGN`), passes 13/13 in the foreground. Classification:
   **TEST BUG** — install.sh is correct in a real terminal (SIGINT
   default); the test assumed the pytest process never inherits
   SIGINT=SIG_IGN. Fix: `_reset_signal_dispositions()` helper passed as
   `preexec_fn=` to both harness `subprocess.Popen` calls in
   test_install_script_npm_ignore_scripts.py, resetting SIGINT/SIGTERM
   to `SIG_DFL` between fork and exec so the harness always gets the
   same signal environment as a real terminal. (The earlier 30 s
   `_wait_for_log_text` timeouts were kept — harmless robustness.)
1. Verification: toctou regression test + test_persistence_db_deleted_file
   (5 passed); the 5 previously failing tests + sigint test file +
   persistence neighborhood (32 passed); full split_1 re-run green
   (758/759, only the sigint test failing pre-final-fix); after the
   preexec_fn fix the two sigint tests pass 5/5 iterations when pytest
   runs as a background job (previously 0/5) and the whole sigint test
   file passes in the foreground (8 passed); `uv run check --full` all
   checks pass.

______________________________________________________________________

# PROGRESS — Resolve stranded-branch merge conflict (kiss/wt-1783912825-c143d801)

## Task

The framework's auto-merge of worktree branch `kiss/wt-1783912825-c143d801`
(which carried the previous merge-conflict-resolution session's 5 commits:
merge 984137bf of the gmail/googlechat KISS_HOME fixes, the Episode 1
PROGRESS restore, marketing mdformat, and the PROGRESS log) reported
"Merge conflict detected. Resolve manually" with `git merge --squash`
instructions. Resolve it and land the branch.

## What was done

1. Investigated: the branch held 6 commits on top of merge-base 886d06b1;
   main had meanwhile advanced by one commit (f56e7a4e, Episode 1
   pre-flight). `git merge-tree` showed exactly ONE conflicted file:
   `marketing/agent-markets-itself/episode-01/LAUNCH_CHECKLIST.md` — main's
   f56e7a4e rewrote its pre-flight section (verified quickstart, launch
   slot, 404 blocker) while the branch's 049df754 mdformatted the older
   version of the same lines.
1. Ran `git merge --no-ff kiss/wt-1783912825-c143d801` (merge commit
   7d7c7fac) and resolved the single conflict by keeping main's newer
   pre-flight CONTENT (verified items, slot, blocker) and applying the
   branch's mdformat indentation style; re-ran `mdformat` on the file.
1. Post-merge review: the ort merge of PROGRESS.md had dropped main's
   "Show HN launch pre-flight" section (from f56e7a4e, absent on the
   branch); restored it verbatim + mdformat (commit 900fc669).
1. `uv run check --full` failed only on the pre-existing unformatted
   `HAND_REWRITE_GUIDE.md` (new in f56e7a4e, never mdformatted);
   formatted it (commit aac21482). Full check now passes.
1. Verified merged-in tests still pass (test_gmail_gchat_isolation.py,
   test_gmail_agent.py, test_tlon_config_isolation.py — 39 passed).
1. Verified the auto-stashes hold nothing unique (every differing blob in
   stash@{0} is the stale dc59fed9 version of PROGRESS.md / docs/api.md /
   index.html.md); left them for the user to drop.
1. Deleted the fully-merged branch `kiss/wt-1783912825-c143d801`
   (tip cb8a0d0b is an ancestor of HEAD).

______________________________________________________________________

# PROGRESS — Resolve stranded-branch merge conflict (kiss/wt-1783911737-192f8ce6) (previous task)

## Task

The framework's auto-merge of worktree branch `kiss/wt-1783911737-192f8ce6`
(which carried commit c8bd1176, the gmail/googlechat KISS_HOME isolation
fixes) reported "Merge conflict detected. Resolve manually" with cherry-pick
instructions. Resolve it and land the branch.

## What was done

1. Investigated: the stranded branch held exactly one commit (c8bd1176) on
   top of dc59fed9; main had advanced to 886d06b1 (Episode 1 marketing
   artifacts). Both sides edited `PROGRESS.md`. `git merge-tree --write-tree`
   showed the merge is now clean (the original conflict came from the
   framework's cherry-pick path against a dirty/stashed main).
1. Merged the branch into the current worktree branch (based on main):
   merge commit 984137bf — brought in c8bd1176 (gmail/googlechat/\_channel
   work_dir KISS_HOME fixes + `test_gmail_gchat_isolation.py` + website
   mdformat fixes) with zero conflicts.
1. Post-merge review of `PROGRESS.md` showed the merged branch's version
   (written before the Episode 1 task) had dropped the Episode 1 section
   that main added; restored it verbatim (commit 2834c7f1) and mdformatted.
1. `uv run check --full` initially failed on 6 pre-existing unformatted
   marketing markdown files from main's 886d06b1; mdformatted them
   (commit 049df754). Full check now passes.
1. Verified merged-in tests: `test_gmail_gchat_isolation.py`,
   `test_gmail_agent.py`, `test_tlon_config_isolation.py` — 39 passed.
1. Verified both auto-stashes (`kiss: auto-stash before merge`) contain no
   unique content: every blob in stash@{0} matches either HEAD or the
   pre-merge commits (its PROGRESS.md/api.md/index.html.md copies are the
   older dc59fed9 versions). The user's uncommitted changes were the same
   c8bd1176 content that is now merged. Stashes left in place for the user
   to drop (`git stash drop`) after confirming.
1. Advised: the stranded branch `kiss/wt-1783911737-192f8ce6` can now be
   deleted (`git branch -D kiss/wt-1783911737-192f8ce6`) since its commit
   is contained in the merge.

______________________________________________________________________

# PROGRESS — Audit third-party agent dirs for KISS_HOME test isolation (previous task)

## Task

Audit the ~20 third-party agent integrations sharing the
`Path.home()`-based directory pattern of `slack_agent.py` and apply the
KISS_HOME/tmp-dir test-isolation fix to prevent cross-process races in
parallel test runs (follow-up to the slack token fix in a4e6c58b).

## Audit results

All 23 module-level `_*_DIR` globals in `src/kiss/agents/third_party_agents/`
were audited:

1. **Already isolated (20 agents, no change needed):** bluebubbles, discord,
   feishu, imessage, irc, line, matrix, mattermost, msteams, nextcloud_talk,
   nostr, phone_control, signal, sms, synology_chat, telegram, tlon, twitch,
   whatsapp, zalo. Each module's `_*_DIR` is used only to construct a
   `ChannelConfig`, whose `.path` property already resolves lazily via
   `_kiss_home()` (honours `$KISS_HOME`, which `src/kiss/tests/conftest.py`
   sets to a fresh temp dir per pytest process).
1. **Gap — gmail_agent.py:** `_GMAIL_DIR = Path.home()/...` was used directly
   by `_token_path()`/`_credentials_path()`, bypassing `KISS_HOME`. Tests
   (`test_gmail_agent.py`, `test_bughunt_gmail_whatsapp.py`) backed up,
   overwrote, and restored the REAL user `token.json`/`credentials.json` —
   the same cross-process race that broke slack, plus a risk of clobbering
   real Gmail credentials on a crash.
1. **Gap — googlechat_agent.py:** same direct-path pattern for
   `token.json`/`credentials.json`/`service_account.json` (latent — no test
   currently writes them).
1. **Gap (minor) — \_channel_agent_utils.py:** `ChannelRunner` and
   `channel_main` defaulted `work_dir` to `Path.home()/".kiss"/"channel_work"`
   instead of `_kiss_home()/"channel_work"`.
1. **Intentionally unchanged:** `slack_sorcar_poller.py` /
   `slack_channel_sorcar_poller.py` `STATE_DIR` constants (live production
   cron state; tests already monkeypatch `mod.LOCK_FILE` etc., and the
   code paths exercised by tests never touch the real state files).

## Fixes applied

1. `gmail_agent.py`: replaced the `_GMAIL_DIR` module global with a lazy
   `_gmail_dir()` helper returning
   `_kiss_home() / "third_party_agents" / "gmail"`; `_token_path()` and
   `_credentials_path()` now call it.
1. `googlechat_agent.py`: same pattern — `_gchat_dir()` helper; the three
   path helpers now resolve lazily.
1. `_channel_agent_utils.py`: both `work_dir` defaults now use
   `_kiss_home() / "channel_work"`.
1. New regression test `src/kiss/tests/agents/channels/test_gmail_gchat_isolation.py`
   (mirrors `test_tlon_config_isolation.py`): proves gmail/googlechat paths
   and the `ChannelRunner` default `work_dir` follow `$KISS_HOME` changes
   made after import, and that a token written under one `KISS_HOME` never
   leaks into another.

## Verification

- New isolation tests: 4 passed.
- Full channels test dir: 620 passed, 32 skipped.
- Real `~/.kiss/third_party_agents/gmail/` untouched (empty before and after).
- `uv run check --full` clean.

______________________________________________________________________

# PROGRESS — Ship llms.txt + pure-Markdown docs on kisssorcar.github.io (previous task)

## Task

Ship **llms.txt** at the kisssorcar.github.io root plus pure-Markdown docs so
LLMs and coding assistants can index and recommend KISS Sorcar, and update the
website.

## Session 1 (research + planning)

1. Read `SORCAR.md`; explored repo. The live site is a separate repo,
   `https://github.com/kisssorcar/kisssorcar.github.io`, mirrored locally at
   `website/kisssorcar.github.io/`. `gh` is authenticated with push permission.
1. Completed the mandatory 10-site web research on the llms.txt convention
   (llmstxt.org spec, Mintlify/Anthropic/Vite practice, `.md` twin pages,
   `llms-full.txt`, `.well-known/` mirror, GitHub Pages static serving).
1. Read source-of-truth content: `README.md`, `API.md`,
   `src/kiss/SAMPLE_TASKS.md`, `src/kiss/INJECTIONS.md`, `src/kiss/TIPS.md`.

## Session 2 (implementation + deploy) — COMPLETE

1. Re-cloned the live site repo to a scratch dir (previous clone was lost with
   the old worktree).
1. Authored new files (all pure Markdown / plain text, no HTML):
   - `llms.txt` — spec-compliant: H1 `# KISS Sorcar`, `>` blockquote summary,
     "Key facts" detail block, then `## Docs` (10 links), `## Papers` (4),
     `## Source & Install` (5), `## Optional` (4) — all `- [name](url): notes`
     lines with absolute URLs. Validated structure with a small Python check.
   - `docs/` — 10 pages: `index.md`, `overview.md` (incl. comparison table +
     citation), `installation.md`, `cli.md` (options + mcp subcommand),
     `api.md` (condensed from API.md), `models.md` (530 models / 9 provider
     categories table), `messaging-agents.md` (23 agents + Govee),
     `sample-tasks.md` (all 12 SAMPLE_TASKS), `prompt-tricks.md` (all 12
     INJECTIONS tricks), `tips.md` (all TIPS content).
   - `index.html.md` — Markdown twin of the homepage.
   - `llms-full.txt` — generated by concatenating the 10 docs pages with
     `<!-- Source: url -->` markers.
   - `.well-known/llms.txt` — copy of `llms.txt`.
   - `robots.txt` (allow all, references llms.txt + sitemap), `sitemap.xml`
     (homepage + all md/txt URLs), `.nojekyll`.
1. Edited `index.html` head:
   `<link rel="alternate" type="text/markdown" href=".../index.html.md">` and
   `<link rel="llms-txt" type="text/plain" href=".../llms.txt">`; footer
   gained `Docs` (docs/index.md) and `llms.txt` links.
1. Verified locally (python3 -m http.server): all 16 URLs returned 200.
1. Committed ("Ship llms.txt + pure-Markdown docs for LLM/coding-assistant
   indexing", b11a7a2) and pushed to `kisssorcar/kisssorcar.github.io` main.
1. Verified LIVE after Pages deploy: 200 for `/llms.txt`, `/llms-full.txt`,
   `/robots.txt`, `/sitemap.xml`, `/index.html.md`, `/.well-known/llms.txt`,
   `/docs/index.md`; rendered llms.txt in browser; homepage contains the two
   llms.txt references.
1. Mirrored everything into the main repo artifact dir
   `website/kisssorcar.github.io/` via rsync (also picked up previously
   missing live assets: og-card.png, sorcar-main.gif, KISS-Sorcar-UI.png,
   swedefend.pdf, cleverest_plus.pdf, .gitignore) and rewrote
   `website/README.md` to document the shipped llms.txt work.
1. Cleaned up `./tmp/site` and research notes; staged website changes in git.

## Possible follow-ups (not required)

- Submit https://kisssorcar.github.io/llms.txt to llms.txt directories
  (llmstxt.site, directory.llmstxt.cloud). — DONE, see next section.

# Task: Submit llms.txt to directories (2026-07)

Submitted https://kisssorcar.github.io/llms.txt to both canonical llms.txt
directories (per https://llmstxt.org/#directories):

1. **llmstxt.site** — submitted via https://llmstxt.site/submit with
   Product Name "KISS Sorcar", Website https://kisssorcar.github.io/,
   llms.txt + llms-full.txt URLs, contact ksen@berkeley.edu, and notes
   (Apache-2.0 framework, `pipx install kiss-agent-framework`, docs at
   /docs/index.md, arXiv 2604.23822). Confirmed via redirect to
   https://llmstxt.site/thankyou. Listing appears after their moderation /
   site refresh.
1. **directory.llmstxt.cloud** — submitted via their Tally form
   (https://tally.so/r/wAydjB): name "KISS Sorcar", llms.txt URL,
   Category "AI", email ksen@berkeley.edu for approval notification,
   adoption note. Confirmed "Form submitted / Thanks for completing this
   form!". Pending curation-team approval; notification will go to
   ksen@berkeley.edu.

Both live URLs (https://kisssorcar.github.io/llms.txt and /llms-full.txt)
were re-verified in the browser before and after submitting.

# Task: Episode 1 of "The Agent That Markets Itself" — Show HN launch post (2026-07)

Drafted and "recorded" (written-serial format) Episode 1: KISS Sorcar wrote
its own Show HN launch post, then the draft was reviewed and refined for the
48-hour launch window.

## Session 1 (research)

Completed the mandatory 10-site web research on Show HN norms 2026:
official Show HN guidelines; dang's tips (item 22336638 — **edited
2026-03-28: HN submission text must be hand-written, no LLM text at all**);
syften.com May-2026 guide (first-comment anatomy, kill-triggers, 9am–12pm ET
weekday); lucasfcosta.com (concrete titles, link the repo, cut 30%);
HN Algolia survey of 223 agent-framework Show HNs (winners: Mastra 442 pts,
AnythingLLM 368, Superset 96/90 comments; anti-example Orcbot 4 pts);
forkoff.xyz agent-native GTM (spec-driven agent workflow); indiehackers KTool
(second-chance pool, reply to everything).

## Session 2 (artifacts) — COMPLETE

Created `marketing/agent-markets-itself/` (git-tracked artifact directory):

- `README.md` — series index + ground rules (always disclose agent work;
  HN text hand-written by the founder; no upvote solicitation).
- `episode-01/SPEC.md` — the 300–800-word brief given to the agent: goal
  (titles + URL + first comment), hard constraints (factual language,
  verifiable numbers, ≤450 words, quickstart + limitations + disclosure),
  3 positive examples (Mastra/Superset/AnythingLLM), 3 disqualifiers,
  and the dang-2026 post-draft rule (agent draft = episode artifact only;
  final HN text must be hand-rewritten by the human).
- `episode-01/SHOW_HN_DRAFT.md` — the agent's refined v2 package:
  5 concrete title candidates (recommended: "Show HN: KISS Sorcar –
  open-source agent framework, ~2,850 LoC core, 530 models"), submission URL
  = the GitHub repo, and a 418-word first comment (intro + credibility line,
  backstory, 5 mechanism-anchored bullets, `pipx install kiss-agent-framework` quickstart, honest limitations incl. single
  maintainer + no benchmark yet, meta disclosure of the agent-drafted
  experiment, links block, specific feedback asks on security model and
  small-core claim), plus 9 annotated v1→v2 edits.
- `episode-01/REVIEW.md` — reproduces rejected draft v1 ("does everything"
  title + buzzword bullets = 2 disqualifiers) and maps all 11 rulings to
  research sources; records the 3 spec deltas (word cap, number-or-mechanism
  rule, specific-ask requirement).
- `episode-01/EPISODE.md` — the episode itself in 5 acts, including the
  twist: while researching, the agent discovered HN's 2026 rule banning
  LLM-written submission text, so the episode's resolution is
  "agent draft = published artifact, human hand-rewrites the HN text,
  the HN comment discloses the experiment" — transparency becomes the hook.
  Includes scorecard and video shot-list note.
- `episode-01/LAUNCH_CHECKLIST.md` — 48-hour runbook: T-7 pre-flight
  (hand-rewrite, README GIF, clean-machine quickstart test, personal HN
  account with profile email), T-0 submission (Tue–Thu 9am–12pm ET, repo
  URL, no upvote solicitation), T+48h rules (reply to every comment
  personally, never AI-written replies, second-chance pool), staggered
  Reddit/newsletter wave after traction, and a metrics table that feeds
  Episode 2.

All artifacts committed to git. Remaining human-only steps: hand-rewrite the
first comment and execute the launch window per the checklist.

# Task: Show HN launch pre-flight — hand-rewrite prep, quickstart test, slot (2026-07-12)

Follow-up to Episode 1 of "The Agent That Markets Itself" (see
marketing/agent-markets-itself/episode-01/). Three deliverables:

1. **Hand-rewrite handled correctly.** Per HN's 2026-03-28 rule, any
   agent-written text is LLM-generated and disallowed as submission text —
   so the agent CANNOT produce the final comment. Instead, created
   `marketing/agent-markets-itself/episode-01/HAND_REWRITE_GUIDE.md`: an
   8-beat content checklist (intro/credibility, backstory, 5
   mechanism-anchored bullets, pipx quickstart, honest limitations,
   meta-disclosure, links, feedback asks) plus a verified-facts table and a
   pre-post self-check — deliberately containing NO prose to copy. The
   founder writes the actual comment by hand from this guide.
1. **Clean-machine quickstart test — PASSED.** Docker as clean machine:
   - `python:3.13-slim`: `pip install pipx && pipx install kiss-agent-framework` → 2026.7.18 installs; `sorcar --help` works;
     **34 seconds end-to-end** (≤5-min target met).
   - `python:3.14-slim`: passes (3.13+ claim holds upward).
   - `python:3.12-slim`: fails gracefully ("No matching distribution
     found") — validates the "Python 3.13+ only" limitation claim.
   - Caveat: pipx PATH warning; `pipx ensurepath` fixes.
1. **Slot picked:** primary **Wed 2026-07-15, 9:30am ET**; fallbacks Thu
   2026-07-16 and Tue 2026-07-21 (Tue 7/14 too soon for human pre-flight).

Also: link pre-flight all 200 (repo, arXiv 2604.23822, docs/index.md,
llms.txt); found one **BLOCKER** — the public repo path
https://github.com/ksenxx/kiss_ai/tree/main/marketing returns 404; the
episode dir must be published publicly (push marketing/ to the public repo
or mirror to kisssorcar.github.io) before T-0, since the comment's meta
paragraph links to it. Updated LAUNCH_CHECKLIST.md (checked off verified
items, added slot + blocker), EPISODE.md (new Act 6: pre-flight
verification + scorecard rows), and the series README (guide + slot).
Remaining human-only steps: hand-rewrite, HN-account karma warm-up, README
hero GIF check, clear the public-repo blocker, submit Wed 9:30am ET.
