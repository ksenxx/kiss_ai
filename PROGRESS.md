# Task: Verify & fix — "Running <duration>" tab status overridden by newer runs in another window

Requirement: EVERY tab must show running time as (now - task start) while
running, and (endTs - startTs) once the task has ended. User reports the
Running timer is still clobbered by newer runs in another window.

## Investigation so far (session 1)

Relevant file: `src/kiss/agents/vscode/media/main.js` (6288 lines), webview UI.

Key state:

- GLOBAL `let t0 = null;` (line ~1128) — timer anchor for the ACTIVE tab.
- GLOBAL `let endTs = 0;` (line ~2430) — recorded end ts, GLOBAL ONLY, **no per-tab endTs field** in makeTab (line ~160-200: tab has `t0: null` but NO `endTs`).
- `_renderTimerTick()` (~2431): uses global t0/endTs; if `endTs>0 && t0 && Date.now()>=endTs` shows `Done (Xm Ys)` else `Running Xm Ys`.
- `startTimer()` (~2452): `if (!t0) t0 = Date.now();` then interval 1s.
- `saveCurrentTab()` (~295): `tab.t0 = t0;` — saves t0 per tab. Does NOT save endTs.
- `restoreTab()` (~381): `t0 = tab.t0 || null;` — restores t0. Does NOT restore endTs.
- `switchToTab` (~552-560) & closeTab (~605-615): after restore, if !tab.isRunning → `t0 = null; stopTimer();` — so a DONE tab loses its t0 ⇒ cannot compute Done duration; relies on saved statusTextContent string instead.
- `status` event handler (~2524-2560): sets `evTab.t0 = ev.startTs` per-tab; anchors GLOBAL t0/endTs only when `ev.tabId === undefined || ev.tabId === activeTabId`. **BUG SUSPECT:** events with `ev.tabId === undefined` (e.g. from another window / non-tab-stamped daemon broadcasts) clobber the active tab's t0 + setRunningState. A newer run in ANOTHER WINDOW whose status events lack tabId (or whose tabId numerically equals this window's activeTabId — tab ids may collide across windows!) overwrites t0.
- `task_events` replay (~2885-2895): sets global t0/endTs from `extra.startTs`/`extra.endTs`.
- `task_done` handler (~3234-3241, also ~3557-3567): computes doneT0 per resolved tab `rt.t0`; uses `ev.endTs - ev.startTs` when both >0; clears `doneTab.t0 = null`, global `t0 = null; endTs = 0`.
- `setRunningState` (~3519): starts/stops timer based on global isRunning.

Tab id generation: `genTabId()` — need to check whether ids are unique
across windows (suspect collision: tab ids like 1,2,3 per window, daemon
broadcasts to ALL clients; other window's tabId===this activeTabId ⇒ clobber).

Tests exist: `src/kiss/agents/vscode/test/bughunt2_status_timer.test.js` — read
it; also check how SorcarSidebarView.ts / server daemon stamp `startTs`,
`endTs`, `tabId` on status events (src/kiss/agents/vscode/src/types.ts:150
mentions startTs anchoring).

## Session 2 findings

- Tab ids are UUIDv4 (`genTabId()` main.js ~147) — no cross-window collision.
- All daemon `status` broadcasts carry tabId (+startTs when running=true):
  task_runner.py:149 (run start), :217 (run end, no startTs), :995 (viewer
  tabs of same chat, with startTs).
- SorcarSidebarView.ts:800 `_startTask` sends webview `{type:'status', running:true, tabId}` WITHOUT startTs → webview startTimer() falls back to
  `t0 = Date.now()` (acceptable: local optimistic start).
- Webview `status` handler is gated by `ev.tabId === undefined || ev.tabId === activeTabId` for the GLOBAL t0/endTs anchor and setRunningState.
  Foreign-window tabIds resolve to no local tab → no clobber there.
- REMAINING SUSPECTS for the cross-window clobber (NOT yet checked):
  1. `task_events` handler ~line 2880-2895 of main.js sets GLOBAL t0/endTs
     from `extra.startTs/endTs` — check whether gated by activeTabId.
  1. `result` / `task_done` handler ~3234, ~3557-3567 clears global
     t0/endTs — check gating by tabId (a done event from ANOTHER window's
     run may stop/zero the active tab's timer; line ~3416
     `setRunningState(false)` mention).
  1. Other handlers calling setRunningState / stopTimer / statusText
     writes without tabId gating (search `setRunningState(` and
     `statusText.textContent` call sites).
  1. server.py resume path line ~1159 sends status with startTs to
     viewers — fine.

## Session 3 findings

- ALL daemon status broadcasts are tab-stamped (server.py:1159 resume w/
  startTs; task_runner.py:149 start w/ startTs, :217 end, :995 viewers w/
  startTs; web_server.py:3794). No un-stamped status event found.
- `setReady(label, tabId)` (main.js ~3550): for bg tab nulls `doneTab.t0`
  but DOES NOT update `doneTab.statusTextContent` to the Done(…) label →
  switching to that tab later: restoreTab paints stale "Running Xs" string,
  then `setRunningState(false)` replaces it with bare "Done" (NO duration).
  **Requirement violated: done tab must show endTs-startTs.**
- `setRunningState(false)` (~3533-3537): blanket `statusText.startsWith ('Running') → 'Done'` destroys duration info.
- NO per-tab `endTs` field exists (only global `endTs`); makeTab lacks it;
  saveCurrentTab/restoreTab don't save/restore it.
- `task_events` bg-tab branch (~2820-2860, before "Active tab:" comment)
  does NOT capture extra.startTs/endTs into the bg tab (teTab) — bg tab
  replay of finished task has no t0/endTs.
- `task_done` handler (~3234): computes duration from ev.endTs-ev.startTs
  (good) and calls setReady(label, ev.tabId) — but label only reaches DOM
  when tabId===activeTabId; bg tab's saved label not set (see above).
- switchToTab/closeTab/restore (555-560, 610-615, 752-756): `if (!tab.isRunning) { t0 = null; stopTimer(); }` — then statusText stays as
  whatever restoreTab painted (stale string) but setRunningState(false)
  morphs "Running…"→"Done" w/o duration.

## ROOT CAUSES (to fix)

1. No per-tab endTs; global-only endTs and reliance on saved status STRING.
1. setReady doesn't persist the Done(…) label/t0/endTs onto the bg tab.
1. setRunningState(false) rewrites "Running…"→"Done" losing duration.
1. bg task_events replay doesn't store startTs/endTs per tab.
1. (cross-window override) statusTextContent of saved tab is repainted then
   morphed; suspect: events for the SAME chat viewed in 2 windows use
   viewer-tab broadcast (task_runner.py:995) — that's intended. The user's
   override: when tab not active, its "Running …" header derives from
   per-tab snapshot; verify in fix that ALL paint paths recompute from
   tab.t0/tab.endTs instead of the snapshot string.

## FIX DESIGN

- makeTab: add `endTs: 0`.
- saveCurrentTab: `tab.t0 = t0; tab.endTs = endTs;` restoreTab: restore both.
- status handler: `if (evTab) { evTab.t0 = ev.startTs; evTab.endTs = 0; }`.
- task_done/setReady: keep `doneTab.t0`; set `doneTab.endTs = ev.endTs` (or
  Date.now()); set `doneTab.statusTextContent = label`; do NOT null t0.
- switchToTab/closeTab: for done tab with t0&&endTs render
  `Done (Xm Ys)` from timestamps; don't blank t0.
- setRunningState(false): only fallback to 'Done' when no t0/endTs known.
- task_events bg branch: store extra.startTs/endTs onto teTab.
- \_renderTimerTick already handles endTs>0 → Done label; make it derive
  duration purely as endTs - t0 (it does).
- Test: extend/add node test alongside test/bughunt2_status_timer.test.js.

## Session 4 findings

- Existing test `test/bughunt2_status_timer.test.js` PASSES (2/2): foreign
  unknown-tabId status + local bg-tab status no longer clobber active timer.
  So the previously-fixed paths are fine; the user's report must come from
  a different path.
- `task_events` case (main.js:2811): `const teTabId = ev.tabId || activeTabId;` — bg branch (teTabId!==activeTabId) ignores extra.startTs/
  endTs for teTab (gap); unknown foreign tab → break (safe). Active branch
  sets GLOBAL t0/endTs from extra (only when teTabId===activeTabId — safe,
  BUT note `ev.tabId || activeTabId` means an event with NO tabId is
  treated as active-tab — check daemon emits of task_events: all stamped).
- NEW HYPOTHESIS for cross-window clobber: if two windows restore the SAME
  persisted tab UUIDs (same workspace opened twice / window reload restores
  tabs from daemon-side or globalState persistence), then window B's new
  run broadcasts status running:true tabId==window A's activeTabId →
  re-anchors A's t0 to the newer startTs. MUST check: persistTabState()
  (vscode.setState — per-webview) AND any daemon-side tab restore
  (SorcarSidebarView resumeSession, server.py listTabs/restoreTabs) that
  could duplicate tab ids across windows.

## Session 5 findings

- vscode.setState/getState is per-webview → two windows don't share tab
  UUIDs via persistence; tab restore (main.js ~821-855) restores st.chatId
  as tab.id only within the same webview.
- task_runner.py: `task_end_event` built at :552 (`{"type": "task_done"}`),
  :571 task_error, :637 task_stopped, :937 task_interrupted. NEED to check
  how task_end_event is BROADCAST (~line 640-760): does it carry tabId,
  startTs, endTs always? If broadcast without tabId → main.js
  `setReady(label, undefined)` & task_done handler treat as ACTIVE tab →
  cross-window clobber ("Done(...)" overriding "Running …"), and
  `markTabDone(undefined)` → marks ACTIVE tab. **PRIME SUSPECT.**
- Next: sed -n '630,770p' task_runner.py to see broadcast of
  task_end_event; check it includes tabId/startTs/endTs.

## Session 6 — CONCLUSION & IMPLEMENTATION PLAN

Verdict on "is the observation still true": the originally-reported clobber
(foreign-tab status re-anchoring the active timer) is FIXED — existing test
test/bughunt2_status_timer.test.js passes. task_done/status/task_events are
all tabId+startTs/endTs-stamped daemon-side (task_runner.py:853-857,
:149,:217,:995; server.py:1159). Remaining REAL violations of the
requirement (every tab must show Running=now-start, Done=endTs-startTs):

V1. Done-in-background tab: setReady nulls doneTab.t0, never stores the
Done(…) label or endTs per-tab → switching to it shows bare 'Done'
(setRunningState(false) morphs 'Running…'→'Done', duration LOST).
V2. No per-tab endTs at all (makeTab/save/restore lack it).
V3. bg task_events replay ignores extra.startTs/endTs for teTab.
V4. setRunningState(false) blanket 'Running…'→'Done' rewrite loses duration.

EDITS planned in media/main.js (all designed, not yet applied):

1. makeTab: add `endTs: 0`.
1. saveCurrentTab: `tab.endTs = endTs;` next to `tab.t0 = t0;` (line ~295).
1. restoreTab (~381): `t0 = tab.t0 || null; endTs = tab.endTs || 0;`
1. status handler (~2536): `if (evTab) { evTab.t0 = ev.startTs; evTab.endTs = 0; }`.
1. task_done handler (~3234): stamp resolved tab `rt.t0=ev.startTs, rt.endTs=ev.endTs` (when >0); pass through so setReady can persist label.
1. setReady(label, tabId): keep doneTab.t0 (do NOT null); set
   doneTab.endTs (param or Date.now()); set doneTab.statusTextContent=label,
   statusTextColor='var(--green)'.
1. switchToTab (~555)/closeTab (~610)/other restore (~752): remove `t0 = null`; keep stopTimer/removeSpinner; rely on restored t0/endTs +
   setRunningState(false) new logic to paint 'Done (Xm Ys)'.
1. setRunningState(false) (~3535): if text starts with 'Running': use
   `endTs>0 && t0` → 'Done (Xm Ys)' else 'Done'.
1. task_events bg branch (~2839 extra parse): also teTab.t0=startTs,
   teTab.endTs=endTs from bgExtra.
   NEW TEST: src/kiss/agents/vscode/test/tab_timer_per_tab.test.js (jsdom,
   mirrors bughunt2_status_timer.test.js harness):
   t1 foreign newer-run status doesn't clobber (regression)
   t2 bg-tab task_done(startTs,endTs) → switch → 'Done (Xm Ys)' from ts diff
   t3 active task_done → 'Done (…)' = endTs-startTs
   t4 switch away & back to done tab → label persists with duration
   t5 bg task_events replay w/ extra{startTs,endTs} → switch → 'Done (…)'
   t6 running tab switch away/back → 'Running ~Ns' still anchored to startTs
   Run: node test/tab_timer_per_tab.test.js (jsdom available in
   src/kiss/agents/vscode/node_modules). Then run existing bughunt2 tests +
   `uv run check --full` (only .js changed → eslint via repo config?
   package.json has eslint.config.mjs; run `npx eslint media/main.js test/...`).

## Session 7 status

- WROTE new test src/kiss/agents/vscode/test/tab_timer_per_tab.test.js
  (6 tests, harness mirrors bughunt2_status_timer.test.js).
- Run result: t1 (foreign newer-run no clobber) PASSES, t2 (active
  task_done endTs-startTs) PASSES. t3 FAILS in the HARNESS: clickTab can't
  find tab titled "background work" — openSubagentTab tab title/DOM
  structure differs. Need to inspect 'openSubagentTab' handler in main.js
  (~line 3340-3400) for how tab title is set (probably truncated
  description, isSubagentTab, and maybe renderTabBar uses different class
  names like .chat-tab-title — verify selector) and how switching works
  (maybe openSubagentTab makes the new tab ACTIVE immediately; my test
  assumed not).
- After fixing harness, expect t3/t5 to FAIL for real (missing per-tab
  endTs etc.), then apply the 9 planned edits from Session 6 to
  media/main.js, re-run both test files.

## Session 8 status

- renderTabBar uses `.chat-tab` with `el.dataset.tabId` + label span class
  `.chat-tab-label`; click on el → switchToTab(tab.id).
- Fixed test harness: clickTab(win, tabId) now selects
  `.chat-tab[data-tab-id="..."]` (DONE).
- REMAINING test edits: replace clickTab title args with tab ids:
  t3: clickTab(win,'bg-tab'); t4: clickTab(win,'sub-x') then
  clickTab(win, activeId); t5: clickTab(win,'hist-tab');
  t6: clickTab(win,'sub-y') then clickTab(win, activeId).
- openSubagentTab handler (main.js:3308): no parent_tab_id in event → OK
  (guard only fires when parent_tab_id set but unknown). Title =
  description.substring(0,40). It does NOT auto-switch unless...(check
  needsPlacement code ~3348-3370 if test misbehaves).
- Then run test → fix main.js per Session 6 edit list.

## Session 9 status — main.js edits IN PROGRESS

Pre-fix test run: t1,t2 pass; t3 FAILS ('Ready' instead of 'Done (3m 21s)').
APPLIED so far to media/main.js:

1. makeTab: added `endTs: 0` field (after t0). DONE
1. saveCurrentTab: `tab.endTs = endTs;` after `tab.t0 = t0;` DONE
1. restoreTab: `endTs = tab.endTs || 0;` after `t0 = tab.t0 || null;` DONE
   REMAINING edits (Session 6 list):
1. status handler (~2540): `if (evTab) evTab.t0 = ev.startTs;` → also
   `evTab.endTs = 0;`
1. task_done handler (~3240): before setReady, resolve tab by ev.tabId
   (or activeTabId) and stamp `tab.t0 = ev.startTs; tab.endTs = ev.endTs`
   when both >0.
1. setReady(label, tabId): REMOVE `doneTab.t0 = null;`; ADD
   `doneTab.statusTextContent = label || 'Ready'; doneTab.statusTextColor = 'var(--green)';` (so bg done tabs restore the duration label). Also
   in the active branch KEEP `t0 = null; endTs = 0`? NO — keep t0/endTs
   so saveCurrentTab snapshots them; statusText set to label anyway.
   CAREFUL: next task on same tab must reset (status running handler
   clears endTs and re-anchors t0 — edit 4 covers per-tab; global cleared
   in existing status handler code).
1. switchToTab (~558) & closeTab (~615) & restore-after-history (~757):
   remove `t0 = null;` lines (keep stopTimer/removeSpinner).
1. setRunningState(false) (~3540): 'Running…' → if (endTs>0 && t0)
   'Done (Xm Ys)' computed, else 'Done'.
1. task_events bg branch (~2845 bgExtra parse): teTab.t0 = bgExtra.startTs
   (>0), teTab.endTs = bgExtra.endTs (>0).
   Then: node test/tab_timer_per_tab.test.js AND
   node test/bughunt2_status_timer.test.js AND bughunt2_demo_continue.test.js;
   npx eslint media/main.js test/tab_timer_per_tab.test.js; uv run check --full.

## Session 10 — FIX APPLIED, TESTS GREEN

All 10 planned edits applied to media/main.js:
makeTab endTs:0; saveCurrentTab/restoreTab save+restore endTs; status
handler stamps evTab.t0 + clears evTab.endTs; task_done & task_error/
stopped/interrupted pass ev.startTs/ev.endTs into setReady; setReady now
KEEPS doneTab.t0, sets doneTab.endTs, persists statusTextContent label +
green color per-tab; switchToTab/closeTab no longer null t0;
setRunningState(false) renders doneLabelFor(t0,endTs) instead of bare
'Done'; bg task_events captures bgExtra.startTs/endTs into teTab and
pre-renders Done label; submit path re-anchors t0=Date.now(), endTs=0;
added helper doneLabelFor(startMs, endMs) used by tick/setRunningState/bg.

Tests: NEW test/tab_timer_per_tab.test.js → 6/6 PASS (was failing t3/t5
pre-fix). ALL other webview tests pass (bughunt2_status_timer 2/2,
bughunt2_demo_continue 2/2, bughunt3 3/3, bughunt5 x2, bughunt6,
panelCopy 12, daemonHealth 21, installerPath 7, reloadGuard 13;
bughunt_isNewFile + syncWorkDir needed `npm run compile` (pre-existing
missing out/), now 3/3 and 4/4). `npx eslint media/main.js` CLEAN (ran
--fix for one prettier wrap). Note: eslint on test/\*.test.js reports
pre-existing no-undef console/process errors in ALL existing test files
(config gap, not introduced by this change).

REMAINING: run `uv run check --full` (repo-wide; .js modified so
mandatory), verify it's clean, re-read modified files, finish.

## Planned next steps (session 2+)

1. Read `genTabId()` + tab id collision behavior across windows; read
   daemon/server side (server.py / SorcarSidebarView.ts / AgentClient.ts)
   for how status events get tabId/startTs/endTs and whether events from
   other windows are filtered (look for clientId/windowId).
1. Read test/bughunt2_status_timer.test.js to learn the test harness style.
1. Design fix:
   - Make timer fully PER-TAB: add `endTs: 0` to makeTab; save/restore endTs
     in saveCurrentTab/restoreTab; in `status` handler only anchor through
     evTab (per-tab), never trust bare `ev.tabId === undefined` events for
     re-anchoring while another tab is targeted; ignore status events whose
     tabId resolves to no local tab (already partly done) AND those from
     other windows (need window/client scoping — check if tabIds are
     globally unique e.g. `Date.now()+random`; if not, make genTabId
     collision-resistant or have daemon stamp client/window id and filter).
   - When a tab is done, keep `tab.t0` + `tab.endTs` so a tab switch can
     re-render `Done (Xm Ys)` from timestamps rather than the saved status
     string; do NOT null t0 on done (lines ~3557, ~3562, switchToTab ~557,
     closeTab ~612, restore path ~754).
1. Write failing test first (node test in src/kiss/agents/vscode/test/,
   style mirrors bughunt2_status_timer.test.js) simulating: tab A running
   with startTs S1; status event for another window's run (different/absent
   tabId, startTs S2) arrives → tab A's displayed Running duration must stay
   anchored to S1; also done-tab shows endTs-startTs after switching tabs.
1. Run test suite for vscode tests + `uv run check --full`.
