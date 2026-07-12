# PROGRESS — Test Slack agent: post message to #sorcar (current task)

## Task

Test the Slack agent by posting a message to the #sorcar channel in the
Sky Computing workspace using `uv run kiss-slack --workspace learningsystems`.

## Steps completed

1. Verified the stored token still works:
   `SlackChannelBackend('learningsystems').connect()` → True
   ("Authenticated as sorcar2 in Sky Computing").
1. Located the channel: `conversations_list` (paginated over 393 channels)
   found `#sorcar` = `C0AKYSNLB7W` (public, bot already a member). Note:
   `find_channel('sorcar')` missed it because it only scans public channels
   without full pagination past 200-per-page default types filter — the
   channel was found by iterating all pages with
   `types="public_channel,private_channel"`.
1. Ran the agent CLI end-to-end:
   `uv run kiss-slack --workspace learningsystems --no-web --no-worktree -t
   "Post the message 'Hello from KISS Sorcar! ...' to channel C0AKYSNLB7W"`
   — completed in ~16 s, cost ≈ $0.17. (Earlier runs by name '#sorcar' also
   succeeded; the CLI prints only timing/cost, not the final answer.)
1. Verified via `read_messages('C0AKYSNLB7W')` that the messages were
   posted by the bot (user U0AKVLSTFGD). Deleted 3 duplicate test messages
   from repeated runs, keeping the final one at ts=1783838044.975019.

## TASK COMPLETE

______________________________________________________________________

# PROGRESS — Slack agent authentication for "learningsystems" (previous task)

## Task

Authenticate the user with the "learningsystems" Slack workspace using the
Slack agent (`src/kiss/agents/third_party_agents/slack_agent.py`).

## Steps completed (across 3 sessions)

1. Verified no token existed at
   `~/.kiss/third_party_agents/slack/learningsystems/token.json`.
1. User chose the browser flow; user signed in manually to
   api.slack.com as ksen@cs.berkeley.edu (reCAPTCHA required human help).
1. The signed-in Slack session only had the "Sky Computing" team available
   (no "learningsystems" team). User approved (option 3) proceeding with
   Sky Computing while saving under the workspace key "learningsystems".
1. Create-app-from-manifest modal proved un-automatable (CodeMirror editor
   not exposed as a textbox; modal overlay intercepted clicks) — abandoned.
1. Reused the EXISTING installed app "sorcar" (A0AKSMD7Q9H) in Sky
   Computing: copied the Bot User OAuth Token (xoxb-…) from its
   OAuth & Permissions page. Bot scopes already granted: channels:history,
   channels:join, channels:read, chat:write, chat:write.public,
   groups:history, groups:read, im:history, im:read, im:write,
   mpim:history, mpim:read, users:read — sufficient for the agent.
1. Saved the token via
   `_save_token(token, 'learningsystems')` →
   `~/.kiss/third_party_agents/slack/learningsystems/token.json`
   (mode 600, key "access_token").
1. Verified end-to-end:
   `SlackChannelBackend('learningsystems').connect()` → True,
   connection_info = "Authenticated as sorcar2 in Sky Computing".
1. Cleaned up temp screenshots; closed the browser.

## TASK COMPLETE

______________________________________________________________________

# PROGRESS — recount Core Agents LoC in README.md (previous task)

## Task

Update the "Core Agents # LoC" figure in `./README.md` after recounting,
excluding empty lines, comment-only lines, and docstrings of private methods.

## Steps completed

1. Identified the five core agent files (matching the definition used in
   `papers/kisssorcar/kiss_sorcar.tex` line 144, which counts the KISS
   Agent, Relentless Agent, Sorcar Agent, Chat Sorcar Agent, and
   Worktree Sorcar Agent with the same exclusion rules):
   - `src/kiss/core/kiss_agent.py`
   - `src/kiss/core/relentless_agent.py`
   - `src/kiss/agents/sorcar/sorcar_agent.py`
   - `src/kiss/agents/sorcar/chat_sorcar_agent.py`
   - `src/kiss/agents/sorcar/worktree_sorcar_agent.py`
1. Wrote a temporary AST/tokenize-based counter (`tmp/count_loc.py`)
   that skips blank lines, comment-only lines (via `tokenize.COMMENT`
   where the stripped line starts with `#`; trailing comments still
   count as code), and docstring lines of private functions/methods
   (name starts with `_`, dunders treated as public). Sanity-checked it
   on a small synthetic file.
1. Recount results: kiss_agent 486, relentless_agent 427, sorcar_agent
   687, chat_sorcar_agent 468, worktree_sorcar_agent 787 → **2855
   total**.
1. Updated README.md line 63: `**~2900**` → `**~2850**`.
1. Deleted the temp counter script and sanity file.

---

# PROGRESS — push website update to GitHub Pages (previous task)

## Task

Push the committed light-theme + transparent-logo website changes to the
GitHub Pages remote so the live kisssorcar.github.io reflects them.

## Steps completed

1. Verified local commit `27a1d8b0` (light theme default + transparent
   hero logo) exists in this repo; the website lives in
   `website/kisssorcar.github.io/` as a drop-in artifact — the actual
   live site is a separate repo
   `https://github.com/kisssorcar/kisssorcar.github.io` (per
   `website/README.md`).
1. Cloned the pages repo into `tmp/kisssorcar.github.io`, copied in the
   updated `index.html` (34 changed lines: light-theme fallback, hero
   CSS, aria-labels) and the transparent `assets/KISS-Sorcar.png`
   (4.4 MB → 0.96 MB).
1. Committed as `dc0f3ab` "Default to light theme and blend hero logo
   seamlessly" and pushed to `origin main`
   (`4091aeb..dc0f3ab main -> main`).
1. Validated the LIVE site after deploy: `curl` confirms
   `saved : 'light'` fallback at line 32 and the 958,897-byte PNG is
   served; a real-browser screenshot of https://kisssorcar.github.io/
   shows the light theme loading by default with the logo blending
   seamlessly (no white box) and the toggle reading "Switch to dark
   theme".
1. Deleted the temp clone and screenshot from `tmp/`.

---

# PROGRESS — full-test-suite run + failure triage (previous task)

## Task

Run ALL tests split into (cores-2)=8 parallel workers via run_parallel;
report the cause of every failing test; classify each failure as a
project bug vs a test bug; fix accordingly.

## Steps completed

1. Collected 6077 test ids, split round-robin into 8 files, ran all 8
   splits in parallel via run_parallel.

1. Results: splits 3/4/5/7 fully passed; split 0: 1 fail (PTY/interrupt
   timing, flaky); split 1: 2 fails + 1 segfault; split 2: 1 fail +
   2 segfaults; split 6: 1 segfault + 1 genuine fail.

1. Genuine PROJECT DATA BUG:
   `test_anthropic_fable_sonnet_5_live_capabilities.py::test_context_length_matches_max_input_tokens[claude-fable-5]`
   — MODEL_INFO.json had `context_length=300000` while the live
   Anthropic `/v1/models` endpoint reports `max_input_tokens=1000000`.
   FIXED: `src/kiss/core/models/MODEL_INFO.json` claude-fable-5
   context_length 300000 → 1000000. Re-ran the test file: 12 passed.

1. Recurring SIGSEGV (splits 1/2/3/6) — TEST BUG: `VSCodeServer.__init__`
   spawns daemon thread `orphan-task-sweep` executing SQL on the
   per-thread sqlite connection aliased into `persistence._db_conn`;
   many vscode tests close that connection / rmtree the temp DB in
   `teardown_method` WITHOUT joining the thread → use-after-close in
   `pysqlite_connection_execute` → SIGSEGV. FIXED centrally in
   `src/kiss/tests/agents/vscode/conftest.py` with a
   `pytest_runtest_call` hookwrapper that joins every live
   `orphan-task-sweep` thread after the test body, before teardown:

   ```python
   @pytest.hookimpl(wrapper=True)
   def pytest_runtest_call(item):
       try:
           yield
       finally:
           for thread in threading.enumerate():
               if thread.name == "orphan-task-sweep" and thread.is_alive():
                   thread.join(timeout=30)
   ```

1. Verified: test_replay_event_coalescing.py 5×/5× pass (previously
   segfaulted 2/3); full vscode folder 1398 passed, 15 skipped, 0
   segfaults; the sweep-asynchrony tests
   (test_web_server_startup_orphan_sweep_nonblocking.py,
   test_orphan_task_recovery.py) still pass — the hook runs after
   the test body, so their in-body timing assertions are unaffected.

1. 4 flaky signal/PTY tests (ctrl-c ×2, install-script SIGINT/SIGHUP)
   pass 4/4 in isolation → load-flakiness only, NOT bugs; left as-is.

1. `uv run check --full` — all checks passed. Both fixes committed
   (6c72db78). tmp/ cleaned.

## TASK COMPLETE

______________________________________________________________________

# PROGRESS — third_party_agents via \_CommandsMixin.\_cmd_run (previous task)

## Task

Update all agents in ./src/kiss/agents/third_party_agents/ so that instead of
calling SorcarAgent.run directly, they use `_CommandsMixin._cmd_run()` to
launch the agents as a kiss-web registered agent. Tests must verify a
third-party agent launched this way is visible/interactable via the remote
webview (UDS transport of RemoteAccessServer / WebPrinter fan-out).
Tests first, then implementation. Dev model: claude-fable-5; review model:
gpt-5.5-xhigh.

## Session 2 findings (exploration)

- Direct `.run()` call sites to convert:
  1. `_channel_agent_utils.py` ChannelRunner.\_handle_message (line ~418) —
     creates `SorcarAgent(self._agent_name)` and calls `.run(...)` with
     extra `tools` (reply tool + backend tools).
  1. `_channel_agent_utils.py` channel_main interactive mode (line ~567) —
     `agent = agent_cls(...); agent.run(**run_kwargs)`.
  1. `slack_sorcar_poller.py` `_run_sorcar` (line ~222) —
     ChatSorcarAgent + resume_chat_by_id/new_chat, returns (text, chat_id).
  1. `slack_channel_sorcar_poller.py` `_run_sorcar` (line ~265) —
     WorktreeSorcarAgent, use_worktree from config.
- `_cmd_run(cmd)` (commands.py:228): needs cmd with non-empty `tabId`;
  spawns thread running `_run_task(cmd)`; `_run_task_inner` calls
  `self._get_tab(tab_id)` which lazily allocates `WorktreeSorcarAgent`
  **only when tab.agent is None** — so pre-populating `tab.agent` with the
  third-party agent instance is the documented injection point.
- `_run_task_inner` requires on tab.agent: `_tab_id`, `_task_start_ms`
  (just set), `_chat_id` (only set if tab.chat_id — SorcarAgent lacks it,
  but assignment works fine on any object; reads `tab.agent.chat_id` —
  **SorcarAgent has NO chat_id property** → channel agents extending
  plain SorcarAgent would crash at task_runner line 618.
  Also `_wt_pending` only accessed when use_worktree=True; `_last_task_id`
  read via `tab.agent._last_task_id` in cleanup (line 348, 928) —
  plain SorcarAgent lacks it too. `resume_from_task_id` only when
  opened_task_id (not our path).
- CONCLUSION: channel agents must derive from ChatSorcarAgent (which
  provides chat_id property, \_chat_id, \_last_task_id, \_task_id_lock,
  resume_from_task_id, \_register_running_state) for the \_cmd_run path.
  Plan: change `BaseChannelAgent, SorcarAgent` bases to
  `BaseChannelAgent, ChatSorcarAgent` in all 26 agent modules, and change
  ChannelRunner to use ChatSorcarAgent. But `_wt_pending` (line 348 finally:
  `tab.use_worktree and tab.agent._wt_pending`) — use_worktree False in
  our path so short-circuits. `tab.agent._last_task_id` at 348 requires
  attr — ChatSorcarAgent has it. OK.
- Result retrieval: `_cmd_run` is async (thread). Helper must wait for
  thread completion and read the result. `agent.run` returns YAML but
  \_run_task discards; result summary is broadcast as "result" event and
  persisted. Easiest: capture the printer broadcasts (server.printer is a
  JsonPrinter whose broadcast we can observe via a subclass or by using
  the server's printer recording), or read `task_history` row via
  persistence, or stash agent's returned value. Cleanest: helper waits on
  tab.task_thread join, then reads persisted result via
  kiss.agents.sorcar.persistence for tab.agent.\_last_task_id... but agent
  is disposed in finally (tab.agent=None) — capture `tab.last_task_id`
  (set from agent.\_last_task_id in finally) then load result from DB.
- Extra tools (ChannelRunner reply tool): \_run_task_inner does NOT pass
  `tools=` to agent.run. Channel agents get tools via \_get_tools()
  override. For ChannelRunner's per-message reply tool: stash extra tools
  on the agent instance (e.g. `agent._extra_run_tools`) and have a
  \_get_tools() override merge them — add support in BaseChannelAgent or
  a small wrapper agent class in \_channel_agent_utils.py.
- Remote webview e2e pattern: see
  src/kiss/tests/agents/sorcar/test_cli_daemon_live_stream.py —
  RemoteAccessServer(uds_path=tmp), asyncio loop thread,
  asyncio.start_unix_server(server.\_uds_handler, path=...), open unix
  connection as "webview", send JSON commands, read newline-JSON events.
- Existing test precedent for stubbing agent run without LLM:
  src/kiss/tests/agents/vscode/test_bughunt_server_runner.py — patches
  `SorcarAgent.__mro__[1].run` (RelentlessAgent.run) with a stub returning
  YAML; uses real VSCodeServer + \_cmd_run.

## IMPORTANT constraint discovered

- Existing contract test src/kiss/tests/agents/channels/
  test_channel_agents_no_chat_session.py asserts every channel agent is
  a SorcarAgent subclass but NOT a ChatSorcarAgent subclass (a prior
  deliberate cleanup). So DO NOT change the agent bases. Instead the
  launcher must make \_cmd_run work with a plain SorcarAgent-based agent:
  task_runner accesses that fail on plain SorcarAgent are:
  - line 617-618: `tab.agent._chat_id = tab.chat_id` (setattr, fine) then
    `tab.agent.chat_id` (AttributeError on plain SorcarAgent!) — but only
    read as `tab.agent.chat_id or tab.chat_id`. SorcarAgent has no
    chat_id attr → crash. Wait — setattr of \_chat_id works, but reading
    `.chat_id` property doesn't exist → AttributeError. tab.chat_id is
    always non-empty (set by \_cmd_run) so line 617 runs, 618 reads.
  - line 348/928: `tab.agent._last_task_id` → AttributeError.
    SOLUTION: give the launcher a small adapter: define a launcher-owned
    subclass? NO — simpler: the launcher sets the missing attributes on
    the agent instance before launch (`agent._chat_id`, `agent._last_task_id = None`, `chat_id` is a property on class...). A property can't be set
    per-instance. ALTERNATIVE: task_runner uses getattr defensively?
    Changing task_runner for this is a wiring change: use
    `getattr(tab.agent, "chat_id", "")` at 618 and
    `getattr(tab.agent, "_last_task_id", None)` at 348/928 — minimal,
    behavior-preserving for existing agents. Line 732 `_wt_pending` is
    guarded by use_worktree (False for third-party launch) but 348 is
    `tab.use_worktree and tab.agent._wt_pending` — also short-circuits.
    resume_from_task_id only on opened_task_id path (never for our tabs).
    Also ChatSorcarAgent.run does registration; plain SorcarAgent.run does
    not register in running_agent_states — but the launcher itself
    pre-registers the \_RunningAgentState (that IS the kiss-web
    registration), so the invariant holds for the duration of the task.

## Additional constraints

- task_runner passes kwargs `use_worktree`, `_skip_persistence`,
  `_subscribe_tab_id`, `_on_task_id_allocated` to tab.agent.run — plain
  SorcarAgent.run raises TypeError. Fix: BaseChannelAgent.run(\*\*kwargs)
  shim that pops those 4 kwargs, merges `self._extra_run_tools` into
  `tools`, then super().run(\*\*kwargs). Channel agents' own run() call
  super().run — MRO routes through BaseChannelAgent.run. For
  ChannelRunner (plain SorcarAgent, no channel class), instantiate the
  agent as a minimal `_RunnerChannelAgent(BaseChannelAgent, SorcarAgent)`
  in \_channel_agent_utils.py so the shim applies.
- Pollers use ChatSorcarAgent/WorktreeSorcarAgent — natively compatible.
- Result capture: helper installs no printer patch; instead it records
  events via a per-tab observer... Simplest robust approach: the helper
  wraps server.printer.broadcast? NO. It reads the "result" broadcast?
  Also persisted task result exists only for ChatSorcarAgent path
  (\_skip_persistence=True stores via task_runner.\_save_task_result in
  finally for chat agents with \_last_task_id). For plain SorcarAgent,
  no task row. ROBUST: launcher captures agent.run return value by
  wrapping agent.run in a one-shot recorder before registering:
  `orig = agent.run; def recording_run(**kw): out["result"]=orig(**kw)`
  — that's a closure/monkeypatch. Cleaner: BaseChannelAgent.run shim
  stores `self.last_run_result = result` before returning; pollers'
  ChatSorcarAgent equally: launcher reads
  getattr(agent, "last_run_result", ""). For chat agents add the same
  attribute via the launcher: wrap? For ChatSorcarAgent we can read the
  persisted result by agent.\_last_task_id via DB. Decision:
  launcher waits for thread, then result =
  getattr(agent, "last_run_result", None) OR persisted row via
  tab.last_task_id. BaseChannelAgent.run sets last_run_result.
  For pollers, read persisted result via \_update? — actually simpler:
  pollers keep agent refs; ChatSorcarAgent.run returns YAML which
  task_runner keeps in `agent_returned` local — inaccessible. Use DB:
  persistence.\_load_chat_events? There's task_history.result saved by
  task_runner finally (\_save_task_result). Verify. So for chat agents:
  after join, read row result via sqlite through persistence helper
  (add tiny `_get_task_result(task_id)` helper? \_load_chat_context by
  chat_id returns task+result — use last entry). Launcher:
  if isinstance(agent, ChatSorcarAgent): result from \_load_chat_context
  (last entry result) else getattr(agent, 'last_run_result', "").

## Design (implemented)

New module: src/kiss/agents/third_party_agents/\_kiss_web_launcher.py

- `run_agent_via_kiss_web(agent, prompt, *, model_name="", work_dir="",  max_budget=None, tools=None, server=None, timeout=None) -> str`
  1. server = server or a process-global lazily-created VSCodeServer()
  1. tab_id = f"tp-{uuid4().hex}"
  1. with server.\_state_lock: create \_RunningAgentState(tab_id,
     server.\_default_model), set state.agent = agent,
     state.auto_commit_mode=False, register.
  1. stash extra tools on agent via agent.\_extra_run_tools (support in
     BaseChannelAgent.\_get_tools; for plain Chat/Worktree agents the
     launcher wraps... — instead: launcher sets agent.\_extra_run_tools
     and SorcarAgent.\_get_tools? No — SorcarAgent builds tools in run()
     via `tools` param. Simplest: BaseChannelAgent.\_get_tools appends
     getattr(self, "\_extra_run_tools", []). For pollers (ChatSorcarAgent)
     there are no extra tools. )
  1. cmd = {"type":"run","tabId":tab_id,"prompt":prompt,"model":...,
     "workDir":..., "useWorktree":bool, "autoCommit":False,
     "useParallel":False}
  1. server.\_cmd_run(cmd); join tab.task_thread; read result from
     persistence via tab.last_task_id.
- Returns the persisted result string (summary) or raises on failure.
  Call sites rewritten to use it. Channel agent base changed
  SorcarAgent → ChatSorcarAgent everywhere in third_party_agents.

## Final concrete plan

Files to change:

1. NEW src/kiss/agents/third_party_agents/\_kiss_web_launcher.py:
   - `default_server()` — lazily-created process-global VSCodeServer.
   - `run_agent_via_kiss_web(agent, prompt_template, *, model_name="",  work_dir="", max_budget=None, tools=None, server=None,  timeout=None) -> str`:
     - stashes tools on agent.\_extra_run_tools, budget on
       agent.\_max_budget_override
     - registers \_RunningAgentState with agent pre-attached
       (auto_commit_mode=False, use_worktree False via cmd)
     - calls server.\_cmd_run({"type":"run","tabId":..., "prompt":...,
       "model":model_name, "workDir":work_dir, "useWorktree":False,
       "autoCommit":False, "useParallel":False})
     - joins tab.task_thread (timeout), marks tab.frontend_closed=True
       so \_dispose_if_closed cleans registry after run
     - returns getattr(agent, "last_run_result", "") — set by
       BaseChannelAgent.run shim AND by chat agents? ChatSorcarAgent has
       no shim... For pollers use `_LauncherResultMixin`? Simpler: the
       launcher wraps nothing; BaseChannelAgent.run sets last_run_result;
       for ChatSorcarAgent-based pollers we define tiny subclasses in the
       launcher module: `class KissWebChatSorcarAgent(ChatSorcarAgent)`
       and `class KissWebWorktreeSorcarAgent(WorktreeSorcarAgent)` whose
       run() records last_run_result then delegates. Pollers instantiate
       those.
1. task_runner.py: 3 defensive getattr changes (lines 348, 618, 928)
   plus max_budget override read:
   `_agent_budget = getattr(tab.agent, "_max_budget_override", None)`
   → pass `max_budget=_agent_budget or _cfg_budget`.
1. \_channel_agent_utils.py:
   - BaseChannelAgent.run shim (pops \_cmd_run-only kwargs, merges
     \_extra_run_tools, applies \_max_budget_override? budget comes via
     task_runner change; shim just records last_run_result).
   - ChannelRunner.\_handle_message → build `_RunnerChannelAgent`
     (module-level class BaseChannelAgent+SorcarAgent) and call
     run_agent_via_kiss_web(...) with tools.
   - channel_main interactive mode → run_agent_via_kiss_web(agent,
     \*\*mapped kwargs) — map run_kwargs (prompt_template, model_name,
     max_budget, work_dir) into launcher params.
1. slack_sorcar_poller.py `_run_sorcar` → KissWebChatSorcarAgent +
   run_agent_via_kiss_web; keep resume_chat_by_id/new_chat; return
   (\_markdown_to_mrkdwn(\_extract_summary(result)), agent.chat_id).
1. slack_channel_sorcar_poller.py similarly with
   KissWebWorktreeSorcarAgent; use_worktree from config → cmd
   useWorktree flag param in launcher (add use_worktree param).
1. NEW tests src/kiss/tests/agents/channels/test_kiss_web_launch.py
   (launcher unit-e2e with stubbed RelentlessAgent.run) and
   test_kiss_web_remote_webview.py (RemoteAccessServer over temp UDS,
   launch SlackAgent via launcher, assert webview receives events and
   appendUserMessage interaction works).

## Remote-webview test wiring notes

- RemoteAccessServer(uds_path=tmp) + asyncio loop thread +
  asyncio.start_unix_server(server.\_uds_handler, path=sock).
  server.\_printer.\_loop = loop. Open unix conn as webview; events
  arrive as newline-JSON. Events with explicit tabId are sent verbatim
  to all clients; task events are stamped per subscribed tab.
  Launcher's tab_id is pre-known so webview just filters on it.
  Interaction: send {"type":"appendUserMessage","tabId":...,"text":...}
  through the UDS connection (dispatched to VSCodeServer.\_handle_command)
  and assert the agent's pre-step drain would see it — with a stubbed
  RelentlessAgent.run we assert pending_user_messages got the text and
  the "prompt" echo event reached the webview.
- Use server=remote.\_vscode_server in launcher so events flow through
  the WebPrinter.

## Env notes

- get_available_models() returns 488 models in dev env → the
  no-model guard in \_run_task_inner passes; cmd "model": "" then
  model = tab.selected_model (default) — fine with stubbed
  RelentlessAgent.run (never hits API).
- Test stub precedent: patch `SorcarAgent.__mro__[1].run`
  (RelentlessAgent.run) as in test_bughunt_server_runner.py.
- conftest sets KISS_HOME to a test dir (isolated sorcar.db).

## Next session concrete TODO (in order)

1. WRITE TESTS FIRST — src/kiss/tests/agents/channels/test_kiss_web_launch.py:
   - stub `SorcarAgent.__mro__[1].run` (RelentlessAgent.run) to return
     "success: true\\nsummary: stub done\\n" (capture kwargs).
   - test_launch_registers_running_state: run_agent_via_kiss_web with a
     SlackAgent; during the (blocked via event) stub run assert
     tab_id in \_RunningAgentState.running_agent_states and
     state.agent is the SlackAgent; after return assert registry entry
     disposed (frontend_closed path) and result == "success..." YAML/
     summary per launcher contract (decide: return raw last_run_result).
   - test_channel_runner_handles_message_via_cmd_run: fake backend,
     ChannelRunner.run_once → assert stub got the reply tool merged and
     backend.send_message got the summary; assert \_cmd_run path used
     (e.g. registry observed during run / thread name, capture that
     stub ran on non-main thread).
   - test_channel_main_interactive_via_cmd_run (argv -t task).
   - test_poller_run_sorcar tests for both pollers (chat id reuse).
   - branch coverage: timeout param, tools=None vs list, model_name
     given/empty, work_dir given/empty, server passed/None, agent
     raising (result failure summary), max_budget override.
1. test_kiss_web_remote_webview.py — per "Remote-webview test wiring
   notes" above; assert webview receives clear/status/prompt events for
   the launcher tab and appendUserMessage lands in
   tab.pending_user_messages + echo "prompt" event.
1. Implement per "Final concrete plan" (launcher module, task_runner
   getattr fixes + budget override, BaseChannelAgent.run shim +
   \_RunnerChannelAgent, channel_main, 2 pollers).
1. Run new tests + existing channel/vscode test suites in parallel
   splits; fix.
1. set_model("gpt-5.5-xhigh") review pass over the diff for missed
   wirings/bugs; fix findings; rerun tests.
1. uv run check --full; clean ./tmp; final PROGRESS.md update; finish.

## Status

- [x] Exploration + design COMPLETE (sessions 1-3)
- [x] PROGRESS.md created + committed
- [x] Tests written first (test_kiss_web_launch.py 24 tests,
  test_kiss_web_remote_webview.py 1 e2e test) — ALL PASSING
- [x] Implementation (launcher, task_runner getattr fixes + budget
  override, BaseChannelAgent.run shim, ChannelRunner +
  channel_main + both slack pollers via run_agent_via_kiss_web)
- [x] Remote webview e2e test passing (open via clear/status/prompt
  events; interact via appendUserMessage → pending_user_messages
  \+ prompt echo)
- [x] Chat-id preservation fix: launcher seeds state.chat_id from the
  agent so resume_chat_by_id survives \_cmd_run's uuid minting
- [x] Contract test rewritten with AST (docstring examples are not
  call sites); zero direct agent.run() launches remain
- [x] Regression suites: channels 607 passed; vscode + sorcar suites
  run in parallel splits — all failures reproduced on baseline
  (SIGINT/SIGHUP signal tests flaky under parallel load; orphan
  sweep sqlite segfault in test_subagent_history_click.py
  pre-existing on main, reproduced with changes stashed)
- [x] gpt-5.5-xhigh review pass (session 6) — findings fixed:
  per-run overrides for model_config/web_tools/is_parallel wired
  through launcher → task_runner; \_cmd_run startup-failure
  disposal; failure-YAML recording in KissWeb\* wrappers and
  BaseChannelAgent.run; defensive server.\_live_task_id; override
  leak-across-reuse fix; 8 new tests covering all findings
- [x] uv run check --full FULLY PASSING (session 7: fixed pyright
  possibly-unbound in webview test, mdformat on PROGRESS.md)
- [x] 100% branch coverage of \_kiss_web_launcher.py (session 8:
  added KeyboardInterrupt worktree-wrapper test for the
  except-BaseException branch — WorktreeSorcarAgent's own
  `except Exception` fallback swallows RuntimeError, so only a
  BaseException reaches the wrapper's except; replaced the dead
  `if thread is not None` branch with `assert` since \_cmd_run
  guarantees task_thread on the non-raise path)
- [x] Final verification (session 8): launcher module coverage
  100% (76 stmts, 4 branches, 0 missed); targeted regression
  suite 97 passed; uv run check --full all checks passed

## TASK COMPLETE

## Session 5 notes

- Segfault forensics: `orphan-task-sweep` thread executes sqlite
  queries in `_log_orphaned_task_forensics` while test teardown
  rmtree's KISS_HOME (deleting the live db) → pre-existing crash on
  main, NOT caused by this diff (verified via git stash A/B run of
  test_subagent_history_click.py — crashes identically both ways).
- Flaky-under-load: test_install_script\_*, test_bughunt9_c_sigterm*,
  test_bughunt4_interrupt_lock, test_bughunt_cli ctrl-c tests — all
  pass in foreground runs with the diff applied.
