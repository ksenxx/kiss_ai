# Progress — Broadcast ask-user prompts across clients

## Task

When the agent asks the user a question, show the prompt on all tabs on all
clients with the same chat id. When the user responds on one tab, close/remove
that prompt window from all tabs on all clients with the same chat id. Reproduce
with an end-to-end test, fix the issue using `claude-opus-4-7`, and review the
full work with `gpt-5.5` (not codex).

## Steps so far

- Read `SORCAR.md` first as required; it is empty.
- Switched to `claude-opus-4-7` for all implementation/test work as requested.
- Started exploring the VS Code/webview and Sorcar ask-user code paths.
- Found that ask prompts are emitted as task events (`type: "askUser"`) and fanned out by `WebPrinter` to task subscribers (tabs viewing the task/chat), but submitting a `userAnswer` only puts the answer into the owner queue and the frontend only clears the modal in the submitting tab. Other subscribed tabs keep a stale modal open.
- Planned changes:
  1. Add an end-to-end backend test proving `askUser` is fanned out to all task/chat subscribers and that answering from one subscriber broadcasts a clear event to all subscribers.
  1. Add a frontend handler for that clear event so every tab with the same chat/task removes the ask-user modal.
  1. Keep existing direct owner-tab answer routing and viewer-tab fallback behavior intact.
- Added `src/kiss/tests/agents/sorcar/test_ask_user_broadcast.py`, an end-to-end backend test that starts `_ask_user_question()` on an agent-like thread, subscribes `owner-tab` and `viewer-tab` to the same task, verifies both receive `askUser`, submits `userAnswer` from the viewer, and expects both tabs to receive `askUserDone`.
- Confirmed the new test fails before the fix: both tabs got `askUser`, the answer reached the waiting agent, but there were no `askUserDone` events for either tab.
- Implemented the backend fix in `commands.py`: after a valid `userAnswer` is enqueued, compute all subscriber tabs sharing the answering tab's task and broadcast `{"type": "askUserDone", "tabId": tab_id}` to each of them.
- Implemented the frontend fix in `media/main.js`: handle `askUserDone` by clearing that tab's pending question/input and hiding the shared modal if that tab is active.
- Updated `types.ts` so `askUser` may carry `tabId` and added the `askUserDone` message type.
- Re-ran impacted tests after the fix: `uv run pytest -q src/kiss/tests/agents/sorcar/test_ask_user_broadcast.py src/kiss/tests/agents/sorcar/test_ask_user_immediate_response.py src/kiss/tests/agents/sorcar/test_vscode_tabs.py` passed with 45 tests.
- Switched to `gpt-5.5` for a full-work review. Review findings:
  - The regression test exercises the actual backend lifecycle: task-local thread state, `_ask_user_question()`, production-equivalent fanout via `MemoryPrinter`, command dispatch through `_handle_command(userAnswer)`, and queue wake-up to the waiting agent. It reproduces the stale-modal problem because pre-fix no `askUserDone` event is emitted.
  - The fix preserves the existing answer-routing behavior: direct owner queue lookup remains first, viewer-tab fallback still resolves through the subscriber graph, and unknown/unrelated tabs still drop safely because no queue is found.
  - Broadcasting `askUserDone` after the queue update is the right protocol-level fix: all clients sharing the task subscription receive an explicit close event, so VS Code webviews and remote browser clients converge even when only one frontend submitted the answer.
  - The frontend change is minimal and stateful: it clears the addressed tab's pending question and input, and only hides the currently mounted modal when the addressed tab is active, so background-tab state is corrected without disrupting the foreground if the event is for a different tab.
  - The message type update in `types.ts` is necessary because VS Code extension code inspects `msg.type === 'askUser'` and forwards the new daemon event through the same typed channel.
  - One concurrency improvement was made during review: compute and broadcast `askUserDone` outside `_state_lock` so transport fanout never runs while holding the registry lock.
