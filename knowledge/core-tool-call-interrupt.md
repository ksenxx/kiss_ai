---
title: Per-tool-call interrupt (panel Stop button) - cooperative event then async
  injection
uuid: 71f28aa8-3545-4844-b704-b08a7e09e0ee
summary: 'Per-tool-call Stop in tool_interrupt.py: token registry, cooperative Event
  and raise_if_interrupted, forced PyThreadState_SetAsyncExc after 1 s, closing/injecting
  flags.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Per-tool-call interrupt

The Stop button on each tool panel stops **one** tool call. The task keeps going, and the model sees
`USER_INTERRUPTED_MESSAGE = "User interrupted the tool call."` as that tool's result.

## Flow
1. UI to server: the `interruptTool` command (`tabId`, `toolName`, `callId`; handled by `_cmd_interrupt_tool`
   in `server/commands.py`) calls `task_runner._interrupt_tool_call(tab, tool_name, call_id)`, which resolves the tab's task thread and calls
   `tool_interrupt.interrupt_tool_call(thread_ident, tool_name, call_id)`.
2. `interrupt_tool_call` (under `_INJECT_LOCK`) finds `_RUNNING[thread_ident]`. It refuses when there is no
   call, the call is `closing`, or the name or `call_id` does not match. The `call_id` check means a late
   click can never hit the next call. It then sets `interrupted = True` and `event.set()` (the
   **cooperative** signal) and starts the daemon watchdog `tool-interrupt-<id>`. A second click returns True and does not start a second watchdog.
3. Cooperative tools (the Bash output loop, the `ask_user_question` wait, the `run_parallel` wait, the
   `run_agent` client loop) watch `current_tool_interrupt_event()` and call `raise_if_interrupted()` at a safe point.
4. **Forced**: after `_INJECT_AFTER_SECONDS = 1.0`, if the call is still running, `_inject_unless_closing` raises
   `ToolCallInterrupted` in that thread with `PyThreadState_SetAsyncExc`. CPython delivers it at the next
   bytecode boundary, **never inside a blocking C call**. It refuses when the thread is dead, because thread idents are reused.
5. `KISSAgent._execute_tool` catches `ToolCallInterrupted` and returns the message. If the interrupt landed
   while a `KeyboardInterrupt` or `BudgetExceededError` was already unwinding, it re-raises that one instead.

## Invariants that make async injection safe
- `ToolCallInterrupted` subclasses `BaseException`, so tool code that catches `except Exception` cannot swallow it.
- **The tool thread never takes a lock** in this module. Registry updates are single dict stores and pops,
  which are atomic under the GIL. An exception injected just after `lock.__enter__` would leak the lock.
  Only the interrupting side holds `_INJECT_LOCK`.
- `new_tool_call` creates the token **before** it can be interrupted. `register_tool_call` runs inside the
  `try`, so `finally` always has a token to unregister.
- `closing` is set as soon as the tool returns, and after that nothing is injected. The `injecting` flag
  brackets the injector's check and verdict, and `unregister_tool_call` spins (up to 1 s) until it clears.
- `end_tool_call` handles an injection that raced the tool's return: `drain_pending_interrupt` busy-waits
  up to 1 s for the pending exception to land, and raises it explicitly otherwise. This stops it from
  surfacing later in the agent loop, attributed to the wrong place.
- A Stop that was only signalled cooperatively, where the tool returned within the grace period, keeps the tool's real result.
- `finish` is never registered, so it cannot be interrupted.

## Limitations
- A tool blocked in C (for example a socket read without a watchdog) is not interrupted until it returns to Python.
- If the watchdog thread cannot start, only the cooperative signal is used.

## Sources
- `src/kiss/core/tool_interrupt.py` (`interrupt_tool_call`, `ToolCallToken`, `new_tool_call`, `register_tool_call`, `end_tool_call`, `unregister_tool_call`, `raise_if_interrupted`, `drain_pending_interrupt`, `inject_async_exception`)
- `src/kiss/core/kiss_agent.py` (`KISSAgent._execute_tool`)
- `src/kiss/server/task_runner.py` (`_interrupt_tool_call`), `src/kiss/server/commands.py`
- `src/kiss/tests/core/test_tool_interrupt.py`; feature commit e01e0ccd9
