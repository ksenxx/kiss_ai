# Plan: Implement Gemini Flash 2.0 Features in VS Code Extension

## Overview

The web-based Sorcar uses `gemini-2.0-flash` (constant `FAST_MODEL` in `shared_utils.py`) as a lightweight, fast LLM for three auxiliary features that augment the main agent (which uses Claude or another expensive model for the core task). The VS Code extension already has **partial** implementations of these features but with key gaps. This plan details exactly what exists, what's missing, and how to bring the VS Code extension to parity.

---

## How Gemini Flash 2.0 Is Used in Web Sorcar

All three features use `KISSAgent` with `is_agentic=False` (single prompt→response, no tools, no conversation loop) and `model_name=FAST_MODEL` (`"gemini-2.0-flash"`).

### Feature 1: Ghost Text Autocomplete (`/complete` endpoint)

**Web flow:**
1. User types in the chat input box. After 200ms debounce, JS calls `GET /complete?q=<partial>`.
2. Server first tries **fast local matching** (`_fast_complete`):
   - Prefix-matches against recent task history strings.
   - Prefix-matches the last word against file paths in `file_cache`.
   - If found, clips it via `clip_autocomplete_suggestion()` and returns immediately (no LLM call).
3. If no fast match, server calls **Gemini Flash 2.0** via `KISSAgent.run()`:
   - System prompt: "You are an inline autocomplete engine..."
   - Context injected: past 20 tasks, up to 200 files, **active editor file content** (up to 10KB), and the partial query.
   - Result clipped by `clip_autocomplete_suggestion()` (max 4 words, stops at boundaries, suppresses low-confidence completions).
4. Suggestion returned as `{"suggestion": "..."}`, displayed as ghost text in the input overlay.

**VS Code extension current state:**
- `server.py` `_complete()` method exists and calls `KISSAgent` with `FAST_MODEL` ✅
- But it does **NOT** have the fast local matching (`_fast_complete`) step ❌
- It does **NOT** include the active editor file content in the context ❌
- It only sends 50 files (vs 200 in web) and truncates task_list to 1000 chars ❌
- `main.js` sends `complete` message, receives `ghost` event, renders ghost text ✅
- No client-side ghost cache (web has `ghostCache` to reuse prefixes) ❌

### Feature 2: Follow-up Task Suggestion (`generate_followup_text`)

**Web flow:**
1. After `task_done`, server calls `generate_followup_text(task, result)` in a background thread.
2. This function creates a `KISSAgent` with `FAST_MODEL` and prompts:
   "A developer just completed this task: {task}. Result summary: {result}. Suggest ONE short follow-up task."
3. Result is broadcast as `{"type": "followup_suggestion", "text": "..."}`.
4. JS renders a clickable "Suggested next" bar that pre-fills the input.

**VS Code extension current state:**
- `server.py` `_generate_followup()` calls `generate_followup_text()` from `shared_utils.py` in a background thread ✅
- `main.js` handles `followup_suggestion` event and renders the bar ✅
- `result` passed to `_generate_followup()` is just `prompt[:200]` instead of the actual agent result/summary ❌ (web passes the full result YAML summary)

### Feature 3: Git Commit Message Generation (`_generate_commit_msg`)

**Web flow:**
1. `POST /commit` or `POST /generate-commit-message` calls `_generate_commit_msg(diff_text)`.
2. Uses `KISSAgent` with `FAST_MODEL` to generate conventional commit messages from git diffs.
3. For the SCM integration, the message is written to `pending-scm-message.json` for the VS Code extension to pick up.

**VS Code extension current state:**
- The VS Code extension does **NOT** have commit message generation ❌
- No `/commit` or `/generate-commit-message` equivalent commands ❌
- The `server.py` has no integration with VS Code's SCM panel ❌

---

## Implementation Plan

### Phase 1: Bring Autocomplete to Parity

**File: `src/kiss/agents/vscode/server.py`**

1. **Add `_fast_complete()` method** to `VSCodeServer`:
   ```python
   def _fast_complete(self, raw_query: str, query: str) -> str:
       """Local prefix matching against history and file paths."""
       query_lower = query.lower()
       for entry in _load_history(limit=50):
           task = str(entry.get("task", ""))
           if task.lower().startswith(query_lower) and len(task) > len(query):
               return task[len(query):]
       words = raw_query.split()
       last_word = words[-1] if words else ""
       if last_word and len(last_word) >= 2:
           lw_lower = last_word.lower()
           for path in self._file_cache:
               if path.lower().startswith(lw_lower) and len(path) > len(last_word):
                   return path[len(last_word):]
       return ""
   ```

2. **Update `_complete()` to use fast path first**, and enrich context:
   - Call `_fast_complete()` first; if it produces a non-empty clipped suggestion, return immediately without calling the LLM.
   - Read the active editor file content from the command's `activeFile` (store `self._last_active_file` when a `run` command arrives).
   - Increase file list from 50 to 200.
   - Remove the 1000-char truncation on task_list (or make it 4000+ like web).
   - Add active file content to the prompt context (matching web's prompt exactly).

3. **Store last active file path** — update `_run_task` to store `self._last_active_file = cmd.get("activeFile", "")`.

**File: `src/kiss/agents/vscode/media/main.js`**

4. **Add client-side ghost cache** to avoid redundant LLM calls:
   ```javascript
   var ghostCache = { q: '', s: '' };
   ```
   In `requestGhost()`, before sending the `complete` message:
   - Check if current input starts with `ghostCache.q` and `ghostCache.s` starts with the extra chars typed. If so, update ghost from cache without an LLM call.
   In the `ghost` event handler, save `ghostCache = { q: inp.value, s: suggestion }`.

5. **Don't request ghost when cursor isn't at end** — match web behavior:
   ```javascript
   if (inp.selectionStart < inp.value.length) { clearGhost(); return; }
   ```

6. **Minimum query length check** — skip if less than 2 non-whitespace characters (matching web).

### Phase 2: Fix Follow-up Suggestion Quality

**File: `src/kiss/agents/vscode/server.py`**

1. **Capture actual result summary from agent run** instead of using `prompt[:200]`:
   ```python
   # In _run_task, after agent.run() completes:
   import yaml
   result_summary = ""
   try:
       parsed = yaml.safe_load(result)
       if isinstance(parsed, dict) and "summary" in parsed:
           result_summary = str(parsed["summary"])
       else:
           result_summary = str(result)[:500]
   except Exception:
       result_summary = str(result)[:500]
   ```
   Pass this proper `result_summary` to `_generate_followup(prompt, result_summary)`.

2. The `_generate_followup` method is already correct — it spawns a background thread and calls `generate_followup_text()` from `shared_utils.py` which uses `FAST_MODEL`.

### Phase 3: Add Git Commit Message Generation

**File: `src/kiss/agents/vscode/server.py`**

1. **Add `generateCommitMessage` command handler**:
   ```python
   elif cmd_type == "generateCommitMessage":
       threading.Thread(target=self._generate_commit_message, daemon=True).start()
   ```

2. **Add `_generate_commit_message()` method**:
   ```python
   def _generate_commit_message(self) -> None:
       from kiss.agents.sorcar.code_server import _capture_untracked, _git
       try:
           diff_result = _git(self.work_dir, "diff")
           cached_result = _git(self.work_dir, "diff", "--cached")
           diff_text = (diff_result.stdout + cached_result.stdout).strip()
           untracked = "\n".join(sorted(_capture_untracked(self.work_dir)))
           if not diff_text and not untracked:
               self.printer.broadcast({"type": "commitMessage", "message": "", "error": "No changes detected"})
               return
           context_parts = []
           if diff_text:
               context_parts.append(f"Diff:\n{diff_text[:4000]}")
           if untracked:
               context_parts.append(f"New untracked files:\n{untracked[:500]}")
           # Reuse _generate_commit_msg from sorcar.py or inline it
           agent = KISSAgent("Commit Message Generator")
           raw = agent.run(
               model_name=FAST_MODEL,
               prompt_template=(
                   "Generate a nicely markdown formatted, informative git commit message for "
                   "these changes. Use conventional commit format with a clear subject "
                   "line (type: description) and optionally a body with bullet points "
                   "for multiple changes. Return ONLY the commit message text, no "
                   "quotes or markdown fences.\n\n{context}"
               ),
               arguments={"context": "\n\n".join(context_parts)},
               is_agentic=False,
           )
           msg = raw.strip().strip('"').strip("'")
           self.printer.broadcast({"type": "commitMessage", "message": msg})
       except Exception:
           logger.debug("Commit message generation failed", exc_info=True)
           self.printer.broadcast({"type": "commitMessage", "message": "", "error": "Failed to generate"})
   ```

**File: `src/kiss/agents/vscode/src/types.ts`**

3. **Add new types**:
   - Add `| { type: 'generateCommitMessage' }` to `FromWebviewMessage` (or make it a VS Code command, not a webview message).
   - Add `| { type: 'commitMessage'; message: string; error?: string }` to `ToWebviewMessage`.
   - Add `'generateCommitMessage'` to `AgentCommand.type` union.

**File: `src/kiss/agents/vscode/src/SorcarPanel.ts`**

4. **Add `generateCommitMessage` handler** in `_handleMessage`:
   ```typescript
   case 'generateCommitMessage':
     this._agentProcess.sendCommand({ type: 'generateCommitMessage' });
     break;
   ```

**File: `src/kiss/agents/vscode/src/extension.ts`**

5. **Register a VS Code command** `kissSorcar.generateCommitMessage` that:
   - Calls `getActiveProvider()._agentProcess.sendCommand({ type: 'generateCommitMessage' })`.
   - Listens for the `commitMessage` response and uses `vscode.scm` API or `vscode.commands.executeCommand('git.stageAll')` to set the SCM input box text.
   - Alternatively, write the message to VS Code's SCM input box via the Git extension API.

### Phase 4: Testing

1. **Unit test `_fast_complete`** — verify prefix matching against history and file paths.
2. **Unit test the enriched `_complete` method** — verify it returns fast path results and falls back to LLM.
3. **Integration test follow-up** — verify actual result summary (not truncated prompt) is passed.
4. **Integration test commit message** — verify `_generate_commit_message` produces output from diffs.

---

## Summary of Changes by File

| File | Changes |
|------|---------|
| `src/kiss/agents/vscode/server.py` | Add `_fast_complete()`, update `_complete()` with fast path + richer context + active file, fix `_run_task` to capture real result summary, add `_generate_commit_message()`, add `generateCommitMessage` command handler, store `_last_active_file` |
| `src/kiss/agents/vscode/src/types.ts` | Add `generateCommitMessage` command type, `commitMessage` event type |
| `src/kiss/agents/vscode/src/SorcarPanel.ts` | Handle `generateCommitMessage` message, forward `commitMessage` to webview or VS Code API |
| `src/kiss/agents/vscode/src/extension.ts` | Register `kissSorcar.generateCommitMessage` command |
| `src/kiss/agents/vscode/media/main.js` | Add ghost cache, add cursor-position check, add min-length check for ghost requests |

## Non-Goals (Already at Parity)

- **Model picker** — both web and VS Code have full model selection with vendor grouping and usage tracking ✅
- **File autocomplete (@-mentions)** — both have `getFiles`/`suggestions` with usage-based sorting ✅
- **Task history** — both have sidebar with search, replay, and welcome suggestions ✅
- **Merge view** — both have accept/reject/navigate via `MergeManager` ✅
- **Follow-up rendering** — both render the "Suggested next" bar correctly ✅
- **Streaming events** — both use identical `BaseBrowserPrinter` event protocol ✅
