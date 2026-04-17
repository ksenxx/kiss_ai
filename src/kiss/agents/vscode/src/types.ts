/**
 * Type definitions for VS Code extension messaging.
 */

/** Attachment for file uploads */
export interface Attachment {
  name: string;
  mimeType: string;
  data: string; // Base64 encoded
}

/** Session/conversation info */
export interface SessionInfo {
  id: number;
  title: string;
  timestamp: number;
  preview: string;
  has_events?: boolean;
}

/** Messages from webview to extension */
export type FromWebviewMessage =
  | { type: 'submit'; prompt: string; model: string; attachments: Attachment[]; useWorktree?: boolean; useParallel?: boolean; tabId?: string }
  | { type: 'stop'; tabId?: string }
  | { type: 'selectModel'; model: string; tabId?: string }
  | { type: 'getModels' }
  | { type: 'getHistory'; query?: string; offset?: number; generation?: number }
  | { type: 'getFiles'; prefix: string }
  | { type: 'userAnswer'; answer: string; tabId?: string }
  | { type: 'userActionDone' }
  | { type: 'openFile'; path: string; line?: number }
  | { type: 'recordFileUsage'; path: string }
  | { type: 'ready'; tabId?: string }
  | { type: 'resumeSession'; id: number; tabId?: string }
  | { type: 'complete'; query: string }
  | { type: 'mergeAction'; action: string }
  | { type: 'newChat'; tabId?: string }
  | { type: 'generateCommitMessage' }
  | { type: 'runPrompt' }
  | { type: 'focusEditor' }
  | { type: 'closeSecondaryBar' }
  | { type: 'getInputHistory' }
  | { type: 'worktreeAction'; action: 'merge' | 'discard' | 'do_nothing' }
  | { type: 'resolveDroppedPaths'; uris: string[] }
  | { type: 'getAdjacentTask'; tabId?: string; task: string; direction: 'prev' | 'next' };

/** Messages from extension to webview (matches browser event protocol) */
export type ToWebviewMessage =
  // Streaming events (same as browser BaseBrowserPrinter)
  | { type: 'thinking_start' }
  | { type: 'thinking_delta'; text: string }
  | { type: 'thinking_end' }
  | { type: 'text_delta'; text: string }
  | { type: 'text_end' }
  | { type: 'tool_call'; name: string; path?: string; lang?: string; description?: string; command?: string; content?: string; old_string?: string; new_string?: string; extras?: Record<string, string> }
  | { type: 'tool_result'; content: string; is_error?: boolean }
  | { type: 'system_output'; text: string }
  | { type: 'result'; text?: string; summary?: string; success?: boolean; total_tokens?: number; cost?: string }
  | { type: 'system_prompt'; text: string }
  | { type: 'prompt'; text: string }
  // Lifecycle events
  | { type: 'clear'; chat_id?: number }
  | { type: 'clearChat' }
  | { type: 'ensureChat' }
  | { type: 'task_done' }
  | { type: 'task_error'; text: string }
  | { type: 'task_stopped' }
  // UI events
  | { type: 'status'; running: boolean }
  | { type: 'models'; models: Array<{name: string; inp: number; out: number; uses: number; vendor: string}>; selected: string }
  | { type: 'history'; sessions: SessionInfo[]; offset?: number; generation?: number }
  | { type: 'files'; files: Array<{type: string; text: string}> }
  | { type: 'askUser'; question: string }
  | { type: 'error'; text: string }
  | { type: 'followup_suggestion'; text: string }
  | { type: 'tasks_updated' }
  | { type: 'task_events'; events: any[]; task?: string; chat_id?: number }
  | { type: 'ghost'; suggestion: string; query: string }
  | { type: 'merge_data'; data: any; hunk_count: number }
  | { type: 'merge_started' }
  | { type: 'merge_ended' }
  | { type: 'commitMessage'; message: string; error?: string }
  | { type: 'activeFileInfo'; isPrompt: boolean; filename: string; path: string }
  | { type: 'inputHistory'; tasks: string[] }
  | { type: 'setTaskText'; text: string }
  | { type: 'appendToInput'; text: string }
  | { type: 'focusInput' }
  | { type: 'worktree_created'; worktreeDir: string; branch: string }
  | { type: 'worktree_done'; branch: string; worktreeDir: string; originalBranch: string; changedFiles: string[]; hasConflict?: boolean }
  | { type: 'worktree_progress'; message: string }
  | { type: 'worktree_result'; success: boolean; message: string }
  | { type: 'droppedPaths'; paths: string[] }
  | { type: 'adjacent_task_events'; direction: string; task: string; events: any[] }
  | { type: 'triggerStop' };

/** Command sent to Python backend */
export interface AgentCommand {
  type: 'run' | 'stop' | 'getModels' | 'selectModel' | 'getHistory' | 'getFiles' | 'userAnswer' | 'recordFileUsage' | 'resumeSession' | 'complete' | 'mergeAction' | 'refreshFiles' | 'newChat' | 'generateCommitMessage' | 'getInputHistory' | 'worktreeAction' | 'getAdjacentTask';
  prompt?: string;
  model?: string;
  workDir?: string;
  activeFile?: string;
  attachments?: Attachment[];
  query?: string;
  offset?: number;
  generation?: number;
  prefix?: string;
  answer?: string;
  path?: string;
  chatId?: number | string;
  activeFileContent?: string;
  action?: 'merge' | 'discard' | 'do_nothing' | 'all-done';
  useWorktree?: boolean;
  useParallel?: boolean;
  task?: string;
  direction?: 'prev' | 'next';
  tabId?: string;
}
