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
  id: string;
  title: string;
  timestamp: number;
  preview: string;
  has_events?: boolean;
}

/** Messages from webview to extension */
export type FromWebviewMessage =
  | { type: 'submit'; prompt: string; model: string; attachments: Attachment[] }
  | { type: 'stop' }
  | { type: 'selectModel'; model: string }
  | { type: 'getModels' }
  | { type: 'getHistory'; query?: string }
  | { type: 'getFiles'; prefix: string }
  | { type: 'userAnswer'; answer: string }
  | { type: 'userActionDone' }
  | { type: 'openFile'; path: string; line?: number }
  | { type: 'recordFileUsage'; path: string }
  | { type: 'ready' }
  | { type: 'resumeSession'; id: string }
  | { type: 'getWelcomeSuggestions' }
  | { type: 'mergeAction'; action: string }
  | { type: 'newChat' }
  | { type: 'generateCommitMessage' }
  | { type: 'runPrompt' }
  | { type: 'focusEditor' };

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
  | { type: 'usage_info'; text: string }
  // Lifecycle events
  | { type: 'clear' }
  | { type: 'clearChat' }
  | { type: 'task_done' }
  | { type: 'task_error'; text: string }
  | { type: 'task_stopped' }
  // UI events
  | { type: 'status'; running: boolean }
  | { type: 'models'; models: Array<{name: string; inp: number; out: number; uses: number; vendor: string}>; selected: string }
  | { type: 'history'; sessions: SessionInfo[] }
  | { type: 'files'; files: Array<{type: string; text: string}> }
  | { type: 'askUser'; question: string }
  | { type: 'waitForUser'; instruction: string; url: string }
  | { type: 'error'; text: string }
  | { type: 'followup_suggestion'; text: string }
  | { type: 'tasks_updated' }
  | { type: 'welcome_suggestions'; suggestions: Array<{text: string; has_events: boolean}> }
  | { type: 'task_events'; events: any[]; task?: string }
  | { type: 'merge_data'; data: any; hunk_count: number }
  | { type: 'merge_started' }
  | { type: 'merge_ended' }
  | { type: 'commitMessage'; message: string; error?: string }
  | { type: 'activeFileInfo'; isPrompt: boolean; filename: string; path: string };

/** Command sent to Python backend */
export interface AgentCommand {
  type: 'run' | 'stop' | 'getModels' | 'selectModel' | 'getHistory' | 'getFiles' | 'userAnswer' | 'recordFileUsage' | 'resumeSession' | 'getLastSession' | 'mergeAction' | 'refreshFiles' | 'newChat' | 'generateCommitMessage';
  prompt?: string;
  model?: string;
  workDir?: string;
  activeFile?: string;
  attachments?: Attachment[];
  query?: string;
  prefix?: string;
  answer?: string;
  path?: string;
  sessionId?: string;
  action?: string;
}
