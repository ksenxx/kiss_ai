/**
 * Type definitions for VS Code extension messaging.
 */

import {MergeData} from './MergeManager';

/** Attachment for file uploads */
export interface Attachment {
  name: string;
  mimeType: string;
  data: string; // Base64 encoded
}

/** Session/conversation info */
export interface SessionInfo {
  id: number;
  task_id?: number;
  title: string;
  timestamp: number;
  preview: string;
  has_events?: boolean;
}

/** Messages from webview to extension */
export type FromWebviewMessage =
  | {
      type: 'submit';
      prompt: string;
      model: string;
      attachments: Attachment[];
      useWorktree?: boolean;
      useParallel?: boolean;
      tabId?: string;
      workDir?: string;
      skipMerge?: boolean;
      reuseProcess?: boolean;
    }
  | {type: 'stop'; tabId?: string}
  | {type: 'selectModel'; model: string; tabId?: string}
  | {type: 'getModels'}
  | {type: 'getHistory'; query?: string; offset?: number; generation?: number}
  | {type: 'getFrequentTasks'; limit?: number}
  | {type: 'deleteTask'; taskId: number}
  | {type: 'getFiles'; prefix: string}
  | {type: 'userAnswer'; answer: string; tabId?: string}
  | {type: 'userActionDone'}
  | {type: 'openFile'; path: string; line?: number}
  | {type: 'recordFileUsage'; path: string}
  | {
      type: 'ready';
      tabId?: string;
      restoredTabs?: Array<{tabId: string; chatId: string}>;
    }
  | {type: 'resumeSession'; id: number; tabId?: string}
  | {type: 'getWelcomeSuggestions'}
  | {type: 'complete'; query: string}
  | {type: 'mergeAction'; action: string; tabId?: string}
  | {type: 'newChat'; tabId?: string}
  | {type: 'generateCommitMessage'}
  | {type: 'focusEditor'}
  | {type: 'closeTab'; tabId: string}
  | {type: 'getInputHistory'}
  | {type: 'worktreeAction'; action: 'merge' | 'discard'; tabId?: string}
  | {type: 'autocommitAction'; action: 'commit' | 'skip'; tabId?: string}
  | {type: 'setSkipMerge'; tabId?: string; skip: boolean}
  | {type: 'resolveDroppedPaths'; uris: string[]}
  | {type: 'webviewFocusChanged'; focused: boolean}
  | {type: 'pickFolder'}
  | {
      type: 'getAdjacentTask';
      tabId?: string;
      task: string;
      direction: 'prev' | 'next';
    }
  | {type: 'getConfig'}
  | {
      type: 'saveConfig';
      config: Record<string, unknown>;
      apiKeys: Record<string, string>;
    };

/** Messages from extension to webview (matches browser event protocol) */
export type ToWebviewMessage = ToWebviewMessageBody & {tabId?: string};

type ToWebviewMessageBody =
  // Streaming events (same as browser BaseBrowserPrinter)
  | {type: 'thinking_start'}
  | {type: 'thinking_delta'; text: string}
  | {type: 'thinking_end'}
  | {type: 'text_delta'; text: string}
  | {type: 'text_end'}
  | {
      type: 'tool_call';
      name: string;
      path?: string;
      lang?: string;
      description?: string;
      command?: string;
      content?: string;
      old_string?: string;
      new_string?: string;
      extras?: Record<string, string>;
    }
  | {type: 'tool_result'; content: string; is_error?: boolean}
  | {type: 'system_output'; text: string}
  | {
      type: 'result';
      text?: string;
      summary?: string;
      success?: boolean;
      total_tokens?: number;
      cost?: string;
      step_count?: number;
    }
  | {
      type: 'usage_info';
      text?: string;
      total_tokens?: number;
      cost?: string;
      total_steps?: number;
    }
  | {type: 'system_prompt'; text: string}
  | {type: 'prompt'; text: string}
  // Lifecycle events
  | {type: 'clear'; chat_id?: number}
  | {type: 'showWelcome'}
  | {type: 'clearChat'}
  | {type: 'ensureChat'}
  | {type: 'task_done'}
  | {type: 'task_error'; text: string}
  | {type: 'task_stopped'}
  // UI events
  | {type: 'status'; running: boolean}
  | {
      type: 'models';
      models: Array<{
        name: string;
        inp: number;
        out: number;
        uses: number;
        vendor: string;
      }>;
      selected: string;
    }
  | {
      type: 'configData';
      config: Record<string, unknown>;
      apiKeys?: Record<string, string>;
    }
  | {
      type: 'history';
      sessions: SessionInfo[];
      offset?: number;
      generation?: number;
    }
  | {type: 'files'; files: Array<{type: string; text: string}>}
  | {type: 'askUser'; question: string}
  | {type: 'error'; text: string}
  | {type: 'followup_suggestion'; text: string}
  | {type: 'tasks_updated'}
  | {type: 'welcome_suggestions'; suggestions: Array<{text: string}>}
  | {type: 'remote_url'; url: string}
  | {type: 'task_events'; events: unknown[]; task?: string; chat_id?: number}
  | {type: 'ghost'; suggestion: string; query: string}
  | {type: 'merge_data'; data: MergeData; hunk_count: number}
  | {type: 'merge_nav'; remaining: number; total: number}
  | {type: 'merge_started'}
  | {type: 'merge_ended'}
  | {type: 'commitMessage'; message: string; error?: string}
  | {type: 'inputHistory'; tasks: string[]}
  | {
      type: 'frequentTasks';
      tasks: Array<{task: string; count: number; timestamp: number}>;
    }
  | {type: 'setTaskText'; text: string}
  | {type: 'appendToInput'; text: string}
  | {type: 'focusInput'}
  | {type: 'worktree_created'; worktreeDir: string; branch: string}
  | {
      type: 'worktree_done';
      branch: string;
      worktreeDir: string;
      originalBranch: string;
      changedFiles: string[];
      hasConflict?: boolean;
    }
  | {type: 'worktree_progress'; message: string}
  | {type: 'worktree_result'; success: boolean; message: string}
  | {type: 'autocommit_prompt'; changedFiles: string[]; tabId?: string}
  | {type: 'autocommit_progress'; message: string; tabId?: string}
  | {
      type: 'autocommit_done';
      success: boolean;
      committed: boolean;
      message: string;
      commitMessage?: string;
      tabId?: string;
    }
  | {type: 'droppedPaths'; paths: string[]}
  | {type: 'folderPicked'; path: string}
  | {
      type: 'adjacent_task_events';
      direction: string;
      task: string;
      events: unknown[];
    }
  | {type: 'triggerStop'}
  | {
      type: 'taskDeleted';
      chatId: number;
      taskId: number;
      chatHasMoreTasks: boolean;
    };

/** Command sent to Python backend */
export interface AgentCommand {
  type:
    | 'run'
    | 'stop'
    | 'getModels'
    | 'selectModel'
    | 'getHistory'
    | 'getFrequentTasks'
    | 'deleteTask'
    | 'getFiles'
    | 'userAnswer'
    | 'recordFileUsage'
    | 'resumeSession'
    | 'complete'
    | 'mergeAction'
    | 'refreshFiles'
    | 'newChat'
    | 'closeTab'
    | 'generateCommitMessage'
    | 'getInputHistory'
    | 'worktreeAction'
    | 'autocommitAction'
    | 'getAdjacentTask'
    | 'setSkipMerge'
    | 'getConfig'
    | 'saveConfig';
  prompt?: string;
  model?: string;
  workDir?: string;
  activeFile?: string;
  attachments?: Attachment[];
  query?: string;
  offset?: number;
  generation?: number;
  limit?: number;
  prefix?: string;
  answer?: string;
  path?: string;
  chatId?: number | string;
  taskId?: number;
  activeFileContent?: string;
  action?: 'merge' | 'discard' | 'all-done' | 'commit' | 'skip';
  useWorktree?: boolean;
  useParallel?: boolean;
  task?: string;
  direction?: 'prev' | 'next';
  tabId?: string;
  skip?: boolean;
  skipMerge?: boolean;
  config?: Record<string, unknown>;
  apiKeys?: Record<string, string>;
}
