// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
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
      autoCommit?: boolean;
      tabId?: string;
      workDir?: string;
    }
  | {type: 'stop'; tabId?: string}
  | {type: 'appendUserMessage'; prompt: string; tabId?: string}
  | {type: 'selectModel'; model: string; tabId?: string}
  | {type: 'getHistory'; query?: string; offset?: number; generation?: number}
  | {type: 'getFrequentTasks'; limit?: number}
  | {type: 'deleteTask'; taskId: number}
  | {type: 'deleteFrequentTask'; task: string}
  | {type: 'setFavorite'; taskId: number; isFavorite: boolean}
  | {type: 'getFiles'; prefix: string; workDir?: string}
  | {type: 'userAnswer'; answer: string; tabId?: string}
  | {type: 'openFile'; path: string; line?: number}
  | {type: 'recordFileUsage'; path: string; workDir?: string}
  | {
      type: 'ready';
      tabId?: string;
      restoredTabs?: Array<{tabId: string; chatId: string}>;
    }
  | {type: 'resumeSession'; id: string; taskId?: number; tabId?: string}
  | {type: 'getWelcomeSuggestions'}
  | {type: 'complete'; query: string; tabId?: string}
  | {type: 'mergeAction'; action: string; tabId?: string; workDir?: string}
  | {type: 'newChat'; tabId?: string}
  | {type: 'focusEditor'}
  | {type: 'closeTab'; tabId: string}
  | {type: 'getInputHistory'}
  | {type: 'worktreeAction'; action: 'merge' | 'discard'; tabId?: string}
  | {
      type: 'autocommitAction';
      action: 'commit' | 'skip';
      tabId?: string;
      workDir?: string;
    }
  | {type: 'resolveDroppedPaths'; uris: string[]}
  | {type: 'webviewFocusChanged'; focused: boolean}
  | {
      type: 'getAdjacentTask';
      tabId?: string;
      // DB row id of the reference task (UUID string; legacy rows may
      // carry ints), or null when the tab has no known task row yet.
      taskId: string | number | null;
      direction: 'prev' | 'next';
    }
  | {type: 'getConfig'}
  | {
      type: 'saveConfig';
      config: Record<string, unknown>;
      apiKeys: Record<string, string>;
    }
  | {type: 'sizeReport'; innerWidth: number; screenWidth: number}
  | {type: 'runUpdate'}
  | {type: 'serverReset'}
  | {type: 'notificationAction'; id: string; action?: string}
  | {type: 'voiceToggle'; enabled: boolean; sensitivity?: number}
  | {type: 'voiceSensitivity'; value: number}
  | {type: 'voiceAck'};

/** Messages from extension to webview (matches browser event protocol) */
export type ToWebviewMessage = ToWebviewMessageBody & {tabId?: string};

type ToWebviewMessageBody =
  // Voice wake-word events (host-side listener → voice.js)
  | {type: 'voiceWake'}
  | {type: 'voiceTranscribing'}
  | {type: 'voiceSpeech'; text: string; speaker?: number; language?: string}
  | {type: 'voiceState'; listening: boolean; error?: string}
  // Streaming events (same as browser JsonPrinter)
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
      /** True when the agent paused to continue in a new session
       *  (``json_printer`` copies ``is_continue`` onto result events;
       *  main.js renders a "Status: Continue" banner for it). */
      is_continue?: boolean;
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
  // Agent-initiated text-to-speech (the ``talk`` tool): the webview
  // plays the GPT-synthesized clip (audioB64), staying silent when no
  // clip can play; talkId dedupes fan-out copies, muted marks copies
  // already played on this machine by another local player.
  | {
      type: 'talk';
      text: string;
      language?: string;
      emotion?: string;
      talkId?: string;
      audioB64?: string;
      audioMime?: string;
      muted?: boolean;
    }
  // Lifecycle events
  | {type: 'clear'; chat_id?: number}
  | {type: 'showWelcome'}
  | {type: 'clearChat'}
  | {type: 'task_done'}
  | {type: 'task_error'; text: string}
  | {type: 'task_stopped'}
  | {type: 'task_interrupted'}
  // UI events
  | {
      type: 'status';
      running: boolean;
      /** Agent's true start timestamp (ms since epoch) supplied by the
       *  backend (``task_runner`` / ``server._replay_session``) so the
       *  webview's "Running …" timer is anchored to agent wall-clock. */
      startTs?: number;
    }
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
      dateRange?: {min: number | null; max: number | null};
    }
  | {
      type: 'files';
      files: Array<{type: string; text: string}>;
      prefix?: string;
      loading?: boolean;
    }
  | {type: 'askUser'; question: string; tabId?: string}
  | {type: 'askUserDone'; tabId?: string}
  | {type: 'error'; text: string}
  | {type: 'followup_suggestion'; text: string}
  | {type: 'tasks_updated'}
  | {type: 'welcome_suggestions'; suggestions: Array<{text: string}>}
  | {type: 'remote_url'; url: string; ntfyUrl?: string; tunnelActive?: boolean}
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
  | {type: 'warning'; message: string; tabId?: string}
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
  | {
      type: 'adjacent_task_events';
      direction: 'prev' | 'next';
      task: string;
      // DB row id (UUID string; legacy rows may carry ints) of the
      // adjacent task, or null when no adjacent row exists.
      task_id: string | number | null;
      events: unknown[];
    }
  | {type: 'triggerStop'}
  | {
      type: 'taskDeleted';
      chatId: number;
      taskId: number;
      chatHasMoreTasks: boolean;
    }
  | {type: 'measureSize'}
  | {type: 'daemonStatus'; connected: boolean}
  | {
      type: 'openSubagentTab';
      tab_id?: string;
      parent_tab_id?: string;
      description?: string;
      taskIndex?: number;
      isSubagentTab?: boolean;
      isDone?: boolean;
    }
  | {type: 'subagentDone'; tab_id?: string; success?: boolean}
  | {
      // Sub-agent tab announcement: ``task_id`` is the sub-agent's
      // persisted ``task_history.id`` (a UUID hex string) and
      // ``parent_tab_id`` is the frontend tab id of the parent
      // run_parallel tab (empty when the parent has no tab).  The
      // broadcast is stamped ``taskId: ''`` so it stays a global
      // system event (see ``ChatSorcarAgent.run``).
      type: 'new_tab';
      task_id: string | number;
      parent_tab_id?: string;
      taskId?: string;
    };

/** Command sent to Python backend */
export interface AgentCommand {
  type:
    | 'run'
    | 'stop'
    | 'appendUserMessage'
    | 'getModels'
    | 'selectModel'
    | 'getHistory'
    | 'getFrequentTasks'
    | 'deleteTask'
    | 'deleteFrequentTask'
    | 'setFavorite'
    | 'getFiles'
    | 'userAnswer'
    | 'recordFileUsage'
    | 'resumeSession'
    | 'complete'
    | 'mergeAction'
    | 'newChat'
    | 'closeTab'
    | 'generateCommitMessage'
    | 'getInputHistory'
    | 'worktreeAction'
    | 'autocommitAction'
    | 'getAdjacentTask'
    | 'setWorkDir'
    | 'getConfig'
    | 'saveConfig'
    | 'serverReset';
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
  taskId?: number | null;
  activeFileContent?: string;
  action?: 'merge' | 'discard' | 'all-done' | 'commit' | 'skip';
  useWorktree?: boolean;
  useParallel?: boolean;
  autoCommit?: boolean;
  task?: string;
  direction?: 'prev' | 'next';
  tabId?: string;
  config?: Record<string, unknown>;
  apiKeys?: Record<string, string>;
  isFavorite?: boolean;
}
