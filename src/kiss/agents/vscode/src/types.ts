// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

export interface Attachment {
  name: string;
  mimeType: string;
  data: string;
}

export interface SessionInfo {
  id: number;
  task_id?: number;
  title: string;
  timestamp: number;
  preview: string;
  has_events?: boolean;
}

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
  | {type: 'deleteFrequentTask'; task: string}
  | {type: 'setFavorite'; taskId: number; isFavorite: boolean}
  | {type: 'getFiles'; prefix: string; workDir?: string; tabId?: string}
  | {type: 'userAnswer'; answer: string; tabId?: string}
  | {
      type: 'openFile';
      path: string;
      line?: number;
      workDir?: string;
      tabId?: string;
    }
  | {type: 'checkPaths'; paths: string[]; workDir?: string; tabId?: string}
  | {
      type: 'shareChat';
      chatId: string;
      html: string;
      title?: string;
      workDir?: string;
      tabId?: string;
    }
  | {type: 'shareChatTasks'; chatId: string; tabId?: string}
  | {type: 'recordFileUsage'; path: string; workDir?: string}
  | {
      type: 'ready';
      tabId?: string;
      restoredTabs?: Array<{
        tabId: string;
        chatId: string;
        title?: string;
        workDir?: string;
      }>;
    }
  | {type: 'openTab'; tabId: string; title?: string; workDir?: string}
  | {
      type: 'resumeSession';
      chatId?: string;
      id?: string;
      taskId?: string | number | null;
      tabId?: string;
    }
  | {type: 'getWelcomeSuggestions'}
  | {type: 'complete'; query: string; tabId?: string}
  | {type: 'newChat'; tabId?: string}
  | {type: 'focusEditor'}
  | {type: 'closeTab'; tabId: string}
  | {type: 'getInputHistory'}
  | {
      type: 'worktreeAction';
      action: 'merge' | 'discard' | 'nothing';
      tabId?: string;
    }
  | {
      type: 'mainTreeAction';
      action: 'discard' | 'nothing';
      tabId?: string;
      workDir?: string;
    }
  | {type: 'autocommitAction'; tabId?: string; workDir?: string}
  | {type: 'resolveDroppedPaths'; uris: string[]; workDir?: string}
  | {type: 'webviewFocusChanged'; focused: boolean}
  | {type: 'activeTabChanged'; tabId: string}
  | {
      type: 'getAdjacentTask';
      tabId?: string;
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
  | {type: 'updateModels'}
  | {type: 'snoozeUpdate'; latest?: string}
  | {type: 'serverReset'}
  | {type: 'notificationAction'; id: string; action?: string}
  | {type: 'voiceToggle'; enabled: boolean; sensitivity?: number}
  | {type: 'voiceSensitivity'; value: number}
  | {type: 'voiceAck'}
  | {type: 'voiceDropped'; tabId?: string; text: string};

export type ToWebviewMessage = ToWebviewMessageBody & {tabId?: string};

type ToWebviewMessageBody =
  // roundId pairs a transcript with the wake that started it. Rounds overlap
  // (the listener re-arms while the previous utterance is transcribed), so the
  // webview needs the id to know which conversation was on screen when those
  // words were spoken.
  | {type: 'voiceWake'; roundId: number}
  | {type: 'voiceTranscribing'}
  | {
      type: 'voiceSpeech';
      roundId: number;
      text: string;
      speaker?: number;
      language?: string;
    }
  | {type: 'voiceState'; listening: boolean; error?: string}
  | {type: 'defaultModel'; model: string}
  | {type: 'kissConfig'; config: Record<string, unknown>}
  | {type: 'kissConfigSaved'; ok: boolean; error?: string}
  | {
      type: 'voiceWakeEvent';
      event: 'ready' | 'wake' | 'transcribing' | 'no_speech' | 'speech';
      text?: string;
      speaker?: number | null;
      language?: string | null;
    }
  | {type: 'voiceWakeState'; listening: boolean; error?: string}
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
  | {
      type: 'tool_result';
      content: string;
      is_error?: boolean;
      tool_name?: string;
      path?: string;
    }
  | {type: 'system_output'; text: string}
  | {
      type: 'pathsExist';
      results: Record<string, boolean>;
      workDir?: string;
    }
  | {
      // Reply to `openFile` (web_server.py _handle_open_file), sent only
      // to the requesting connection: `content` on success, `error`
      // otherwise; `tabId` echoes the request's (possibly '').
      type: 'fileContent';
      path: string;
      name: string;
      content?: string;
      error?: string;
    }
  | {type: 'share_done'; ok: boolean; path?: string; error?: string}
  | {
      type: 'share_tasks';
      chatId: string;
      tasks: Array<{task: string; task_id: string; events: unknown[]}>;
      truncated?: boolean;
      tabId?: string;
    }
  | {
      type: 'result';
      text?: string;
      summary?: string;
      success?: boolean;
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
  // chat_id is the chat's uuid string (task_runner.py re-announces the
  // overridden chat and clears viewer tabs with it).
  | {type: 'clear'; chat_id?: string}
  | {type: 'showWelcome'}
  | {type: 'clearChat'}
  // The four task-end events are one Python broadcast
  // (task_runner.py: {**task_end_event, tabId, startTs, endTs}); the
  // webview derives the per-tab "Done in …" label from the timestamps.
  | {type: 'task_done'; startTs?: number; endTs?: number}
  | {type: 'task_error'; text: string; startTs?: number; endTs?: number}
  | {type: 'task_stopped'; startTs?: number; endTs?: number}
  | {type: 'task_interrupted'; startTs?: number; endTs?: number}
  // Emitted once per run (chat_sorcar_agent.py) and synthesised into a
  // replay (json_printer.py task_settings_event); the static task panel
  // renders these like a history row.
  | {
      type: 'task_settings';
      settings: {
        model: string;
        work_dir: string;
        is_parallel: boolean;
        is_worktree: boolean;
        start_ts?: number;
        max_budget?: number;
        chat_id: string;
        task_id: string;
        is_subagent: boolean;
        parent_task_id?: string;
      };
      taskId?: string;
    }
  | {
      type: 'status';
      running: boolean;
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
      /** The server machine's hostname, shown in the status bar. */
      machine?: string;
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
  // A session replay (server.py): task_id is the history row id (None
  // for a task still running without a row), chat_id the chat's uuid
  // string, extra the row's JSON-encoded extra column ('' when absent).
  | {
      type: 'task_events';
      events: unknown[];
      task?: string;
      task_id?: string | null;
      chat_id?: string;
      extra?: string;
    }
  | {type: 'ghost'; suggestion: string; query: string}
  | {type: 'commitMessage'; message: string; error?: string}
  | {type: 'inputHistory'; tasks: string[]}
  | {
      type: 'frequentTasks';
      tasks: Array<{task: string; count: number; timestamp: number}>;
    }
  | {type: 'setTaskText'; text: string}
  | {type: 'appendToInput'; text: string}
  | {type: 'insertAndSubmit'; text: string}
  | {type: 'focusInput'}
  | {
      type: 'worktree_created';
      worktreeDir: string;
      worktreeWorkDir?: string;
      branch: string;
    }
  | {
      type: 'worktree_done';
      branch: string;
      worktreeDir: string;
      worktreeWorkDir?: string;
      originalBranch: string;
      changedFiles: string[];
      hasConflict?: boolean;
    }
  | {type: 'worktree_progress'; message: string}
  | {
      type: 'worktree_result';
      success: boolean;
      message: string;
      kept?: boolean;
      // A failure the user can retry (merge_flow.py: deferred discard
      // while a sub-agent still holds the worktree): the webview keeps
      // the Merge / Discard bar instead of stripping it.
      retryable?: boolean;
    }
  | {
      type: 'main_tree_done';
      workDir?: string;
      changedFiles: string[];
    }
  | {type: 'main_tree_result'; success: boolean; message: string}
  | {type: 'warning'; message: string; tabId?: string}
  | {type: 'autocommit_progress'; message: string; tabId?: string}
  | {
      type: 'autocommit_done';
      success: boolean;
      committed: boolean;
      message: string;
      commitMessage?: string;
      tabId?: string;
      manual?: boolean;
      workDir?: string;
    }
  | {type: 'droppedPaths'; paths: string[]}
  | {
      type: 'adjacent_task_events';
      direction: 'prev' | 'next';
      task: string;
      task_id: string | number | null;
      events: unknown[];
    }
  | {type: 'triggerStop'}
  | {type: 'measureSize'}
  | {type: 'daemonStatus'; connected: boolean}
  // A toast rendered by the webview (media/main.js updateNotification).
  // Posted by the extension host in place of vscode.window.show*Message
  // (WebviewNotifications.ts) and by the daemon (manual-commit outcome,
  // server-reset progress).  A stable `id` lets a later message replace
  // the toast in place; `close` retires it; `progress` marks a spinner
  // toast whose text is `progressMessage`.
  | {
      type: 'notification';
      id: string;
      severity?: 'info' | 'warning' | 'error';
      message?: string;
      actions?: string[];
      sticky?: boolean;
      progress?: boolean;
      progressMessage?: string;
      close?: boolean;
    }
  // Daemon: a plain informational line for one connection (e.g. "an
  // update is already running").
  | {type: 'notice'; text: string}
  // Daemon: answer to a `complete` command (the input-box ghost /
  // autocomplete list), scoped to the requesting connection and tab.
  | {
      type: 'completions';
      completions: Array<{type: string; text: string}>;
      query: string;
    }
  // Daemon: a run_parallel sub-agent tab was retired everywhere.
  | {type: 'closeSubagentTab'; tab_id: string}
  // Daemon: `openTab` was refused (tab limit); `text` explains why.
  | {type: 'openTabRejected'; text: string}
  // Daemon: cached PyPI check result for the Update button/badge.
  // `snoozed` marks an active "Remind me later" snooze: the webview
  // keeps the badge but suppresses the sticky toast.
  | {
      type: 'update_available';
      available: boolean;
      latest: string;
      current: string;
      snoozed?: boolean;
    }
  // The window's workspace folder changed; the webview re-scopes its
  // workspace-filtered surfaces (tab bar, history) to this directory.
  | {type: 'workspaceWorkDir'; workDir: string}
  | {
      // Canonical shared-tab snapshot broadcast by the daemon after
      // every tab-registry mutation; clients reconcile against it.
      type: 'tabs_state';
      tabs: Array<{
        tabId: string;
        chatId: string;
        title: string;
        workDir: string;
        // Workspace-visibility scope, distinct from workDir (the
        // execution directory): a run_agent sub-task runs in a
        // channel/cron scratch dir but is scoped to the calling
        // workspace. Empty means "scope by workDir".
        scopeWorkDir: string;
      }>;
    }
  | {
      type: 'openSubagentTab';
      tab_id?: string;
      parent_tab_id?: string;
      description?: string;
      // The sub-agent's history-row id (server.py), null when the row
      // has none; main.js treats null/undefined as ''.
      task_id?: string | null;
      taskIndex?: number;
      isSubagentTab?: boolean;
      isDone?: boolean;
    }
  | {type: 'subagentDone'; tab_id?: string; success?: boolean}
  | {
      // Transient picker label update: 'agent' is the model a running
      // agent switched itself to, 'restore' the end-of-task revert to
      // the model the user picked in that tab.
      type: 'modelPick';
      model: string;
      source: 'agent' | 'restore';
      tabId: string;
    }
  | {
      // Receipt for a Stop click: `accepted` is false when the daemon
      // found no running task owning `tabId`, so the UI can say so
      // instead of leaving the button looking dead.
      type: 'stop_ack';
      accepted: boolean;
      tabId: string;
    }
  | {
      type: 'new_tab';
      task_id: string | number;
      parent_tab_id?: string;
      taskId?: string;
    };

export interface AgentCommand {
  type:
    | 'run'
    | 'stop'
    | 'appendUserMessage'
    | 'getModels'
    | 'selectModel'
    | 'getHistory'
    | 'getFrequentTasks'
    | 'deleteFrequentTask'
    | 'setFavorite'
    | 'getFiles'
    | 'userAnswer'
    | 'recordFileUsage'
    | 'resumeSession'
    | 'complete'
    | 'newChat'
    | 'openTab'
    | 'closeTab'
    | 'ready'
    | 'generateCommitMessage'
    | 'autocommitAction'
    | 'getInputHistory'
    | 'worktreeAction'
    | 'mainTreeAction'
    | 'getAdjacentTask'
    | 'setWorkDir'
    | 'getConfig'
    | 'saveConfig'
    | 'serverReset'
    | 'shareChat'
    | 'shareChatTasks'
    | 'snoozeUpdate';
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
  html?: string;
  chatId?: number | string;
  taskId?: string | number | null;
  activeFileContent?: string;
  action?: 'merge' | 'discard' | 'nothing';
  useWorktree?: boolean;
  useParallel?: boolean;
  autoCommit?: boolean;
  task?: string;
  direction?: 'prev' | 'next';
  tabId?: string;
  config?: Record<string, unknown>;
  apiKeys?: Record<string, string>;
  isFavorite?: boolean;
  title?: string;
  latest?: string;
  restoredTabs?: Array<{
    tabId: string;
    chatId: string;
    title?: string;
    workDir?: string;
  }>;
}
