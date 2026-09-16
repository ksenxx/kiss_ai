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
      webTools?: boolean;
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
  // Remote webapp only: write an editable content tab's Monaco text back
  // to the file it was opened from (web_server.py _handle_save_file);
  // the daemon drops it when a VS Code window sends it.
  | {
      type: 'saveFile';
      path: string;
      content: string;
      workDir?: string;
      tabId?: string;
      /** `<content tab id>:<save sequence>`, echoed so the reply settles
       * exactly the request still awaited (a straggler for an earlier,
       * timed-out save finds no taker). */
      token?: string;
      /** The `version` stamp the fileContent reply reported; a file whose
       * stamp changed since is refused (`conflict`) unless `force` is set. */
      version?: string;
      force?: boolean;
    }
  // Remote webapp only (the activity bar's Explorer / Source Control
  // views); the daemon drops these when a VS Code window sends them.
  | {
      type: 'listDir';
      path?: string;
      workDir?: string;
      tabId?: string;
      token?: string;
    }
  | {type: 'gitStatus'; workDir?: string; tabId?: string; token?: string}
  | {
      type: 'gitLog';
      workDir?: string;
      tabId?: string;
      token?: string;
      limit?: number;
    }
  | {
      type: 'shareChat';
      chatId: string;
      html: string;
      title?: string;
      workDir?: string;
      tabId?: string;
    }
  | {type: 'shareChatTasks'; chatId: string; tabId?: string; taskId?: string}
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
  | {
      type: 'resolveDroppedPaths';
      uris: string[];
      workDir?: string;
      // The originating chat tab: echoed back on the droppedPaths reply
      // so a tab switch during the round trip cannot leak the paths into
      // another tab's composer.
      tabId?: string;
    }
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
  | {type: 'getMyModels'}
  | {
      type: 'saveMyModel';
      name: string;
      endpoint?: string;
      apiKey?: string;
      headers?: string;
      /** The entry's name before an edit-and-rename ('' for adds). */
      originalName?: string;
    }
  | {type: 'deleteMyModel'; name: string}
  | {type: 'sizeReport'; innerWidth: number; screenWidth: number}
  | {type: 'runUpdate'}
  | {type: 'updateModels'}
  | {type: 'snoozeUpdate'; latest?: string}
  | {type: 'serverReset'}
  | {type: 'notificationAction'; id: string; action?: string}
  | {type: 'voiceToggle'; enabled: boolean; sensitivity?: number}
  // In-page (browser-mic) capture fallback: the webview recorded the
  // post-wake utterance itself and ships it to the daemon for
  // transcription — the exact message the remote webapp sends over its
  // WebSocket. The host forwards it verbatim (FORWARDED_COMMANDS); the
  // daemon answers with a `voiceSpeech` the client relay passes back.
  | {
      type: 'voiceTranscribe';
      audio: string;
      wakePrefixed?: boolean;
      wakeSamples?: number;
    }
  | {type: 'voiceSensitivity'; value: number}
  | {type: 'voiceAck'}
  | {type: 'voiceDropped'; tabId?: string; text: string}
  // Editor-tabs mode (host-only, never forwarded to the daemon): the
  // webview's root chat tab renamed itself or its task's status
  // changed, so the hosting editor tab should follow. `state` mirrors
  // the internal tab strip's status dot: '' (no task yet), 'running',
  // 'ok' or 'fail'.
  | {type: 'panelTitle'; title: string; tabId?: string; state?: string}
  // Editor-tabs mode: a task in this panel just finished — bring the
  // hosting editor tab forward (sidebar mode's finished-task switch).
  | {type: 'revealPanel'}
  // Editor-tabs mode: open another chat as a new editor tab — a fresh
  // conversation when chatId is absent, a history resume otherwise.
  | {
      type: 'openChatPanel';
      chatId?: string;
      taskId?: string | number | null;
      title?: string;
      // Fresh conversations only: the opening webview's composer draft,
      // stamped onto the new panel as data-kiss-pending-text so the new
      // chat's textarea starts out with the same text.
      pendingText?: string;
    }
  // Editor-tabs mode: close this panel — because the daemon's registry
  // no longer lists its chat tab (another client closed it; retire
  // absent/false), or because the user closed the root chat inside the
  // panel (retire true: the host must also retire the tab from the
  // registry).
  | {type: 'closePanel'; retire?: boolean}
  // The settings UI's editor-tabs toggle (both modes).
  | {type: 'setEditorTabsMode'; enabled: boolean};

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
      // The host's own listener stamps the round id; a daemon reply to a
      // forwarded `voiceTranscribe` (in-page capture fallback) carries
      // none — voice.js then answers its oldest unkeyed round.
      roundId?: number;
      text: string;
      speaker?: number | null;
      language?: string | null;
    }
  // hostMicUnavailable: the host machine cannot run the wake listener at
  // all (no microphone/PortAudio, uv missing — it died before READY).
  // The webview shows a calm "unavailable" state instead of an error.
  | {
      type: 'voiceState';
      listening: boolean;
      error?: string;
      hostMicUnavailable?: boolean;
    }
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
      /** Images the tool call generated, embedded for inline display. */
      images?: Array<{path?: string; mime: string; b64: string}>;
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
      /** True when `path` is a directory and `content` is its plain-text
       * listing (rendered as text even for md/html-looking names). */
      isDirectory?: boolean;
      error?: string;
      /** Echo of the request's `line` (a path:NN link's line number). */
      line?: number;
      /** The file's `"<st_mtime_ns>:<st_size>"` stamp as read; `saveFile`
       * hands it back so a file changed on disk while open is not silently
       * overwritten. A string: nanosecond mtimes exceed 2^53. */
      version?: string;
    }
  | {
      // Reply to `saveFile` (web_server.py _handle_save_file), sent only
      // to the requesting connection.
      type: 'fileSaved';
      ok: boolean;
      path: string;
      name: string;
      tabId?: string;
      token?: string;
      /** The file's version stamp after the write (on success). */
      version?: string;
      error?: string;
      /** True when the write was refused because the file changed on
       * disk since it was opened (retry with `force` to overwrite). */
      conflict?: boolean;
    }
  | {
      // Reply to `listDir` (web_server.py _handle_list_dir), sent only to
      // the requesting connection: the Explorer view's folder contents.
      type: 'dirListing';
      path: string;
      root: string;
      tabId?: string;
      token?: string;
      entries?: Array<{name: string; path: string; isDir: boolean}>;
      truncated?: boolean;
      error?: string;
    }
  | {
      // Reply to `gitStatus` (web_server.py _handle_git_status): the
      // Source Control view's "Changes" rows.
      type: 'gitStatus';
      workDir: string;
      tabId?: string;
      token?: string;
      repo?: string;
      branch?: string;
      changes?: Array<{
        path: string;
        absPath: string;
        status: string;
        group: 'merge' | 'staged' | 'changes';
        origPath?: string;
      }>;
      error?: string;
    }
  | {
      // Reply to `gitLog` (web_server.py _handle_git_log): the Source
      // Control view's commit graph rows, newest first.
      type: 'gitLog';
      workDir: string;
      tabId?: string;
      token?: string;
      repo?: string;
      head?: string;
      commits?: Array<{
        sha: string;
        shortSha: string;
        parents: string[];
        author: string;
        date: string;
        refs: string[];
        subject: string;
        files: Array<{path: string; status: string; origPath?: string}>;
      }>;
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
      type: 'myModelsData';
      models: Array<{
        name: string;
        endpoint: string;
        api_key: string;
        headers: string;
      }>;
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
  | {type: 'droppedPaths'; paths: string[]; tabId?: string}
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
  // Host (editor-tabs mode): open the webview's settings panel — the
  // editor-title gear button's action.
  | {type: 'openSettings'}
  // Host (editor-tabs mode): run the manual Git Commit of the active
  // chat tab's working tree — the editor-title git-commit button's
  // action (same flow as the settings drawer's Git Commit button).
  | {type: 'gitCommit'}
  // Host (editor-tabs mode): bring one of the panel's own chat's tasks
  // on screen — a history-panel click on a task of a chat whose editor
  // tab is already open. The webview scrolls to the task's transcript
  // region, or replays the task when it is not rendered.
  | {type: 'showTask'; taskId: string}
  // Daemon: answer to a `complete` command (the input-box ghost /
  // autocomplete list), scoped to the requesting connection and tab.
  | {
      type: 'completions';
      completions: Array<{type: string; text: string}>;
      query: string;
    }
  // Daemon: a sub-agent tab (run_parallel or run_agent child) was
  // retired everywhere.
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
      // The sub-agent row's wall-clock start (ms), 0/absent when
      // unknown; main.js attributes the sub-agent to the fan-out call
      // (run_parallel / run_agent) that was running at that time.
      startTs?: number;
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
    | 'getTabsState'
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
    | 'getMyModels'
    | 'saveMyModel'
    | 'deleteMyModel'
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
  webTools?: boolean;
  task?: string;
  direction?: 'prev' | 'next';
  tabId?: string;
  config?: Record<string, unknown>;
  apiKeys?: Record<string, string>;
  isFavorite?: boolean;
  title?: string;
  latest?: string;
  /** saveMyModel / deleteMyModel: the custom model's name. */
  name?: string;
  /** saveMyModel: OpenAI-compatible base URL for the model. */
  endpoint?: string;
  /** saveMyModel: API key sent to the endpoint. */
  apiKey?: string;
  /** saveMyModel: extra HTTP headers, `Key: Value` one per line. */
  headers?: string;
  /** saveMyModel: the entry's name before an edit-and-rename. */
  originalName?: string;
  restoredTabs?: Array<{
    tabId: string;
    chatId: string;
    title?: string;
    workDir?: string;
  }>;
}
