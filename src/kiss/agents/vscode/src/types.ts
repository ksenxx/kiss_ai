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
  | {type: 'worktreeAction'; action: 'merge' | 'discard'; tabId?: string}
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
  | {type: 'clear'; chat_id?: number}
  | {type: 'showWelcome'}
  | {type: 'clearChat'}
  | {type: 'task_done'}
  | {type: 'task_error'; text: string}
  | {type: 'task_stopped'}
  | {type: 'task_interrupted'}
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
  | {type: 'worktree_result'; success: boolean; message: string}
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
  | {
      // Canonical shared-tab snapshot broadcast by the daemon after
      // every tab-registry mutation; clients reconcile against it.
      type: 'tabs_state';
      tabs: Array<{
        tabId: string;
        chatId: string;
        title: string;
        workDir: string;
      }>;
    }
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
  taskId?: string | number | null;
  activeFileContent?: string;
  action?: 'merge' | 'discard';
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
  restoredTabs?: Array<{
    tabId: string;
    chatId: string;
    title?: string;
    workDir?: string;
  }>;
}
