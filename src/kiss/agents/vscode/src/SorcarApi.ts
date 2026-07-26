// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * SorcarApi — the extension host's typed facade over the Sorcar
 * server API.
 *
 * The ONLY way the VS Code extension talks to the kiss-web daemon.
 * Each method maps 1:1 onto a command of the server API catalog
 * defined in ``src/kiss/server/sorcar.py`` (the single source of
 * truth); the daemon validates every command against that catalog.
 * The transport (one persistent UDS connection, queuing, reconnect)
 * stays in {@link AgentClient} — this class owns all message
 * construction so no caller ever hand-builds a protocol message.
 */

import {AgentClient} from './AgentClient';
import {AgentCommand, Attachment} from './types';

/** Fields accepted by {@link SorcarApi.run}. */
export interface RunFields {
  prompt: string;
  model: string;
  workDir?: string;
  activeFile?: string;
  attachments?: Attachment[];
  useWorktree?: boolean;
  useParallel?: boolean;
  autoCommit?: boolean;
  tabId?: string;
}

export class SorcarApi {
  constructor(private readonly client: AgentClient) {}

  /** Start a task on the daemon. */
  run(fields: RunFields): void {
    this._post({...fields, type: 'run'});
  }

  /** Stop the task running on ``tabId``. */
  stop(tabId?: string): void {
    this._post({type: 'stop', tabId});
  }

  /** Inject a follow-up user message into ``tabId``'s live task. */
  appendUserMessage(prompt: string, tabId?: string): void {
    this._post({type: 'appendUserMessage', prompt, tabId});
  }

  /** Answer the agent's pending ask-user question on ``tabId``. */
  userAnswer(answer: string, tabId?: string): void {
    this._post({type: 'userAnswer', answer, tabId});
  }

  /** Resume a chat session (optionally at a specific task row). */
  resumeSession(fields: {
    chatId?: string;
    taskId?: string | number | null;
    tabId?: string;
  }): void {
    this._post({...fields, type: 'resumeSession'});
  }

  /** Announce this window's workspace folder to the daemon. */
  setWorkDir(workDir: string): void {
    this._post({type: 'setWorkDir', workDir});
  }

  /** Select the model used for new tasks on ``tabId``. */
  selectModel(model: string, tabId?: string): void {
    this._post({type: 'selectModel', model, tabId});
  }

  /** Request the daemon's model list (``models`` event reply). */
  getModels(): void {
    this._post({type: 'getModels'});
  }

  /** Request the prompt input history (``inputHistory`` reply). */
  getInputHistory(): void {
    this._post({type: 'getInputHistory'});
  }

  /** Request the daemon configuration (``configData`` reply). */
  getConfig(): void {
    this._post({type: 'getConfig'});
  }

  /** Request ghost-text completion for the task input. */
  complete(fields: {
    query: string;
    tabId?: string;
    activeFile?: string;
    activeFileContent?: string;
  }): void {
    this._post({...fields, type: 'complete'});
  }

  /** Record a file mention so autocomplete ranks it higher. */
  recordFileUsage(path: string, workDir?: string): void {
    this._post({type: 'recordFileUsage', path, workDir});
  }

  /** Merge or discard ``tabId``'s worktree changes. */
  worktreeAction(action: 'merge' | 'discard', tabId?: string): void {
    this._post({type: 'worktreeAction', action, tabId});
  }

  /** Commit or skip ``tabId``'s pending auto-commit. */
  autocommitAction(
    action: 'commit' | 'skip',
    tabId?: string,
    workDir?: string,
  ): void {
    this._post({type: 'autocommitAction', action, tabId, workDir});
  }

  /** Advance the interactive merge flow for ``tabId``. */
  mergeAction(
    action: 'merge' | 'discard' | 'all-done' | 'commit' | 'skip',
    tabId?: string,
    workDir?: string,
  ): void {
    this._post({type: 'mergeAction', action, tabId, workDir});
  }

  /** Ask the daemon to draft an SCM commit message. */
  generateCommitMessage(model: string, tabId: string, workDir: string): void {
    this._post({type: 'generateCommitMessage', model, tabId, workDir});
  }

  /** Tell the daemon a webview tab was closed. */
  closeTab(tabId: string): void {
    this._post({type: 'closeTab', tabId});
  }

  /** Restart the kiss-web daemon. */
  serverReset(): void {
    this._post({type: 'serverReset'});
  }

  /**
   * Forward a webview-built command unchanged.  Used for the
   * whitelisted pass-through commands (``FORWARDED_COMMANDS``) whose
   * payloads the webview's own API facade (``media/api.js``) already
   * validated and constructed.
   */
  forward(cmd: AgentCommand): void {
    this._post(cmd);
  }

  private _post(cmd: AgentCommand): void {
    this.client.sendCommand(cmd);
  }
}
