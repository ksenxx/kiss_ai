// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import {AgentClient} from './AgentClient';
import {AgentCommand, Attachment} from './types';

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

  run(fields: RunFields): void {
    this._post({...fields, type: 'run'});
  }

  stop(tabId?: string): void {
    this._post({type: 'stop', tabId});
  }

  appendUserMessage(prompt: string, tabId?: string): void {
    this._post({type: 'appendUserMessage', prompt, tabId});
  }

  userAnswer(answer: string, tabId?: string): void {
    this._post({type: 'userAnswer', answer, tabId});
  }

  resumeSession(fields: {
    chatId?: string;
    taskId?: string | number | null;
    tabId?: string;
  }): void {
    this._post({...fields, type: 'resumeSession'});
  }

  setWorkDir(workDir: string): void {
    this._post({type: 'setWorkDir', workDir});
  }

  selectModel(model: string, tabId?: string): void {
    this._post({type: 'selectModel', model, tabId});
  }

  getModels(): void {
    this._post({type: 'getModels'});
  }

  getInputHistory(): void {
    this._post({type: 'getInputHistory'});
  }

  getConfig(): void {
    this._post({type: 'getConfig'});
  }

  complete(fields: {
    query: string;
    tabId?: string;
    activeFile?: string;
    activeFileContent?: string;
  }): void {
    this._post({...fields, type: 'complete'});
  }

  recordFileUsage(path: string, workDir?: string): void {
    this._post({type: 'recordFileUsage', path, workDir});
  }

  worktreeAction(
    action: 'merge' | 'discard' | 'nothing',
    tabId?: string,
  ): void {
    this._post({type: 'worktreeAction', action, tabId});
  }

  mainTreeAction(
    action: 'discard' | 'nothing',
    tabId?: string,
    workDir?: string,
  ): void {
    this._post({type: 'mainTreeAction', action, tabId, workDir});
  }

  generateCommitMessage(model: string, tabId: string, workDir: string): void {
    this._post({type: 'generateCommitMessage', model, tabId, workDir});
  }

  autocommitAction(tabId?: string, workDir?: string): void {
    this._post({type: 'autocommitAction', tabId, workDir});
  }

  closeTab(tabId: string): void {
    this._post({type: 'closeTab', tabId});
  }

  serverReset(): void {
    this._post({type: 'serverReset'});
  }

  forward(cmd: AgentCommand): void {
    this._post(cmd);
  }

  private _post(cmd: AgentCommand): void {
    this.client.sendCommand(cmd);
  }
}
