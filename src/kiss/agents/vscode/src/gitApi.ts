// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';

export interface GitApi {
  repositories: Array<{
    inputBox: {value: string};
    state: {indexChanges: unknown[]};
  }>;
  openRepository?: (uri: vscode.Uri) => Promise<unknown>;
}

export async function getGitApi(): Promise<GitApi | null> {
  const gitExt = vscode.extensions.getExtension('vscode.git');
  if (!gitExt) return null;
  const git = gitExt.isActive ? gitExt.exports : await gitExt.activate();
  return git.getAPI(1) as GitApi;
}
