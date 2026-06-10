// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Shared access to the built-in ``vscode.git`` extension API.
 */

import * as vscode from 'vscode';

/** Minimal surface of the ``vscode.git`` v1 API used by KISS Sorcar. */
export interface GitApi {
  repositories: Array<{
    inputBox: {value: string};
    state: {indexChanges: unknown[]};
  }>;
  openRepository?: (uri: vscode.Uri) => Promise<unknown>;
}

/**
 * Activate (if needed) the built-in ``vscode.git`` extension and return
 * its v1 API, or ``null`` when the extension is unavailable.
 */
export async function getGitApi(): Promise<GitApi | null> {
  const gitExt = vscode.extensions.getExtension('vscode.git');
  if (!gitExt) return null;
  const git = gitExt.isActive ? gitExt.exports : await gitExt.activate();
  return git.getAPI(1) as GitApi;
}
