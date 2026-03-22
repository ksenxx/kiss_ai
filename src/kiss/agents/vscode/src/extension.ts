/**
 * KISS Sorcar VS Code Extension entry point.
 */

import * as vscode from 'vscode';
import { SorcarViewProvider } from './SorcarPanel';
import { MergeManager } from './MergeManager';

let primaryProvider: SorcarViewProvider | undefined;
let secondaryProvider: SorcarViewProvider | undefined;
let mergeManager: MergeManager | undefined;

function getActiveProvider(): SorcarViewProvider | undefined {
  return secondaryProvider ?? primaryProvider;
}

function getActiveMergeManager(): MergeManager | undefined {
  return getActiveProvider()?.mergeManager ?? mergeManager;
}

export function activate(context: vscode.ExtensionContext): void {
  console.log('KISS Sorcar extension activating...');

  // Check if VS Code supports secondary sidebar (1.98+)
  const supportsSecondarySidebar = typeof vscode.ViewColumn !== 'undefined';

  mergeManager = new MergeManager();
  context.subscriptions.push({ dispose: () => mergeManager?.dispose() });

  // Create and register the primary (activitybar) webview provider
  primaryProvider = new SorcarViewProvider(context.extensionUri, mergeManager);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'kissSorcar.chatView',
      primaryProvider,
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );

  // Create and register the secondary sidebar webview provider
  secondaryProvider = new SorcarViewProvider(context.extensionUri, mergeManager);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'kissSorcar.chatViewSecondary',
      secondaryProvider,
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.openPanel', () => {
      vscode.commands.executeCommand('kissSorcar.chatViewSecondary.focus').then(
        undefined,
        () => vscode.commands.executeCommand('kissSorcar.chatView.focus')
      );
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.newConversation', () => {
      getActiveProvider()?.newConversation();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.stopTask', () => {
      getActiveProvider()?.stopTask();
    })
  );

  // Merge commands
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.acceptChange', () => {
      getActiveMergeManager()?.acceptChange();
    }),
    vscode.commands.registerCommand('kissSorcar.rejectChange', () => {
      getActiveMergeManager()?.rejectChange();
    }),
    vscode.commands.registerCommand('kissSorcar.prevChange', () => {
      getActiveMergeManager()?.prevChange();
    }),
    vscode.commands.registerCommand('kissSorcar.nextChange', () => {
      getActiveMergeManager()?.nextChange();
    }),
    vscode.commands.registerCommand('kissSorcar.acceptAll', () => {
      getActiveMergeManager()?.acceptAll();
    }),
    vscode.commands.registerCommand('kissSorcar.rejectAll', () => {
      getActiveMergeManager()?.rejectAll();
    })
  );

  // Set context for conditional view visibility, then auto-open on startup
  vscode.commands.executeCommand(
    'setContext',
    'kissSorcar:doesNotSupportSecondarySidebar',
    !supportsSecondarySidebar
  ).then(() => {
    vscode.commands.executeCommand('kissSorcar.openPanel');
  });

  console.log('KISS Sorcar extension activated');
}

export function deactivate(): void {
  primaryProvider?.dispose();
  primaryProvider = undefined;
  secondaryProvider?.dispose();
  secondaryProvider = undefined;
  console.log('KISS Sorcar extension deactivated');
}
