/**
 * KISS Sorcar VS Code Extension entry point.
 */

import * as vscode from 'vscode';
import { SorcarViewProvider } from './SorcarPanel';

let primaryProvider: SorcarViewProvider | undefined;
let secondaryProvider: SorcarViewProvider | undefined;

function getActiveProvider(): SorcarViewProvider | undefined {
  return secondaryProvider ?? primaryProvider;
}

export function activate(context: vscode.ExtensionContext): void {
  console.log('KISS Sorcar extension activating...');

  // Check if VS Code supports secondary sidebar (1.98+)
  const supportsSecondarySidebar = typeof vscode.ViewColumn !== 'undefined';

  // Create and register the primary (activitybar) webview provider
  primaryProvider = new SorcarViewProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'kissSorcar.chatView',
      primaryProvider,
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );

  // Create and register the secondary sidebar webview provider
  secondaryProvider = new SorcarViewProvider(context.extensionUri);
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
      // Try secondary sidebar first, then fall back to primary
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

  // Auto-open the chat view on startup
  vscode.commands.executeCommand(
    'setContext',
    'kissSorcar:doesNotSupportSecondarySidebar',
    !supportsSecondarySidebar
  ).then(() => {
    const viewId = supportsSecondarySidebar
      ? 'kissSorcar.chatViewSecondary.focus'
      : 'kissSorcar.chatView.focus';
    vscode.commands.executeCommand(viewId);
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
