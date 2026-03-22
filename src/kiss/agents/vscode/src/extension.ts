/**
 * KISS Sorcar VS Code Extension entry point.
 */

import * as vscode from 'vscode';
import { SorcarViewProvider } from './SorcarPanel';

let sorcarProvider: SorcarViewProvider | undefined;

export function activate(context: vscode.ExtensionContext): void {
  console.log('KISS Sorcar extension activating...');

  // Create the webview provider
  sorcarProvider = new SorcarViewProvider(context.extensionUri);

  // Register the webview provider for the sidebar view
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      SorcarViewProvider.viewType,
      sorcarProvider,
      {
        webviewOptions: {
          retainContextWhenHidden: true,
        },
      }
    )
  );

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.openPanel', () => {
      // Focus on the sorcar view
      vscode.commands.executeCommand('kissSorcar.chatView.focus');
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.newConversation', () => {
      if (sorcarProvider) {
        sorcarProvider.newConversation();
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.stopTask', () => {
      if (sorcarProvider) {
        sorcarProvider.stopTask();
      }
    })
  );

  console.log('KISS Sorcar extension activated');
}

export function deactivate(): void {
  if (sorcarProvider) {
    sorcarProvider.dispose();
    sorcarProvider = undefined;
  }
  console.log('KISS Sorcar extension deactivated');
}
