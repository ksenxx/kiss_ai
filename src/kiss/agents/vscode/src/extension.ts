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

  // Temporary test command for file picker
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.testFilePicker', () => {
      if (sorcarProvider) {
        // Inject @ into the textarea and show file picker via webview eval
        sorcarProvider.sendToWebview({
          type: 'test_file_picker',
          files: [
            { type: 'frequent', text: 'src/main.ts' },
            { type: 'frequent', text: 'pyproject.toml' },
            { type: 'file', text: 'test/file1.py' },
            { type: 'file', text: 'README.md' },
            { type: 'file', text: 'src/kiss/agents/vscode/media/main.js' },
            { type: 'file', text: 'src/kiss/agents/vscode/media/main.css' },
          ]
        } as any);
      }
    })
  );

  // Temporary test command for model picker
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.testModelPicker', () => {
      if (sorcarProvider) {
        sorcarProvider.sendToWebview({
          type: 'test_model_picker',
          models: [
            { name: 'claude-opus-4-6', inp: 15.0, out: 75.0, uses: 12, vendor: 'Anthropic' },
            { name: 'claude-sonnet-4-20250514', inp: 3.0, out: 15.0, uses: 5, vendor: 'Anthropic' },
            { name: 'gpt-4o', inp: 5.0, out: 15.0, uses: 0, vendor: 'OpenAI' },
            { name: 'gpt-4o-mini', inp: 0.15, out: 0.6, uses: 0, vendor: 'OpenAI' },
            { name: 'gemini-2.0-flash', inp: 0.1, out: 0.4, uses: 0, vendor: 'Gemini' },
            { name: 'o4-mini', inp: 1.10, out: 4.40, uses: 3, vendor: 'OpenAI' },
          ],
          selected: 'claude-opus-4-6'
        } as any);
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
