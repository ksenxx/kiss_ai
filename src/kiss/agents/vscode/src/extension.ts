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

  let _chatFocused = false;
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.toggleFocus', async () => {
      if (_chatFocused) {
        _chatFocused = false;
        await vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
      } else {
        _chatFocused = true;
        await getActiveProvider()?.focusChatInput();
      }
    })
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.focusEditor', () => {
      _chatFocused = false;
      vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.newConversation', () => {
      vscode.commands.executeCommand('kissSorcar.openPanel');
      getActiveProvider()?.newConversation();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.stopTask', () => {
      getActiveProvider()?.stopTask();
    })
  );

  // Commit message generation — sets the Git SCM input box
  const setScmMessage = async (message: string) => {
    try {
      const gitExt = vscode.extensions.getExtension('vscode.git');
      if (gitExt) {
        const git = gitExt.isActive ? gitExt.exports : await gitExt.activate();
        const api = git.getAPI(1);
        if (api.repositories.length > 0) {
          api.repositories[0].inputBox.value = message;
          vscode.commands.executeCommand('workbench.view.scm');
        }
      }
    } catch (err) {
      console.error('[kissSorcar] Failed to set SCM message:', err);
    }
  };

  // Listen for commitMessage events from both providers
  for (const provider of [primaryProvider, secondaryProvider]) {
    if (provider) {
      context.subscriptions.push(
        provider.onCommitMessage((ev) => {
          if (ev.error) {
            vscode.window.showWarningMessage(`Commit message: ${ev.error}`);
          } else if (ev.message) {
            setScmMessage(ev.message);
          }
        })
      );
    }
  }

  const triggerCommitMessageGeneration = () => {
    const provider = getActiveProvider();
    if (!provider) return;
    provider.generateCommitMessage();
  };

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.generateCommitMessage', triggerCommitMessageGeneration)
  );

  // Try to take over common commit-message commands so the SCM sparkle
  // button uses Gemini instead of Copilot.  registerCommand throws if the
  // command is already registered, so we silently ignore failures.
  for (const cmdId of [
    'github.copilot.git.generateCommitMessage',
    'git.generateCommitMessage',
  ]) {
    try {
      context.subscriptions.push(
        vscode.commands.registerCommand(cmdId, triggerCommitMessageGeneration)
      );
    } catch {
      // Already registered by another extension — ignored
    }
  }

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
