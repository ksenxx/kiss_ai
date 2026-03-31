/**
 * KISS Sorcar VS Code Extension entry point.
 */

import * as vscode from 'vscode';
import { SorcarViewProvider } from './SorcarPanel';
import { MergeManager } from './MergeManager';
import { ensureDependencies, ensureLocalBinInPath } from './DependencyInstaller';

let primaryProvider: SorcarViewProvider | undefined;
let secondaryProvider: SorcarViewProvider | undefined;
let mergeManager: MergeManager | undefined;
let mergeOwner: SorcarViewProvider | undefined;

function getActiveProvider(): SorcarViewProvider | undefined {
  return secondaryProvider ?? primaryProvider;
}

function getActiveMergeManager(): MergeManager | undefined {
  return getActiveProvider()?.mergeManager ?? mergeManager;
}

export function activate(context: vscode.ExtensionContext): void {
  // Ensure ~/.local/bin is in PATH before any tool lookups or process spawns
  ensureLocalBinInPath();

  console.log('KISS Sorcar extension activating...');

  // Check if VS Code supports secondary sidebar (1.98+)
  const supportsSecondarySidebar = typeof vscode.ViewColumn !== 'undefined';

  mergeManager = new MergeManager();
  mergeManager.on('allDone', () => {
    // X4 fix: route to merge owner, not just active provider
    (mergeOwner ?? getActiveProvider())?.sendMergeAllDone();
    mergeOwner = undefined;
  });
  context.subscriptions.push({ dispose: () => mergeManager?.dispose() });

  const setMergeOwner = (provider: SorcarViewProvider) => { mergeOwner = provider; };

  // Create and register the primary (activitybar) webview provider
  primaryProvider = new SorcarViewProvider(context.extensionUri, mergeManager);
  primaryProvider.mergeOwnerCallback = setMergeOwner;
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'kissSorcar.chatView',
      primaryProvider,
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );

  // Create and register the secondary sidebar webview provider
  secondaryProvider = new SorcarViewProvider(context.extensionUri, mergeManager);
  secondaryProvider.mergeOwnerCallback = setMergeOwner;
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
  let _focusToggling = false;
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(() => {
      _chatFocused = false;
    })
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.toggleFocus', async () => {
      if (_focusToggling) return;
      _focusToggling = true;
      try {
        if (_chatFocused) {
          _chatFocused = false;
          await vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
        } else {
          _chatFocused = true;
          await getActiveProvider()?.focusChatInput();
        }
      } finally {
        _focusToggling = false;
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
    vscode.commands.registerCommand('kissSorcar.newConversation', async () => {
      await vscode.commands.executeCommand('kissSorcar.openPanel');
      const provider = getActiveProvider();
      provider?.newConversation();
      await provider?.focusChatInput();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.runSelection', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const sel = editor.document.getText(editor.selection);
      if (!sel || !sel.trim()) {
        vscode.window.showInformationMessage('No text selected');
        return;
      }
      const provider = getActiveProvider();
      if (!provider) return;
      provider.submitTask(sel.trim());
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

  // VS Code SCM passes (rootUri, context, cancellationToken) to scm/inputBox commands.
  // Returning a Promise makes the sparkle button show a stop/cancel button while pending.
  const triggerCommitMessageGeneration = (
    _rootUri?: unknown,
    _context?: unknown,
    token?: vscode.CancellationToken
  ): Thenable<void> | void => {
    const provider = getActiveProvider();
    if (!provider) return;
    return provider.generateCommitMessage(token);
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
  for (const cmd of ['acceptChange', 'rejectChange', 'prevChange', 'nextChange', 'acceptAll', 'rejectAll', 'acceptFile', 'rejectFile'] as const) {
    context.subscriptions.push(
      vscode.commands.registerCommand(`kissSorcar.${cmd}`, () => {
        getActiveMergeManager()?.[cmd]();
      })
    );
  }

  // Set context for conditional view visibility, then auto-open on startup
  vscode.commands.executeCommand(
    'setContext',
    'kissSorcar:doesNotSupportSecondarySidebar',
    !supportsSecondarySidebar
  ).then(() => {
    vscode.commands.executeCommand('kissSorcar.openPanel');
  });

  // Auto-install dependencies (uv, Python, Playwright Chromium) in background
  ensureDependencies().catch(err => {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[KISS Sorcar] Dependency setup error:', err);
    vscode.window.showErrorMessage(
      `KISS Sorcar: Setup failed — ${msg}. Check ~/.kiss/install.log for details.`
    );
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
