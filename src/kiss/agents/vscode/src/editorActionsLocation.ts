// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Relocates the editor-actions toolbar — the strip that hosts the four
// Sorcar editor-title buttons (+ new chat, git commit, settings gear,
// KS history) — from the top-right of the editor window into the
// window title bar above it, whenever editor-tabs mode is on.
//
// VS Code offers no extension contribution point for the window title
// bar (the `menus` contribution targets editor/title, view/title,
// scm/title, ...). The one supported relocation mechanism is the user
// setting `workbench.editor.editorActionsLocation`: the value
// "titleBar" renders the editor actions in the title bar's right-hand
// toolbar (next to the layout controls) instead of the editor tab bar.
//
// The setting is written at the GLOBAL (user) scope — it is a
// per-user UI preference, and editor-tabs mode itself defaults to on
// globally. The extension never reverts it in deactivate(): shutdown
// hooks cannot reliably await configuration writes, and a window
// closing must not strip the title-bar buttons from the user's other
// windows. The restore instead happens when the user turns the mode
// off (or disables the custom title bar) in a running window.

import * as vscode from 'vscode';

const EDITOR_ACTIONS_SETTING = 'workbench.editor.editorActionsLocation';

/**
 * globalState key remembering the user's own global value of
 * `workbench.editor.editorActionsLocation` from before this extension
 * moved the actions to the title bar. The record's shape is
 * `{prior: string | null}` (null = the user had no global override);
 * its very presence means "the title-bar value is ours to undo".
 */
export const PRIOR_LOCATION_KEY = 'kissSorcar.priorEditorActionsLocation';

// Chain serializing the syncs: a mode flip immediately followed by
// another must apply its configuration reads and writes strictly
// after the first finished, or the interleaved awaits could leave the
// setting reflecting the OLDER of the two states. Errors are logged
// and swallowed so one failed write never wedges the chain (or
// surfaces as an unhandled rejection from the fire-and-forget call
// sites in extension.ts).
let syncChain: Promise<void> = Promise.resolve();

/**
 * Queue an alignment of `workbench.editor.editorActionsLocation` with
 * editor-tabs mode: ON moves the editor actions (and with them the
 * four Sorcar buttons) into the window title bar, OFF restores the
 * user's own value — but only if the title-bar move was made by this
 * extension and the user has not overridden it since.
 *
 * The move is also undone (and re-done) as
 * `window.customTitleBarVisibility` flips through "never": VS Code
 * then HIDES title-bar editor actions instead of falling back to the
 * tab bar, which would make the buttons disappear entirely.
 *
 * Calls are serialized in arrival order, so the state of the last
 * call always wins.
 *
 * @param context The extension context whose globalState remembers the
 *     user's prior setting across sessions.
 * @param editorTabsModeOn Current state of kissSorcar.editorTabsMode.
 * @returns Settles when this call's sync (and every earlier one) is
 *     done; never rejects.
 */
export function syncEditorActionsLocation(
  context: vscode.ExtensionContext,
  editorTabsModeOn: boolean,
): Promise<void> {
  syncChain = syncChain
    .then(() => applyEditorActionsLocation(context, editorTabsModeOn))
    .catch(err => {
      console.error('KISS Sorcar: editor-actions location sync failed', err);
    });
  return syncChain;
}

/**
 * Perform one sync (see syncEditorActionsLocation for the contract).
 *
 * @param context The extension context holding the restore record.
 * @param editorTabsModeOn Current state of kissSorcar.editorTabsMode.
 */
async function applyEditorActionsLocation(
  context: vscode.ExtensionContext,
  editorTabsModeOn: boolean,
): Promise<void> {
  // Guarded like the other optional host APIs (test stubs may not
  // model the configuration surface); the sync is then a no-op.
  if (typeof vscode.workspace?.getConfiguration !== 'function') return;
  const cfg = vscode.workspace.getConfiguration();
  if (typeof cfg?.inspect !== 'function' || typeof cfg?.update !== 'function') {
    return;
  }

  const globalValue = cfg.inspect<string>(EDITOR_ACTIONS_SETTING)?.globalValue;
  const wantTitleBar =
    editorTabsModeOn &&
    cfg.get<string>('window.customTitleBarVisibility') !== 'never';

  if (wantTitleBar) {
    // Already in the title bar (by us in an earlier session, or by the
    // user's own hand) — nothing to move, and any existing restore
    // record must survive for the eventual switch back.
    if (globalValue === 'titleBar') return;
    await context.globalState.update(PRIOR_LOCATION_KEY, {
      prior: globalValue ?? null,
    });
    await cfg.update(
      EDITOR_ACTIONS_SETTING,
      'titleBar',
      vscode.ConfigurationTarget.Global,
    );
    return;
  }

  const record = context.globalState.get<{prior: string | null}>(
    PRIOR_LOCATION_KEY,
  );
  // No record: the extension never moved the actions, so whatever the
  // user configured (including a deliberate "titleBar") stays put.
  if (!record) return;
  // Restore only while our value is still in effect; if the user has
  // meanwhile picked something else, their choice wins.
  if (globalValue === 'titleBar') {
    await cfg.update(
      EDITOR_ACTIONS_SETTING,
      record.prior ?? undefined,
      vscode.ConfigurationTarget.Global,
    );
  }
  await context.globalState.update(PRIOR_LOCATION_KEY, undefined);
}
