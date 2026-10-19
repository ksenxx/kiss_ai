// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// Helpers shared by the settings-sheet suites (settings*.test.js,
// appearanceMenuAndMachineSettings, audit0911_api_key_inventory_parity,
// audit1017SettingsSheetRemoteMachine, config*Toggle*, ...).  They only
// touch the jsdom window and the harness's `posted` array, so a suite
// with its own page bootstrap can use them too.

const assert = require('assert');

/** The last message of `type` the page posted, or null. */
function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

/** Dispatch a bubbling click on `el` in `win`. */
function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** The element with `id`, asserting it exists. */
function byId(win, id) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  return node;
}

/** Click the tab bar's settings gear (asserting it is rendered). */
function openSettings(win) {
  click(win, byId(win, 'settings-btn'));
}

/** Click the settings sheet's close button. */
function closeSettings(win) {
  click(win, byId(win, 'settings-panel-close'));
}

/** The status line `#cfg-<subpanel>-status` of an agent subpanel. */
function statusLine(win, subpanel) {
  return byId(win, `cfg-${subpanel}-status`);
}

module.exports = {
  lastMsg,
  click,
  byId,
  openSettings,
  closeSettings,
  statusLine,
};
