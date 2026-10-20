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

const {click, byId} = require('./simplify2_harness');

/** The last message of `type` the page posted, or null. */
function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

/** Click the tab bar's settings gear (asserting it is rendered). */
function openSettings(win) {
  click(win, byId(win, 'settings-btn'));
}

/** Click the settings sheet's close button. */
function closeSettings(win) {
  click(win, byId(win, 'settings-panel-close'));
}

/** Set the value of the input `#id` and fire its `input` event. */
function typeInto(win, id, value) {
  const node = byId(win, id);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
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
  typeInto,
  statusLine,
};
