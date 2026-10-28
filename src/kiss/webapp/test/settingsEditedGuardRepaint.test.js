// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the OTHER half of the
// ``settingsEditedFields`` guard.
//
// ``settingsPanelStaleConfig.test.js`` proves the guard does its job:
// a field the user is editing is not painted over by a ``configData``
// that lands mid-edit.  A guard like that has a natural failure mode in
// the opposite direction -- marks that outlive the edit and then
// suppress a value the server legitimately changed, leaving the panel
// showing a number the daemon no longer holds and re-saving it on
// close.
//
// The marks are dropped in two places (``openSettingsPanel`` and
// ``closeSettingsPanel``).
//
// These tests pin the release valve rather than the guard.

'use strict';

const assert = require('assert');
const {makePage, send, lastMsg} = require('./simplify2_harness');
const {openSettings, closeSettings, typeInto} = require('./settings_helpers');

function valueOf(win, id) {
  return win.document.getElementById(id).value;
}

const STORED = {
  max_budget: 100,
  custom_endpoint: 'https://one.example/v1',
  custom_api_key: '',
  custom_headers: 'X-One: 1',
  remote_password: 'pw-one',
};

// A push that arrives with the panel shut belongs to nobody's edit, so
// it must land -- otherwise the next time the panel opens it shows a
// value the daemon has not held for minutes.
function testServerChangeWhilePanelClosedRepaints() {
  const {win} = makePage();
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {}});
  typeInto(win, 'cfg-custom-headers', 'X-Mine: 2');
  closeSettings(win);

  // Another browser window saves the same setting and the daemon
  // pushes the new configData.
  send(win, {
    type: 'configData',
    config: {...STORED, custom_headers: 'X-Other: 9'},
    apiKeys: {},
  });
  assert.strictEqual(
    valueOf(win, 'cfg-custom-headers'),
    'X-Other: 9',
    'a field edited in a FINISHED session must accept the next server ' +
      'value: closeSettingsPanel() drops the edited marks, so nothing ' +
      'may still be suppressing the repaint',
  );
  win.close();
  console.log('  ok - a server change with the panel closed repaints');
}

// Reopening asks for the config again; that reply must win even though
// the user never touched the field in the NEW session.
function testReopenedPanelShowsTheServerValueNotTheOldEdit() {
  const {win, posted} = makePage();
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {}});
  typeInto(win, 'cfg-custom-endpoint', 'https://mine.example/v1');
  closeSettings(win);

  openSettings(win);
  assert.ok(
    lastMsg(posted, 'getConfig'),
    'reopening the panel must re-request the config',
  );
  // The daemon rejected/normalised the edit and reports something else.
  send(win, {
    type: 'configData',
    config: {...STORED, custom_endpoint: 'https://server.example/v1'},
    apiKeys: {},
  });
  assert.strictEqual(
    valueOf(win, 'cfg-custom-endpoint'),
    'https://server.example/v1',
    'a reopened panel must show what the daemon actually holds, not the ' +
      'value this window typed in the previous session',
  );

  // ...and closing it must not resurrect the stale edit: nothing was
  // touched in this session, so nothing is sent (the daemon keeps what
  // it holds).
  const before = posted.length;
  closeSettings(win);
  const save = posted
    .slice(before)
    .filter(m => m && m.type === 'saveConfig')
    .pop();
  assert.strictEqual(
    save,
    undefined,
    'closing an untouched reopened panel must not save the previous ' +
      "session's edit",
  );
  win.close();
  console.log('  ok - a reopened panel shows the server value and saves nothing');
}

// The run toggles are seeded by configData for every session, so a mark
// stuck on one of them would silently run tasks with the wrong flags.
function testToggleMarksAreReleasedBetweenSessions() {
  const {win} = makePage();
  const doc = win.document;
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {}});

  const wt = doc.getElementById('cfg-use-web-tools');
  wt.checked = false;
  wt.dispatchEvent(new win.Event('change', {bubbles: true}));
  closeSettings(win);

  // The daemon reports web tools back ON (another window re-enabled it).
  send(win, {
    type: 'configData',
    config: {...STORED, use_web_browser: true},
    apiKeys: {},
  });
  assert.strictEqual(
    wt.checked,
    true,
    'a toggle edited in a finished session must accept the next server ' +
      'value; a stuck mark would run every later task with the wrong flag',
  );
  win.close();
  console.log('  ok - toggle marks are released between sessions');
}

function main() {
  testServerChangeWhilePanelClosedRepaints();
  testReopenedPanelShowsTheServerValueNotTheOldEdit();
  testToggleMarksAreReleasedBetweenSessions();
  console.log('settingsEditedGuardRepaint.test.js: all tests passed');
}

main();
