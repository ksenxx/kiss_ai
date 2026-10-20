// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) regression tests, driving the real chat.html +
// api.js + main.js, for the settings sheet opened for a registered
// machine other than this daemon (audit 2026-10-17, mainjs-3 F1):
//
// * until that machine's forwarded configData arrives, the form is
//   blank, not this daemon's password / API keys / budget under the
//   other machine's heading;
// * a forwarded error reply is toasted and paints nothing, so the
//   form stays blank and a close saves nothing;
// * a value typed into the blank form is the only thing a close
//   saves, and it goes to that machine;
// * reopening for this daemon right after a remote sheet closed does
//   not show the remote values as this daemon's; reopening for the
//   same machine keeps its values on show;
// * a custom-model edit cancelled before the machine answered restores
//   blank boxes, not the previous machine's endpoint and key.

'use strict';

const assert = require('assert');
const {makePage, send, pickMachine} = require('./simplify2_harness');
const {byId, click} = require('./settings_helpers');

const MACHINES = [
  {name: 'laptop', status: 'reachable', local: true},
  {name: 'gpu-box', status: 'reachable', address: 'ssh me@gpu'},
];

const LAPTOP_CONFIG = {
  type: 'configData',
  config: {
    remote_password: 'laptop-pw',
    max_budget: 3,
    custom_endpoint: 'http://laptop:8080/v1',
    custom_api_key: 'sk-laptop-custom',
  },
  apiKeys: {OPENAI_API_KEY: 'sk-laptop'},
  machine: 'laptop',
};

const GPU_CONFIG = {
  type: 'configData',
  forwarded: true,
  machine: 'gpu-box',
  config: {remote_password: 'gpu-pw', max_budget: 42},
  apiKeys: {OPENAI_API_KEY: 'sk-gpu'},
};

function typeInto(win, id, value) {
  const node = byId(win, id);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

/** The form's text boxes that could leak another machine's values. */
function formValues(win) {
  return {
    password: byId(win, 'cfg-remote-password').value,
    openai: byId(win, 'cfg-key-OPENAI_API_KEY').value,
    budget: byId(win, 'cfg-max-budget').value,
    endpoint: byId(win, 'cfg-custom-endpoint').value,
  };
}

const BLANK = {password: '', openai: '', budget: '', endpoint: ''};
const LAPTOP = {
  password: 'laptop-pw',
  openai: 'sk-laptop',
  budget: '3',
  endpoint: 'http://laptop:8080/v1',
};
const GPU = {password: 'gpu-pw', openai: 'sk-gpu', budget: '42', endpoint: ''};

function msgsOfType(posted, type) {
  return posted.filter(m => m && m.type === type);
}

/** A page showing laptop's settings values, with a chat tab open. */
function pageWithLaptopValues() {
  const page = makePage();
  send(page.win, LAPTOP_CONFIG);
  send(page.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  page.win._testApi.createNewTab();
  assert.deepStrictEqual(formValues(page.win), LAPTOP, 'laptop values on load');
  return page;
}

function errorToast(win) {
  return win.document.querySelector(
    '.kiss-notification[data-notification-id="settings-machine-error"]',
  );
}

function testRemoteSheetStartsBlankAndErrorKeepsItBlank() {
  const {win, posted} = pageWithLaptopValues();
  pickMachine(win, 'gpu-box');
  click(win, byId(win, 'settings-btn'));
  assert.strictEqual(
    byId(win, 'settings-machine').textContent,
    'Settings of gpu-box',
  );
  assert.deepStrictEqual(
    formValues(win),
    BLANK,
    "the sheet for gpu-box does not show laptop's values while waiting",
  );

  // This daemon's own late reply and a stray forwarded error for
  // another machine leave the form alone...
  send(win, LAPTOP_CONFIG);
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'other-box',
    error: 'nope',
  });
  assert.deepStrictEqual(formValues(win), BLANK, 'still blank');
  // ...and gpu-box not answering is toasted, with the form left blank
  // under its heading.
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    error: 'gpu-box unreachable',
  });
  assert.deepStrictEqual(
    formValues(win),
    BLANK,
    'a failed reply paints nothing',
  );
  const toast = errorToast(win);
  assert.ok(toast, 'the failure is toasted');
  assert.ok(
    toast.textContent.includes('gpu-box unreachable'),
    toast.textContent,
  );
  assert.strictEqual(
    byId(win, 'settings-machine').textContent,
    'Settings of gpu-box',
  );

  // The form was never populated: a close must not flush its blanks
  // over gpu-box's password and endpoint.
  const saves = msgsOfType(posted, 'saveConfig').length;
  click(win, byId(win, 'settings-overlay'));
  assert.strictEqual(
    msgsOfType(posted, 'saveConfig').length,
    saves,
    'closing a blank sheet posts no saveConfig',
  );
  win.close();
}

function testTypedValueIsTheOnlySave() {
  const {win, posted} = pageWithLaptopValues();
  pickMachine(win, 'gpu-box');
  click(win, byId(win, 'settings-btn'));
  typeInto(win, 'cfg-remote-password', 'typed-pw');
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    error: 'timed out',
  });
  assert.strictEqual(
    byId(win, 'cfg-remote-password').value,
    'typed-pw',
    "the user's own value survives the failed reply",
  );
  assert.strictEqual(byId(win, 'cfg-key-OPENAI_API_KEY').value, '');

  click(win, byId(win, 'settings-overlay'));
  const saves = msgsOfType(posted, 'saveConfig');
  assert.strictEqual(saves.length, 1, 'the edit is saved on close');
  assert.strictEqual(saves[0].machine, 'gpu-box');
  assert.strictEqual(
    JSON.stringify(saves[0].config),
    JSON.stringify({remote_password: 'typed-pw'}),
    'only the typed field is saved',
  );
  assert.strictEqual(
    JSON.stringify(saves[0].apiKeys),
    '{}',
    'the blank boxes are not saved',
  );
  win.close();
}

function testReopenForThisDaemonDoesNotShowTheRemoteValues() {
  const {win} = pageWithLaptopValues();
  pickMachine(win, 'gpu-box');
  click(win, byId(win, 'settings-btn'));
  send(win, GPU_CONFIG);
  assert.deepStrictEqual(formValues(win), GPU);
  click(win, byId(win, 'settings-overlay'));

  // Reopened for laptop before laptop's configData came back.
  pickMachine(win, 'laptop');
  click(win, byId(win, 'settings-btn'));
  assert.strictEqual(byId(win, 'settings-machine').hidden, true);
  assert.deepStrictEqual(
    formValues(win),
    BLANK,
    "gpu-box's values are not shown as this daemon's",
  );
  send(win, LAPTOP_CONFIG);
  assert.deepStrictEqual(
    formValues(win),
    LAPTOP,
    "laptop's reply fills the form",
  );
  click(win, byId(win, 'settings-overlay'));

  // Reopened for the same machine: its values stay on show.
  click(win, byId(win, 'settings-btn'));
  assert.deepStrictEqual(
    formValues(win),
    LAPTOP,
    'a same-machine reopen keeps the values',
  );
  click(win, byId(win, 'settings-overlay'));
  win.close();
}

function testCancelledModelEditRestoresBlankBoxes() {
  const {win} = pageWithLaptopValues();
  pickMachine(win, 'gpu-box');
  click(win, byId(win, 'settings-btn'));
  // gpu-box's model list arrived, its config did not (yet).
  send(win, {
    type: 'myModelsData',
    forwarded: true,
    machine: 'gpu-box',
    models: [
      {
        name: 'gpu-model',
        endpoint: 'http://gpu:9/v1',
        api_key: 'sk-m',
        headers: '',
      },
    ],
  });
  const editBtn = win.document.querySelector('.custom-model-edit-btn');
  assert.ok(editBtn, "gpu-box's row has an Edit button");
  click(win, editBtn);
  assert.strictEqual(byId(win, 'cfg-custom-endpoint').value, 'http://gpu:9/v1');
  click(win, byId(win, 'custom-model-cancel-btn'));
  assert.strictEqual(
    byId(win, 'cfg-custom-endpoint').value,
    '',
    "cancel restores blank boxes, not laptop's endpoint",
  );
  assert.strictEqual(byId(win, 'cfg-custom-api-key').value, '');
  click(win, byId(win, 'settings-overlay'));
  win.close();
}

testRemoteSheetStartsBlankAndErrorKeepsItBlank();
testTypedValueIsTheOnlySave();
testReopenForThisDaemonDoesNotShowTheRemoteValues();
testCancelledModelEditRestoresBlankBoxes();
console.log('audit1017SettingsSheetRemoteMachine.test.js: all tests passed');
