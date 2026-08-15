// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests: the client owns no default budget.
//
// The product default lives in exactly one place, `kiss.core.config
// .DEFAULT_MAX_BUDGET`, which `vscode_config.DEFAULTS['max_budget']`
// reads and `load_config()` seeds every reply from -- so the effective
// value is always present in the `configData` the daemon sends.
//
// main.js used to keep its own copy of the number, twice:
//
//     setValue('cfg-max-budget', cfg.max_budget != null ? cfg.max_budget : 100)
//     cfg.max_budget = parseFloat(el('cfg-max-budget').value) || 100
//
// plus a third in chat.html's `value="100"`. They agreed with Python by
// coincidence, and the second one is not merely cosmetic: it is a
// WRITE. Whenever the field did not parse -- the user cleared it, or the
// reply had not landed -- closing the settings panel saved the client's
// guess over whatever the daemon actually held. A user on a 250 budget
// who cleared the box to retype it was silently put back on 100.
//
// The rule these tests pin: the client shows the server's number and
// says nothing when it has none. `vscode_config.save_config` MERGES the
// payload it receives, so omitting the key leaves the stored budget
// exactly as it was -- which is the only honest thing a client with no
// opinion can do.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

function openSettings(win) {
  const gear = win.document.querySelector('.chat-tab-settings');
  assert.ok(gear, 'the tab bar must render the settings gear');
  gear.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function closeSettings(win) {
  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function typeInto(win, id, value) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function budgetField(win) {
  return win.document.getElementById('cfg-max-budget').value;
}

// A daemon whose default has moved on (or a user who set their own).
const STORED = {
  max_budget: 250,
  custom_endpoint: '',
  custom_api_key: '',
  custom_headers: '',
  remote_password: '',
  auto_commit_mode: true,
  is_worktree: true,
};

// Nothing may appear in the box before the daemon has said anything:
// whatever number the client could put there would be its own, and the
// client has none.
function testNoBudgetIsShownBeforeTheConfigArrives() {
  const {win} = makeWebview();
  openSettings(win);
  assert.strictEqual(
    budgetField(win),
    '',
    'before configData lands the budget box must be empty rather than ' +
      'showing a number main.js/chat.html made up',
  );
  win.close();
  console.log('  ok - no budget is shown before the config arrives');
}

function testServerValueIsShownAndSavedBack() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {}});
  assert.strictEqual(
    budgetField(win),
    '250',
    "the box must show the daemon's effective budget",
  );

  closeSettings(win);
  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing a populated panel must save the form');
  assert.strictEqual(
    save.config.max_budget,
    250,
    'an untouched budget must round-trip unchanged',
  );
  win.close();
  console.log('  ok - the server value is shown and round-trips');
}

// The regression: the client used to answer an unparseable box with its
// own 100 and write it, quietly demoting a 250 budget.
function testClearingTheBoxDoesNotWriteAClientDefault() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {}});

  // The user selects the box and deletes it, meaning to retype -- and
  // then closes the panel instead.
  typeInto(win, 'cfg-max-budget', '');
  closeSettings(win);

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing a populated panel must save the form');
  assert.ok(
    !('max_budget' in save.config),
    'an empty budget box means "no opinion", not "100": the daemon ' +
      'merges what it is sent, so leaving the key out is what keeps the ' +
      `stored 250. Got ${JSON.stringify(save.config.max_budget)}`,
  );
  win.close();
  console.log('  ok - clearing the box writes no client-side default');
}

// Same write, reached the other way: a reply that carries no budget at
// all must not cause one to be invented on the next save.
function testAConfigWithoutABudgetInventsNone() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {
    type: 'configData',
    config: {custom_endpoint: '', custom_headers: ''},
    apiKeys: {},
  });
  assert.strictEqual(
    budgetField(win),
    '',
    'a configData without max_budget must leave the box empty, not ' +
      'fill in a number of the client\u2019s own',
  );

  // The user edits something else entirely and closes.
  typeInto(win, 'cfg-custom-headers', 'X-Trace: 1');
  closeSettings(win);

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing a populated panel must save the form');
  assert.ok(
    !('max_budget' in save.config),
    'editing an unrelated field must not smuggle a client-side budget ' +
      `into the saved config. Got ${JSON.stringify(save.config)}`,
  );
  win.close();
  console.log('  ok - a config without a budget invents none');
}

// `parseFloat(v) || 100` also swallowed a legitimate zero -- the one
// value a user might type to mean "stop immediately".
function testTypedValuesAreSavedVerbatim() {
  [
    ['42', 42],
    ['0', 0],
    ['12.5', 12.5],
  ].forEach(([typed, expected]) => {
    const {win, posted} = makeWebview();
    openSettings(win);
    send(win, {type: 'configData', config: STORED, apiKeys: {}});
    typeInto(win, 'cfg-max-budget', typed);
    closeSettings(win);

    const save = lastMsg(posted, 'saveConfig');
    assert.ok(save, 'closing a populated panel must save the form');
    assert.strictEqual(
      save.config.max_budget,
      expected,
      `a budget the user typed must be saved as typed; ${typed} must not ` +
        'be rewritten by the client',
    );
    win.close();
  });
  console.log('  ok - typed budgets (0 included) are saved verbatim');
}

function main() {
  testNoBudgetIsShownBeforeTheConfigArrives();
  testServerValueIsShownAndSavedBack();
  testClearingTheBoxDoesNotWriteAClientDefault();
  testAConfigWithoutABudgetInventsNone();
  testTypedValuesAreSavedVerbatim();
  console.log('settingsMaxBudgetDefault.test.js: all tests passed');
}

main();
