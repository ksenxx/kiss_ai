// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The model picker belongs to the user. Each tab shows the model that tab's
// user last picked, and a task launched there runs with that model.
//
// The one exception is a live agent: while a task is running, an agent may
// call `set_model` on itself, and the tabs watching that task show what it is
// actually running -- but only for as long as it runs, and without ever
// overwriting the user's own choice underneath. When the task ends the user's
// pick comes straight back.
//
// This file drives those rules through the real webview: media/chat.html plus
// media/main.js in jsdom, fed the same `models`, `modelPick` and `status`
// events the daemon emits.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const USER_MODEL = 'claude-opus-5';
const AGENT_MODEL = 'gpt-5.6-sol';
const OTHER_MODEL = 'gemini-3.5-flash';

const MODEL_LIST = [
  {name: USER_MODEL, inp: 5, out: 25, uses: 3, vendor: 'anthropic'},
  {name: AGENT_MODEL, inp: 2, out: 8, uses: 0, vendor: 'openai'},
  {name: OTHER_MODEL, inp: 1, out: 4, uses: 0, vendor: 'google'},
];

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, USER_MODEL);
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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=modelpick-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function label(win) {
  return win.document.getElementById('model-name').textContent;
}

/** Deliver the daemon's model list with the user's pick as `selected`. */
function sendModels(win, selected) {
  send(win, {type: 'models', models: MODEL_LIST, selected: selected});
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** Open a second chat tab; returns both ids with `second` on screen. */
function twoTabs(win) {
  const api = win._testApi;
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  assert.ok(second && second !== first, 'a fresh second tab must be active');
  return {api, first, second};
}

/** Open the picker and click *model*. */
function pickFromDropdown(win, model) {
  win.document
    .getElementById('model-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const items = Array.from(
    win.document.querySelectorAll('#model-list .model-item'),
  );
  const wanted = items.find(el => el.textContent.indexOf(model) === 0);
  assert.ok(wanted, 'the model list must offer ' + model);
  wanted.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** Submit a prompt and return the `submit` message the webview posted. */
function submit(win, posted, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  win.document
    .getElementById('send-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const sent = posted.filter(m => m && m.type === 'submit');
  assert.ok(sent.length > 0, 'no submit message was posted');
  return sent[sent.length - 1];
}

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

// ---------------------------------------------------------------------------
// The user's pick
// ---------------------------------------------------------------------------

test('the picker shows the model the daemon reports the user last picked', () => {
  const {win} = makeWebview();

  sendModels(win, OTHER_MODEL);

  assert.strictEqual(label(win), OTHER_MODEL);
  win.close();
});

test('a model list refresh leaves a tab that has its own pick alone', () => {
  const {win, posted} = makeWebview();
  sendModels(win, USER_MODEL);
  const {first, second} = twoTabs(win);

  // The user gives the tab on screen its own model, which the daemon
  // records as its new default; a reconnect re-sends that default.
  pickFromDropdown(win, OTHER_MODEL);
  sendModels(win, OTHER_MODEL);

  assert.strictEqual(
    label(win),
    OTHER_MODEL,
    'a refresh must not take away the model this tab was given',
  );
  const msg = submit(win, posted, 'go');
  assert.strictEqual(
    msg.model,
    OTHER_MODEL,
    'and the task must still run with it',
  );
  clickTab(win, first);
  assert.strictEqual(
    label(win),
    USER_MODEL,
    'the other tab keeps the model it already had',
  );
  clickTab(win, second);
  win.close();
});

test('picking a model from the dropdown tells the daemon and shows it', () => {
  const {win, posted} = makeWebview();
  sendModels(win, USER_MODEL);

  pickFromDropdown(win, OTHER_MODEL);

  assert.strictEqual(label(win), OTHER_MODEL);
  const sel = posted.filter(m => m && m.type === 'selectModel');
  assert.strictEqual(sel.length, 1, 'exactly one selectModel must be sent');
  assert.strictEqual(sel[0].model, OTHER_MODEL);
  win.close();
});

// ---------------------------------------------------------------------------
// The agent's override
// ---------------------------------------------------------------------------

test('a running agent model is shown in the tab running it', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const tabId = win._testApi.getActiveTabId();

  send(win, {type: 'status', running: true, tabId: tabId});
  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: tabId,
  });

  assert.strictEqual(label(win), AGENT_MODEL);
  win.close();
});

test('an agent override never leaks into another tab', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const {first, second} = twoTabs(win);

  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: first,
  });

  assert.strictEqual(
    label(win),
    USER_MODEL,
    'the tab on screen is not the one running the agent',
  );
  clickTab(win, first);
  assert.strictEqual(
    label(win),
    AGENT_MODEL,
    'switching to the running tab must show what its agent is running',
  );
  clickTab(win, second);
  assert.strictEqual(label(win), USER_MODEL, 'and switching back undoes it');
  win.close();
});

test('the agent override is display only: a submit still uses the user model', () => {
  const {win, posted} = makeWebview();
  sendModels(win, USER_MODEL);
  const tabId = win._testApi.getActiveTabId();

  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: tabId,
  });
  send(win, {type: 'status', running: false, tabId: tabId});
  const msg = submit(win, posted, 'do a thing');

  assert.strictEqual(
    msg.model,
    USER_MODEL,
    'the next task must run with the model the USER picked',
  );
  win.close();
});

test('a model list refresh does not blank a live agent override', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const tabId = win._testApi.getActiveTabId();
  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: tabId,
  });

  // Any window reconnecting re-requests the list; the reply must not
  // yank the label away from the agent that is still running.
  sendModels(win, USER_MODEL);

  assert.strictEqual(label(win), AGENT_MODEL);
  win.close();
});

test('picking a model by hand beats a live agent override', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const tabId = win._testApi.getActiveTabId();
  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: tabId,
  });

  pickFromDropdown(win, OTHER_MODEL);

  assert.strictEqual(label(win), OTHER_MODEL);
  win.close();
});

// ---------------------------------------------------------------------------
// Handing the picker back
// ---------------------------------------------------------------------------

test('the finished task hands the picker back to the user', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tabId});
  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: tabId,
  });

  send(win, {
    type: 'modelPick',
    model: USER_MODEL,
    source: 'restore',
    tabId: tabId,
  });

  assert.strictEqual(label(win), USER_MODEL);
  win.close();
});

test('a task that dies without a restore still hands the picker back', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tabId});
  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: tabId,
  });

  // A killed daemon sends the stop but never the restore.
  send(win, {type: 'status', running: false, tabId: tabId});

  assert.strictEqual(label(win), USER_MODEL);
  win.close();
});

test('a background tab whose task ends is restored too', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const {first, second} = twoTabs(win);
  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: first,
  });

  send(win, {
    type: 'modelPick',
    model: USER_MODEL,
    source: 'restore',
    tabId: first,
  });

  clickTab(win, first);
  assert.strictEqual(label(win), USER_MODEL);
  clickTab(win, second);
  win.close();
});

test('an open dropdown follows the model change under it', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const tabId = win._testApi.getActiveTabId();

  // The user is browsing the picker, filtered, when the agent switches.
  win.document
    .getElementById('model-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const search = win.document.getElementById('model-search');
  search.value = 'g';
  search.dispatchEvent(new win.Event('input', {bubbles: true}));
  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: tabId,
  });

  assert.strictEqual(label(win), AGENT_MODEL);
  const shown = Array.from(
    win.document.querySelectorAll('#model-list .model-item'),
  ).map(el => el.textContent);
  assert.ok(
    shown.length > 0 && shown.every(t => t.indexOf('g') >= 0),
    `the typed filter must survive the repaint; got ${JSON.stringify(shown)}`,
  );
  const active = win.document.querySelector('#model-list .model-item.active');
  assert.ok(
    !active || active.textContent.indexOf(USER_MODEL) !== 0,
    'the tick must not still claim the user model is in use',
  );
  win.close();
});

test('a reconnect drops an override that may already be stale', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tabId});
  send(win, {
    type: 'modelPick',
    model: AGENT_MODEL,
    source: 'agent',
    tabId: tabId,
  });

  // The daemon went away and came back; the hand-back may have been
  // sent while this window was not listening.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});

  assert.strictEqual(label(win), USER_MODEL);
  win.close();
});

test('a modelPick with no model is ignored', () => {
  const {win} = makeWebview();
  sendModels(win, USER_MODEL);
  const tabId = win._testApi.getActiveTabId();

  send(win, {type: 'modelPick', model: '', source: 'agent', tabId: tabId});

  assert.strictEqual(label(win), USER_MODEL);
  win.close();
});

console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length > 0 ? 1 : 0);
