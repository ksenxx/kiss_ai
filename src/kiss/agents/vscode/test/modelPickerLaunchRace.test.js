// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: on a fresh VS Code launch the chat
// webview must not silently drop the model name that the extension
// substituted into the ``{{MODEL_NAME}}`` template placeholder.
//
// Two races collude to make the picker blank shortly after launch:
//
//   1. The IIFE that builds ``tabs`` (around line 1235 of main.js)
//      runs BEFORE the script reads ``#model-name`` from the DOM
//      (around line 1297) into the closure variable ``selectedModel``.
//      So every tab created during init records
//      ``selectedModel: ''`` regardless of the template value.
//
//   2. The ``case 'models':`` handler updates the global
//      ``selectedModel`` plus the ``#model-name`` text node, but
//      never propagates the new value into ``tab.selectedModel``.
//      Switching to any tab then calls ``restoreTab`` which runs
//      ``selectedModel = tab.selectedModel || ''`` and writes ``''``
//      into ``#model-name`` — the picker reverts to blank, even
//      though the daemon already broadcast a real model name.
//
// Either race alone is enough to make the picker render blank after
// the first tab switch following launch.  The fix:
//
//   * Move the ``selectedModel = #model-name.textContent`` read so it
//     runs BEFORE the tab-restore IIFE — every tab created on launch
//     now starts with the template value rather than ``''``.
//   * In ``case 'models':``, also propagate the new ``ev.selected``
//     into every tab that still holds the prior default — preserve
//     tabs where the user explicitly picked a different model.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/modelPickerLaunchRace.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the real chat webview, substituting
 * *templateModel* into the ``{{MODEL_NAME}}`` placeholder so the test
 * can drive both the "No model" launch case and the real-model case.
 *
 * *seededState* (optional) is what ``vscode.getState()`` returns on
 * the first call before the webview ever persists, so the test can
 * simulate the cross-restart restore path that creates multiple tabs
 * during the launch IIFE.
 */
function makeWebview(templateModel, seededState) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, templateModel);
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
    let state = seededState;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Click a real tab-bar element to drive the production switchToTab flow. */
function clickTab(win, tabId) {
  const tabEl = win.document.querySelector(
    `.chat-tab[data-tab-id="${tabId}"]`,
  );
  assert.ok(tabEl, `tab ${tabId} must exist in the tab bar`);
  tabEl.click();
}

/** Return the textContent of the model picker label. */
function pickerText(win) {
  const el = win.document.getElementById('model-name');
  return (el && el.textContent) || '';
}

// ---------------------------------------------------------------------
// Race 1: tab init IIFE runs before the DOM-read of #model-name.  When
// VS Code restores multiple tabs from the previous session, every
// restored tab gets ``selectedModel: ''`` because the closure variable
// is still empty when ``makeTab`` runs.  Switching to any non-active
// restored tab then drives the picker blank — even though the active
// tab and the daemon both have a valid model name.
// ---------------------------------------------------------------------
function testRestoredTabInheritsTemplateModelOnLaunch() {
  // Mimic VS Code restoring two persisted tabs (the user had a chat
  // open in tab A and tab B before reload).
  const seeded = {
    tabs: [
      {title: 'first', chatId: 'persisted-tab-a', backendChatId: ''},
      {title: 'second', chatId: 'persisted-tab-b', backendChatId: ''},
    ],
    activeTabIndex: 0,
    chatId: 'persisted-tab-a',
  };
  const {win} = makeWebview('claude-opus-4-7', seeded);

  // Sanity: on launch the picker shows the substituted template
  // value (this is what the user sees pre-daemon-reply).
  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'launch-time picker must reflect the {{MODEL_NAME}} template',
  );

  // User clicks the OTHER restored tab.  ``switchToTab`` calls
  // ``saveCurrentTab`` (which saves the active tab's live state)
  // and then ``restoreTab(other)`` which writes
  // ``other.selectedModel || ''`` into the picker.  If the restored
  // tab stored '' during launch the picker turns blank.
  clickTab(win, 'persisted-tab-b');
  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'BUG: model picker turned blank after switching to the second ' +
      'restored tab — the launch IIFE ran before the DOM-read of ' +
      '#model-name, so every restored tab inherited an empty ' +
      'selectedModel from the closure variable',
  );

  win.close();
  console.log(
    '  ok - restored tabs inherit template model on launch',
  );
}

// ---------------------------------------------------------------------
// Race 2: ``case 'models':`` updates the global selectedModel but not
// the tab's stored selectedModel, so switching tabs reverts the picker.
// ---------------------------------------------------------------------
function testModelsEventPropagatesIntoTabState() {
  // Simulate the typical fresh-install flow: the extension could not
  // resolve a real default model synchronously (env vars not visible
  // to the activation process) so MODEL_NAME was substituted with
  // ``No model``; the daemon later replies with the real default;
  // the user had multiple tabs persisted from the previous session.
  const seeded = {
    tabs: [
      {title: 'first', chatId: 'persisted-tab-a', backendChatId: ''},
      {title: 'second', chatId: 'persisted-tab-b', backendChatId: ''},
    ],
    activeTabIndex: 0,
    chatId: 'persisted-tab-a',
  };
  const {win} = makeWebview('No model', seeded);

  assert.strictEqual(pickerText(win), 'No model');

  // Daemon broadcasts ``models`` with a real default.  Pricing fields
  // are required by the dropdown renderer (it calls .toFixed on them).
  send(win, {
    type: 'models',
    models: [
      {name: 'claude-opus-4-7', inp: 15, out: 75, uses: 0, vendor: 'Anthropic'},
      {name: 'gpt-5.5', inp: 5, out: 25, uses: 0, vendor: 'OpenAI'},
    ],
    selected: 'claude-opus-4-7',
  });
  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'models event must update the live picker label',
  );

  // The user switches to a sibling restored tab.  Every restored
  // tab must reflect the post-models default — they all started out
  // with ``selectedModel: ''`` and were never user-picked, so the
  // ``case 'models':`` handler must propagate ``ev.selected`` into
  // their state.  Without this, ``restoreTab`` reverts the picker
  // to ``''``.
  clickTab(win, 'persisted-tab-b');
  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'BUG: tab state was not updated by the models event; ' +
      'switching tabs reverts the picker to the stale launch-time value',
  );

  win.close();
  console.log(
    '  ok - models event propagates new selection into existing tab state',
  );
}

// ---------------------------------------------------------------------
// Sanity: a tab where the user has explicitly picked a different
// model must NOT be overwritten by a later ``models`` event.
// ---------------------------------------------------------------------
function testModelsEventPreservesUserPickedTabs() {
  // Seed two restored tabs so the test can simulate "user picks
  // gpt-5.5 in tab A; daemon later broadcasts claude-opus-4-7;
  // tab A must keep gpt-5.5".
  const seeded = {
    tabs: [
      {title: 'first', chatId: 'persisted-tab-a', backendChatId: ''},
      {title: 'second', chatId: 'persisted-tab-b', backendChatId: ''},
    ],
    activeTabIndex: 0,
    chatId: 'persisted-tab-a',
  };
  const {win, posted} = makeWebview('claude-opus-4-7', seeded);

  // Models that include the pricing fields the dropdown renderer
  // expects (the renderer calls ``.toFixed()`` on inp/out).
  const modelList = [
    {name: 'claude-opus-4-7', inp: 15, out: 75, uses: 0, vendor: 'Anthropic'},
    {name: 'gpt-5.5', inp: 5, out: 25, uses: 0, vendor: 'OpenAI'},
  ];

  send(win, {type: 'models', models: modelList, selected: 'claude-opus-4-7'});

  // Pick gpt-5.5 from the dropdown to drive the production
  // ``selectModel`` flow on the active (first) tab.
  const modelItems = win.document.querySelectorAll('#model-list .model-item');
  let picked = null;
  modelItems.forEach(it => {
    if ((it.textContent || '').includes('gpt-5.5')) picked = it;
  });
  assert.ok(picked, 'gpt-5.5 must be listed in the model dropdown');
  picked.click();
  assert.strictEqual(pickerText(win), 'gpt-5.5');

  // Switch away to the sibling tab — it must show its own model
  // (claude-opus-4-7 from the models event), not the user's pick.
  clickTab(win, 'persisted-tab-b');
  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'sibling tab must keep its own model (not inherit the just-picked value)',
  );

  // Switch back — the user-picked value must persist.
  clickTab(win, 'persisted-tab-a');
  assert.strictEqual(
    pickerText(win),
    'gpt-5.5',
    'user-picked model must persist across tab switches',
  );

  // (silence "unused" lint warning for posted)
  void posted;

  win.close();
  console.log('  ok - user-picked tab is not overwritten by models event');
}

function runTests() {
  testModelsEventPropagatesIntoTabState();
  testRestoredTabInheritsTemplateModelOnLaunch();
  testModelsEventPreservesUserPickedTabs();
}

try {
  runTests();
  console.log('\n3 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
