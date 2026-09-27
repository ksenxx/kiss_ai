// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-pattern A7 (focus stealing) in
// media/main.js: a finishing task, an autocommit result or a tab
// activation must not yank the caret out of a field the user is typing
// in, must not focus the composer behind an open sheet, and must not
// pull the user onto the finished tab once they have interacted with
// the UI since submitting.  Explicit host requests (focusInput) still
// focus the composer.  Each test fails on the old code.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

function startTask(win, tabId) {
  h.send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
}

function finishTask(win, tabId) {
  h.send(win, {type: 'task_done', tabId, startTs: 1000, endTs: 3000});
}

/** Open the promptlets sheet and put the caret in its search box. */
function focusTricksSearch(win) {
  win.__TRICKS__ = ['do the thing'];
  h.click(win, h.byId(win, 'tricks-btn'));
  const search = h.byId(win, 'tricks-search');
  search.focus();
  assert.strictEqual(win.document.activeElement, search, 'precondition');
  return search;
}

async function main() {
  await test('task_done does not pull the caret out of the promptlet search', () => {
    const {win} = h.makeWebview();
    const tabId = win._testApi.getActiveTabId();
    startTask(win, tabId);
    const search = focusTricksSearch(win);
    finishTask(win, tabId);
    assert.strictEqual(
      win.document.activeElement,
      search,
      'setReady must leave the caret where the user is typing',
    );
    win.close();
  });

  await test('task_done does not focus the composer behind the open settings sheet', () => {
    const {win} = h.makeWebview();
    const tabId = win._testApi.getActiveTabId();
    startTask(win, tabId);
    h.click(win, h.byId(win, 'settings-btn'));
    assert.ok(h.byId(win, 'settings-panel').classList.contains('open'));
    win.document.body.focus();
    finishTask(win, tabId);
    assert.notStrictEqual(
      win.document.activeElement,
      h.byId(win, 'task-input'),
      'the composer under a sheet must not take focus',
    );
    win.close();
  });

  await test('autocommit result does not steal focus from another text field', async () => {
    const {win} = h.makeWebview();
    const search = focusTricksSearch(win);
    h.send(win, {
      type: 'autocommit_done',
      tabId: win._testApi.getActiveTabId(),
      success: false,
      manual: true,
      message: 'nothing to commit',
    });
    await h.sleep(350);
    assert.strictEqual(
      win.document.activeElement,
      search,
      'focusInputWithRetry (and its retries) must not move the caret',
    );
    win.close();
  });

  await test("the host's focusInput request still focuses the composer", () => {
    const {win} = h.makeWebview();
    focusTricksSearch(win);
    h.click(win, h.byId(win, 'tricks-panel-close'));
    h.byId(win, 'tricks-search').focus();
    h.send(win, {type: 'focusInput'});
    assert.strictEqual(
      win.document.activeElement,
      h.byId(win, 'task-input'),
      'an explicit request wins over the typing-elsewhere guard',
    );
    win.close();
  });

  await test('a finished background tab is not raised after the user interacted', () => {
    const {win} = h.makeWebview();
    const tabA = win._testApi.getActiveTabId();
    startTask(win, tabA);
    win._testApi.endLaunch();
    win._testApi.createNewTab();
    const tabB = win._testApi.getActiveTabId();
    assert.notStrictEqual(tabA, tabB, 'precondition: two tabs');
    // The user is reading / working in tab B.
    win.document.body.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true}),
    );
    finishTask(win, tabA);
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      tabB,
      'the user keeps their place',
    );
    const doneTab = win.document.querySelector(
      `.chat-tab[data-tab-id=${JSON.stringify(tabA)}]`,
    );
    assert.ok(doneTab, 'the finished tab is still in the strip');
    win.close();
  });

  await test('an agent-made switch is still undone when the task finishes', () => {
    const {win} = h.makeWebview();
    const tabA = win._testApi.getActiveTabId();
    startTask(win, tabA);
    win._testApi.endLaunch();
    // No key press, click or wheel since the submit: the move away
    // from tab A was not the user's.
    win._testApi.createNewTab();
    finishTask(win, tabA);
    assert.strictEqual(win._testApi.getActiveTabId(), tabA);
    win.close();
  });

  report('ui_antipattern_focus');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
