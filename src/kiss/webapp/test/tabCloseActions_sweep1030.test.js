// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for which tabs the user can close and which
// tab comes on screen after a close (media/main.js):
// * a chat's and a sub-agent's tab have no close control and no
//   gesture closes them; a file's tab keeps its close control;
// * closing a file on the stacked (phone) layout lands on the chat that
//   opened it, not on whichever chat sits next to it in the tab list;
// * when the page's last chat tab closes (a finished sub-agent's), a
//   file takes the screen on the stacked layout, a fresh chat anywhere
//   else, and closing the last file opens a fresh chat too;
// * widening the window while a file whose chat is gone is on screen
//   puts another open chat in the chat pane instead of a new chat.
'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const h = require('./ui_antipattern_harness');
const {tabEl} = require('./subagent_tab_helpers');

const {test, report} = h.makeRunner();

/** A remote page on the desktop (split) or phone (stacked) layout;
 *  `fire(matches)` crosses the desktop breakpoint. */
function page(desktop) {
  let fire = null;
  const {win, posted} = harness.makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js', 'contentContextMenu.js'],
    beforeScripts(w) {
      harness.stubRaf(w);
      fire = harness.fakeMatchMedia(w, desktop);
      harness.stubMonaco(w);
    },
  });
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  return {win, posted, tabA, fire};
}

/** Open /ws/*name* from chat *owner*; returns the file tab's id. */
function openFile(win, owner, name = 'a.py') {
  const path = '/ws/' + name;
  h.send(win, {
    type: 'fileContent',
    tabId: owner,
    path,
    name,
    content: 'print(1)\n',
    version: 'v1',
  });
  const file = win._testApi.openTabs().find(t => t.contentPath === path);
  assert.ok(file, 'the file opened in a tab');
  return file.id;
}

const SOLO = 'task__sub_solo';

/**
 * A page whose last chat tab is a sub-agent's, on screen: the
 * sub-agent of a task that has no tab (its `new_tab` names no parent)
 * is a group of its own, and the registry dropped the page's chat.
 * The files *names*, opened from that chat, stay.
 */
function lastChatTabIsASubagents(desktop, names) {
  const {win, tabA} = page(desktop);
  const files = names.map(name => openFile(win, tabA, name));
  h.send(win, {type: 'new_tab', task_id: 'solo', parent_tab_id: '', taskId: ''});
  h.send(win, {
    type: 'tabs_state',
    tabs: [{tabId: tabA, chatId: 'c-a', title: 'A', workDir: ''}],
  });
  h.send(win, {type: 'tabs_state', tabs: []});
  win._testApi.switchToTab(SOLO);
  // The daemon's answer to the tab's resumeSession: what the sub-agent
  // did so far, under its task.
  h.send(win, {
    type: 'task_events',
    tabId: SOLO,
    task: 'solo task',
    task_id: 'solo',
    events: [{type: 'text_delta', text: 'working'}],
  });
  assert.deepStrictEqual(harness.tabIds(win), [...files, SOLO]);
  return {win, files};
}

/** Asserts that the page's one chat tab is a fresh chat, on screen. */
function assertFreshChatShows(win, why) {
  const chats = win._testApi.openTabs().filter(t => !t.isContentTab);
  assert.strictEqual(chats.length, 1, why);
  assert.ok(!chats[0].isSubagentTab, why);
  assert.strictEqual(win._testApi.getActiveTabId(), chats[0].id, why);
}

function closeWithX(win, tabId) {
  const el = tabEl(win, tabId);
  assert.ok(el, `tab ${tabId} is on the strip`);
  h.click(win, el.querySelector('.chat-tab-close'));
}

/** A press of the middle mouse button on *el*. */
function middleClick(win, el) {
  for (const type of ['mousedown', 'mouseup', 'auxclick']) {
    el.dispatchEvent(
      new win.MouseEvent(type, {bubbles: true, cancelable: true, button: 1}),
    );
  }
}

async function main() {
  await test("a chat's and a sub-agent's tab cannot be closed by the user", () => {
    const {win, posted, tabA} = page(false);
    h.send(win, {
      type: 'openSubagentTab',
      tab_id: 'sub-a',
      parent_tab_id: tabA,
      description: 'a sub-agent',
      task_id: 'sub-task-a',
      taskIndex: 0,
    });
    // Both layouts' rows: the group strip lists the chat and its
    // sub-agent, whichever of the two is on screen.
    for (const shown of [tabA, 'sub-a']) {
      win._testApi.switchToTab(shown);
      for (const id of [tabA, 'sub-a']) {
        const el = tabEl(win, id);
        assert.ok(el, `tab ${id} is on the strip`);
        assert.strictEqual(
          el.querySelector('.chat-tab-close'),
          null,
          `tab ${id} has no close control`,
        );
        el.focus();
        for (const key of ['Delete', 'Backspace', 'Escape']) h.key(win, el, key);
        for (const mod of ['ctrlKey', 'metaKey']) {
          h.key(win, el, 'w', {[mod]: true});
          h.key(win, el, 'F4', {[mod]: true});
        }
        middleClick(win, el);
        el.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
      }
    }
    assert.deepStrictEqual(harness.tabIds(win), [tabA, 'sub-a']);
    assert.deepStrictEqual(h.ofType(posted, 'closeTab'), []);
    // A file's tab is the user's to close.
    win._testApi.switchToTab(tabA);
    const file = openFile(win, tabA);
    assert.ok(tabEl(win, file).querySelector('.chat-tab-close'));
    win.close();
  });

  await test('closing a file on the phone layout lands on its own chat', () => {
    const {win, tabA} = page(false);
    h.send(win, {
      type: 'tabs_state',
      tabs: [
        {tabId: tabA, chatId: '', title: 'A', workDir: ''},
        {tabId: 'tab-b', chatId: 'c-b', title: 'B', workDir: ''},
      ],
    });
    const file = openFile(win, tabA);
    assert.strictEqual(win._testApi.getActiveTabId(), file, 'the file shows');
    closeWithX(win, file);
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      tabA,
      'the chat that opened the file comes back',
    );
    win.close();
  });

  await test('the last chat tab closing leaves a file or a fresh chat', () => {
    const done = {type: 'subagentDone', tab_id: SOLO, task_id: 'solo'};
    const shown = w => w._testApi.getActiveTabId();

    // No other tab: a fresh chat opens.
    let {win, files} = lastChatTabIsASubagents(false, []);
    h.send(win, done);
    assertFreshChatShows(win, 'a fresh chat replaces the last tab');
    assert.strictEqual(harness.tabIds(win).length, 1);
    win.close();

    // Phone layout, files open: the nearest file takes the screen over
    // a blank chat surface, and closing the last file opens a chat.
    ({win, files} = lastChatTabIsASubagents(false, ['a.py', 'b.py']));
    h.send(win, done);
    assert.deepStrictEqual(harness.tabIds(win), files);
    assert.strictEqual(shown(win), files[1], 'the nearest file shows');
    assert.strictEqual(
      win.document.getElementById('output').children.length,
      0,
      "the sub-agent's transcript is gone from under the file",
    );
    closeWithX(win, files[1]);
    assert.strictEqual(shown(win), files[0], 'then the other file');
    closeWithX(win, files[0]);
    assertFreshChatShows(win, 'a fresh chat replaces the last file');
    assert.strictEqual(harness.tabIds(win).length, 1);
    win.close();

    // Desktop layout: the chat pane never shows a file, so it gets a
    // fresh chat and the file stays in the content pane.
    ({win, files} = lastChatTabIsASubagents(true, ['a.py']));
    h.send(win, done);
    assertFreshChatShows(win, 'the chat pane gets a fresh chat');
    assert.ok(harness.tabIds(win).includes(files[0]), 'the file stays open');
    win.close();
  });

  await test('widening onto a file whose chat closed shows an open chat', () => {
    const {win, posted, tabA, fire} = page(false);
    h.send(win, {
      type: 'tabs_state',
      tabs: [
        {tabId: tabA, chatId: '', title: 'A', workDir: ''},
        {tabId: 'tab-b', chatId: 'c-b', title: 'B', workDir: ''},
      ],
    });
    const file = openFile(win, tabA);
    // The registry no longer lists chat A; its file stays on screen.
    h.send(win, {
      type: 'tabs_state',
      tabs: [{tabId: 'tab-b', chatId: 'c-b', title: 'B', workDir: ''}],
    });
    assert.strictEqual(win._testApi.getActiveTabId(), file, 'the file shows');
    const before = h.ofType(posted, 'newChat').length;
    fire(true);
    assert.strictEqual(
      h.ofType(posted, 'newChat').length,
      before,
      'no new chat is opened while chat B is there',
    );
    assert.strictEqual(win._testApi.getActiveTabId(), 'tab-b');
    assert.deepStrictEqual(harness.tabIds(win), ['tab-b', file]);
    win.close();
  });

  report('tabCloseActions_sweep1030');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
