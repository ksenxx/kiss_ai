// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The desktop remote page's split layout (chat pane | content pane):
// the chat the user leaves is retired only once its state is known,
// an emptied registry never leaves a vanished chat on screen beside
// surviving files, a browser tab survives a round trip through the
// stacked layout, the composer never takes the keyboard from the
// content pane, and the grid fits between the docked side panels.

'use strict';

const assert = require('assert');
const {
  makePage: makeSharedPage,
  send,
  stubDesktop,
  captureTimers,
  loadMainCss,
} = require('./simplify2_harness');
const {installTokenisedCss} = require('./designTokens');

function makePage(opts) {
  const {desktopMatches = true} = opts || {};
  let fireChange;
  const {win, posted} = makeSharedPage({
    bodyClass: 'remote-chat',
    beforeScripts: win => {
      ({fireChange} = stubDesktop(win, desktopMatches));
      stubBrowserTabView(win);
    },
  });
  return {win, posted, fireChange};
}

function stubBrowserTabView(win) {
  const noop = function () {};
  win.BrowserTabView = {
    create: function () {
      const el = win.document.createElement('div');
      el.className = 'browser-screen';
      el.tabIndex = 0;
      return {
        el: el,
        setBadge: noop,
        setVisible: noop,
        state: noop,
        error: noop,
        frame: noop,
        resubscribe: noop,
        dispose: noop,
      };
    },
  };
}

function openTabs(win) {
  return Array.from(win._testApi.openTabs());
}

function chatIds(win) {
  return openTabs(win)
    .filter(t => !t.isContentTab)
    .map(t => t.id);
}

function activeId(win) {
  return win._testApi.getActiveTabId();
}

function clickNewChat(win) {
  win.document
    .getElementById('new-chat-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function shownContentTab(win) {
  const el = win.document.querySelector('#content-tab-list .chat-tab.active');
  return el ? el.dataset.tabId : null;
}

function twoChats(win) {
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws'},
      {tabId: 'a2', chatId: 'chat-2', title: 'a2', workDir: '/ws'},
    ],
  });
}

function openBrowser(win) {
  send(win, {
    type: 'openBrowserTab',
    tab_id: 'browser-1',
    url: 'https://example.com',
    title: 'Example',
    focus: true,
  });
}

function testRegistryChatIsRetiredOnlyOnceItsStateIsKnown() {
  const {win} = makePage();
  twoChats(win);
  assert.strictEqual(activeId(win), 'a1', 'the first registry chat is shown');

  // Leaving a1 before its replay arrived: its task may be running on
  // another surface, so the chat stays.
  clickNewChat(win);
  assert.ok(chatIds(win).includes('a1'), 'a1 survives + before its replay');
  const fresh = activeId(win);
  assert.notStrictEqual(fresh, 'a1');

  // The replay says a1 is idle: leaving it now retires it.
  send(win, {type: 'task_events', tabId: 'a1', task: 'old task', events: []});
  win._testApi.switchToTab('a1');
  assert.strictEqual(activeId(win), 'a1');
  clickNewChat(win);
  assert.ok(!chatIds(win).includes('a1'), 'a1 retired after its replay');

  // A status event settles a chat's state as well: a2 is running, so
  // it stays; once it stops it goes.
  send(win, {type: 'status', running: true, tabId: 'a2'});
  win._testApi.switchToTab('a2');
  clickNewChat(win);
  assert.ok(chatIds(win).includes('a2'), 'a running a2 is never retired');
  send(win, {type: 'status', running: false, tabId: 'a2'});
  win._testApi.switchToTab('a2');
  clickNewChat(win);
  assert.ok(!chatIds(win).includes('a2'), 'an idle a2 is retired');
  win.close();
}

function testEmptiedRegistryKeepsAChatBesideOpenFiles() {
  const {win} = makePage();
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws'}],
  });
  send(win, {type: 'task_events', tabId: 'a1', task: 'the task', events: []});
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/notes.md',
    name: 'notes.md',
    content: '# Notes',
    version: 'v1',
  });
  const fileTab = openTabs(win).find(t => t.isContentTab);
  assert.ok(fileTab, 'the file opened as a content tab');
  assert.strictEqual(
    activeId(win),
    'a1',
    'split layout: the chat stays active',
  );
  assert.strictEqual(
    shownContentTab(win),
    fileTab.id,
    'the file is in the pane',
  );

  // Another surface closed the chat: the registry is empty, the file
  // is still open, and the chat pane must not keep showing a1.
  send(win, {type: 'tabs_state', tabs: []});
  assert.ok(!chatIds(win).includes('a1'), 'a1 is gone');
  assert.strictEqual(chatIds(win).length, 1, 'one placeholder chat');
  assert.strictEqual(
    activeId(win),
    chatIds(win)[0],
    'the placeholder is shown',
  );
  assert.ok(
    openTabs(win).some(t => t.id === fileTab.id),
    'the file tab survived',
  );
  assert.strictEqual(shownContentTab(win), fileTab.id, 'and is still shown');
  win.close();
}

function testBrowserTabComesBackAfterStackedRoundTrip() {
  const {win, fireChange} = makePage();
  twoChats(win);
  openBrowser(win);
  assert.strictEqual(shownContentTab(win), 'browser-1');
  assert.strictEqual(activeId(win), 'a1', 'the chat pane keeps the chat');

  fireChange(false);
  assert.ok(!win.document.body.classList.contains('remote-desktop'));
  assert.strictEqual(activeId(win), 'a1', 'stacked: the chat stays on screen');
  const area = win.document.getElementById('content-tab-area');
  assert.strictEqual(area.style.display, 'none', 'the content pane is gone');

  fireChange(true);
  assert.ok(win.document.body.classList.contains('remote-desktop'));
  assert.strictEqual(
    shownContentTab(win),
    'browser-1',
    'the browser tab is back in the content pane',
  );
  assert.notStrictEqual(area.style.display, 'none');
  win.close();
}

function testComposerFocusRetriesLeaveTheBrowserScreen() {
  const {win} = makePage();
  twoChats(win);
  openBrowser(win);
  const screen = win.document.querySelector('.browser-screen');
  const inp = win.document.getElementById('task-input');
  screen.focus();
  assert.strictEqual(win.document.activeElement, screen);

  // The daemon's connect-time nudge: not at once...
  send(win, {type: 'focusInput'});
  assert.strictEqual(win.document.activeElement, screen, 'not at once');

  // ...nor by a deferred retry: the nudge lands while the composer has
  // the keyboard (so the 100 / 300 ms retries are scheduled), and the
  // user clicks into the browser before they fire.
  inp.focus();
  assert.strictEqual(win.document.activeElement, inp);
  const timers = captureTimers(win);
  send(win, {type: 'focusInput'});
  const retries = timers.filter(t => t.ms === 100 || t.ms === 300);
  assert.strictEqual(retries.length, 2, 'the two retries are armed');
  screen.focus();
  // The retries are fired by hand instead of waited for.
  for (const t of retries) t.cb();
  assert.strictEqual(
    win.document.activeElement,
    screen,
    'a deferred retry pulled the keyboard out of the browser',
  );
  win.close();
}

/**
 * The computed split-grid styles of a desktop page with a content tab
 * open: `#app`'s columns and the "new output" button's cell.  jsdom
 * never matches a width media feature, so `desktopMedia` installs
 * remote-codex.css's `@media (width >= 900px)` block as `@media screen`
 * (false leaves it unmatched, as on a stacked phone layout).
 */
function splitGridStyles(desktopMedia) {
  const {win} = makePage();
  loadMainCss(win);
  installTokenisedCss(win, 'remote-codex.css', {
    applyMedia: desktopMedia ? /\(width >= 900px\)/ : null,
  });
  const doc = win.document;
  doc.body.classList.add('content-pane-open');
  const btn = doc.createElement('button');
  btn.className = 'new-output-btn';
  doc.getElementById('app').appendChild(btn);
  const styles = {
    columns: win.getComputedStyle(doc.getElementById('app')).gridTemplateColumns,
    button: win.getComputedStyle(btn),
  };
  win.close();
  return styles;
}

function testGridFitsBetweenDockedPanels() {
  const {columns, button} = splitGridStyles(true);
  const mins = columns.match(/minmax\(\s*min\(280px, 30%\)/g) || [];
  assert.strictEqual(
    mins.length,
    2,
    'each pane yields below 280px when #app is narrow: ' + columns,
  );
  assert.strictEqual(button.gridArea, 'chat', 'the button shares the chat cell');
  assert.strictEqual(button.alignSelf, 'end', 'pinned to the cell bottom');
  assert.strictEqual(button.justifySelf, 'center', 'centred in the cell');
  // The rule belongs to the desktop grid: outside the >= 900px media
  // block (the stacked flex layout) neither the grid nor the cell apply.
  const stacked = splitGridStyles(false);
  assert.strictEqual(stacked.columns, 'none', 'no split grid when stacked');
  assert.notStrictEqual(stacked.button.gridArea, 'chat', 'no chat cell when stacked');
}

const tests = [
  testRegistryChatIsRetiredOnlyOnceItsStateIsKnown,
  testEmptiedRegistryKeepsAChatBesideOpenFiles,
  testBrowserTabComesBackAfterStackedRoundTrip,
  testComposerFocusRetriesLeaveTheBrowserScreen,
  testGridFitsBetweenDockedPanels,
];
try {
  for (const t of tests) {
    t();
    console.log('ok -', t.name);
  }
} catch (e) {
  console.error(e);
  process.exit(1);
}
