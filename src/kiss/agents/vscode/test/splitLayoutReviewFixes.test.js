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
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {inlineDesignTokens} = require('./designTokens');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  const {desktopMatches = true} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
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
  win.Element.prototype.setPointerCapture = function () {};
  win.Element.prototype.releasePointerCapture = function () {};
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
  const listeners = [];
  const mql = {
    matches: desktopMatches,
    media: '(min-width: 900px)',
    addEventListener: (ev, fn) => {
      if (ev === 'change') listeners.push(fn);
    },
    removeEventListener: () => {},
    addListener: fn => listeners.push(fn),
    removeListener: () => {},
  };
  win.matchMedia = function (query) {
    if (query === '(min-width: 900px)') return mql;
    return {
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  function fireChange(matches) {
    mql.matches = matches;
    listeners.forEach(fn => fn(mql));
  }
  return {win, posted, fireChange};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
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

async function testRegistryChatIsRetiredOnlyOnceItsStateIsKnown() {
  const {win} = makeWebview();
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

async function testEmptiedRegistryKeepsAChatBesideOpenFiles() {
  const {win} = makeWebview();
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

async function testBrowserTabComesBackAfterStackedRoundTrip() {
  const {win, fireChange} = makeWebview();
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

async function testComposerFocusRetriesLeaveTheBrowserScreen() {
  const {win} = makeWebview();
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
  send(win, {type: 'focusInput'});
  screen.focus();
  await new Promise(r => setTimeout(r, 450));
  assert.strictEqual(
    win.document.activeElement,
    screen,
    'a deferred retry pulled the keyboard out of the browser',
  );
  win.close();
}

const CSS = inlineDesignTokens(
  fs.readFileSync(path.join(MEDIA, 'remote-codex.css'), 'utf8'),
);

function cssRule(selector) {
  const source = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(source + String.raw`\s*(?:,[^{]*)?\{([^}]*)\}`, 'g');
  let body = null;
  let m;
  while ((m = re.exec(CSS)) !== null) body = m[1];
  assert.ok(body !== null, `CSS rule for ${selector} missing`);
  return body;
}

async function testGridFitsBetweenDockedPanels() {
  // The split grid (two panes) applies while a content tab is open;
  // without one #app is the chat column alone.
  const app = cssRule('body.remote-chat.remote-desktop.content-pane-open #app');
  const columns = /grid-template-columns:\s*([^;]+);/.exec(app);
  assert.ok(columns, 'the split grid declares its columns');
  const mins = columns[1].match(/minmax\(\s*min\(280px, 30%\)/g) || [];
  assert.strictEqual(
    mins.length,
    2,
    'each pane yields below 280px when #app is narrow: ' + columns[1].trim(),
  );
  const btn = cssRule('body.remote-chat.remote-desktop .new-output-btn');
  assert.ok(btn.includes('grid-area: chat'), 'the button shares the chat cell');
  assert.ok(btn.includes('align-self: end'), 'pinned to the cell bottom');
  assert.ok(btn.includes('justify-self: center'), 'centred in the cell');
  // The rule belongs to the desktop grid: it must sit inside the
  // >= 900px media block, never leak into the stacked flex layout.
  const at = CSS.indexOf('body.remote-chat.remote-desktop .new-output-btn');
  const mediaOpen = CSS.lastIndexOf('@media (width >= 900px)', at);
  assert.ok(mediaOpen >= 0, 'the button rule is under a desktop media query');
  const between = CSS.slice(mediaOpen, at);
  let depth = 0;
  for (const ch of between) {
    if (ch === '{') depth += 1;
    else if (ch === '}') depth -= 1;
  }
  assert.strictEqual(depth, 1, 'the button rule is inside that media block');
}

(async () => {
  const tests = [
    testRegistryChatIsRetiredOnlyOnceItsStateIsKnown,
    testEmptiedRegistryKeepsAChatBesideOpenFiles,
    testBrowserTabComesBackAfterStackedRoundTrip,
    testComposerFocusRetriesLeaveTheBrowserScreen,
    testGridFitsBetweenDockedPanels,
  ];
  for (const t of tests) {
    await t();
    console.log('ok -', t.name);
  }
})().catch(e => {
  console.error(e);
  process.exit(1);
});
