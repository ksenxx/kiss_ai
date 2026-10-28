// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM end-to-end tests for the desktop remote page's content pane
// and task-info panel (media/chat.html, main.js, main.css,
// remote-codex.css):
//
//   * the content pane (its tab row, area and handle) exists only while
//     a file, browser or terminal tab is open: body.content-pane-open
//     follows the open content tabs, and without it #app is the chat
//     column alone, clear of the docked panel;
//   * with the pane open the panel lies on top of it; opening a file, a
//     browser or a terminal (the user or the agent) and a press
//     anywhere in the pane slide the panel off (body.meta-hidden, the
//     panel inert) and the #meta-drawer tab brings it back; closing the
//     last content tab brings it back too;
//   * the hide button at the right end of the pane's tab row folds the
//     pane away (body.content-pane-folded: the chat alone, the panel
//     docked, the content tabs kept); the #content-pane-show tab or the
//     next content tab shown unfolds it; the fold is kept in
//     localStorage (kiss-content-pane-folded) across reloads, like the
//     chat pane's share of the split, until unfolded or the last
//     content tab closes;
//   * the Explorer and Source Control sections catch up on task news
//     that arrived while the panel was hidden;
//   * a tool-call panel starts folded and is active (panel.panel-active,
//     main.css pulses its header text) until its tool_result, a
//     Question and a fan-out panel start open.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {
  makePage: makeHarnessPage,
  send,
  click,
  byId,
  stubDesktop,
  loadMainCss,
  MEDIA,
  ofType,
} = require('./simplify2_harness');
const {installTokenisedCss, cssRule} = require('./designTokens');

/**
 * A fresh page.  `storage` is the localStorage contents of the page
 * instance this one reloads (see storageOf); `denyStorage` makes every
 * localStorage access throw, as a browser with storage blocked does.
 */
function makePage(opts) {
  const {desktop = true, storage = null, denyStorage = false} = opts || {};
  let fireChange;
  const {win, posted} = makeHarnessPage({
    bodyClass: 'remote-chat',
    beforeScripts(win) {
      if (storage) {
        for (const [k, v] of Object.entries(storage))
          win.localStorage.setItem(k, v);
      }
      if (denyStorage) {
        Object.defineProperty(win, 'localStorage', {
          get() {
            throw new Error('localStorage denied');
          },
        });
      }
      ({fireChange} = stubDesktop(win, desktop));
    },
  });
  return {win, posted, fireChange};
}

function press(win, el) {
  el.dispatchEvent(new win.Event('pointerdown', {bubbles: true}));
}

function hasClass(win, cls) {
  return win.document.body.classList.contains(cls);
}

/** Open a chat tab `a1` on /ws and a file in the content pane. */
function openChatAndFile(win) {
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
  const fileTab = win._testApi.openTabs().find(t => t.isContentTab);
  assert.ok(fileTab, 'the file opened as a content tab');
  return fileTab;
}

/** Everything a page left in localStorage, for the page that reloads it. */
function storageOf(win) {
  const items = {};
  for (let i = 0; i < win.localStorage.length; i++) {
    const k = win.localStorage.key(i);
    items[k] = win.localStorage.getItem(k);
  }
  return items;
}

const FOLD_KEY = 'kiss-content-pane-folded';

function closeContentTab(win, tabId) {
  const btn = win.document.querySelector(
    '#content-tab-list .chat-tab[data-tab-id="' + tabId + '"] .chat-tab-close',
  );
  assert.ok(btn, 'the content tab has a close button');
  click(win, btn);
}

const TESTS = [];
function test(name, fn) {
  TESTS.push({name, fn});
}

test('the content pane exists only while a content tab is open', () => {
  const {win} = makePage();
  assert.ok(hasClass(win, 'remote-desktop'));
  assert.ok(!hasClass(win, 'content-pane-open'), 'no content tab: no pane');
  const fileTab = openChatAndFile(win);
  assert.ok(hasClass(win, 'content-pane-open'), 'a file opened the pane');
  closeContentTab(win, fileTab.id);
  assert.ok(
    !win._testApi.openTabs().some(t => t.isContentTab),
    'the file tab closed',
  );
  assert.ok(!hasClass(win, 'content-pane-open'), 'the pane went away');
  win.close();
});

test('a press in the pane hides the panel; the drawer and the pane closing bring it back', () => {
  const {win} = makePage();
  const panel = byId(win, 'meta-panel');
  const drawer = byId(win, 'meta-drawer');
  // Without a content pane the panel is docked beside the chat: a
  // press in the (absent) pane changes nothing.
  press(win, byId(win, 'content-tab-bar'));
  assert.ok(!hasClass(win, 'meta-hidden'));

  openChatAndFile(win);
  assert.ok(
    hasClass(win, 'meta-hidden'),
    'the file opened: the panel slid off',
  );
  assert.ok(panel.hasAttribute('inert'), 'the hidden panel is inert');
  assert.strictEqual(drawer.getAttribute('aria-expanded'), 'false');

  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'), 'the drawer brings it back');
  assert.ok(!panel.hasAttribute('inert'));
  assert.strictEqual(drawer.getAttribute('aria-expanded'), 'true');

  press(win, byId(win, 'content-tab-area'));
  assert.ok(hasClass(win, 'meta-hidden'), 'a press in the pane hides it');
  assert.strictEqual(drawer.getAttribute('aria-expanded'), 'false');
  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'));

  // The pane's tab row counts as the pane.
  press(win, byId(win, 'content-tab-bar'));
  assert.ok(hasClass(win, 'meta-hidden'));
  // A second file keeps it off the pane.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/other.md',
    name: 'other.md',
    content: '# Other',
    version: 'v1',
  });
  assert.ok(hasClass(win, 'meta-hidden'), 'another file keeps it hidden');
  // Closing every content tab docks the panel beside the chat again.
  for (const t of win._testApi.openTabs().filter(t => t.isContentTab))
    closeContentTab(win, t.id);
  assert.ok(!hasClass(win, 'content-pane-open'));
  assert.ok(!hasClass(win, 'meta-hidden'), 'docked: never hidden');
  assert.ok(!panel.hasAttribute('inert'));
  win.close();
});

test('focus taken by a frame of the pane counts as a press in it', () => {
  const {win} = makePage();
  openChatAndFile(win);
  click(win, byId(win, 'meta-drawer'));
  assert.ok(!hasClass(win, 'meta-hidden'), 'the panel is back after the open');
  const area = byId(win, 'content-tab-area');
  const frame = win.document.createElement('iframe');
  area.appendChild(frame);
  // The window blurring for any other reason (another window, a frame
  // elsewhere) changes nothing.
  win.dispatchEvent(new win.Event('blur'));
  assert.ok(!hasClass(win, 'meta-hidden'));
  const outside = win.document.createElement('iframe');
  win.document.body.appendChild(outside);
  outside.focus();
  win.dispatchEvent(new win.Event('blur'));
  assert.ok(!hasClass(win, 'meta-hidden'), 'a frame outside the pane');
  frame.focus();
  assert.strictEqual(win.document.activeElement, frame);
  win.dispatchEvent(new win.Event('blur'));
  assert.ok(hasClass(win, 'meta-hidden'), 'the preview frame took the press');
  win.close();
});

test('leaving the desktop layout drops the hidden state', () => {
  const {win, fireChange} = makePage();
  openChatAndFile(win);
  press(win, byId(win, 'content-tab-area'));
  assert.ok(hasClass(win, 'meta-hidden'));
  fireChange(false);
  assert.ok(!hasClass(win, 'remote-desktop'));
  assert.ok(!hasClass(win, 'meta-hidden'), 'the phone drawer is never hidden');
  assert.ok(byId(win, 'meta-panel').hasAttribute('inert'), 'closed drawer');
  win.close();
});

test('the workspace views catch up on task news once the panel is back', () => {
  const {win, posted} = makePage();
  send(win, {type: 'configData', config: {work_dir: '/ws'}});
  const listed = ofType(posted, 'listDir').length;
  assert.ok(listed >= 1, 'the Explorer listed the pinned workspace');
  // Answer the root listing so task news has a loaded folder to re-list.
  const req = ofType(posted, 'listDir').pop();
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: req.path,
    root: req.path,
    entries: [{name: 'a.txt', type: 'file'}],
  });
  openChatAndFile(win);
  press(win, byId(win, 'content-tab-area'));
  assert.ok(hasClass(win, 'meta-hidden'));
  const before = ofType(posted, 'listDir').length;
  // Any timer the page arms on the news is run by hand, so no
  // wall-clock wait decides whether a deferred reload would have shown.
  const armed = [];
  const realSetTimeout = win.setTimeout;
  win.setTimeout = function (cb, ms, ...args) {
    armed.push(() => cb(...args));
    return 0;
  };
  send(win, {type: 'tasks_updated'});
  win.setTimeout = realSetTimeout;
  while (armed.length) armed.shift()();
  assert.strictEqual(
    ofType(posted, 'listDir').length,
    before,
    'hidden: the Explorer does not reload yet',
  );
  click(win, byId(win, 'meta-drawer'));
  assert.ok(
    ofType(posted, 'listDir').length > before,
    'back on screen: the Explorer reloads',
  );
  win.close();
});

test('a tool-call panel starts folded and pulses until its result', () => {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: 1000});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la',
    description: 'list files',
    tabId: tab,
    ts: 1000,
  });
  const O = byId(win, 'output');
  const tc = O.querySelector('.ev.tc');
  assert.ok(tc.classList.contains('collapsed'), 'folded at birth');
  assert.ok(tc.classList.contains('panel-active'), 'active until the result');
  const hdr = tc.querySelector(':scope > .tc-h');
  const name = hdr.querySelector('.tc-h-name');
  const prev = hdr.querySelector('.collapse-preview');
  assert.strictEqual(name.textContent, 'Bash');
  assert.strictEqual(name.children.length, 0, 'the name is plain text');
  assert.strictEqual(prev.textContent, 'list files', 'the folded preview');
  assert.strictEqual(prev.children.length, 0, 'the preview is plain text');
  assert.strictEqual(hdr.getAttribute('aria-expanded'), 'false');
  // The user may unfold it; the preview empties, the header keeps pulsing.
  click(win, hdr);
  assert.ok(!tc.classList.contains('collapsed'));
  assert.strictEqual(prev.textContent, '');
  assert.ok(tc.classList.contains('panel-active'));
  click(win, hdr);
  assert.strictEqual(prev.textContent, 'list files');

  send(win, {
    type: 'tool_result',
    name: 'Bash',
    content: 'ok',
    tabId: tab,
    ts: 2000,
  });
  assert.ok(
    !tc.classList.contains('panel-active'),
    'the result ends the pulse',
  );
  assert.strictEqual(name.textContent, 'Bash');
  assert.strictEqual(prev.textContent, 'list files');
  win.close();
});

test('the Thoughts panel pulses while the model thinks and writes', () => {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: 1000});
  send(win, {type: 'thinking_start', tabId: tab, ts: 1000});
  send(win, {type: 'thinking_delta', text: 'hmm', tabId: tab});
  const panel = byId(win, 'output').querySelector('.llm-panel');
  assert.ok(
    panel.classList.contains('panel-active'),
    'active from the first token',
  );
  assert.ok(
    panel.querySelector(':scope > .llm-panel-hdr'),
    'the pulsing header',
  );
  send(win, {type: 'thinking_end', tabId: tab});
  send(win, {type: 'text_delta', text: 'Done.', tabId: tab});
  send(win, {type: 'text_end', tabId: tab, ts: 1500});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    tabId: tab,
    ts: 2000,
  });
  assert.ok(
    !panel.classList.contains('panel-active'),
    'the next event ends it',
  );
  win.close();
});

test('Question and fan-out panels start open; the task end stops every pulse', () => {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: 1000});
  send(win, {
    type: 'tool_call',
    name: 'ask_user_question',
    question: 'Proceed?',
    tabId: tab,
    ts: 1000,
  });
  send(win, {
    type: 'tool_result',
    name: 'ask_user_question',
    content: 'yes',
    tabId: tab,
    ts: 1100,
  });
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tasks: ['one', 'two'],
    tabId: tab,
    ts: 1200,
  });
  const O = byId(win, 'output');
  const q = O.querySelector('.tc-question');
  const rp = O.querySelector('.tc-run-parallel');
  assert.ok(
    q && !q.classList.contains('collapsed'),
    'a Question is read, not folded',
  );
  assert.ok(
    rp && !rp.classList.contains('collapsed'),
    'a fan-out keeps its sub-agent tabs open',
  );
  assert.ok(rp.classList.contains('panel-active'));
  assert.ok(
    !q.classList.contains('panel-active'),
    'the answered Question is done',
  );
  // The task stops without a tool_result (or task_done) for the
  // fan-out: the bare status flip ends the pulse.
  send(win, {type: 'status', running: false, tabId: tab});
  assert.strictEqual(O.querySelectorAll('.panel-active').length, 0);
  // Not sealed, though: a refused submit on a busy tab sends the same
  // status while the task runs on, so a late tool_result still times it.
  assert.ok(!rp.dataset.timeDone, 'paused, not sealed');
  send(win, {
    type: 'tool_result',
    name: 'run_parallel',
    content: 'ok',
    tabId: tab,
    ts: 9000,
  });
  assert.strictEqual(rp.dataset.timeDone, '1', 'the result seals it');
  assert.ok(rp.querySelector('.panel-elapsed'), 'with its duration');
  win.close();
});

test('the panel close button hides the docked panel; focus follows the controls', () => {
  const {win, fireChange} = makePage();
  const panel = byId(win, 'meta-panel');
  const drawer = byId(win, 'meta-drawer');
  const close = byId(win, 'meta-close');
  openChatAndFile(win);
  // The open slid the panel off; the drawer brings it back.
  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'));
  // A panel wide enough to cover the whole pane leaves nothing to
  // press there: its close button hides it.
  close.focus();
  assert.strictEqual(win.document.activeElement, close);
  click(win, close);
  assert.ok(hasClass(win, 'meta-hidden'), 'the close button hides the panel');
  assert.strictEqual(
    win.document.activeElement,
    drawer,
    'focus moves to the drawer (the panel went inert)',
  );
  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'));
  assert.strictEqual(
    win.document.activeElement,
    close,
    'focus leaves the faded drawer for the panel',
  );
  // Focus elsewhere stays where it is.
  byId(win, 'task-input').focus();
  press(win, byId(win, 'content-tab-area'));
  assert.ok(hasClass(win, 'meta-hidden'));
  assert.strictEqual(win.document.activeElement, byId(win, 'task-input'));
  click(win, drawer);
  assert.strictEqual(win.document.activeElement, byId(win, 'task-input'));
  // On the phone the same button closes the drawer.
  fireChange(false);
  click(win, byId(win, 'meta-drawer-btn'));
  assert.ok(panel.classList.contains('open'), 'the phone drawer opened');
  click(win, close);
  assert.ok(!panel.classList.contains('open'), 'the phone drawer closed');
  assert.ok(!hasClass(win, 'meta-hidden'));
  win.close();
});

test('a copied running panel reads the tool name as one word', () => {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: 1000});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    tabId: tab,
    ts: 1000,
  });
  const tc = byId(win, 'output').querySelector('.ev.tc');
  assert.ok(win.PanelCopy, 'panelCopy.js loaded');
  const copied = win.PanelCopy.getRawText(tc.querySelector(':scope > .tc-h'));
  assert.strictEqual(copied.split('\n')[0], 'Bash', 'copied as a word');
  win.close();
});

test('replaying a finished task leaves no header pulsing', () => {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: false, tabId: tab});
  // A call that never got its result (the task was stopped); replay
  // skips the terminal event that would have closed it.
  send(win, {
    type: 'task_events',
    tabId: tab,
    task: 'the task',
    events: [
      {type: 'tool_call', name: 'Bash', command: 'sleep 99', ts: 1000},
      {type: 'task_stopped', ts: 2000},
    ],
  });
  const O = byId(win, 'output');
  assert.ok(O.querySelector('.tc'), 'the call replayed');
  assert.strictEqual(O.querySelectorAll('.panel-active').length, 0);
  win.close();
});

/**
 * Give the page the browser and terminal tab views.  jsdom has no
 * xterm.js (terminalTab.js fetches it from a CDN unless window.Terminal
 * and window.FitAddon exist) and no ResizeObserver, so the terminal
 * gets the same minimal stand-ins terminalTabView.test.js uses; the
 * browser view runs as is.
 */
function addBrowserAndTerminal(win) {
  class Terminal {
    constructor() {
      this.cols = 80;
      this.rows = 24;
      this.options = {};
    }
    open() {}
    loadAddon() {}
    attachCustomKeyEventHandler() {}
    write() {}
    onData() {}
    onResize() {}
    focus() {}
    getSelection() {
      return '';
    }
    dispose() {}
  }
  class FitAddon {
    fit() {}
  }
  win.Terminal = Terminal;
  win.FitAddon = {FitAddon};
  win.ResizeObserver = class {
    observe() {}
    disconnect() {}
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'browserTab.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'terminalTab.js'), 'utf8'));
}

test('a file, a browser or a terminal opening slides the panel off', async () => {
  const {win} = makePage();
  addBrowserAndTerminal(win);
  const drawer = byId(win, 'meta-drawer');
  // The agent's file (openChatAndFile) hid the panel; the drawer
  // brings it back before each further open.
  openChatAndFile(win);
  assert.ok(hasClass(win, 'meta-hidden'), 'the file slid the panel off');
  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'));

  // A file picked by the user in the Explorer, i.e. one the daemon
  // answers with the tab's own id, hides it the same way; one opened
  // "to the side" (background) does not take the pane, so it leaves
  // the panel alone.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/aside.md',
    name: 'aside.md',
    content: '# Aside',
    version: 'v1',
    background: true,
  });
  assert.ok(
    !hasClass(win, 'meta-hidden'),
    'a background open leaves the panel',
  );
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/picked.md',
    name: 'picked.md',
    content: '# Picked',
    version: 'v1',
  });
  assert.ok(hasClass(win, 'meta-hidden'), 'the picked file slid the panel off');
  click(win, drawer);

  send(win, {
    type: 'openBrowserTab',
    tab_id: 'b1',
    url: 'https://example.com/',
    title: 'Example',
    focus: true,
  });
  const browserTab = win._testApi.openTabs().find(t => t.id === 'b1');
  assert.ok(
    browserTab && browserTab.isContentTab,
    'the browser opened as a content tab',
  );
  assert.ok(hasClass(win, 'meta-hidden'), 'the browser slid the panel off');
  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'));

  click(win, byId(win, 'terminal-btn'));
  const termTab = win._testApi.openTabs().find(t => t.title === 'Terminal');
  assert.ok(
    termTab && termTab.isContentTab,
    'the terminal opened as a content tab',
  );
  assert.ok(hasClass(win, 'meta-hidden'), 'the terminal slid the panel off');
  // Let xterm attach (ensureXterm resolves in a microtask) before the
  // window goes; a shown content tab left behind keeps node alive.
  await Promise.resolve();
  await Promise.resolve();
  win.close();
});

test('the shown file reopened, a successor and a desktop restore slide the panel off too', () => {
  const {win, fireChange} = makePage();
  addBrowserAndTerminal(win);
  const drawer = byId(win, 'meta-drawer');
  const hideBtn = byId(win, 'content-pane-hide');
  const notes = {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/notes.md',
    name: 'notes.md',
    content: '# Notes',
    version: 'v1',
  };
  openChatAndFile(win);
  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'));
  // The file already on screen, picked again in the Explorer: the
  // reply refreshes it and still slides the panel off.
  send(win, notes);
  assert.ok(
    hasClass(win, 'meta-hidden'),
    'the shown file reopened hides the panel',
  );
  // Folded, the same reopen unfolds the pane; a background reload of
  // the shown file (a save echo) leaves the fold alone.
  click(win, hideBtn);
  send(win, Object.assign({}, notes, {background: true}));
  assert.ok(
    hasClass(win, 'content-pane-folded'),
    'a background reload keeps the fold',
  );
  send(win, notes);
  assert.ok(
    hasClass(win, 'content-pane-open'),
    'the shown file reopened unfolds the pane',
  );
  assert.ok(hasClass(win, 'meta-hidden'));

  // A browser tab shown over the file, the pane folded, the browser
  // closed by the daemon: the file takes the pane, unfolded and clear
  // of the panel.
  send(win, {
    type: 'openBrowserTab',
    tab_id: 'b1',
    url: 'https://example.com/',
    title: 'Example',
    focus: true,
  });
  click(win, hideBtn);
  assert.ok(hasClass(win, 'content-pane-folded'));
  assert.ok(!hasClass(win, 'meta-hidden'));
  send(win, {type: 'closeBrowserTab', tab_id: 'b1'});
  assert.ok(
    hasClass(win, 'content-pane-open'),
    'the successor unfolds the pane',
  );
  assert.ok(hasClass(win, 'meta-hidden'), 'and slides the panel off');
  assert.ok(
    win.document.querySelector(
      '#content-tab-list .chat-tab[data-tab-id="b1"]',
    ) === null,
  );

  // The panel brought back, the window narrowed and widened again: the
  // file restored to the pane is clear of the panel.
  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'));
  fireChange(false);
  assert.ok(!hasClass(win, 'content-pane-open'));
  fireChange(true);
  assert.ok(hasClass(win, 'content-pane-open'), 'the file is back in the pane');
  assert.ok(hasClass(win, 'meta-hidden'), 'the restore slides the panel off');
  win.close();
});

test('the hide button folds the pane away; the show tab or the next open unfolds it', () => {
  const {win, fireChange} = makePage();
  addBrowserAndTerminal(win);
  const hideBtn = byId(win, 'content-pane-hide');
  const showBtn = byId(win, 'content-pane-show');
  assert.ok(hideBtn && showBtn, 'both controls are in the markup');
  assert.ok(
    byId(win, 'content-tab-bar').lastElementChild === hideBtn,
    'the hide button ends the tab row',
  );
  // Nothing to fold without a content tab.
  click(win, hideBtn);
  assert.ok(!hasClass(win, 'content-pane-folded'));

  const fileTab = openChatAndFile(win);
  assert.ok(hasClass(win, 'content-pane-open'));
  assert.ok(hasClass(win, 'meta-hidden'));
  hideBtn.focus();
  click(win, hideBtn);
  assert.ok(!hasClass(win, 'content-pane-open'), 'the pane is gone');
  assert.ok(hasClass(win, 'content-pane-folded'), 'but folded, not closed');
  assert.ok(
    !hasClass(win, 'meta-hidden'),
    'the panel is docked beside the chat again',
  );
  assert.ok(
    win._testApi.openTabs().some(t => t.id === fileTab.id),
    'the content tab stays open',
  );
  assert.strictEqual(
    win.document.activeElement,
    showBtn,
    'focus moved to the show tab',
  );
  // Folding again is a no-op.
  click(win, hideBtn);
  assert.ok(hasClass(win, 'content-pane-folded'));

  click(win, showBtn);
  assert.ok(
    hasClass(win, 'content-pane-open'),
    'the show tab unfolds the pane',
  );
  assert.ok(!hasClass(win, 'content-pane-folded'));
  assert.ok(
    hasClass(win, 'meta-hidden'),
    'unfolded to be seen: the panel slid off',
  );
  assert.strictEqual(
    win.document.activeElement,
    hideBtn,
    'focus moved to the hide button',
  );
  assert.ok(
    win.document.querySelector(
      '#content-tab-list .chat-tab[data-tab-id="' + fileTab.id + '"]',
    ),
    'the tab row still lists the file',
  );

  // Folded, a file opened in the background stays out of sight; one
  // opened to be seen unfolds the pane and slides the panel off it.
  click(win, hideBtn);
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/aside.md',
    name: 'aside.md',
    content: '# Aside',
    version: 'v1',
    background: true,
  });
  assert.ok(
    hasClass(win, 'content-pane-folded'),
    'a background open keeps the fold',
  );
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/other.md',
    name: 'other.md',
    content: '# Other',
    version: 'v1',
  });
  assert.ok(
    hasClass(win, 'content-pane-open'),
    'the next open unfolds the pane',
  );
  assert.ok(!hasClass(win, 'content-pane-folded'));
  assert.ok(hasClass(win, 'meta-hidden'));

  // Focus elsewhere stays where it is when the pane folds.
  byId(win, 'meta-drawer').focus();
  click(win, hideBtn);
  assert.ok(hasClass(win, 'content-pane-folded'));
  assert.notStrictEqual(win.document.activeElement, showBtn);
  click(win, showBtn);
  assert.notStrictEqual(win.document.activeElement, hideBtn);

  // Closing every content tab drops the fold.  The folded pane's tab
  // row is out of reach, so the last tab goes the agent's way: a
  // browser tab alone in the pane, closed from the daemon
  // (closeBrowserTab, the close_browser tool).  The next file opened
  // in the background then shows the pane as it did before.
  click(win, showBtn);
  for (const t of win._testApi.openTabs().filter(t => t.isContentTab)) {
    closeContentTab(win, t.id);
  }
  assert.ok(!hasClass(win, 'content-pane-open'));
  send(win, {
    type: 'openBrowserTab',
    tab_id: 'b1',
    url: 'https://example.com/',
    title: 'Example',
    focus: true,
  });
  assert.ok(hasClass(win, 'content-pane-open'));
  click(win, hideBtn);
  assert.ok(hasClass(win, 'content-pane-folded'));
  send(win, {type: 'closeBrowserTab', tab_id: 'b1'});
  assert.ok(!hasClass(win, 'content-pane-folded'));
  assert.ok(!hasClass(win, 'content-pane-open'));
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/later.md',
    name: 'later.md',
    content: '# Later',
    version: 'v1',
    background: true,
  });
  assert.ok(
    hasClass(win, 'content-pane-open'),
    'the fold did not outlive the tabs',
  );

  // Leaving the desktop layout drops the fold too (no pane to fold).
  click(win, hideBtn);
  assert.ok(hasClass(win, 'content-pane-folded'));
  fireChange(false);
  assert.ok(!hasClass(win, 'content-pane-folded'));
  assert.ok(!hasClass(win, 'content-pane-open'));
  win.close();
});

/**
 * The daemon's `browserTabs` snapshot, sent on every `ready`: a reload
 * gets its browser tabs back through it, unfocused.
 */
function sendBrowserSnapshot(win) {
  send(win, {
    type: 'browserTabs',
    tabs: [
      {
        tab_id: 'b1',
        url: 'https://example.com/',
        title: 'Example',
        focus: false,
      },
    ],
  });
}

/** Reload `win`: a fresh page with its localStorage, chat and browser tabs. */
function reload(win) {
  const next = makePage({storage: storageOf(win)});
  win.close();
  addBrowserAndTerminal(next.win);
  send(next.win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws'}],
  });
  sendBrowserSnapshot(next.win);
  return next.win;
}

test('the fold is remembered across reloads in localStorage', () => {
  let {win} = makePage();
  addBrowserAndTerminal(win);
  const hideBtn = byId(win, 'content-pane-hide');
  // The hide button without a content tab records nothing.
  click(win, hideBtn);
  assert.strictEqual(win.localStorage.getItem(FOLD_KEY), null);

  openChatAndFile(win);
  send(win, {
    type: 'openBrowserTab',
    tab_id: 'b1',
    url: 'https://example.com/',
    title: 'Example',
    focus: true,
  });
  assert.strictEqual(
    win.localStorage.getItem(FOLD_KEY),
    null,
    'unfolded: no key',
  );
  click(win, hideBtn);
  assert.ok(hasClass(win, 'content-pane-folded'));
  assert.strictEqual(
    win.localStorage.getItem(FOLD_KEY),
    '1',
    'the fold is written',
  );

  // Reloaded: the page starts with no content tab (file tabs are not
  // mirrored), so there is no pane and no fold on screen yet.  The
  // browser tab the daemon announces again comes back behind the
  // fold: the chat alone, the show tab on its edge, the panel docked.
  win = reload(win);
  assert.ok(!hasClass(win, 'content-pane-open'));
  assert.ok(
    hasClass(win, 'content-pane-folded'),
    'the browser tab came back folded',
  );
  assert.ok(
    !hasClass(win, 'meta-hidden'),
    'the panel is docked beside the chat',
  );
  assert.ok(
    win._testApi.openTabs().some(t => t.id === 'b1'),
    'the browser tab is open',
  );
  assert.strictEqual(
    win.localStorage.getItem(FOLD_KEY),
    '1',
    'still remembered',
  );

  // A file opened in the background after the reload stays behind the
  // fold; the show tab unfolds the pane and forgets the fold.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/aside.md',
    name: 'aside.md',
    content: '# Aside',
    version: 'v1',
    background: true,
  });
  assert.ok(
    hasClass(win, 'content-pane-folded'),
    'a background open keeps the fold',
  );
  click(win, byId(win, 'content-pane-show'));
  assert.ok(hasClass(win, 'content-pane-open'));
  assert.ok(!hasClass(win, 'content-pane-folded'));
  assert.strictEqual(
    win.localStorage.getItem(FOLD_KEY),
    null,
    'unfolded: forgotten',
  );

  // Reloaded unfolded: the browser tab comes back in an open pane.
  win = reload(win);
  assert.ok(hasClass(win, 'content-pane-open'), 'the pane is back open');
  assert.ok(!hasClass(win, 'content-pane-folded'));

  // Folded, reloaded, then a file opened to be seen: the open unfolds
  // the pane and forgets the fold, as before the reload.
  click(win, byId(win, 'content-pane-hide'));
  assert.strictEqual(win.localStorage.getItem(FOLD_KEY), '1');
  win = reload(win);
  assert.ok(hasClass(win, 'content-pane-folded'));
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/notes.md',
    name: 'notes.md',
    content: '# Notes',
    version: 'v1',
  });
  assert.ok(
    hasClass(win, 'content-pane-open'),
    'the shown file unfolds the pane',
  );
  assert.ok(!hasClass(win, 'content-pane-folded'));
  assert.strictEqual(win.localStorage.getItem(FOLD_KEY), null);

  // Folded, then the last content tab closed: the fold is forgotten,
  // so the next reload shows the pane as soon as a tab comes back.
  click(win, byId(win, 'content-pane-hide'));
  assert.strictEqual(win.localStorage.getItem(FOLD_KEY), '1');
  send(win, {type: 'closeBrowserTab', tab_id: 'b1'});
  assert.strictEqual(
    win.localStorage.getItem(FOLD_KEY),
    '1',
    'a file tab is still open',
  );
  for (const t of win._testApi.openTabs().filter(t => t.isContentTab)) {
    closeContentTab(win, t.id);
  }
  assert.strictEqual(
    win.localStorage.getItem(FOLD_KEY),
    null,
    'no tab left: forgotten',
  );
  win = reload(win);
  assert.ok(hasClass(win, 'content-pane-open'));
  assert.ok(!hasClass(win, 'content-pane-folded'));
  win.close();
});

test('without localStorage the fold still works for the page instance', () => {
  const {win} = makePage({denyStorage: true});
  addBrowserAndTerminal(win);
  openChatAndFile(win);
  assert.ok(
    hasClass(win, 'content-pane-open'),
    'the page came up without storage',
  );
  click(win, byId(win, 'content-pane-hide'));
  assert.ok(hasClass(win, 'content-pane-folded'), 'folded in memory');
  click(win, byId(win, 'content-pane-show'));
  assert.ok(hasClass(win, 'content-pane-open'));
  assert.ok(!hasClass(win, 'content-pane-folded'));
  win.close();
});

function computedOf(win, id) {
  return win.getComputedStyle(byId(win, id));
}

test('the stylesheets carry the layout, the pulse and the machine strip', () => {
  // Everything is read as computed style on the live page.  jsdom never
  // matches a width media feature, so remote-codex.css's desktop block
  // (`@media (width >= 900px)`) is installed as `@media screen`; the
  // body classes main.js toggles (`content-pane-open`, `meta-hidden`,
  // `content-pane-folded`) are set by hand to read each state.
  const {win} = makePage();
  loadMainCss(win);
  installTokenisedCss(win, 'remote-codex.css', {
    applyMedia: /\(width >= 900px\)/,
  });
  const body = win.document.body;
  assert.ok(body.classList.contains('remote-desktop'));
  // The pane's hide button sits at the right end of its tab row.
  const hide = computedOf(win, 'content-pane-hide');
  assert.strictEqual(hide.display, 'inline-flex');
  assert.strictEqual(hide.flexShrink, '0');
  // The chat header strip ("machine: directory"): centred, bold, green.
  const header = computedOf(win, 'chat-header');
  assert.strictEqual(header.justifyContent, 'center');
  assert.strictEqual(header.fontWeight, '700');
  assert.strictEqual(header.color, 'var(--green)');
  assert.strictEqual(computedOf(win, 'chat-machine').textAlign, 'center');
  // The pulse: an active panel's header text, tool call and Thoughts
  // alike, breathes.
  send(win, {type: 'tool_call', name: 'Bash', command: 'sleep 1', ts: 1000});
  const active = byId(win, 'output').querySelector('.tc.panel-active');
  assert.ok(active, 'the call is active until its result');
  assert.ok(
    /^panel-pulse\b/.test(win.getComputedStyle(active.querySelector('.tc-h')).animation),
    'the tool-call header pulses',
  );
  const thoughts = win.document.createElement('div');
  thoughts.className = 'panel-active';
  thoughts.innerHTML = '<div class="llm-panel-hdr">Thoughts</div>';
  win.document.body.appendChild(thoughts);
  assert.ok(
    /^panel-pulse\b/.test(win.getComputedStyle(thoughts.firstChild).animation),
    'the Thoughts header pulses too',
  );
  // The machine drop-down is gone while empty.
  assert.strictEqual(byId(win, 'chat-machine').options.length, 0);
  assert.strictEqual(computedOf(win, 'chat-machine').display, 'none');

  // No content tab: one chat column, clear of the docked panel; the
  // drawer, the show tab and the pane handle are not on screen.
  const app = () => computedOf(win, 'app');
  assert.strictEqual(app().gridTemplateColumns, 'minmax(0, 1fr)');
  assert.ok(/^var\(--meta-w/.test(app().marginRight));
  assert.strictEqual(computedOf(win, 'meta-drawer').display, 'none');
  assert.strictEqual(computedOf(win, 'content-pane-show').display, 'none');
  assert.strictEqual(computedOf(win, 'pane-resizer').display, 'none');
  // The panel is fixed and slides.
  assert.strictEqual(computedOf(win, 'meta-panel').position, 'fixed');
  assert.ok(/^transform\b/.test(computedOf(win, 'meta-panel').transition));
  assert.strictEqual(computedOf(win, 'meta-panel').transform, 'none');

  // A content tab: the split grid runs under the panel, the handle
  // shows, the panel's close button doubles as its "hide" over the pane,
  // and the drawer tab is faded out of the tab order.
  body.classList.add('content-pane-open');
  assert.strictEqual(app().marginRight, '0px');
  assert.ok(/'machine\s+handle ctabs'/.test(app().gridTemplateAreas));
  assert.strictEqual(computedOf(win, 'pane-resizer').display, 'block');
  assert.strictEqual(computedOf(win, 'meta-close').display, 'block');
  const drawer = computedOf(win, 'meta-drawer');
  assert.strictEqual(drawer.position, 'fixed');
  assert.strictEqual(drawer.right, '0px');
  assert.strictEqual(drawer.opacity, '0');
  assert.strictEqual(drawer.visibility, 'hidden', 'out of the tab order');
  assert.ok(/^opacity\b/.test(drawer.transition));
  // The panel slid away: the drawer tab comes on.
  body.classList.add('meta-hidden');
  const panel = computedOf(win, 'meta-panel');
  assert.strictEqual(panel.transform, 'translateX(100%)');
  assert.strictEqual(computedOf(win, 'meta-drawer').opacity, '1');
  assert.strictEqual(computedOf(win, 'meta-drawer').visibility, 'visible');

  // The pane folded away: the show tab sits on the chat's right edge
  // against the docked panel (jsdom leaves the var() fallback as text).
  body.classList.remove('content-pane-open', 'meta-hidden');
  body.classList.add('content-pane-folded');
  assert.strictEqual(computedOf(win, 'content-pane-show').position, 'fixed');
  assert.strictEqual(
    computedOf(win, 'content-pane-show').right,
    'var(--meta-w, var(--meta-panel-w))',
  );
  win.close();

  // The pulse holds still under prefers-reduced-motion, a media feature
  // jsdom cannot be put into; that one rule is read from the text.
  const main = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  assert.ok(/@keyframes panel-pulse/.test(main));
  assert.ok(/animation:\s*none/.test(cssRule(main, '.panel-active > .tc-h')));
});

async function main() {
  let failed = 0;
  for (const {name, fn} of TESTS) {
    try {
      await fn();
      console.log('  ok - ' + name);
    } catch (e) {
      failed++;
      console.log('  FAIL - ' + name);
      console.log('      ' + (e.stack || e.message));
    }
  }
  console.log(`${TESTS.length - failed} passed, ${failed} failed`);
  if (failed) process.exit(1);
}

main();
