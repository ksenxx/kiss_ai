// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// The page scripts in the order chat.html loads them.  makePage evals the
// three the chat always needs plus whatever `extraScripts` names, in this
// order, so a suite gets the same globals the browser page has.
const PAGE_SCRIPTS = [
  'highlight.min.js',
  'marked.min.js',
  'panelCopy.js',
  'tips.js',
  'api.js',
  'contentContextMenu.js',
  'treeContextMenu.js',
  'browserTab.js',
  'terminalTab.js',
  'pdfView.js',
  'mobileSwipe.js',
  'appWindow.js',
  'main.js',
  'voice.js',
];
const DEFAULT_SCRIPTS = ['panelCopy.js', 'api.js', 'main.js'];

/**
 * chat.html with its placeholders filled and its script tags stripped.
 *
 * @param {{bodyClass?: string, modelName?: string,
 *   placeholders?: Object<string, string>}} [opts] `bodyClass` fills the
 *   `<body {{BODY_CLASS_ATTR}}>` placeholder (`'remote-chat'` builds the
 *   remote webapp page); `modelName` fills `{{MODEL_NAME}}` (default
 *   `'test-model'`); `placeholders` fills further `{{NAME}}` slots
 *   (`{PRODUCT_NAME: 'KISS Sorcar'}`).  Every other placeholder is
 *   blanked.
 */
function chatHtml(opts) {
  const {bodyClass = '', modelName = 'test-model', placeholders} = opts || {};
  const fills = Object.assign(
    {
      MODEL_NAME: modelName,
      BODY_CLASS_ATTR: bodyClass ? ' class="' + bodyClass + '"' : '',
    },
    placeholders,
  );
  const html = fs
    .readFileSync(path.join(MEDIA, 'chat.html'), 'utf8')
    .replace(/\{\{([A-Z_]+)\}\}/g, (m, name) => fills[name] || '');
  return html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
}

/**
 * A fresh jsdom window on chat.html with no page script loaded yet and
 * the scroll methods jsdom lacks stubbed out.
 *
 * @param {string|{runScripts?: string, url?: string|null,
 *   userAgent?: string, bodyClass?: string, modelName?: string,
 *   placeholders?: object, virtualConsole?: object}} [opts] The jsdom
 *   `runScripts` mode (default `'dangerously'`), either alone or in an
 *   options bag with the `chatHtml` options, the page `url` (default
 *   `https://localhost/`; `null` opens an opaque `about:blank` origin
 *   whose `localStorage` throws), the `navigator.userAgent` to report
 *   (jsdom ignores its own `userAgent` option, so it is defined on the
 *   navigator) and a jsdom `VirtualConsole` to receive the page's
 *   console output and jsdom errors (a `location.reload()` surfaces
 *   there as a "Not implemented: navigation" error).
 * @returns {object} The window.
 */
function newDom(opts) {
  if (typeof opts === 'string') opts = {runScripts: opts};
  const {
    runScripts = 'dangerously',
    url = 'https://localhost/',
    userAgent,
    virtualConsole,
  } = opts || {};
  const domOpts = {runScripts, pretendToBeVisual: true};
  if (url) domOpts.url = url;
  if (virtualConsole) domOpts.virtualConsole = virtualConsole;
  const dom = new JSDOM(chatHtml(opts), domOpts);
  const win = dom.window;
  if (userAgent) {
    Object.defineProperty(win.navigator, 'userAgent', {
      value: userAgent,
      configurable: true,
    });
  }
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  return win;
}

/**
 * Install a recording transport on *win*: `posted` collects every
 * message the page posts and the persisted state starts as
 * *initialState*.
 *
 * @returns {{posted: Array<object>, getState: Function}} The message
 *   ledger and a reader of the state the page last persisted.
 */
function installRecordingTransport(win, initialState) {
  const posted = [];
  let state = initialState;
  win.acquireTransport = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  return {posted, getState: () => state};
}

/**
 * Build a jsdom chat page running the real panelCopy.js + api.js +
 * main.js over a recording transport.
 *
 * @param {{bodyClass?: string, modelName?: string, placeholders?: object,
 *   url?: string|null, userAgent?: string, initialState?: object,
 *   beforeScripts?: Function, extraScripts?: Array<string>,
 *   mainSource?: Function, sourceUrl?: string|Object<string, string>}}
 *   [opts]
 *   `bodyClass`, `modelName`, `placeholders`, `url` and `userAgent` go to
 *   `newDom`;
 *   `initialState` is what the transport's `getState()` returns before
 *   the page saves anything (a reload of an open chat);
 *   `beforeScripts(win)` runs after the transport is installed and
 *   before the scripts are eval'd (stub browser APIs there);
 *   `extraScripts` names further media scripts (`'marked.min.js'`,
 *   `'voice.js'`, …), eval'd with the defaults in chat.html order;
 *   `mainSource(src)` returns the main.js text to eval given the real
 *   one (a suite patching a constant);
 *   `sourceUrl` names the eval'd main.js for V8 (`//# sourceURL=…`), which
 *   the `*.coverage.js` gates use to find its coverage entries; an object
 *   names several scripts (`{'main.js': 'x-main.js', 'voice.js': …}`).
 * @returns {{win: object, posted: Array<object>, getState: Function}}
 *   The window, every message the page posted in order, and a reader of
 *   the state the page last persisted through the transport.
 */
function makePage(opts) {
  const {
    initialState,
    beforeScripts,
    extraScripts = [],
    mainSource,
    sourceUrl,
  } = opts || {};
  const win = newDom(opts);
  const {posted, getState} = installRecordingTransport(win, initialState);

  if (beforeScripts) beforeScripts(win);

  const unknown = extraScripts.find(name => !PAGE_SCRIPTS.includes(name));
  assert.ok(!unknown, unknown + ' is not a chat.html page script');
  const wanted = DEFAULT_SCRIPTS.concat(extraScripts);
  const sourceUrls =
    typeof sourceUrl === 'string' ? {'main.js': sourceUrl} : sourceUrl || {};
  for (const name of PAGE_SCRIPTS.filter(n => wanted.includes(n))) {
    let src = fs.readFileSync(path.join(MEDIA, name), 'utf8');
    if (name === 'main.js' && mainSource) src = mainSource(src);
    if (sourceUrls[name]) src += '\n//# sourceURL=' + sourceUrls[name];
    win.eval(src);
  }

  return {win, posted, getState};
}

/**
 * Build a chat page whose transport already holds one persisted tab
 * *tabId* (selected) and that has received the daemon's matching
 * `tabs_state` snapshot, as a reload of an open chat sees (an
 * `outside-only` jsdom: nothing but the test runs scripts in it).
 *
 * @param {string} tabId The persisted tab's id.
 * @returns {{win: object, posted: Array<object>, close: Function}} The
 *   window, every message it posted, and a closer for it.
 */
function makePreloadedPage(tabId) {
  const win = newDom('outside-only');
  const {posted} = installRecordingTransport(win, {
    tabs: [
      {title: 'new chat', chatId: tabId, backendChatId: '', parentTabId: ''},
    ],
    activeTabIndex: 0,
    chatId: tabId,
  });
  for (const name of DEFAULT_SCRIPTS) {
    win.eval(fs.readFileSync(path.join(MEDIA, name), 'utf8'));
  }
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId, chatId: '', title: 'new chat', workDir: ''}],
  });
  return {win, posted, close: () => win.close()};
}

/** Append media/main.css to *win*'s document, for computed-style asserts. */
function loadMainCss(win) {
  const styleEl = win.document.createElement('style');
  styleEl.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(styleEl);
}

/** Dispatch a bubbling click on `el` in `win`. */
function click(win, el) {
  const ev = new win.MouseEvent('click', {bubbles: true, cancelable: true});
  el.dispatchEvent(ev);
  return ev;
}

/**
 * Click the strip entry of an open tab, falling back to the page's
 * `switchToTab` when the strip does not list it (chat tabs have no
 * entry while the strip is hidden).
 */
function clickTab(win, tabId) {
  assert.ok(
    win._testApi.openTabs().some(t => t.id === tabId),
    `tab ${tabId} must be open`,
  );
  const el = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  if (el) click(win, el);
  else win._testApi.switchToTab(tabId);
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    `tab ${tabId} is on screen`,
  );
}

/** The element with `id`, asserting it exists. */
function byId(win, id) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  return node;
}

/**
 * A `beforeScripts` hook: stand a recording `window.monaco` in for the
 * CDN build jsdom never fetches, so `ensureMonaco()` resolves at once.
 * Each `editor.create` is recorded as `{holder, editor, value, language,
 * opts}`; `editor._type(text)` edits the model and fires its content
 * listeners.
 *
 * @returns {Array<object>} The record of every editor created.
 */
function stubMonaco(win) {
  const created = [];
  win.monaco = {
    editor: {
      // main.js defines a kiss-light/kiss-dark theme from the page palette
      // before every create(); real Monaco always exposes both calls.
      defineTheme: () => {},
      setTheme: () => {},
      create: (holder, opts) => {
        let value = String(opts.value === undefined ? '' : opts.value);
        let version = 1;
        const listeners = [];
        const model = {
          getAlternativeVersionId: () => version,
          getValue: () => value,
          getLineCount: () => value.split('\n').length,
        };
        const editor = {
          getModel: () => model,
          onDidChangeModelContent: cb => listeners.push(cb),
          setPosition: () => {},
          revealLineInCenter: () => {},
          getDomNode: () => holder,
          dispose: () => {},
          layout: () => {},
          focus: () => {},
          _type: text => {
            value = String(text);
            version += 1;
            listeners.slice().forEach(cb => cb());
          },
        };
        created.push({
          holder,
          editor,
          value: opts.value,
          language: opts.language,
          opts,
        });
        return editor;
      },
    },
  };
  return created;
}

/**
 * A `beforeScripts` hook: answer every `checkPaths` command the page
 * posts with the `pathsExist` reply the host would send, so file-path
 * links get linkified.  The command is still recorded in `posted`.
 *
 * @param {Function} [results] `results(msg)` returns the
 *   `{path: boolean}` map for one `checkPaths` message; by default every
 *   path exists.
 */
function answerCheckPaths(win, results) {
  const acquire = win.acquireTransport;
  win.acquireTransport = function () {
    const transport = acquire();
    const record = transport.postMessage;
    transport.postMessage = msg => {
      record(msg);
      if (msg.type !== 'checkPaths') return;
      send(win, {
        type: 'pathsExist',
        results: results
          ? results(msg)
          : Object.fromEntries(msg.paths.map(p => [p, true])),
        workDir: msg.workDir,
        tabId: msg.tabId,
      });
    };
    return transport;
  };
}

/**
 * A `beforeScripts` hook: record every `setTimeout` the page arms as
 * `{cb, ms}` while still arming it for real, so a suite can see which
 * delays main.js chose.
 *
 * @returns {Array<{cb: Function, ms: number}>} The timers, in order.
 */
function captureTimers(win) {
  const timers = [];
  const realSetTimeout = win.setTimeout;
  win.setTimeout = function (cb, ms, ...args) {
    timers.push({cb, ms});
    return realSetTimeout.call(win, cb, ms, ...args);
  };
  return timers;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

/**
 * A `beforeScripts` hook that runs animation frames synchronously, for
 * suites that assert right after a DOM update main.js defers to a frame.
 */
function stubRaf(win) {
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};
}

const DESKTOP_QUERY = '(min-width: 900px)';

/**
 * Install a fake window.matchMedia on which the desktop breakpoint
 * '(min-width: 900px)' reports `desktop` and every other query never
 * matches.  Install it before main.js is evaluated.
 *
 * @param {Window} win The jsdom window.
 * @param {boolean} desktop Whether the breakpoint matches at boot.
 * @returns {function(boolean)} `fire(matches)`: flip the breakpoint and
 *   notify the change listeners main.js registered on it.
 */
function fakeMatchMedia(win, desktop) {
  const listeners = [];
  const mql = {
    matches: desktop,
    media: DESKTOP_QUERY,
    addEventListener: (type, fn) => {
      if (type === 'change') listeners.push(fn);
    },
    removeEventListener: () => {},
    addListener: fn => listeners.push(fn),
    removeListener: () => {},
  };
  win.matchMedia = function (query) {
    if (query === DESKTOP_QUERY) return mql;
    return {
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  return function fire(matches) {
    mql.matches = matches;
    for (const fn of listeners.slice()) fn(mql);
  };
}

/**
 * Replace the window's setInterval/clearInterval with a virtual clock so
 * a periodic poll (the 5 s task-update poll) is fired deterministically
 * instead of waited out.  setTimeout stays real (message handlers
 * debounce with it).  Install it before main.js is evaluated.
 *
 * @param {Window} win The jsdom window.
 * @returns {{tick: function(number), intervals: Array}} `tick(ms)`
 *   advances the clock and runs every due interval callback in time
 *   order; `intervals` lists the live intervals as {id, cb, ms, next}.
 */
function fakeIntervals(win) {
  let now = 0;
  let nextId = 1;
  const intervals = [];
  win.setInterval = function (cb, ms, ...args) {
    const period = Number(ms) || 0;
    const iv = {id: nextId++, cb, args, ms: period, next: now + period};
    intervals.push(iv);
    return iv.id;
  };
  win.clearInterval = function (id) {
    const i = intervals.findIndex(iv => iv.id === id);
    if (i >= 0) intervals.splice(i, 1);
  };
  function tick(ms) {
    const target = now + ms;
    for (;;) {
      const due = intervals.filter(iv => iv.next <= target);
      if (due.length === 0) break;
      due.sort((a, b) => a.next - b.next);
      const iv = due[0];
      now = iv.next;
      iv.next += iv.ms;
      iv.cb(...iv.args);
    }
    now = target;
  }
  return {tick, intervals};
}

/**
 * Poll `predicate` every 20 ms until it returns a truthy value (returned)
 * or `timeoutMs` pass (throws `message`).
 */
async function waitFor(predicate, message, timeoutMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = predicate();
    if (value) return value;
    await sleep(20);
  }
  throw new Error(message || 'waitFor timed out');
}

/** The messages of `type` the page posted. */
function ofType(posted, type) {
  return posted.filter(m => m && m.type === type);
}

/** The last message of `type` the page posted, or null. */
function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

/** A `tabs_state` entry as the daemon's registry snapshot carries it. */
function snapshotEntry(tabId, title, chatId, workDir) {
  return {
    tabId: tabId,
    chatId: chatId || '',
    title: title || 'new chat',
    workDir: workDir || '',
  };
}

/** Resolve after `n` of the window's animation frames have run. */
function nextFrames(win, n = 3) {
  return new Promise(resolve => {
    let left = n;
    function step() {
      if (--left <= 0) return resolve();
      win.requestAnimationFrame(step);
    }
    win.requestAnimationFrame(step);
  });
}

/** Set `el.value` and dispatch the bubbling `input` event a keystroke would. */
function typeInto(win, el, value) {
  el.value = value;
  el.dispatchEvent(new win.Event('input', {bubbles: true}));
}

/** Whether `el` and every ancestor up to #output is displayed. */
function isDisplayed(win, el) {
  for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
    if (win.getComputedStyle(n).getPropertyValue('display').trim() === 'none')
      return false;
    if (n.id === 'output') break;
  }
  return true;
}

/** The tabId main.js posted in its `ready` message. */
function readyTabId(posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'the page posts ready with a tabId');
  return ready.tabId;
}

/** Send the configData of a chat with no working directory. */
function sendEmptyWorkDir(win) {
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
}

/** The transcript element. */
function output(win) {
  return win.document.getElementById('output');
}

/** The history-list rows keyed by their title text. */
function rowsByTitle(win) {
  const list = win.document.getElementById('history-list');
  const map = {};
  list.querySelectorAll('.sidebar-item').forEach(r => {
    const t = r.querySelector('.sidebar-item-text');
    if (t) map[t.textContent] = r;
  });
  return map;
}

/** Whether the page is answering an ask_user_question. */
function answering(win) {
  return win.document.body.classList.contains('ask-answering');
}

/** Pick `value` in the chat's machine drop-down. */
function pickMachine(win, value) {
  const select = byId(win, 'chat-machine');
  select.value = value;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/**
 * Open a chat on `task` the way the daemon does when the user submits a
 * prompt: an empty `task_events` for the chat, `status running`, then the
 * prompt echo.  `posted` is the page's ledger (for its ready tabId).
 */
function startTask(win, posted, task) {
  send(win, {
    type: 'task_events',
    events: [],
    task,
    tabId: readyTabId(posted),
    chat_id: 'chat-' + task.replace(/\W+/g, '-'),
  });
  send(win, {type: 'status', running: true});
  send(win, {type: 'prompt', text: task});
}

/**
 * What a collapse header reads as: its title and its preview span, which
 * the stylesheet sets apart with a margin, joined by one space.
 */
function headerText(hdr) {
  return Array.from(hdr.childNodes)
    .map(n => n.textContent.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .join(' ');
}

/** Fold `panel` (the streaming sweep may have already) and read its header. */
function collapsedHeaderText(panel) {
  const hdr = panel.querySelector('.collapse-header');
  if (!panel.classList.contains('collapsed')) hdr.click();
  assert.ok(panel.classList.contains('collapsed'));
  return headerText(hdr);
}

/**
 * The transcript of `win` serialized the way a shared page is and
 * re-driven by media/share.js in a fresh window, which is returned
 * (close it when done).  `win` is closed.
 */
function shareTranscript(win) {
  const html =
    '<!doctype html><html><body><div id="output">' +
    output(win).innerHTML +
    '</div></body></html>';
  win.close();
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    url: 'https://localhost/',
  });
  dom.window.eval(fs.readFileSync(path.join(MEDIA, 'share.js'), 'utf8'));
  return dom.window;
}

/**
 * The header text of every `.tc` panel on a shared page, each folded by
 * share.js itself: a panel the live sweep had folded is opened first and
 * its serialized preview overwritten, so the text is share.js's own.
 */
function foldedShareHeaders(sw) {
  return Array.from(sw.document.querySelectorAll('.tc')).map(p => {
    const hdr = p.querySelector('.collapse-header');
    if (p.classList.contains('collapsed')) hdr.click();
    hdr.querySelector('.collapse-preview').textContent = 'stale';
    hdr.click();
    assert.ok(p.classList.contains('collapsed'), 'share.js folds the panel');
    return headerText(hdr);
  });
}

/**
 * The html frames of the content views on screen (a background content
 * tab keeps its view in the DOM, hidden).
 */
function htmlFrames(win) {
  return Array.from(
    win.document.querySelectorAll('.content-html-frame'),
  ).filter(f => f.closest('.content-tab-view').style.display !== 'none');
}

/** The srcdoc of the one html frame on screen. */
function activeFrameDoc(win) {
  const fr = htmlFrames(win);
  assert.strictEqual(fr.length, 1, 'exactly one html frame on screen');
  return fr[0].srcdoc;
}

/** A panel's time-spent label ("1m 2.5s", "2.5s", "40ms") in ms, else NaN. */
function parsePanelTimeMs(text) {
  const t = String(text || '').trim();
  let m = t.match(/^(\d+)m\s+(\d+(?:\.\d+)?)s$/);
  if (m) return (parseInt(m[1], 10) * 60 + parseFloat(m[2])) * 1000;
  m = t.match(/^(\d+(?:\.\d+)?)s$/);
  if (m) return parseFloat(m[1]) * 1000;
  m = t.match(/^(\d+)ms$/);
  if (m) return parseInt(m[1], 10);
  return NaN;
}

/** Open a second chat tab; `{api, first, second}` with `second` active. */
function twoTabs(win) {
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  assert.ok(second && second !== first, 'a fresh second tab must be active');
  return {api, first, second};
}

/**
 * Mount a tips sheet showing `tips` (markdown strings) the way tips.js
 * does for the Tips button, and return its `<kiss-tips-panel>`.
 */
function showTips(win, tips) {
  const el = win.document.createElement('kiss-tips-panel');
  el.tips = tips;
  win.document.body.appendChild(el);
  return el;
}

module.exports = {
  showTips,
  MEDIA,
  newDom,
  makePage,
  makePreloadedPage,
  loadMainCss,
  click,
  clickTab,
  byId,
  send,
  sleep,
  stubRaf,
  stubMonaco,
  answerCheckPaths,
  captureTimers,
  fakeMatchMedia,
  fakeIntervals,
  waitFor,
  ofType,
  readyTabId,
  sendEmptyWorkDir,
  output,
  rowsByTitle,
  answering,
  pickMachine,
  lastMsg,
  snapshotEntry,
  nextFrames,
  isDisplayed,
  startTask,
  headerText,
  collapsedHeaderText,
  shareTranscript,
  foldedShareHeaders,
  htmlFrames,
  activeFrameDoc,
  parsePanelTimeMs,
  twoTabs,
  typeInto,
};
