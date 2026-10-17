// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/** chat.html with its placeholders filled and its script tags stripped. */
function chatHtml() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  return html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
}

/**
 * A fresh jsdom window on chat.html with no page script loaded yet and
 * the scroll methods jsdom lacks stubbed out.
 *
 * @param {string} [runScripts='dangerously'] The jsdom `runScripts` mode.
 * @returns {object} The window.
 */
function newDom(runScripts = 'dangerously') {
  const dom = new JSDOM(chatHtml(), {
    runScripts,
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  return win;
}

/**
 * Build a jsdom chat page running the real panelCopy.js + api.js +
 * main.js over a recording transport.
 *
 * @param {{beforeScripts?: Function, sourceUrl?: string}} [opts]
 *   `beforeScripts(win)` runs after the transport is installed and
 *   before the scripts are eval'd (stub browser APIs there);
 *   `sourceUrl` names the eval'd main.js for V8 (`//# sourceURL=…`), which
 *   the `*.coverage.js` gates use to find its coverage entries.
 * @returns {{win: object, posted: Array<object>}} The window and every
 *   message the page posted, in order.
 */
function makePage(opts) {
  const win = newDom();
  const posted = [];
  win.acquireTransport = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  if (opts && opts.beforeScripts) opts.beforeScripts(win);

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  const sourceUrl = opts && opts.sourceUrl;
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      (sourceUrl ? '\n//# sourceURL=' + sourceUrl : ''),
  );

  return {win, posted};
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
  const posted = [];
  let state = {
    tabs: [
      {title: 'new chat', chatId: tabId, backendChatId: '', parentTabId: ''},
    ],
    activeTabIndex: 0,
    chatId: tabId,
  };
  win.acquireTransport = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  for (const name of ['panelCopy.js', 'api.js', 'main.js']) {
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

module.exports = {
  chatHtml,
  newDom,
  makePage,
  makePreloadedPage,
  loadMainCss,
  send,
  sleep,
  stubRaf,
  waitFor,
};
