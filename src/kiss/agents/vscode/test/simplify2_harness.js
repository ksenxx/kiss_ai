// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Shared jsdom harness for the simplify2_*.test.js regression suite.
// Boots the REAL chat webview (media/chat.html + media/panelCopy.js +
// media/main.js) exactly like test/bughunt2_status_timer.test.js does —
// no mocks of project code, only the VS Code host API stub every
// webview test uses.
'use strict';

const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview.
 *
 * Returns ``{win, posted}`` where ``posted`` records every message the
 * webview sends to the extension host via ``vscode.postMessage``.
 *
 * ``opts.beforeScripts(win)`` — optional hook run after the DOM is
 * built but before panelCopy.js/main.js are evaluated, so a test can
 * add markup that other hosts of main.js provide (e.g. the remote
 * webapp's #frequent-tasks-btn) without mocking any project code.
 */
function makeWebview(opts) {
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

  if (opts && opts.beforeScripts) opts.beforeScripts(win);

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Promise that resolves after ``ms`` milliseconds. */
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

module.exports = {makeWebview, send, sleep};
