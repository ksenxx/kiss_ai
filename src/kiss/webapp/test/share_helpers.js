// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// Helpers shared by the share-page suites (shareChat, shareSubagents,
// shareThemeToggle): clicking the share button, answering the page's
// shareChatTasks request, and running media/share.js in the document
// _build_share_page (web_server.py) writes.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {MEDIA, send, lastMsg, click: clickIn} = require('./simplify2_harness');

const SHARE_JS = fs.readFileSync(path.join(MEDIA, 'share.js'), 'utf8');

/** Dispatch a bubbling, cancelable click on `el`. */
function click(el) {
  clickIn(el.ownerDocument.defaultView, el);
}

/**
 * Click the share button, answer the page's shareChatTasks request with
 * `tasks` (plus the fields of `replyExtra`), and return the resulting
 * shareChat command (or undefined when none was sent).
 */
function shareWithTasks(wv, tasks, replyExtra) {
  const win = wv.win;
  click(win.document.getElementById('share-btn'));
  const req = lastMsg(wv.posted, 'shareChatTasks');
  assert.ok(req, 'clicking share must ask the daemon for the chat tasks');
  send(win, {
    type: 'share_tasks',
    tabId: req.tabId,
    chatId: req.chatId,
    tasks: tasks || [],
    truncated: false,
    ...(replyExtra || {}),
  });
  return lastMsg(wv.posted, 'shareChat') || undefined;
}

/**
 * Load share.js into a fresh page whose #app holds `bodyHtml`, the
 * exact document shape _build_share_page writes.  `sourceUrl` labels
 * the script in stack traces.
 */
function makeSharePage(bodyHtml, sourceUrl = 'share.js') {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><div id="app">' +
      bodyHtml +
      '</div></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true},
  );
  const win = dom.window;
  win.scrollTo = function () {};
  win.eval(SHARE_JS + '\n//# sourceURL=' + sourceUrl);
  return win;
}

module.exports = {click, shareWithTasks, makeSharePage};
