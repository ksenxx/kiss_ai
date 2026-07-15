// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Regression tests for the error/notice/warning banner rendering in
// media/main.js (handleEvent cases 'error' / 'notice' / 'warning').
// Locks in the exact DOM (class list, <strong> label, HTML-escaped
// body) and the tab-id gating so the banner helpers can be
// deduplicated without any visible change.
//
//     node src/kiss/agents/vscode/test/simplify2_banners.test.js
'use strict';

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness');

function run() {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const activeId = ready.tabId;
  const O = win.document.getElementById('output');

  // 1. Un-stamped (broadcast) error renders an err banner, escaped.
  send(win, {type: 'error', text: 'boom <b>&</b>'});
  let banner = O.querySelector('div.ev.tr.err');
  assert.ok(banner, 'error event renders div.ev.tr.err');
  assert.strictEqual(
    banner.innerHTML,
    '<strong>Error:</strong> boom &lt;b&gt;&amp;&lt;/b&gt;',
    'error banner: bold label + HTML-escaped text',
  );

  // 2. Tab-stamped error for the ACTIVE tab renders too.
  send(win, {type: 'error', tabId: activeId, text: 'active-err'});
  assert.strictEqual(
    O.querySelectorAll('div.ev.tr.err').length,
    2,
    'active-tab error renders a second err banner',
  );

  // 3. Error stamped for a FOREIGN tab is dropped.
  send(win, {type: 'error', tabId: 'foreign-tab', text: 'foreign-err'});
  assert.strictEqual(
    O.querySelectorAll('div.ev.tr.err').length,
    2,
    'foreign-tab error must be dropped',
  );

  // 4. Notice banner.
  send(win, {type: 'notice', text: 'note <i>here</i>'});
  banner = O.querySelector('div.ev.tr.note');
  assert.ok(banner, 'notice event renders div.ev.tr.note');
  assert.strictEqual(
    banner.innerHTML,
    '<strong>Note:</strong> note &lt;i&gt;here&lt;/i&gt;',
    'notice banner: bold label + HTML-escaped text',
  );

  // 5. Warning banner (backend sends ``message``; ``text`` fallback).
  send(win, {type: 'warning', message: 'warn <x>'});
  banner = O.querySelector('div.ev.tr.warn');
  assert.ok(banner, 'warning event renders div.ev.tr.warn');
  assert.strictEqual(
    banner.innerHTML,
    '<strong>Warning:</strong> warn &lt;x&gt;',
    'warning banner: bold label + HTML-escaped ev.message',
  );
  send(win, {type: 'warning', text: 'warn-text-only'});
  const warns = O.querySelectorAll('div.ev.tr.warn');
  assert.strictEqual(warns.length, 2, 'text-only warning also renders');
  assert.strictEqual(
    warns[1].innerHTML,
    '<strong>Warning:</strong> warn-text-only',
    'warning banner falls back to ev.text',
  );

  win.close();
  console.log('  ok - error/notice/warning banners render identically');
}

run();
console.log('simplify2_banners.test.js: all tests passed');
