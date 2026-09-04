// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for three stale-flash-timer races in
// media/main.js.  Each site armed a bare setTimeout to end a visual
// flash and never cleared it, so re-triggering the flash while one
// was showing let the FIRST trigger's stale timer cut the second
// flash short:
//
//  1. flashShareBtn — the share button's green/red flash after a
//     share_done message (2000 ms).  A second share_done during the
//     flash was cut short; an ok flash chased by an error flash also
//     left both classes stacked because the new class was added
//     without removing the old one.
//  2. The remote-URL bar's copy button (1500 ms icon revert).
//  3. The sidebar history copy button: wireCopyButton took a
//     resetFlashTimer flag and makeSidebarCopyButton passed false,
//     keeping the race there while the ids-copy buttons (flag true)
//     were already fixed.  The flag is gone: every caller restarts
//     the timer.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function makeWebview() {
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

  const clipboardWrites = [];
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: text => {
        clipboardWrites.push(String(text));
        return Promise.resolve();
      },
    },
  });

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=audit0903-main.js',
  );

  return {win, posted, clipboardWrites};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

async function testShareFlashSurvivesSecondShareDone() {
  const {win} = makeWebview();
  const shareBtn = win.document.getElementById('share-btn');
  assert.ok(shareBtn, '#share-btn must exist');

  send(win, {type: 'share_done', ok: true, path: 'reports/a.html'});
  assert.ok(shareBtn.classList.contains('share-ok'), 'first flash shows');

  // A second share completes while the first flash is still showing.
  await sleep(1300);
  send(win, {type: 'share_done', ok: true, path: 'reports/b.html'});
  assert.ok(shareBtn.classList.contains('share-ok'), 'second flash shows');

  // 2000 ms after the FIRST share_done its stale timer would fire here.
  await sleep(900);
  assert.ok(
    shareBtn.classList.contains('share-ok'),
    "the first share's stale timer must not end the second flash early",
  );

  await sleep(1300);
  assert.ok(
    !shareBtn.classList.contains('share-ok'),
    'the flash must still end after the second timer',
  );
  win.close();
  console.log('  ok - the share flash survives a second share_done');
}

async function testShareErrorFlashReplacesOkFlash() {
  const {win} = makeWebview();
  const shareBtn = win.document.getElementById('share-btn');

  send(win, {type: 'share_done', ok: true, path: 'reports/a.html'});
  assert.ok(shareBtn.classList.contains('share-ok'));

  await sleep(300);
  send(win, {type: 'share_done', ok: false});
  assert.ok(
    shareBtn.classList.contains('share-err'),
    'the error flash must show',
  );
  assert.ok(
    !shareBtn.classList.contains('share-ok'),
    'the superseded ok flash must not stay stacked under the error flash',
  );
  await sleep(2200);
  assert.ok(
    !shareBtn.classList.contains('share-err') &&
      !shareBtn.classList.contains('share-ok'),
    'the error flash ends on its own timer',
  );
  win.close();
  console.log('  ok - an error flash replaces an ok flash cleanly');
}

async function testRemoteUrlCopySurvivesSecondClick() {
  const {win, clipboardWrites} = makeWebview();
  send(win, {type: 'remote_url', url: 'https://example.test/chat'});
  const bar = win.document.getElementById('remote-url');
  const btn = bar && bar.querySelector('.remote-url-copy');
  assert.ok(btn, 'the remote-url bar must render a copy button');

  const isCheck = () => !!btn.querySelector('svg polyline');

  click(win, btn);
  await sleep(20);
  assert.deepStrictEqual(clipboardWrites, ['https://example.test/chat']);
  assert.ok(isCheck(), 'first click shows the check icon');

  await sleep(1200);
  click(win, btn);
  await sleep(20);
  assert.ok(isCheck(), 'second click shows the check icon');

  // 1500 ms after the FIRST click its stale timer would fire here.
  await sleep(500);
  assert.ok(
    isCheck(),
    "the first click's stale timer must not revert the second flash early",
  );

  await sleep(1200);
  assert.ok(!isCheck(), 'the icon must still revert after the second timer');
  win.close();
  console.log('  ok - the remote-url copy flash survives a second click');
}

const WS = '/Users/koushik/work/repo';

function loadHistoryRow(win) {
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  const ws = win.document.getElementById('hf-workspace');
  if (ws && ws.checked) {
    ws.checked = false;
    ws.dispatchEvent(new win.Event('change', {bubbles: true}));
  }
  send(win, {
    type: 'history',
    sessions: [
      {
        id: 'chat-a',
        task_id: 'task-a',
        title: 'row A',
        preview: 'row A preview',
        timestamp: 1_700_000_000,
        has_events: false,
        failed: false,
        is_running: false,
        tokens: 100,
        cost: 0.01,
        steps: 1,
        is_favorite: false,
        work_dir: WS,
        model: 'gpt-5',
        is_worktree: true,
        is_parallel: false,
        auto_commit_mode: true,
        startTs: 1_700_000_000_000,
        endTs: 1_700_000_005_000,
      },
    ],
    offset: 0,
  });
  const row = win.document.querySelector('#history-list .sidebar-item');
  assert.ok(row, 'the history row must render');
  return row;
}

async function testSidebarCopySurvivesSecondClick() {
  const {win, clipboardWrites} = makeWebview();
  const row = loadHistoryRow(win);
  const btn = row.querySelector('.sidebar-item-copy');
  assert.ok(btn, 'the history row must render a copy button');

  click(win, btn);
  await sleep(20);
  assert.deepStrictEqual(clipboardWrites, ['row A preview']);
  assert.ok(btn.classList.contains('copied'), 'first click flashes');

  await sleep(1200);
  click(win, btn);
  await sleep(20);
  assert.ok(btn.classList.contains('copied'), 'second click flashes');

  // 1500 ms after the FIRST click its stale timer would fire here.
  await sleep(500);
  assert.ok(
    btn.classList.contains('copied'),
    "the first click's stale timer must not end the second flash early",
  );

  await sleep(1200);
  assert.ok(
    !btn.classList.contains('copied'),
    'the flash must still revert after the second timer',
  );
  win.close();
  console.log('  ok - the sidebar copy flash survives a second click');
}

async function testSidebarCopyFallbackStillWorks() {
  const {win} = makeWebview();
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: undefined,
  });
  const execCalls = [];
  win.document.execCommand = function (cmd) {
    execCalls.push(cmd);
    return true;
  };
  const row = loadHistoryRow(win);
  const btn = row.querySelector('.sidebar-item-copy');
  click(win, btn);
  await sleep(20);
  assert.deepStrictEqual(execCalls, ['copy']);
  assert.ok(btn.classList.contains('copied'), 'the fallback path flashes');
  win.close();
  console.log('  ok - the sidebar copy execCommand fallback still works');
}

async function main() {
  await testShareFlashSurvivesSecondShareDone();
  await testShareErrorFlashReplacesOkFlash();
  await testRemoteUrlCopySurvivesSecondClick();
  await testSidebarCopySurvivesSecondClick();
  await testSidebarCopyFallbackStillWorks();
  console.log('audit0903_main_flash_timers.test.js: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
