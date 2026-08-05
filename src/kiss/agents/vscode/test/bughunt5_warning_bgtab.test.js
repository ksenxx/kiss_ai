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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

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
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function testBackgroundTabWarningSurvivesTabSwitch() {
  const {win} = makeWebview();
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');

  const tab1 = api.getActiveTabId();
  assert.ok(tab1, 'initial tab id must exist');

  api.createNewTab();
  const tab2 = api.getActiveTabId();
  assert.ok(tab2 && tab2 !== tab1, 'a fresh second tab must be active');

  send(win, {type: 'system_output', text: 'control-sysout-QQQ7', tabId: tab1});
  send(win, {
    type: 'warning',
    message: 'bg stash-pop warning ZZZ9',
    tabId: tab1,
  });

  const activeText = win.document.getElementById('output').textContent;
  assert.ok(
    !activeText.includes('control-sysout-QQQ7') &&
      !activeText.includes('bg stash-pop warning ZZZ9'),
    'background-tab events must not render in the active tab',
  );

  const tabEl = win.document.querySelector(
    '.chat-tab[data-tab-id="' + tab1 + '"]',
  );
  assert.ok(tabEl, 'tab1 element must exist in the tab bar');
  tabEl.click();
  assert.strictEqual(api.getActiveTabId(), tab1, 'tab1 must be active now');

  const text = win.document.getElementById('output').textContent;
  assert.ok(
    text.includes('control-sysout-QQQ7'),
    'control display event must be in the restored transcript ' +
      '(default bg-tab route)',
  );
  assert.ok(
    text.includes('bg stash-pop warning ZZZ9'),
    'BUG: a live warning for a background tab was dropped instead of ' +
      'being routed into the tab outputFragment — the user never sees ' +
      'that their uncommitted changes are stuck in the git stash',
  );
  win.close();
  console.log('  ok - bg-tab warning survives switching to the tab');
}

function testForeignWindowTabWarningStillDropped() {
  const {win} = makeWebview();
  send(win, {
    type: 'warning',
    message: 'foreign-window stash warning',
    tabId: 'some-other-window-tab',
  });
  const text = win.document.getElementById('output').textContent;
  assert.ok(
    !text.includes('foreign-window stash warning'),
    'a warning for an unknown (foreign-window) tab must not render',
  );
  win.close();
  console.log('  ok - foreign-window tab warning still dropped');
}

function testActiveTabWarningStillRendersOnce() {
  const {win} = makeWebview();
  send(win, {type: 'warning', message: 'active live warning AAA1'});
  const banners = win.document.querySelectorAll('#output .warn');
  assert.strictEqual(banners.length, 1, 'active-tab warning renders once');
  assert.ok(
    banners[0].textContent.includes('active live warning AAA1'),
    'active-tab warning text must render',
  );
  win.close();
  console.log('  ok - active-tab live warning still renders exactly once');
}

function runTests() {
  testBackgroundTabWarningSurvivesTabSwitch();
  testForeignWindowTabWarningStillDropped();
  testActiveTabWarningStillRendersOnce();
}

try {
  runTests();
  console.log('\n3 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
