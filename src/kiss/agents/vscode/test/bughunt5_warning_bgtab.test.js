// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Bug-hunt 5 integration test: a LIVE ``warning`` event stamped for a
// BACKGROUND tab must be routed into that tab's saved output fragment
// (like every other display event), not silently dropped.
//
// Bug locked in:
//
//   The top-level message switch in ``media/main.js`` routes
//   tab-stamped display events for non-active tabs through its
//   ``default:`` branch into ``processOutputEventForBgTab`` so the
//   event lands in the background tab's ``outputFragment`` and is
//   visible when the user switches to that tab.  ``case 'warning'``
//   however did a plain ``break`` on tabId mismatch — so when a
//   worktree task finished with a stash-pop warning while the user
//   was viewing ANOTHER tab, the warning never reached the owning
//   tab's transcript: switching to the tab showed the task's result
//   but not the warning that the user's uncommitted changes are
//   stuck in the git stash.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bughunt5_warning_bgtab.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
 */
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function testBackgroundTabWarningSurvivesTabSwitch() {
  const {win} = makeWebview();
  const api = win._demoApi;
  assert.ok(api, '_demoApi must be exposed by main.js');

  const tab1 = api.getActiveTabId();
  assert.ok(tab1, 'initial tab id must exist');

  // Open a second tab — tab1 becomes a background tab.
  api.createNewTab();
  const tab2 = api.getActiveTabId();
  assert.ok(tab2 && tab2 !== tab1, 'a fresh second tab must be active');

  // Tab1's worktree task streams its final events while the user is
  // looking at tab2: an ordinary display event (control) and the
  // stash-pop warning, both stamped with tab1's id.
  send(win, {type: 'system_output', text: 'control-sysout-QQQ7', tabId: tab1});
  send(win, {
    type: 'warning',
    message: 'bg stash-pop warning ZZZ9',
    tabId: tab1,
  });

  // Neither event may leak into the ACTIVE (tab2) transcript.
  const activeText = win.document.getElementById('output').textContent;
  assert.ok(
    !activeText.includes('control-sysout-QQQ7') &&
      !activeText.includes('bg stash-pop warning ZZZ9'),
    'background-tab events must not render in the active tab',
  );

  // Switch back to tab1 by clicking its tab-bar element (the real
  // user gesture) — restoreTab re-attaches tab1's outputFragment.
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
  // A warning stamped for a tab of ANOTHER VS Code window (no local
  // tab with that id) must still be dropped — regression guard for
  // the bughunt3 foreign-tab behaviour.
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
