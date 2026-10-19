// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console */

// A `checkPaths` forwarded to a registered machine that is unreachable
// comes back as `{type:'pathsExist', tabId, machine, error}` with no
// `results` (sidebar_panels.py `_machine_reply`).  main.js must then drop
// the in-flight dedup keys of that tab+machine, or every later panel of
// the tab that mentions the same path is never checked again and its
// links stay inert for the rest of the session.  The candidate spans
// stay registered (see conc1018_k_paths_exist_error_hidden_tab.test.js)
// so the next reply that covers their path resolves them.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

const BOX_TAB = 'tab-on-box';

function makeBoundPage() {
  const page = makePage();
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: BOX_TAB, chatId: 'chat-box', title: 'box', workDir: '/box/work'},
    ],
  });
  send(page.win, {type: 'tabMachine', tabId: BOX_TAB, machine: 'box'});
  return page;
}

function checks(posted) {
  return posted.filter(m => m.type === 'checkPaths');
}

// The probe's object comes from the page realm, whose Object.prototype
// deepStrictEqual would reject: compare the two numbers directly.
function assertCounts(win, spans, checks_, message) {
  const counts = win._testApi.pendingFileLinkCounts();
  assert.strictEqual(counts.spans, spans, message + ' (spans)');
  assert.strictEqual(counts.checks, checks_, message + ' (checks)');
}

function mention(win, text) {
  send(win, {type: 'prompt', text, tabId: BOX_TAB});
  send(win, {type: 'text_end', tabId: BOX_TAB});
}

function testErrorReplyReleasesTheCheck() {
  const {win, posted} = makeBoundPage();
  mention(win, 'see src/box.py please');
  const first = checks(posted);
  assert.strictEqual(first.length, 1, 'one checkPaths for the first panel');
  assert.strictEqual(first[0].machine, 'box');
  assertCounts(win, 1, 1, 'the first check is in flight');

  send(win, {
    type: 'pathsExist',
    tabId: BOX_TAB,
    machine: 'box',
    error: "no machine named 'box'",
  });
  assertCounts(win, 1, 0, 'the refused check is released; its span stays');
  const span = win.document.querySelector('#output [data-path-candidate]');
  assert.ok(span, 'the unverified span stays a candidate (not promoted)');
  assert.ok(!span.classList.contains('kiss-filelink'));

  mention(win, 'again src/box.py');
  const second = checks(posted);
  assert.strictEqual(
    second.length,
    2,
    'the same path is re-checked once the machine refused the first check',
  );
  assert.deepStrictEqual(Array.from(second[1].paths), ['src/box.py']);
  assert.strictEqual(second[1].machine, 'box');

  // The machine is back: its reply promotes both the new span and the
  // old one still on screen.
  send(win, {
    type: 'pathsExist',
    tabId: BOX_TAB,
    machine: 'box',
    workDir: '/box/work',
    results: {'src/box.py': true},
  });
  const links = win.document.querySelectorAll('#output [data-path]');
  assert.strictEqual(links.length, 2, 'both mentions are clickable links');
  win.close();
  console.log('  ok - a machine error reply releases the in-flight check');
}

function testErrorOfAnotherMachineKeepsTheCheck() {
  const {win, posted} = makeBoundPage();
  mention(win, 'see src/box.py please');
  assert.strictEqual(checks(posted).length, 1);
  send(win, {
    type: 'pathsExist',
    tabId: BOX_TAB,
    machine: 'other',
    error: 'unreachable',
  });
  assertCounts(
    win,
    1,
    1,
    "another machine's error leaves the box check in flight",
  );
  send(win, {
    type: 'pathsExist',
    tabId: 'some-other-tab',
    machine: 'box',
    error: 'unreachable',
  });
  assertCounts(win, 1, 1, "another tab's error leaves the box check in flight");
  mention(win, 'again src/box.py');
  assert.strictEqual(
    checks(posted).length,
    1,
    'the in-flight check still dedups the same path',
  );
  win.close();
  console.log("  ok - another owner's error reply does not release the check");
}

testErrorReplyReleasesTheCheck();
testErrorOfAnotherMachineKeepsTheCheck();
