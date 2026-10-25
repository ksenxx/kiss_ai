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
// stay registered so the next reply that covers their path resolves
// them -- which matters most for a hidden tab, whose transcript is
// parked in a detached fragment that a document query does not see:
// only the registry (_pendingFileLinkSpans) can find its spans.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {pendingCounts} = require('./file_link_helpers');

const HOME = 'home';
const BOX = 'box';

// A page whose tab BOX is bound to machine `box`; with `hidden`, a
// second tab HOME is on screen and BOX is parked.
function makeBoundPage(hidden) {
  const page = makePage();
  const tabs = [{tabId: BOX, chatId: 'chat-box', title: 'box', workDir: '/box/work'}];
  if (hidden) {
    tabs.unshift({tabId: HOME, chatId: 'chat-home', title: 'home', workDir: '/home/w'});
  }
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {type: 'tabs_state', tabs});
  send(page.win, {type: 'tabMachine', tabId: BOX, machine: 'box'});
  if (hidden) {
    page.win._testApi.switchToTab(HOME);
    page.win._testApi.endLaunch();
    assert.strictEqual(page.win._testApi.getActiveTabId(), HOME);
  }
  return page;
}

function checks(posted) {
  return posted.filter(m => m.type === 'checkPaths');
}

function assertCounts(win, spans, checks_, message) {
  assert.deepStrictEqual(pendingCounts(win), {spans, checks: checks_}, message);
}

function mention(win, text) {
  send(win, {type: 'prompt', text, tabId: BOX});
  send(win, {type: 'text_end', tabId: BOX});
}

function testErrorReplyReleasesTheCheck() {
  const {win, posted} = makeBoundPage(false);
  mention(win, 'see src/box.py please');
  const first = checks(posted);
  assert.strictEqual(first.length, 1, 'one checkPaths for the first panel');
  assert.strictEqual(first[0].machine, 'box');
  assertCounts(win, 1, 1, 'the first check is in flight');

  send(win, {
    type: 'pathsExist',
    tabId: BOX,
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
    tabId: BOX,
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
  const {win, posted} = makeBoundPage(false);
  mention(win, 'see src/box.py please');
  assert.strictEqual(checks(posted).length, 1);
  send(win, {
    type: 'pathsExist',
    tabId: BOX,
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

function testRefusedCheckKeepsOtherHiddenCandidatesResolvable() {
  const {win, posted} = makeBoundPage(true);
  mention(win, 'see src/failed.py please');
  mention(win, 'see src/good.py please');
  const sent = checks(posted);
  assert.strictEqual(sent.length, 2, 'two separate checkPaths requests');
  assert.deepStrictEqual(Array.from(sent[0].paths), ['src/failed.py']);
  assert.deepStrictEqual(Array.from(sent[1].paths), ['src/good.py']);
  assert.strictEqual(win._testApi.getActiveTabId(), HOME, 'box stays hidden');
  assert.strictEqual(
    win.document.querySelectorAll('#output [data-path-candidate]').length,
    0,
    "the hidden tab's spans are not in the document",
  );

  // The first check is refused; the second is still in flight.
  send(win, {
    type: 'pathsExist',
    tabId: BOX,
    machine: 'box',
    error: 'timed out',
  });
  assertCounts(
    win,
    2,
    0,
    "the refused machine's dedup keys are released; the live candidates " +
      'stay registered: no reply has resolved them yet',
  );

  // The other check's reply arrives and must still promote its link.
  send(win, {
    type: 'pathsExist',
    tabId: BOX,
    machine: 'box',
    workDir: '/box/work',
    results: {'src/good.py': true},
  });
  win._testApi.switchToTab(BOX);
  const good = win.document.querySelector('#output [data-path="src/good.py"]');
  assert.ok(good, 'the valid reply promotes src/good.py in the hidden tab');
  assert.ok(good.classList.contains('kiss-filelink'));
  assert.ok(!good.hasAttribute('data-path-candidate'));
  const failed = win.document.querySelector(
    '#output [data-path-candidate="src/failed.py"]',
  );
  assert.ok(failed, 'the refused path stays an unverified candidate');
  assert.strictEqual(
    pendingCounts(win).spans,
    1,
    'only the unresolved candidate remains registered',
  );

  // The machine is back: a new mention re-checks the refused path (its
  // key was released) and the reply promotes the old span too.
  mention(win, 'again src/failed.py');
  const after = checks(posted);
  assert.strictEqual(after.length, 3, 'the refused path is checked again');
  assert.deepStrictEqual(Array.from(after[2].paths), ['src/failed.py']);
  send(win, {
    type: 'pathsExist',
    tabId: BOX,
    machine: 'box',
    workDir: '/box/work',
    results: {'src/failed.py': true},
  });
  assert.strictEqual(
    win.document.querySelectorAll('#output [data-path="src/failed.py"]').length,
    2,
    'both mentions of the once-refused path are clickable',
  );
  assert.strictEqual(pendingCounts(win).spans, 0);
  win.close();
  console.log(
    "  ok - a refused check does not strand a hidden tab's other candidates",
  );
}

testErrorReplyReleasesTheCheck();
testErrorOfAnotherMachineKeepsTheCheck();
testRefusedCheckKeepsOtherHiddenCandidatesResolvable();
