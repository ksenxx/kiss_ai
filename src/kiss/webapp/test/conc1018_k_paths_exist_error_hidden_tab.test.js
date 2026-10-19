// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console */

// A hidden tab's transcript is parked in a detached fragment, so a
// `pathsExist` reply can only find its candidate spans through the
// registry (_pendingFileLinkSpans): a document query does not see
// them.  When one check of the tab's machine comes back refused
// (`error`, no `results`), the release must drop the stuck dedup keys
// but keep the still-live candidates registered, or a valid reply for
// another outstanding check of the same tab can no longer promote its
// link and that link stays inert for the rest of the session.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

const HOME = 'home';
const BOX = 'box';

function makeTwoTabPage() {
  const page = makePage();
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: HOME, chatId: 'chat-home', title: 'home', workDir: '/home/w'},
      {tabId: BOX, chatId: 'chat-box', title: 'box', workDir: '/box/work'},
    ],
  });
  send(page.win, {type: 'tabMachine', tabId: BOX, machine: 'box'});
  page.win._testApi.switchToTab(HOME);
  page.win._testApi.endLaunch();
  assert.strictEqual(page.win._testApi.getActiveTabId(), HOME);
  return page;
}

function mention(win, text) {
  send(win, {type: 'prompt', text, tabId: BOX});
  send(win, {type: 'text_end', tabId: BOX});
}

function testRefusedCheckKeepsOtherHiddenCandidatesResolvable() {
  const {win, posted} = makeTwoTabPage();
  mention(win, 'see src/failed.py please');
  mention(win, 'see src/good.py please');
  const checks = posted.filter(m => m.type === 'checkPaths');
  assert.strictEqual(checks.length, 2, 'two separate checkPaths requests');
  assert.deepStrictEqual(Array.from(checks[0].paths), ['src/failed.py']);
  assert.deepStrictEqual(Array.from(checks[1].paths), ['src/good.py']);
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
  assert.strictEqual(
    win._testApi.pendingFileLinkCounts().checks,
    0,
    "the refused machine's dedup keys are released",
  );
  assert.strictEqual(
    win._testApi.pendingFileLinkCounts().spans,
    2,
    'the live candidates stay registered: no reply has resolved them yet',
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
    win._testApi.pendingFileLinkCounts().spans,
    1,
    'only the unresolved candidate remains registered',
  );

  // The machine is back: a new mention re-checks the refused path (its
  // key was released) and the reply promotes the old span too.
  mention(win, 'again src/failed.py');
  const after = posted.filter(m => m.type === 'checkPaths');
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
  assert.strictEqual(win._testApi.pendingFileLinkCounts().spans, 0);
  win.close();
  console.log(
    "  ok - a refused check does not strand a hidden tab's other candidates",
  );
}

testRefusedCheckKeepsOtherHiddenCandidatesResolvable();
