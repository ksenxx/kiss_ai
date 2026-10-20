// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function testWarningEventIsRendered() {
  const {win} = makePage();

  const msg =
    'Failed to restore your uncommitted changes: `git stash pop` ' +
    'conflicted. Your changes are preserved in the git stash.';
  send(win, {type: 'warning', message: msg});

  const output = win.document.getElementById('output');
  const text = output ? output.textContent : '';
  assert.ok(
    text.includes('git stash pop'),
    'BUG: a backend warning event (stash-pop failure) was silently ' +
      'dropped by the page — the user never learns their ' +
      'uncommitted changes are stuck in the stash',
  );
  const banners = output.querySelectorAll('.warn');
  assert.strictEqual(
    banners.length,
    1,
    'an active-tab warning renders exactly once, got ' + banners.length,
  );
  win.close();
  console.log('  ok - warning event renders once in the transcript');
}

function testWarningEscapesHtml() {
  const {win} = makePage();

  send(win, {
    type: 'warning',
    message: 'conflict in <img src=x onerror=alert(1)> & "branch"',
  });

  const output = win.document.getElementById('output');
  assert.ok(output, 'output container must exist');
  assert.strictEqual(
    // The welcome page's own logos are the only legitimate images here.
    output.querySelector('img:not(.welcome-logo)'),
    null,
    'BUG: warning message HTML was not escaped (XSS)',
  );
  assert.ok(
    output.textContent.includes('<img src=x onerror=alert(1)> & "branch"'),
    'warning text must be shown verbatim (escaped)',
  );
  win.close();
  console.log('  ok - warning message is HTML-escaped');
}

function testForeignTabWarningNotRendered() {
  const {win} = makePage();

  send(win, {
    type: 'warning',
    message: 'foreign-window stash warning',
    tabId: 'some-other-window-tab',
  });

  const output = win.document.getElementById('output');
  const text = output ? output.textContent : '';
  assert.ok(
    !text.includes('foreign-window stash warning'),
    'a warning stamped for another tab must not render in this tab',
  );
  win.close();
  console.log('  ok - foreign-tab warning is not rendered');
}

function runTests() {
  testWarningEventIsRendered();
  testWarningEscapesHtml();
  testForeignTabWarningNotRendered();
}

try {
  runTests();
  console.log('\n3 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
