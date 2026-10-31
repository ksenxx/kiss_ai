// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Wiring test for the autocomplete tab stamp over the page's direct
// transport (media/main.js -> media/api.js -> postMessage).
//
// The page stamps `tabId` on getFiles / complete so the daemon's reply
// can be matched against the conversation on screen: a reply without
// the stamp, or stamped with another tab, is rejected by the page's
// guard (isForActiveTab), which would kill the @-mention picker and
// ghost text outright if the stamp were ever dropped on the way out.
// So this test checks both ends of the page: the commands carry the
// active tab's id, and only replies echoing it render.

'use strict';

const assert = require('assert');
const {makePage, send, waitFor, recordTimers, fireTimers} = require('./simplify2_harness.js');

// requestGhost()'s 300 ms debounce is recorded and fired by hand.
function prepare(win) {
  win.ghostTimers = recordTimers(win, 300);
}

function typeInto(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.selectionStart = text.length;
  inp.selectionEnd = text.length;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  return inp;
}

function pickerVisible(win) {
  return win.document.getElementById('autocomplete').style.display === 'block';
}

function ghostText(win) {
  const el = win.document
    .getElementById('ghost-overlay')
    .querySelector('.ghost-text');
  return el ? el.textContent : '';
}

async function main() {
  const {win, posted} = makePage({beforeScripts: prepare});
  const tabId = win._testApi.getActiveTabId();
  assert.ok(tabId, 'the page has an active tab');

  // Typing an @-mention posts getFiles WITH the tab stamp.
  typeInto(win, 'look at @sr');
  const getFiles = await waitFor(
    () => posted.find(c => c.type === 'getFiles'),
    'typing @ must post getFiles',
  );
  assert.strictEqual(
    getFiles.tabId,
    tabId,
    'getFiles must carry the active tab id (the @-mention picker ' +
      'renders nothing for an untagged reply)',
  );
  assert.strictEqual(getFiles.prefix, 'sr');
  console.log('  ok - getFiles carries the tab stamp');

  // Only a reply echoing the stamp opens the picker.
  send(win, {
    type: 'files',
    files: [{type: 'file', text: './src/main.py'}],
    prefix: 'sr',
    tabId: 'some-other-tab',
  });
  assert.strictEqual(
    pickerVisible(win),
    false,
    "another tab's files reply must not open this tab's picker",
  );
  send(win, {
    type: 'files',
    files: [{type: 'file', text: './src/main.py'}],
    prefix: 'sr',
    tabId,
  });
  assert.strictEqual(
    pickerVisible(win),
    true,
    'the files reply stamped with this tab opens the picker',
  );
  console.log('  ok - the files reply is matched by the tab stamp');

  // The ghost-text request carries the stamp too (300 ms debounce).
  typeInto(win, 'write a haiku');
  fireTimers(win.ghostTimers);
  const complete = posted.find(c => c.type === 'complete');
  assert.ok(complete, 'typing must post complete');
  assert.strictEqual(
    complete.tabId,
    tabId,
    'complete must carry the active tab id (ghost text would render ' +
      'nothing for an untagged reply)',
  );
  console.log('  ok - complete carries the tab stamp');

  send(win, {
    type: 'ghost',
    suggestion: ' about rain',
    query: 'write a haiku',
    tabId: 'some-other-tab',
  });
  assert.strictEqual(
    ghostText(win),
    '',
    "another tab's ghost reply must not paint here",
  );
  send(win, {
    type: 'ghost',
    suggestion: ' about rain',
    query: 'write a haiku',
    tabId,
  });
  assert.strictEqual(
    ghostText(win),
    ' about rain',
    'the ghost reply stamped with this tab paints the suggestion',
  );
  console.log('  ok - the ghost reply is matched by the tab stamp');
  win.close();
}

main()
  .then(() => console.log('autocompleteTabIdWiring.test.js: all tests passed'))
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
