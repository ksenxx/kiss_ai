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
const {
  makePage,
  send,
  waitFor,
  recordTimers,
  fireTimers,
  ghostText,
} = require('./simplify2_harness.js');

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

// A list of files answers the mention it was asked for.  Once the text
// typed, or the folder or the machine the tab browses, is another, the
// list still on screen (the answer to the new mention is on its way)
// completes nothing: Tab, Enter or a click would put a file of the old
// list in place of the new mention.  Nor once the caret moved inside
// the word typed: the mention is the word up to the caret.
function testAListForAnotherMentionIsNotAccepted() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  const inp = win.document.getElementById('task-input');
  const config = extra =>
    send(win, {
      type: 'configData',
      config: Object.assign({work_dir: '/work/a'}, extra),
      apiKeys: {},
    });
  const files = (prefix, text, extra) =>
    send(
      win,
      Object.assign(
        {type: 'files', tabId, prefix, workDir: '/work/a', files: [{type: 'file', text}]},
        extra,
      ),
    );
  const accept = how => {
    if (how === 'click') {
      win.document.querySelector('#autocomplete .ac-item').click();
      return;
    }
    inp.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: how, bubbles: true, cancelable: true}),
    );
  };
  // Type *text*, the caret where its "|" is.
  const typeAt = text => {
    const caret = text.indexOf('|');
    inp.value = text.replace('|', '');
    inp.setSelectionRange(caret, caret);
    inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  };
  const drafts = {'the caret': 'Review @alpha|zebra'};
  const changes = {
    'the text': () => typeInto(win, 'Review @zebra'),
    'the folder': () => config({work_dir: '/work/b'}),
    'the machine': () => config({last_machine: 'gpu-box'}),
    'the caret': () => {
      inp.setSelectionRange(inp.value.length, inp.value.length);
      win.document.dispatchEvent(new win.Event('selectionchange'));
    },
  };
  // What answers the mention after each change.
  const answers = {
    'the text': () => files('zebra', './zebra.txt'),
    'the folder': () => files('alpha', './alpha-b.txt', {workDir: '/work/b'}),
    'the machine': () =>
      files('alpha', './alpha-gpu.txt', {machine: 'gpu-box', workDir: ''}),
    // No keystroke asked for the word the caret ends now: the next does.
    'the caret': () => {
      typeInto(win, inp.value);
      files('alphazebra', './alphazebra.txt');
    },
  };

  for (const what of Object.keys(changes)) {
    for (const how of ['Tab', 'Enter', 'click']) {
      config({});
      typeAt(drafts[what] || 'Review @alpha|');
      files('alpha', './alpha-private.txt');
      assert.ok(pickerVisible(win), 'the list is on screen');
      changes[what]();
      const typed = inp.value;
      posted.length = 0;
      accept(how);
      const label = what + ', ' + how;
      assert.strictEqual(
        inp.value,
        typed,
        label + ': BUG: a file of the old list replaced the mention',
      );
      assert.deepStrictEqual(
        posted.filter(c => c.type === 'recordFileUsage' || c.type === 'submit'),
        [],
        label,
      );
      // The answer arrives: its list completes the mention.
      answers[what]();
      assert.ok(pickerVisible(win), label + ': the new list is on screen');
      accept(how);
      assert.strictEqual(
        inp.value,
        'Review ' + win.document.querySelector('#autocomplete .ac-item').dataset.text + ' ',
        label,
      );
    }
  }
  console.log('  ok - a list for another mention is not accepted');
  win.close();
}

main()
  .then(testAListForAnotherMentionIsNotAccepted)
  .then(() => console.log('autocompleteTabIdWiring.test.js: all tests passed'))
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
