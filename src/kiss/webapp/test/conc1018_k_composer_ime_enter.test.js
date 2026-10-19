// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console */

// The Enter that commits an IME composition (CJK keyboards) reaches the
// composer as keydown key='Enter' with isComposing=true, or keyCode=229 on
// engines that report the real key.  The composer must leave it to the
// IME: swallowing it cancels the commit and posts the uncommitted text.
// A plain Enter afterwards posts exactly one submit.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function submits(posted) {
  return posted.filter(m => m.type === 'submit');
}

function enter(win, inp, init) {
  const ev = new win.KeyboardEvent('keydown', {
    key: 'Enter',
    bubbles: true,
    cancelable: true,
    ...init,
  });
  inp.dispatchEvent(ev);
  return ev;
}

function main() {
  const {win, posted} = makePage();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 't1', chatId: 'chat-1', title: 't1', workDir: '/ws'}],
  });
  const inp = win.document.getElementById('task-input');
  inp.value = '変換中';
  for (const init of [{isComposing: true}, {keyCode: 229}]) {
    const ev = enter(win, inp, init);
    assert.ok(
      !ev.defaultPrevented,
      'composition Enter is left to the IME: ' + JSON.stringify(init),
    );
    assert.strictEqual(
      submits(posted).length,
      0,
      'composition Enter posts nothing: ' + JSON.stringify(init),
    );
    assert.strictEqual(inp.value, '変換中', 'the composer keeps its text');
  }
  const plain = enter(win, inp, {});
  assert.ok(plain.defaultPrevented, 'a plain Enter is the send key');
  assert.strictEqual(submits(posted).length, 1, 'a plain Enter posts once');
  assert.strictEqual(submits(posted)[0].prompt, '変換中');
  win.close();
  console.log('  ok - composer Enter during IME composition is not a send');
}

main();
