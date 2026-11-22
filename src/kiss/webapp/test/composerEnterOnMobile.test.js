// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// The composer's Enter key on a phone or tablet (main.js enterSends):
//
// * The on-screen keyboard shows its Enter key (enterkeyhint="enter"),
//   and Enter adds a line to the task: neither the keydown nor the
//   insertLineBreak it produces is cancelled, and nothing is sent.
// * The Send button sends the task, line breaks included, and the
//   answer box of a question says so ("tap Send", not "press Enter").
// * A desktop browser keeps Enter as the send key and Shift+Enter as
//   the line break, and its composer is not marked enterkeyhint="enter".

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {submits} = require('./attachment_helpers');

const PHONE_UA =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1';
const DESKTOP_UA =
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36';

/**
 * A connected chat page with one tab, under the given user agent.
 *
 * @param {string} ua The browser's user agent string.
 * @returns {{win: object, posted: Array<object>, inp: object}} The
 *   window, the commands the page posted, and the composer.
 */
function chatPage(ua) {
  const {win, posted} = makePage({
    userAgent: ua,
    beforeScripts(w) {
      Object.defineProperty(w.navigator, 'userAgentData', {
        value: undefined,
        configurable: true,
      });
    },
  });
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 't1', chatId: 'chat-1', title: 't1', workDir: '/ws'}],
  });
  return {win, posted, inp: win.document.getElementById('task-input')};
}

/** Press Enter in *inp*; returns the keydown event. */
function pressEnter(win, inp, shiftKey) {
  const ev = new win.KeyboardEvent('keydown', {
    key: 'Enter',
    shiftKey: shiftKey,
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  return ev;
}

/** Deliver the line break an Enter key produces; returns the event. */
function lineBreak(win, inp) {
  const ev = new win.InputEvent('beforeinput', {
    inputType: 'insertLineBreak',
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  return ev;
}

function phoneEnterAddsALine() {
  const {win, posted, inp} = chatPage(PHONE_UA);
  assert.ok(win.document.body.classList.contains('remote-mobile'));
  assert.strictEqual(inp.getAttribute('enterkeyhint'), 'enter');
  inp.value = 'line one';
  assert.ok(!pressEnter(win, inp, false).defaultPrevented, 'Enter is typed');
  assert.ok(!lineBreak(win, inp).defaultPrevented, 'the line break is kept');
  assert.strictEqual(submits(posted).length, 0, 'Enter sends nothing');
  inp.value = 'line one\nline two';
  win.document.getElementById('send-btn').click();
  assert.strictEqual(submits(posted).length, 1, 'the Send button sends');
  assert.strictEqual(submits(posted)[0].prompt, 'line one\nline two');
  send(win, {type: 'askUser', question: 'Which branch?', tabId: 't1'});
  assert.strictEqual(inp.placeholder, 'Type your answer and tap Send');
  win.close();
  console.log('  ok - phone: Enter adds a line, the Send button sends');
}

function desktopEnterSends() {
  const {win, posted, inp} = chatPage(DESKTOP_UA);
  assert.ok(!win.document.body.classList.contains('remote-mobile'));
  assert.notStrictEqual(inp.getAttribute('enterkeyhint'), 'enter');
  inp.value = 'first';
  assert.ok(!pressEnter(win, inp, true).defaultPrevented, 'Shift+Enter types');
  assert.strictEqual(submits(posted).length, 0, 'Shift+Enter sends nothing');
  assert.ok(pressEnter(win, inp, false).defaultPrevented, 'Enter sends');
  assert.strictEqual(submits(posted).length, 1);
  assert.strictEqual(submits(posted)[0].prompt, 'first');
  send(win, {type: 'askUser', question: 'Which branch?', tabId: 't1'});
  assert.strictEqual(inp.placeholder, 'Type your answer and press Enter');
  win.close();
  console.log('  ok - desktop: Enter sends, Shift+Enter adds a line');
}

function desktopLineBreakSends() {
  const {win, posted, inp} = chatPage(DESKTOP_UA);
  inp.value = 'second';
  assert.ok(lineBreak(win, inp).defaultPrevented, 'the line break sends');
  assert.strictEqual(submits(posted).length, 1);
  assert.strictEqual(submits(posted)[0].prompt, 'second');
  win.close();
  console.log('  ok - desktop: an Enter line break sends');
}

phoneEnterAddsALine();
desktopEnterSends();
desktopLineBreakSends();
