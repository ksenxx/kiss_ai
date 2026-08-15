// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end coverage for the "Auto submit spoken task" setting.  The real
// chat.html, main.js and voice.js run inside JSDOM so the whole path from an
// incoming `voiceSpeech` message to either a posted `submit` or a caret
// insertion in the task textarea is exercised.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const AUTO_SUBMIT_KEY = 'kissVoiceAutoSubmit';

function makeWebview(options) {
  const opts = options || {};
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

  if (opts.stored !== undefined) {
    win.localStorage.setItem(AUTO_SUBMIT_KEY, opts.stored);
  }

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

  win.__VOICE__ = {mode: 'webview'};
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function input(win) {
  return win.document.getElementById('task-input');
}

function toggle(win) {
  return win.document.getElementById('cfg-voice-auto-submit');
}

function setToggle(win, value) {
  const sel = toggle(win);
  sel.value = value;
  sel.dispatchEvent(new win.Event('change', {bubbles: true}));
}

function caret(win, start, end) {
  const inp = input(win);
  inp.setSelectionRange(start, end === undefined ? start : end);
}

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
}

test('the settings panel exposes an "Auto submit spoken task" select', () => {
  const {win} = makeWebview();
  const sel = toggle(win);
  assert.ok(sel, 'select widget is missing from chat.html');
  assert.strictEqual(sel.tagName, 'SELECT');
  const values = Array.from(sel.options).map(o => o.value);
  assert.deepStrictEqual(values, ['on', 'off']);
  const label = sel.closest('label');
  assert.ok(label, 'the select must live inside a config label');
  assert.ok(
    /Auto submit spoken task/.test(label.textContent),
    'label text must read "Auto submit spoken task"',
  );
});

test('auto submit defaults to on and honours a stored choice', () => {
  assert.strictEqual(toggle(makeWebview().win).value, 'on');
  assert.strictEqual(toggle(makeWebview({stored: 'off'}).win).value, 'off');
  assert.strictEqual(toggle(makeWebview({stored: 'on'}).win).value, 'on');
  assert.strictEqual(
    toggle(makeWebview({stored: 'garbage'}).win).value,
    'on',
    'an unparsable stored value must fall back to on',
  );
});

test('changing the select persists the choice', () => {
  const {win} = makeWebview();
  setToggle(win, 'off');
  assert.strictEqual(win.localStorage.getItem(AUTO_SUBMIT_KEY), 'off');
  setToggle(win, 'on');
  assert.strictEqual(win.localStorage.getItem(AUTO_SUBMIT_KEY), 'on');
});

test('with auto submit on a spoken task is still sent immediately', () => {
  const {win, posted} = makeWebview();
  posted.length = 0;
  send(win, {type: 'voiceWake'});
  send(win, {type: 'voiceSpeech', text: 'Fix the parser bug', speaker: 1});
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(submits.length, 1, JSON.stringify(posted));
  assert.strictEqual(
    submits[0].prompt,
    'Speaker #1 says that: Fix the parser bug',
  );
  assert.strictEqual(input(win).value, '');
});

test('with auto submit off nothing is sent and the draft is kept', () => {
  const {win, posted} = makeWebview({stored: 'off'});
  posted.length = 0;
  send(win, {type: 'voiceWake'});
  send(win, {type: 'voiceSpeech', text: 'Fix the parser bug', speaker: 1});
  assert.strictEqual(
    posted.filter(m => m.type === 'submit' || m.type === 'appendUserMessage')
      .length,
    0,
    JSON.stringify(posted),
  );
  assert.strictEqual(input(win).value, 'Fix the parser bug');
});

test('with auto submit off no "working on it" ack is played', () => {
  const {win, posted} = makeWebview({stored: 'off'});
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'just a draft'});
  assert.strictEqual(
    posted.filter(m => m.type === 'voiceAck').length,
    0,
    'drafting a task must not claim work has started: ' +
      JSON.stringify(posted),
  );

  const auto = makeWebview();
  auto.posted.length = 0;
  send(auto.win, {type: 'voiceSpeech', text: 'go now'});
  assert.strictEqual(
    auto.posted.filter(m => m.type === 'voiceAck').length,
    1,
    'a submitted task must still be acknowledged',
  );
});

test('caret is restored before listeners see the input event', () => {
  const {win, posted} = makeWebview({stored: 'off'});
  const inp = input(win);
  inp.value = 'head tail';
  caret(win, 4);
  const seen = [];
  inp.addEventListener('input', () => {
    seen.push({start: inp.selectionStart, value: inp.value});
  });
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'MID'});
  assert.strictEqual(inp.value, 'head MID tail');
  assert.deepStrictEqual(
    seen,
    [{start: 8, value: 'head MID tail'}],
    'the input handler must observe the post-insert caret, not the end',
  );

  // main.js only asks for a ghost completion when the caret sits at the end
  // of the draft, so a mid-draft insertion must not schedule one.
  const clock = win.setTimeout;
  let fired = false;
  win.setTimeout = (fn, ms) => clock(() => {
    fired = true;
    fn();
  }, ms);
  assert.strictEqual(
    posted.filter(m => m.type === 'complete').length,
    0,
    'a mid-draft insertion must not request a completion',
  );
  win.setTimeout = clock;
  assert.strictEqual(fired, false);
});

test('with auto submit off the exact spoken text is inserted, with no speaker or language prefix', () => {
  const {win} = makeWebview({stored: 'off'});
  const inp = input(win);
  inp.value = 'before after';
  caret(win, 6);
  send(win, {
    type: 'voiceSpeech',
    text: 'Fix the bug',
    speaker: 2,
    language: 'fr',
  });
  assert.strictEqual(inp.value, 'before Fix the bug after');
  assert.strictEqual(inp.selectionStart, 18);
  assert.strictEqual(inp.selectionEnd, 18);
});

test('with auto submit off the text lands at the caret', () => {
  const {win} = makeWebview({stored: 'off'});
  const inp = input(win);
  inp.value = 'before after';
  caret(win, 6);
  send(win, {type: 'voiceSpeech', text: 'MIDDLE'});
  assert.strictEqual(inp.value, 'before MIDDLE after');
  assert.strictEqual(inp.selectionStart, 13);
  assert.strictEqual(inp.selectionEnd, 13);
});

test('with auto submit off a selection is replaced by the speech', () => {
  const {win} = makeWebview({stored: 'off'});
  const inp = input(win);
  inp.value = 'keep DROP keep';
  caret(win, 5, 9);
  send(win, {type: 'voiceSpeech', text: 'NEW'});
  assert.strictEqual(inp.value, 'keep NEW keep');
  assert.strictEqual(inp.selectionStart, 8);
});

test('caret insertion pads only where whitespace is missing', () => {
  const cases = [
    {value: '', pos: 0, text: 'hi', want: 'hi', caret: 2},
    {value: 'a', pos: 1, text: 'hi', want: 'a hi', caret: 4},
    {value: 'a ', pos: 2, text: 'hi', want: 'a hi', caret: 4},
    {value: 'ab', pos: 0, text: 'hi', want: 'hi ab', caret: 2},
    {value: ' ab', pos: 0, text: 'hi', want: 'hi ab', caret: 2},
  ];
  cases.forEach(c => {
    const {win} = makeWebview({stored: 'off'});
    const inp = input(win);
    inp.value = c.value;
    caret(win, c.pos);
    send(win, {type: 'voiceSpeech', text: c.text});
    assert.strictEqual(
      inp.value,
      c.want,
      `inserting "${c.text}" into "${c.value}" at ${c.pos}`,
    );
    assert.strictEqual(inp.selectionStart, c.caret, `caret for "${c.want}"`);
  });
});

test('toggling the select at runtime switches behaviour both ways', () => {
  const {win, posted} = makeWebview();
  setToggle(win, 'off');
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'draft this'});
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
  assert.strictEqual(input(win).value, 'draft this');

  setToggle(win, 'on');
  send(win, {type: 'voiceSpeech', text: 'and go'});
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(submits.length, 1, JSON.stringify(posted));
  assert.strictEqual(submits[0].prompt, 'draft this and go');
  assert.strictEqual(input(win).value, '');
});

test('with auto submit off a running agent is not steered', () => {
  const {win, posted} = makeWebview({stored: 'off'});
  send(win, {type: 'status', running: true});
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'Also update the docs', speaker: 2});
  assert.strictEqual(
    posted.filter(m => m.type === 'appendUserMessage').length,
    0,
    JSON.stringify(posted),
  );
  assert.strictEqual(input(win).value, 'Also update the docs');
});

test('an empty transcript never edits the draft in either mode', () => {
  ['on', 'off'].forEach(mode => {
    const {win, posted} = makeWebview({stored: mode});
    input(win).value = 'precious draft';
    caret(win, 0);
    posted.length = 0;
    send(win, {type: 'voiceSpeech', text: '   ', speaker: 1});
    assert.strictEqual(
      posted.filter(m => m.type === 'submit').length,
      0,
      `mode ${mode}: ${JSON.stringify(posted)}`,
    );
    assert.strictEqual(input(win).value, 'precious draft', `mode ${mode}`);
  });
});

test('answering an ask-user question always submits, even with the ' +
  'toggle off', () => {
  const {win, posted} = makeWebview({stored: 'off'});
  send(win, {
    type: 'askUser',
    question: 'Which branch?',
    taskId: null,
  });
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'the main branch'});
  const answers = posted.filter(m => m.type === 'userAnswer');
  assert.strictEqual(answers.length, 1, JSON.stringify(posted));
  assert.strictEqual(answers[0].answer, 'the main branch');
  assert.strictEqual(
    input(win).value,
    '',
    'the answer must not leak into the task textarea',
  );
});

test('a sibling tab picks up the setting through a storage event', () => {
  const {win, posted} = makeWebview();
  const sel = toggle(win);
  assert.strictEqual(sel.value, 'on');

  const evt = new win.Event('storage');
  evt.key = AUTO_SUBMIT_KEY;
  evt.newValue = 'off';
  win.dispatchEvent(evt);
  assert.strictEqual(sel.value, 'off', 'the select must follow the change');

  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'drafted elsewhere'});
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
  assert.strictEqual(input(win).value, 'drafted elsewhere');

  const back = new win.Event('storage');
  back.key = AUTO_SUBMIT_KEY;
  back.newValue = 'on';
  win.dispatchEvent(back);
  assert.strictEqual(sel.value, 'on');
});

test('an unrelated storage key never disturbs the setting', () => {
  const {win, posted} = makeWebview({stored: 'off'});
  const evt = new win.Event('storage');
  evt.key = 'kissVoiceSensitivity';
  evt.newValue = '40';
  win.dispatchEvent(evt);
  assert.strictEqual(toggle(win).value, 'off');
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'still a draft'});
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length > 0 ? 1 : 0);
