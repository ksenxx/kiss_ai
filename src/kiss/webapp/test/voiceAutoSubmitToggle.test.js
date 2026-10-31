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
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  recordTimers,
  fireTimers,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {test, report} = makeRunner();

const AUTO_SUBMIT_KEY = 'kissVoiceAutoSubmit';

// Every page built; closed after the run so node exits on its own.
const windows = [];

function makePage(options) {
  const opts = options || {};
  // The "working on it" ack is an audio clip the page plays itself.
  const played = [];
  function beforeScripts(win) {
    stubRaf(win);
    if (opts.stored !== undefined) {
      win.localStorage.setItem(AUTO_SUBMIT_KEY, opts.stored);
    }
    win.Audio = function (src) {
      return {src, play: () => (played.push(src), Promise.resolve())};
    };
    win.__VOICE__ = {ackAudioUrl: '/media/working-on-it.mp3'};
  }
  const {win, posted} = makeChatPage({
    beforeScripts,
    extraScripts: ['voice.js'],
  });
  windows.push(win);
  return {win, posted, played};
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

test('the settings panel exposes an "Auto submit spoken task" select', () => {
  const {win} = makePage();
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
  assert.strictEqual(toggle(makePage().win).value, 'on');
  assert.strictEqual(toggle(makePage({stored: 'off'}).win).value, 'off');
  assert.strictEqual(toggle(makePage({stored: 'on'}).win).value, 'on');
  assert.strictEqual(
    toggle(makePage({stored: 'garbage'}).win).value,
    'on',
    'an unparsable stored value must fall back to on',
  );
});

test('changing the select persists the choice', () => {
  const {win} = makePage();
  setToggle(win, 'off');
  assert.strictEqual(win.localStorage.getItem(AUTO_SUBMIT_KEY), 'off');
  setToggle(win, 'on');
  assert.strictEqual(win.localStorage.getItem(AUTO_SUBMIT_KEY), 'on');
});

test('with auto submit off nothing is sent and the draft is kept', () => {
  const {win, posted} = makePage({stored: 'off'});
  posted.length = 0;
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
  const {win, played} = makePage({stored: 'off'});
  send(win, {type: 'voiceSpeech', text: 'just a draft'});
  assert.deepStrictEqual(
    played,
    [],
    'drafting a task must not claim work has started',
  );

  const auto = makePage();
  send(auto.win, {type: 'voiceSpeech', text: 'go now'});
  assert.deepStrictEqual(
    auto.played,
    ['/media/working-on-it.mp3'],
    'a submitted task must still be acknowledged',
  );
});

test('caret is restored before listeners see the input event', () => {
  const {win, posted} = makePage({stored: 'off'});
  // main.js asks for a ghost completion 300 ms after the input event.
  const ghostTimers = recordTimers(win, 300);
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
  fireTimers(ghostTimers);
  assert.strictEqual(
    posted.filter(m => m.type === 'complete').length,
    0,
    'a mid-draft insertion must not request a completion',
  );
  // Spoken at the end of the draft, the same insertion does request one.
  caret(win, inp.value.length);
  send(win, {type: 'voiceSpeech', text: 'END'});
  fireTimers(ghostTimers);
  assert.strictEqual(
    posted.filter(m => m.type === 'complete').length,
    1,
    'an insertion at the end of the draft requests a completion',
  );
});

test('with auto submit off the exact spoken text is inserted, with no speaker or language prefix', () => {
  const {win} = makePage({stored: 'off'});
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

test('with auto submit off a selection is replaced by the speech', () => {
  const {win} = makePage({stored: 'off'});
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
  const {win} = makePage({stored: 'off'});
  const inp = input(win);
  cases.forEach(c => {
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
  const {win, posted} = makePage();
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
  const {win, posted} = makePage({stored: 'off'});
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

// With auto submit on, an empty transcript leaving the draft alone is
// voiceSubmitToAgent's 'empty translation submits nothing and keeps a
// draft intact'.
test('with auto submit off an empty transcript never edits the draft', () => {
  const {win, posted} = makePage({stored: 'off'});
  input(win).value = 'precious draft';
  caret(win, 0);
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: '   ', speaker: 1});
  assert.strictEqual(
    posted.filter(m => m.type === 'submit').length,
    0,
    JSON.stringify(posted),
  );
  assert.strictEqual(input(win).value, 'precious draft');
});

test(
  'answering an ask-user question always submits, even with the ' +
    'toggle off',
  () => {
    const {win, posted} = makePage({stored: 'off'});
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
  },
);

test('a sibling tab picks up the setting through a storage event', () => {
  const {win, posted} = makePage();
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
  const {win, posted} = makePage({stored: 'off'});
  const evt = new win.Event('storage');
  evt.key = 'kissVoiceSensitivity';
  evt.newValue = '40';
  win.dispatchEvent(evt);
  assert.strictEqual(toggle(win).value, 'off');
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'still a draft'});
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
});

windows.forEach(win => win.close());
report('voiceAutoSubmitToggle');
