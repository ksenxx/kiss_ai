// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end coverage for the "Auto submit spoken task" setting.  The real
// chat.html, main.js and voice.js run inside JSDOM so the whole path from a
// dictated round's `voiceSpeech` transcript to either a posted `submit` or
// a caret insertion in the task textarea is exercised.

'use strict';

const assert = require('assert');
const {send, recordTimers, fireTimers, byId} = require('./simplify2_harness');
const {makeVoicePage, startPipeline, dictate} = require('./voice_page_helpers');
const {installAudio} = require('./speech_helpers');
const {makeRunner} = require('./ui_antipattern_harness');
const {test, report} = makeRunner();

const AUTO_SUBMIT_KEY = 'kissVoiceAutoSubmit';

// Every page built; closed after the run so node exits on its own.
const windows = [];

function makePage(options) {
  const opts = options || {};
  // The "working on it" ack is an audio clip the page plays itself.
  let players;
  const page = makeVoicePage({
    beforeScripts(win) {
      if (opts.stored !== undefined) {
        win.localStorage.setItem(AUTO_SUBMIT_KEY, opts.stored);
      }
      players = installAudio(win);
    },
  });
  windows.push(page.win);
  return Object.assign(page, {players});
}

// The same page with the voice pipeline live: voice.js types only the
// transcript of a round it posted, so the tests below dictate real ones.
async function makeLivePage(options) {
  const page = makePage(options);
  await startPipeline(page);
  return page;
}

const input = win => byId(win, 'task-input');

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

async function main() {
  await test('the settings panel exposes an "Auto submit spoken task" select', () => {
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

  await test('auto submit defaults to on and honours a stored choice', () => {
    assert.strictEqual(toggle(makePage().win).value, 'on');
    assert.strictEqual(toggle(makePage({stored: 'off'}).win).value, 'off');
    assert.strictEqual(toggle(makePage({stored: 'on'}).win).value, 'on');
    assert.strictEqual(
      toggle(makePage({stored: 'garbage'}).win).value,
      'on',
      'an unparsable stored value must fall back to on',
    );
  });

  await test('changing the select persists the choice', () => {
    const {win} = makePage();
    setToggle(win, 'off');
    assert.strictEqual(win.localStorage.getItem(AUTO_SUBMIT_KEY), 'off');
    setToggle(win, 'on');
    assert.strictEqual(win.localStorage.getItem(AUTO_SUBMIT_KEY), 'on');
  });

  await test('with auto submit off nothing is sent and the draft is kept', async () => {
    const page = await makeLivePage({stored: 'off'});
    const {win, posted} = page;
    posted.length = 0;
    dictate(page, 'Fix the parser bug', {speaker: 1});
    assert.strictEqual(
      posted.filter(m => m.type === 'submit' || m.type === 'appendUserMessage')
        .length,
      0,
      JSON.stringify(posted),
    );
    assert.strictEqual(input(win).value, 'Fix the parser bug');
  });

  await test('with auto submit off no "working on it" ack is played', async () => {
    const off = await makeLivePage({stored: 'off'});
    dictate(off, 'just a draft');
    assert.deepStrictEqual(
      off.players,
      [],
      'drafting a task must not claim work has started',
    );

    const auto = await makeLivePage();
    dictate(auto, 'go now');
    assert.deepStrictEqual(
      auto.players.map(p => [p.src, p.playCalls]),
      [['/media/working-on-it.mp3', 1]],
      'a submitted task must still be acknowledged',
    );
  });

  await test('caret is restored before listeners see the input event', async () => {
    const page = await makeLivePage({stored: 'off'});
    const {win, posted} = page;
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
    dictate(page, 'MID');
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
    dictate(page, 'END');
    fireTimers(ghostTimers);
    assert.strictEqual(
      posted.filter(m => m.type === 'complete').length,
      1,
      'an insertion at the end of the draft requests a completion',
    );
  });

  await test('with auto submit off the exact spoken text is inserted, with no speaker or language prefix', async () => {
    const page = await makeLivePage({stored: 'off'});
    const inp = input(page.win);
    inp.value = 'before after';
    caret(page.win, 6);
    dictate(page, 'Fix the bug', {speaker: 2, language: 'fr'});
    assert.strictEqual(inp.value, 'before Fix the bug after');
    assert.strictEqual(inp.selectionStart, 18);
    assert.strictEqual(inp.selectionEnd, 18);
  });

  await test('with auto submit off a selection is replaced by the speech', async () => {
    const page = await makeLivePage({stored: 'off'});
    const inp = input(page.win);
    inp.value = 'keep DROP keep';
    caret(page.win, 5, 9);
    dictate(page, 'NEW');
    assert.strictEqual(inp.value, 'keep NEW keep');
    assert.strictEqual(inp.selectionStart, 8);
  });

  await test('caret insertion pads only where whitespace is missing', async () => {
    const cases = [
      {value: '', pos: 0, text: 'hi', want: 'hi', caret: 2},
      {value: 'a', pos: 1, text: 'hi', want: 'a hi', caret: 4},
      {value: 'a ', pos: 2, text: 'hi', want: 'a hi', caret: 4},
      {value: 'ab', pos: 0, text: 'hi', want: 'hi ab', caret: 2},
      {value: ' ab', pos: 0, text: 'hi', want: 'hi ab', caret: 2},
    ];
    const page = await makeLivePage({stored: 'off'});
    const inp = input(page.win);
    cases.forEach(c => {
      inp.value = c.value;
      caret(page.win, c.pos);
      dictate(page, c.text);
      assert.strictEqual(
        inp.value,
        c.want,
        `inserting "${c.text}" into "${c.value}" at ${c.pos}`,
      );
      assert.strictEqual(inp.selectionStart, c.caret, `caret for "${c.want}"`);
    });
  });

  await test('toggling the select at runtime switches behaviour both ways', async () => {
    const page = await makeLivePage();
    const {win, posted} = page;
    setToggle(win, 'off');
    posted.length = 0;
    dictate(page, 'draft this');
    assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
    assert.strictEqual(input(win).value, 'draft this');

    setToggle(win, 'on');
    dictate(page, 'and go');
    const submits = posted.filter(m => m.type === 'submit');
    assert.strictEqual(submits.length, 1, JSON.stringify(posted));
    assert.strictEqual(submits[0].prompt, 'draft this and go');
    assert.strictEqual(input(win).value, '');
  });

  await test('with auto submit off a running agent is not steered', async () => {
    const page = await makeLivePage({stored: 'off'});
    const {win, posted} = page;
    send(win, {type: 'status', running: true});
    posted.length = 0;
    dictate(page, 'Also update the docs', {speaker: 2});
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
  await test('with auto submit off an empty transcript never edits the draft', async () => {
    const page = await makeLivePage({stored: 'off'});
    const {win, posted} = page;
    input(win).value = 'precious draft';
    caret(win, 0);
    posted.length = 0;
    dictate(page, '   ', {speaker: 1});
    assert.strictEqual(
      posted.filter(m => m.type === 'submit').length,
      0,
      JSON.stringify(posted),
    );
    assert.strictEqual(input(win).value, 'precious draft');
  });

  await test(
    'answering an ask-user question always submits, even with the ' +
      'toggle off',
    async () => {
      const page = await makeLivePage({stored: 'off'});
      const {win, posted} = page;
      send(win, {
        type: 'askUser',
        question: 'Which branch?',
        taskId: null,
      });
      posted.length = 0;
      dictate(page, 'the main branch');
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

  await test('a sibling tab picks up the setting through a storage event', async () => {
    const page = await makeLivePage();
    const {win, posted} = page;
    const sel = toggle(win);
    assert.strictEqual(sel.value, 'on');

    const evt = new win.Event('storage');
    evt.key = AUTO_SUBMIT_KEY;
    evt.newValue = 'off';
    win.dispatchEvent(evt);
    assert.strictEqual(sel.value, 'off', 'the select must follow the change');

    posted.length = 0;
    dictate(page, 'drafted elsewhere');
    assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
    assert.strictEqual(input(win).value, 'drafted elsewhere');

    const back = new win.Event('storage');
    back.key = AUTO_SUBMIT_KEY;
    back.newValue = 'on';
    win.dispatchEvent(back);
    assert.strictEqual(sel.value, 'on');
  });

  await test('an unrelated storage key never disturbs the setting', async () => {
    const page = await makeLivePage({stored: 'off'});
    const {win, posted} = page;
    const evt = new win.Event('storage');
    evt.key = 'kissVoiceSensitivity';
    evt.newValue = '40';
    win.dispatchEvent(evt);
    assert.strictEqual(toggle(win).value, 'off');
    posted.length = 0;
    dictate(page, 'still a draft');
    assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
  });

  windows.forEach(win => win.close());
  report('voiceAutoSubmitToggle');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
