// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end page tests for dictation, the composer's mic (#dictate-btn,
// media/voice.js): while it is on, what the user says is sent to the
// daemon for its English transcript, with no wake word, and the
// transcript is typed at the cursor of the text box.
//
// jsdom has no microphone and no audio graph, so voice_page_helpers
// supplies them: a microphone the test grants or refuses and processor
// nodes it feeds audio blocks to.  The daemon is the test: it reads what
// the page posts and sends the `voiceSpeech` reply.  Everything else is
// the real chat.html, main.js and voice.js.  dictation.coverage.js runs
// this suite and requires every line of the dictation code to be run.
//
// A real browser, with its own microphone and audio graph and a real
// daemon, runs the same code in src/kiss/tests/webapp/test_dictation.py.
// What a real microphone cannot be made to do on demand is tested here
// only: a prompt nobody answers, a device lost in mid-sentence, a
// transcript that never comes, audio that will not start.

const assert = require('assert');
const {send, twoTabs, clickTab} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {submits} = require('./attachment_helpers');
const {
  makeVoicePage,
  micBtn,
  settle,
  startMic,
  feed,
  lastProcessor,
  fireTimer,
} = require('./voice_page_helpers');

const windows = [];
const {test, report} = makeRunner({windows});

const SOURCE_URL = {'voice.js': 'dictation-voice.js'};
// One audio block of the fake microphone at its default 16 kHz.
const BLOCK_BYTES = 4096 * 2;

function makePage(opts) {
  const page = makeVoicePage(Object.assign({sourceUrl: SOURCE_URL}, opts));
  windows.push(page.win);
  return page;
}

function dictateBtn(win) {
  return win.document.getElementById('dictate-btn');
}

function taskInput(win) {
  return win.document.getElementById('task-input');
}

/** The utterances the page has sent for a transcript, in order. */
function sent(page) {
  return page.posted.filter(m => m.type === 'voiceTranscribe');
}

function lastMicCall(page) {
  return page.mic.calls[page.mic.calls.length - 1];
}

function classes(el) {
  return Array.from(el.classList).sort().join(' ');
}

function errorToast(win) {
  return win.document.querySelector(
    '.kiss-notification[data-notification-id="dictation-error"]',
  );
}

/** Turn the mic on with its button and grant it the microphone. */
async function micOn(page) {
  dictateBtn(page.win).click();
  await settle();
  lastMicCall(page).grant();
  await settle();
  assert.ok(
    dictateBtn(page.win).classList.contains('active'),
    `the mic must be listening: ${dictateBtn(page.win).className}`,
  );
}

/**
 * Speak for half a second and fall silent for two: the utterance ends
 * and its audio is sent.  Returns the tag sent with it, which the daemon
 * echoes on its transcript.
 */
function say(page) {
  const proc = lastProcessor(page);
  const before = sent(page).length;
  feed(proc, true, 2);
  feed(proc, false, 8);
  assert.strictEqual(sent(page).length, before + 1, 'the utterance is sent');
  return sent(page)[before].round;
}

/** The daemon's transcript of the utterance tagged `round`. */
function transcript(page, round, text) {
  send(page.win, {type: 'voiceSpeech', text, round});
}

async function main() {
  await test(
    'the mic needs no wake word: what is said goes to the daemon as plain ' +
      'audio and its English transcript is typed at the cursor, unsent',
    async () => {
      const page = makePage();
      const {win} = page;
      const btn = dictateBtn(win);
      const inp = taskInput(win);
      assert.strictEqual(btn.getAttribute('aria-pressed'), 'false');

      btn.click();
      assert.strictEqual(classes(btn), 'dictate-loading toggle-btn');
      assert.strictEqual(btn.getAttribute('aria-pressed'), 'true');
      assert.strictEqual(
        btn.getAttribute('data-tooltip'),
        'Dictation: starting ...',
      );
      await settle();
      assert.strictEqual(page.mic.calls.length, 1, 'the microphone is asked');
      lastMicCall(page).grant();
      await settle();
      assert.strictEqual(classes(btn), 'active toggle-btn');
      assert.ok(/^Dictation on:/.test(btn.getAttribute('data-tooltip')));
      assert.strictEqual(
        page.state.recognizers.length,
        0,
        'no wake-word recognizer is built for dictation',
      );

      // The user has typed, and put the cursor between two words.
      inp.value = 'Hello world';
      inp.setSelectionRange(5, 5);
      // Five quiet blocks, of which the last two (0.5 s) lead the
      // utterance in; two loud ones; two seconds of silence end it.
      const proc = lastProcessor(page);
      const output = feed(proc, false, 5);
      assert.ok(output[0] > 0, 'the audio sink is kept awake');
      assert.strictEqual(sent(page).length, 0, 'silence sends nothing');
      feed(proc, true, 2);
      assert.strictEqual(classes(btn), 'active dictate-hearing toggle-btn');
      feed(proc, false, 8);
      assert.strictEqual(
        classes(btn),
        'active dictate-transcribing toggle-btn',
      );

      assert.strictEqual(sent(page).length, 1);
      const request = sent(page)[0];
      assert.deepStrictEqual(
        Object.keys(request).sort(),
        ['audio', 'dictation', 'round', 'type'],
        'no wake-word field: the daemon transcribes and translates it all',
      );
      assert.strictEqual(
        request.dictation,
        true,
        'marked as dictation, so the daemon cuts no wake word from it',
      );
      assert.strictEqual(typeof request.round, 'number');
      assert.strictEqual(
        Buffer.from(request.audio, 'base64').length,
        (2 + 2 + 8) * BLOCK_BYTES,
        'the lead-in, the speech and the closing silence',
      );

      transcript(page, request.round, '  big  ');
      assert.strictEqual(inp.value, 'Hello big world');
      assert.strictEqual(inp.selectionStart, 9, 'the cursor follows the words');
      assert.strictEqual(inp.selectionEnd, 9);
      assert.strictEqual(classes(btn), 'active toggle-btn');
      assert.strictEqual(submits(page.posted).length, 0, 'nothing is sent');
    },
  );

  await test(
    'utterances are typed in the order they were spoken, over the ' +
      'selection; one with no words types nothing',
    async () => {
      const page = makePage();
      const {win} = page;
      const btn = dictateBtn(win);
      const inp = taskInput(win);
      await micOn(page);
      const first = say(page);
      const second = say(page);
      const third = say(page);
      assert.notStrictEqual(first, second);
      assert.strictEqual(classes(btn), 'active dictate-transcribing toggle-btn');

      transcript(page, first, 'Fix the bug now');
      assert.strictEqual(inp.value, 'Fix the bug now');
      transcript(page, second, '   ');
      assert.strictEqual(inp.value, 'Fix the bug now', 'no words, no typing');
      assert.ok(
        btn.classList.contains('dictate-transcribing'),
        'the third transcript is still awaited',
      );
      // The user selects "bug": the next words replace it.
      inp.setSelectionRange(8, 11);
      transcript(page, third, 'crash');
      assert.strictEqual(inp.value, 'Fix the crash now');
      assert.strictEqual(classes(btn), 'active toggle-btn');
      // A transcript that answers no utterance of this page is refused.
      transcript(page, third, 'again');
      transcript(page, undefined, 'untagged');
      assert.strictEqual(inp.value, 'Fix the crash now');
      assert.strictEqual(submits(page.posted).length, 0);
    },
  );

  await test(
    'a knock is not sent; someone who speaks on is cut at a breath after ' +
      '20 s and at 30 s at the latest',
    async () => {
      const page = makePage();
      const btn = dictateBtn(page.win);
      await micOn(page);
      const proc = lastProcessor(page);

      // Two 85 ms blocks of noise, then silence: too short to be speech.
      feed(proc, true, 2, 48000);
      assert.ok(btn.classList.contains('dictate-hearing'));
      feed(proc, false, 24, 48000);
      assert.strictEqual(sent(page).length, 0, 'a knock is not sent');
      assert.strictEqual(classes(btn), 'active toggle-btn');

      // A half-second pause does not end an utterance ...
      feed(proc, true, 40);
      feed(proc, false, 2);
      feed(proc, true, 38);
      assert.strictEqual(sent(page).length, 0, 'under 20 s: no cut');
      // ... until it is 20 s long: then a breath of 0.3 s does.
      feed(proc, false, 1);
      assert.strictEqual(sent(page).length, 0, '0.26 s is no breath');
      feed(proc, false, 1);
      assert.strictEqual(sent(page).length, 1, 'cut at the breath');
      assert.strictEqual(
        Buffer.from(sent(page)[0].audio, 'base64').length,
        (40 + 2 + 38 + 2) * BLOCK_BYTES,
        'the whole utterance, its pause and the breath included',
      );

      // Without a breath the utterance ends at 30 s (118 blocks), and
      // the next block begins another.
      feed(proc, true, 117);
      assert.strictEqual(sent(page).length, 1);
      feed(proc, true, 1);
      assert.strictEqual(sent(page).length, 2, 'cut at 30 s');
      feed(proc, true, 1);
      assert.ok(btn.classList.contains('dictate-hearing'), 'a new utterance');
      assert.notStrictEqual(sent(page)[0].round, sent(page)[1].round);
    },
  );

  await test(
    'turning the mic off sends what was being said and frees the ' +
      'microphone; the transcript is still typed',
    async () => {
      const page = makePage();
      const {win, log} = page;
      const btn = dictateBtn(win);
      await micOn(page);
      feed(lastProcessor(page), true, 2);
      assert.strictEqual(sent(page).length, 0, 'still speaking');

      btn.click();
      assert.strictEqual(sent(page).length, 1, 'what was said is sent');
      assert.deepStrictEqual(log, ['ctx-open', 'track-stopped', 'ctx-closed']);
      assert.strictEqual(lastProcessor(page).onaudioprocess, null);
      assert.strictEqual(classes(btn), 'dictate-transcribing toggle-btn');
      assert.strictEqual(btn.getAttribute('aria-pressed'), 'false');
      assert.ok(/^Dictate:/.test(btn.getAttribute('data-tooltip')));

      transcript(page, sent(page)[0].round, 'last words');
      assert.strictEqual(taskInput(win).value, 'last words');
      assert.strictEqual(classes(btn), 'toggle-btn');

      // Off while silent: nothing more to send.
      await micOn(page);
      btn.click();
      assert.strictEqual(sent(page).length, 1);
      assert.strictEqual(classes(btn), 'toggle-btn');
    },
  );

  await test(
    'a mic turned off before it listens holds nothing: neither a ' +
      'microphone granted late nor an audio context that starts late',
    async () => {
      // Off while the browser still asks for the microphone.
      const asking = makePage();
      const askBtn = dictateBtn(asking.win);
      askBtn.click();
      await settle();
      askBtn.click();
      assert.strictEqual(classes(askBtn), 'toggle-btn');
      lastMicCall(asking).grant();
      await settle();
      assert.deepStrictEqual(asking.log, ['track-stopped'], 'released at once');
      assert.strictEqual(asking.state.processors.length, 0);

      // Off while the browser keeps the audio suspended (a browser that
      // only has the prefixed audio context, as old Safari).
      const suspended = makePage({
        beforeScripts(win) {
          win.webkitAudioContext = class extends win.AudioContext {
            constructor() {
              super();
              this.state = 'suspended';
            }
            resume() {
              return new Promise(resolve => {
                win.resumeAudio = resolve;
              });
            }
          };
          delete win.AudioContext;
        },
      });
      const btn = dictateBtn(suspended.win);
      btn.click();
      await settle();
      lastMicCall(suspended).grant();
      await settle();
      assert.strictEqual(classes(btn), 'dictate-loading toggle-btn');
      btn.click();
      assert.deepStrictEqual(suspended.log, [
        'ctx-open',
        'track-stopped',
        'ctx-closed',
      ]);
      suspended.win.resumeAudio();
      await settle();
      assert.strictEqual(suspended.state.processors.length, 0);
      assert.strictEqual(classes(btn), 'toggle-btn');
      assert.strictEqual(errorToast(suspended.win), null);
    },
  );

  await test(
    'a refused microphone turns the mic off and says why and what to do; ' +
      'the next start clears it; a browser with no microphone says so too',
    async () => {
      const page = makePage();
      const {win} = page;
      const btn = dictateBtn(win);
      const remedy =
        'Microphone access was denied. Allow the microphone for this site ' +
        'in the browser settings, then turn dictation on again.';
      btn.click();
      await settle();
      lastMicCall(page).refuse('Permission denied');
      await settle();
      assert.strictEqual(classes(btn), 'dictate-error toggle-btn');
      assert.strictEqual(btn.getAttribute('aria-pressed'), 'false');
      assert.strictEqual(
        btn.getAttribute('data-tooltip'),
        'Dictation error: ' + remedy,
      );
      assert.strictEqual(
        errorToast(win).querySelector('.kiss-notification-message').textContent,
        remedy,
      );
      assert.deepStrictEqual(page.log, [], 'nothing is held');

      // The next start takes the error off the button, and the toast
      // down once it listens.
      btn.click();
      assert.strictEqual(classes(btn), 'dictate-loading toggle-btn');
      assert.ok(errorToast(win), 'still shown while the mic starts');
      await settle();
      lastMicCall(page).grant();
      await settle();
      assert.strictEqual(classes(btn), 'active toggle-btn');
      assert.strictEqual(errorToast(win), null);

      const noMic = makePage({
        beforeScripts(w) {
          Object.defineProperty(w.navigator, 'mediaDevices', {
            configurable: true,
            value: undefined,
          });
        },
      });
      const noMicBtn = dictateBtn(noMic.win);
      noMicBtn.click();
      await settle();
      assert.strictEqual(classes(noMicBtn), 'dictate-error toggle-btn');
      assert.ok(
        /^Dictation error: Dictation could not start \(.*\)\. Check the microphone, then turn dictation on again\.$/.test(
          noMicBtn.getAttribute('data-tooltip'),
        ),
        noMicBtn.getAttribute('data-tooltip'),
      );
    },
  );

  await test(
    'a microphone that goes away mid-sentence: what was said is sent, ' +
      'the mic turns off and says why; one that ends later is no news',
    async () => {
      const page = makePage();
      const {win} = page;
      const btn = dictateBtn(win);
      await micOn(page);
      feed(lastProcessor(page), true, 2);
      page.state.tracks[0].end();
      assert.strictEqual(sent(page).length, 1, 'what was said is sent');
      assert.strictEqual(
        classes(btn),
        'dictate-error dictate-transcribing toggle-btn',
      );
      assert.ok(
        btn
          .getAttribute('data-tooltip')
          .includes(
            'the microphone was disconnected or taken by another application',
          ),
      );
      assert.ok(errorToast(win), 'the reason is shown');
      assert.deepStrictEqual(page.log, ['ctx-open', 'track-stopped', 'ctx-closed']);

      // The track of a mic the user turned off ends: nothing to report.
      const off = makePage();
      await micOn(off);
      dictateBtn(off.win).click();
      off.state.tracks[0].end();
      assert.strictEqual(classes(dictateBtn(off.win)), 'toggle-btn');
      assert.strictEqual(errorToast(off.win), null);
    },
  );

  await test(
    'the mic takes the microphone from a voice trigger that is on, and ' +
      'gives it back; what was dictated is typed, never sent',
    async () => {
      const page = makePage({enabled: true});
      const {win} = page;
      await startMic(page);

      dictateBtn(win).click();
      assert.ok(micBtn(win).classList.contains('voice-off'), 'trigger paused');
      assert.deepStrictEqual(
        page.log,
        ['ctx-open', 'track-stopped', 'ctx-closed'],
        'the trigger lets the microphone go',
      );
      assert.strictEqual(
        win.localStorage.getItem('kissVoiceEnabled'),
        '1',
        'the trigger is still on for the user',
      );
      await settle();
      assert.strictEqual(page.mic.calls.length, 2);
      lastMicCall(page).grant();
      await settle();
      assert.ok(dictateBtn(win).classList.contains('active'));
      const round = say(page);
      assert.deepStrictEqual(Object.keys(sent(page)[0]).sort(), [
        'audio',
        'dictation',
        'round',
        'type',
      ]);

      dictateBtn(win).click();
      assert.ok(micBtn(win).classList.contains('voice-loading'), 'trigger back');
      await startMic(page);
      assert.strictEqual(page.mic.calls.length, 3);
      // The voice trigger sends what it hears by itself; dictation does not.
      transcript(page, round, 'dictated words');
      assert.strictEqual(taskInput(win).value, 'dictated words');
      assert.strictEqual(submits(page.posted).length, 0);
    },
  );

  await test(
    'turning the voice trigger on ends dictation: the trigger takes the ' +
      'microphone and is the one that stays on',
    async () => {
      const page = makePage();
      const {win} = page;
      await micOn(page);
      micBtn(win).click();
      assert.strictEqual(classes(dictateBtn(win)), 'toggle-btn');
      assert.strictEqual(dictateBtn(win).getAttribute('aria-pressed'), 'false');
      assert.strictEqual(win.localStorage.getItem('kissVoiceEnabled'), '1');
      await startMic(page);
      // The trigger's button is its own toggle again.
      micBtn(win).click();
      assert.ok(micBtn(win).classList.contains('voice-off'));
      assert.strictEqual(win.localStorage.getItem('kissVoiceEnabled'), '0');
      assert.strictEqual(classes(dictateBtn(win)), 'toggle-btn');
    },
  );

  await test(
    'a transcript that takes over a minute is given up, and is not typed ' +
      'when it comes after all',
    async () => {
      const page = makePage();
      const {win} = page;
      const btn = dictateBtn(win);
      await micOn(page);
      const round = say(page);
      assert.ok(btn.classList.contains('dictate-transcribing'));
      fireTimer(page, 60000);
      assert.strictEqual(classes(btn), 'active toggle-btn');
      transcript(page, round, 'too late');
      assert.strictEqual(taskInput(win).value, '');
    },
  );

  await test('words spoken to one chat are not typed into another', async () => {
    const page = makePage();
    const {win} = page;
    const {first, second} = twoTabs(win);
    clickTab(win, first);
    await micOn(page);
    const round = say(page);
    // The user moves to the other chat before the transcript comes.
    clickTab(win, second);
    transcript(page, round, 'for the first chat');
    assert.strictEqual(taskInput(win).value, '', 'not into this chat');
    clickTab(win, first);
    assert.strictEqual(taskInput(win).value, '', 'the words are dropped');
    // What is said to the chat on screen is typed there.
    transcript(page, say(page), 'typed here');
    assert.strictEqual(taskInput(win).value, 'typed here');
    assert.strictEqual(submits(page.posted).length, 0);
  });

  report('dictation');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
