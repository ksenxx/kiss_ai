// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for natural, emotive delivery in the agent ``talk``
// tool playback: the webview must strip markdown/emoji artifacts the
// engine would read aloud, split the text into one utterance per
// sentence (natural breathing pauses), and shape each sentence's
// rate/pitch from the agent's ``emotion`` (or a vibe inferred from
// the wording) plus the sentence's own punctuation — questions rise,
// exclamations energize, ellipses trail off — instead of a flat
// robotic monotone.
//
// Runs the REAL production ``media/main.js`` in jsdom (only the
// vscode host API and the Web Speech API are recording stubs, as in
// every webview test).  Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/talkEmotionProsody.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
 */
function makeWebview() {
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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win};
}

/**
 * Install a recording Web Speech API on *win* (jsdom has none).  The
 * stub records every spoken utterance — including the ``rate``,
 * ``pitch``, ``lang`` and ``voice`` the production code assigns —
 * and returns the array of spoken utterances.
 */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    getVoices: () => [],
    speak: u => spoken.push(u),
  };
  return spoken;
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Send one talk event stamped for this webview's active tab. */
function talk(win, language, text, emotion) {
  send(win, {
    type: 'talk',
    language,
    text,
    emotion,
    tabId: win._demoApi.getActiveTabId(),
  });
}

/** Speak *text* with *emotion* in a fresh webview; return utterances. */
function speak(text, emotion) {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  talk(win, 'en-US', text, emotion);
  return spoken;
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

// ---------------------------------------------------------------------------

test('splits multi-sentence text into one utterance per sentence', () => {
  const spoken =
      speak('First sentence here. Second sentence here. Third one here.');
  assert.strictEqual(spoken.length, 3, 'one utterance per sentence');
  assert.strictEqual(spoken[0].text, 'First sentence here.');
  assert.strictEqual(spoken[1].text, 'Second sentence here.');
  assert.strictEqual(spoken[2].text, 'Third one here.');
});

test('splits at blank lines (paragraph breaks) too', () => {
  const spoken = speak('Hello there\n\nGoodbye now');
  assert.strictEqual(spoken.length, 2);
  assert.strictEqual(spoken[0].text, 'Hello there');
  assert.strictEqual(spoken[1].text, 'Goodbye now');
});

test('a question rises in pitch above a plain statement', () => {
  const spoken =
      speak('This is a plain statement. Would you like to proceed?');
  assert.strictEqual(spoken.length, 2);
  assert.ok(spoken[1].pitch > spoken[0].pitch,
            `question pitch ${spoken[1].pitch} must exceed ` +
                `statement pitch ${spoken[0].pitch}`);
});

test('an exclamation adds energy: higher rate and pitch than a statement',
     () => {
  const spoken = speak('Here is the plan. We ship it now!', 'neutral');
  assert.strictEqual(spoken.length, 2);
  assert.ok(spoken[1].rate > spoken[0].rate,
            'exclamation must speed up relative to the statement');
  assert.ok(spoken[1].pitch > spoken[0].pitch,
            'exclamation must raise pitch relative to the statement');
});

test('a trailing ellipsis slows the rate below neutral', () => {
  const spoken = speak('Let me think about this...');
  assert.strictEqual(spoken.length, 1);
  assert.ok(spoken[0].rate < 1.0,
            `ellipsis rate ${spoken[0].rate} must trail off below 1.0`);
});

test('emotion "excited" speaks faster and higher than neutral', () => {
  const excited = speak('The tests are all green now.', 'excited');
  const neutral = speak('The tests are all green now.', 'neutral');
  assert.strictEqual(excited.length, 1);
  assert.strictEqual(neutral.length, 1);
  assert.ok(excited[0].rate > neutral[0].rate, 'excited rate > neutral');
  assert.ok(excited[0].pitch > neutral[0].pitch, 'excited pitch > neutral');
});

test('emotion "calm" slows below the neutral rate', () => {
  const spoken = speak('Take a breath and relax.', 'calm');
  assert.strictEqual(spoken.length, 1);
  assert.ok(spoken[0].rate < 1.0, `calm rate ${spoken[0].rate} must be < 1`);
});

test('emotion "sad" lowers pitch below neutral', () => {
  const spoken = speak('The run did not go as planned.', 'sad');
  assert.strictEqual(spoken.length, 1);
  assert.ok(spoken[0].pitch < 1.0, 'sad pitch must be below neutral');
});

test('an unknown emotion falls back to neutral prosody', () => {
  const spoken = speak('This is a plain statement.', 'bogus-emotion');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(spoken[0].rate, 1.0);
  assert.strictEqual(spoken[0].pitch, 1.0);
});

test('inherited object property names are not treated as emotions', () => {
  for (const evil of ['constructor', 'toString', 'hasOwnProperty']) {
    const spoken = speak('This is a plain statement.', evil);
    assert.strictEqual(spoken.length, 1, `still speaks for "${evil}"`);
    assert.strictEqual(spoken[0].rate, 1.0,
                       `"${evil}" must fall back to a neutral rate`);
    assert.strictEqual(spoken[0].pitch, 1.0,
                       `"${evil}" must fall back to a neutral pitch`);
  }
});

test('no emotion given: infers an apologetic vibe from the wording', () => {
  const spoken = speak('Sorry about the noise in the last run.');
  assert.strictEqual(spoken.length, 1);
  assert.ok(spoken[0].rate < 1.0, 'apologetic slows down');
  assert.ok(spoken[0].pitch < 1.0, 'apologetic lowers pitch');
});

test('no emotion given: infers excitement from double exclamations', () => {
  const spoken = speak('It works! It really works!');
  assert.strictEqual(spoken.length, 2);
  assert.ok(spoken[0].rate > 1.0, 'excited inference speeds up');
  assert.ok(spoken[0].pitch > 1.0, 'excited inference raises pitch');
});

test('strips markdown emphasis and emoji before speaking', () => {
  const spoken = speak('**Done!** \u{1F389}');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(spoken[0].text, 'Done!');
});

test('strips code fences, backticks, headings and bullets', () => {
  const spoken = speak(
      '# Status\n\n- ran `pytest` fine\n\n```sh\necho hi\n```\nAll good.');
  assert.strictEqual(spoken.length, 3);
  assert.strictEqual(spoken[0].text, 'Status');
  assert.strictEqual(spoken[1].text, 'ran pytest fine');
  assert.ok(!/[`#*]/.test(spoken.map(u => u.text).join(' ')),
            'no markdown punctuation may reach the speech engine');
});

test('speaks nothing when the text is only emoji', () => {
  const spoken = speak('\u{1F389}\u{1F680} \u2728');
  assert.strictEqual(spoken.length, 0, 'nothing speakable remains');
});

test('sets the requested language on every sentence utterance', () => {
  const spoken = speak('First sentence here. Second sentence here.');
  assert.strictEqual(spoken.length, 2);
  for (const u of spoken) assert.strictEqual(u.lang, 'en-US');
});

test('rate and pitch always stay within the Web Speech safe ranges', () => {
  const emotions = ['cheerful', 'excited', 'playful', 'curious', 'warm',
                    'proud', 'calm', 'empathetic', 'reassuring',
                    'apologetic', 'serious', 'sad', ''];
  const text = 'Wow, look at that! Really? Hmm, let me see... ' +
      'Okay. Amazing! And one more thing. Done!';
  for (const emotion of emotions) {
    const spoken = speak(text, emotion);
    assert.ok(spoken.length > 1, 'multi-sentence text splits');
    for (const u of spoken) {
      assert.ok(Number.isFinite(u.rate), `rate finite for "${emotion}"`);
      assert.ok(Number.isFinite(u.pitch), `pitch finite for "${emotion}"`);
      assert.ok(u.rate >= 0.6 && u.rate <= 1.6,
                `rate ${u.rate} in [0.6, 1.6] for "${emotion}"`);
      assert.ok(u.pitch >= 0.5 && u.pitch <= 1.8,
                `pitch ${u.pitch} in [0.5, 1.8] for "${emotion}"`);
    }
  }
});

test('long replies vary pitch across sentences (no monotone)', () => {
  const spoken = speak(
      'Sentence number one. Sentence number two. Sentence number three.',
      'neutral');
  assert.strictEqual(spoken.length, 3);
  const pitches = new Set(spoken.map(u => u.pitch));
  assert.ok(pitches.size > 1, 'per-sentence drift must vary the pitch');
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) {
    console.error(`FAIL: ${f.name}`);
    console.error(f.error && f.error.stack ? f.error.stack : f.error);
  }
  process.exit(1);
}
console.log('PASS: talkEmotionProsody.test.js');
