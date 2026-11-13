// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// voice.js's window 'message' listener must only act on the daemon's
// synthetic MessageEvents (dispatched by media/transport.js with no
// source window).  A sandboxed iframe on the page can postMessage to
// its parent, and such an event carries the iframe's window as
// event.source: it must not be able to type a fake transcript into the
// task input.  The listener also requires the documented wire shape
// ({type: 'voiceSpeech', text: string}) and types only the transcript
// of a round the page posted (the frame's `round` tag).

const assert = require('assert');
const {
  makeBareVoicePage,
  startPipeline,
  speakRound,
} = require('./voice_page_helpers');

const {test, report} = require('./ui_antipattern_harness').makeRunner();

// Every page built; closed after the run so node exits on its own.
const windows = [];

// A sandboxed iframe whose contentWindow serves as a foreign `source`.
function addPreviewFrame(win) {
  win.document.body.insertAdjacentHTML(
    'beforeend',
    '<iframe id="preview" sandbox="allow-scripts"></iframe>',
  );
}

// A page with one round spoken and posted; `round` is its tag.  Every
// frame below carries it: an untagged frame is refused whatever else is
// wrong with it, so only a tagged one shows which guard kept it out.
async function makeRound() {
  const page = makeBareVoicePage({}, addPreviewFrame);
  windows.push(page.win);
  await startPipeline(page);
  return {win: page.win, round: speakRound(page)};
}

function dispatch(win, init) {
  win.dispatchEvent(new win.MessageEvent('message', init));
}

function inputValue(win) {
  return win.document.getElementById('task-input').value;
}

async function main() {
  await test('a sourceless daemon event still types the transcript', async () => {
    const {win, round} = await makeRound();
    dispatch(win, {data: {type: 'voiceSpeech', text: 'hello there', round}});
    assert.strictEqual(inputValue(win), 'hello there');
  });

  await test('an event with a source window is ignored', async () => {
    const {win, round} = await makeRound();
    const iframeWin = win.document.getElementById('preview').contentWindow;
    dispatch(win, {
      data: {type: 'voiceSpeech', text: 'rm -rf everything', round},
      source: iframeWin,
      origin: 'null',
    });
    dispatch(win, {
      data: {type: 'voiceSpeech', text: 'from self', round},
      source: win,
    });
    assert.strictEqual(inputValue(win), '');
    // The fakes did not use the round up either: its real transcript,
    // the same frame without a source, is still typed.
    dispatch(win, {data: {type: 'voiceSpeech', text: 'from the daemon', round}});
    assert.strictEqual(inputValue(win), 'from the daemon');
  });

  await test('events whose data has no string text are ignored', async () => {
    const {win, round} = await makeRound();
    dispatch(win, {data: {type: 'voiceSpeech', text: 42, round}});
    dispatch(win, {data: {type: 'voiceSpeech', round}});
    dispatch(win, {data: 'voiceSpeech'});
    dispatch(win, {data: null});
    dispatch(win, {});
    assert.strictEqual(inputValue(win), '');
    dispatch(win, {data: {type: 'voiceSpeech', text: 'the words', round}});
    assert.strictEqual(inputValue(win), 'the words');
  });

  // A real iframe postMessage is not exercised here: jsdom's
  // window.postMessage never sets event.source (a TODO in
  // jsdom/lib/jsdom/browser/Window.js), so it looks exactly like the
  // daemon's synthetic event.  The sourced MessageEvents above model what
  // a browser delivers for a cross-window postMessage.

  windows.forEach(win => win.close());
  report('audit1015VoiceMessageGuard');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
