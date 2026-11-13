// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  installSpeech,
  assertNoSpeech,
  installAudio,
} = require('./speech_helpers');
const {makeRunner} = require('./ui_antipattern_harness');
const {settle} = require('./simplify2_harness');
const {
  makeBareVoicePage,
  startPipeline,
  dictate,
} = require('./voice_page_helpers');
const {test, report} = makeRunner();

// Every page built; closed after the run so node exits on its own.
const windows = [];

/** What a browser's autoplay policy answers a play() it refuses. */
function refusedPlay() {
  return Promise.reject(new Error('NotAllowedError: autoplay'));
}

// A bare voice page with the pipeline live (a transcript is typed only
// for a round the page posted), its Audio players recorded (every play()
// refused) or the Audio API absent.
async function makePage(cfg, {withAudio = true} = {}) {
  const record = {players: [], spoken: null};
  const page = makeBareVoicePage(cfg, w => {
    record.spoken = installSpeech(w);
    if (withAudio) record.players = installAudio(w, refusedPlay);
    else w.Audio = undefined;
  });
  windows.push(page.win);
  await startPipeline(page);
  return {page, record};
}

async function main() {
  await test('a dictated task plays the synthesized clip; rejected play() stays silent', async () => {
    const {page, record} = await makePage({
      ackAudioUrl: 'https://localhost/media/working-on-it.mp3',
    });
    dictate(page, 'Deploy it', {speaker: 1});
    await settle();
    assert.deepStrictEqual(
      record.players.map(p => p.src),
      ['https://localhost/media/working-on-it.mp3'],
      'the page must try the synthesized ack clip',
    );
    assert.strictEqual(
      record.players[0].playCalls,
      1,
      'clip play() must be attempted',
    );
    assertNoSpeech(record.spoken, 'rejected play() must degrade to silence');
    assert.deepStrictEqual(
      page.posted.map(m => m.type),
      ['voiceTranscribe'],
      "the ack is played in the page: only the round's audio is posted",
    );
  });

  await test('without an Audio API: silent, no robotic fallback, no crash', async () => {
    const {page, record} = await makePage(
      {ackAudioUrl: 'https://localhost/media/working-on-it.mp3'},
      {withAudio: false},
    );
    dictate(page, 'Ship it', {speaker: 3});
    await settle();
    assertNoSpeech(record.spoken, 'missing Audio API');
    assert.deepStrictEqual(
      page.posted.map(m => m.type),
      ['voiceTranscribe'],
      "nothing but the round's audio is posted",
    );
  });

  windows.forEach(win => win.close());
  report('voiceAckNoRoboticFallback');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
