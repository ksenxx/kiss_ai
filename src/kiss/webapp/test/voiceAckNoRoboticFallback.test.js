// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {installSpeech, assertNoSpeech} = require('./speech_helpers');
const {makeRunner} = require('./ui_antipattern_harness');
const {send: sendHostMessage, settle} = require('./simplify2_harness');
const {makeBareVoiceWindow} = require('./voice_page_helpers');
const {test, report} = makeRunner();

function makeWindow(cfg, {withAudio = true} = {}) {
  const record = {audioUrls: [], audioPlays: 0, posts: [], spoken: null};
  const win = makeBareVoiceWindow(cfg, w => {
    record.spoken = installSpeech(w);
    if (withAudio) {
      w.Audio = function Audio(url) {
        record.audioUrls.push(String(url));
        this.play = () => {
          record.audioPlays++;
          return Promise.reject(new Error('NotAllowedError: autoplay'));
        };
      };
    } else {
      w.Audio = undefined;
    }
    w.addEventListener('kiss-voice-post', event => {
      record.posts.push(event.detail);
    });
  });
  return {win, record};
}

async function main() {
  await test('a dictated task plays the synthesized clip; rejected play() stays silent', async () => {
    const {win, record} = makeWindow({
      ackAudioUrl: 'https://localhost/media/working-on-it.mp3',
    });
    sendHostMessage(win, {type: 'voiceSpeech', text: 'Deploy it', speaker: 1});
    await settle();
    assert.deepStrictEqual(
      record.audioUrls,
      ['https://localhost/media/working-on-it.mp3'],
      'the page must try the synthesized ack clip',
    );
    assert.strictEqual(record.audioPlays, 1, 'clip play() must be attempted');
    assertNoSpeech(record.spoken, 'rejected play() must degrade to silence');
    assert.deepStrictEqual(
      record.posts,
      [],
      'the ack is played in the page, nothing is posted for it',
    );
  });

  await test('without an Audio API: silent, no robotic fallback, no crash', async () => {
    const {win, record} = makeWindow(
      {ackAudioUrl: 'https://localhost/media/working-on-it.mp3'},
      {withAudio: false},
    );
    sendHostMessage(win, {type: 'voiceSpeech', text: 'Ship it', speaker: 3});
    await settle();
    assertNoSpeech(record.spoken, 'missing Audio API');
    assert.deepStrictEqual(record.audioUrls, [], 'no Audio constructor to call');
    assert.deepStrictEqual(record.posts, [], 'nothing is posted for the ack');
  });

  report('voiceAckNoRoboticFallback');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
