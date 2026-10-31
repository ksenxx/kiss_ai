// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// voice.js pairs a `voiceSpeech` reply with the dictation (round) it
// answers.  Round A is spoken in chat X and its audio posted; the user
// turns the mic off (A is cancelled while transcribing), moves to chat Y,
// turns the mic on and dictates round B, whose audio is posted too.  Only
// then does A's late reply arrive.
//
// * With the round tag the daemon echoes (web_server.py
//   `_handle_voice_transcribe`, serialized as `{type, text, speaker,
//   language, round}`), A's reply names a cancelled round and is refused.
// * A reply without a usable tag (`round: null`, or none at all) could be
//   A's late reply or B's; voice.js used to give it to B, typing chat X's
//   words into chat Y.  It must fail closed: the cancelled round is
//   charged first, and B's own reply still answers on Y.
//
// Real chat.html, main.js and voice.js on jsdom; voice_page_helpers
// supplies the microphone, Vosk recognizer and audio graph.

const assert = require('assert');
const {send, twoTabs, clickTab} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  makeVoicePage,
  micBtn,
  startPipeline,
  wake,
  speak,
} = require('./voice_page_helpers');

const {test, report} = makeRunner();

function submits(posted) {
  return posted.filter(m => m.type === 'submit');
}

function audioRounds(posted) {
  return posted.filter(m => m.type === 'voiceTranscribe').map(m => m.round);
}

/** The reply exactly as the daemon serializes it. */
function daemonReply(text, round) {
  return {type: 'voiceSpeech', text, speaker: null, language: null, round};
}

/**
 * Round A spoken on chat X and cancelled by mic-off while transcribing,
 * then round B spoken and posted on chat Y.  Returns the tags of A and B.
 */
async function cancelAThenPostBElsewhere(page, chatX, chatY) {
  clickTab(page.win, chatX);
  await startPipeline(page);
  wake(page);
  speak(page); // round A posts its audio
  micBtn(page.win).click(); // mic off: round A is cancelled
  clickTab(page.win, chatY);
  await startPipeline(page);
  wake(page);
  speak(page); // round B posts its audio before A's reply lands
  const [roundA, roundB] = audioRounds(page.posted);
  assert.strictEqual(typeof roundA, 'number', 'the audio carries its round');
  assert.notStrictEqual(roundA, roundB, 'each round has its own tag');
  return {roundA, roundB};
}

/** No submit and no draft carries `marker`. */
function assertRefused(page, marker) {
  const {win, posted} = page;
  assert.strictEqual(
    submits(posted).length,
    0,
    `${marker} must not be submitted: ${JSON.stringify(posted)}`,
  );
  assert.ok(
    !win.document.getElementById('task-input').value.includes(marker),
    `${marker} must not be drafted into the chat on screen`,
  );
}

/** Exactly one submit, on `tabId`, carrying `marker`. */
function assertSubmittedOn(page, tabId, marker) {
  const subs = submits(page.posted);
  assert.strictEqual(subs.length, 1, JSON.stringify(page.posted));
  assert.ok(subs[0].prompt.includes(marker), subs[0].prompt);
  assert.strictEqual(subs[0].tabId, tabId);
}

async function main() {
  await test(
    "the daemon's tagged reply for a cancelled round is refused after the " +
      'next round posted; the next round answers on its own chat',
    async () => {
      const page = makeVoicePage();
      const {first, second} = twoTabs(page.win);
      const {roundA, roundB} = await cancelAThenPostBElsewhere(
        page,
        first,
        second,
      );
      page.posted.length = 0;
      send(page.win, daemonReply('chat X words QF1', roundA));
      assertRefused(page, 'QF1');
      send(page.win, daemonReply('chat Y words QF2', roundB));
      assertSubmittedOn(page, second, 'QF2');
      page.win.close();
    },
  );

  await test(
    'an untagged reply while a cancelled and a posted round are both ' +
      'outstanding fails closed; the posted round still answers on its chat',
    async () => {
      const page = makeVoicePage();
      const {first, second} = twoTabs(page.win);
      await cancelAThenPostBElsewhere(page, first, second);
      page.posted.length = 0;
      // The daemon's serialization of a reply it could not tag.
      send(page.win, daemonReply('chat X words QF3', null));
      assertRefused(page, 'QF3');
      // A reply with no round field at all is untagged too.
      send(page.win, {type: 'voiceSpeech', text: 'chat Y words QF4'});
      assertSubmittedOn(page, second, 'QF4');
      page.win.close();
    },
  );

  await test(
    'every cancelled round is charged before a posted one: two late ' +
      'untagged replies are refused, the third answers the live round',
    async () => {
      const page = makeVoicePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first);
      await startPipeline(page);
      wake(page);
      speak(page); // round A1
      wake(page);
      speak(page); // round A2
      micBtn(win).click(); // mic off: both are cancelled
      clickTab(win, second);
      await startPipeline(page);
      wake(page);
      speak(page); // round B
      assert.strictEqual(audioRounds(posted).length, 3);
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'late A1 words QF5'});
      send(win, {type: 'voiceSpeech', text: 'late A2 words QF6'});
      assertRefused(page, 'QF5');
      assertRefused(page, 'QF6');
      send(win, {type: 'voiceSpeech', text: 'chat Y words QF7'});
      assertSubmittedOn(page, second, 'QF7');
      assert.ok(
        !micBtn(win).classList.contains('voice-transcribing'),
        'no round is left awaiting a transcript',
      );
      win.close();
    },
  );

  await test('a tag this page never posted is refused', async () => {
    const page = makeVoicePage();
    const {win, posted} = page;
    const {first} = twoTabs(win);
    clickTab(win, first);
    await startPipeline(page);
    wake(page);
    speak(page);
    const [round] = audioRounds(posted);
    posted.length = 0;
    send(win, daemonReply('stray words QF8', round + 100));
    assertRefused(page, 'QF8');
    assert.ok(
      micBtn(win).classList.contains('voice-transcribing'),
      'the posted round still awaits its own transcript',
    );
    send(win, daemonReply('own words QF9', round));
    assertSubmittedOn(page, first, 'QF9');
    win.close();
  });

  report('voiceRoundFailsClosed_wave3');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
