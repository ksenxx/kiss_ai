// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E page tests for round-owner hygiene in the in-page (browser mic)
// voice pipeline of voice.js, and for the debug-mode free recognizer.
//
// Defects being reproduced (each test failed against the pre-fix code):
//
//  CAND-1  Stopping the pipeline mid-capture (stopBrowserPipeline) threw
//          the active capture away but left its just-begun round in
//          roundOwners. No voiceTranscribe was ever posted for it, so no
//          transcript could ever retire it, and the NEXT transcript was
//          paired with the stale owner: words spoken later on tab B were
//          charged to tab A and silently dropped or typed into the wrong
//          task.
//
//  CAND-2  finishCapture's no-audio branch retired the OLDEST round.
//          Rounds overlap by design (wake B can begin while round A's
//          audio is still being transcribed), and the round that produced
//          no audio is the one that JUST began — the NEWEST. Retiring the
//          oldest consumed round A's owner, so A's transcript was then
//          paired with owner B: one tab's words submitted into another
//          tab's conversation.
//
//  CAND-3  With kissVoiceDebug=1, startBrowserPipeline created a second,
//          grammar-free KaldiRecognizer for logging, held only by a
//          closure local. stopBrowserPipeline removed the module-level
//          `recognizer` but never the free one, leaking one worker-side
//          recognizer (a model.recognizers map entry) per pipeline start.
//
//  REV-1   The transcribe-flash safety timer FORGOT the outstanding rounds
//          (no cancelled credit), so a reply arriving after it had fired,
//          with no newer round to claim it, retired as "no round at all"
//          and was typed into whatever chat was on screen: words spoken
//          in chat A submitted into chat B.
//
//  REV-2   A reply lost with its socket connection has no transcript on
//          its way, so the credit a later mic-off turned it into must not
//          eat the next valid dictation. The daemon echoes each round's
//          tag, so the next round's reply names its round and never
//          touches that credit. (An untagged reply cannot tell the two
//          apart, so it is charged to the credit and fails closed.)
//
// jsdom provides no speech engine, audio graph, or microphone, so the
// harness supplies the minimal window.Vosk / AudioContext / mediaDevices
// capabilities the page expects from its embedder — the same approach as
// voiceBrowserPipelineLifecycle.test.js. The fake KaldiRecognizer keeps a
// model.recognizers map exactly like the real vosk.js Model does, and the
// fake ScriptProcessorNode hands its onaudioprocess callback to the test
// so real audio blocks drive the real feedCapture/finishCapture logic.
// Everything else (the real voice.js, main.js, tabs, message plumbing) is
// exercised for real.

const assert = require('assert');
const {send, twoTabs, clickTab} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  makeVoicePage,
  micBtn,
  startPipeline,
  feed,
  lastProcessor,
  wake,
  speak,
  fireTimer,
} = require('./voice_page_helpers');

const {test, report} = makeRunner();

function submits(posted) {
  return posted.filter(m => m.type === 'submit');
}

async function main() {
  await test(
    'CAND-1: turning the mic off mid-capture retires the just-begun round, ' +
      'so the next transcript pairs with the round that spoke it',
    async () => {
      const page = makeVoicePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first); // tab A on screen

      await startPipeline(page);
      wake(page); // capture begins: round owned by tab A

      // The user turns the mic off mid-capture: the capture's audio dies
      // with the pipeline (no voiceTranscribe was posted), so its round
      // must die with it too.
      micBtn(win).click();
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        0,
        'the abandoned capture must not have posted audio',
      );

      // Mic back on, move to tab B, and speak a full round there.
      await startPipeline(page);
      clickTab(win, second); // tab B on screen
      wake(page); // round B begins on tab B
      speak(page); // its audio is posted

      // The daemon answers round B. Its words belong to tab B, the tab on
      // screen — never to the abandoned round's tab A.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'round B words QC1'});
      const subs = submits(posted);
      assert.strictEqual(
        subs.length,
        1,
        "round B's words belong to the tab that spoke them: " +
          JSON.stringify(posted),
      );
      assert.ok(subs[0].prompt.includes('round B words QC1'));
      assert.strictEqual(subs[0].tabId, second);
      win.close();
    },
  );

  await test(
    'CAND-2: a wake that captures no audio retires ITS OWN round, not the ' +
      'oldest one still awaiting its transcript',
    async () => {
      const page = makeVoicePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first); // tab A on screen

      await startPipeline(page);

      // Round A: wake, speech, 2 s silence -> audio posted, transcript
      // pending. Its owner (tab A) must stay queued until the reply.
      wake(page);
      speak(page);

      // Round B begins on tab B while A transcribes, then the user says
      // nothing for 5 s: the no-audio branch must retire round B itself.
      clickTab(win, second);
      wake(page);
      feed(lastProcessor(page), false, 20); // 5.12 s, no speech ever started
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
        'the silent round must not post audio',
      );

      // Round A's transcript arrives. Its words were spoken on tab A; tab
      // B is on screen, so they are dropped — never submitted into tab B.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'round A words QC3'});
      assert.strictEqual(
        submits(posted).length,
        0,
        "round A's words must not be typed into tab B's conversation: " +
          JSON.stringify(posted),
      );
      assert.ok(
        !win.document.getElementById('task-input').value.includes('QC3'),
        "round A's words must not be drafted into tab B's input either",
      );
      win.close();
    },
  );

  await test(
    'CAND-2: a rounds reset mid-capture leaves the silent finish with ' +
      'nothing to retire and the accounting sound',
    async () => {
      const page = makeVoicePage();
      const {win, posted} = page;
      const {first} = twoTabs(win);
      clickTab(win, first);

      await startPipeline(page);
      wake(page); // capture begins: one round
      // The 45 s wake-flash safety timer fires mid-capture (the audio clock
      // stalled); voice.js cancels every outstanding round. The capture
      // itself lives on and ends in the no-audio branch, which now has
      // nothing of its own left to retire — and must not eat into
      // anything else.
      fireTimer(page, 45000);
      assert.ok(
        !micBtn(win).classList.contains('voice-triggered'),
        'the safety timer clears the wake flash',
      );
      feed(lastProcessor(page), false, 20); // 5.12 s: silent finish

      // The cancellation moved the capture's round into the cancelled
      // credit, so the next transcript fails CLOSED against it (by
      // design: a cancelled round's words may not be typed anywhere).
      // The silent finish must not have consumed that credit, nor pushed
      // any counter negative.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'cancelled words QC4'});
      assert.strictEqual(
        submits(posted).length,
        0,
        "a cancelled round's transcript must fail closed: " +
          JSON.stringify(posted),
      );

      // With the credit spent, an unsolicited transcript flows again: the
      // accounting is exactly one round behind, never stale forever.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'after reset words QC5'});
      assert.strictEqual(
        submits(posted).length,
        1,
        'an unsolicited transcript must still be delivered: ' +
          JSON.stringify(posted),
      );
      win.close();
    },
  );

  await test(
    'REV-1: a reply arriving after the transcribe window ran out is refused, ' +
      'never typed into the chat now on screen',
    async () => {
      const page = makeVoicePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first); // chat A on screen

      await startPipeline(page);
      wake(page); // round A begins on chat A
      speak(page); // audio posted, transcript pending

      // The user moves to chat B; the daemon is slow and the 60 s
      // transcribing flash runs out before its reply lands.
      clickTab(win, second);
      fireTimer(page, 60000);
      assert.ok(
        !micBtn(win).classList.contains('voice-transcribing'),
        'the safety timer clears the transcribing flash',
      );

      // Round A's delayed reply arrives with no newer round to claim it.
      // Its words were spoken in chat A: they must not be submitted into
      // (or drafted for) chat B.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'words spoken in chat A QR1'});
      assert.deepStrictEqual(
        submits(posted).map(m => ({onB: m.tabId === second, prompt: m.prompt})),
        [],
        "a cancelled round's late words must not be submitted: " +
          JSON.stringify(posted),
      );
      assert.ok(
        !win.document.getElementById('task-input').value.includes('QR1'),
        "nor drafted into chat B's input",
      );
      win.close();
    },
  );

  await test(
    'REV-2: a reply lost with the connection, then a mic restart, never ' +
      'eats the next valid dictation',
    async () => {
      const page = makeVoicePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first); // chat A on screen

      await startPipeline(page);
      wake(page); // round A begins on chat A
      speak(page); // audio posted

      // The socket drops while the daemon transcribes: its reply to
      // round A is lost for good. The user turns the mic off and on
      // again, the page reconnects, and they dictate in chat B.
      send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
      micBtn(win).click(); // off: round A is cancelled
      send(win, {type: 'daemonStatus', connected: true});
      await startPipeline(page);
      clickTab(win, second);
      wake(page); // round B begins on chat B
      speak(page);
      const audio = posted.filter(m => m.type === 'voiceTranscribe');
      assert.strictEqual(audio.length, 2, 'only rounds A and B ever posted audio');

      // Only round B's reply ever arrives, with the tag the daemon echoes.
      // It belongs to the live round, never to the credit the lost round
      // left behind.
      posted.length = 0;
      send(win, {
        type: 'voiceSpeech',
        text: 'valid words spoken in chat B QR2',
        round: audio[1].round,
      });
      assert.deepStrictEqual(
        submits(posted).map(m => ({onB: m.tabId === second, prompt: m.prompt})),
        [{onB: true, prompt: 'valid words spoken in chat B QR2'}],
        "round B's words must reach the chat that spoke them: " +
          JSON.stringify(posted),
      );
      win.close();
    },
  );

  await test('CAND-3: stopping the pipeline removes the debug free recognizer too', async () => {
    const page = makeVoicePage({debug: true});
    const {win, state, model} = page;

    await startPipeline(page);
    assert.strictEqual(
      state.recognizers.length,
      2,
      'debug mode runs a wake recognizer plus a free-text one',
    );
    assert.strictEqual(model.recognizers.size, 2);

    micBtn(win).click(); // off: stopBrowserPipeline
    assert.strictEqual(
      model.recognizers.size,
      0,
      'every recognizer the pipeline registered must be removed: ' +
        JSON.stringify(Array.from(model.recognizers.keys())),
    );
    for (const rec of state.recognizers) {
      assert.ok(
        rec.removed,
        `recognizer ${rec.id} (grammar: ${rec.grammar || 'free'}) leaked`,
      );
    }
    win.close();
  });

  await test('CAND-3: repeated debug-mode pipeline restarts leave no recognizer behind', async () => {
    const page = makeVoicePage({debug: true});
    const {win, state, model} = page;
    await startPipeline(page);

    // Off and on again restart the pipeline; each cycle creates a fresh
    // recognizer pair and must free the previous one entirely.
    micBtn(win).click(); // off
    await startPipeline(page);
    assert.strictEqual(
      model.recognizers.size,
      2,
      'only the live pipeline pair may be registered after a restart: ' +
        JSON.stringify(Array.from(model.recognizers.keys())),
    );
    assert.strictEqual(state.recognizers.length, 4);

    micBtn(win).click(); // off
    assert.strictEqual(
      model.recognizers.size,
      0,
      'stopping must free the last pair as well',
    );
    win.close();
  });

  report('conc2026_voice_rounds');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
