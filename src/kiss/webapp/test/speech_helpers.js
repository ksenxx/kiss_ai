// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');

/**
 * Install a recording Web Speech stub on a jsdom window.
 *
 * media/ never references speechSynthesis; the talk/voice suites install the
 * stub anyway so that a guarded browser-speech fallback added to any audio
 * failure path (missing Audio API, throwing constructor, rejected play(),
 * error/abort events) would run and be caught by assertNoSpeech.
 *
 * @returns {Array} every utterance handed to speechSynthesis.speak().
 */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    speak: u => spoken.push(u),
    cancel: () => {},
    resume: () => {},
    getVoices: () => [],
  };
  return spoken;
}

/** Assert that the robotic Web Speech voice was never used. */
function assertNoSpeech(spoken, scenario) {
  assert.strictEqual(
    spoken.length, 0,
    `${scenario}: Web Speech must never speak, spoke ` +
      JSON.stringify(spoken.map(u => u.text)));
}

module.exports = {installSpeech, assertNoSpeech};
