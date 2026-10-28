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
    spoken.length,
    0,
    `${scenario}: Web Speech must never speak, spoke ` +
      JSON.stringify(spoken.map(u => u.text)),
  );
}

/**
 * Install a recording `Audio` constructor on a jsdom window.  Every
 * player built is returned with its `src`, `playCalls` / `pauseCalls`
 * counters, a `currentTime` that only moves when a test calls
 * `advance(seconds)` (so a stalled clip is one the test stops advancing),
 * and `fire(type)`, which invokes the `on<type>` handler the page set
 * (`fire('ended')`, `fire('error')`).
 *
 * @param {*} [playResult] What `play()` returns: a promise, or a function
 *   of the player returning one; `undefined` resolves at once.
 * @returns {Array<object>} Every player the page constructed, in order.
 */
function installAudio(win, playResult) {
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    this.currentTime = 0;
    this.playCalls = 0;
    this.pauseCalls = 0;
    this.pause = () => {
      this.pauseCalls++;
    };
    this.advance = seconds => {
      this.currentTime += seconds;
    };
    this.play = () => {
      this.playCalls++;
      const result =
        typeof playResult === 'function' ? playResult(this) : playResult;
      return result === undefined ? Promise.resolve() : result;
    };
    this.fire = type => {
      const handler = this['on' + type];
      if (typeof handler === 'function') handler({type});
    };
    players.push(this);
  };
  return players;
}

module.exports = {installSpeech, assertNoSpeech, installAudio};
