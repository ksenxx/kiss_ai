// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const VOICE_JS_PATH = path.join(__dirname, '..', 'media', 'voice.js');

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
}

function flush() {
  return new Promise(resolve => setTimeout(resolve, 25));
}

function makeWindow(cfg, {withAudio = true} = {}) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn" class="toggle-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.__VOICE__ = cfg;

  const record = {
    audioUrls: [],
    audioPlays: 0,
    spoken: [],
    posts: [],
  };

  if (withAudio) {
    win.Audio = function Audio(url) {
      record.audioUrls.push(String(url));
      this.play = () => {
        record.audioPlays++;
        return Promise.reject(new Error('NotAllowedError: autoplay'));
      };
    };
  } else {
    win.Audio = undefined;
  }
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
  };
  win.speechSynthesis = {
    speak: utter => record.spoken.push(utter && utter.text),
    cancel: () => {},
    getVoices: () => [],
  };
  win.addEventListener('kiss-voice-post', event => {
    record.posts.push(event.detail);
  });

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);
  return {win, record};
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

async function main() {
  await test(
    'webview mode: dictated task delegates the ack to the host (voiceAck)',
    async () => {
      const {win, record} = makeWindow({
        mode: 'webview',
        ackAudioUrl: 'https://localhost/media/working-on-it.mp3',
      });
      sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the bug', speaker: 1});
      await flush();
      const acks = record.posts.filter(m => m && m.type === 'voiceAck');
      assert.strictEqual(
        acks.length,
        1,
        `expected exactly one voiceAck post, got ${JSON.stringify(record.posts)}`,
      );
    },
  );

  await test(
    'webview mode: never plays Audio and never speaks with the robotic voice',
    async () => {
      const {win, record} = makeWindow({
        mode: 'webview',
        ackAudioUrl: 'https://localhost/media/working-on-it.mp3',
      });
      sendHostMessage(win, {type: 'voiceSpeech', text: 'Run the tests', speaker: 2});
      await flush();
      assert.deepStrictEqual(
        record.audioUrls,
        [],
        'webview must not construct an Audio element for the ack',
      );
      assert.deepStrictEqual(
        record.spoken,
        [],
        `robotic Web Speech ack must never fire, spoke: ${JSON.stringify(record.spoken)}`,
      );
    },
  );

  await test(
    'browser mode: plays the synthesized clip; rejected play() stays silent',
    async () => {
      const {win, record} = makeWindow({
        mode: 'browser',
        ackAudioUrl: 'https://localhost/media/working-on-it.mp3',
      });
      sendHostMessage(win, {type: 'voiceSpeech', text: 'Deploy it', speaker: 1});
      await flush();
      assert.deepStrictEqual(
        record.audioUrls,
        ['https://localhost/media/working-on-it.mp3'],
        'browser mode must try the synthesized ack clip',
      );
      assert.strictEqual(record.audioPlays, 1, 'clip play() must be attempted');
      assert.deepStrictEqual(
        record.spoken,
        [],
        `rejected play() must degrade to silence, spoke: ${JSON.stringify(record.spoken)}`,
      );
      const acks = record.posts.filter(m => m && m.type === 'voiceAck');
      assert.deepStrictEqual(acks, [], 'browser mode must not post voiceAck');
    },
  );

  await test(
    'browser mode without an Audio API: silent, no robotic fallback, no crash',
    async () => {
      const {win, record} = makeWindow(
        {mode: 'browser', ackAudioUrl: 'https://localhost/media/working-on-it.mp3'},
        {withAudio: false},
      );
      sendHostMessage(win, {type: 'voiceSpeech', text: 'Ship it', speaker: 3});
      await flush();
      assert.deepStrictEqual(
        record.spoken,
        [],
        `robotic Web Speech ack must never fire, spoke: ${JSON.stringify(record.spoken)}`,
      );
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
