// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit J (2026-10-05) of media/main.js, end-to-end under jsdom.
//
// 1. A `tabs_state` snapshot that drops a chat tab (another surface closed
//    it) removes that tab and its sub-agent descendants, dismisses the
//    "Waiting for your answer" notice of a removed tab, and leaves every
//    content tab (a file opened from that chat) untouched: content tabs
//    are per-surface editors, never in the registry, and have no
//    `parentTabId`, so reconcileTabs has no editor to dispose.
// 2. `talk` playback: a muted event is dropped before it reaches the
//    queue, an audible one plays unmuted, and the queue moves on when the
//    clip ends.

/* global require, __dirname, console, process, setTimeout */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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
  const sent = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => sent.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, sent};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabIds(win) {
  return Array.from(
    win.document.querySelectorAll('.chat-tab[data-tab-id]'),
  ).map(el => el.getAttribute('data-tab-id'));
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must be in the strip`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function notice(win, id) {
  return win.document.querySelector(
    `.kiss-notification[data-notification-id=${JSON.stringify(id)}]`,
  );
}

// One event-loop turn: lets the theme MutationObserver callbacks a class
// change queued run while the document still exists, so win.close() tears
// down a quiet window.
function tick() {
  return new Promise(r => setTimeout(r, 0));
}

async function testRegistryRemovalKeepsContentTabs() {
  const {win, sent} = makeWebview();
  const T1 = 'tab-one';
  const T2 = 'tab-two';
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: T1, chatId: 'chat-1', title: 'first', workDir: ''},
      {tabId: T2, chatId: 'chat-2', title: 'second', workDir: ''},
    ],
  });
  assert.deepStrictEqual(tabIds(win), [T1, T2]);

  // A live sub-agent of T1 and a file opened while T1 is on screen.
  clickTab(win, T1);
  const SUB = T1 + '__sub_child-1';
  send(win, {
    type: 'openSubagentTab',
    tab_id: SUB,
    parent_tab_id: T1,
    description: 'child',
    task_id: 'child-1',
    isSubagentTab: true,
    isDone: false,
  });
  send(win, {
    type: 'fileContent',
    name: 'notes.html',
    path: '/tmp/notes.html',
    content: '<h1>notes</h1>',
    tabId: T1,
  });
  const fileTabId = tabIds(win).find(
    id => id !== T1 && id !== T2 && id !== SUB,
  );
  assert.ok(fileTabId, 'the file opened as a tab');
  const fileView = win.document.querySelector('.content-tab-view');
  assert.ok(fileView, 'the file tab rendered its view');

  // T2 is waiting for an answer: its sticky notice is up.
  send(win, {type: 'askUser', tabId: T2, question: 'Which one?'});
  assert.ok(notice(win, 'ask:' + T2), 'the ask notice shows for T2');

  // Another surface closed T1 and T2: the snapshot lists neither.
  sent.length = 0;
  send(win, {type: 'tabs_state', tabs: []});
  const after = tabIds(win);
  assert.ok(!after.includes(T1), 'T1 left with the registry');
  assert.ok(!after.includes(T2), 'T2 left with the registry');
  assert.ok(!after.includes(SUB), "T1's sub-agent tab followed its parent");
  assert.ok(after.includes(fileTabId), 'the file tab survived');
  assert.strictEqual(
    fileView.isConnected,
    true,
    'the file view was not disposed',
  );
  assert.strictEqual(
    notice(win, 'ask:' + T2),
    null,
    'the removed tab took its ask notice away',
  );
  // A removal mirrored from the registry is never echoed as a close.
  assert.ok(
    !sent.some(m => m && m.type === 'closeTab'),
    'no closeTab echoed for a registry removal',
  );
  // The file tab is still usable: switching to it shows its view.
  clickTab(win, fileTabId);
  assert.notStrictEqual(fileView.style.display, 'none');
  await tick();
  win.close();
}

function installAudio(win) {
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    this.playCalls = 0;
    this.play = () => {
      this.playCalls++;
      return Promise.resolve();
    };
    this.fire = type => {
      const handler = this['on' + type];
      if (typeof handler === 'function') handler({type});
    };
    players.push(this);
  };
  return players;
}

async function testTalkPlaysUnmutedAndSkipsMuted() {
  const {win} = makeWebview();
  const players = installAudio(win);
  send(win, {type: 'talk', text: 'quiet', audioB64: 'QUJD', muted: true});
  assert.strictEqual(players.length, 0, 'a muted talk creates no player');
  send(win, {type: 'talk', text: 'one', audioB64: 'QUJD', talkId: 'a'});
  send(win, {type: 'talk', text: 'two', audioB64: 'QUJD', talkId: 'b'});
  await tick();
  assert.strictEqual(players.length, 1, 'one clip plays at a time');
  assert.strictEqual(players[0].playCalls, 1);
  assert.ok(!players[0].muted, 'an audible clip is never muted');
  players[0].fire('ended');
  await tick();
  assert.strictEqual(players.length, 2, 'the queue moved on to the next clip');
  assert.ok(!players[1].muted);
  players[1].fire('ended');
  await tick();
  win.close();
}

async function main() {
  await testRegistryRemovalKeepsContentTabs();
  await testTalkPlaysUnmutedAndSkipsMuted();
  console.log('audit1005_main_js_scope_j.test.js: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
