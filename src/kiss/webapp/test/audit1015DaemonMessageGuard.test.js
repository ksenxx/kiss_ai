// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// main.js's window 'message' listener feeds handleEvent, the page's
// daemon-event dispatcher.  Daemon frames arrive as the synthetic
// MessageEvents media/transport.js dispatches (no source window).  An
// opened HTML file or agent report renders in a sandboxed iframe that
// can postMessage to its parent, and such an event carries the
// iframe's window as event.source: it must not be able to impersonate
// the daemon (clear the transcript, open tabs, ask a question...).  A
// self-post (event.source === window) comes from the page's own
// scripts and is handled: the Playwright suites inject events that way.
// A payload that is not an object is dropped instead of throwing inside
// the listener.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function dispatch(win, init) {
  win.dispatchEvent(new win.MessageEvent('message', init));
}

function previewFrame(win) {
  const tabId = win._testApi.getActiveTabId();
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/report.html',
    name: 'report.html',
    tabId,
    content: '<p>report</p>',
    version: '1:1',
  });
  const frame = win.document.querySelector('iframe.content-html-frame');
  assert.ok(frame, 'the html file opened in a sandboxed preview frame');
  assert.strictEqual(frame.getAttribute('sandbox'), 'allow-scripts');
  return frame;
}

function testSourcedMessageIsIgnored() {
  const {win} = makePage();
  const chatTab = win._testApi.getActiveTabId();
  send(win, {type: 'setTaskText', text: 'the real task'});
  const frame = previewFrame(win);
  const before = win._testApi.openTabs().length;

  // An iframe-style post: the same payloads the daemon sends, but with
  // the preview's window as the source.
  dispatch(win, {
    data: {type: 'openSubagentTab', tab_id: 'evil-sub', parent_tab_id: chatTab},
    source: frame.contentWindow,
    origin: 'null',
  });
  dispatch(win, {
    data: {type: 'setTaskText', text: 'injected task', tabId: chatTab},
    source: frame.contentWindow,
    origin: 'null',
  });

  assert.strictEqual(
    win._testApi.openTabs().length,
    before,
    'a sourced openSubagentTab opens nothing',
  );
  assert.ok(
    !win._testApi.openTabs().some(t => t.id === 'evil-sub'),
    'no tab from the iframe post',
  );
  const chat = win._testApi.openTabs().find(t => t.id === chatTab);
  win._testApi.switchToTab(chatTab);
  assert.ok(chat, 'the chat tab is still there');
  assert.ok(
    !win.document.body.textContent.includes('injected task'),
    'a sourced setTaskText changes nothing on screen',
  );
  win.close();
  console.log('  ok - a message with a source window is ignored');
}

function testSourcelessMessageIsStillHandled() {
  const {win} = makePage();
  const chatTab = win._testApi.getActiveTabId();
  dispatch(win, {
    data: {type: 'openSubagentTab', tab_id: 'real-sub', parent_tab_id: chatTab},
  });
  assert.ok(
    win._testApi.openTabs().some(t => t.id === 'real-sub'),
    "the daemon's synthetic (sourceless) event opens the tab",
  );
  // The page's own scripts posting to their window (what a Playwright
  // test's page.evaluate(window.postMessage(...)) delivers).
  dispatch(win, {
    data: {type: 'openSubagentTab', tab_id: 'self-sub', parent_tab_id: chatTab},
    source: win,
  });
  assert.ok(
    win._testApi.openTabs().some(t => t.id === 'self-sub'),
    'a self-post is handled',
  );
  win.close();
  console.log('  ok - sourceless daemon events and self-posts are handled');
}

function testNonObjectPayloadIsDropped() {
  const {win} = makePage();
  const errors = [];
  win.addEventListener('error', e => errors.push(e.message));
  const before = win._testApi.openTabs().length;
  dispatch(win, {data: null});
  dispatch(win, {data: 'clear'});
  dispatch(win, {data: 42});
  dispatch(win, {});
  assert.deepStrictEqual(errors, [], 'no listener error for a bad payload');
  assert.strictEqual(win._testApi.openTabs().length, before);
  win.close();
  console.log('  ok - a non-object payload is dropped without throwing');
}

// A real iframe postMessage is not exercised here: jsdom's
// window.postMessage never sets event.source (a TODO in
// jsdom/lib/jsdom/browser/Window.js), so it looks exactly like the
// daemon's synthetic event.  The sourced MessageEvents above model what
// a browser delivers for a cross-window postMessage.

function main() {
  testSourcedMessageIsIgnored();
  testSourcelessMessageIsStillHandled();
  testNonObjectPayloadIsDropped();
  console.log('\n3 passed, 0 failed');
}

main();
