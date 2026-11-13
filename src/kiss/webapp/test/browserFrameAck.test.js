// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Every browserFrame the page gets is answered with a browserFrameAck
// naming the tab (kiss/server/frame_flow.py on the daemon keeps at most
// two frames in flight per client and holds the newest one back until
// then), whether the frame is painted into a browser tab's view or
// arrives for a tab the page does not show.

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  ofType,
  installXtermStub,
  openChat,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {assertJsonEqual: same} = require('./workdir_helpers');

function makePage() {
  return makeChatPage({
    beforeScripts: win => {
      stubRaf(win);
      installXtermStub(win);
    },
    extraScripts: ['browserTab.js', 'terminalTab.js'],
  });
}

const {test, report} = makeRunner();

async function main() {
  await test('each frame of a browser tab is acknowledged', () => {
    const {win, posted} = makePage();
    openChat(win);
    send(win, {
      type: 'openBrowserTab',
      tab_id: 'browser__1',
      url: 'https://example.com/',
      title: 'Example',
      focus: true,
    });
    const tab = win._testApi.openTabs().find(t => t.id === 'browser__1');
    assert.ok(tab && tab.isContentTab, 'the browser tab is open');
    assert.ok(win.document.querySelector('.browser-tab-view'), 'with its view');
    send(win, {type: 'browserFrame', tab_id: 'browser__1', data: 'AAAA', width: 10, height: 10});
    send(win, {type: 'browserFrame', tab_id: 'browser__1', data: 'BBBB', width: 10, height: 10});
    same(ofType(posted, 'browserFrameAck'), [
      {tab_id: 'browser__1', type: 'browserFrameAck'},
      {tab_id: 'browser__1', type: 'browserFrameAck'},
    ]);
    const img = win.document.querySelector('.browser-tab-view img');
    assert.ok(img && img.src.endsWith('BBBB'), 'the last frame is painted');
    win.close();
  });

  await test('a frame for a tab the page does not show is acknowledged too', () => {
    const {win, posted} = makePage();
    openChat(win);
    send(win, {type: 'browserFrame', tab_id: 'browser__9@box', data: 'CCCC'});
    same(ofType(posted, 'browserFrameAck'), [
      {tab_id: 'browser__9@box', type: 'browserFrameAck'},
    ]);
    win.close();
  });

  report('browserFrameAck');
}

main();
