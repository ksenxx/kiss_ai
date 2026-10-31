// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for the task-update poll of the desktop remote
// page (media/main.js): every getTaskUpdate may run the daemon's paid
// update agent, so the poll runs only while the task-info panel is on
// screen.  Opening a file in the content pane slides the panel off; the
// poll stops then and comes back, with an immediate poll, when the drawer
// tab brings the panel back.

'use strict';

const assert = require('assert');
const {
  makePage,
  click,
  send,
  stubRaf,
  fakeIntervals,
  fakeMatchMedia,
  ofType,
} = require('./simplify2_harness');

const POLL_MS = 5000;

function testNoPollWhileThePanelIsSlidOff() {
  let clock;
  const {win, posted} = makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts(w) {
      clock = fakeIntervals(w);
      stubRaf(w);
      fakeMatchMedia(w, true);
    },
  });
  const polls = () => ofType(posted, 'getTaskUpdate').length;
  send(win, {type: 'configData', config: {work_dir: '/cfg/dir'}, apiKeys: {}});
  send(win, {type: 'status', running: true});
  assert.ok(polls() > 0, 'a running task on the desktop page polls');
  send(win, {
    type: 'fileContent',
    name: 'report.md',
    path: '/cfg/dir/report.md',
    content: '# report',
  });
  assert.ok(
    win.document.body.classList.contains('meta-hidden'),
    'the file slid the panel off',
  );
  const hidden = polls();
  clock.tick(POLL_MS * 3);
  assert.strictEqual(polls(), hidden, 'no poll while the panel is off screen');
  click(win, win.document.getElementById('meta-drawer'));
  assert.ok(!win.document.body.classList.contains('meta-hidden'));
  assert.strictEqual(polls(), hidden + 1, 'the panel back on screen polls');
  clock.tick(POLL_MS);
  assert.strictEqual(polls(), hidden + 2, 'and keeps polling');
  win.close();
  console.log('  ok - the task update is not polled while the panel is off');
}

console.log('metaPanelPoll_sweep1030');
testNoPollWhileThePanelIsSlidOff();
