// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test of media/main.js's `status` handling: a
// `running: true` status without a start (the daemon's submit broadcast)
// clears the end of the task before it, so the timer does not read the
// new run as finished and flip the tab back to "Done".  A `task_events`
// replay of a running task (no end) drops the old end too, so that
// status leaves the clock on the replayed task's start.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  fakeIntervals,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

function makePage() {
  let clock;
  const {win, posted} = makeChatPage({
    beforeScripts(w) {
      stubRaf(w);
      clock = fakeIntervals(w);
    },
  });
  return {win, posted, clock};
}

async function main() {
  await test('a running status with no start keeps the new run running', () => {
    const {win, clock} = makePage();
    const tabId = win._testApi.getActiveTabId();
    const start = Date.now() - 120000;
    send(win, {type: 'status', running: true, tabId, startTs: start});
    send(win, {type: 'task_done', tabId, startTs: start, endTs: start + 60000});
    send(win, {type: 'status', running: false, tabId});
    const status = win.document.getElementById('status-text');
    assert.ok(/Done/.test(status.textContent), status.textContent);
    // The next submit: the daemon's broadcast carries no start.
    send(win, {type: 'status', running: true, tabId});
    clock.tick(1000);
    const tab = win._testApi.openTabs().find(t => t.id === tabId);
    assert.ok(tab.isRunning, 'the tab is running');
    assert.notStrictEqual(
      win.document.getElementById('stop-btn').style.display,
      'none',
      'Stop is on screen',
    );
    assert.ok(!/Done/.test(status.textContent), status.textContent);
    // The task's own status brings the real start.
    send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
    clock.tick(1000);
    assert.ok(win._testApi.openTabs().find(t => t.id === tabId).isRunning);
    win.close();
  });

  await test('a background tab is not finished by its old end either', () => {
    const {win} = makePage();
    const home = win._testApi.getActiveTabId();
    const start = Date.now() - 120000;
    win._testApi.createNewTab();
    send(win, {type: 'status', running: true, tabId: home, startTs: start});
    send(win, {
      type: 'task_done',
      tabId: home,
      startTs: start,
      endTs: start + 60000,
    });
    send(win, {type: 'status', running: false, tabId: home});
    send(win, {type: 'status', running: true, tabId: home});
    win._testApi.switchToTab(home);
    const tab = win._testApi.openTabs().find(t => t.id === home);
    assert.ok(tab.isRunning, 'still running once on screen');
    assert.ok(
      !/Done/.test(win.document.getElementById('status-text').textContent),
    );
    win.close();
  });

  for (const visible of [true, false]) {
    const where = visible ? 'on screen' : 'in the background';
    await test(`a replayed running task keeps its start ${where}`, () => {
      const {win, clock} = makePage();
      const home = win._testApi.getActiveTabId();
      const now = Date.now();
      const start = now - 300000;
      send(win, {type: 'status', running: true, tabId: home, startTs: start});
      send(win, {
        type: 'task_done',
        tabId: home,
        startTs: start,
        endTs: start + 60000,
      });
      send(win, {type: 'status', running: false, tabId: home});
      if (!visible) win._testApi.createNewTab();
      // The tab's next task, started 100 s ago and still running, is
      // replayed; its row has no end yet.
      send(win, {
        type: 'task_events',
        tabId: home,
        task: 'second task',
        task_id: 2,
        events: [{type: 'text_delta', text: 'working'}],
        extra: JSON.stringify({startTs: now - 100000}),
      });
      send(win, {type: 'status', running: true, tabId: home});
      if (!visible) win._testApi.switchToTab(home);
      clock.tick(1000);
      const status = win.document.getElementById('status-text').textContent;
      assert.ok(/^Running 1m/.test(status), status);
      win.close();
    });
  }

  report('runningStatusNoStart_sweep1030');
}

main();
