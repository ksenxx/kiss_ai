// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the time-remaining line of the Task Progress
// subpanel (#meta-info-eta in media/chat.html, the tasketa block of
// media/main.js).  A `taskUpdate` reply carries the update agent's
// estimate of when the task finishes (etaAt, etaLowAt, etaHighAt, epoch
// ms; zeros without one, kiss.server.task_update):
//
// * the line words it as the time left, with the plausible range and
//   the clock time of the finish, under the status line;
// * the page counts it down between updates, every 15 s;
// * under a minute it reads "Less than a minute left", past the latest
//   plausible finish "Taking longer than estimated";
// * without an estimate there is no line and no timer, and the timer
//   stops when the task ends, the tab changes or the next task starts.
//
// setInterval and Date.now are the page's own, driven by hand: the
// tests advance virtual time instead of sleeping.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  click,
  send,
  stubRaf,
  fakeIntervals,
  fakeMatchMedia,
  readyTabId,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {lastPoll, replyToPoll} = require('./remote_drawer_helpers');

// Every page a test built: the runner closes them when it has ended.
const openWindows = [];
const {test, report} = makeRunner({windows: openWindows});

const MIN = 60000;
const TICK_MS = 15000;
// Local noon, so a finish minutes away is on the same day in every
// time zone and one 14 hours away is on the next.
const NOON = new Date(2026, 0, 15, 12, 0, 0).getTime();

/**
 * Boot a remote page whose visible tab runs a task, at NOON on the
 * page's clock.  `desktop` (default true) is the (min-width: 900px)
 * media query: false boots the phone layout with its task-info drawer
 * open.  `page.now` is what the page's Date.now() returns.
 */
function runningPage(opts) {
  const {desktop = true} = opts || {};
  let clock;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    sourceUrl: 'tasketa-main.js',
    extraScripts: ['marked.min.js'],
    beforeScripts(win) {
      clock = fakeIntervals(win);
      stubRaf(win);
      fakeMatchMedia(win, desktop);
    },
  });
  openWindows.push(win);
  const page = {win, posted, clock, now: NOON};
  win.Date.now = () => page.now;
  send(win, {type: 'configData', config: {work_dir: '/cfg/dir'}, apiKeys: {}});
  if (!desktop) click(win, win.document.getElementById('meta-drawer-btn'));
  send(win, {type: 'status', running: true});
  return page;
}

/** Let *ms* (whole seconds) pass on the page's clock, timers included. */
function advance(page, ms) {
  for (let t = 0; t < ms; t += 1000) {
    page.now += 1000;
    page.clock.tick(1000);
  }
}

let sigs = 0;

/**
 * Answer the page's last poll with a task update; *fields* are the
 * reply's own (the estimate among them).
 */
function update(page, fields) {
  replyToPoll(
    page.win,
    lastPoll(page),
    Object.assign(
      {
        sig: 'sig-' + ++sigs,
        content: 'Half of the files are converted.',
        updatedAt: page.now,
      },
      fields,
    ),
  );
}

/** An update estimating the finish *at* minutes from now (*low* to *high*). */
function estimate(page, at, low, high) {
  update(page, {
    etaAt: page.now + at * MIN,
    etaLowAt: page.now + low * MIN,
    etaHighAt: page.now + high * MIN,
  });
}

function etaLine(page) {
  return page.win.document.getElementById('meta-info-eta').textContent;
}

/** How many countdown timers the page keeps. */
function etaTimers(page) {
  return page.clock.intervals.filter(iv => iv.ms === TICK_MS).length;
}

/** The HH:MM clock time main.js renders for an epoch-ms stamp. */
function clockLabel(page, epochMs) {
  return new page.win.Date(epochMs).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

async function main() {
  await test('an estimate shows the time left under the status line and is counted down every 15 s', () => {
    const page = runningPage();
    const line = page.win.document.getElementById('meta-info-eta');
    assert.strictEqual(etaLine(page), '');
    assert.strictEqual(etaTimers(page), 0, 'no estimate, no timer');
    estimate(page, 12, 8, 20);
    const done = clockLabel(page, NOON + 12 * MIN);
    assert.strictEqual(
      etaLine(page),
      `About 12 min left (8 to 20 min), done around ${done}`,
    );
    // One line of the subpanel, between the status and the report.
    assert.strictEqual(line.previousElementSibling.id, 'meta-info-status');
    assert.strictEqual(line.nextElementSibling.id, 'meta-info-content');
    assert.strictEqual(line.parentElement.id, 'meta-info');
    assert.strictEqual(line.getAttribute('role'), 'timer');
    assert.strictEqual(etaTimers(page), 1);
    // 30 s on, the rounded minutes are the same; at 45 s they drop.
    // The finish time does not move.
    advance(page, 30000);
    assert.strictEqual(
      etaLine(page),
      `About 12 min left (8 to 20 min), done around ${done}`,
    );
    advance(page, 15000);
    assert.strictEqual(
      etaLine(page),
      `About 11 min left (7 to 19 min), done around ${done}`,
    );
    // A poll answered "unchanged" leaves the countdown alone.
    const poll = lastPoll(page);
    send(page.win, {
      type: 'taskUpdate',
      tabId: poll.tabId,
      token: poll.token,
      exists: true,
      sig: 'sig-' + sigs,
      unchanged: true,
    });
    advance(page, 60000);
    assert.strictEqual(
      etaLine(page),
      `About 10 min left (6 to 18 min), done around ${done}`,
    );
    // The next update's estimate replaces it, on the same single timer.
    estimate(page, 30, 20, 45);
    assert.strictEqual(
      etaLine(page),
      'About 30 min left (20 to 45 min), done around ' +
        clockLabel(page, page.now + 30 * MIN),
    );
    assert.strictEqual(etaTimers(page), 1);
  });

  await test('the line words hours, a point estimate, an open lower bound and a finish on another day', () => {
    const page = runningPage();
    const around = minutes =>
      ', done around ' + clockLabel(page, NOON + minutes * MIN);
    estimate(page, 80, 50, 120);
    assert.strictEqual(
      etaLine(page),
      'About 1 h 20 min left (50 min to 2 h)' + around(80),
    );
    // The three times agree: no range to give.
    estimate(page, 180, 180, 180);
    assert.strictEqual(etaLine(page), 'About 3 h left' + around(180));
    // The likely time alone is a point estimate too.
    update(page, {etaAt: NOON + 25 * MIN});
    assert.strictEqual(etaLine(page), 'About 25 min left' + around(25));
    // It may be over within the minute, or take up to ten.
    estimate(page, 5, 0.5, 10);
    assert.strictEqual(
      etaLine(page),
      'About 5 min left (up to 10 min)' + around(5),
    );
    // "up to 5 min" would say nothing "About 5 min" does not.
    estimate(page, 5, 0.5, 5);
    assert.strictEqual(etaLine(page), 'About 5 min left' + around(5));
    // 59.6 minutes round to the hour.
    estimate(page, 59.6, 59.6, 59.6);
    assert.strictEqual(etaLine(page), 'About 1 h left' + around(59.6));
    // Tomorrow's 02:00 is named with its day.
    estimate(page, 14 * 60, 13 * 60, 15.5 * 60);
    const finish = NOON + 14 * 60 * MIN;
    assert.strictEqual(
      etaLine(page),
      'About 14 h left (13 h to 15 h 30 min), done around ' +
        new page.win.Date(finish).toLocaleDateString([], {weekday: 'short'}) +
        ' ' +
        clockLabel(page, finish),
    );
  });

  await test('under a minute the line says so; past the latest finish the task is taking longer than estimated', () => {
    const page = runningPage();
    estimate(page, 1.5, 1, 3);
    assert.strictEqual(
      etaLine(page),
      'About 2 min left (1 to 3 min), done around ' +
        clockLabel(page, NOON + 1.5 * MIN),
    );
    // 45 s left by the likely time, 2 min 15 s at the latest.
    advance(page, 45000);
    assert.strictEqual(etaLine(page), 'Less than a minute left (up to 2 min)');
    // 30 s before the latest finish.
    advance(page, 105000);
    assert.strictEqual(etaLine(page), 'Less than a minute left');
    // At the latest finish itself, still.
    advance(page, 30000);
    assert.strictEqual(etaLine(page), 'Less than a minute left');
    advance(page, 15000);
    assert.strictEqual(etaLine(page), 'Taking longer than estimated');
    assert.strictEqual(etaTimers(page), 1, 'the task still runs');
    // The next update brings a new estimate.
    estimate(page, 4, 2, 6);
    assert.strictEqual(
      etaLine(page),
      'About 4 min left (2 to 6 min), done around ' +
        clockLabel(page, page.now + 4 * MIN),
    );
  });

  await test('an update without an estimate shows no line and keeps no timer', () => {
    const page = runningPage();
    const report = page.win.document.getElementById('meta-info-content');
    // The agent could not tell (the task waits for the user): zeros.
    update(page, {
      content: 'Waiting for your answer.',
      etaAt: 0,
      etaLowAt: 0,
      etaHighAt: 0,
    });
    assert.ok(report.textContent.includes('Waiting for your answer.'));
    assert.strictEqual(etaLine(page), '');
    assert.strictEqual(etaTimers(page), 0);
    // An estimate, then an update that has none (no fields at all).
    estimate(page, 12, 8, 20);
    assert.notStrictEqual(etaLine(page), '');
    update(page, {content: 'Waiting again.'});
    assert.strictEqual(etaLine(page), '');
    assert.strictEqual(etaTimers(page), 0);
    // An estimate, then the daemon has no update any more.
    estimate(page, 12, 8, 20);
    assert.strictEqual(etaTimers(page), 1);
    update(page, {exists: false, sig: ''});
    assert.strictEqual(etaLine(page), '');
    assert.strictEqual(etaTimers(page), 0);
  });

  await test('the countdown ends with the task; the next task starts without the old estimate', () => {
    const page = runningPage();
    estimate(page, 12, 8, 20);
    assert.strictEqual(etaTimers(page), 1);
    send(page.win, {type: 'status', running: false});
    assert.strictEqual(etaLine(page), '');
    assert.strictEqual(etaTimers(page), 0);
    advance(page, 30000);
    assert.strictEqual(etaLine(page), '', 'nothing counts on');
    // The next task has no estimate until its own update brings one.
    send(page.win, {type: 'status', running: true});
    assert.strictEqual(etaLine(page), '');
    assert.strictEqual(etaTimers(page), 0);
    estimate(page, 3, 2, 5);
    assert.strictEqual(
      etaLine(page),
      'About 3 min left (2 to 5 min), done around ' +
        clockLabel(page, page.now + 3 * MIN),
    );
  });

  await test('a tab switch drops the estimate of the tab left behind', () => {
    const page = runningPage();
    const {win} = page;
    const first = readyTabId(page.posted);
    estimate(page, 12, 8, 20);
    // A second, idle tab takes the screen.
    click(win, win.document.querySelector('#new-chat-btn'));
    assert.notStrictEqual(win._testApi.getActiveTabId(), first);
    assert.strictEqual(etaLine(page), '');
    assert.strictEqual(etaTimers(page), 0);
    // Back on the running tab the line waits for that tab's update.
    win._testApi.switchToTab(first);
    assert.strictEqual(etaLine(page), '');
    assert.strictEqual(lastPoll(page).tabId, first);
    estimate(page, 9, 6, 15);
    assert.strictEqual(
      etaLine(page),
      'About 9 min left (6 to 15 min), done around ' +
        clockLabel(page, page.now + 9 * MIN),
    );
    assert.strictEqual(etaTimers(page), 1);
  });

  await test('the phone drawer counts down too, and a task that ends behind a file view stops the countdown', () => {
    const page = runningPage({desktop: false});
    const {win} = page;
    const chat = readyTabId(page.posted);
    estimate(page, 12, 8, 20);
    const done = clockLabel(page, NOON + 12 * MIN);
    assert.strictEqual(
      etaLine(page),
      `About 12 min left (8 to 20 min), done around ${done}`,
    );
    // A file view takes the screen: the panel stays the chat's.
    send(win, {
      type: 'fileContent',
      name: 'report.html',
      path: '/cfg/dir/reports/report.html',
      content: '<h1>report</h1>',
    });
    assert.notStrictEqual(win._testApi.getActiveTabId(), chat);
    advance(page, 45000);
    assert.strictEqual(
      etaLine(page),
      `About 11 min left (7 to 19 min), done around ${done}`,
    );
    send(win, {type: 'task_done', tabId: chat});
    assert.strictEqual(etaLine(page), '');
    assert.strictEqual(etaTimers(page), 0);
  });

  report('taskProgressEta');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
