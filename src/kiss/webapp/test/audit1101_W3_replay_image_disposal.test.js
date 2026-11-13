// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) regressions for the 2026-11-01 audit of
// media/main.js (W3, lines 15200-22800): the lazy pictures of a
// replayed transcript (`tr-img-lazy`, fetched with getReplayImage once
// they scroll into view) are dropped with the transcript they belong
// to.
//
// * a transcript cleared or replaced, or a tab closed, leaves none of
//   its pictures registered with the page-lifetime IntersectionObserver
//   (which would otherwise keep every discarded transcript alive),
// * a sighting the observer queued before the discard asks the daemon
//   for nothing, and a late reply changes nothing,
// * a hidden tab's transcript keeps its pictures: an ordinary tab
//   switch disposes nothing, and the pictures load once shown,
// * a reconnect re-watches only the pictures still awaited,
// * a replacement that fails half way hands the old transcript its
//   pictures back (watched again, or asked for afresh without the
//   observer).

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  stubRaf,
  ofType,
  readyTabId,
  settle,
} = require('./simplify2_harness');
const {
  bootParallelRun,
  clickClose,
  announceRetaggedSubagent,
} = require('./subagent_tab_helpers');
const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();
const pages = [];

/** A page whose IntersectionObserver records its registrations. */
function pageWithObserver() {
  const observers = [];
  const page = makePage({
    beforeScripts(win) {
      stubRaf(win);
      win.IntersectionObserver = class {
        constructor(cb, options) {
          this.callback = cb;
          this.options = options;
          this.targets = new Set();
          this.unobserved = [];
          observers.push(this);
        }
        observe(el) {
          this.targets.add(el);
        }
        unobserve(el) {
          this.targets.delete(el);
          this.unobserved.push(el);
        }
        disconnect() {
          this.targets.clear();
        }
      };
    },
  });
  pages.push(page.win);
  const observer = observers.find(o => o.options.rootMargin === '400px');
  assert.ok(observer, 'the replay image observer is created');
  return Object.assign(page, {observer});
}

/** The two events of a screenshot whose picture is lazy picture *index* of *task*. */
function screenshotEvents(task, path, index) {
  return [
    {type: 'tool_call', name: 'screenshot'},
    {
      type: 'tool_result',
      tool_name: 'screenshot',
      content: 'image',
      images: [{mime: 'image/png', path, lazy: {taskId: task, index}}],
    },
  ];
}

function replay(win, tab, task) {
  send(win, {
    type: 'task_events',
    tabId: tab,
    task_id: task,
    task: 'images',
    events: screenshotEvents(task, 'shot.png', 0),
  });
}

/**
 * Replay a corrupted transcript of `parent-task` into *tab*: a null
 * event after picture 1 makes the replay throw half way (the page logs
 * the TypeError).
 */
function replayCorrupted(win, tab) {
  send(win, {
    type: 'task_events',
    tabId: tab,
    task_id: 'parent-task',
    task: 'images',
    events: [...screenshotEvents('parent-task', 'x.png', 1), null],
  });
}

function lazyImages(observer, owner) {
  return Array.from(observer.targets).filter(
    img => img.dataset.replayOwner === owner,
  );
}

async function main() {
  await test('a cleared transcript leaves no picture with the observer', () => {
    const page = pageWithObserver();
    const {win, observer} = page;
    const {parentId} = bootParallelRun(1, page);
    for (let i = 0; i < 20; i++) {
      replay(win, parentId, 'parent-task');
      assert.strictEqual(
        lazyImages(observer, parentId).length,
        1,
        'the replay registers its picture',
      );
      send(win, {type: 'clear', tabId: parentId});
    }
    assert.strictEqual(
      observer.targets.size,
      0,
      'BUG: ' +
        observer.targets.size +
        ' discarded pictures are still observed',
    );
    assert.strictEqual(
      observer.unobserved.length,
      20,
      'each clear unobserves its picture',
    );
  });

  await test("a closed sub-agent tab's picture is neither observed, asked for nor changed", () => {
    const page = pageWithObserver();
    const {win, posted, observer} = page;
    const {subTabIds} = bootParallelRun(2, page);
    replay(win, subTabIds[0], 'sub-task-1');
    const [img] = lazyImages(observer, subTabIds[0]);
    assert.ok(img, "the child's picture is observed while its tab is open");
    // The second child asks for its picture while its tab is open.
    replay(win, subTabIds[1], 'sub-task-2');
    const [awaited] = lazyImages(observer, subTabIds[1]);
    observer.callback([{target: awaited, isIntersecting: true}]);
    const req = ofType(posted, 'getReplayImage').pop();
    assert.strictEqual(req.tabId, subTabIds[1], 'the open child asks');
    for (const id of subTabIds) send(win, {type: 'subagentDone', tab_id: id});
    assert.ok(
      !win._testApi.openTabs().some(t => subTabIds.includes(t.id)),
      'the child tabs closed',
    );
    assert.ok(
      !observer.targets.has(img),
      "BUG: the closed tab's picture is still observed",
    );
    // A sighting queued before the close delivers now.
    const before = posted.length;
    observer.callback([{target: img, isIntersecting: true}]);
    const asked = ofType(posted.slice(before), 'getReplayImage');
    assert.strictEqual(
      asked.length,
      0,
      'BUG: a closed tab asked for its picture',
    );
    assert.strictEqual(img.getAttribute('src'), null);
    // A reply to a request made before the close changes nothing either.
    send(win, {
      type: 'replayImage',
      token: req.token,
      mime: 'image/png',
      b64: 'AAAA',
    });
    assert.strictEqual(
      awaited.getAttribute('src'),
      null,
      'BUG: a closed tab took its late reply',
    );
    assert.ok(awaited.classList.contains('tr-img-lazy'));
  });

  await test('a picture asked for before its transcript was cleared ignores the late reply', () => {
    const page = pageWithObserver();
    const {win, posted, observer} = page;
    const {parentId} = bootParallelRun(1, page);
    replay(win, parentId, 'parent-task');
    const [img] = lazyImages(observer, parentId);
    observer.callback([{target: img, isIntersecting: true}]);
    const req = ofType(posted, 'getReplayImage').pop();
    assert.ok(req, 'the picture on screen is asked for');
    assert.strictEqual(req.tabId, parentId);
    send(win, {type: 'clear', tabId: parentId});
    // The connection drops: only pictures still awaited are re-watched.
    send(win, {type: 'daemonStatus', connected: false});
    assert.ok(
      !observer.targets.has(img),
      'BUG: the reconnect re-watches a discarded picture',
    );
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {
      type: 'replayImage',
      token: req.token,
      mime: 'image/png',
      b64: 'AAAA',
    });
    assert.strictEqual(
      img.getAttribute('src'),
      null,
      'BUG: the late reply filled a discarded picture',
    );
  });

  await test("a hidden tab's transcript keeps its pictures until shown", () => {
    const page = pageWithObserver();
    const {win, posted, observer} = page;
    const {parentId} = bootParallelRun(1, page);
    const api = win._testApi;
    api.createNewTab();
    const other = api.getActiveTabId();
    assert.notStrictEqual(other, parentId);
    // The hidden parent's transcript is replaced (its fragment is
    // detached DOM): its picture is registered all the same.
    replay(win, parentId, 'parent-task');
    let imgs = lazyImages(observer, parentId);
    assert.strictEqual(imgs.length, 1, 'the hidden tab registers its picture');
    assert.ok(!imgs[0].isConnected, 'in a detached fragment');
    // A second replay of the hidden tab discards the first's picture.
    replay(win, parentId, 'parent-task');
    imgs = lazyImages(observer, parentId);
    assert.strictEqual(
      imgs.length,
      1,
      "BUG: the replaced transcript's picture is still observed",
    );
    // Switching to the tab (and back) disposes nothing.
    api.switchToTab(parentId);
    api.switchToTab(other);
    api.switchToTab(parentId);
    assert.strictEqual(
      lazyImages(observer, parentId).length,
      1,
      'a tab switch keeps the picture',
    );
    const [img] = lazyImages(observer, parentId);
    assert.ok(img.isConnected, 'on screen again');
    observer.callback([{target: img, isIntersecting: true}]);
    const req = ofType(posted, 'getReplayImage').pop();
    assert.ok(req && req.tabId === parentId, 'the shown picture is asked for');
    send(win, {
      type: 'replayImage',
      token: req.token,
      mime: 'image/png',
      b64: 'AAAA',
    });
    assert.strictEqual(img.getAttribute('src'), 'data:image/png;base64,AAAA');
    assert.ok(!img.classList.contains('tr-img-lazy'));
    // Clearing the transcript now has nothing left to drop; the other
    // tab's transcript is untouched.
    send(win, {type: 'clear', tabId: parentId});
    assert.strictEqual(observer.targets.size, 0);
  });

  await test("a retagged sub-agent tab's pictures are dropped under its new id", () => {
    const page = pageWithObserver();
    const {win, posted, observer} = page;
    const parentId = readyTabId(posted);
    win._testApi.hideWelcome();
    send(win, {type: 'status', running: true, tabId: parentId, startTs: 1});
    announceRetaggedSubagent(win, parentId, 'sa-1');
    send(win, {type: 'status', running: true, tabId: 'sa-1', startTs: 1});
    replay(win, 'sa-1', 't1');
    assert.strictEqual(lazyImages(observer, 'sa-1').length, 1);
    // The daemon re-announces the sub-agent under a new id: the tab is
    // renamed and its picture follows.
    announceRetaggedSubagent(win, parentId, 'sa-2');
    assert.ok(posted.some(m => m.type === 'closeTab' && m.tabId === 'sa-1'));
    assert.strictEqual(
      lazyImages(observer, 'sa-1').length,
      0,
      'no picture under the retired id',
    );
    assert.strictEqual(
      lazyImages(observer, 'sa-2').length,
      1,
      'BUG: the picture kept the old owner',
    );
    clickClose(win, 'sa-2');
    assert.strictEqual(
      observer.targets.size,
      0,
      "BUG: the closed tab's picture is still observed",
    );
  });

  await test("a hidden tab's failed replay hands its old pictures back", () => {
    const page = pageWithObserver();
    const {win, observer} = page;
    const {parentId} = bootParallelRun(1, page);
    const api = win._testApi;
    api.createNewTab();
    replay(win, parentId, 'parent-task');
    const [kept] = lazyImages(observer, parentId);
    assert.ok(kept, 'the hidden tab has its picture');
    // The corrupted replay throws half way: the tab gets its old
    // transcript back, with its picture watched again, and the failed
    // replay's picture (index 1) is dropped.
    replayCorrupted(win, parentId);
    const imgs = lazyImages(observer, parentId);
    assert.strictEqual(imgs.length, 1, 'one picture is watched for the tab');
    assert.strictEqual(
      imgs[0],
      kept,
      "the old transcript's picture, not the failed replay's",
    );
    api.switchToTab(parentId);
    assert.ok(kept.isConnected, 'the old transcript is back on screen');
  });

  await test("without the observer, a hidden tab's failed replay asks for its old pictures again", () => {
    // A page without IntersectionObserver (plain jsdom, an old
    // browser) asks for each picture as soon as it is rendered.
    const page = makePage({beforeScripts: stubRaf});
    const {win, posted} = page;
    pages.push(win);
    assert.strictEqual(typeof win.IntersectionObserver, 'undefined');
    const {parentId} = bootParallelRun(1, page);
    const api = win._testApi;
    api.createNewTab();
    replay(win, parentId, 'parent-task');
    const first = ofType(posted, 'getReplayImage').pop();
    assert.ok(first, 'the picture is asked for at once');
    assert.strictEqual(first.index, 0);
    const kept = win.document.querySelector(
      'img[data-replay-owner="' + parentId + '"]',
    );
    assert.strictEqual(kept, null, 'the hidden transcript is detached');
    const before = posted.length;
    replayCorrupted(win, parentId);
    // The failed replay asked for its own picture (index 1) before it
    // threw; the restored transcript's picture is asked for afresh.
    const asked = ofType(posted.slice(before), 'getReplayImage');
    assert.deepStrictEqual(
      asked.map(m => m.index),
      [1, 0],
      'BUG: the restored picture is not asked for again',
    );
    assert.notStrictEqual(asked[1].token, first.token, 'with a new token');
    // The answers to the dropped requests change nothing; the answer
    // to the fresh one shows the picture.
    for (const token of [first.token, asked[0].token])
      send(win, {type: 'replayImage', token, mime: 'image/png', b64: 'AAAA'});
    api.switchToTab(parentId);
    const img = win.document.querySelector(
      'img[data-replay-owner="' + parentId + '"]',
    );
    assert.ok(img, 'the old transcript is back on screen');
    assert.strictEqual(img.dataset.replayIndex, '0');
    assert.strictEqual(
      img.getAttribute('src'),
      null,
      'BUG: a dropped reply shown',
    );
    send(win, {
      type: 'replayImage',
      token: asked[1].token,
      mime: 'image/png',
      b64: 'BBBB',
    });
    assert.strictEqual(img.getAttribute('src'), 'data:image/png;base64,BBBB');
    assert.ok(!img.classList.contains('tr-img-lazy'));
  });

  await test('without the observer, a reconnect asks afresh for the pictures still awaited', () => {
    // The dropped connection never answers the request in flight; with
    // no observer to re-watch the picture, it is asked for again at
    // once (as a restored transcript's is), not left blank for good.
    const page = makePage({beforeScripts: stubRaf});
    const {win, posted} = page;
    pages.push(win);
    assert.strictEqual(typeof win.IntersectionObserver, 'undefined');
    const {parentId} = bootParallelRun(1, page);
    replay(win, parentId, 'parent-task');
    const first = ofType(posted, 'getReplayImage').pop();
    assert.ok(first, 'the picture is asked for at once');
    const before = posted.length;
    send(win, {type: 'daemonStatus', connected: false});
    send(win, {type: 'daemonStatus', connected: true});
    const again = ofType(posted.slice(before), 'getReplayImage');
    assert.strictEqual(
      again.length,
      1,
      'BUG: the awaited picture is not asked for again after the outage',
    );
    assert.strictEqual(again[0].index, 0);
    assert.notStrictEqual(again[0].token, first.token, 'with a new token');
    const img = win.document.querySelector(
      'img[data-replay-owner="' + parentId + '"]',
    );
    assert.ok(img, 'the picture is on screen');
    send(win, {
      type: 'replayImage',
      token: first.token,
      mime: 'image/png',
      b64: 'AAAA',
    });
    assert.strictEqual(img.getAttribute('src'), null, 'a dropped reply shown');
    send(win, {
      type: 'replayImage',
      token: again[0].token,
      mime: 'image/png',
      b64: 'CCCC',
    });
    assert.strictEqual(img.getAttribute('src'), 'data:image/png;base64,CCCC');
    assert.ok(!img.classList.contains('tr-img-lazy'));
  });

  report('audit1101_W3_replay_image_disposal');
  await settle();
  for (const win of pages) win.close();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
