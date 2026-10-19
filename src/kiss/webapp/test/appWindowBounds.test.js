// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

/* global require, __dirname, console */

'use strict';

// media/appWindow.js keeps the bounds of the macOS standalone window
// (the popup media/appWindow.html opens, named "kiss-app-window") in
// localStorage for the next launch.  Driven here in jsdom windows:
//
//   * in the popup the bounds (screen position, outer size) are saved
//     at once, on resize, on pagehide, and by a once-a-second poll
//     after a move (the poll is captured and fired by hand);
//   * unchanged bounds are not written again, and a window with no
//     size (hidden) is not recorded;
//   * storage that throws is tolerated;
//   * any other window (no name: a tab, the Linux app window) is left
//     alone: nothing saved, nothing listened to.
//
// The companion page's own behaviour (opening the popup, closing
// itself) needs a real browser and is covered by
// test_standalone_app.py's macOS test.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const SOURCE = fs.readFileSync(
  path.join(__dirname, '..', 'media', 'appWindow.js'),
  'utf8',
);
const KEY = 'kissAppWindowBounds';

/**
 * A jsdom window named `name` with `bounds` and `storage` as its
 * localStorage, running media/appWindow.js.  With `polls`, every
 * `setInterval` the script arms is recorded there as `{fn, ms}` instead
 * of running, so a test fires the bounds poll itself.
 */
function makeWindow(name, bounds, storage, polls) {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    url: 'http://localhost/',
    runScripts: 'outside-only',
    pretendToBeVisual: true,
  });
  const win = dom.window;
  win.name = name;
  setBounds(win, bounds);
  if (storage) {
    Object.defineProperty(win, 'localStorage', {value: storage});
  }
  if (polls) {
    win.setInterval = function (fn, ms) {
      polls.push({fn, ms});
      return polls.length;
    };
  }
  win.eval(SOURCE);
  return win;
}

function setBounds(win, bounds) {
  win.screenX = bounds[0];
  win.screenY = bounds[1];
  win.outerWidth = bounds[2];
  win.outerHeight = bounds[3];
}

function saved(win) {
  return JSON.parse(win.localStorage.getItem(KEY));
}

function countingStorage() {
  const store = {writes: 0, values: {}};
  store.getItem = key => (key in store.values ? store.values[key] : null);
  store.setItem = (key, value) => {
    store.writes += 1;
    store.values[key] = value;
  };
  return store;
}

function testPopupBoundsAreKept() {
  const storage = countingStorage();
  const polls = [];
  const win = makeWindow(
    'kiss-app-window',
    [100, 50, 800, 600],
    storage,
    polls,
  );
  assert.deepStrictEqual(JSON.parse(storage.values[KEY]), {
    left: 100,
    top: 50,
    width: 800,
    height: 600,
  });
  assert.strictEqual(storage.writes, 1, 'saved once at load');

  win.dispatchEvent(new win.Event('resize'));
  assert.strictEqual(storage.writes, 1, 'unchanged bounds are not rewritten');

  setBounds(win, [100, 50, 640, 480]);
  win.dispatchEvent(new win.Event('resize'));
  assert.deepStrictEqual(JSON.parse(storage.values[KEY]), {
    left: 100,
    top: 50,
    width: 640,
    height: 480,
  });
  assert.strictEqual(storage.writes, 2, 'a resize is saved');

  setBounds(win, [10, 20, 640, 480]);
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(JSON.parse(storage.values[KEY]).left, 10);
  assert.strictEqual(
    storage.writes,
    3,
    'the window closing saves its last bounds',
  );

  // A move raises no event: the once-a-second look catches it.
  assert.deepStrictEqual(
    polls.map(p => p.ms),
    [1000],
    'one poll is armed, once a second',
  );
  setBounds(win, [300, 200, 640, 480]);
  polls[0].fn();
  assert.deepStrictEqual(JSON.parse(storage.values[KEY]), {
    left: 300,
    top: 200,
    width: 640,
    height: 480,
  });
  assert.strictEqual(storage.writes, 4, 'a move is saved by the poll');

  // A window without a size (hidden) is not recorded.
  setBounds(win, [0, 0, 0, 0]);
  win.dispatchEvent(new win.Event('resize'));
  assert.strictEqual(storage.writes, 4, 'zero bounds are ignored');
  win.close();
}

function testStorageThatThrowsIsTolerated() {
  const storage = {
    getItem: () => null,
    setItem: () => {
      throw new Error('QuotaExceededError');
    },
  };
  const win = makeWindow('kiss-app-window', [1, 2, 300, 400], storage);
  setBounds(win, [1, 2, 301, 400]);
  win.dispatchEvent(new win.Event('resize'));
  win.close();
}

function testOtherWindowsAreLeftAlone() {
  for (const name of ['', 'something-else']) {
    const storage = countingStorage();
    const win = makeWindow(name, [100, 50, 800, 600], storage);
    setBounds(win, [100, 50, 640, 480]);
    win.dispatchEvent(new win.Event('resize'));
    win.dispatchEvent(new win.Event('pagehide'));
    assert.strictEqual(
      storage.writes,
      0,
      `nothing saved for window name ${name}`,
    );
    win.close();
  }
}

function testRealLocalStorage() {
  const win = makeWindow('kiss-app-window', [5, 6, 700, 500]);
  assert.deepStrictEqual(saved(win), {
    left: 5,
    top: 6,
    width: 700,
    height: 500,
  });
  win.close();
}

testPopupBoundsAreKept();
testStorageThatThrowsIsTolerated();
testOtherWindowsAreLeftAlone();
testRealLocalStorage();
console.log('appWindowBounds.test.js passed');
