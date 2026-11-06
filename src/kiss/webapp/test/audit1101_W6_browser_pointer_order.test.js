// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E regression for audit 2026-11-01 W6 F2: browserTab.js coalesces
// pointer moves to one per animation frame, but pointerdown / pointerup
// / wheel were sent at once without flushing the move still waiting for
// its frame.  A drag whose release came before the next frame reached
// the daemon (which forwards the order unchanged over CDP) as press,
// release, then a move with the button still down: the remote page saw
// no drag and an impossible buttons-down move after mouseup.  Likewise
// a hover move queued before a press arrived after it.
//
// Animation frames are queued by hand, so the test chooses the moment
// the frame runs.  The ordering assertions failed against the pre-fix
// browserTab.js; the coalescing assertion guards what must stay.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

const MEDIA = path.join(__dirname, '..', 'media');

function makeView() {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
  });
  const win = dom.window;
  const frames = [];
  let nextFrame = 0;
  win.requestAnimationFrame = cb => {
    const id = ++nextFrame;
    frames.push({id, cb});
    return id;
  };
  win.cancelAnimationFrame = id => {
    const i = frames.findIndex(f => f.id === id);
    if (i >= 0) frames.splice(i, 1);
  };
  /** Run the frames queued so far. */
  function runFrames() {
    for (const f of frames.splice(0)) f.cb();
  }
  win.eval(fs.readFileSync(path.join(MEDIA, 'browserTab.js'), 'utf8'));
  const sent = [];
  const view = win.BrowserTabView.create('tab-1', {
    send: msg => sent.push(JSON.parse(JSON.stringify(msg))),
  });
  win.document.body.appendChild(view.el);
  const screen = view.el.querySelector('.browser-screen');
  return {win, view, sent, screen, frames, runFrames};
}

function pointer(win, screen, type, x, y, buttons) {
  screen.dispatchEvent(
    new win.PointerEvent(type, {
      clientX: x,
      clientY: y,
      button: 0,
      buttons,
      pointerType: 'mouse',
      bubbles: true,
      cancelable: true,
    }),
  );
}

/** The mouse events sent so far as "action@x:buttons". */
function mouseTrace(sent) {
  return sent
    .filter(m => m.type === 'browserInput' && m.event.kind === 'mouse')
    .map(m => m.event.action + '@' + m.event.x + ':' + m.event.buttons);
}

async function main() {
  await test(
    'W6 F2: a move still waiting for its frame goes out before the ' +
      'release, so the daemon sees press, drag, release in user order',
    async () => {
      const {win, sent, screen, frames, runFrames} = makeView();
      pointer(win, screen, 'pointerdown', 10, 10, 1);
      pointer(win, screen, 'pointermove', 100, 10, 1);
      pointer(win, screen, 'pointerup', 120, 10, 0);
      assert.deepStrictEqual(
        mouseTrace(sent),
        ['mousePressed@10:1', 'mouseMoved@100:1', 'mouseReleased@120:0'],
        'the queued move must be flushed before the release',
      );
      assert.strictEqual(frames.length, 0, 'the flushed frame is cancelled');
      runFrames();
      assert.strictEqual(
        mouseTrace(sent).length,
        3,
        'no buttons-down move may follow the release: ' +
          JSON.stringify(mouseTrace(sent)),
      );
    },
  );

  await test(
    'W6 F2: a hover move queued before a press, and before a wheel, ' +
      'goes out first',
    async () => {
      const {win, sent, screen, runFrames} = makeView();
      pointer(win, screen, 'pointermove', 50, 50, 0);
      pointer(win, screen, 'pointerdown', 50, 50, 1);
      assert.deepStrictEqual(mouseTrace(sent), [
        'mouseMoved@50:0',
        'mousePressed@50:1',
      ]);
      pointer(win, screen, 'pointerup', 50, 50, 0);
      pointer(win, screen, 'pointermove', 60, 60, 0);
      screen.dispatchEvent(
        new win.WheelEvent('wheel', {
          clientX: 60,
          clientY: 60,
          deltaX: 0,
          deltaY: 120,
          deltaMode: 0,
          bubbles: true,
          cancelable: true,
        }),
      );
      assert.deepStrictEqual(mouseTrace(sent).slice(2), [
        'mouseReleased@50:0',
        'mouseMoved@60:0',
        'mouseWheel@60:0',
      ]);
      runFrames();
      assert.strictEqual(mouseTrace(sent).length, 5, 'nothing left queued');
    },
  );

  await test(
    'W6 F2 guard: between transitions moves still coalesce to one per ' +
      'frame, the last one wins, and a disposed view sends none',
    async () => {
      const {win, view, sent, screen, frames, runFrames} = makeView();
      pointer(win, screen, 'pointermove', 1, 1, 0);
      pointer(win, screen, 'pointermove', 2, 2, 0);
      pointer(win, screen, 'pointermove', 3, 3, 0);
      assert.deepStrictEqual(mouseTrace(sent), [], 'moves wait for the frame');
      assert.strictEqual(frames.length, 1, 'one frame for the burst');
      runFrames();
      assert.deepStrictEqual(mouseTrace(sent), ['mouseMoved@3:0']);
      pointer(win, screen, 'pointermove', 4, 4, 0);
      view.dispose();
      runFrames();
      pointer(win, screen, 'pointerdown', 5, 5, 1);
      assert.deepStrictEqual(
        mouseTrace(sent).filter(t => t.startsWith('mouseMoved')),
        ['mouseMoved@3:0'],
        'a move queued when the view was disposed is dropped',
      );
    },
  );

  // A pointer the browser cancels after the press (the OS took the
  // device, a native drag began) gets no pointerup: the remote page
  // would be left with the button down until the next click.  The
  // release goes out on pointercancel, after the move that preceded
  // it, once; a lost capture after it, or a cancel with nothing
  // pressed, sends nothing more.
  await test(
    'a cancelled pointer releases the remote button it pressed, once',
    async () => {
      const {win, sent, screen, runFrames} = makeView();
      pointer(win, screen, 'pointerdown', 10, 10, 1);
      pointer(win, screen, 'pointermove', 30, 10, 1);
      screen.dispatchEvent(
        new win.PointerEvent('pointercancel', {
          button: -1,
          buttons: 0,
          pointerType: 'mouse',
          bubbles: true,
        }),
      );
      screen.dispatchEvent(
        new win.PointerEvent('lostpointercapture', {
          pointerType: 'mouse',
          bubbles: true,
        }),
      );
      runFrames();
      assert.deepStrictEqual(mouseTrace(sent), [
        'mousePressed@10:1',
        'mouseMoved@30:1',
        'mouseReleased@10:0',
      ]);
      // A cancel with nothing pressed (a touch, or after a pointerup)
      // releases nothing.
      pointer(win, screen, 'pointerdown', 20, 20, 1);
      pointer(win, screen, 'pointerup', 20, 20, 0);
      screen.dispatchEvent(
        new win.PointerEvent('pointercancel', {
          button: -1,
          pointerType: 'mouse',
          bubbles: true,
        }),
      );
      assert.deepStrictEqual(mouseTrace(sent).slice(3), [
        'mousePressed@20:1',
        'mouseReleased@20:0',
      ]);
    },
  );

  report('audit1101_W6_browser_pointer_order');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
