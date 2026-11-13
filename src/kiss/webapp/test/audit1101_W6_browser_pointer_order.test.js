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
const {MEDIA, queueRaf} = require('./simplify2_harness');

const {test, report} = makeRunner();

function makeView() {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
  });
  const win = dom.window;
  const {frames, run: runFrames} = queueRaf(win);
  win.eval(fs.readFileSync(path.join(MEDIA, 'browserTab.js'), 'utf8'));
  const sent = [];
  const view = win.BrowserTabView.create('tab-1', {
    send: msg => sent.push(JSON.parse(JSON.stringify(msg))),
  });
  win.document.body.appendChild(view.el);
  const screen = view.el.querySelector('.browser-screen');
  return {win, view, sent, screen, frames, runFrames};
}

function pointer(win, screen, type, x, y, buttons, button) {
  screen.dispatchEvent(
    new win.PointerEvent(type, {
      clientX: x,
      clientY: y,
      button: button || 0,
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
  // would be left with the button down until the next click.  A
  // release sent instead clicked the control under the press (Chromium
  // fires click on a release at the press point whatever its click
  // count): a cancelled gesture must activate nothing, so what goes
  // out on pointercancel, after the move that preceded it and once, is
  // a move with no buttons down, which ends the press on the remote
  // page (its pressed state, :active) without mouseup or click, as the
  // cancel ended it here; a lost capture after it, or a cancel with
  // nothing pressed, sends nothing more.
  await test(
    'a cancelled pointer ends the remote press without a release, once',
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
        'mouseMoved@30:0',
      ]);
      assert.ok(
        !sent.some(m => m.event && m.event.action === 'mouseReleased'),
        'no release, which would click the control under the press',
      );
      // A cancel with nothing pressed (a touch, or after a pointerup)
      // releases nothing.
      pointer(win, screen, 'pointerdown', 20, 20, 1);
      screen.dispatchEvent(
        new win.PointerEvent('pointercancel', {pointerType: 'touch'}),
      );
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

  // A button the remote page has no name for (a pen's eraser, bit 32)
  // is no press there.  It was counted as one: cancelled before any
  // other pointer event, the move that ends a press was built from a
  // pointer that had never been forwarded and threw.
  await test(
    'a button the remote page has no name for is not a press there',
    async () => {
      const {win, sent, screen, runFrames} = makeView();
      const errors = [];
      win.addEventListener('error', e => {
        errors.push(e.message);
        e.preventDefault();
      });
      screen.dispatchEvent(
        new win.PointerEvent('pointerdown', {
          clientX: 10,
          clientY: 10,
          button: 5,
          buttons: 32,
          pointerType: 'pen',
          bubbles: true,
          cancelable: true,
        }),
      );
      screen.dispatchEvent(
        new win.PointerEvent('pointercancel', {
          pointerType: 'pen',
          bubbles: true,
        }),
      );
      assert.deepStrictEqual(errors, []);
      assert.deepStrictEqual(mouseTrace(sent), [], 'nothing to end');
      // Moves with it down are still coalesced to one per frame.
      pointer(win, screen, 'pointermove', 20, 10, 32);
      pointer(win, screen, 'pointermove', 30, 10, 32);
      runFrames();
      assert.deepStrictEqual(mouseTrace(sent), ['mouseMoved@30:32']);
    },
  );

  // A button let go in another window or browser tab gets no pointerup
  // here, only the screen's blur, and the first move back (no buttons)
  // went out as a release: a click on the control under the press.
  // That move now ends the press as a cancel does.
  await test('a press let go elsewhere ends without a release', async () => {
    const {win, sent, screen, runFrames} = makeView();
    screen.focus();
    screen.blur();
    assert.deepStrictEqual(mouseTrace(sent), [], 'nothing was pressed');
    pointer(win, screen, 'pointerdown', 10, 10, 1);
    pointer(win, screen, 'pointermove', 30, 10, 1);
    screen.blur();
    pointer(win, screen, 'pointermove', 40, 10, 0);
    runFrames();
    assert.deepStrictEqual(mouseTrace(sent), [
      'mousePressed@10:1',
      'mouseMoved@30:1',
      'mouseMoved@30:0',
      'mouseMoved@40:0',
    ]);
  });

  await test('a button still down after a blur keeps its press', async () => {
    const {win, view, sent, screen} = makeView();
    pointer(win, screen, 'pointerdown', 100, 10, 1);
    pointer(win, screen, 'pointermove', 120, 10, 1);
    view.el.querySelector('.browser-url').focus(); // the focus only moved
    pointer(win, screen, 'pointerup', 180, 10, 0);
    assert.deepStrictEqual(mouseTrace(sent.splice(0)), [
      'mousePressed@100:1',
      'mouseMoved@120:1',
      'mouseReleased@180:0',
    ]);
    pointer(win, screen, 'pointerdown', 10, 10, 1);
    screen.blur(); // to another window and back, still holding
    pointer(win, screen, 'pointermove', 40, 10, 1);
    pointer(win, screen, 'pointermove', 45, 10, 0); // let go later: a release
    assert.deepStrictEqual(mouseTrace(sent.splice(0)), [
      'mousePressed@10:1',
      'mouseMoved@40:1',
      'mouseReleased@45:0',
    ]);
  });

  // Pointer Events fire pointerdown for the first button pressed and
  // pointerup for the last released; a button pressed or released while
  // another stays down (a chord) comes as a pointermove whose `buttons`
  // changed.  Forwarded as moves, the remote page got a left press and a
  // right release: each changed button is now its own press / release,
  // in order, before the move.
  await test(
    'a chorded button press / release is forwarded as its own',
    async () => {
      const {win, sent, screen, runFrames} = makeView();
      pointer(win, screen, 'pointerdown', 10, 10, 1);
      pointer(win, screen, 'pointermove', 10, 10, 3, 2); // right pressed too
      pointer(win, screen, 'pointermove', 12, 10, 2, 0); // left released
      pointer(win, screen, 'pointerup', 12, 10, 0, 2); // right released
      runFrames();
      const buttons = sent
        .filter(m => m.type === 'browserInput' && m.event.kind === 'mouse')
        .map(m => [m.event.action, m.event.button, m.event.buttons].join(':'));
      assert.deepStrictEqual(buttons, [
        'mousePressed:left:1',
        'mousePressed:right:3',
        'mouseMoved:none:3',
        'mouseReleased:left:2',
        'mouseMoved:none:2',
        'mouseReleased:right:0',
      ]);
    },
  );

  // A view hidden in the middle of a press (the agent opened another
  // tab) has no box left to map a pointer through, yet its screen
  // keeps the captured pointer: the moves and the release went out at
  // window coordinates, a drag ending elsewhere on the remote page.
  // Hiding ends the press as a cancel does, at the point forwarded
  // last, and the pointer is ignored until it lets go, also when the
  // view is back by then; a move waiting for its frame is dropped.
  await test('a view hidden in the middle of a press ends it', async () => {
    const {win, view, sent, screen, runFrames} = makeView();
    // jsdom lays nothing out: the box follows the view's display.
    screen.getBoundingClientRect = () =>
      view.el.style.display === 'none'
        ? {left: 0, top: 0, width: 0, height: 0}
        : {left: 25, top: 0, width: 800, height: 500};
    function shown(on) {
      view.el.style.display = on ? '' : 'none';
      view.setVisible(on);
    }
    shown(true);
    pointer(win, screen, 'pointerdown', 400, 300, 1);
    pointer(win, screen, 'pointermove', 410, 300, 1); // waits for its frame
    shown(false);
    shown(false); // every tab switch hides the hidden views again
    pointer(win, screen, 'pointermove', 430, 300, 1); // the captured pointer
    shown(true);
    pointer(win, screen, 'pointermove', 440, 300, 1); // back, still down
    pointer(win, screen, 'pointerup', 440, 300, 0);
    pointer(win, screen, 'pointermove', 310, 300, 0); // waits for its frame
    shown(false);
    runFrames();
    // The next press, hidden too; its release is never seen.
    shown(true);
    pointer(win, screen, 'pointerdown', 300, 300, 1);
    shown(false);
    shown(true);
    pointer(win, screen, 'pointermove', 320, 300, 0);
    runFrames();
    assert.deepStrictEqual(mouseTrace(sent), [
      'mousePressed@375:1',
      'mouseMoved@375:0',
      'mousePressed@275:1',
      'mouseMoved@275:0',
      'mouseMoved@295:0',
    ]);
  });

  report('audit1101_W6_browser_pointer_order');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
