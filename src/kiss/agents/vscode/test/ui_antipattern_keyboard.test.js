// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-pattern fixes A5/A7/A15 in
// media/main.js: the custom tooltip opens on keyboard focus (and closes
// on blur / Escape) with role="tooltip"; toolbar buttons keep focus after
// a keyboard-originated click; collapsible transcript panel headers are
// focusable buttons toggled by Enter / Space with aria-expanded kept in
// sync (the thinking text inside a Thoughts panel has no header of its own).
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();
const TS = 1767225600000;

function startRun(win) {
  win._testApi.endLaunch();
  win._testApi.hideWelcome();
  const tab = win._testApi.getActiveTabId();
  h.send(win, {type: 'status', running: true, tabId: tab, startTs: TS});
  return tab;
}

function stream(win, tab, events) {
  for (const ev of events) h.send(win, Object.assign({tabId: tab, ts: TS}, ev));
}

/** Top-level collapsible panels of the visible transcript, in order. */
function panels(win) {
  const out = win.document.getElementById('output');
  return Array.from(out.children).filter(
    el => el.classList.contains('collapsible') && !el.classList.contains('rc'),
  );
}

function focusEvent(win, el, type) {
  el.dispatchEvent(new win.FocusEvent(type, {bubbles: true}));
}

async function main() {
  await test('the tooltip has role="tooltip" and opens at once on keyboard focus', async () => {
    const {win} = h.makeWebview();
    const tip = h.byId(win, 'custom-tooltip');
    assert.ok(tip, 'the tooltip element exists');
    assert.strictEqual(tip.getAttribute('role'), 'tooltip');
    const btn = h.byId(win, 'new-chat-btn');
    assert.strictEqual(btn.dataset.tooltip, 'New chat');
    btn.focus();
    focusEvent(win, btn, 'focusin');
    assert.ok(
      tip.classList.contains('visible'),
      'focus shows it without delay',
    );
    assert.strictEqual(tip.textContent, 'New chat');
    focusEvent(win, btn, 'focusout');
    assert.ok(!tip.classList.contains('visible'), 'blur hides it');
    win.close();
  });

  await test('Escape closes an open tooltip; hover keeps its 400 ms delay', async () => {
    const {win} = h.makeWebview();
    const tip = h.byId(win, 'custom-tooltip');
    const btn = h.byId(win, 'menu-btn');
    focusEvent(win, btn, 'focusin');
    assert.ok(tip.classList.contains('visible'));
    h.key(win, btn, 'Escape');
    assert.ok(!tip.classList.contains('visible'), 'Escape hides it');

    btn.dispatchEvent(new win.MouseEvent('mouseover', {bubbles: true}));
    assert.ok(
      !tip.classList.contains('visible'),
      'hover does not show at once',
    );
    await h.sleep(450);
    assert.ok(tip.classList.contains('visible'), 'hover shows after the delay');
    btn.dispatchEvent(new win.MouseEvent('mouseout', {bubbles: true}));
    assert.ok(!tip.classList.contains('visible'));
    win.close();
  });

  await test('a keyboard click on a toolbar button keeps focus; a mouse click drops it', () => {
    const {win} = h.makeWebview();
    // A toggle whose own click handler leaves focus alone (the New
    // chat button, by contrast, deliberately focuses the task input).
    const btn = h.byId(win, 'theme-btn');
    btn.focus();
    assert.strictEqual(win.document.activeElement, btn);
    h.keyboardClick(win, btn);
    assert.strictEqual(
      win.document.activeElement,
      btn,
      'Enter / Space (detail 0) must not blur the button',
    );
    h.click(win, btn);
    assert.notStrictEqual(
      win.document.activeElement,
      btn,
      'a pointer click (detail 1) still drops the focus ring',
    );
    win.close();
  });

  await test('a collapsible panel header is a focusable button with aria-expanded', () => {
    const {win} = h.makeWebview();
    const tab = startRun(win);
    stream(win, tab, [
      {type: 'tool_call', name: 'Read', path: 'src/one.py'},
      {type: 'tool_result', content: 'one'},
    ]);
    const ps = panels(win);
    assert.ok(ps.length >= 1, 'a tool panel rendered');
    const panel = ps[0];
    const header = panel.querySelector('.collapse-header');
    assert.ok(header, 'the panel has a collapse header');
    assert.strictEqual(header.tabIndex, 0, 'header is in the tab order');
    assert.strictEqual(header.getAttribute('role'), 'button');
    assert.strictEqual(
      header.getAttribute('aria-expanded'),
      panel.classList.contains('collapsed') ? 'false' : 'true',
      'aria-expanded matches the initial state',
    );
    const before = panel.classList.contains('collapsed');
    header.focus();
    h.key(win, header, 'Enter');
    assert.strictEqual(
      panel.classList.contains('collapsed'),
      !before,
      'Enter toggles',
    );
    assert.strictEqual(
      header.getAttribute('aria-expanded'),
      before ? 'true' : 'false',
    );
    h.key(win, header, ' ');
    assert.strictEqual(
      panel.classList.contains('collapsed'),
      before,
      'Space toggles back',
    );
    assert.strictEqual(
      header.getAttribute('aria-expanded'),
      before ? 'false' : 'true',
    );
    // Only the header itself answers Enter / Space: a key pressed on a
    // focusable child (a button placed in the header) must be its own.
    assert.ok(
      !header.querySelector('.collapse-chv'),
      'the header carries no chevron',
    );
    const inner = header.querySelector('.collapse-preview');
    assert.ok(inner, 'the header carries the collapse preview');
    h.key(win, inner, 'Enter');
    assert.strictEqual(
      panel.classList.contains('collapsed'),
      before,
      'child key ignored',
    );
    win.close();
  });

  await test('auto-folded panels keep aria-expanded in sync', () => {
    const {win} = h.makeWebview();
    const tab = startRun(win);
    stream(win, tab, [
      {type: 'tool_call', name: 'Read', path: 'src/one.py'},
      {type: 'tool_result', content: 'one'},
      {type: 'tool_call', name: 'Read', path: 'src/two.py'},
      {type: 'tool_result', content: 'two'},
      {type: 'tool_call', name: 'Write', path: 'src/three.py'},
      {type: 'tool_result', content: 'three'},
    ]);
    const ps = panels(win);
    assert.ok(
      ps.some(p => p.classList.contains('collapsed')),
      'some folded',
    );
    for (const p of ps) {
      const header = p.querySelector('.collapse-header');
      assert.strictEqual(
        header.getAttribute('aria-expanded'),
        p.classList.contains('collapsed') ? 'false' : 'true',
        'every header reports its panel state',
      );
    }
    win.close();
  });

  await test('the thinking text has no header of its own to operate', () => {
    // Thinking tokens are plain text inside the Thoughts panel; the
    // panel's own header (tested above) is the one disclosure control.
    const {win} = h.makeWebview();
    const tab = startRun(win);
    stream(win, tab, [
      {type: 'thinking_start'},
      {type: 'thinking_delta', text: 'pondering'},
      {type: 'thinking_end'},
    ]);
    const think = win.document.querySelector('.llm-panel > .think');
    assert.ok(think, 'the thinking text block rendered inside the Thoughts panel');
    assert.strictEqual(think.textContent, 'pondering');
    assert.ok(!think.querySelector('[role="button"]'), 'no button inside the thinking text');
    assert.ok(!win.document.querySelector('.think .lbl'), 'no "Thinking" header');
    const hdr = think.parentElement.querySelector(':scope > .llm-panel-hdr');
    assert.strictEqual(hdr.getAttribute('role'), 'button', "the Thoughts header is the control");
    win.close();
  });

  report('ui_antipattern_keyboard');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
