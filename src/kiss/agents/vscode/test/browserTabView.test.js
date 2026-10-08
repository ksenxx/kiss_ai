// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for the remote webapp's streamed browser tab view
// (browserTab.js): the address bar follows browserState unless it is
// being edited, Enter / Escape in it, the navigation error bar, the
// first frame replacing the placeholder, viewport reports on show /
// hide / resubscribe, and dispose.  jsdom lays nothing out, so the
// screen's size is stubbed.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeView() {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
  });
  const win = dom.window;
  win.eval(fs.readFileSync(path.join(MEDIA, 'browserTab.js'), 'utf8'));
  const sent = [];
  const view = win.BrowserTabView.create('tab-1', {
    send: msg => sent.push(JSON.parse(JSON.stringify(msg))),
  });
  win.document.body.appendChild(view.el);
  const screen = view.el.querySelector('.browser-screen');
  Object.defineProperty(screen, 'clientWidth', {value: 640});
  Object.defineProperty(screen, 'clientHeight', {value: 480});
  return {win, view, sent, screen};
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function keydown(win, el, key) {
  el.dispatchEvent(new win.KeyboardEvent('keydown', {key, bubbles: true}));
}

async function testAddressBarAndState() {
  const {win, view, sent} = makeView();
  const url = view.el.querySelector('.browser-url');
  const [back, fwd] = view.el.querySelectorAll('.browser-nav-btn');
  view.state({url: 'https://a.test/', canGoBack: false, canGoForward: true});
  assert.strictEqual(url.value, 'https://a.test/');
  assert.strictEqual(back.disabled, true);
  assert.strictEqual(fwd.disabled, false);

  // A state arriving while the user types does not clobber the field;
  // leaving the field without navigating restores the page's URL.
  url.focus();
  url.value = 'b.test';
  view.state({url: 'https://c.test/', canGoBack: true});
  assert.strictEqual(url.value, 'b.test');
  keydown(win, url, 'Escape');
  assert.strictEqual(url.value, 'https://c.test/');
  url.focus();
  url.value = ' d.test ';
  keydown(win, url, 'Enter');
  assert.deepStrictEqual(sent.pop(), {
    type: 'browserNavigate',
    tab_id: 'tab-1',
    action: 'go',
    url: 'd.test',
  });
  url.focus();
  url.value = 'typing';
  url.blur();
  assert.strictEqual(url.value, 'https://c.test/');

  back.click();
  assert.strictEqual(sent.pop().action, 'back');
  fwd.click();
  assert.strictEqual(sent.length, 0, 'a disabled Forward sends nothing');
  view.state({url: 'https://c.test/', canGoForward: true});
  fwd.click();
  assert.strictEqual(sent.pop().action, 'forward');

  // The error bar shows a navigation error until the next state.
  const errorBar = view.el.querySelector('.browser-error');
  view.error('net::ERR_NAME_NOT_RESOLVED');
  assert.strictEqual(errorBar.style.display, '');
  assert.strictEqual(errorBar.textContent, 'net::ERR_NAME_NOT_RESOLVED');
  view.state({url: 'https://c.test/'});
  assert.strictEqual(errorBar.style.display, 'none');
  view.error('');
  assert.strictEqual(errorBar.style.display, 'none');

  // The first frame replaces the placeholder and records the page size.
  assert.ok(view.el.querySelector('.browser-placeholder'));
  view.frame({data: 'AAAA', width: 1200, height: 900});
  assert.strictEqual(view.el.querySelector('.browser-placeholder'), null);
  assert.strictEqual(view.pageWidth, 1200);
  assert.strictEqual(view.pageHeight, 900);
  view.frame({});
  view.frame({data: 'BBBB'});
  assert.strictEqual(view.pageWidth, 1200);

  view.setBadge('chromium', true);
  const badge = view.el.querySelector('.browser-badge');
  assert.strictEqual(badge.textContent, 'chromium');
  assert.ok(badge.title.includes('default browser'));
  view.setBadge('firefox', false);
  assert.ok(!badge.title.includes('default'));
  view.setBadge('firefox', false, 'Chrome is not installed.');
  assert.ok(badge.title.includes('Showing firefox instead.'));
}

async function testViewportAndDispose() {
  const {view, sent} = makeView();
  view.setVisible(true);
  await sleep(10);
  assert.deepStrictEqual(sent, [
    {
      type: 'browserViewport',
      tab_id: 'tab-1',
      width: 640,
      height: 480,
      visible: true,
    },
  ]);
  view.setVisible(true);
  await sleep(10);
  assert.strictEqual(sent.length, 1, 'an unchanged viewport is not resent');
  view.resubscribe();
  assert.strictEqual(sent.length, 2, 'resubscribe reports again');
  view.setVisible(false);
  assert.strictEqual(sent[2].visible, false);
  view.setVisible(false);
  assert.strictEqual(sent.length, 3);
  view.resubscribe();
  assert.strictEqual(sent.length, 3, 'hidden: nothing to report');

  view.setVisible(true);
  await sleep(10);
  assert.strictEqual(sent.length, 4);
  view.dispose();
  view.dispose();
  assert.ok(view.disposed);
  assert.strictEqual(sent.length, 5);
  assert.strictEqual(sent[4].visible, false);
  view.frame({data: 'AAAA', width: 10, height: 10});
  assert.strictEqual(view.pageWidth, 0, 'no frame after dispose');
}

(async () => {
  await testAddressBarAndState();
  await testViewportAndDispose();
  console.log('browserTabView.test.js passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
