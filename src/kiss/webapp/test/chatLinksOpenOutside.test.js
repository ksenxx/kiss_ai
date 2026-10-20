// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the web links the chat shows (a markdown
// link of a prompt or result, a URL in a tool's output): a click never
// navigates the page.  From a browser the link opens in a new tab;
// from the standalone app window (kiss-app, a Chromium --app window of
// the daemon on this machine) the daemon opens it in the user's default
// browser (``openExternal``), and the page falls back to its own
// window.open when the daemon answers that nothing opened.  A download
// link keeps its default behaviour.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  click,
  ofType,
} = require('./simplify2_harness');

const URL = 'https://example.com/docs/page';

function makePage(opts) {
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    url: (opts && opts.url) || 'https://localhost/',
    extraScripts: ['marked.min.js'],
    beforeScripts(win) {
      stubRaf(win);
      if (opts && opts.name) win.name = opts.name;
      if (opts && opts.standalone) {
        win.matchMedia = query => ({
          matches: query === '(display-mode: standalone)',
          media: query,
          addEventListener: () => {},
          removeEventListener: () => {},
          addListener: () => {},
          removeListener: () => {},
        });
      }
    },
  });
  const opened = [];
  win.open = (url, target, features) => {
    opened.push({url, target, features});
    return {};
  };
  const tab = posted.find(m => m.type === 'ready').tabId;
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'clear', chat_id: 'chat-l', tabId: tab});
  send(win, {type: 'status', running: true, tabId: tab, startTs: Date.now()});
  send(win, {
    type: 'prompt',
    text: 'read [the docs](' + URL + ')',
    tabId: tab,
    taskId: '7',
  });
  const link = win.document.querySelector('#output .prompt a[href]');
  assert.ok(link, 'the markdown link is rendered in the transcript');
  assert.strictEqual(link.href, URL);
  return {win, posted, opened, link, tab};
}

function testBrowserOpensANewTab() {
  const {win, posted, opened, link} = makePage();
  const ev = click(win, link);
  assert.ok(ev.defaultPrevented, 'the page itself never navigates');
  assert.deepStrictEqual(opened, [
    {url: URL, target: '_blank', features: 'noopener'},
  ]);
  assert.strictEqual(
    ofType(posted, 'openExternal').length,
    0,
    'a browser needs no daemon help',
  );
  win.close();
  console.log('  ok - from a browser a chat link opens in a new tab');
}

function testAppWindowAsksTheDaemon() {
  const {win, posted, opened, link} = makePage({
    url: 'http://127.0.0.1:8787/',
    name: 'kiss-app-window',
  });
  const ev = click(win, link);
  assert.ok(ev.defaultPrevented);
  assert.deepStrictEqual(
    ofType(posted, 'openExternal').map(m => m.url),
    [URL],
  );
  assert.strictEqual(
    opened.length,
    0,
    'the app window opens no window of its own',
  );
  // The daemon could not start a browser: the page falls back.
  send(win, {type: 'openExternalResult', url: URL, opened: false});
  assert.deepStrictEqual(opened, [
    {url: URL, target: '_blank', features: 'noopener'},
  ]);
  send(win, {type: 'openExternalResult', url: URL, opened: true});
  assert.strictEqual(opened.length, 1, 'an opened link needs no fallback');
  win.close();
  console.log('  ok - the macOS app window hands the link to the daemon');
}

function testStandaloneAppWindowAsksTheDaemon() {
  const {win, posted, opened, link} = makePage({
    url: 'http://localhost:8787/',
    standalone: true,
  });
  click(win, link);
  assert.deepStrictEqual(
    ofType(posted, 'openExternal').map(m => m.url),
    [URL],
  );
  assert.strictEqual(opened.length, 0);
  win.close();
  console.log('  ok - a Chromium --app window hands the link to the daemon');
}

function testRemoteStandaloneUsesTheBrowser() {
  // A PWA of a remote daemon: that daemon cannot open a browser here.
  const {win, posted, opened, link} = makePage({
    url: 'https://box.example.net/',
    standalone: true,
  });
  click(win, link);
  assert.strictEqual(ofType(posted, 'openExternal').length, 0);
  assert.strictEqual(opened.length, 1);
  win.close();
  console.log('  ok - a remote standalone page opens links itself');
}

function testDownloadLinksAndOtherSchemesAreLeftAlone() {
  const {win, posted, opened, link} = makePage();
  link.setAttribute('download', 'page.html');
  let ev = click(win, link);
  assert.ok(!ev.defaultPrevented, 'a download link keeps its default');
  link.removeAttribute('download');
  link.href = 'mailto:someone@example.com';
  ev = click(win, link);
  assert.ok(!ev.defaultPrevented, 'only http(s) links are handled');
  assert.strictEqual(opened.length, 0);
  assert.strictEqual(ofType(posted, 'openExternal').length, 0);
  // Not the primary button, and a click something else already handled.
  link.href = URL;
  const middle = new win.MouseEvent('click', {
    bubbles: true,
    cancelable: true,
    button: 1,
  });
  link.dispatchEvent(middle);
  const handled = new win.MouseEvent('click', {
    bubbles: true,
    cancelable: true,
  });
  handled.preventDefault();
  link.dispatchEvent(handled);
  assert.strictEqual(opened.length, 0);
  win.close();
  console.log(
    '  ok - download, mailto, middle and handled clicks are left alone',
  );
}

testBrowserOpensANewTab();
testAppWindowAsksTheDaemon();
testStandaloneAppWindowAsksTheDaemon();
testRemoteStandaloneUsesTheBrowser();
testDownloadLinksAndOtherSchemesAreLeftAlone();
console.log('chatLinksOpenOutside.test.js passed');
