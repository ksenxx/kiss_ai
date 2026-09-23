// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Wherever the Cloudflare tunnel URL is shown (the settings panel's
// #remote-url and the welcome page's #welcome-remote-url), the webview
// must also show the 127.0.0.1 URL for the local machine and the
// https://<lan-ip>:PORT URL(s) for other devices on the LAN.  These
// tests drive the real `remote_url` message through main.js inside
// jsdom and assert on the rendered DOM of BOTH containers.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=remoteurllan-main.js',
  );
  return {win};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
}

function barsIn(win, containerId) {
  const container = win.document.getElementById(containerId);
  assert.ok(container, `#${containerId} must exist`);
  return Array.from(container.querySelectorAll('.remote-url-bar')).map(
    bar => ({
      label: bar.querySelector('.remote-url-label').textContent,
      url: bar.querySelector('.remote-url-link').getAttribute('href'),
    }),
  );
}

test('tunnel URL is shown together with 127.0.0.1 and LAN URLs in both containers', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: 'https://random-words.trycloudflare.com',
    tunnelActive: true,
    loopbackUrl: 'https://127.0.0.1:8787',
    lanUrls: ['https://192.168.1.42:8787', 'https://10.0.0.7:8787'],
  });
  for (const id of ['remote-url', 'welcome-remote-url']) {
    const bars = barsIn(win, id);
    assert.strictEqual(bars.length, 4, `#${id} must show 4 URL bars`);
    assert.strictEqual(bars[0].url, 'https://random-words.trycloudflare.com');
    assert.strictEqual(bars[1].url, 'https://127.0.0.1:8787');
    assert.strictEqual(bars[1].label, 'Local (this machine)');
    assert.strictEqual(bars[2].url, 'https://192.168.1.42:8787');
    assert.strictEqual(bars[2].label, 'LAN (local network)');
    assert.strictEqual(bars[3].url, 'https://10.0.0.7:8787');
    assert.strictEqual(bars[3].label, 'LAN (local network)');
  }
  win.close();
});

test('without a tunnel the 127.0.0.1 and LAN URLs are still shown', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: 'https://localhost:8787',
    tunnelActive: false,
    loopbackUrl: 'https://127.0.0.1:8787',
    lanUrls: ['https://192.168.1.42:8787'],
  });
  const bars = barsIn(win, 'remote-url');
  assert.strictEqual(bars.length, 3);
  assert.strictEqual(bars[0].url, 'https://localhost:8787');
  assert.strictEqual(bars[1].url, 'https://127.0.0.1:8787');
  assert.strictEqual(bars[2].url, 'https://192.168.1.42:8787');
  win.close();
});

test('local/LAN URLs equal to the primary URL are not duplicated', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: 'https://127.0.0.1:8787',
    tunnelActive: false,
    loopbackUrl: 'https://127.0.0.1:8787',
    lanUrls: ['https://127.0.0.1:8787', 'https://192.168.1.9:8787', ''],
  });
  const bars = barsIn(win, 'remote-url');
  assert.strictEqual(bars.length, 2, 'primary URL must not repeat');
  assert.strictEqual(bars[0].url, 'https://127.0.0.1:8787');
  assert.strictEqual(bars[1].url, 'https://192.168.1.9:8787');
  win.close();
});

test('legacy remote_url without the new fields still renders one bar', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: 'https://legacy.trycloudflare.com',
    tunnelActive: true,
  });
  const bars = barsIn(win, 'remote-url');
  assert.strictEqual(bars.length, 1);
  assert.strictEqual(bars[0].url, 'https://legacy.trycloudflare.com');
  win.close();
});

test('ntfy URL keeps its label and is followed by local + LAN URLs', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: 'https://tunnel.trycloudflare.com',
    ntfyUrl: 'https://ntfy.sh/some-topic',
    tunnelActive: true,
    loopbackUrl: 'https://127.0.0.1:8787',
    lanUrls: ['https://192.168.0.5:8787'],
  });
  const bars = barsIn(win, 'remote-url');
  assert.strictEqual(bars.length, 3);
  assert.strictEqual(bars[0].url, 'https://ntfy.sh/some-topic');
  assert.strictEqual(
    bars[0].label,
    'Webapp: click the link in the first post at URL:',
  );
  assert.strictEqual(bars[1].url, 'https://127.0.0.1:8787');
  assert.strictEqual(bars[2].url, 'https://192.168.0.5:8787');
  win.close();
});

test('empty remote_url clears both containers except local/LAN info', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: '',
    tunnelActive: false,
    loopbackUrl: 'https://127.0.0.1:8787',
    lanUrls: [],
  });
  const bars = barsIn(win, 'remote-url');
  assert.strictEqual(bars.length, 1, 'only the loopback URL renders');
  assert.strictEqual(bars[0].url, 'https://127.0.0.1:8787');
  win.close();
});

// With the daemon's auto-generated certificate (localCa: true) the
// Local/LAN URLs are signed by the machine-local CA; both containers
// must end with one hint telling the user how to trust that CA, linking
// /ca.crt on the LAN URL (what a phone can reach).  An explicitly
// configured certificate (no localCa) has no CA to offer: no hint.
function hintIn(win, containerId) {
  const hints = win.document
    .getElementById(containerId)
    .querySelectorAll('.remote-url-tls-hint');
  return hints.length === 1 ? hints[0] : null;
}

test('a trust-the-CA hint linking /ca.crt on the LAN URL follows the bars', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: 'https://tunnel.trycloudflare.com',
    tunnelActive: true,
    loopbackUrl: 'https://127.0.0.1:8787',
    lanUrls: ['https://192.168.0.5:8787/', 'https://10.0.0.7:8787'],
    localCa: true,
  });
  for (const id of ['remote-url', 'welcome-remote-url']) {
    const hint = hintIn(win, id);
    assert.ok(hint, `#${id} shows exactly one TLS hint`);
    assert.ok(hint.textContent.includes('kiss-web --trust-ca'));
    assert.strictEqual(
      hint.querySelector('a').getAttribute('href'),
      'https://192.168.0.5:8787/ca.crt',
    );
    const container = win.document.getElementById(id);
    assert.strictEqual(container.lastElementChild, hint, 'hint comes last');
  }
  win.close();
});

test('the hint falls back to the loopback URL when there is no LAN URL', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: '',
    tunnelActive: false,
    loopbackUrl: 'https://127.0.0.1:8787',
    lanUrls: [],
    localCa: true,
  });
  const hint = hintIn(win, 'remote-url');
  assert.ok(hint);
  assert.strictEqual(
    hint.querySelector('a').getAttribute('href'),
    'https://127.0.0.1:8787/ca.crt',
  );
  win.close();
});

test('no local/LAN URL means no TLS hint', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: 'https://legacy.trycloudflare.com',
    tunnelActive: true,
    localCa: true,
  });
  assert.strictEqual(hintIn(win, 'remote-url'), null);
  assert.strictEqual(
    win.document.querySelectorAll('.remote-url-tls-hint').length,
    0,
  );
  win.close();
});

test('an explicitly configured certificate (no localCa) shows no TLS hint', () => {
  const {win} = makeWebview();
  send(win, {
    type: 'remote_url',
    url: 'https://tunnel.trycloudflare.com',
    tunnelActive: true,
    loopbackUrl: 'https://127.0.0.1:8787',
    lanUrls: ['https://192.168.0.5:8787'],
  });
  assert.strictEqual(barsIn(win, 'remote-url').length, 3, 'bars still render');
  assert.strictEqual(
    win.document.querySelectorAll('.remote-url-tls-hint').length,
    0,
  );
  win.close();
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`FAILED: ${f.name}`);
    console.error(f.error && f.error.stack ? f.error.stack : String(f.error));
  }
  process.exit(1);
}
