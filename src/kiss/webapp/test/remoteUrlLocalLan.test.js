// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Where the Cloudflare tunnel URL is shown (the settings panel's
// #remote-url), the page must also show the 127.0.0.1 URL for the
// local machine and the https://<lan-ip>:PORT URL(s) for other devices
// on the LAN.  These tests drive the real `remote_url` message through
// main.js inside jsdom and assert on the rendered DOM.

'use strict';

const assert = require('assert');
const {
  makePage: makeSharedPage,
  send,
  stubRaf,
} = require('./simplify2_harness');

function makePage() {
  const {win} = makeSharedPage({beforeScripts: stubRaf});
  return {win};
}

const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

function barsIn(win, containerId) {
  const container = win.document.getElementById(containerId);
  assert.ok(container, `#${containerId} must exist`);
  return Array.from(container.querySelectorAll('.remote-url-bar')).map(bar => ({
    label: bar.querySelector('.remote-url-label').textContent,
    url: bar.querySelector('.remote-url-link').getAttribute('href'),
  }));
}

// With the daemon's auto-generated certificate (localCa: true) the
// Local/LAN URLs are signed by the machine-local CA; the container
// must end with one hint telling the user how to trust that CA, linking
// /ca.crt on the LAN URL (what a phone can reach).  An explicitly
// configured certificate (no localCa) has no CA to offer: no hint.
function hintIn(win, containerId) {
  const hints = win.document
    .getElementById(containerId)
    .querySelectorAll('.remote-url-tls-hint');
  return hints.length === 1 ? hints[0] : null;
}

async function main() {
  await test('tunnel URL is shown together with 127.0.0.1 and LAN URLs', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: 'https://random-words.trycloudflare.com',
      loopbackUrl: 'https://127.0.0.1:8787',
      lanUrls: ['https://192.168.1.42:8787', 'https://10.0.0.7:8787'],
    });
    const bars = barsIn(win, 'remote-url');
    assert.strictEqual(bars.length, 4, '#remote-url must show 4 URL bars');
    assert.strictEqual(bars[0].url, 'https://random-words.trycloudflare.com');
    assert.strictEqual(bars[1].url, 'https://127.0.0.1:8787');
    assert.strictEqual(bars[1].label, 'Local (this machine)');
    assert.strictEqual(bars[2].url, 'https://192.168.1.42:8787');
    assert.strictEqual(bars[2].label, 'LAN (local network)');
    assert.strictEqual(bars[3].url, 'https://10.0.0.7:8787');
    assert.strictEqual(bars[3].label, 'LAN (local network)');
    win.close();
  });

  await test('without a tunnel the 127.0.0.1 and LAN URLs are still shown', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: 'https://localhost:8787',
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

  await test('local/LAN URLs equal to the primary URL are not duplicated', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: 'https://127.0.0.1:8787',
      loopbackUrl: 'https://127.0.0.1:8787',
      lanUrls: ['https://127.0.0.1:8787', 'https://192.168.1.9:8787', ''],
    });
    const bars = barsIn(win, 'remote-url');
    assert.strictEqual(bars.length, 2, 'primary URL must not repeat');
    assert.strictEqual(bars[0].url, 'https://127.0.0.1:8787');
    assert.strictEqual(bars[1].url, 'https://192.168.1.9:8787');
    win.close();
  });

  await test('legacy remote_url without the new fields still renders one bar', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: 'https://legacy.trycloudflare.com',
    });
    const bars = barsIn(win, 'remote-url');
    assert.strictEqual(bars.length, 1);
    assert.strictEqual(bars[0].url, 'https://legacy.trycloudflare.com');
    win.close();
  });

  await test('ntfy URL keeps its label and is followed by local + LAN URLs', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: 'https://tunnel.trycloudflare.com',
      ntfyUrl: 'https://ntfy.sh/some-topic',
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

  await test('empty remote_url clears the container except local/LAN info', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: '',
      loopbackUrl: 'https://127.0.0.1:8787',
      lanUrls: [],
    });
    const bars = barsIn(win, 'remote-url');
    assert.strictEqual(bars.length, 1, 'only the loopback URL renders');
    assert.strictEqual(bars[0].url, 'https://127.0.0.1:8787');
    win.close();
  });

  await test('a trust-the-CA hint linking /ca.crt on the LAN URL follows the bars', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: 'https://tunnel.trycloudflare.com',
      loopbackUrl: 'https://127.0.0.1:8787',
      lanUrls: ['https://192.168.0.5:8787/', 'https://10.0.0.7:8787'],
      localCa: true,
    });
    const hint = hintIn(win, 'remote-url');
    assert.ok(hint, '#remote-url shows exactly one TLS hint');
    assert.ok(hint.textContent.includes('kiss-web --trust-ca'));
    assert.strictEqual(
      hint.querySelector('a').getAttribute('href'),
      'https://192.168.0.5:8787/ca.crt',
    );
    const container = win.document.getElementById('remote-url');
    assert.strictEqual(container.lastElementChild, hint, 'hint comes last');
    win.close();
  });

  await test('the hint falls back to the loopback URL when there is no LAN URL', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: '',
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

  await test('no local/LAN URL means no TLS hint', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: 'https://legacy.trycloudflare.com',
      localCa: true,
    });
    assert.strictEqual(hintIn(win, 'remote-url'), null);
    assert.strictEqual(
      win.document.querySelectorAll('.remote-url-tls-hint').length,
      0,
    );
    win.close();
  });

  await test('an explicitly configured certificate (no localCa) shows no TLS hint', () => {
    const {win} = makePage();
    send(win, {
      type: 'remote_url',
      url: 'https://tunnel.trycloudflare.com',
      loopbackUrl: 'https://127.0.0.1:8787',
      lanUrls: ['https://192.168.0.5:8787'],
    });
    assert.strictEqual(
      barsIn(win, 'remote-url').length,
      3,
      'bars still render',
    );
    assert.strictEqual(
      win.document.querySelectorAll('.remote-url-tls-hint').length,
      0,
    );
    win.close();
  });

  await test('a daemon in a Docker container (inDocker) shows no URL bars at all', () => {
    const {win} = makePage();
    // Bars painted before the daemon said so are cleared too.
    send(win, {
      type: 'remote_url',
      url: 'https://tunnel.trycloudflare.com',
      loopbackUrl: 'https://127.0.0.1:8787',
      lanUrls: ['https://192.168.0.5:8787'],
      localCa: true,
    });
    assert.strictEqual(
      barsIn(win, 'remote-url').length,
      3,
      'bars render on a host',
    );
    send(win, {
      type: 'remote_url',
      url: 'https://tunnel.trycloudflare.com',
      loopbackUrl: 'https://127.0.0.1:8787',
      lanUrls: ['https://192.168.0.5:8787'],
      localCa: true,
      inDocker: true,
    });
    const container = win.document.getElementById('remote-url');
    assert.strictEqual(
      barsIn(win, 'remote-url').length,
      0,
      'no URL bar in Docker',
    );
    assert.strictEqual(
      win.document.querySelectorAll('.remote-url-tls-hint').length,
      0,
      'no TLS hint in Docker',
    );
    assert.strictEqual(container.childNodes.length, 0, '#remote-url is empty');
    // Anything but boolean true means a host: the bars come back.
    send(win, {
      type: 'remote_url',
      url: '',
      loopbackUrl: 'https://127.0.0.1:8787',
      lanUrls: [],
      inDocker: 'true',
    });
    assert.strictEqual(
      barsIn(win, 'remote-url').length,
      1,
      'bars back on a host',
    );
    win.close();
  });

  report('remoteUrlLocalLan');
}

main();
