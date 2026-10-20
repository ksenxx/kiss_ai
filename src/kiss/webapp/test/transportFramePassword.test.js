'use strict';
// transport.js against the daemon's per-frame password contract:
//   * the socket scheme follows the page (ws:// on http://, wss:// on https://);
//   * every frame sent after auth_ok repeats the password the socket
//     authenticated with (the server checks it on each command), including
//     the frames queued while the socket was down and their trailing ping;
//   * a post-auth error/auth_failed (the password was rotated) forgets the
//     stored password so the reconnect probes with the empty one;
//   * a browser the daemon admits without a password (http://127.0.0.1,
//     nothing stored) sends its frames bare: no password key at all.
const assert = require('assert');
const {sleep} = require('./simplify2_harness');
const {
  newAuthDom,
  bootShim,
  installFakeWebSocket,
  frames,
  waitForSocket,
} = require('./transport_page_helpers');

function ok(msg) {
  console.log('  ok -', msg);
}

function fail(msg, err) {
  console.error('  FAIL -', msg);
  if (err) console.error('       ', err.message || err);
  process.exit(1);
}

async function run() {
  // 1. Scheme follows the page.
  for (const [url, scheme] of [
    ['http://127.0.0.1:8787/', 'ws://127.0.0.1:8787/ws'],
    ['https://example.test/', 'wss://example.test/ws'],
  ]) {
    const {window} = newAuthDom(url);
    const sockets = [];
    installFakeWebSocket(window, sockets);
    bootShim(window);
    try {
      assert.strictEqual(sockets[0].url, scheme);
      ok(`page ${url} opens ${scheme}`);
    } catch (err) {
      fail(`socket scheme for ${url}`, err);
    }
    window.close();
  }

  // 2. Every post-auth frame carries the password, queued ones included.
  {
    const {window} = newAuthDom('http://127.0.0.1:8787/');
    window.localStorage.setItem('sorcar-remote-pwd', 'Secret20Chars0000000');
    const sockets = [];
    installFakeWebSocket(window, sockets);
    bootShim(window);
    const transport = window.acquireTransport();
    transport.postMessage({type: 'ready'});
    transport.postMessage({});
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});
    transport.postMessage({type: 'runTask', prompt: 'a "quoted" {thing}'});
    try {
      assert.deepStrictEqual(frames(sock), [
        {type: 'auth', password: 'Secret20Chars0000000'},
        {password: 'Secret20Chars0000000', type: 'ready'},
        {password: 'Secret20Chars0000000'},
        {password: 'Secret20Chars0000000', type: 'ping'},
        {password: 'Secret20Chars0000000', type: 'runTask', prompt: 'a "quoted" {thing}'},
      ]);
      ok('the auth frame, the queued batch, its ping and a live frame all carry the password');
    } catch (err) {
      fail('per-frame password', err);
    }

    // 3. A rotated password: the server refuses a frame and closes;
    //    the stored password is forgotten, the reconnect probes with "".
    sock.fireMessage({type: 'error', code: 'auth_failed', text: 'Wrong password.'});
    sock.close();
    try {
      assert.strictEqual(window.localStorage.getItem('sorcar-remote-pwd'), null);
      await waitForSocket(sockets, 2);
      sockets[1].fireOpen();
      assert.deepStrictEqual(frames(sockets[1]), [{type: 'auth', password: ''}]);
      ok('post-auth auth_failed forgets the password; the reconnect probes with ""');
    } catch (err) {
      fail('rotated password handling', err);
    }

    // 4. The password typed into the dialog is the one later frames carry.
    sockets[1].fireMessage({type: 'auth_required'});
    window.document.getElementById('auth-modal-input').value = 'NewPassword000000000';
    window.document.getElementById('auth-modal-ok').click();
    // The dialog resolves a promise before the auth frame goes out.
    const authDeadline = Date.now() + 3000;
    while (sockets[1].sent.length < 2) {
      if (Date.now() > authDeadline) throw new Error('the typed password never went out');
      await sleep(20);
    }
    sockets[1].fireMessage({type: 'auth_ok'});
    transport.postMessage({type: 'ping'});
    try {
      // After the auth frame: the unconfirmed batch of socket 1 (no
      // pong came back) is re-sent, then its ping, then the live ping.
      assert.deepStrictEqual(frames(sockets[1]).slice(1), [
        {type: 'auth', password: 'NewPassword000000000'},
        {password: 'NewPassword000000000', type: 'ready'},
        {password: 'NewPassword000000000'},
        {password: 'NewPassword000000000', type: 'ping'},
        {password: 'NewPassword000000000', type: 'ping'},
      ]);
      ok('frames after a typed password carry that password');
    } catch (err) {
      fail('typed password in frames', err);
    }
    window.close();
  }

  // 5. Admitted without a password (a loopback browser): frames go out
  //    bare, with no password key, and no dialog opens.
  {
    const {window} = newAuthDom('http://127.0.0.1:8787/');
    const sockets = [];
    installFakeWebSocket(window, sockets);
    bootShim(window);
    const transport = window.acquireTransport();
    transport.postMessage({type: 'ready'});
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok', local: false});
    transport.postMessage({type: 'getConfig'});
    try {
      assert.deepStrictEqual(frames(sock), [
        {type: 'auth', password: ''},
        {type: 'ready'},
        {type: 'ping'},
        {type: 'getConfig'},
      ]);
      const modal = window.document.getElementById('auth-modal');
      assert.ok(!modal || modal.style.display !== 'flex', 'no password dialog opened');
      ok('a browser admitted without a password sends bare frames and sees no dialog');
    } catch (err) {
      fail('bare frames without a password', err);
    }
    // Such a page saving a new password (Settings > Remote password)
    // keeps sending bare frames: it was never admitted by password.
    sock.fireMessage({type: 'remotePasswordChanged', password: 'Rotated0Password0001'});
    transport.postMessage({type: 'ping'});
    try {
      assert.deepStrictEqual(frames(sock).at(-1), {type: 'ping'});
      assert.strictEqual(window.localStorage.getItem('sorcar-remote-pwd'), null);
      ok('a page admitted without a password ignores remotePasswordChanged for its frames');
    } catch (err) {
      fail('remotePasswordChanged on a bare session', err);
    }
    window.close();
  }

  // 6. Settings > Remote password saved from a password-authenticated
  //    page: the remotePasswordChanged event reaches the page AND the transport
  //    adopts the new password for every later frame (the server checks
  //    each frame against config.json, where the new one now is) and
  //    for the next page load.  A malformed event changes nothing.
  {
    const {window} = newAuthDom('https://example.test/');
    window.localStorage.setItem('sorcar-remote-pwd', 'Secret20Chars0000000');
    const sockets = [];
    installFakeWebSocket(window, sockets);
    bootShim(window);
    const received = []; // what the page sees, minus the shim's own daemonStatus
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type !== 'daemonStatus') received.push(ev.data);
    });
    const transport = window.acquireTransport();
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok', local: false});
    transport.postMessage({
      type: 'saveConfig', config: {remote_password: 'Rotated0Password0001'},
    });
    sock.fireMessage({type: 'remotePasswordChanged', password: 'Rotated0Password0001'});
    sock.fireMessage({type: 'configData', config: {remote_password: 'Rotated0Password0001'}});
    transport.postMessage({type: 'getConfig'});
    for (const bad of [{}, {password: 42}]) {
      sock.fireMessage({type: 'remotePasswordChanged', ...bad});
    }
    transport.postMessage({type: 'ping'});
    try {
      assert.deepStrictEqual(frames(sock), [
        {type: 'auth', password: 'Secret20Chars0000000'},
        {
          password: 'Secret20Chars0000000', type: 'saveConfig',
          config: {remote_password: 'Rotated0Password0001'},
        },
        {password: 'Rotated0Password0001', type: 'getConfig'},
        {password: 'Rotated0Password0001', type: 'ping'},
      ]);
      assert.strictEqual(
        window.localStorage.getItem('sorcar-remote-pwd'), 'Rotated0Password0001',
      );
      assert.deepStrictEqual(
        received.map((m) => m.type),
        ['remotePasswordChanged', 'configData', 'remotePasswordChanged',
         'remotePasswordChanged'],
      );
      assert.strictEqual(received[0].password, 'Rotated0Password0001');
      ok('after remotePasswordChanged the page sees the event and frames carry the new password');
    } catch (err) {
      fail('remotePasswordChanged adoption', err);
    }

    // 7. The same page clears the password (remote access off).  It is
    //    signed out with everyone else: the old credential is forgotten
    //    at once, the next frame goes bare and the server refuses it;
    //    the reconnect probes bare and shows the handshake's refusal.
    sock.fireMessage({type: 'remotePasswordChanged', password: ''});
    transport.postMessage({type: 'ping'});
    try {
      assert.deepStrictEqual(frames(sock).at(-1), {type: 'ping'});
      assert.strictEqual(window.localStorage.getItem('sorcar-remote-pwd'), null);
      sock.fireMessage({type: 'error', code: 'auth_failed', text: 'Wrong password.'});
      sock.close();
      await waitForSocket(sockets, 2);
      sockets[1].fireOpen();
      assert.deepStrictEqual(frames(sockets[1]), [{type: 'auth', password: ''}]);
      const refusal = 'Remote access is turned off: no remote password is set.';
      sockets[1].fireMessage({type: 'error', code: 'localhost_only', text: refusal});
      const overlay = window.document.getElementById('kiss-server-loading-msg');
      assert.ok(overlay, 'the connecting overlay message node exists');
      assert.ok(overlay.textContent.indexOf(refusal) >= 0, overlay.textContent);
      ok('clearing the password signs the page out; the reconnect shows remote access is off');
    } catch (err) {
      fail('remotePasswordChanged with an empty password', err);
    }
    window.close();
  }
  process.exit(0);
}

run().catch((err) => fail('unexpected error', err));
