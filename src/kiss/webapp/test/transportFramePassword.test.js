'use strict';
// transport.js against the daemon's per-frame password contract:
//   * the socket scheme follows the page (ws:// on http://, wss:// on https://);
//   * every frame sent after auth_ok repeats the password the socket
//     authenticated with (the server checks it on each command), including
//     the frames queued while the socket was down and their trailing ping;
//   * a post-auth error/auth_failed (the password was rotated) forgets the
//     stored password so the reconnect probes with the empty one;
//   * a browser the daemon admits without a password (http://127.0.0.1,
//     nothing stored) sends its frames bare: no password key at all;
//   * the auth frame says the page joins the parts of a big frame
//     (`parts: true`), and the shim joins them (`_joinPart`): the page
//     sees the frame whole, a part out of sequence is dropped.
const assert = require('assert');
const {
  newAuthDom,
  bootShim,
  installFakeWebSocket,
  installFakeTimers,
  openNextSocket,
  unlock,
  overlayText,
  frames,
  ok,
  fail,
} = require('./transport_page_helpers');

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
    const timers = installFakeTimers(window);
    bootShim(window);
    const transport = window.acquireTransport();
    transport.postMessage({type: 'ready'});
    transport.postMessage({});
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});
    transport.postMessage({type: 'runTask', prompt: 'a "quoted" {thing}'});
    try {
      assert.strictEqual(frames(sock).length, 4, 'the runTask waits for the pong');
      // The daemon answers the batch's ping; the held frame goes out.
      sock.fireMessage({type: 'pong'});
      assert.deepStrictEqual(frames(sock), [
        {type: 'auth', password: 'Secret20Chars0000000', parts: true},
        {password: 'Secret20Chars0000000', type: 'ready'},
        {password: 'Secret20Chars0000000'},
        {password: 'Secret20Chars0000000', type: 'ping'},
        {password: 'Secret20Chars0000000', type: 'runTask', prompt: 'a "quoted" {thing}'},
      ]);
      ok('the auth frame, the queued batch, its ping and a held frame all carry the password');
    } catch (err) {
      fail('per-frame password', err);
    }

    // 3. A rotated password: the server refuses a frame and closes;
    //    the stored password is forgotten, the reconnect probes with "".
    sock.fireMessage({type: 'error', code: 'auth_failed', text: 'Wrong password.'});
    sock.close();
    try {
      assert.strictEqual(window.localStorage.getItem('sorcar-remote-pwd'), null);
      openNextSocket(sockets, timers, 0);
      sockets[1].fireOpen();
      assert.deepStrictEqual(frames(sockets[1]), [
        {type: 'auth', password: '', parts: true},
      ]);
      ok('post-auth auth_failed forgets the password; the reconnect probes with ""');
    } catch (err) {
      fail('rotated password handling', err);
    }

    // 4. The password typed into the dialog is the one later frames carry.
    sockets[1].fireMessage({type: 'auth_required'});
    await unlock(window.document, 'NewPassword000000000');
    assert.strictEqual(sockets[1].sent.length, 2, 'the typed password went out');
    // Posted while socket 2 waits for its auth_ok: queued for it.
    transport.postMessage({type: 'getConfig'});
    sockets[1].fireMessage({type: 'auth_ok'});
    sockets[1].fireMessage({type: 'pong'});
    transport.postMessage({type: 'ping'});
    try {
      // After the auth frame: the batch queued for this socket, its
      // ping, then a live frame.  Socket 1 confirmed its own batch, so
      // nothing of it is sent again.
      assert.deepStrictEqual(frames(sockets[1]).slice(1), [
        {type: 'auth', password: 'NewPassword000000000', parts: true},
        {password: 'NewPassword000000000', type: 'getConfig'},
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
    sock.fireMessage({type: 'pong'}); // the daemon answers the batch's ping
    transport.postMessage({type: 'getConfig'});
    try {
      assert.deepStrictEqual(frames(sock), [
        {type: 'auth', password: '', parts: true},
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
    const timers = installFakeTimers(window);
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
        {type: 'auth', password: 'Secret20Chars0000000', parts: true},
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

    // 7. The password is cleared (remote access off), by this page or
    //    any other.  A page admitted by password is told nothing but the
    //    shut-out: the old credential is forgotten, the reconnect probes
    //    bare and shows the handshake's refusal.
    try {
      sock.fireMessage({type: 'error', code: 'auth_failed', text: 'Wrong password.'});
      assert.strictEqual(window.localStorage.getItem('sorcar-remote-pwd'), null);
      sock.close();
      openNextSocket(sockets, timers, 0);
      sockets[1].fireOpen();
      assert.deepStrictEqual(frames(sockets[1]), [
        {type: 'auth', password: '', parts: true},
      ]);
      const refusal = 'Remote access is turned off: no remote password is set.';
      sockets[1].fireMessage({type: 'error', code: 'localhost_only', text: refusal});
      assert.ok(overlayText(window).includes(refusal), overlayText(window));
      ok('clearing the password signs the page out; the reconnect shows remote access is off');
    } catch (err) {
      fail('a cleared password on a password page', err);
    }
    window.close();
  }

  // 8. The empty password is told only to a page the daemon serves
  //    without one: the browser on its own machine.  That page may
  //    still hold a password in localStorage (typed before loopback
  //    pages stopped being asked for one) and repeats it, unread, in
  //    every frame.  Told the empty one, it forgets the stored copy
  //    and sends bare frames.
  {
    const {window} = newAuthDom('http://127.0.0.1:8787/');
    window.localStorage.setItem('sorcar-remote-pwd', 'Secret20Chars0000000');
    const sockets = [];
    installFakeWebSocket(window, sockets);
    bootShim(window);
    const transport = window.acquireTransport();
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok', local: false});
    transport.postMessage({type: 'saveConfig', config: {remote_password: ''}});
    sock.fireMessage({type: 'remotePasswordChanged', password: ''});
    transport.postMessage({type: 'getConfig'});
    try {
      assert.deepStrictEqual(frames(sock), [
        {type: 'auth', password: 'Secret20Chars0000000', parts: true},
        {
          password: 'Secret20Chars0000000', type: 'saveConfig',
          config: {remote_password: ''},
        },
        {type: 'getConfig'},
      ]);
      assert.strictEqual(window.localStorage.getItem('sorcar-remote-pwd'), null);
      ok('a loopback page told the empty password forgets its stored one and sends bare frames');
    } catch (err) {
      fail('remotePasswordChanged with an empty password', err);
    }
    window.close();
  }

  // 9. A big frame comes as numbered parts (the daemon's frame_parts):
  //    the page sees it whole once the last part landed; a part out of
  //    sequence (the start was never seen) is dropped with what was
  //    gathered; a plain frame passes through as it is.
  {
    const {window} = newAuthDom('http://127.0.0.1:8787/');
    const sockets = [];
    installFakeWebSocket(window, sockets);
    bootShim(window);
    // What the page sees, as JSON text (the window's objects are of
    // another realm, which deepStrictEqual would tell apart).
    const received = [];
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type !== 'daemonStatus')
        received.push(JSON.stringify(ev.data));
    });
    window.acquireTransport();
    const sock = sockets[0];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok', local: false});
    sock.fireMessage({type: 'pong'});
    const big = JSON.stringify({type: 'task_events', events: [{text: 'x'.repeat(200)}]});
    const half = Math.ceil(big.length / 2);
    // A part as the daemon serialises it (Python's json.dumps: a space
    // after each colon, `type` first), which is what the shim looks for.
    const part = (seq, data, more) =>
      sock.onmessage({
        data:
          '{"type": "part", "seq": ' + seq + ', "data": ' + JSON.stringify(data) +
          (more ? ', "more": true}' : '}'),
      });
    try {
      assert.deepStrictEqual(frames(sock).slice(0, 1), [
        {type: 'auth', password: '', parts: true},
      ]);
      received.length = 0;
      part(0, big.slice(0, half), true);
      assert.deepStrictEqual(received, [], 'a part with more to come shows nothing');
      part(1, big.slice(half), false);
      assert.deepStrictEqual(received, [big], 'the last part yields the whole frame');
      // A sequence whose start was missed: dropped until a seq 0.
      received.length = 0;
      part(1, big.slice(half), false);
      part(2, 'tail', false);
      assert.deepStrictEqual(received, [], 'parts out of sequence are dropped');
      sock.fireMessage({type: 'tasks_updated'});
      part(0, big.slice(0, half), true);
      part(1, big.slice(half), false);
      assert.deepStrictEqual(
        received, [JSON.stringify({type: 'tasks_updated'}), big],
        'a plain frame passes through; the next seq 0 starts a frame again',
      );
      ok('the parts of a big frame are joined; a part out of sequence is dropped');
    } catch (err) {
      fail('parts joining', err);
    }
    window.close();
  }
}

run().catch((err) => fail('unexpected error', err));
