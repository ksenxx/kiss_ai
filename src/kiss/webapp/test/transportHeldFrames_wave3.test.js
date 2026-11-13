// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// media/transport.js re-sends the batch it flushed on a reconnect when
// the socket dies before the server's `pong` confirms it.  A frame the
// page posted between that flush and the `pong` (a `submit`, a `stop`,
// a tab close) used to go straight out AND join the batch, so the next
// connection sent it a second time: a prompt the daemon had already
// started ran twice.  Such a frame now waits behind the batch and goes
// out exactly once: after the `pong`, or after the `pong` that confirms
// the batch re-sent on the next connection (wave 4: it never joins the
// re-sent batch, see transportUnconfirmedBatch_wave4.test.js).
//
// The real shim over a fake WebSocket whose sockets are born CONNECTING
// (transport_page_helpers.js); the reconnect backoff timers are fired by
// hand.

const assert = require('assert');
const {SORCAR_API_COMMANDS} = require('../media/api');
const {
  WD,
  makeRunner,
  click,
  showSection,
  toast,
  toastButton,
  explorerRow,
  runMenuItem,
} = require('./ui_antipattern_harness');
const {makePage, stubRaf, fakeMatchMedia} = require('./simplify2_harness');
const {
  newAuthDom,
  bootShim,
  readShimJs,
  installFakeWebSocket,
  installFakeTimers,
  openNextSocket,
  frames,
  submitCount,
  settle,
} = require('./transport_page_helpers');

// The chat.html pages a test built: the runner closes them when the
// test has ended, so a failed one leaves no page timer running.
const windows = [];
const {test, report} = makeRunner({windows});

// One frame of every command a re-sent batch must never carry.  The
// daemon would run the action again (a prompt, an answer, a keystroke,
// a browser Back, Forward or New-tab click, a daemon restart or update,
// a worktree, commit, cron or machine action, a sign-in step, a paid
// transcription or task update, a file or model usage count), or would
// refuse the second delivery of the very change the first one made (a
// promptlet edit or delete, a custom model added).  The Explorer and Git
// actions are once-only too; they are among GIVEN_UP_FRAMES below.
const ONCE_FRAMES = [
  {type: 'submit', prompt: 'run once'},
  {type: 'appendUserMessage', tabId: 'chat-1', prompt: 'steer once'},
  {type: 'userAnswer', tabId: 'chat-1', answer: 'yes'},
  {type: 'newChat', tabId: 'chat-2'},
  {type: 'cronAction', action: 'run_now', jobId: 'j'},
  {type: 'machineAction', action: 'remove', name: 'gpu-box'},
  {type: 'addTrick', text: 'a promptlet'},
  {type: 'editTrick', text: 'a promptlet', newText: 'reworded'},
  {type: 'deleteTrick', text: 'a promptlet'},
  {type: 'saveMyModel', name: 'my-model'},
  {type: 'selectModel', tabId: 'chat-1', model: 'my-model'},
  {type: 'recordFileUsage', path: 'a.txt'},
  {type: 'getTaskUpdate', tabId: 'chat-1', refresh: true},
  {type: 'voiceTranscribe', audio: 'UklGRg=='},
  {type: 'startCodexLogin'},
  {type: 'openExternal', url: 'https://example.test/'},
  {type: 'shareChat', tabId: 'chat-1', html: '<p>chat</p>'},
  {type: 'browserOpen', url: ''},
  {type: 'browserNavigate', tab_id: 'browser-1', action: 'back', url: ''},
  {type: 'browserNavigate', tab_id: 'browser-1', action: 'forward', url: ''},
  {type: 'browserInput', tab_id: 'browser-1', event: {kind: 'key', key: 'a'}},
  {type: 'terminalInput', tab_id: 'terminal-1', data: 'ls\r'},
  {type: 'serverReset'},
  {type: 'runUpdate'},
  {type: 'updateWhenIdle'},
  {type: 'updateModels'},
  {type: 'worktreeAction', action: 'merge', tabId: 'chat-1'},
  {type: 'autocommitAction', tabId: 'chat-1'},
  {type: 'signinLogin', provider: 'x'},
  {type: 'signinPaste', text: 'code-123'},
  {type: 'signinCancel'},
];

// The requests main.js gives up the moment it hears that the connection
// is gone: a file save ("Save failed: the server connection dropped")
// and whatever the Explorer or the Source Control view awaited a reply
// to ("the pending Explorer/Git action was not completed. Try again once
// it reconnects").  Its reply would find no request any more.
const SAVE_FRAME = {
  type: 'saveFile',
  path: 'notes.txt',
  content: 'new text',
  token: 'tab:1',
};
const SHOW_FRAME = {type: 'gitShow', sha: 'abc1234', token: 'show:1'};
const PASTE_FRAME = {
  type: 'fsAction',
  action: 'copy',
  path: '/workspace/a.txt',
  dest: '/workspace',
  token: 'fs:1',
};
const PICK_FRAME = {
  type: 'gitAction',
  action: 'cherryPick',
  sha: 'abc1234',
  token: 'git:1',
};
// Posted while the daemon is unreachable, a save or a show travels in
// the reconnect's batch; a paste or a cherry-pick waits for its pong
// (the daemon would act twice on a re-sent one).
const BATCHED_GIVEN_UP = [SAVE_FRAME, SHOW_FRAME];
const HELD_GIVEN_UP = [PASTE_FRAME, PICK_FRAME];
const GIVEN_UP_FRAMES = [...BATCHED_GIVEN_UP, ...HELD_GIVEN_UP];

// Every other command of the page (media/api.js).  A second delivery
// changes nothing: a read, a setting that names its value, a close or a
// delete of what is already gone, an open the daemon re-attaches.  These
// travel in the batch, which is re-sent until a pong confirms it.
// `stop` is the one open limit (its test is below).
const REPLAYABLE = [
  'stop',
  'interruptTool',
  'openTab',
  'closeTab',
  'resumeSession',
  'ready',
  'getHistory',
  'getAdjacentTask',
  'setFavorite',
  'setChatSummary',
  'deleteTask',
  'deleteChat',
  'getInputHistory',
  'getSeaCommands',
  'getWelcomeInfo',
  'getModels',
  'getConfig',
  'saveConfig',
  'getMyModels',
  'getClaudeCodeStatus',
  'getCodexStatus',
  'deleteMyModel',
  'setWorkDir',
  'setMachine',
  'getFiles',
  'openFile',
  'checkPaths',
  'getReplayImage',
  'getCronJobs',
  'getMachines',
  'getMachineWorkDirs',
  'getAppsStatus',
  'getSpendReport',
  'getRecommendations',
  'listDir',
  'gitStatus',
  'gitLog',
  'shareChatTasks',
  'complete',
  'snoozeUpdate',
  'tipsOptOut',
  'dismissNotice',
  'browserClose',
  'browserViewport',
  'browserFrameAck',
  'terminalOpen',
  'terminalResize',
  'terminalClose',
];

/** A page on the real shim; sockets are born CONNECTING, timers recorded. */
function newPage() {
  const {window} = newAuthDom('https://example.test/');
  const sockets = [];
  installFakeWebSocket(window, sockets, {connecting: true});
  const timers = installFakeTimers(window);
  bootShim(window);
  return {window, sockets, timers, transport: window.acquireTransport()};
}

/**
 * The daemon's side of a handshake on *sock*: `auth_ok`, then the `pong`
 * of the ping that ended the batch flushed on it, if there was one.
 */
function authenticate(sock) {
  sock.fireOpen();
  sock.fireMessage({type: 'auth_ok'});
  if (sentTypes(sock).includes('ping')) sock.fireMessage({type: 'pong'});
}

/**
 * chat.html with main.js and the tree menus on the real shim, its first
 * socket authenticated.  `sent(type)` is every frame of *type* the
 * daemon was sent so far, over all sockets.
 */
async function newAppPage() {
  const sockets = [];
  const {win} = makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['contentContextMenu.js', 'treeContextMenu.js'],
    beforeScripts(w) {
      stubRaf(w);
      fakeMatchMedia(w, true);
      installFakeWebSocket(w, sockets, {connecting: true});
      w.eval(readShimJs()); // in place of makePage's recording transport
    },
  });
  windows.push(win);
  authenticate(sockets[0]);
  // The shim hands main.js its events once jsdom has ended the page load.
  await settle();
  return {
    win,
    sockets,
    sent: type => sockets.flatMap(frames).filter(f => f.type === type),
  };
}

/** The frame types one socket carried after its auth frame. */
function sentTypes(sock) {
  return frames(sock)
    .slice(1)
    .map(f => f.type);
}

/**
 * Post *frame* while the socket is CLOSING (the server began the close
 * handshake; onclose has not fired, so main.js has not heard of the
 * drop), let the next connection die before its pong and the third get
 * one.  Return the frame types each socket carried (the third's before
 * and after its pong) and every frame of *frame*'s type that went out.
 */
function postWhileClosing(frame) {
  const {window, sockets, timers, transport} = newPage();
  sockets[0].fireOpen();
  sockets[0].fireMessage({type: 'auth_ok'});
  sockets[0].readyState = window.WebSocket.CLOSING;
  transport.postMessage(frame);
  sockets[0].fireClose();
  const second = openNextSocket(sockets, timers, 0);
  second.fireOpen();
  second.fireMessage({type: 'auth_ok'});
  second.fireClose(); // the daemon would have acted; the pong never came
  const third = openNextSocket(sockets, timers, 1);
  third.fireOpen();
  third.fireMessage({type: 'auth_ok'});
  const beforePong = sentTypes(third);
  third.fireMessage({type: 'pong'});
  window.close();
  return {
    closing: sentTypes(sockets[0]),
    second: sentTypes(second),
    beforePong,
    third: sentTypes(third),
    sent: sockets.flatMap(frames).filter(f => f.type === frame.type),
  };
}

async function main() {
  await test(
    'a submit posted before the pong goes out once when the socket dies ' +
      'and the batch is re-sent',
    async () => {
      const {window, sockets, timers, transport} = newPage();
      transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      transport.postMessage({type: 'submit', prompt: 'run once'});
      assert.deepStrictEqual(
        sentTypes(sockets[0]),
        ['saveConfig', 'ping'],
        'the submit waits behind the unconfirmed batch',
      );
      sockets[0].fireClose(); // died before the pong
      const next = openNextSocket(sockets, timers, 0);
      next.fireOpen();
      next.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(next), ['saveConfig', 'ping']);
      next.fireMessage({type: 'pong'});
      assert.deepStrictEqual(sentTypes(next), ['saveConfig', 'ping', 'submit']);
      assert.strictEqual(submitCount(sockets), 1, 'the prompt runs once');
      window.close();
    },
  );

  await test(
    'frames held behind the batch go out once, in order, when the pong ' +
      'arrives, and a later reconnect re-sends none of them',
    async () => {
      const {window, sockets, timers, transport} = newPage();
      transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      transport.postMessage({type: 'submit', prompt: 'first'});
      transport.postMessage({type: 'stop'});
      sockets[0].fireMessage({type: 'pong'});
      assert.deepStrictEqual(sentTypes(sockets[0]), [
        'saveConfig',
        'ping',
        'submit',
        'stop',
      ]);
      // Confirmed: a frame posted now goes straight out.
      transport.postMessage({type: 'getConfig'});
      assert.strictEqual(sentTypes(sockets[0]).at(-1), 'getConfig');
      sockets[0].fireClose();
      const next = openNextSocket(sockets, timers, 0);
      next.fireOpen();
      next.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(next), [], 'nothing is owed any more');
      assert.strictEqual(submitCount(sockets), 1);
      window.close();
    },
  );

  await test('with no batch to confirm, a frame posted after auth_ok goes straight out', async () => {
    const {window, sockets, transport} = newPage();
    sockets[0].fireOpen();
    sockets[0].fireMessage({type: 'auth_ok'});
    transport.postMessage({type: 'submit', prompt: 'live'});
    assert.deepStrictEqual(sentTypes(sockets[0]), ['submit']);
    window.close();
  });

  for (const frame of ONCE_FRAMES) {
    await test(
      `${frame.type}${frame.action ? ':' + frame.action : ''} posted while ` +
        'the socket is CLOSING (main.js has not heard of the drop yet) ' +
        'never joins a re-sent batch: it goes out once, after a pong',
      async () => {
        assert.deepStrictEqual(postWhileClosing(frame), {
          closing: [],
          second: ['ping'], // not in the batch: the ping alone asks for the pong
          beforePong: ['ping'],
          third: ['ping', frame.type],
          sent: [frame], // one user action is one command, as posted
        });
      },
    );
  }

  for (const frame of GIVEN_UP_FRAMES) {
    await test(
      `${frame.type} still queued when main.js hears that the connection ` +
        'is gone (it gives the request up there and then) is never sent; ' +
        'the one the user makes once it is back is delivered, once',
      async () => {
        const {window, sockets, timers, transport} = newPage();
        const told = [];
        window.addEventListener('message', e => {
          if (e.data.type === 'daemonStatus') told.push(e.data.connected);
        });
        sockets[0].fireOpen();
        sockets[0].fireMessage({type: 'auth_ok'});
        sockets[0].readyState = window.WebSocket.CLOSING;
        transport.postMessage(frame); // main.js has not heard of the drop
        sockets[0].fireClose();
        assert.deepStrictEqual(told, [true, false]);
        const second = openNextSocket(sockets, timers, 0);
        second.fireOpen();
        second.fireMessage({type: 'auth_ok'});
        assert.deepStrictEqual(sentTypes(second), [], 'it was given up');
        transport.postMessage(frame); // the user asks again
        assert.deepStrictEqual(
          sockets.flatMap(frames).filter(f => f.type === frame.type),
          [frame],
          'the one request main.js still awaits',
        );
        window.close();
      },
    );
  }

  for (const frame of BATCHED_GIVEN_UP) {
    await test(
      `a ${frame.type} main.js has given up on is forgotten wherever it ` +
        'waits: in a batch no pong has confirmed, or held behind one',
      async () => {
        const {window, sockets, timers, transport} = newPage();
        transport.postMessage(frame); // before the handshake: in its batch
        transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
        sockets[0].fireOpen();
        sockets[0].fireMessage({type: 'auth_ok'});
        assert.deepStrictEqual(sentTypes(sockets[0]), [
          frame.type,
          'saveConfig',
          'ping',
        ]);
        transport.postMessage(frame); // held behind the unconfirmed batch
        // A wake-up replaces the dying socket before its onclose fires.
        sockets[0].readyState = window.WebSocket.CLOSING;
        window.dispatchEvent(new window.Event('focus'));
        sockets[1].fireOpen();
        sockets[1].fireMessage({type: 'auth_ok'});
        assert.deepStrictEqual(
          sentTypes(sockets[1]),
          ['saveConfig', 'ping'],
          'the batch again, less the request given up',
        );
        transport.postMessage(frame); // held behind the unconfirmed batch
        sockets[1].fireClose(); // died before the pong
        const third = openNextSocket(sockets, timers, 1);
        third.fireOpen();
        third.fireMessage({type: 'auth_ok'});
        third.fireMessage({type: 'pong'});
        assert.deepStrictEqual(sentTypes(third), ['saveConfig', 'ping']);
        window.close();
      },
    );
  }

  for (const frame of HELD_GIVEN_UP) {
    await test(
      `${frame.type} posted under the reconnecting banner goes out once, ` +
        'after the pong of the next connection; when that connection dies ' +
        'before its pong main.js gives the request up, and so does the shim',
      async () => {
        const {window, sockets, timers, transport} = newPage();
        sockets[0].fireOpen();
        sockets[0].fireMessage({type: 'auth_ok'});
        sockets[0].fireClose(); // main.js hears of the drop: the banner
        transport.postMessage(frame);
        const second = openNextSocket(sockets, timers, 0);
        second.fireOpen();
        second.fireMessage({type: 'auth_ok'});
        assert.deepStrictEqual(sentTypes(second), ['ping'], 'held back');
        second.fireMessage({type: 'pong'});
        assert.deepStrictEqual(sentTypes(second), ['ping', frame.type]);
        // The same again, but this time the pong never comes.
        second.fireClose();
        transport.postMessage(frame);
        const third = openNextSocket(sockets, timers, 1);
        third.fireOpen();
        third.fireMessage({type: 'auth_ok'});
        assert.deepStrictEqual(sentTypes(third), ['ping'], 'held back');
        third.fireClose(); // main.js: "not completed. Try again ..."
        const fourth = openNextSocket(sockets, timers, 2);
        fourth.fireOpen();
        fourth.fireMessage({type: 'auth_ok'});
        assert.deepStrictEqual(sentTypes(fourth), [], 'nothing is owed');
        assert.deepStrictEqual(
          sockets.flatMap(frames).filter(f => f.type === frame.type),
          [frame],
          'the one a pong let through',
        );
        window.close();
      },
    );
  }

  await test(
    'on the page: an Explorer delete confirmed while the socket is CLOSING ' +
      'is reported as not completed and never reaches the daemon; the ' +
      'retry goes out once and its reply is acted on',
    async () => {
      const {win, sockets, sent} = await newAppPage();
      // Answer the newest listing the page asked for, as the daemon does.
      const listFolder = sock => {
        const asked = sent('listDir').pop();
        sock.fireMessage({
          type: 'dirListing',
          token: asked.token,
          path: asked.path,
          root: asked.path,
          entries: [{name: 'foo.py', path: WD + '/foo.py', isDir: false}],
        });
      };
      sockets[0].fireMessage({type: 'configData', config: {work_dir: WD}});
      showSection(win, 'meta-explorer');
      listFolder(sockets[0]);
      runMenuItem(win, explorerRow(win, WD + '/foo.py'), 'Delete');
      // The server began the close handshake; onclose has not fired.
      sockets[0].readyState = win.WebSocket.CLOSING;
      click(win, toastButton(toast(win, 'fs-delete'), 'Delete'));
      assert.deepStrictEqual(sent('fsAction'), [], 'it waits in the shim');
      sockets[0].fireClose();
      assert.strictEqual(
        toast(win, 'sidebar-action-error').querySelector(
          '.kiss-notification-message',
        ).textContent,
        'The server connection dropped; the pending Explorer/Git action was ' +
          'not completed. Try again once it reconnects.',
      );
      win.dispatchEvent(new win.Event('focus')); // the user is back
      authenticate(sockets[1]);
      listFolder(sockets[1]); // the reconnect lists the tree again
      assert.deepStrictEqual(
        sent('fsAction'),
        [],
        'what the user was told had not been done is not done behind them',
      );
      runMenuItem(win, explorerRow(win, WD + '/foo.py'), 'Delete');
      click(win, toastButton(toast(win, 'fs-delete'), 'Delete'));
      const retries = sent('fsAction');
      assert.strictEqual(retries.length, 1, 'the retry, once');
      const listed = sent('listDir').length;
      sockets[1].fireMessage({
        type: 'fsResult',
        token: retries[0].token,
        action: 'delete',
        path: WD + '/foo.py',
      });
      assert.ok(
        sent('listDir').length > listed,
        'the reply finds its request: the tree is listed again',
      );
    },
  );

  await test(
    'every command of media/api.js is delivered one way or the other: ' +
      'the replayable ones travel, in order, in each re-sent batch',
    async () => {
      assert.deepStrictEqual(
        [
          ...new Set(ONCE_FRAMES.map(f => f.type)),
          ...GIVEN_UP_FRAMES.map(f => f.type),
          ...REPLAYABLE,
        ].sort(),
        [...SORCAR_API_COMMANDS].sort(),
        'a command of api.js is in exactly one of the three lists',
      );
      const {window, sockets, timers, transport} = newPage();
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      sockets[0].readyState = window.WebSocket.CLOSING;
      for (const type of REPLAYABLE) transport.postMessage({type});
      sockets[0].fireClose();
      const second = openNextSocket(sockets, timers, 0);
      second.fireOpen();
      second.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(second), [...REPLAYABLE, 'ping']);
      second.fireClose(); // died before the pong: the batch is owed again
      const third = openNextSocket(sockets, timers, 1);
      third.fireOpen();
      third.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(third), [...REPLAYABLE, 'ping']);
      window.close();
    },
  );

  await test(
    'stop posted while the socket is CLOSING travels in the batch, as it ' +
      'always has: it goes out at once on the next connection, and again ' +
      'when that connection is replaced before its pong',
    async () => {
      // The second Stop is an open limit, pinned here: a Stop names no
      // run, so the daemon cannot tell it from a new click.  Held back for
      // the pong instead, a Stop would wait out the resync of a slow link.
      const stop = {type: 'stop', tabId: 'chat-1'};
      assert.deepStrictEqual(postWhileClosing(stop), {
        closing: [],
        second: ['stop', 'ping'],
        beforePong: ['stop', 'ping'],
        third: ['stop', 'ping'],
        sent: [stop, stop],
      });
    },
  );

  await test(
    'a cron run_now posted while disconnected waits behind the batch with ' +
      'every frame after it, and a lost pong re-sends the batch alone',
    async () => {
      const {window, sockets, timers, transport} = newPage();
      sockets[0].fireOpen();
      sockets[0].fireMessage({type: 'auth_ok'});
      sockets[0].fireClose();
      transport.postMessage({type: 'saveConfig', config: {value: 'A'}});
      transport.postMessage({
        type: 'cronAction',
        action: 'run_now',
        jobId: 'j',
      });
      transport.postMessage({type: 'getCronJobs'}); // keeps its place
      const second = openNextSocket(sockets, timers, 0);
      second.fireOpen();
      second.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(second), ['saveConfig', 'ping']);
      second.fireClose(); // died before the pong
      const third = openNextSocket(sockets, timers, 1);
      third.fireOpen();
      third.fireMessage({type: 'auth_ok'});
      assert.deepStrictEqual(sentTypes(third), ['saveConfig', 'ping']);
      third.fireMessage({type: 'pong'});
      assert.deepStrictEqual(sentTypes(third), [
        'saveConfig',
        'ping',
        'cronAction',
        'getCronJobs',
      ]);
      assert.strictEqual(
        sockets.flatMap(frames).filter(f => f.type === 'cronAction').length,
        1,
        'the job runs once',
      );
      window.close();
    },
  );

  report('transportHeldFrames_wave3');
}

main();
