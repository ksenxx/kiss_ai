// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for a clicked link to a file no tab can show
// (media/main.js against the real chat.html).  The daemon answers the
// click's openFile with the file's error flagged `unsupported`, and
//  1. a page on the daemon's own machine has the daemon open the file
//     with its default application (`openWithDefaultApp`) and shows
//     the error only when the daemon answers that nothing opened; an
//     answer no request waits for is ignored;
//  2. everywhere else (another device, a daemon that can start no
//     program, a file of a registered machine) the page asks for the
//     file's bytes (`openFile` with `download`) and offers them as a
//     download link, whatever their type; the tab's reload asks for
//     them the same way;
//  3. a reply the user did not ask to see (a background open, another
//     chat's, one without a click's token, a reload's) does neither.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  click,
  ofType,
  plain,
  stubMonaco,
} = require('./simplify2_harness.js');
const {contentTabs} = require('./subagent_tab_helpers');

const WD = '/ws/proj';
// A picture no browser tab draws: offered as a file, never as an <img>.
const FILE = WD + '/out/scan.tiff';
const ERROR = 'Cannot display binary file: out/scan.tiff';
const LOCAL = 'https://localhost:8787/';
const BYTES = 'iVBORw0KGgo=';

/**
 * A connected page at *url* whose daemon said whether it can start a
 * program, with a transcript link to out/scan.tiff (a file of the
 * registered *machine*, when one is named).
 */
function page(url, opensBrowser, machine) {
  const made = makePage({
    url,
    beforeScripts: win => {
      stubMonaco(win);
      win.URL.createObjectURL = () => 'blob:fake';
      win.URL.revokeObjectURL = () => {};
    },
  });
  send(made.win, {type: 'daemonStatus', connected: true});
  send(made.win, {
    type: 'configData',
    config: {work_dir: WD},
    apiKeys: {},
    opensBrowser,
  });
  const link = made.win.document.createElement('span');
  link.className = 'kiss-filelink';
  link.dataset.path = 'out/scan.tiff';
  if (machine) link.setAttribute('data-path-machine', machine);
  made.win.document.getElementById('output').appendChild(link);
  const tab = made.win._testApi.getActiveTabId();
  return Object.assign(made, {link, tab, stamp: machine ? {machine} : {}});
}

/**
 * Click the link and answer its openFile as the daemon does for a file
 * no tab can show, *fields* on top.  Returns the request.
 */
function clickAndRefuse(ctx, fields) {
  click(ctx.win, ctx.link);
  const request = ofType(ctx.posted, 'openFile').pop();
  send(ctx.win, {
    type: 'fileContent',
    path: FILE,
    name: 'scan.tiff',
    tabId: request.tabId,
    token: request.token,
    error: ERROR,
    unsupported: true,
    ...ctx.stamp,
    ...fields,
  });
  return request;
}

/** The daemon's answer to the openWithDefaultApp numbered *token*. */
function answer(ctx, opened, token) {
  send(ctx.win, {type: 'openWithDefaultAppResult', opened, path: FILE, token});
}

/** The message of the file-open-error toast ('' when none is up); closes it. */
function takeToast(win) {
  const toast = win.document.querySelector(
    '.kiss-notification[data-notification-id="file-open-error"]',
  );
  if (!toast) return '';
  click(win, toast.querySelector('.kiss-notification-close'));
  return toast.querySelector('.kiss-notification-message').textContent;
}

/** The page's requests for a file's bytes. */
function downloads(ctx) {
  return ofType(ctx.posted, 'openFile').filter(m => m.download);
}

/** Click the reload button of the content view on screen; its request. */
function reload(ctx) {
  const views = ctx.win.document.querySelectorAll('.content-tab-view');
  click(ctx.win, views[views.length - 1].querySelector('.content-refresh-btn'));
  return ofType(ctx.posted, 'openFile').pop();
}

function testTheDaemonOpensTheFileOnItsOwnMachine() {
  const ctx = page(LOCAL, true);
  const request = clickAndRefuse(ctx);
  assert.deepStrictEqual(plain(request), {
    type: 'openFile',
    path: 'out/scan.tiff',
    workDir: WD,
    tabId: ctx.tab,
    token: 'open:1',
  });
  assert.deepStrictEqual(plain(ofType(ctx.posted, 'openWithDefaultApp')), [
    {path: FILE, tabId: ctx.tab, token: 'app:1', type: 'openWithDefaultApp'},
  ]);
  assert.strictEqual(takeToast(ctx.win), '', 'the daemon is asked first');
  answer(ctx, true, 'app:1');
  // Answers no request waits for: to one already answered, to a token
  // never issued, and without a token.
  answer(ctx, false, 'app:1');
  answer(ctx, false, 'app:9');
  answer(ctx, false, undefined);
  assert.strictEqual(takeToast(ctx.win), '', 'it opened: nothing to say');
  // The daemon opened nothing (a program, no application for the type).
  clickAndRefuse(ctx);
  answer(ctx, false, 'app:2');
  assert.strictEqual(takeToast(ctx.win), ERROR);
  // An answer that was on its way when the connection dropped is lost
  // with it: the request is forgotten, a straggler says nothing.
  clickAndRefuse(ctx);
  send(ctx.win, {type: 'daemonStatus', connected: false});
  send(ctx.win, {type: 'daemonStatus', connected: true});
  answer(ctx, false, 'app:3');
  assert.strictEqual(takeToast(ctx.win), '');
  assert.strictEqual(ofType(ctx.posted, 'openWithDefaultApp').length, 3);
  assert.strictEqual(downloads(ctx).length + contentTabs(ctx.win).length, 0);
  // An answer that finds another chat on screen says nothing there:
  // the file was this chat's.
  clickAndRefuse(ctx);
  ctx.win._testApi.createNewTab();
  answer(ctx, false, 'app:4');
  assert.strictEqual(takeToast(ctx.win), '');
  ctx.win.close();
  console.log('  ok - on its own machine the daemon opens the file');
}

function testEverywhereElseTheFileIsOfferedAsADownload() {
  const pages = [
    page('https://box.example/', true),
    page(LOCAL, false),
    page(LOCAL, true, 'gpu-box'),
  ];
  for (const ctx of pages) {
    clickAndRefuse(ctx);
    assert.deepStrictEqual(plain(downloads(ctx)), [
      {
        path: FILE,
        tabId: ctx.tab,
        token: 'open:2',
        download: true,
        ...ctx.stamp,
        type: 'openFile',
      },
    ]);
    const bytes = {
      type: 'fileContent',
      path: FILE,
      name: 'scan.tiff',
      tabId: ctx.tab,
      binary: true,
      download: true,
      mime: 'image/tiff',
      base64: BYTES,
      ...ctx.stamp,
    };
    send(ctx.win, {...bytes, token: 'open:2'});
    const [tab] = contentTabs(ctx.win);
    assert.deepStrictEqual(
      [tab.contentPath, tab.ownerBrowseMachine],
      [FILE, ctx.stamp.machine || ''],
    );
    // The reload asks for the bytes again, and shows them the same way.
    const again = reload(ctx);
    assert.deepStrictEqual(
      [again.token, again.download, again.machine],
      ['reload:1', true, ctx.stamp.machine],
    );
    send(ctx.win, {...bytes, token: 'reload:1'});
    const view = ctx.win.document.querySelector('.content-tab-view');
    const link = view.querySelector('a.content-binary-note');
    assert.deepStrictEqual(
      [link.download, link.textContent, link.getAttribute('href')],
      ['scan.tiff', 'Download scan.tiff', 'blob:fake'],
    );
    assert.strictEqual(view.querySelector('img'), null, 'not drawn');
    assert.strictEqual(contentTabs(ctx.win).length, 1);
    assert.strictEqual(ofType(ctx.posted, 'openWithDefaultApp').length, 0);
    assert.strictEqual(takeToast(ctx.win), '');
    // A picture a tab shows comes without `download`, and is drawn.
    const png = {path: WD + '/dot.png', name: 'dot.png', mime: 'image/png'};
    send(ctx.win, {...bytes, ...png, download: undefined});
    assert.ok(ctx.win.document.querySelector('img.content-image'));
    ctx.win.close();
  }
  console.log('  ok - everywhere else the file is offered as a download');
}

function testAReplyNobodyAskedToSeeOpensNothing() {
  const ctx = page(LOCAL, true);
  // "Open to the Side" and another chat's open are answered in the
  // background: their failure is not this conversation's to hear.
  clickAndRefuse(ctx, {background: true});
  clickAndRefuse(ctx, {tabId: 'another-chat'});
  assert.strictEqual(takeToast(ctx.win), '');
  // A reply without a click's token (a path sent as a prompt) is the
  // error it was.
  clickAndRefuse(ctx, {token: undefined});
  assert.strictEqual(takeToast(ctx.win), ERROR);
  // A file a tab shows reloads without asking for a download, and a
  // reload that finds the file turned binary keeps the tab as it is.
  const notes = {
    type: 'fileContent',
    path: WD + '/notes.txt',
    name: 'notes.txt',
  };
  send(ctx.win, {...notes, tabId: ctx.tab, content: 'notes\n'});
  const again = reload(ctx);
  assert.deepStrictEqual(
    [again.token, 'download' in again],
    ['reload:1', false],
  );
  send(ctx.win, {
    ...notes,
    tabId: again.tabId,
    token: 'reload:1',
    error: 'Cannot display binary file: notes.txt',
    unsupported: true,
  });
  const kept = contentTabs(ctx.win);
  assert.strictEqual(kept.length, 1);
  assert.strictEqual(kept[0].contentPath, WD + '/notes.txt');
  assert.deepStrictEqual(
    [ofType(ctx.posted, 'openWithDefaultApp'), downloads(ctx)],
    [[], []],
  );
  ctx.win.close();
  console.log('  ok - a reply nobody asked to see opens nothing');
}

testTheDaemonOpensTheFileOnItsOwnMachine();
testEverywhereElseTheFileIsOfferedAsADownload();
testAReplyNobodyAskedToSeeOpensNothing();
console.log('openWithDefaultApp.test.js: all tests passed');
