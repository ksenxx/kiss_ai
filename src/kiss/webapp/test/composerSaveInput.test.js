// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end (jsdom) tests of the composer's Save button
// (#input-save-btn; the `savedinput-coverage` region of media/main.js),
// one per scenario of its contract: the text survives a crash of the
// page or of the daemon and a lost connection, and typed text is never
// overwritten.  The real main.js runs over a recording transport, and
// over the real media/transport.js on a fake socket where the scenario
// is about the connection.  The daemon's replies are played as
// server/input_draft.py gives them: tests/server/test_input_draft.py
// runs the commands asserted here on a live daemon and pins those
// replies and the file.

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  byId,
  typeInto,
  ofType,
  loadMainCss,
  sendTabsState,
  snapshotEntry,
  plain,
  settle,
} = require('./simplify2_harness');
const {makeRunner, toast} = require('./ui_antipattern_harness');
const {loadHistory, makeSession, historyRow} = require('./history_helpers');
const {
  installFakeWebSocket,
  readShimJs,
  frames,
} = require('./transport_page_helpers');

const KEY = 'kiss-saved-input';
const FILE = '/home/u/.kiss/saved_input.txt';
const SAVED = `Saved to ${FILE}.`;
const OFFLINE =
  'Saved in this browser. The file is written when the server ' +
  'connection is back.';
const NOT_KEPT = 'Not saved: this browser cannot keep the text.';
const TOO_LONG = 'Not saved: the text is longer than 1 MB.';

// The runner closes a test's pages when it has ended.
const windows = [];
const {test, report} = makeRunner({windows});

/** The composer of a page: its window, text box and buttons. */
function composer(win) {
  windows.push(win);
  return {
    win,
    inp: byId(win, 'task-input'),
    saveBtn: byId(win, 'input-save-btn'),
    clearBtn: byId(win, 'input-clear-btn'),
    sendBtn: byId(win, 'send-btn'),
  };
}

/**
 * A chat page.  `stored` is what an earlier page left under the browser
 * copy's key (a copy, or a raw string), `state` the session state it
 * persisted; `url: null` opens a page without localStorage.
 */
function makePage(opts) {
  const {stored, state, url, beforeScripts} = opts || {};
  const page = makeChatPage({
    sourceUrl: 'savedinput-main.js',
    initialState: state,
    url,
    beforeScripts(win) {
      if (typeof stored === 'string') win.localStorage.setItem(KEY, stored);
      else if (stored) win.localStorage.setItem(KEY, JSON.stringify(stored));
      if (beforeScripts) beforeScripts(win);
    },
  });
  return Object.assign(page, composer(page.win));
}

/**
 * The page started again after `page` died without a `pagehide` (a
 * crash): the browser copy is what it wrote last, and so is the session
 * state unless the browser itself was started again (`session` false).
 */
function restart(page, session = true) {
  return makePage({
    stored: page.win.localStorage.getItem(KEY) || undefined,
    state: session ? plain(page.getState()) : undefined,
  });
}

/**
 * The browser copy as stored, but for its `rev` (the name of the save or
 * removal it is of, a new one each time); null when the browser keeps
 * none.
 */
function browserCopy(win) {
  const copy = JSON.parse(win.localStorage.getItem(KEY));
  if (copy) delete copy.rev;
  return copy;
}

/** The copy of a text the daemon confirmed, held by the box of `tabId`. */
function kept(text, tabId) {
  return {text, tabId, owed: false, was: [text]};
}

/**
 * The copy a removal the daemon confirmed leaves: no text, and (not
 * shown by browserCopy) the removal's `rev`.
 */
const REMOVED = {text: '', tabId: '', owed: false, was: []};

/** The session drafts as persisted last: no `pagehide` is fired. */
function sessionDrafts(getState) {
  return plain(getState().inputDrafts);
}

/** The saveInputDraft command number `seq`: `text`, naming the texts `was`. */
function cmd(text, seq, ...was) {
  return {type: 'saveInputDraft', text, was, seq};
}

/** The saveInputDraft commands posted so far, in order. */
function saves(posted) {
  return plain(ofType(posted, 'saveInputDraft'));
}

/** The newest saveInputDraft command `page` posted. */
function newest(page) {
  return saves(page.posted).pop();
}

function connect(win) {
  send(win, {type: 'daemonStatus', connected: true});
}

function disconnect(win) {
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
}

/**
 * The daemon's answer to the save command `sent`: its file holds the
 * command's text now, or `holds` (a text it kept: another browser's, or
 * the old one when `error` says that the file could not be changed).
 */
function answer(win, sent, holds, error) {
  const text = holds === undefined ? sent.text : holds;
  send(win, {type: 'inputDraft', text, path: FILE, seq: sent.seq, error});
}

/** The daemon's answer to getInputDraft: its file holds `text`. */
function fileHolds(win, text) {
  send(win, {type: 'inputDraft', text, path: FILE});
}

/** The Save toast as {error, message}; null when none is up. */
function saveToast(win) {
  const el = toast(win, 'saved-input');
  if (!el) return null;
  return {
    error: el.classList.contains('kiss-notification-error'),
    message: el.querySelector('.kiss-notification-message').textContent,
  };
}

/** Type `text` into the composer and click Save. */
function saveText(page, text) {
  typeInto(page.win, page.inp, text);
  click(page.win, page.saveBtn);
}

/**
 * A connected page: the daemon has said that its file holds `holds`,
 * and the first snapshot lists the chats `tabIds`.
 */
function pageWithChats(tabIds, holds = '', opts) {
  const page = makePage(opts);
  connect(page.win);
  fileHolds(page.win, holds);
  sendTabsState(
    page.win,
    tabIds.map(tabId => ({tabId})),
  );
  return page;
}

/** Fill the browser's storage until only `room` more characters fit. */
function fillStorage(win, room) {
  let size = 0;
  for (let step = 1 << 23; step >= 1; step >>= 1) {
    try {
      win.localStorage.setItem('filler', 'x'.repeat(size + step));
      size += step;
    } catch {
      // Over the quota: half the step is tried next.
    }
  }
  win.localStorage.setItem('filler', 'x'.repeat(size - room));
}

/** A page of pageWithChats whose chat on screen holds the saved `text`. */
function savedPage(tabIds, text) {
  const page = pageWithChats(tabIds);
  saveText(page, text);
  answer(page.win, newest(page));
  assert.deepStrictEqual(saveToast(page.win), {error: false, message: SAVED});
  toast(page.win, 'saved-input').remove();
  return page;
}

/** A chat page over the real transport and the sockets it opens. */
function makeTransportPage() {
  const sockets = [];
  const {win} = makeChatPage({
    bodyClass: 'remote-chat',
    sourceUrl: 'savedinput-main.js',
    beforeScripts(w) {
      installFakeWebSocket(w, sockets, {connecting: true});
      w.eval(readShimJs()); // in place of the recording transport
    },
  });
  return Object.assign(composer(win), {sockets});
}

/**
 * Play the daemon accepting socket `n` of the transport page `page`;
 * on the first one it says that its file is empty and lists chat `t1`.
 */
async function openSocket(page, n) {
  const socket = page.sockets[n];
  socket.fireOpen();
  socket.fireMessage({type: 'auth_ok'});
  socket.fireMessage({type: 'pong'});
  if (n > 0) return;
  // The transport hands main.js its events once the page has loaded.
  await settle();
  socket.fireMessage({type: 'inputDraft', text: '', path: FILE});
  socket.fireMessage({type: 'tabs_state', tabs: [snapshotEntry('t1')]});
}

/** The connection of a transport page is lost, and the user comes back. */
function reconnect(page) {
  page.sockets[page.sockets.length - 1].fireClose();
  page.win.dispatchEvent(new page.win.Event('focus'));
}

/** The commands of `type` sent on socket `n` of a transport page. */
function sentOn(page, n, type) {
  return frames(page.sockets[n]).filter(f => f.type === type);
}

async function main() {
  await test('the Save button stands left of the clear button and shows with it', () => {
    const {win, inp, saveBtn, clearBtn, posted} = makePage({
      beforeScripts: loadMainCss,
    });
    assert.strictEqual(saveBtn.parentElement.id, 'input-text-wrap');
    assert.strictEqual(saveBtn.nextElementSibling, clearBtn);
    assert.strictEqual(saveBtn.getAttribute('aria-label'), 'Save input');
    assert.ok(/crash|connection/.test(saveBtn.getAttribute('data-tooltip')));
    assert.ok(saveBtn.querySelector('svg'), 'an icon button');
    const shown = () => [saveBtn.style.display, clearBtn.style.display];
    assert.deepStrictEqual(shown(), ['none', 'none']);
    typeInto(win, inp, 'draft');
    assert.deepStrictEqual(shown(), ['', '']);
    const [save, clear] = [saveBtn, clearBtn].map(b => win.getComputedStyle(b));
    assert.deepStrictEqual(
      [save.width, save.height, save.display],
      [clear.width, clear.height, clear.display],
      'the same box as the clear button',
    );
    // Clearing a text that was never saved posts nothing, nor does a
    // click that reaches the hidden Save button.
    click(win, clearBtn);
    assert.deepStrictEqual(shown(), ['none', 'none']);
    click(win, saveBtn);
    assert.deepStrictEqual(saves(posted), []);
    assert.strictEqual(browserCopy(win), null);
  });

  await test('scenario 1: after a restart the saved text is back in its chat, or in the chat on screen', () => {
    const page = pageWithChats(['t1', 't2']);
    const {win, inp, posted, getState} = page;
    win._testApi.switchToTab('t2');
    saveText(page, 'a long prompt');
    assert.deepStrictEqual(saves(posted), [
      cmd('a long prompt', 1, 'a long prompt'),
    ]);
    assert.deepStrictEqual(browserCopy(win), {
      ...kept('a long prompt', 't2'),
      owed: true,
    });
    assert.deepStrictEqual(sessionDrafts(getState), {t2: 'a long prompt'});
    assert.strictEqual(win.document.activeElement, inp);
    assert.strictEqual(saveToast(win), null, 'the daemon has not answered');
    answer(win, newest(page));
    assert.deepStrictEqual(saveToast(win), {error: false, message: SAVED});
    assert.deepStrictEqual(browserCopy(win), kept('a long prompt', 't2'));
    // The browser is started again: the session's drafts are gone.  The
    // text waits for the daemon's word and for the chats.
    const next = restart(page, false);
    connect(next.win);
    fileHolds(next.win, 'a long prompt');
    assert.strictEqual(next.inp.value, '');
    sendTabsState(next.win, [{tabId: 't1'}, {tabId: 't2'}]);
    assert.strictEqual(next.win._testApi.getActiveTabId(), 't1');
    assert.strictEqual(next.inp.value, '', 'not the chat it was saved in');
    next.win._testApi.switchToTab('t2');
    assert.strictEqual(next.inp.value, 'a long prompt');
    assert.strictEqual(saveToast(next.win), null, 'a restore is not a save');
    // Once per page load: a box emptied by hand stays empty, whatever
    // the daemon sends later, and the file is not read again.
    typeInto(next.win, next.inp, '');
    fileHolds(next.win, 'a long prompt');
    sendTabsState(next.win, [{tabId: 't1'}, {tabId: 't2'}]);
    disconnect(next.win);
    connect(next.win);
    assert.strictEqual(next.inp.value, '');
    assert.strictEqual(ofType(next.posted, 'getInputDraft').length, 1);
    // Its chat is gone: the chat on screen takes the text.  Here the
    // chats are known first, and the first read is lost with its
    // connection: it is asked again.
    const gone = restart(page, false);
    sendTabsState(gone.win, [{tabId: 't1'}]);
    connect(gone.win);
    disconnect(gone.win);
    connect(gone.win);
    assert.strictEqual(ofType(gone.posted, 'getInputDraft').length, 2);
    assert.strictEqual(gone.inp.value, '');
    fileHolds(gone.win, 'a long prompt');
    assert.strictEqual(gone.inp.value, 'a long prompt');
    assert.deepStrictEqual(browserCopy(gone.win), kept('a long prompt', 't1'));
  });

  await test("scenario 2: with the session draft present the box is left alone and is the saved text's box", () => {
    const next = restart(savedPage(['t1'], 'my prompt'));
    const {win, inp, sendBtn, posted} = next;
    assert.strictEqual(inp.value, 'my prompt', "the session's draft");
    let written = 0;
    inp.addEventListener('input', () => {
      written += 1;
    });
    connect(win);
    fileHolds(win, 'my prompt');
    sendTabsState(win, [{tabId: 't1'}]);
    assert.deepStrictEqual([inp.value, written], ['my prompt', 0]);
    assert.deepStrictEqual(browserCopy(win), kept('my prompt', 't1'));
    click(win, sendBtn);
    assert.deepStrictEqual(saves(posted), [cmd('', 1, 'my prompt')]);
    // Its chat is gone and the chat on screen holds exactly the text:
    // that box becomes its box.
    const moved = pageWithChats(['t1'], 'my prompt', {
      stored: kept('my prompt', 't9'),
      state: {chatId: 't1', inputDrafts: {t1: 'my prompt'}},
    });
    assert.strictEqual(moved.inp.value, 'my prompt');
    assert.deepStrictEqual(browserCopy(moved.win), kept('my prompt', 't1'));
    click(moved.win, moved.clearBtn);
    assert.deepStrictEqual(saves(moved.posted), [cmd('', 1, 'my prompt')]);
  });

  await test('scenario 3: a save made offline says so, is written at the reconnect and places nothing', async () => {
    const page = makeTransportPage();
    const {win, inp} = page;
    await openSocket(page, 0);
    assert.strictEqual(sentOn(page, 0, 'getInputDraft').length, 1);
    page.sockets[0].fireClose();
    saveText(page, 'typed offline');
    assert.deepStrictEqual(saveToast(win), {error: false, message: OFFLINE});
    assert.deepStrictEqual(browserCopy(win), {
      ...kept('typed offline', 't1'),
      owed: true,
    });
    // The user empties the box by hand before the connection is back.
    typeInto(win, inp, '');
    win.dispatchEvent(new win.Event('focus'));
    await openSocket(page, 1);
    assert.deepStrictEqual(sentOn(page, 0, 'saveInputDraft'), []);
    assert.deepStrictEqual(sentOn(page, 1, 'saveInputDraft'), [
      cmd('typed offline', 2, 'typed offline'),
    ]);
    assert.deepStrictEqual(sentOn(page, 1, 'getInputDraft'), []);
    page.sockets[1].fireMessage({
      type: 'inputDraft',
      text: 'typed offline',
      path: FILE,
      seq: 2,
    });
    assert.deepStrictEqual(saveToast(win), {error: false, message: SAVED});
    assert.deepStrictEqual(browserCopy(win), kept('typed offline', 't1'));
    assert.strictEqual(inp.value, '', 'this page load has placed already');
  });

  await test('scenario 4: a save made offline is written and restored by the next page', () => {
    const page = pageWithChats(['t1']);
    disconnect(page.win);
    saveText(page, 'typed offline');
    assert.deepStrictEqual(saves(page.posted), []);
    assert.strictEqual(saveToast(page.win).message, OFFLINE);
    // The page dies with its browser; the next one finds the daemon
    // after it knows the chats.
    const next = restart(page, false);
    const {win, inp, posted} = next;
    sendTabsState(win, [{tabId: 't1'}]);
    assert.strictEqual(inp.value, '', 'the daemon has not spoken');
    connect(win);
    assert.deepStrictEqual(saves(posted), [
      cmd('typed offline', 1, 'typed offline'),
    ]);
    assert.deepStrictEqual(ofType(posted, 'getInputDraft'), []);
    answer(win, newest(next));
    assert.strictEqual(saveToast(win).message, SAVED);
    assert.strictEqual(inp.value, 'typed offline');
    assert.deepStrictEqual(browserCopy(win), kept('typed offline', 't1'));
  });

  await test('scenario 5: a saved text that was sent or cleared does not come back', () => {
    for (const how of ['send', 'cross', 'edit and send']) {
      const page = savedPage(['t1'], 'run this');
      const {win, inp, sendBtn, clearBtn, posted, getState} = page;
      assert.deepStrictEqual(sessionDrafts(getState), {t1: 'run this'}, how);
      if (how === 'edit and send') typeInto(win, inp, 'run this, edited');
      click(win, how === 'cross' ? clearBtn : sendBtn);
      assert.strictEqual(inp.value, '', how);
      // Gone from the browser at once; the daemon is told what to delete.
      const owed = {text: '', tabId: '', owed: true, was: ['run this']};
      assert.deepStrictEqual(browserCopy(win), owed, how);
      assert.deepStrictEqual(newest(page), cmd('', 2, 'run this'), how);
      // The session draft the Save wrote goes with a clear (a sent
      // prompt is the session's own until the daemon echoes it).
      const cleared = how === 'cross';
      if (cleared) assert.deepStrictEqual(sessionDrafts(getState), {}, how);
      // The page dies before the daemon answers.  The next one has
      // nothing to put back, and posts the removal again.
      const next = restart(page, cleared);
      connect(next.win);
      sendTabsState(next.win, [{tabId: 't1'}]);
      assert.deepStrictEqual(saves(next.posted), [cmd('', 1, 'run this')], how);
      assert.deepStrictEqual(ofType(next.posted, 'getInputDraft'), [], how);
      answer(next.win, newest(next));
      assert.deepStrictEqual(
        [browserCopy(next.win), next.inp.value, saveToast(next.win)],
        [REMOVED, '', null],
        how,
      );
    }
  });

  await test('the cross clicked before the saved text is placed drops the text its box showed', () => {
    // The last session showed chat t1, whose draft the boot tab holds.
    const page = makePage({
      stored: kept('run this', 't1'),
      state: {chatId: 't1', inputDrafts: {t1: 'run this'}},
    });
    connect(page.win);
    assert.strictEqual(page.inp.value, 'run this');
    click(page.win, page.clearBtn);
    assert.deepStrictEqual(newest(page), cmd('', 1, 'run this'));
    // The chats, and what the file held before: nothing comes back.
    sendTabsState(page.win, [{tabId: 't1'}]);
    fileHolds(page.win, 'run this');
    assert.strictEqual(page.inp.value, '');
    answer(page.win, newest(page));
    assert.deepStrictEqual(
      [browserCopy(page.win), page.inp.value],
      [REMOVED, ''],
    );
    // A boot tab that holds another chat's draft is not the text's box.
    const other = makePage({
      stored: kept('run this', 't1'),
      state: {chatId: 't2', inputDrafts: {t2: 'other draft'}},
    });
    connect(other.win);
    assert.strictEqual(other.inp.value, 'other draft');
    click(other.win, other.clearBtn);
    assert.deepStrictEqual(saves(other.posted), []);
    assert.deepStrictEqual(browserCopy(other.win), kept('run this', 't1'));
    // Nor is the tab of a chat picked from the Chats list meanwhile.
    const picked = makePage({
      stored: kept('run this', 't1'),
      state: {chatId: 't1', inputDrafts: {t1: 'run this'}},
    });
    connect(picked.win);
    const boot = picked.win._testApi.getActiveTabId();
    typeInto(picked.win, picked.inp, '');
    loadHistory(picked.win, [makeSession({id: 'chat-y'})]);
    click(picked.win, historyRow(picked.win));
    assert.notStrictEqual(picked.win._testApi.getActiveTabId(), boot);
    for (const btn of [picked.clearBtn, picked.sendBtn]) {
      typeInto(picked.win, picked.inp, 'unrelated');
      click(picked.win, btn);
      assert.strictEqual(picked.inp.value, '');
    }
    assert.deepStrictEqual(saves(picked.posted), []);
    assert.deepStrictEqual(browserCopy(picked.win), kept('run this', 't1'));
    // Its chat is gone and the chat on screen took the draft, as it was
    // or edited in the boot tab: that box is the text's box once the
    // chats are known, before the daemon speaks.
    for (const edit of ['', ', edited']) {
      const moved = makePage({
        stored: kept('run this', 't1'),
        state: {chatId: 't1', inputDrafts: {t1: 'run this'}},
      });
      connect(moved.win);
      if (edit) typeInto(moved.win, moved.inp, 'run this' + edit);
      sendTabsState(moved.win, [{tabId: 't3'}]);
      assert.deepStrictEqual(
        [moved.inp.value, browserCopy(moved.win)],
        ['run this' + edit, kept('run this', 't3')],
      );
      click(moved.win, moved.clearBtn);
      assert.deepStrictEqual(newest(moved), cmd('', 1, 'run this'), edit);
      fileHolds(moved.win, 'run this');
      assert.strictEqual(moved.inp.value, '', edit);
    }
  });

  await test('scenario 6: a removal names every text the file may hold', async () => {
    const page = makeTransportPage();
    const {win, inp, clearBtn, sockets} = page;
    await openSocket(page, 0);
    saveText(page, 'A');
    sockets[0].fireMessage({type: 'inputDraft', text: 'A', path: FILE, seq: 1});
    assert.strictEqual(saveToast(win).message, SAVED);
    toast(win, 'saved-input').remove();
    // B is posted as the socket closes: its answer never comes, and
    // the transport keeps the command for the next connection.
    sockets[0].readyState = win.WebSocket.CLOSING;
    saveText(page, 'B');
    reconnect(page);
    click(win, clearBtn);
    assert.deepStrictEqual(browserCopy(win), {
      text: '',
      tabId: '',
      owed: true,
      was: ['A', 'B'],
    });
    await openSocket(page, 1);
    // The transport delivers B first; the removal that follows deletes
    // the file whether it holds A or B.
    assert.deepStrictEqual(sentOn(page, 1, 'saveInputDraft'), [
      cmd('B', 2, 'A', 'B'),
      cmd('', 4, 'A', 'B'),
    ]);
    sockets[1].fireMessage({type: 'inputDraft', text: 'B', path: FILE, seq: 2});
    assert.strictEqual(browserCopy(win).owed, true, 'an older answer');
    sockets[1].fireMessage({type: 'inputDraft', text: '', path: FILE, seq: 4});
    assert.deepStrictEqual(
      [browserCopy(win), inp.value, saveToast(win)],
      [REMOVED, '', null],
    );
  });

  await test("scenario 7: another chat's send or clear, or an answer to a question, leaves the text saved", () => {
    const page = pageWithChats(['x', 'y'], '', {beforeScripts: loadMainCss});
    const {win, inp, saveBtn, clearBtn, sendBtn, posted} = page;
    const api = win._testApi;
    saveText(page, 'prompt of x');
    answer(win, newest(page));
    api.switchToTab('y');
    typeInto(win, inp, 'other work');
    click(win, sendBtn);
    assert.strictEqual(ofType(posted, 'submit').length, 1);
    typeInto(win, inp, 'a note');
    click(win, clearBtn);
    // A question of x's agent brings x on screen: its prompt is parked,
    // and the composer is the answer box, which offers no Save.
    send(win, {type: 'askUser', question: 'Deploy?', tabId: 'x'});
    assert.deepStrictEqual([api.getActiveTabId(), inp.value], ['x', '']);
    typeInto(win, inp, 'no');
    const display = el => win.getComputedStyle(el).display;
    assert.deepStrictEqual([display(saveBtn), display(clearBtn)], ['none', 'flex']);
    click(win, clearBtn);
    typeInto(win, inp, 'yes');
    click(win, sendBtn);
    assert.strictEqual(ofType(posted, 'userAnswer').length, 1);
    assert.strictEqual(inp.value, 'prompt of x', 'the prompt is back');
    assert.deepStrictEqual(saves(posted), [cmd('prompt of x', 1, 'prompt of x')]);
    assert.deepStrictEqual(browserCopy(win), kept('prompt of x', 'x'));
  });

  await test('scenario 8: a box that holds text, an answer box and a file tab are never written', () => {
    // Its chat is gone and the chat on screen holds a draft: the text
    // stays in the file, and is not that chat's to drop.
    const other = pageWithChats(['t1'], 'saved', {
      stored: kept('saved', 't9'),
      state: {chatId: 't1', inputDrafts: {t1: 'typed draft'}},
    });
    assert.strictEqual(other.inp.value, 'typed draft');
    assert.deepStrictEqual(browserCopy(other.win), kept('saved', ''));
    click(other.win, other.clearBtn);
    assert.deepStrictEqual(saves(other.posted), []);
    // Its own chat holds the text as edited since: that stays, and the
    // chat stays the one whose clear drops the saved text.
    const edited = pageWithChats(['t1'], 'saved', {
      stored: kept('saved', 't1'),
      state: {chatId: 't1', inputDrafts: {t1: 'saved, and more'}},
    });
    assert.strictEqual(edited.inp.value, 'saved, and more');
    click(edited.win, edited.clearBtn);
    assert.deepStrictEqual(saves(edited.posted), [cmd('', 1, 'saved')]);
    // The chat on screen is answering a question when the daemon speaks.
    const asked = makePage({stored: kept('saved', 't9')});
    sendTabsState(asked.win, [{tabId: 't1'}]);
    send(asked.win, {type: 'askUser', question: 'Deploy?', tabId: 't1'});
    connect(asked.win);
    fileHolds(asked.win, 'saved');
    assert.strictEqual(asked.inp.value, '');
    assert.deepStrictEqual(browserCopy(asked.win), kept('saved', ''));
    // No chat is left: a report stays on screen while its chat is
    // replaced, and the replacement closed, on another surface.
    const report = makePage();
    connect(report.win);
    send(report.win, {
      type: 'fileContent',
      path: '/ws/report.html',
      name: 'report.html',
      content: '<!DOCTYPE html><html><body>report</body></html>',
    });
    send(report.win, {
      type: 'tabs_state',
      tabs: [{tabId: 'b2', chatId: 'c2', title: 'b2', workDir: '/ws'}],
    });
    send(report.win, {type: 'tabs_state', tabs: []});
    assert.deepStrictEqual(
      Array.from(report.win._testApi.openTabs(), t => t.isContentTab),
      [true],
    );
    fileHolds(report.win, 'saved');
    assert.strictEqual(report.inp.value, '');
    assert.deepStrictEqual(browserCopy(report.win), kept('saved', ''));
  });

  await test('scenario 9: a browser that cannot keep the text fails the Save and claims nothing', () => {
    // No storage at all (an opaque origin).
    const blocked = pageWithChats(['t1'], '', {url: null});
    saveText(blocked, 'abc');
    assert.deepStrictEqual(saves(blocked.posted), []);
    assert.deepStrictEqual(saveToast(blocked.win), {
      error: true,
      message: NOT_KEPT,
    });
    // No room left: the text saved before stays the saved one.
    const full = pageWithChats(['t1'], 'old', {
      stored: kept('old', 't1'),
      beforeScripts(win) {
        win.localStorage.setItem('filler', 'x'.repeat(4900000));
      },
    });
    assert.strictEqual(full.inp.value, 'old');
    saveText(full, 'y'.repeat(300000));
    assert.deepStrictEqual(saves(full.posted), []);
    assert.deepStrictEqual(saveToast(full.win), {error: true, message: NOT_KEPT});
    assert.deepStrictEqual(browserCopy(full.win), kept('old', 't1'));
    // No room for the newer text of another browser: it is not offered,
    // and the older copy is not kept in its place.  Of an older copy
    // that names its save, only the name stays.
    for (const [rev, left] of [
      [undefined, null],
      ['r1', REMOVED],
    ]) {
      const stale = makePage({stored: {...kept('old', 't1'), rev}});
      connect(stale.win);
      sendTabsState(stale.win, [{tabId: 't1'}]);
      fillStorage(stale.win, 0);
      fileHolds(stale.win, 'a newer text of another browser');
      assert.deepStrictEqual(
        [stale.inp.value, browserCopy(stale.win)],
        ['', left],
      );
    }
    // No room to note the box of a text that is in none: not filled.
    const unplaced = makePage({stored: kept('T', '')});
    connect(unplaced.win);
    sendTabsState(unplaced.win, [{tabId: 't1'}]);
    fillStorage(unplaced.win, 0);
    fileHolds(unplaced.win, 'T');
    assert.deepStrictEqual(
      [unplaced.inp.value, browserCopy(unplaced.win)],
      ['', kept('T', '')],
    );
    // No room to note the daemon's answer: the save stays owed, and is
    // the saved text all the same.
    const exact = pageWithChats(['t1']);
    const owed = {text: 'T', tabId: 't1', owed: true, was: ['T']};
    // As stored, the copy names its save too: 36 characters.
    const stored = JSON.stringify({...owed, rev: '-'.repeat(36)});
    fillStorage(exact.win, KEY.length + stored.length);
    saveText(exact, 'T');
    answer(exact.win, newest(exact));
    assert.deepStrictEqual(saveToast(exact.win), {error: false, message: SAVED});
    assert.deepStrictEqual(browserCopy(exact.win), owed);
    click(exact.win, exact.clearBtn);
    assert.deepStrictEqual(newest(exact), cmd('', 2, 'T'));
  });

  await test('scenario 10: a removal spares the newer text of another browser, which is offered at the next load', () => {
    const a = savedPage(['t1'], 'T');
    // Browser B has a storage of its own.  It is offered the file's
    // text, and saves another one.
    const b = pageWithChats(['t1'], 'T');
    assert.strictEqual(b.inp.value, 'T');
    saveText(b, 'T2');
    assert.deepStrictEqual(saves(b.posted), [cmd('T2', 1, 'T', 'T2')]);
    // A sends its box.  Its removal names T only: the daemon keeps T2
    // and says so.
    click(a.win, a.sendBtn);
    assert.deepStrictEqual(newest(a), cmd('', 2, 'T'));
    answer(a.win, newest(a), 'T2');
    assert.deepStrictEqual(
      [browserCopy(a.win), a.inp.value, saveToast(a.win)],
      [kept('T2', ''), '', null],
    );
    // At A's next page load T2 is offered, in the chat on screen.
    const next = pageWithChats(['t1'], 'T2', {
      stored: a.win.localStorage.getItem(KEY),
    });
    assert.strictEqual(next.inp.value, 'T2');
    assert.deepStrictEqual(browserCopy(next.win), kept('T2', 't1'));
  });

  await test('scenario 11: only the answer to the newest save announces it', () => {
    const page = pageWithChats(['t1']);
    const {win, posted} = page;
    saveText(page, 'A');
    saveText(page, 'B');
    saveText(page, 'A');
    const [first, second, third] = saves(posted);
    assert.deepStrictEqual(
      [first, second, third],
      [cmd('A', 1, 'A'), cmd('B', 2, 'A', 'B'), cmd('A', 3, 'A', 'B')],
    );
    // The first answer names the same words but another save; a read
    // of the file comes late.
    answer(win, first);
    answer(win, second);
    fileHolds(win, 'B');
    assert.deepStrictEqual(
      [browserCopy(win), saveToast(win)],
      [{text: 'A', tabId: 't1', owed: true, was: ['A', 'B']}, null],
    );
    answer(win, third);
    assert.deepStrictEqual(
      [browserCopy(win), saveToast(win).message],
      [kept('A', 't1'), SAVED],
    );
    // The transport sent the command twice: its second answer is no news.
    toast(win, 'saved-input').remove();
    answer(win, third);
    assert.strictEqual(saveToast(win), null);
  });

  await test('a text over 1 MB is refused before it is kept or posted', () => {
    const page = pageWithChats(['t1']);
    const {win, posted} = page;
    // 1,000,001 bytes of ASCII; 500,001 two-byte characters.
    for (const text of ['a'.repeat(1000001), '\u00e9'.repeat(500001)]) {
      saveText(page, text);
      assert.deepStrictEqual(saves(posted), []);
      assert.strictEqual(browserCopy(win), null);
      assert.deepStrictEqual(saveToast(win), {error: true, message: TOO_LONG});
      toast(win, 'saved-input').remove();
    }
    saveText(page, 'a'.repeat(1000000));
    assert.strictEqual(saves(posted).length, 1);
    assert.strictEqual(saveToast(win), null);
  });

  await test('a write the daemon could not make is reported and posted again at the next connection', () => {
    const page = savedPage(['t1'], 'old');
    const {win, posted} = page;
    saveText(page, 'precious');
    const error = `The saved text in ${FILE} was not changed: [Errno 28] No space left on device`;
    answer(win, newest(page), 'old', error);
    assert.deepStrictEqual(saveToast(win), {error: true, message: error});
    const owed = {text: 'precious', tabId: 't1', owed: true, was: ['old', 'precious']};
    assert.deepStrictEqual(browserCopy(win), owed);
    disconnect(win);
    connect(win);
    assert.deepStrictEqual(newest(page), cmd('precious', 3, 'old', 'precious'));
    assert.strictEqual(saves(posted).length, 3);
    answer(win, newest(page));
    assert.deepStrictEqual(
      [browserCopy(win), saveToast(win).message],
      [kept('precious', 't1'), SAVED],
    );
  });

  await test('the saved text follows its draft to the chat that takes it', () => {
    // Saved in the boot tab and edited there, before the first
    // snapshot: the chat that takes the screen takes the draft and the
    // save.
    const page = makePage();
    const {win, inp, clearBtn} = page;
    connect(win);
    fileHolds(win, '');
    saveText(page, 'carry me');
    answer(win, newest(page));
    assert.strictEqual(browserCopy(win).tabId, win._testApi.getActiveTabId());
    typeInto(win, inp, 'carry me, edited');
    sendTabsState(win, [{tabId: 't1'}]);
    assert.deepStrictEqual(
      [inp.value, browserCopy(win).tabId],
      ['carry me, edited', 't1'],
    );
    // "+" from a chat that has a task: the new chat takes both.
    send(win, {type: 'setTaskText', text: 'a long task', tabId: 't1'});
    send(win, {type: 'status', running: true, tabId: 't1'});
    click(win, byId(win, 'new-chat-btn'));
    const fresh = win._testApi.getActiveTabId();
    assert.notStrictEqual(fresh, 't1');
    assert.deepStrictEqual(
      [inp.value, browserCopy(win).tabId],
      ['carry me, edited', fresh],
    );
    click(win, clearBtn);
    assert.deepStrictEqual(newest(page), cmd('', 2, 'carry me'));
    // "+" out of an answer box leaves the parked prompt, and its save,
    // with the chat that asks.
    saveText(page, 'stay here');
    send(win, {type: 'setTaskText', text: 'another task', tabId: fresh});
    send(win, {type: 'askUser', question: 'Deploy?', tabId: fresh});
    click(win, byId(win, 'new-chat-btn'));
    assert.notStrictEqual(win._testApi.getActiveTabId(), fresh);
    assert.deepStrictEqual([inp.value, browserCopy(win).tabId], ['', fresh]);
  });

  await test('what the browser stores that is no copy counts as nothing saved', () => {
    for (const stored of ['not json', '{"text": 5}', '{"text": "half"}']) {
      const page = makePage({stored});
      sendTabsState(page.win, [{tabId: 't1'}]);
      connect(page.win);
      assert.strictEqual(ofType(page.posted, 'getInputDraft').length, 1, stored);
      saveText(page, 'new');
      assert.deepStrictEqual(saves(page.posted), [cmd('new', 1, 'new')], stored);
    }
  });

  await test('two windows of one browser: the answer to an older save leaves the other window\u2019s newer save or removal owed, and a read\u2019s answer a save confirmed since', () => {
    // The browser's copy is one for all its windows.
    const a = pageWithChats(['t1']);
    const shared = {
      beforeScripts(win) {
        Object.defineProperty(win, 'localStorage', {value: a.win.localStorage});
      },
    };
    const b = pageWithChats(['t1'], '', shared);
    // A's save is on its way when B, its connection lost, saves another
    // text: A's answer is not B's, whose text is written at B's return.
    saveText(a, 'older');
    disconnect(b.win);
    saveText(b, 'newer');
    answer(a.win, newest(a));
    assert.deepStrictEqual(
      [browserCopy(a.win), saveToast(a.win)],
      [{text: 'newer', tabId: 't1', owed: true, was: ['older', 'newer']}, null],
    );
    connect(b.win);
    assert.deepStrictEqual(newest(b), cmd('newer', 2, 'older', 'newer'));
    answer(b.win, newest(b));
    assert.deepStrictEqual(browserCopy(a.win), kept('newer', 't1'));
    // Likewise a removal.  A clears the chat's box; before the daemon's
    // answer is here, B saves another text and, its connection lost,
    // clears its box: A's answer is not that of B's removal.
    click(a.win, a.clearBtn);
    const removal = newest(a);
    saveText(b, 'again');
    answer(b.win, newest(b));
    disconnect(b.win);
    click(b.win, b.clearBtn);
    answer(a.win, removal);
    const owed = {text: '', tabId: '', owed: true, was: ['again']};
    assert.deepStrictEqual(removal, cmd('', 2, 'newer'));
    assert.deepStrictEqual(browserCopy(a.win), owed);
    connect(b.win);
    assert.deepStrictEqual(newest(b), cmd('', 5, 'again'));
    answer(b.win, newest(b));
    assert.deepStrictEqual(browserCopy(a.win), REMOVED);
    // Likewise a read.  A window starts and asks what the file holds;
    // before the answer is here, B saves another text and the daemon
    // confirms it: the answer, of the file as it was, is not of that
    // text.  The window asks again at its next connection.
    saveText(b, 'old');
    answer(b.win, newest(b));
    const c = makePage(shared);
    connect(c.win);
    sendTabsState(c.win, [{tabId: 't1'}]);
    saveText(b, 'fresh');
    answer(b.win, newest(b));
    fileHolds(c.win, 'old');
    assert.deepStrictEqual(
      [browserCopy(c.win), c.inp.value],
      [kept('fresh', 't1'), ''],
    );
    disconnect(c.win);
    connect(c.win);
    assert.strictEqual(ofType(c.posted, 'getInputDraft').length, 2);
    fileHolds(c.win, 'fresh');
    assert.deepStrictEqual(
      [browserCopy(c.win), c.inp.value],
      [kept('fresh', 't1'), 'fresh'],
    );
    // Nor is it of a text removed since.  A window of a browser that
    // keeps no copy asks; before the answer is here, another window
    // saves a text and clears its box, and the daemon confirms both.
    // The copy is as empty as it was when the window asked, but it
    // names the removal: the answer puts the removed text nowhere.
    const d = makePage();
    connect(d.win);
    sendTabsState(d.win, [{tabId: 't1'}]);
    const e = pageWithChats(['t1'], '', {
      beforeScripts(win) {
        Object.defineProperty(win, 'localStorage', {value: d.win.localStorage});
      },
    });
    saveText(e, 'gone');
    answer(e.win, newest(e));
    click(e.win, e.clearBtn);
    answer(e.win, newest(e));
    fileHolds(d.win, 'gone');
    assert.deepStrictEqual([browserCopy(d.win), d.inp.value], [REMOVED, '']);
  });

  report('composerSaveInput');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
