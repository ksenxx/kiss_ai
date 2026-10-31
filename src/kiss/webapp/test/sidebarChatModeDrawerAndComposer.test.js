// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the remote webapp page on a PHONE-SIZED window
// (media/chat.html + media/main.js with body.remote-chat and a
// `(min-width: 900px)` media query that does not match):
//
// * the task-info panel is a right-hand drawer: the toggle / close
//   button / backdrop / Escape flows work, the closed drawer is inert,
//   and the 5s getTaskUpdate poll runs only while the drawer is open
//   and a task runs; the #meta-info refresh button posts a
//   getTaskUpdate with refresh:true and mirrors the agent's running
//   state;
// * on a desktop-sized window the burger and the drawer toggle go
//   (remote-codex.css): the history sidebar and the task-info panel
//   are docked;
// * a history row click switches to the chat's tab, resumes the chat
//   in a fresh tab, or shows the task text read-only;
// * body.content-tab-open (a file/page tab in front) keeps the
//   composer's button row (+ and ...) while hiding the text box and
//   the chat-only controls (Inject promptlet, model picker, Send);
// * Stop, Share and a voice submit on a content tab act on the chat
//   that owns it.

'use strict';

const assert = require('assert');

const POLL_MS = 5000;

const {
  makePage: makeChatPage,
  send,
  loadMainCss,
  fakeIntervals,
  fakeMatchMedia,
  click,
  byId,
} = require('./simplify2_harness');
const {installTokenisedCss} = require('./designTokens');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  drawerOpen,
  overlayOpen,
  pollCount,
  lastPoll,
} = require('./remote_drawer_helpers');

const {test, report} = makeRunner();

function makePage(desktop = false) {
  let timers;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    sourceUrl: 'remote-mobile-main.js',
    beforeScripts(win) {
      // A phone-sized window (no media query matches, so the task-info
      // panel is a drawer and the history sidebar slides in from the
      // burger), or a desktop-sized one when *desktop* is true.
      fakeMatchMedia(win, desktop);
      timers = fakeIntervals(win);
    },
  });
  return {win, posted, tick: timers.tick, intervals: timers.intervals};
}

/** A taskUpdate reply answering *poll*, with the given overrides. */
function reply(win, poll, fields) {
  send(
    win,
    Object.assign(
      {
        type: 'taskUpdate',
        tabId: poll.tabId,
        token: poll.token,
        taskId: 'task-1',
        exists: true,
        sig: '',
        content: '',
        error: '',
        running: false,
        cost: 0,
        updatedAt: 0,
      },
      fields,
    ),
  );
}

function chatTabs(win) {
  // The chat tabs the window holds.  They have no row of their own
  // (the Chats panel picks them; the strip only lists the group of the
  // chat on screen), so the tab records are the source.
  return Array.from(win._testApi.openTabs()).filter(
    t => !t.isContentTab && !t.isSubagentTab,
  );
}

function display(win, id) {
  return win.getComputedStyle(win.document.getElementById(id)).display;
}

/**
 * Open the history sidebar from the burger, answer its getHistory with
 * *sessions* and return the rendered task rows.
 */
function showHistory(win, posted, sessions) {
  click(win, byId(win, 'menu-btn'));
  const req = posted.filter(m => m.type === 'getHistory').pop();
  assert.ok(req, 'opening the sidebar asks for the history');
  send(win, {
    type: 'history',
    generation: req.generation,
    offset: 0,
    sessions: sessions,
  });
  const rows = Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item'),
  );
  assert.strictEqual(rows.length, sessions.length, 'every session has a row');
  return rows;
}

/**
 * The computed `display` of the burger and the task-info drawer toggle
 * on a desktop- or phone-sized window styled by main.css and
 * remote-codex.css.  jsdom never matches a width media feature, so the
 * desktop window installs remote-codex.css's `(width >= 900px)` block
 * as `@media screen`.
 */
function burgerAndToggleDisplay(desktop) {
  const {win} = makePage(desktop);
  loadMainCss(win);
  installTokenisedCss(win, 'remote-codex.css', {
    applyMedia: desktop ? /\(width >= 900px\)/ : null,
  });
  const shown = [display(win, 'menu-btn'), display(win, 'meta-drawer-btn')];
  win.close();
  return shown;
}

function taskPanelText(win) {
  return win.document.querySelector('#output .task-panel-text').textContent;
}

async function main() {
  await test(
    'a desktop-sized window hides the burger and the drawer toggle ' +
      '(remote-codex.css): the sidebar and the task-info panel are docked',
    () => {
      assert.deepStrictEqual(
        burgerAndToggleDisplay(true),
        ['none', 'none'],
        'the docked panels need neither button',
      );
      const phone = burgerAndToggleDisplay(false);
      assert.ok(!phone.includes('none'), 'a phone keeps both buttons: ' + phone);
    },
  );

  await test(
    'the drawer opens/closes from the button, close button, backdrop ' +
      'and Escape; closed it is inert',
    () => {
      const {win} = makePage();
      const btn = win.document.getElementById('meta-drawer-btn');
      const panel = win.document.getElementById('meta-panel');
      assert.ok(btn, 'the toggle exists in the composer footer');
      assert.strictEqual(
        btn.parentElement.id,
        'footer-tools',
        'the toggle is a composer footer icon (there is no main tab row)',
      );
      assert.strictEqual(
        btn.previousElementSibling && btn.previousElementSibling.id,
        'menu-btn',
        'it sits right after the chats burger',
      );
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'false');
      assert.ok(!drawerOpen(win));
      assert.ok(
        panel.hasAttribute('inert'),
        'the drawer starts inert: its off-screen close button must not ' +
          'sit in the tab order before the drawer was ever opened',
      );
      click(win, byId(win, 'meta-drawer-btn'));
      assert.ok(drawerOpen(win), 'button opens the drawer');
      assert.ok(overlayOpen(win), 'the backdrop opens with the drawer');
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'true');
      assert.ok(!panel.hasAttribute('inert'), 'an open drawer is usable');
      click(win, byId(win, 'meta-close'));
      assert.ok(!drawerOpen(win), 'the close button dismisses the drawer');
      assert.ok(!overlayOpen(win));
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'false');
      assert.ok(
        panel.hasAttribute('inert'),
        'the closed drawer leaves the keyboard tab order',
      );
      click(win, byId(win, 'meta-drawer-btn'));
      assert.ok(drawerOpen(win));
      click(win, byId(win, 'meta-overlay'));
      assert.ok(!drawerOpen(win), 'the backdrop dismisses the drawer');
      assert.ok(!overlayOpen(win));
      click(win, byId(win, 'meta-drawer-btn'));
      win.document.dispatchEvent(
        new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
      );
      assert.ok(!drawerOpen(win), 'Escape dismisses the drawer');
    },
  );

  await test(
    'getTaskUpdate polls every 5s only while the drawer is open and a ' +
      'task runs, and the reply renders in the drawer',
    () => {
      const wv = makePage();
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      wv.tick(POLL_MS * 2);
      assert.strictEqual(pollCount(wv), 0, 'no task, no poll');
      send(win, {type: 'status', running: true});
      wv.tick(POLL_MS * 2);
      assert.strictEqual(pollCount(wv), 0, 'a closed drawer must not poll');
      click(win, byId(win, 'meta-drawer-btn'));
      assert.strictEqual(pollCount(wv), 1, 'opening the drawer polls at once');
      assert.ok(
        wv.intervals.some(iv => iv.ms === POLL_MS),
        'the poll timer ticks every 5000 ms',
      );
      const poll = lastPoll(wv);
      assert.ok(poll.tabId, 'the poll targets the visible chat tab');
      wv.tick(POLL_MS - 1);
      assert.strictEqual(pollCount(wv), 1, 'no poll before the interval');
      wv.tick(1);
      assert.strictEqual(pollCount(wv), 2, 'the timer polls at 5000 ms');

      // The poll's shape and the reply handling (sig, unchanged,
      // sanitising, exists:false, the refresh button) are the same code
      // as on the docked panel: remoteMetaPanelInfo.test.js.
      reply(win, poll, {
        sig: '5:9',
        content: 'working on **it**\n',
        updatedAt: Date.now(),
        cost: 0.12,
      });
      const info = win.document.getElementById('meta-info');
      const content = win.document.getElementById('meta-info-content');
      assert.ok(info.classList.contains('visible'));
      assert.ok(
        content.innerHTML.includes('<strong>it</strong>'),
        'the report renders in the drawer',
      );

      click(win, byId(win, 'meta-close'));
      const count = pollCount(wv);
      wv.tick(POLL_MS * 3);
      assert.strictEqual(
        pollCount(wv),
        count,
        'closing the drawer stops the poll',
      );
      assert.ok(!wv.intervals.some(iv => iv.ms === POLL_MS));
    },
  );

  await test('the refresh button in the drawer posts getTaskUpdate refresh:true', () => {
    const wv = makePage();
    const win = wv.win;
    send(win, {type: 'status', running: true});
    click(win, byId(win, 'meta-drawer-btn'));
    const btn = win.document.getElementById('meta-info-refresh');
    assert.ok(btn, 'the refresh button sits in the subpanel header');
    assert.strictEqual(
      btn.parentElement.parentElement.id,
      'meta-info',
      'the button belongs to the task-update subpanel',
    );
    const before = pollCount(wv);
    click(win, byId(win, 'meta-info-refresh'));
    assert.strictEqual(pollCount(wv), before + 1, 'the click polls at once');
    assert.strictEqual(lastPoll(wv).refresh, true, 'the click asks for a run');
  });

  // "a history row click switches to the tab already bound to the chat"
  // lives in historyClickNoInputCopy.test.js; the closeTab the
  // retired idle chat posts is pinned by historyMachineFilter.test.js.

  await test('a history row click resumes an unopened chat in a fresh tab', () => {
    const {win, posted} = makePage();
    const bootTab = win._testApi.getActiveTabId();
    const rows = showHistory(win, posted, [
      {
        id: 'chat-far',
        task_id: 7,
        title: 'A chat from another day',
        preview: 'A chat from another day',
        has_events: true,
        ts: Date.now() / 1000,
      },
    ]);
    rows[0].dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
    const fresh = win._testApi.getActiveTabId();
    assert.notStrictEqual(fresh, bootTab, 'the chat opens in a fresh tab');
    assert.strictEqual(
      chatTabs(win).length,
      1,
      'the idle boot chat left behind is retired',
    );
    const resume = posted.filter(m => m.type === 'resumeSession').pop();
    assert.ok(resume, 'the fresh tab resumes the chat');
    assert.strictEqual(resume.id, 'chat-far');
    assert.strictEqual(resume.taskId, 7);
    assert.strictEqual(
      taskPanelText(win),
      'A chat from another day',
      'the fresh transcript opens with the clicked task',
    );
  });

  await test('a history row with nothing to resume shows the text read-only', () => {
    const {win, posted} = makePage();
    const resumesBefore = posted.filter(m => m.type === 'resumeSession').length;
    // A task that never persisted an event and is not running.
    const rows = showHistory(win, posted, [
      {
        id: 'chat-empty',
        task_id: 3,
        title: 'An eventless task',
        preview: 'An eventless task',
        has_events: false,
        is_running: false,
        ts: Date.now() / 1000,
      },
    ]);
    rows[0].dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
    assert.strictEqual(
      posted.filter(m => m.type === 'resumeSession').length,
      resumesBefore,
      'nothing to resume',
    );
    assert.strictEqual(taskPanelText(win), 'An eventless task');
    assert.strictEqual(
      win.document.getElementById('welcome').style.display,
      'none',
      'the task text replaces the welcome screen',
    );
  });

  await test(
    'a content tab keeps the + and ... buttons but hides the text box, ' +
      'Inject promptlet, model picker and Send (main.css)',
    () => {
      const {win} = makePage();
      loadMainCss(win);
      assert.notStrictEqual(display(win, 'input-wrap'), 'none');
      assert.notStrictEqual(display(win, 'tricks-btn'), 'none');
      win.document.body.classList.add('content-tab-open');
      assert.notStrictEqual(
        display(win, 'input-area'),
        'none',
        'the composer area stays',
      );
      assert.notStrictEqual(
        display(win, 'input-footer'),
        'none',
        'the button row stays',
      );
      assert.notStrictEqual(display(win, 'new-chat-btn'), 'none', '+ stays');
      assert.notStrictEqual(display(win, 'more-btn'), 'none', '... stays');
      assert.strictEqual(display(win, 'input-wrap'), 'none', 'no text box');
      assert.strictEqual(display(win, 'file-chips'), 'none');
      assert.strictEqual(display(win, 'input-drawer-btn'), 'none');
      assert.strictEqual(display(win, 'autocomplete'), 'none');
      assert.strictEqual(
        display(win, 'tricks-btn'),
        'none',
        'Inject promptlet hides',
      );
      assert.strictEqual(
        display(win, 'model-picker'),
        'none',
        'the model picker hides',
      );
      assert.strictEqual(display(win, 'send-btn'), 'none', 'Send hides');
      // Stop is NOT hidden by the content-tab CSS: while the owning
      // chat runs, its inline display keeps the button reachable to
      // stop the task from the file view.
      const stop = win.document.getElementById('stop-btn');
      stop.style.display = 'flex';
      assert.strictEqual(
        display(win, 'stop-btn'),
        'flex',
        'Stop stays reachable when the owner chat runs',
      );
      stop.style.display = 'none';
      // A composer hidden with an INLINE display:none (a finished
      // sub-agent tab, a worktree action bar) must still show the
      // button row on a content tab.
      const container = win.document.getElementById('input-container');
      container.style.display = 'none';
      assert.notStrictEqual(
        display(win, 'input-container'),
        'none',
        'the row survives an inline-hidden composer',
      );
      assert.notStrictEqual(display(win, 'new-chat-btn'), 'none');
      win.document.body.classList.remove('content-tab-open');
      assert.strictEqual(
        display(win, 'input-container'),
        'none',
        'the inline hide applies again once the content tab closes',
      );
      container.style.display = '';
      assert.notStrictEqual(display(win, 'input-wrap'), 'none');
    },
  );

  await test(
    'Stop clicked on a content tab stops the OWNING chat, and the ' +
      "button mirrors that chat's running state",
    () => {
      const {win, posted} = makePage();
      const api = win._testApi;
      const chatId = api.getActiveTabId();
      send(win, {type: 'status', running: true, tabId: chatId});
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'notes_stop.txt',
        path: '/tmp/notes_stop.txt',
        content: 'stop target body',
      });
      const contentId = api.getActiveTabId();
      assert.notStrictEqual(contentId, chatId, 'the file tab is in front');
      assert.strictEqual(
        win.document.getElementById('stop-btn').style.display,
        'flex',
        'Stop shows on the file tab while the owner chat runs',
      );
      posted.length = 0;
      click(win, byId(win, 'stop-btn'));
      const stop = posted.find(m => m.type === 'stop');
      assert.ok(stop, 'the click posts a stop');
      assert.strictEqual(
        stop.tabId,
        chatId,
        'the stop lands on the owning chat, not on the file view id',
      );
      // The owner idle: a later content-tab show hides Stop again.
      send(win, {type: 'status', running: false, tabId: chatId});
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'notes_stop2.txt',
        path: '/tmp/notes_stop2.txt',
        content: 'second file',
      });
      assert.strictEqual(
        win.document.getElementById('stop-btn').style.display,
        'none',
        'no Stop on a file tab whose owner chat is idle',
      );
    },
  );

  await test(
    'Stop on a file opened FROM a file view, and the pending pulse, ' +
      'follow the chat at the root of the owner chain',
    () => {
      const {win, posted} = makePage();
      const api = win._testApi;
      const chatId = api.getActiveTabId();
      send(win, {type: 'status', running: true, tabId: chatId});
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'nested_a.txt',
        path: '/tmp/nested_a.txt',
        content: 'file A',
      });
      const fileA = api.getActiveTabId();
      assert.notStrictEqual(fileA, chatId);
      // Opening a file FROM the file view chains the ownership:
      // B -> A -> chat. Every chat-scoped action lands on the chat.
      send(win, {
        type: 'fileContent',
        tabId: fileA,
        name: 'nested_b.txt',
        path: '/tmp/nested_b.txt',
        content: 'file B',
      });
      assert.notStrictEqual(api.getActiveTabId(), fileA);
      const stop = win.document.getElementById('stop-btn');
      assert.strictEqual(
        stop.style.display,
        'flex',
        'Stop shows two hops away from the running chat',
      );
      posted.length = 0;
      click(win, byId(win, 'stop-btn'));
      const stopMsg = posted.find(m => m.type === 'stop');
      assert.ok(stopMsg, 'the click posts a stop');
      assert.strictEqual(
        stopMsg.tabId,
        chatId,
        'the stop lands on the chat at the root of the owner chain',
      );
      assert.ok(
        stop.classList.contains('stopping'),
        'the pending stop pulses on the file tab too',
      );
      win.close();
    },
  );

  await test(
    'Share chat clicked on a content tab exports the OWNING chat: the ' +
      'chat comes back on screen and the share names its id',
    () => {
      const {win, posted} = makePage();
      const api = win._testApi;
      const chatId = api.getActiveTabId();
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'notes_share.txt',
        path: '/tmp/notes_share.txt',
        content: 'share target body',
      });
      assert.notStrictEqual(api.getActiveTabId(), chatId);
      posted.length = 0;
      click(win, byId(win, 'share-btn'));
      assert.strictEqual(
        api.getActiveTabId(),
        chatId,
        'the owning chat is back on screen (the export splices the ' +
          'LIVE transcript, which only exists there)',
      );
      const share = posted.find(m => m.type === 'shareChatTasks');
      assert.ok(share, "the click asks for the chat's persisted tasks");
      assert.strictEqual(share.tabId, chatId);
      assert.strictEqual(share.chatId, chatId);
    },
  );

  await test(
    'a voice submit landing on a content tab posts under the OWNING ' +
      'chat, never under the file view',
    () => {
      const {win, posted} = makePage();
      const api = win._testApi;
      const chatId = api.getActiveTabId();
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'notes_voice.txt',
        path: '/tmp/notes_voice.txt',
        content: 'voice target body',
      });
      assert.notStrictEqual(api.getActiveTabId(), chatId);
      assert.strictEqual(
        win.kissVoiceOwner().tabId,
        chatId,
        'an utterance started on the file tab is owned by the chat',
      );
      win.document.getElementById('task-input').value = 'voice prompt QX1';
      posted.length = 0;
      win.dispatchEvent(new win.CustomEvent('kiss-voice-submit'));
      assert.strictEqual(
        api.getActiveTabId(),
        chatId,
        'the submit brings the conversation back on screen',
      );
      const submit = posted.find(m => m.type === 'submit');
      assert.ok(submit, 'the prompt is submitted');
      assert.strictEqual(submit.tabId, chatId);
      assert.strictEqual(submit.prompt, 'voice prompt QX1');
    },
  );

  report('sidebarChatModeDrawerAndComposer');
  process.exit(0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
