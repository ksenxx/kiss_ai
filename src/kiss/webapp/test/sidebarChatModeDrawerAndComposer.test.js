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
// * Stop, Share and a voice submit on the file screen act on the chat
//   the user swiped in from, whichever chat owns the file.

'use strict';

const assert = require('assert');

const {
  makePage: makeChatPage,
  send,
  loadMainCss,
  fakeIntervals,
  fakeMatchMedia,
  click,
  byId,
  ofType,
  plain,
  withTeardown,
} = require('./simplify2_harness');
const {installTokenisedCss} = require('./designTokens');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  POLL_MS,
  drawerOpen,
  overlayOpen,
  pollCount,
  lastPoll,
  replyToPoll,
} = require('./remote_drawer_helpers');
const {chatTabs} = require('./history_helpers');
const {swipeLeftToFileScreen} = require('./subagent_tab_helpers');
const {
  VOICE_CFG,
  voicePageState,
  stubVoiceSurface,
  startMic,
  speakRound,
} = require('./voice_page_helpers');
const {trackWindow, closeWindows} = require('./workdir_helpers');

const {test: runTest, report} = makeRunner();
// Each test's windows are closed once it has settled.
const test = withTeardown(runTest, closeWindows);

/**
 * The remote page.  With *voice* (a voicePageState record) the real
 * voice.js runs too, over a stubbed microphone, its mic remembered ON.
 */
function makePage(desktop = false, voice = null) {
  let timers;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js', 'mobileSwipe.js'].concat(
      voice ? ['voice.js'] : [],
    ),
    // Named for sidebarChatModeDrawerAndComposer.coverage.js.
    sourceUrl: 'filechat-main.js',
    beforeScripts(win) {
      if (voice) {
        stubVoiceSurface(win, voice);
        win.__VOICE__ = Object.assign({}, VOICE_CFG);
        win.localStorage.setItem('kissVoiceEnabled', '1');
      }
      // A phone-sized window (no media query matches, so the task-info
      // panel is a drawer and the history sidebar slides in from the
      // burger), or a desktop-sized one when *desktop* is true.
      fakeMatchMedia(win, desktop);
      timers = fakeIntervals(win);
    },
  });
  trackWindow(win);
  if (voice) Object.assign(voice, {win, posted});
  return {win, posted, tick: timers.tick, intervals: timers.intervals};
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
  return shown;
}

function taskPanelText(win) {
  return win.document.querySelector('#output .task-panel-text').textContent;
}

/**
 * The chat on screen opens the file *name*; the user moves on to a new
 * chat and swipes left: the first chat's file is on the file screen,
 * reached from the second.  *beforeSwipe(owner, chat)* runs on the
 * second chat's screen.
 */
function swipeToFileOfAnotherChat(win, name, beforeSwipe) {
  const api = win._testApi;
  // Past the launch, a running status no longer moves the screen.
  api.endLaunch();
  const owner = api.getActiveTabId();
  send(win, {
    type: 'fileContent',
    tabId: owner,
    name: name,
    path: '/tmp/' + name,
    content: name,
  });
  const file = api.getActiveTabId();
  assert.notStrictEqual(file, owner, 'the file tab is in front');
  api.createNewTab();
  const chat = api.getActiveTabId();
  if (beforeSwipe) beforeSwipe(owner, chat);
  swipeLeftToFileScreen(win);
  assert.strictEqual(api.getActiveTabId(), file, 'the swipe shows the file');
  return {owner, chat, file};
}

/** The tabs the page asked the daemon to stop. */
function stopsPosted(posted) {
  return ofType(posted, 'stop').map(m => m.tabId);
}

/** The daemon's tab registry now lists the chat tabs *ids*. */
function registryLists(win, ids) {
  send(win, {
    type: 'tabs_state',
    tabs: ids.map(id => ({tabId: id, chatId: '', title: 'new chat'})),
  });
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
      // An open drawer with no running task has nothing to poll for.
      click(win, byId(win, 'meta-drawer-btn'));
      wv.tick(POLL_MS * 2);
      assert.strictEqual(pollCount(wv), 0, 'no task, no poll');
      click(win, byId(win, 'meta-close'));
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
      replyToPoll(win, poll, {
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
    assert.strictEqual(resume.chatId, 'chat-far');
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
      // Stop is NOT hidden by the content-tab CSS: while the chat
      // behind the file runs, its inline display keeps the button
      // reachable to stop the task from the file view.
      const stop = win.document.getElementById('stop-btn');
      stop.style.display = 'flex';
      assert.strictEqual(
        display(win, 'stop-btn'),
        'flex',
        'Stop stays reachable when the chat behind the file runs',
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
    'Stop on the file screen acts on the chat swiped in from, not on ' +
      "the file's owner: it shows, stops and pulses for that chat's run",
    () => {
      const {win, posted} = makePage();
      const api = win._testApi;
      const stop = byId(win, 'stop-btn');
      // The owner runs; the chat the user swipes in from is idle.
      const {owner, chat, file} = swipeToFileOfAnotherChat(
        win,
        'notes_stop.txt',
        ownerId => send(win, {type: 'status', running: true, tabId: ownerId}),
      );
      assert.strictEqual(
        stop.style.display,
        'none',
        "no Stop for the owner's run: the chat swiped in from is idle",
      );
      send(win, {type: 'status', running: true, tabId: chat});
      assert.strictEqual(api.getActiveTabId(), file, 'the file stays');
      assert.strictEqual(
        stop.style.display,
        'flex',
        'Stop shows once the chat swiped in from runs',
      );
      posted.length = 0;
      click(win, stop);
      assert.deepStrictEqual(
        stopsPosted(posted),
        [chat],
        'the stop lands on the chat swiped in from',
      );
      assert.ok(
        stop.classList.contains('stopping'),
        'its pending stop pulses on the file screen',
      );
      send(win, {type: 'status', running: false, tabId: chat});
      assert.strictEqual(stop.style.display, 'none', 'and goes with its run');

      // That chat closes on another surface: the screen is handed to
      // the file's owner (the chat a swipe back now returns to), whose
      // run Stop then shows and stops.
      registryLists(win, [owner, chat]);
      registryLists(win, [owner]);
      assert.strictEqual(api.getActiveTabId(), file, 'the file stays');
      assert.strictEqual(
        stop.style.display,
        'flex',
        'Stop shows the run of the chat the screen was handed to',
      );
      posted.length = 0;
      click(win, stop);
      assert.deepStrictEqual(stopsPosted(posted), [owner]);
      send(win, {type: 'status', running: false, tabId: owner});
      assert.strictEqual(stop.style.display, 'none', 'until that run ends');
      assert.ok(
        !stop.classList.contains('stopping'),
        'and its pending stop ends with the run',
      );
    },
  );

  await test(
    "the file screen's Browser button stays with the file's own chat; " +
      'with no chat left behind the file there is nothing to stop or share',
    () => {
      const {win, posted} = makePage();
      const browser = byId(win, 'browser-btn');
      // The file's own chat works on the machine `box`, the chat the
      // user swipes in from on this one.
      const {owner, chat, file} = swipeToFileOfAnotherChat(
        win,
        'notes_box.txt',
        ownerId => {
          send(win, {
            type: 'machines',
            machines: [
              {name: 'here', local: true, status: 'reachable'},
              {name: 'box', label: 'Box', status: 'reachable'},
            ],
            checkedAt: 1,
          });
          send(win, {type: 'tabMachine', tabId: ownerId, machine: 'box'});
        },
      );
      click(win, browser);
      assert.deepStrictEqual(
        plain(ofType(posted, 'browserOpen')),
        [{type: 'browserOpen', url: '', machine: 'box'}],
        "the page opens where the file's own chat works",
      );

      // Both chats close on another surface while they run.
      send(win, {type: 'status', running: true, tabId: owner});
      send(win, {type: 'status', running: true, tabId: chat});
      registryLists(win, [owner, chat]);
      registryLists(win, []);
      assert.strictEqual(win._testApi.getActiveTabId(), file, 'the file stays');
      assert.strictEqual(chatTabs(win).length, 0, 'setup: no chat is left');
      assert.strictEqual(byId(win, 'stop-btn').style.display, 'none');
      posted.length = 0;
      click(win, byId(win, 'share-btn'));
      assert.deepStrictEqual(ofType(posted, 'shareChatTasks'), []);
      assert.ok(
        Array.from(win.document.querySelectorAll('.kiss-notification')).some(
          el => el.textContent.includes('this tab has no chat to share'),
        ),
        'Share says there is no chat to share',
      );
      click(win, browser);
      assert.deepStrictEqual(
        plain(ofType(posted, 'browserOpen')),
        [{type: 'browserOpen', url: ''}],
        'the page opens on this machine',
      );
    },
  );

  await test(
    'Stop on a file opened FROM a file view, and the pending pulse, ' +
      'follow the chat the file screen was reached from',
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
      // B -> A -> chat. The screen still stands for the chat it was
      // reached from.
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
        'the stop lands on the chat the file screen was reached from',
      );
      assert.ok(
        stop.classList.contains('stopping'),
        'the pending stop pulses on the file tab too',
      );
    },
  );

  await test(
    'Share chat clicked on the file screen exports the chat swiped in ' +
      "from, not the file's owner: that chat comes back on screen and " +
      'the share names its id',
    () => {
      const {win, posted} = makePage();
      const {chat} = swipeToFileOfAnotherChat(win, 'notes_share.txt');
      posted.length = 0;
      click(win, byId(win, 'share-btn'));
      assert.strictEqual(
        win._testApi.getActiveTabId(),
        chat,
        'the chat swiped in from is back on screen (the export splices ' +
          'the LIVE transcript, which only exists there)',
      );
      const share = posted.find(m => m.type === 'shareChatTasks');
      assert.ok(share, "the click asks for the chat's persisted tasks");
      assert.strictEqual(share.tabId, chat);
      assert.strictEqual(share.chatId, chat);
    },
  );

  await test(
    'words spoken on the file screen join the draft of the chat swiped ' +
      "in from and are sent to it, never to the file's owner",
    async () => {
      const voice = voicePageState({});
      const {win, posted} = makePage(false, voice);
      const api = win._testApi;
      const input = byId(win, 'task-input');
      input.value = 'draft of the owner QX1';
      const {owner, chat} = swipeToFileOfAnotherChat(
        win,
        'notes_voice.txt',
        () => {
          input.value = 'draft of the chat QX1';
        },
      );
      await startMic(voice);
      const round = speakRound(voice);
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'spoken QX1', round});
      assert.deepStrictEqual(
        ofType(posted, 'submit').map(m => [m.tabId, m.prompt]),
        [[chat, 'draft of the chat QX1 spoken QX1']],
      );
      assert.strictEqual(
        api.getActiveTabId(),
        chat,
        'the chat swiped in from is back on screen',
      );
      api.switchToTab(owner);
      assert.strictEqual(
        input.value,
        'draft of the owner QX1',
        "the file's owner keeps its draft",
      );
    },
  );

  report('sidebarChatModeDrawerAndComposer');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
