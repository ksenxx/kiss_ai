// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) regressions for the 2026-11-01 audit of
// media/main.js (W2, lines 7600-15200): a reply, or a menu item left
// open, must act on the form, clipboard, repository and chat it was
// made for, not on whatever is current when it lands.
//
// * F1: the acknowledgment of a Schedule / Machines form's save closes
//   or re-arms the form instance that sent it.  A form cancelled with
//   its save in flight and the one saved in its place both await a
//   reply: the first reply answers the first (the daemon echoes the
//   action and the job / machine it named), so a success for the old
//   form keeps the new one waiting, and the new one's refusal brings
//   its draft back,
// * F2: a `move` reply consumes the Cut it answers only on the machine
//   it ran on: the same path Cut on another machine survives,
// * F3: a Source Control commit menu left open while the sidebar moves
//   to another machine or working directory still acts on the
//   repository it was opened on,
// * F4: a New File reply that lands after a chat switch opens the
//   editor for the chat that created the file, and a closed creator
//   gets no editor at all.
// * Round 2 (F3): a save keeps its owner across machine changes.  The
//   subpanels moved away and back before the reply came: the retained
//   draft, edited and saved again, has a second save in flight with
//   the same action and name, and the reply to the first (its late
//   success) must not close the form the second one re-armed -- the
//   second's refusal then brings its draft back.  Saves are keyed by
//   the machine they went to, and a reply for a machine no longer
//   shown retires its save without touching the form on show.

'use strict';

const assert = require('assert');
const {
  WD,
  SHA_A,
  makePage: makeHarnessPage,
  makeRunner,
  send,
  click,
  key,
  byId,
  showSection,
  ofType,
  menuItem,
  toast,
  toastButton,
  toastInput,
  rightClick,
  openExplorer,
  explorerRow,
  openScm,
  runMenuItem,
  pinWorkspace,
} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();
const pages = [];

function makePage() {
  const page = makeHarnessPage();
  pages.push(page.win);
  return page;
}

function job(id) {
  return {
    id,
    name: 'Job ' + id,
    schedule: 'every 5m',
    what: 'draft ' + id,
    kind: 'prompt',
    enabled: true,
    running: false,
  };
}

function saveBtn(form) {
  return form.querySelector('.sidebar-inline-form-save');
}

function submit(win, form) {
  click(win, saveBtn(form));
}

function cancel(win, form) {
  click(win, form.querySelector('.sidebar-confirm-no'));
}

function rows(win, id) {
  return byId(win, id).querySelectorAll('.sidebar-panel-row');
}

function setupMachines(win) {
  send(win, {
    type: 'machines',
    machines: [
      {name: 'local', local: true, status: 'reachable'},
      {name: 'gpu-box', status: 'reachable'},
    ],
  });
}

/** Bind the active tab to gpu-box (the sidebar follows) and answer its listing. */
function bindGpuBox(win, posted, entries) {
  send(win, {
    type: 'tabMachine',
    tabId: win._testApi.getActiveTabId(),
    machine: 'gpu-box',
  });
  send(win, {
    type: 'machineWorkDirs',
    machine: 'gpu-box',
    workDir: WD,
    recentWorkDirs: [{path: WD, ts: 1}],
    error: '',
  });
  if (entries) {
    const req = ofType(posted, 'listDir')
      .filter(m => m.path === WD && m.machine === 'gpu-box')
      .pop();
    assert.ok(req, 'the Explorer lists gpu-box');
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD,
      root: WD,
      entries,
      machine: 'gpu-box',
    });
  }
}

async function main() {
  await test("F1: an earlier job editor's success keeps the newer editor waiting", () => {
    const {win, posted} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    showSection(win, 'meta-schedule');
    const jobs = [job('a'), job('b')];
    send(win, {type: 'cronJobs', jobs});
    const list = byId(win, 'meta-schedule-list');
    click(
      win,
      rows(win, 'meta-schedule-list')[0].querySelector('.sidebar-item-edit'),
    );
    let form = list.querySelector('form');
    form.elements.what.value = 'save A';
    submit(win, form);
    assert.strictEqual(ofType(posted, 'cronAction').pop().jobId, 'a');
    cancel(win, form);
    click(
      win,
      rows(win, 'meta-schedule-list')[1].querySelector('.sidebar-item-edit'),
    );
    form = list.querySelector('form');
    form.elements.what.value = 'save B';
    submit(win, form);
    assert.strictEqual(ofType(posted, 'cronAction').pop().jobId, 'b');
    assert.ok(
      saveBtn(form).disabled,
      "B's Save is disabled while its update is in flight",
    );
    // A's update went through: it says nothing about B.
    send(win, {
      type: 'cronJobs',
      jobs,
      action: 'update',
      jobId: 'a',
      error: '',
    });
    form = list.querySelector('form');
    assert.ok(form, "BUG: A's acknowledgment closed B's editor");
    assert.strictEqual(form.elements.what.value, 'save B');
    assert.ok(saveBtn(form).disabled, 'B still awaits its own reply');
    // B's update was refused: its editor is re-armed with the draft.
    send(win, {
      type: 'cronJobs',
      jobs,
      action: 'update',
      jobId: 'b',
      error: 'B rejected',
    });
    form = list.querySelector('form');
    assert.ok(form, "B's refusal re-opens its editor");
    assert.strictEqual(form.elements.what.value, 'save B', 'with its draft');
    assert.ok(!saveBtn(form).disabled, 'Save works again');
    assert.strictEqual(
      byId(win, 'meta-schedule-status').textContent,
      'B rejected',
    );
    // Its retry closes it.
    submit(win, form);
    send(win, {
      type: 'cronJobs',
      jobs,
      action: 'update',
      jobId: 'b',
      error: '',
    });
    assert.strictEqual(
      list.querySelector('form'),
      null,
      'its own success closes it',
    );
  });

  await test('F1: the same job saved, cancelled, re-opened and saved again', () => {
    const {win, posted} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    showSection(win, 'meta-schedule');
    const jobs = [job('a')];
    send(win, {type: 'cronJobs', jobs});
    const list = byId(win, 'meta-schedule-list');
    click(
      win,
      rows(win, 'meta-schedule-list')[0].querySelector('.sidebar-item-edit'),
    );
    let form = list.querySelector('form');
    form.elements.what.value = 'first draft';
    submit(win, form);
    cancel(win, form);
    click(
      win,
      rows(win, 'meta-schedule-list')[0].querySelector('.sidebar-item-edit'),
    );
    form = list.querySelector('form');
    form.elements.what.value = 'second draft';
    submit(win, form);
    assert.strictEqual(ofType(posted, 'cronAction').length, 2);
    // The first save's success answers the first form (the daemon
    // answers in the order asked): the second keeps waiting.
    send(win, {
      type: 'cronJobs',
      jobs,
      action: 'update',
      jobId: 'a',
      error: '',
    });
    form = list.querySelector('form');
    assert.ok(
      form,
      "BUG: the first save's success closed the re-opened editor",
    );
    assert.ok(saveBtn(form).disabled);
    send(win, {
      type: 'cronJobs',
      jobs,
      action: 'update',
      jobId: 'a',
      error: 'nope',
    });
    form = list.querySelector('form');
    assert.ok(form, 'the refusal re-arms the editor');
    assert.strictEqual(form.elements.what.value, 'second draft');
    assert.ok(!saveBtn(form).disabled);
  });

  for (const kind of ['schedule', 'machines']) {
    await test(
      'F1: an earlier ' +
        kind +
        " add form's success keeps the newer one's draft",
      () => {
        const {win, posted} = makePage();
        send(win, {type: 'daemonStatus', connected: true});
        if (kind === 'machines') setupMachines(win);
        else send(win, {type: 'cronJobs', jobs: []});
        const list = byId(win, 'meta-' + kind + '-list');
        const add = byId(win, 'meta-' + kind + '-add');
        for (const name of ['first', 'second']) {
          click(win, add);
          const form = list.querySelector('form');
          form.elements.name.value = name;
          if (kind === 'machines') {
            form.elements.address.value = 'https://' + name + ':8787';
          } else {
            form.elements.what.value = name + ' unsaved task';
            form.elements.schedule.value = 'every 1h';
          }
          submit(win, form);
          assert.ok(saveBtn(form).disabled);
          if (name === 'first') cancel(win, form);
        }
        const type = kind === 'machines' ? 'machines' : 'cronJobs';
        const action = kind === 'machines' ? 'add' : 'create';
        const requests = ofType(
          posted,
          kind === 'machines' ? 'machineAction' : 'cronAction',
        );
        assert.strictEqual(requests.length, 2);
        send(win, {
          type,
          action,
          name: 'first',
          jobId: '',
          machines: [],
          jobs: [],
          error: '',
        });
        let form = list.querySelector('form');
        assert.ok(form, "BUG: the first form's success closed the second form");
        assert.strictEqual(form.elements.name.value, 'second');
        assert.ok(saveBtn(form).disabled, 'the second still awaits its reply');
        send(win, {
          type,
          action,
          name: 'second',
          jobId: '',
          machines: [],
          jobs: [],
          error: 'second refused',
        });
        form = list.querySelector('form');
        assert.ok(form, "the second form's refusal re-opens it");
        assert.strictEqual(
          form.elements.name.value,
          'second',
          'with its draft',
        );
        assert.ok(!saveBtn(form).disabled, 'Save works again');
      },
    );
  }

  await test('F1: machine add replies arriving out of order answer their own forms', () => {
    const {win, posted} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    setupMachines(win);
    const list = byId(win, 'meta-machines-list');
    const add = byId(win, 'meta-machines-add');
    for (const name of ['slow', 'quick']) {
      click(win, add);
      const form = list.querySelector('form');
      form.elements.name.value = name;
      form.elements.address.value = 'https://' + name + ':8787';
      submit(win, form);
      if (name === 'slow') cancel(win, form);
    }
    assert.strictEqual(ofType(posted, 'machineAction').length, 2);
    // The daemon adds machines in the background: quick's reply (its
    // success) comes first and closes the form that sent it.
    send(win, {
      type: 'machines',
      action: 'add',
      name: 'quick',
      machines: [],
      error: '',
    });
    assert.strictEqual(
      list.querySelector('form'),
      null,
      "quick's success closes its form",
    );
    // slow's refusal answers a form that was cancelled: no form reopens.
    send(win, {
      type: 'machines',
      action: 'add',
      name: 'slow',
      machines: [],
      error: 'slow refused',
    });
    assert.strictEqual(
      list.querySelector('form'),
      null,
      'the cancelled form stays closed',
    );
    assert.strictEqual(
      byId(win, 'meta-machines-status').textContent,
      'slow refused',
    );
  });

  await test('2026-11-04 W2 F1: a form is locked while its save is in flight', () => {
    const {win, posted} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'cronJobs', jobs: [job('a')]});
    const list = byId(win, 'meta-schedule-list');
    click(win, list.querySelector('.sidebar-item-edit'));
    let form = list.querySelector('form');
    form.elements.what.value = 'submitted prompt';
    submit(win, form);
    assert.strictEqual(ofType(posted, 'cronAction').pop().what, 'submitted prompt');
    // Typed into a live field meanwhile, the newer text would never be
    // sent, and the success closing the form would discard it.
    for (const field of form.querySelectorAll('[name]'))
      assert.ok(field.disabled, field.name + ' is locked while saving');
    assert.ok(form.querySelector('.sidebar-confirm-no').disabled === false);
    // The poll's rebuild keeps the form locked; a refusal unlocks it
    // with the submitted values.
    send(win, {type: 'cronJobs', jobs: [job('a')]});
    form = list.querySelector('form');
    assert.ok(form.elements.what.disabled, 'still locked after a poll');
    send(win, {type: 'cronJobs', jobs: [job('a')], action: 'update', jobId: 'a', error: 'refused'});
    form = list.querySelector('form');
    assert.ok(!form.elements.what.disabled, 'a refusal unlocks the fields');
    assert.ok(!saveBtn(form).disabled);
    assert.strictEqual(form.elements.what.value, 'submitted prompt');
    // The same for the add forms, and a disconnect unlocks them too.
    for (const [kind, type] of [['schedule', 'cronJobs'], ['machines', 'machines']]) {
      if (kind === 'machines') setupMachines(win);
      click(win, byId(win, 'meta-' + kind + '-add'));
      const add = byId(win, 'meta-' + kind + '-list').querySelector('form');
      add.elements.name.value = 'n';
      if (kind === 'schedule') {
        add.elements.schedule.value = 'every 1h';
        add.elements.what.value = 'w';
      } else add.elements.address.value = 'https://n:8787';
      submit(win, add);
      for (const field of add.querySelectorAll('[name]'))
        assert.ok(field.disabled, kind + ' add: ' + field.name + ' is locked');
      send(win, {type, jobs: [job('a')], machines: []});
      const rebuilt = byId(win, 'meta-' + kind + '-list').querySelector('form');
      assert.ok(rebuilt.elements.name.disabled, kind + ' add: locked after a poll');
      send(win, {type: 'daemonStatus', connected: false});
      assert.ok(!rebuilt.elements.name.disabled, kind + ' add: a disconnect unlocks');
      assert.ok(!saveBtn(rebuilt).disabled);
      send(win, {type: 'daemonStatus', connected: true});
      cancel(win, rebuilt);
    }
  });

  await test('2026-11-04 W2 F2: unnamed machine adds are answered by their generation', () => {
    const {win, posted} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    setupMachines(win);
    const list = byId(win, 'meta-machines-list');
    for (const host of ['slow', 'quick']) {
      click(win, byId(win, 'meta-machines-add'));
      const form = list.querySelector('form');
      form.elements.address.value = 'https://' + host + '.example:8787';
      submit(win, form);
      if (host === 'slow') cancel(win, form);
    }
    const reqs = ofType(posted, 'machineAction');
    assert.deepStrictEqual(reqs.map(r => r.name), ['', '']);
    assert.ok(reqs[0].generation > 0 && reqs[1].generation > reqs[0].generation);
    // The daemon adds the two hosts in parallel (one lock per host)
    // and echoes each add's generation: quick's refusal comes first
    // and re-arms the form that sent it, with its draft.
    send(win, {type: 'machines', action: 'add', name: '', generation: reqs[1].generation, machines: [], error: 'quick refused'});
    let form = list.querySelector('form');
    assert.ok(form, 'the quick form is still open');
    assert.strictEqual(form.elements.address.value, 'https://quick.example:8787');
    assert.ok(!saveBtn(form).disabled, "quick's refusal re-arms its Save");
    // slow's late success answers the cancelled form: quick's stays.
    send(win, {type: 'machines', action: 'add', name: '', generation: reqs[0].generation, machines: [{name: 'slow.example', status: 'reachable'}], error: ''});
    form = list.querySelector('form');
    assert.ok(form, "slow's success leaves the quick form open");
    assert.strictEqual(form.elements.address.value, 'https://quick.example:8787');
    assert.ok(!saveBtn(form).disabled);
    // A reply without a generation (a forwarded machine's) still
    // answers by name.
    submit(win, form);
    send(win, {type: 'machines', action: 'add', name: '', machines: [], error: ''});
    assert.strictEqual(list.querySelector('form'), null);
  });

  await test('F1: a reply to a save the panels dropped (machine change) resolves nothing', () => {
    const {win, posted} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    setupMachines(win);
    const list = byId(win, 'meta-machines-list');
    click(win, byId(win, 'meta-machines-add'));
    let form = list.querySelector('form');
    form.elements.name.value = 'draft';
    form.elements.address.value = 'https://draft:8787';
    submit(win, form);
    assert.ok(saveBtn(form).disabled);
    // The subpanels move to gpu-box: the add in flight went to this
    // daemon, whose reply is dropped; the draft comes back, not saving.
    bindGpuBox(win, posted, null);
    send(win, {
      type: 'machines',
      forwarded: true,
      machine: 'gpu-box',
      machines: [{name: 'local', local: true, status: 'reachable'}],
      action: '',
      error: '',
    });
    form = list.querySelector('form');
    assert.ok(form, 'the draft is back on gpu-box');
    assert.ok(!saveBtn(form).disabled, 'and Save works');
    submit(win, form);
    const sent = ofType(posted, 'machineAction').pop();
    assert.strictEqual(sent.machine, 'gpu-box');
    // gpu-box's own reply closes the form (the dropped save of this
    // daemon must not have been matched instead).
    send(win, {
      type: 'machines',
      forwarded: true,
      machine: 'gpu-box',
      machines: [],
      action: 'add',
      name: 'draft',
      error: '',
    });
    assert.strictEqual(
      list.querySelector('form'),
      null,
      "gpu-box's success closes the form",
    );
  });

  await test('F3 (round 2): an add saved, left for another machine, returned to and saved again keeps both owners', () => {
    const {win, posted} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    setupMachines(win);
    const list = byId(win, 'meta-machines-list');
    click(win, byId(win, 'meta-machines-add'));
    let form = list.querySelector('form');
    form.elements.name.value = 'pending-host';
    form.elements.address.value = 'https://old-address:8787';
    submit(win, form);
    assert.strictEqual(ofType(posted, 'machineAction').length, 1);
    const id = win._testApi.getActiveTabId();
    // Away to gpu-box (its registry answers) and back to this daemon
    // before the add was answered.
    bindGpuBox(win, posted, null);
    send(win, {
      type: 'machines',
      forwarded: true,
      machine: 'gpu-box',
      machines: [],
      action: '',
      error: '',
    });
    send(win, {type: 'tabMachine', tabId: id, machine: ''});
    send(win, {
      type: 'machines',
      machines: [
        {name: 'local', local: true, status: 'reachable'},
        {name: 'gpu-box', status: 'reachable'},
      ],
      action: '',
      error: '',
    });
    form = list.querySelector('form');
    assert.ok(form, 'the draft is retained');
    assert.ok(!saveBtn(form).disabled, 'and Save works');
    form.elements.address.value = 'https://new-address:8787';
    submit(win, form);
    const sent = ofType(posted, 'machineAction');
    assert.strictEqual(sent.length, 2);
    assert.strictEqual(sent[1].address, 'https://new-address:8787');
    assert.ok(saveBtn(form).disabled);
    // The first add's late success answers the first form, not this one.
    send(win, {
      type: 'machines',
      machines: [],
      action: 'add',
      name: 'pending-host',
      error: '',
    });
    form = list.querySelector('form');
    assert.ok(form, "BUG: the old reply closed the new save's form");
    assert.ok(saveBtn(form).disabled, 'still awaiting its own reply');
    // The second add's refusal brings its draft back.
    send(win, {
      type: 'machines',
      machines: [],
      action: 'add',
      name: 'pending-host',
      error: 'new address refused',
    });
    form = list.querySelector('form');
    assert.ok(form, 'the refused draft is back');
    assert.strictEqual(form.elements.address.value, 'https://new-address:8787');
    assert.ok(!saveBtn(form).disabled, 'for a fix');
    assert.strictEqual(
      byId(win, 'meta-machines-status').textContent,
      'new address refused',
    );
  });

  await test('F3 (round 2): a job create answered while its machine is not shown settles its own save only', () => {
    const {win, posted} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    setupMachines(win);
    showSection(win, 'meta-schedule');
    send(win, {type: 'cronJobs', jobs: []});
    const list = byId(win, 'meta-schedule-list');
    click(win, byId(win, 'meta-schedule-add'));
    let form = list.querySelector('form');
    form.elements.name.value = 'n';
    form.elements.schedule.value = 'every 1h';
    form.elements.what.value = 'w';
    submit(win, form);
    const id = win._testApi.getActiveTabId();
    bindGpuBox(win, posted, null);
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [],
      action: '',
      error: '',
    });
    // This daemon's reply lands while gpu-box is shown: not painted,
    // but its save is answered.
    send(win, {type: 'cronJobs', jobs: [job('n')], action: 'create', error: ''});
    form = list.querySelector('form');
    assert.ok(form, 'the draft is kept on gpu-box');
    assert.ok(!saveBtn(form).disabled);
    // Back on this daemon the draft is saved again: only this second
    // create is awaited, so its reply closes the form.
    send(win, {type: 'tabMachine', tabId: id, machine: ''});
    send(win, {type: 'cronJobs', jobs: [job('n')], action: '', error: ''});
    form = list.querySelector('form');
    assert.ok(form, 'the draft is back on this daemon');
    submit(win, form);
    assert.strictEqual(ofType(posted, 'cronAction').length, 2);
    send(win, {type: 'cronJobs', jobs: [job('n')], action: 'create', error: ''});
    assert.strictEqual(
      list.querySelector('form'),
      null,
      'the second create closes its form',
    );
  });

  await test('F1: a disconnect releases the pending saves', () => {
    const {win} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'cronJobs', jobs: []});
    const list = byId(win, 'meta-schedule-list');
    click(win, byId(win, 'meta-schedule-add'));
    const form = list.querySelector('form');
    form.elements.name.value = 'n';
    form.elements.schedule.value = 'every 1h';
    form.elements.what.value = 'w';
    submit(win, form);
    assert.ok(saveBtn(form).disabled);
    send(win, {type: 'daemonStatus', connected: false});
    assert.ok(!saveBtn(form).disabled, 'Save works again while away');
    send(win, {type: 'daemonStatus', connected: true});
    submit(win, form);
    // Only the second create is awaited: the first's reply never came.
    send(win, {type: 'cronJobs', jobs: [], action: 'create', error: ''});
    assert.strictEqual(
      list.querySelector('form'),
      null,
      'the re-sent create closes it',
    );
  });

  await test('F2: a move reply spends the Cut of its own machine only', () => {
    const {win, posted} = makePage();
    setupMachines(win);
    const entries = [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
      {name: 'dest', path: WD + '/dest', isDir: true},
    ];
    openExplorer(win, posted, entries);
    runMenuItem(win, explorerRow(win, WD + '/a.txt'), 'Cut');
    runMenuItem(win, explorerRow(win, WD + '/dest'), 'Paste');
    const move = ofType(posted, 'fsAction').pop();
    assert.strictEqual(move.action, 'move');
    assert.strictEqual(
      move.machine,
      undefined,
      'the move runs on this machine',
    );
    // Before its reply the tree shows gpu-box, whose a.txt is Cut.
    bindGpuBox(win, posted, entries);
    runMenuItem(win, explorerRow(win, WD + '/a.txt'), 'Cut');
    rightClick(win, explorerRow(win, WD + '/dest'));
    assert.ok(!menuItem(win, 'Paste').classList.contains('disabled'));
    send(win, {
      type: 'fsResult',
      token: move.token,
      path: WD + '/dest/a.txt',
      action: 'move',
      ok: true,
    });
    rightClick(win, explorerRow(win, WD + '/dest'));
    assert.ok(
      !menuItem(win, 'Paste').classList.contains('disabled'),
      "BUG: this machine's move reply cleared the Cut of the same path on gpu-box",
    );
    click(win, menuItem(win, 'Paste'));
    const paste = ofType(posted, 'fsAction').pop();
    assert.strictEqual(paste.action, 'move');
    assert.strictEqual(
      paste.machine,
      'gpu-box',
      'the Paste moves the gpu-box entry',
    );
  });

  await test('F3: a commit menu left open acts on the machine it was opened on', () => {
    const {win, posted} = makePage();
    setupMachines(win);
    const row = openScm(win, posted);
    rightClick(win, row);
    bindGpuBox(win, posted, null);
    click(win, menuItem(win, 'Checkout (Detached)'));
    const req = ofType(posted, 'gitAction').pop();
    assert.strictEqual(req.sha, SHA_A);
    assert.strictEqual(
      req.machine,
      undefined,
      'BUG: the local commit was checked out on gpu-box',
    );
    assert.strictEqual(req.workDir, WD);
  });

  await test('F3: a commit menu left open acts on the repository it was opened on', () => {
    const {win, posted} = makePage();
    const row = openScm(win, posted);
    rightClick(win, row);
    pinWorkspace(win, '/ws/other');
    click(win, menuItem(win, 'Cherry Pick'));
    const req = ofType(posted, 'gitAction').pop();
    assert.strictEqual(req.sha, SHA_A);
    assert.strictEqual(
      req.workDir,
      WD,
      'BUG: the cherry-pick went to the new working directory',
    );
  });

  await test("F3: a commit menu's prompt opened after the move keeps the repository too", () => {
    const {win, posted} = makePage();
    const row = openScm(win, posted);
    rightClick(win, row);
    pinWorkspace(win, '/ws/other');
    click(win, menuItem(win, 'Create Branch...'));
    const prompt = toast(win, 'git-create-branch');
    assert.ok(prompt, 'the branch name prompt is open');
    const input = toastInput(prompt);
    input.value = 'feature';
    click(win, toastButton(prompt, 'Create branch'));
    const branch = ofType(posted, 'gitAction').pop();
    assert.strictEqual(branch.action, 'createBranch');
    assert.strictEqual(branch.name, 'feature');
    assert.strictEqual(branch.sha, SHA_A);
    assert.strictEqual(
      branch.workDir,
      WD,
      'BUG: the branch went to the new working directory',
    );
  });

  await test('F4: a late New File reply opens the editor for the chat that made it', () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, []);
    const owner = win._testApi.getActiveTabId();
    runMenuItem(win, explorerRow(win, WD), 'New File...');
    const input = win.document.querySelector('input.explorer-input');
    input.value = 'late.txt';
    key(win, input, 'Enter');
    const req = ofType(posted, 'fsAction').pop();
    assert.strictEqual(req.tabId, owner);
    win._testApi.createNewTab();
    const current = win._testApi.getActiveTabId();
    assert.notStrictEqual(current, owner);
    send(win, {
      type: 'fsResult',
      token: req.token,
      tabId: owner,
      action: 'newFile',
      path: WD + '/late.txt',
      ok: true,
    });
    const open = ofType(posted, 'openFile').pop();
    assert.ok(open, 'the new file is opened');
    assert.strictEqual(
      open.tabId,
      owner,
      'BUG: the editor was asked for the current chat',
    );
    assert.strictEqual(
      open.background,
      true,
      'without leaving the current chat',
    );
  });

  await test('F4: a closed creator gets no editor', () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, []);
    const owner = win._testApi.getActiveTabId();
    runMenuItem(win, explorerRow(win, WD), 'New File...');
    const input = win.document.querySelector('input.explorer-input');
    input.value = 'closed-owner.txt';
    key(win, input, 'Enter');
    const req = ofType(posted, 'fsAction').pop();
    win._testApi.createNewTab();
    const current = win._testApi.getActiveTabId();
    send(win, {
      type: 'tabs_state',
      tabs: [{tabId: current, chatId: 'chat-b', workDir: WD, title: 'B'}],
    });
    assert.ok(
      !win._testApi.openTabs().some(t => t.id === owner),
      'A is closed',
    );
    const before = ofType(posted, 'openFile').length;
    send(win, {
      type: 'fsResult',
      token: req.token,
      tabId: owner,
      action: 'newFile',
      path: WD + '/closed-owner.txt',
      ok: true,
    });
    assert.strictEqual(
      ofType(posted, 'openFile').length,
      before,
      'BUG: a closed chat opened an editor',
    );
  });

  await test('F4: a New File reply for the chat on screen opens the editor there', () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, []);
    const owner = win._testApi.getActiveTabId();
    runMenuItem(win, explorerRow(win, WD), 'New File...');
    const input = win.document.querySelector('input.explorer-input');
    input.value = 'now.txt';
    key(win, input, 'Enter');
    const req = ofType(posted, 'fsAction').pop();
    send(win, {
      type: 'fsResult',
      token: req.token,
      tabId: owner,
      action: 'newFile',
      path: WD + '/now.txt',
      ok: true,
    });
    const open = ofType(posted, 'openFile').pop();
    assert.strictEqual(open.tabId, owner);
    assert.strictEqual(
      open.background,
      undefined,
      'a foreground open, as before',
    );
  });

  report('audit1101_W2_sidebar_reply_owners');
  await new Promise(r => setTimeout(r, 0));
  for (const win of pages) win.close();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
