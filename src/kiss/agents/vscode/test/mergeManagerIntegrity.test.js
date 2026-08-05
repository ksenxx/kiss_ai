// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for MergeManager state-integrity findings:
// VS-005: a failed save-time hunk removal must not advance hunk state.
// VS-006: resolve-file/all must not report success after failed edits.
// VS-007: a failed base-line applyEdit must not install bogus hunk state.
// VS-008: a pre-existing empty file (exec present) must not be deleted
//         on reject; a truly new file (exec absent) still is.
// VS-009: reject must restore the pre-task exec mode (tri-state).
// VS-010: a failed merge-open must not leave stale pending data that
//         replays over a newer merge.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const docs = new Map();
// Failure injection: fp -> remaining editor.edit failures (-1 = always).
const editFailures = new Map();
let applyEditFailures = 0;

function offsetOf(text, line, character) {
  const lines = text.split('\n');
  let off = 0;
  for (let i = 0; i < line && i < lines.length; i++) off += lines[i].length + 1;
  return off + character;
}

function makeDoc(fp) {
  const doc = {
    uri: {fsPath: fp, scheme: 'file'},
    _text: fs.readFileSync(fp, 'utf8'),
    isDirty: false,
    get lineCount() {
      return doc._text.split('\n').length;
    },
    lineAt(i) {
      const lines = doc._text.split('\n');
      const t = lines[i] || '';
      return {text: t, range: {end: {line: i, character: t.length}}};
    },
    _applyDelete(range) {
      const o1 = offsetOf(doc._text, range.start.line, range.start.character);
      const o2 = offsetOf(doc._text, range.end.line, range.end.character);
      doc._text = doc._text.slice(0, o1) + doc._text.slice(o2);
      doc.isDirty = true;
    },
    _applyInsert(pos, txt) {
      const o = offsetOf(doc._text, pos.line, pos.character);
      doc._text = doc._text.slice(0, o) + txt + doc._text.slice(o);
      doc.isDirty = true;
    },
    _applyReplace(range, txt) {
      doc._applyDelete(range);
      doc._applyInsert(range.start, txt);
    },
    _save() {
      fs.writeFileSync(fp, doc._text);
      doc.isDirty = false;
    },
  };
  return doc;
}

function getDoc(fp) {
  let d = docs.get(fp);
  if (!d) {
    d = makeDoc(fp);
    docs.set(fp, d);
  }
  return d;
}

function makeEditor(doc) {
  return {
    document: doc,
    selection: null,
    revealRange() {},
    setDecorations() {},
    edit(cb) {
      const fp = doc.uri.fsPath;
      const remaining = editFailures.get(fp) || 0;
      if (remaining !== 0) {
        if (remaining > 0) editFailures.set(fp, remaining - 1);
        return Promise.resolve(false);
      }
      cb({
        delete: range => doc._applyDelete(range),
        insert: (pos, txt) => doc._applyInsert(pos, txt),
        replace: (range, txt) => doc._applyReplace(range, txt),
      });
      return Promise.resolve(true);
    },
  };
}

class StubWorkspaceEdit {
  constructor() {
    this._textOps = [];
    this._deletes = [];
  }
  get size() {
    return this._textOps.length;
  }
  insert(uri, pos, txt) {
    this._textOps.push({kind: 'insert', uri, pos, txt});
  }
  replace(uri, range, txt) {
    this._textOps.push({kind: 'replace', uri, range, txt});
  }
  deleteFile(uri) {
    this._deletes.push(uri);
  }
}

let willSaveCb = null;

const vscodeStub = {
  Uri: {file: p => ({fsPath: p, scheme: 'file'})},
  Position: class {
    constructor(line, character) {
      this.line = line;
      this.character = character;
    }
  },
  Range: class {
    constructor(a, b, c, d) {
      this.start = {line: a, character: b};
      this.end = {line: c, character: d};
    }
  },
  Selection: class {
    constructor(a, b, c, d) {
      this.anchor = {line: a, character: b};
      this.active = {line: c, character: d};
    }
  },
  TextEditorRevealType: {InCenter: 2},
  ViewColumn: {One: 1},
  WorkspaceEdit: StubWorkspaceEdit,
  window: {
    visibleTextEditors: [],
    activeTextEditor: undefined,
    createTextEditorDecorationType: () => ({dispose() {}}),
    onDidChangeVisibleTextEditors: () => ({dispose() {}}),
    showTextDocument: doc => Promise.resolve(makeEditor(doc)),
    showInformationMessage: () => undefined,
    showWarningMessage: () => undefined,
  },
  workspace: {
    onWillSaveTextDocument: cb => {
      willSaveCb = cb;
      return {dispose() {}};
    },
    onDidSaveTextDocument: () => ({dispose() {}}),
    openTextDocument: uri => Promise.resolve(getDoc(uri.fsPath)),
    saveAll: () => {
      for (const d of docs.values()) {
        if (fs.existsSync(d.uri.fsPath)) d._save();
      }
      return Promise.resolve(true);
    },
    applyEdit: edit => {
      if (applyEditFailures !== 0 && edit._textOps.length > 0) {
        if (applyEditFailures > 0) applyEditFailures--;
        return Promise.resolve(false);
      }
      for (const op of edit._textOps) {
        const d = getDoc(op.uri.fsPath);
        if (op.kind === 'insert') d._applyInsert(op.pos, op.txt);
        else d._applyReplace(op.range, op.txt);
      }
      for (const uri of edit._deletes) {
        try {
          fs.rmSync(uri.fsPath, {force: true});
        } catch {}
        docs.delete(uri.fsPath);
      }
      return Promise.resolve(true);
    },
  },
  commands: {executeCommand: () => Promise.resolve()},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const sourcePath = path.join(__dirname, '..', 'out', 'MergeManager.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`npm run compile\` first`,
);
const {MergeManager} = require(sourcePath);

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-merge-integ-'));
const work = path.join(tmp, 'work');
const mergeDir = path.join(tmp, 'merge');
fs.mkdirSync(work, {recursive: true});
fs.mkdirSync(path.join(mergeDir, 'base'), {recursive: true});

function freshMgr() {
  docs.clear();
  editFailures.clear();
  applyEditFailures = 0;
  willSaveCb = null;
  return new MergeManager();
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// VS-005: failed save-time hunk removal must not advance hunk state.
async function testWillSaveFailureKeepsState() {
  const fp = path.join(work, 'ws.txt');
  const base = path.join(mergeDir, 'base', 'ws.txt');
  fs.writeFileSync(base, 'A\nB\nC\n');
  fs.writeFileSync(fp, 'A\nX\nC\n');

  const mgr = freshMgr();
  await mgr.openMerge({
    files: [
      {name: 'ws.txt', base, current: fp, hunks: [{bs: 1, bc: 1, cs: 1, cc: 1}]},
    ],
  });
  const s = mgr._ms[fp];
  assert.strictEqual(getDoc(fp)._text, 'A\nB\nX\nC\n');
  assert.deepStrictEqual(
    {os: s.hunks[0].os, oc: s.hunks[0].oc, ns: s.hunks[0].ns, nc: s.hunks[0].nc},
    {os: 1, oc: 1, ns: 2, nc: 1},
  );

  // Fire willSave with the edit failing (both attempts).
  editFailures.set(fp, -1);
  let waited = null;
  willSaveCb({document: getDoc(fp), waitUntil: p => (waited = p)});
  await waited;
  editFailures.delete(fp);

  assert.strictEqual(
    getDoc(fp)._text,
    'A\nB\nX\nC\n',
    'failed edit must leave the document unchanged',
  );
  assert.deepStrictEqual(
    {os: s.hunks[0].os, oc: s.hunks[0].oc, ns: s.hunks[0].ns, nc: s.hunks[0].nc},
    {os: 1, oc: 1, ns: 2, nc: 1},
    'VS-005: hunk state advanced even though the save-time edit failed',
  );
  mgr.dispose();
  console.log('ok - VS-005 willSave failure keeps hunk state consistent');
}

// VS-006: rejectAll with a failing edit must not report success/allDone.
async function testResolveAllFailureKeepsFile() {
  const fp1 = path.join(work, 'f1.txt');
  const fp2 = path.join(work, 'f2.txt');
  const base1 = path.join(mergeDir, 'base', 'f1.txt');
  const base2 = path.join(mergeDir, 'base', 'f2.txt');
  fs.writeFileSync(base1, 'A\nB\n');
  fs.writeFileSync(base2, 'P\nQ\n');
  fs.writeFileSync(fp1, 'A\nNEW1\nB\n');
  fs.writeFileSync(fp2, 'P\nNEW2\nQ\n');

  const mgr = freshMgr();
  let allDone = false;
  mgr.on('allDone', () => (allDone = true));
  await mgr.openMerge({
    files: [
      {name: 'f1.txt', base: base1, current: fp1, hunks: [{bs: 0, bc: 0, cs: 1, cc: 1}]},
      {name: 'f2.txt', base: base2, current: fp2, hunks: [{bs: 0, bc: 0, cs: 1, cc: 1}]},
    ],
  });

  editFailures.set(fp1, -1);
  await mgr.rejectAll();
  editFailures.delete(fp1);
  await delay(20);

  assert.strictEqual(
    allDone,
    false,
    'VS-006: allDone emitted although an edit failed',
  );
  assert.ok(
    mgr._ms[fp1],
    'VS-006: failed file state was discarded, losing the review',
  );
  assert.ok(!mgr._ms[fp2], 'successfully rejected file must be resolved');
  assert.strictEqual(fs.readFileSync(fp2, 'utf8'), 'P\nQ\n');
  mgr.dispose();
  console.log('ok - VS-006 rejectAll failure keeps state and withholds allDone');
}

// VS-007: failed base-line applyEdit must not install bogus hunk state.
async function testOpenMergeApplyEditFailure() {
  const fp = path.join(work, 'ae.txt');
  const base = path.join(mergeDir, 'base', 'ae.txt');
  fs.writeFileSync(base, 'A\nB\nC\n');
  fs.writeFileSync(fp, 'A\nX\nC\n');

  const mgr = freshMgr();
  applyEditFailures = -1;
  await mgr.openMerge({
    files: [
      {name: 'ae.txt', base, current: fp, hunks: [{bs: 1, bc: 1, cs: 1, cc: 1}]},
    ],
  });
  applyEditFailures = 0;

  assert.strictEqual(
    getDoc(fp)._text,
    'A\nX\nC\n',
    'document must be untouched when applyEdit fails',
  );
  const s = mgr._ms[fp];
  if (s) {
    // If the file is tracked, its coordinates must describe the real
    // document (no phantom inserted base lines).
    for (const h of s.hunks) {
      assert.strictEqual(
        h.oc,
        0,
        'VS-007: hunk claims base lines were inserted after failed applyEdit',
      );
    }
  }
  mgr.dispose();
  console.log('ok - VS-007 failed applyEdit does not install bogus hunk state');
}

// VS-008: a pre-existing empty file must not be deleted on reject.
async function testPreExistingEmptyFileNotDeleted() {
  const fp = path.join(work, 'empty.txt');
  const base = path.join(mergeDir, 'base', 'empty.txt');
  fs.writeFileSync(base, ''); // pre-task content was empty
  fs.writeFileSync(fp, 'NEW\n');

  const mgr = freshMgr();
  await mgr.openMerge({
    files: [
      {
        name: 'empty.txt',
        base,
        current: fp,
        exec: false, // pre-task file existed (server emits exec for it)
        hunks: [{bs: 0, bc: 0, cs: 0, cc: 1}],
      },
    ],
  });
  await mgr.rejectFile();
  await delay(20);

  assert.ok(
    fs.existsSync(fp),
    'VS-008: pre-existing empty file was deleted on reject',
  );
  mgr.dispose();

  // A truly new file (no exec in the manifest) must still be deleted.
  const nf = path.join(work, 'brandnew.txt');
  const nbase = path.join(mergeDir, 'base', 'brandnew.txt');
  fs.writeFileSync(nbase, '');
  fs.writeFileSync(nf, 'CREATED\n');
  const mgr2 = freshMgr();
  await mgr2.openMerge({
    files: [
      {name: 'brandnew.txt', base: nbase, current: nf, hunks: [{bs: 0, bc: 0, cs: 0, cc: 1}]},
    ],
  });
  await mgr2.rejectFile();
  await delay(20);
  assert.ok(
    !fs.existsSync(nf),
    'rejecting a truly new file must still delete it',
  );
  mgr2.dispose();
  console.log('ok - VS-008 empty-base classification uses exec presence');
}

// VS-009: reject restores the tri-state pre-task exec mode in place.
async function testRejectRestoresExecMode() {
  if (process.platform === 'win32') return;
  const fp = path.join(work, 'script.sh');
  const base = path.join(mergeDir, 'base', 'script.sh');
  fs.writeFileSync(base, 'echo old\n');
  fs.writeFileSync(fp, 'echo new\n');
  // Agent made the file executable; pre-task mode was NOT executable.
  fs.chmodSync(fp, 0o755);

  const mgr = freshMgr();
  await mgr.openMerge({
    files: [
      {
        name: 'script.sh',
        base,
        current: fp,
        exec: false,
        hunks: [{bs: 0, bc: 1, cs: 0, cc: 1}],
      },
    ],
  });
  await mgr.rejectFile();
  await delay(20);

  assert.strictEqual(
    fs.statSync(fp).mode & 0o111,
    0,
    'VS-009: rejecting must clear exec bits when the pre-task mode was -x',
  );

  // And the reverse: pre-task executable, agent cleared the bit.
  const fp2 = path.join(work, 'tool.sh');
  const base2 = path.join(mergeDir, 'base', 'tool.sh');
  fs.writeFileSync(base2, 'echo a\n');
  fs.writeFileSync(fp2, 'echo b\n');
  fs.chmodSync(fp2, 0o644);
  const mgr2 = freshMgr();
  await mgr2.openMerge({
    files: [
      {
        name: 'tool.sh',
        base: base2,
        current: fp2,
        exec: true,
        hunks: [{bs: 0, bc: 1, cs: 0, cc: 1}],
      },
    ],
  });
  await mgr2.rejectFile();
  await delay(20);
  assert.notStrictEqual(
    fs.statSync(fp2).mode & 0o111,
    0,
    'VS-009: rejecting must set exec bits when the pre-task mode was +x',
  );
  mgr.dispose();
  mgr2.dispose();
  console.log('ok - VS-009 reject restores tri-state exec mode in place');
}

// VS-010: a failed merge-open must not replay stale pending data.
async function testFailedOpenDoesNotReplayStalePending() {
  const fpB = path.join(work, 'pb.txt');
  const fpC = path.join(work, 'pc.txt');
  const baseB = path.join(mergeDir, 'base', 'pb.txt');
  const baseC = path.join(mergeDir, 'base', 'pc.txt');
  fs.writeFileSync(baseB, 'B\n');
  fs.writeFileSync(baseC, 'C\n');
  fs.writeFileSync(fpB, 'B\nNEWB\n');
  fs.writeFileSync(fpC, 'C\nNEWC\n');
  const missing = path.join(work, 'does-not-exist.txt');

  const mgr = freshMgr();
  // A: fails (current file unreadable). B: queued while A is in flight.
  const pA = mgr.openMerge({
    files: [
      {name: 'missing', base: baseB, current: missing, hunks: [{bs: 0, bc: 0, cs: 0, cc: 1}]},
    ],
  });
  const pB = mgr.openMerge({
    files: [
      {name: 'pb.txt', base: baseB, current: fpB, hunks: [{bs: 0, bc: 0, cs: 1, cc: 1}]},
    ],
  });
  await Promise.allSettled([pA, pB]);

  // Newest payload C arrives afterwards; stale B must not replace it.
  await mgr.openMerge({
    files: [
      {name: 'pc.txt', base: baseC, current: fpC, hunks: [{bs: 0, bc: 0, cs: 1, cc: 1}]},
    ],
  });
  await delay(20);

  assert.deepStrictEqual(
    Object.keys(mgr._ms),
    [fpC],
    'VS-010: stale pending merge payload replayed over the newest merge',
  );
  mgr.dispose();
  console.log('ok - VS-010 failed open does not replay stale pending merge');
}

async function run() {
  await testWillSaveFailureKeepsState();
  await testResolveAllFailureKeepsFile();
  await testOpenMergeApplyEditFailure();
  await testPreExistingEmptyFileNotDeleted();
  await testRejectRestoresExecMode();
  await testFailedOpenDoesNotReplayStalePending();
  console.log('\nAll merge-manager integrity tests passed');
}

run().then(
  () => {
    fs.rmSync(tmp, {recursive: true, force: true});
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err);
    fs.rmSync(tmp, {recursive: true, force: true});
    process.exit(1);
  },
);
