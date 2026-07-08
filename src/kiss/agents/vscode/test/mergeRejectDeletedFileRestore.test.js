// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Bug-hunt integration test for MergeManager's reject-side restore.
//
// Bugs locked in (all mirror behavior the server-side reject path in
// web_server.py already implements — the native VS Code path ignored
// the manifest's ``target`` / ``link_target`` / ``exec`` fields):
//
//   1. Rejecting an agent-DELETED tracked file never restored it: the
//      manifest points ``current`` at an empty ``.deleted`` placeholder
//      and carries the real workspace path in ``target``; MergeManager
//      only ever wrote to ``current``, so the user's file stayed
//      deleted after clicking Reject.
//
//   2. ``_restoreBinaryBase`` wrote THROUGH a symlink at the reviewed
//      path — silently clobbering the symlink's destination (possibly
//      a precious file outside the repo).
//
//   3. A ``link_target`` entry must recreate the symlink itself on
//      reject, not write the blob's target string as file content.
//
//   4. Rejecting a deleted ``100755`` script must restore it with the
//      exec bit (``exec: true`` in the manifest).
//
// Drives the real compiled ``out/MergeManager.js`` with the same
// in-memory vscode stub as bughunt_isNewFile.test.js.
//
// Run directly with ``node`` (after ``npm run compile``):
//
//     node src/kiss/agents/vscode/test/mergeRejectDeletedFileRestore.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

// ---------------------------------------------------------------------------
// In-memory text-document model shared by the vscode stub (same as
// bughunt_isNewFile.test.js).
// ---------------------------------------------------------------------------

const docs = new Map();

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
    onWillSaveTextDocument: () => ({dispose() {}}),
    onDidSaveTextDocument: () => ({dispose() {}}),
    openTextDocument: uri => Promise.resolve(getDoc(uri.fsPath)),
    saveAll: () => {
      for (const d of docs.values()) {
        if (fs.existsSync(d.uri.fsPath)) d._save();
      }
      return Promise.resolve(true);
    },
    applyEdit: edit => {
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
fs.writeFileSync(
  path.join(__dirname, '_vscode-stub.js'),
  `'use strict';\nmodule.exports = global.__kissVscodeStub;\n`,
);
global.__kissVscodeStub = vscodeStub;

const sourcePath = path.join(__dirname, '..', 'out', 'MergeManager.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`npm run compile\` first`,
);
const {MergeManager} = require(sourcePath);

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-merge-del-'));
const work = path.join(tmp, 'work');
const mergeDir = path.join(tmp, 'merge');
fs.mkdirSync(work, {recursive: true});
fs.mkdirSync(path.join(mergeDir, '.deleted'), {recursive: true});
fs.mkdirSync(path.join(mergeDir, 'base'), {recursive: true});

const BASE_TEXT = 'line one\nline two\nline three\n';

function freshMgr() {
  docs.clear();
  return new MergeManager();
}

async function testRejectDeletedTextFileRestores() {
  const target = path.join(work, 'app.py');
  const placeholder = path.join(mergeDir, '.deleted', 'app.py');
  const base = path.join(mergeDir, 'base', 'app.py');
  fs.writeFileSync(placeholder, '');
  fs.writeFileSync(base, BASE_TEXT);
  assert.ok(!fs.existsSync(target), 'precondition: agent deleted app.py');

  const mgr = freshMgr();
  await mgr.openMerge({
    files: [
      {
        name: 'app.py',
        base,
        current: placeholder,
        target,
        exec: true,
        hunks: [{bs: 0, bc: 3, cs: 0, cc: 0}],
      },
    ],
  });
  await mgr.rejectFile();
  mgr.dispose();

  assert.ok(
    fs.existsSync(target),
    'BUG 1: rejecting an agent-DELETED file must restore it at the real ' +
      'workspace path (manifest `target`), not leave it deleted',
  );
  assert.strictEqual(
    fs.readFileSync(target, 'utf8'),
    BASE_TEXT,
    'restored file must carry the base content',
  );
  if (process.platform !== 'win32') {
    assert.ok(
      (fs.statSync(target).mode & 0o111) !== 0,
      'BUG 4: an `exec: true` manifest entry must restore the exec bit',
    );
  }
  fs.rmSync(target, {force: true});
}

async function testAcceptDeletedTextFileStaysDeleted() {
  const target = path.join(work, 'gone.py');
  const placeholder = path.join(mergeDir, '.deleted', 'gone.py');
  const base = path.join(mergeDir, 'base', 'gone.py');
  fs.writeFileSync(placeholder, '');
  fs.writeFileSync(base, BASE_TEXT);

  const mgr = freshMgr();
  await mgr.openMerge({
    files: [
      {
        name: 'gone.py',
        base,
        current: placeholder,
        target,
        hunks: [{bs: 0, bc: 3, cs: 0, cc: 0}],
      },
    ],
  });
  await mgr.acceptFile();
  mgr.dispose();

  assert.ok(
    !fs.existsSync(target),
    'accepting an agent deletion must leave the workspace file deleted',
  );
}

async function testRejectBinaryDoesNotWriteThroughSymlink() {
  if (process.platform === 'win32') return;
  const precious = path.join(tmp, 'precious.txt');
  fs.writeFileSync(precious, 'PRECIOUS\n');
  const target = path.join(work, 'cfg');
  try {
    fs.unlinkSync(target);
  } catch {}
  fs.symlinkSync(precious, target);
  const base = path.join(mergeDir, 'base', 'cfg');
  fs.writeFileSync(base, 'BASEBYTES\n');

  const mgr = freshMgr();
  await mgr.openMerge({
    files: [
      {
        name: 'cfg',
        base,
        current: target,
        target,
        binary: true,
        hunks: [{bs: 0, bc: 0, cs: 0, cc: 0}],
      },
    ],
  });
  await mgr.rejectFile();
  mgr.dispose();

  assert.strictEqual(
    fs.readFileSync(precious, 'utf8'),
    'PRECIOUS\n',
    'BUG 2: restoring a base must never write THROUGH a symlink — the ' +
      "symlink's destination was silently clobbered",
  );
  assert.strictEqual(
    fs.readFileSync(target, 'utf8'),
    'BASEBYTES\n',
    'the reviewed path itself must carry the restored base bytes',
  );
  fs.rmSync(target, {force: true});
  fs.rmSync(precious, {force: true});
}

async function testRejectLinkTargetRecreatesSymlink() {
  if (process.platform === 'win32') return;
  const dest = path.join(tmp, 'dest.txt');
  fs.writeFileSync(dest, 'DEST\n');
  const target = path.join(work, 'lnk');
  const placeholder = path.join(mergeDir, '.deleted', 'lnk');
  fs.writeFileSync(placeholder, '');
  const base = path.join(mergeDir, 'base', 'lnk');
  fs.writeFileSync(base, dest); // blob content = the target string

  const mgr = freshMgr();
  await mgr.openMerge({
    files: [
      {
        name: 'lnk',
        base,
        current: placeholder,
        target,
        binary: true,
        link_target: dest,
        hunks: [{bs: 0, bc: 0, cs: 0, cc: 0}],
      },
    ],
  });
  await mgr.rejectFile();
  mgr.dispose();

  assert.ok(
    fs.lstatSync(target).isSymbolicLink(),
    'BUG 3: rejecting a `link_target` entry must recreate the symlink ' +
      "itself, not write the blob's target string as file content",
  );
  assert.strictEqual(fs.readlinkSync(target), dest);
  fs.rmSync(target, {force: true});
}

async function run() {
  await testRejectDeletedTextFileRestores();
  await testAcceptDeletedTextFileStaysDeleted();
  await testRejectBinaryDoesNotWriteThroughSymlink();
  await testRejectLinkTargetRecreatesSymlink();
  console.log('\nAll merge reject-restore tests passed');
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
