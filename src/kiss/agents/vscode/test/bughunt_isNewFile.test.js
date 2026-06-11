// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Bug-hunt integration test for ``MergeManager._doOpenMerge``'s
// ``isNewFile`` classification.
//
// Bug locked in:
//
//   ``isNewFile`` was computed as ``processed.every(h => h.oc === 0)``
//   — "every hunk has zero old lines".  But the Python side
//   (``diff_merge._hunk_to_dict`` / ``_file_as_new_hunks``) produces
//   ``bc = 0`` (⇒ ``oc = 0``) hunks for ANY pure insertion, including
//   appends to a file that existed long before the task.  Only a
//   brand-new file gets an EMPTY base copy
//   (``diff_merge._write_base_copy`` writes ``b""`` when git has no
//   blob).  So when the agent merely appended lines to an existing
//   file and the user clicked "Reject File" (or "Reject Rest"), the
//   extension treated the file as agent-created and DELETED it from
//   disk via ``_deleteNewFile`` — destroying the user's pre-existing
//   file instead of just reverting the appended lines.
//
// This test drives the real compiled ``MergeManager.js`` (no mocks of
// project code) with an in-memory ``vscode`` stub that performs real
// file deletions for ``WorkspaceEdit.deleteFile`` — exactly what the
// VS Code host does.
//
// Run directly with ``node`` (after ``npm run compile``):
//
//     node src/kiss/agents/vscode/test/bughunt_isNewFile.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

// ---------------------------------------------------------------------------
// In-memory text-document model shared by the vscode stub.
// ---------------------------------------------------------------------------

const docs = new Map(); // fsPath -> doc model

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
      return {
        text: t,
        range: {end: {line: i, character: t.length}},
      };
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
        delete: (range) => doc._applyDelete(range),
        insert: (pos, txt) => doc._applyInsert(pos, txt),
        replace: (range, txt) => doc._applyReplace(range, txt),
      });
      return Promise.resolve(true);
    },
  };
}

// ---------------------------------------------------------------------------
// vscode stub
// ---------------------------------------------------------------------------

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
  deleteFile(uri, _opts) {
    this._deletes.push(uri);
  }
}

const vscodeStub = {
  Uri: {file: (p) => ({fsPath: p, scheme: 'file'})},
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
    showTextDocument: (doc) => Promise.resolve(makeEditor(doc)),
    showInformationMessage: () => undefined,
    showWarningMessage: () => undefined,
  },
  workspace: {
    onWillSaveTextDocument: () => ({dispose() {}}),
    onDidSaveTextDocument: () => ({dispose() {}}),
    openTextDocument: (uri) => Promise.resolve(getDoc(uri.fsPath)),
    saveAll: () => {
      for (const d of docs.values()) {
        if (fs.existsSync(d.uri.fsPath)) d._save();
      }
      return Promise.resolve(true);
    },
    applyEdit: (edit) => {
      for (const op of edit._textOps) {
        const d = getDoc(op.uri.fsPath);
        if (op.kind === 'insert') d._applyInsert(op.pos, op.txt);
        else d._applyReplace(op.range, op.txt);
      }
      // Real VS Code deletes the file from disk for deleteFile().
      for (const uri of edit._deletes) {
        try {
          fs.rmSync(uri.fsPath, {force: true});
        } catch {
          /* ignore */
        }
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

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

const compiled = path.join(__dirname, '..', 'out', 'MergeManager.js');
assert.ok(
  fs.existsSync(compiled),
  `compiled extension missing: ${compiled} — run \`npm run compile\` first`,
);
delete require.cache[require.resolve(compiled)];
const {MergeManager} = require(compiled);

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-isnewfile-'));

async function testRejectFileOnExistingFileWithAppendOnlyHunk() {
  docs.clear();
  // Pre-existing file: the agent appended one line ("D") at the end.
  // The pre-task base copy (written by diff_merge._write_base_copy)
  // is NON-empty — proof the file existed before the task.
  const basePath = path.join(tmpDir, 'base-existing.txt');
  const curPath = path.join(tmpDir, 'existing.txt');
  fs.writeFileSync(basePath, 'A\nB\nC\n');
  fs.writeFileSync(curPath, 'A\nB\nC\nD\n');

  const mgr = new MergeManager();
  // Insertion-only hunk exactly as diff_merge emits it for an append:
  // bc=0 (no old lines) ⇒ the TS side computes oc=0 for every hunk.
  await mgr.openMerge({
    files: [
      {
        name: 'existing.txt',
        base: basePath,
        current: curPath,
        hunks: [{bs: 3, bc: 0, cs: 3, cc: 1}],
      },
    ],
  });

  let allDone = false;
  mgr.on('allDone', () => {
    allDone = true;
  });

  await mgr.rejectFile();
  // Let the saveAll/allDone promise chain settle.
  await new Promise((r) => setTimeout(r, 50));

  assert.ok(
    fs.existsSync(curPath),
    'BUG: rejecting an append-only change to a PRE-EXISTING file ' +
      'deleted the file from disk (isNewFile mis-classification)',
  );
  assert.strictEqual(
    fs.readFileSync(curPath, 'utf8'),
    'A\nB\nC\n',
    'rejecting the appended hunk must restore the pre-task content',
  );
  assert.ok(allDone, 'allDone must fire after the last file is resolved');
  mgr.dispose();
  console.log(
    '  ok - rejectFile on existing file with append-only hunk keeps the file',
  );
}

async function testRejectFileOnGenuinelyNewFileStillDeletes() {
  docs.clear();
  // Brand-new file: diff_merge._write_base_copy writes an EMPTY base
  // when git has no blob for the file.
  const basePath = path.join(tmpDir, 'base-new.txt');
  const curPath = path.join(tmpDir, 'new.txt');
  fs.writeFileSync(basePath, '');
  fs.writeFileSync(curPath, 'X\nY\n');

  const mgr = new MergeManager();
  await mgr.openMerge({
    files: [
      {
        name: 'new.txt',
        base: basePath,
        current: curPath,
        hunks: [{bs: 0, bc: 0, cs: 0, cc: 2}],
      },
    ],
  });

  await mgr.rejectFile();
  await new Promise((r) => setTimeout(r, 50));

  assert.ok(
    !fs.existsSync(curPath),
    'rejecting ALL changes of an agent-created file must delete it',
  );
  mgr.dispose();
  console.log('  ok - rejectFile on genuinely new file still deletes it');
}

async function testAcceptFileOnExistingFileKeepsAppendedLine() {
  docs.clear();
  const basePath = path.join(tmpDir, 'base-accept.txt');
  const curPath = path.join(tmpDir, 'accept.txt');
  fs.writeFileSync(basePath, 'A\nB\nC\n');
  fs.writeFileSync(curPath, 'A\nB\nC\nD\n');

  const mgr = new MergeManager();
  await mgr.openMerge({
    files: [
      {
        name: 'accept.txt',
        base: basePath,
        current: curPath,
        hunks: [{bs: 3, bc: 0, cs: 3, cc: 1}],
      },
    ],
  });

  await mgr.acceptFile();
  await new Promise((r) => setTimeout(r, 50));

  assert.ok(fs.existsSync(curPath), 'accepting must never delete the file');
  assert.strictEqual(
    fs.readFileSync(curPath, 'utf8'),
    'A\nB\nC\nD\n',
    'accepting the appended hunk must keep the new content',
  );
  mgr.dispose();
  console.log('  ok - acceptFile keeps the appended line and the file');
}

async function runTests() {
  await testRejectFileOnExistingFileWithAppendOnlyHunk();
  await testRejectFileOnGenuinelyNewFileStillDeletes();
  await testAcceptFileOnExistingFileKeepsAppendedLine();
}

runTests().then(
  () => {
    fs.rmSync(tmpDir, {recursive: true, force: true});
    console.log('\n3 passed, 0 failed');
    process.exit(0);
  },
  (err) => {
    console.error('FAIL:', err && err.message ? err.message : err);
    fs.rmSync(tmpDir, {recursive: true, force: true});
    process.exit(1);
  },
);
