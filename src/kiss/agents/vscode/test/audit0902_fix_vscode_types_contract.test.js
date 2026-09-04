// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Compile-time contract test for review-vscode.md #8: the messages the
// Python daemon emits must be assignable to `ToWebviewMessage`
// (src/types.ts), which media/main.js reads.  The union used to omit
// task_done.startTs/endTs, worktree_result.retryable, task_events.task_id
// / .extra, typed the replay chat_id as a number although the emitters
// send uuid strings, and had no task_settings variant at all.
//
// Round 2 (review2-vscode.md #4): `openSubagentTab` also lacked the
// emitted `task_id` (server.py) and the live `fileContent` reply
// (web_server.py _handle_open_file, consumed by main.js
// handleFileContent) had no variant at all; both payload shapes are
// checked below.
//
// The task_settings payload is produced by the REAL Python synthesiser
// (kiss.server.json_printer._task_settings_event_from_session, which the
// replay path calls) through `uv run python`; the remaining payloads are
// the literal dicts the emitters build, with their source lines cited.
// Each one is written into a scratch .ts file as `... satisfies
// ToWebviewMessage` and type-checked with the project's own tsc and
// tsconfig options: a field the union does not declare, or a wrong type,
// fails the compile and therefore this test.

/* global require, process, console, __dirname */

'use strict';

const assert = require('assert');
const {execFileSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const EXT_DIR = path.join(__dirname, '..');
const TYPES_TS = path.join(EXT_DIR, 'src', 'types.ts');
const KISS_PROJECT = path.resolve(EXT_DIR, '..', '..', '..', '..');
// kissPaths.js imports the vscode API module, which only exists inside an
// extension host; the stub stands in for it.
global.__kissVscodeStub = {workspace: {workspaceFolders: []}};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, ...rest);
};
const {findUvPath} = require(path.join(EXT_DIR, 'out', 'kissPaths.js'));
const UV = findUvPath();
assert.ok(UV, 'this suite needs a real uv binary');

const PY = `
import json
from kiss.server.json_printer import _task_settings_event_from_session
session = {
    "task": "t", "task_id": "0123abcd", "chat_id": "chat-uuid",
    "events": [],
    "extra": json.dumps({
        "model": "m", "work_dir": "/w", "is_parallel": True,
        "is_worktree": False, "startTs": 1700000000000, "max_budget": 2.5,
        "subagent": {"parent_task_id": "parent-1"},
    }),
}
print(json.dumps(_task_settings_event_from_session(session)))
`;

function pythonTaskSettings() {
  const out = execFileSync(UV, ['run', 'python', '-c', PY], {
    cwd: KISS_PROJECT,
    encoding: 'utf-8',
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  const ev = JSON.parse(out);
  assert.strictEqual(ev.type, 'task_settings');
  assert.strictEqual(ev.settings.chat_id, 'chat-uuid');
  assert.strictEqual(ev.settings.parent_task_id, 'parent-1');
  return ev;
}

// The literal dicts the daemon broadcasts.
const EMITTED = {
  // task_runner.py: {**task_end_event, "tabId", "startTs", "endTs"} for
  // task_done / task_error / task_stopped / task_interrupted.
  task_done: {type: 'task_done', tabId: 'tab-1', startTs: 1, endTs: 2},
  task_error: {
    type: 'task_error',
    text: 'boom',
    tabId: 'tab-1',
    startTs: 1,
    endTs: 2,
  },
  task_stopped: {type: 'task_stopped', tabId: 'tab-1', startTs: 1, endTs: 2},
  task_interrupted: {
    type: 'task_interrupted',
    tabId: 'tab-1',
    startTs: 1,
    endTs: 2,
  },
  // task_runner.py: re-announce of the overridden chat / viewer clear.
  clear: {type: 'clear', chat_id: 'chat-uuid', tabId: 'tab-1'},
  // server.py: live replay (no history row yet) and stored replay.
  task_events_live: {
    type: 'task_events',
    events: [],
    task: '',
    task_id: 'task-1',
    chat_id: 'chat-uuid',
    extra: '',
    tabId: 'tab-1',
  },
  task_events_stored: {
    type: 'task_events',
    events: [{type: 'prompt', text: 'x'}],
    task: 'do it',
    task_id: null,
    chat_id: 'chat-uuid',
    extra: '{"model":"m"}',
    tabId: 'tab-1',
  },
  // merge_flow.py: retryable failures keep the Merge / Discard bar.
  worktree_result: {
    type: 'worktree_result',
    tabId: 'tab-1',
    success: false,
    message: 'Discard deferred',
    retryable: true,
  },
  // server.py _handle_ready (subagent_info branch): task_id is the
  // history row's id or null; main.js maps null/undefined to ''.
  openSubagentTab_resumed: {
    type: 'openSubagentTab',
    tab_id: 'tab-1__sub_abc',
    parent_tab_id: 'tab-1',
    description: 'sub task',
    task_id: null,
    isSubagentTab: true,
    isDone: true,
  },
  // server.py _announce_subagent_tabs: one per stored sub-agent row.
  openSubagentTab_announced: {
    type: 'openSubagentTab',
    tab_id: 'tab-1__sub_abc',
    parent_tab_id: 'tab-1',
    description: 'sub task',
    task_id: 'abc',
    taskIndex: 0,
    isSubagentTab: true,
    isDone: false,
  },
  // web_server.py _handle_open_file: the success and error replies.
  fileContent_ok: {
    type: 'fileContent',
    path: '/w/README.md',
    name: 'README.md',
    tabId: 'tab-1',
    content: '# hi\n',
  },
  fileContent_error: {
    type: 'fileContent',
    path: 'missing.md',
    name: 'missing.md',
    tabId: '',
    error: 'File not found: missing.md',
  },
};

function main() {
  const payloads = {...EMITTED, task_settings: pythonTaskSettings()};
  const scratch = fs.mkdtempSync(
    path.join(os.tmpdir(), 'kiss-types-contract-'),
  );
  try {
    const lines = [
      `import {ToWebviewMessage} from ${JSON.stringify(TYPES_TS.replace(/\.ts$/, ''))};`,
      'export const contract = {',
    ];
    for (const [name, payload] of Object.entries(payloads)) {
      lines.push(
        `  ${name}: ${JSON.stringify(payload)} satisfies ToWebviewMessage,`,
      );
    }
    lines.push('};', '');
    const file = path.join(scratch, 'contract.ts');
    fs.writeFileSync(file, lines.join('\n'));
    const tsc = path.join(EXT_DIR, 'node_modules', 'typescript', 'bin', 'tsc');
    let output = '';
    try {
      output = execFileSync(
        process.execPath,
        [
          tsc,
          '--noEmit',
          '--strict',
          '--target',
          'ES2022',
          '--module',
          'Node16',
          '--moduleResolution',
          'Node16',
          '--skipLibCheck',
          file,
        ],
        {encoding: 'utf-8', stdio: ['ignore', 'pipe', 'pipe']},
      );
    } catch (err) {
      const out = (err.stdout || '') + (err.stderr || '');
      assert.fail(`emitted payloads are not ToWebviewMessage:\n${out}`);
    }
    assert.strictEqual(output.trim(), '');
    console.log(
      `  ✓ ${Object.keys(payloads).length} emitted payloads satisfy ToWebviewMessage`,
    );
  } finally {
    fs.rmSync(scratch, {recursive: true, force: true});
  }
  console.log('audit0902_fix_vscode_types_contract: all tests passed');
}

main();
