// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

global.__kissVscodeStub = {};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

const sourcePath = path.join(__dirname, '..', 'out', 'SorcarTab.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
);
delete require.cache[require.resolve(sourcePath)];
const {readSampleTasks} = require(sourcePath);
assert.strictEqual(
  typeof readSampleTasks,
  'function',
  'readSampleTasks must be exported from the compiled SorcarTab',
);

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  FAIL - ${name}: ${err && err.message}`);
  }
}

function mkExt() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sample-tasks-'));
}

function packageSampleTasksPath(ext) {
  return path.join(
    ext,
    'kiss_project',
    'src',
    'kiss',
    'SAMPLE_TASKS.md',
  );
}

function writePackageSampleTasks(ext, content) {
  const file = packageSampleTasksPath(ext);
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, content);
}

function withTempKissHome(fn) {
  const ext = mkExt();
  const kissHome = mkExt();
  const prev = process.env.KISS_HOME;
  process.env.KISS_HOME = kissHome;
  try {
    fn(ext, kissHome);
  } finally {
    if (prev === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prev;
    fs.rmSync(ext, {recursive: true, force: true});
    fs.rmSync(kissHome, {recursive: true, force: true});
  }
}

test('auto-creates MY_TASK_TEMPLATES.md with seed "Hi!" when missing', () => {
  withTempKissHome((ext, kissHome) => {
    const myTasks = path.join(kissHome, 'MY_TASK_TEMPLATES.md');
    assert.ok(!fs.existsSync(myTasks), 'user copy must start absent');
    const tasks = readSampleTasks(ext);
    assert.ok(
      fs.existsSync(myTasks),
      'reader must create ~/.kiss/MY_TASK_TEMPLATES.md when missing',
    );
    assert.strictEqual(
      fs.readFileSync(myTasks, 'utf-8'),
      '## Task\n\nHi!\n',
      'seed content must be a single Hi! Task section',
    );
    assert.deepStrictEqual(tasks, [{text: 'Hi!'}]);
  });
});

test('does NOT copy SAMPLE_TASKS.md into ~/.kiss/', () => {
  withTempKissHome((ext, kissHome) => {
    writePackageSampleTasks(ext, '## Task\n\nPackage-only chip\n');
    const tasks = readSampleTasks(ext);
    assert.deepStrictEqual(tasks, [
      {text: 'Hi!'},
      {text: 'Package-only chip'},
    ]);
    assert.ok(
      !fs.existsSync(path.join(kissHome, 'SAMPLE_TASKS.md')),
      '~/.kiss/SAMPLE_TASKS.md must never be seeded by readSampleTasks',
    );
  });
});

test('MY_TASK_TEMPLATES.md chips appear before SAMPLE_TASKS.md chips', () => {
  withTempKissHome((ext, kissHome) => {
    fs.mkdirSync(kissHome, {recursive: true});
    fs.writeFileSync(
      path.join(kissHome, 'MY_TASK_TEMPLATES.md'),
      '## Task\n\nMy first task\n\n## Task\n\nMy second task\n',
    );
    writePackageSampleTasks(
      ext,
      '## Task\n\nBundled A\n\n## Task\n\nBundled B\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'My first task'},
      {text: 'My second task'},
      {text: 'Bundled A'},
      {text: 'Bundled B'},
    ]);
  });
});

test('preserves user edits to MY_TASK_TEMPLATES.md across reads', () => {
  withTempKissHome((ext, kissHome) => {
    fs.mkdirSync(kissHome, {recursive: true});
    const myTasks = path.join(kissHome, 'MY_TASK_TEMPLATES.md');
    fs.writeFileSync(myTasks, '## Task\n\nCurated chip\n');
    writePackageSampleTasks(ext, '## Task\n\nFresh bundled\n');
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'Curated chip'},
      {text: 'Fresh bundled'},
    ]);
    assert.strictEqual(
      fs.readFileSync(myTasks, 'utf-8'),
      '## Task\n\nCurated chip\n',
    );
  });
});

test('returns only MY_TASK_TEMPLATES.md chips when package SAMPLE_TASKS.md is missing', () => {
  withTempKissHome((ext, kissHome) => {
    fs.mkdirSync(kissHome, {recursive: true});
    fs.writeFileSync(
      path.join(kissHome, 'MY_TASK_TEMPLATES.md'),
      '## Task\n\nOnly mine\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'Only mine'}]);
  });
});

test('returns only seed Hi! + bundled chips when MY_TASK_TEMPLATES.md was empty', () => {
  withTempKissHome((ext, kissHome) => {
    fs.mkdirSync(kissHome, {recursive: true});
    fs.writeFileSync(path.join(kissHome, 'MY_TASK_TEMPLATES.md'), '');
    writePackageSampleTasks(ext, '## Task\n\nBundled only\n');
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'Bundled only'}]);
  });
});

test('parses multiple ## Task sections in source order', () => {
  withTempKissHome(ext => {
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '',
    );
    writePackageSampleTasks(
      ext,
      '## Task\n\nFirst task\n\n## Task\n\nSecond task\n\n## Task\n\nThird\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'First task'},
      {text: 'Second task'},
      {text: 'Third'},
    ]);
  });
});

test('preserves multi-line bodies and inline punctuation', () => {
  withTempKissHome(ext => {
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '',
    );
    const body =
      'Line one with **bold**, "quotes", and a <<placeholder>>.\n' +
      'Line two continues the same task.';
    writePackageSampleTasks(ext, `## Task\n\n${body}\n`);
    assert.deepStrictEqual(readSampleTasks(ext), [{text: body}]);
  });
});

test('skips sections whose heading is not Task', () => {
  withTempKissHome(ext => {
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '## Intro\n\nignored\n\n## Task\n\nmy kept\n',
    );
    writePackageSampleTasks(
      ext,
      '## Intro\n\nignored\n\n## Task\n\nkept\n\n## Notes\n\nalso ignored\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'my kept'},
      {text: 'kept'},
    ]);
  });
});

test('skips empty-bodied ## Task sections', () => {
  withTempKissHome(ext => {
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '',
    );
    writePackageSampleTasks(ext, '## Task\n\n\n## Task\n\nreal body\n');
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'real body'}]);
  });
});

test('unescapes mdformat backslash escapes (\\<< -> <<)', () => {
  withTempKissHome(ext => {
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '',
    );
    writePackageSampleTasks(
      ext,
      '## Task\n\nRun on \\<<your dataset>> with **bold** \\*literal\\*\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'Run on <<your dataset>> with **bold** *literal*'},
    ]);
  });
});

test('skips user chips when ~/.kiss/ is unwritable (ensureUserAssetFromDefault returns null)', () => {
  if (process.getuid && process.getuid() === 0) {
    console.log('  ok - SKIPPED (root) - returns only bundled chips when ~/.kiss/ is unwritable');
    passed += 1;
    return;
  }
  withTempKissHome((ext, kissHome) => {
    fs.mkdirSync(kissHome, {recursive: true});
    fs.chmodSync(kissHome, 0o500);
    try {
      writePackageSampleTasks(ext, '## Task\n\nOnly bundled\n');
      assert.deepStrictEqual(readSampleTasks(ext), [{text: 'Only bundled'}]);
      assert.ok(
        !fs.existsSync(path.join(kissHome, 'MY_TASK_TEMPLATES.md')),
        'seed file must not be created on read-only ~/.kiss/',
      );
    } finally {
      fs.chmodSync(kissHome, 0o700);
    }
  });
});

test('falls back to dev-checkout SAMPLE_TASKS.md when packaged copy is absent', () => {
  const prev = process.env.KISS_HOME;
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-dev-fallback-'));
  const ext = path.join(root, 'a', 'b');
  fs.mkdirSync(ext, {recursive: true});
  const devFile = path.join(root, 'SAMPLE_TASKS.md');
  fs.writeFileSync(devFile, '## Task\n\nDev checkout chip\n');
  const kissHome = path.join(root, '.kiss');
  fs.mkdirSync(kissHome);
  fs.writeFileSync(path.join(kissHome, 'MY_TASK_TEMPLATES.md'), '');
  process.env.KISS_HOME = kissHome;
  try {
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'Dev checkout chip'},
    ]);
  } finally {
    if (prev === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prev;
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test('shipped SAMPLE_TASKS.md tasks never contain a leading backslash before <<', () => {
  withTempKissHome(() => {
    const ext = path.join(__dirname, '..');
    const tasks = readSampleTasks(ext);
    for (const t of tasks) {
      assert.ok(
        !/\\</.test(t.text),
        `task should not retain mdformat escapes: ${JSON.stringify(t.text)}`,
      );
    }
  });
});

test('parses the shipped SAMPLE_TASKS.md (sanity)', () => {
  withTempKissHome(() => {
    const ext = path.join(__dirname, '..');
    const shipped = path.join(ext, '..', '..', 'SAMPLE_TASKS.md');
    assert.ok(
      fs.existsSync(shipped),
      `shipped SAMPLE_TASKS.md missing at ${shipped}`,
    );
    const tasks = readSampleTasks(ext);
    assert.ok(Array.isArray(tasks), 'must return an array');
    assert.ok(tasks.length > 0, 'shipped file must contain at least one task');
    for (const t of tasks) {
      assert.strictEqual(typeof t.text, 'string');
      assert.ok(t.text.length > 0, 'each task body must be non-empty');
      assert.strictEqual(t.text, t.text.trim());
    }
  });
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`\n${f.name}:\n`, f.err);
  }
  process.exit(1);
}
