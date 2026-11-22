// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate of Projects and of the default working directory: every
// byte of every code line of media/projects.js (the Projects page) and
// of the regions of media/main.js fenced by `// projects-coverage:start`
// / `// projects-coverage:end` (the "Default working directory"
// drop-downs, the chat header's project drop-down, the Projects tab, and
// the project panels of the Chats list) must have run, so an untaken arm
// of a `?:`, `||` or `&&` fails the gate as an untested line does.
//
// The three functional jsdom suites are run in turn and their coverage
// is added up: test/projectsPage.test.js (the page and the header),
// test/historyProjects.test.js (the Chats list) and
// test/settingsWorkDirField.test.js (the Settings sheet's drop-down).

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: [
    'projectsPage.test.js',
    'historyProjects.test.js',
    'settingsWorkDirField.test.js',
  ],
  mark: 'projects',
  label: 'projects',
  everyByte: true,
  targets: [
    {file: 'main.js', sourceUrl: 'projects-main.js', regions: 3},
    {file: 'projects.js', sourceUrl: 'projects-page.js', wholeFile: true},
  ],
});
