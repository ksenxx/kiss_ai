// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Shared fixture of the suites that drive a chat tab picked onto a
// registered machine: the machine registry (MACHINES) and a page whose
// daemon knows its own folder, picked onto gpu-box (bootOnGpuBox).

'use strict';

const {makePage, send} = require('./workdir_helpers');
const {pickMachine} = require('./simplify2_harness');

/** The daemon's machine registry: this laptop and a reachable gpu-box. */
const MACHINES = [
  {
    name: 'laptop',
    local: true,
    status: 'reachable',
    address: 'https://10.0.0.5:8787',
  },
  {name: 'gpu-box', status: 'reachable', address: 'https://gpu:8787'},
];
const BOX_DIRS = {
  type: 'machineWorkDirs',
  machine: 'gpu-box',
  workDir: '/srv/proj',
  recentWorkDirs: [{path: '/srv/proj', ts: Date.now() / 1000}],
  error: '',
};

/**
 * A page whose daemon knows its own folder and the machines, picked
 * onto gpu-box (whose folder is /srv/proj).  The messages the boot
 * posted are cleared, so `posted` holds only what the test drives.
 *
 * @returns {{win: object, posted: Array<object>}}
 */
function bootOnGpuBox() {
  const made = makePage();
  send(made.win, {
    type: 'configData',
    config: {work_dir: '/work/ws', recent_work_dirs: []},
    machine: 'laptop',
  });
  send(made.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  pickMachine(made.win, 'gpu-box');
  send(made.win, BOX_DIRS);
  made.posted.length = 0;
  return made;
}

module.exports = {MACHINES, bootOnGpuBox};
