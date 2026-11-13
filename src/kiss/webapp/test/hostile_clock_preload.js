// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// `node -r` preload of suitesUnderHostileClock.test.js: every read of a
// jsdom window's clock comes 25 hours after the previous one, as if the
// page had stalled in between, past midnight.  A suite that freezes its
// pages' clock and dates its fixtures from the same instant never sees it.

'use strict';

const jsdom = require('jsdom');

const STALL_MS = 25 * 3600 * 1000;

jsdom.JSDOM = class JSDOM extends jsdom.JSDOM {
  constructor(...args) {
    super(...args);
    let now = Date.now();
    const stalled = () => (now += STALL_MS);
    this.window.Date = class Date extends this.window.Date {
      constructor(...given) {
        if (given.length) super(...given);
        else super(stalled());
      }
      static now() {
        return stalled();
      }
    };
  }
};
