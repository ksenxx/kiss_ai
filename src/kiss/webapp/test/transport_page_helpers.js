// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// Pieces for booting chat.html on the REAL webapp transport
// (media/transport.js) over a fake WebSocket, so the frames the daemon
// would see are exactly the ones a suite inspects.

const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/** chat.html with its placeholders filled and its script tags stripped. */
function chatHtml() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  return html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
}

/** A fresh jsdom window on chat.html with no page script loaded yet. */
function newDom() {
  const dom = new JSDOM(chatHtml(), {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  return win;
}

/** Eval panelCopy.js, api.js and main.js into *win*, in that order. */
function loadPageScripts(win) {
  for (const f of ['panelCopy.js', 'api.js', 'main.js']) {
    win.eval(fs.readFileSync(path.join(MEDIA, f), 'utf8'));
  }
}

/** The source of media/transport.js (the webapp shim). */
function readShimJs() {
  return fs.readFileSync(path.join(MEDIA, 'transport.js'), 'utf8');
}

/**
 * Replace *win*.WebSocket with a fake that records every socket it opens
 * into *sockets* and every frame sent on it into `socket.sent`.
 */
function installFakeWebSocket(win, sockets) {
  function FakeWebSocket(url) {
    this.url = url;
    this.readyState = FakeWebSocket.OPEN;
    this.sent = [];
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
    sockets.push(this);
  }
  FakeWebSocket.CONNECTING = 0;
  FakeWebSocket.OPEN = 1;
  FakeWebSocket.CLOSING = 2;
  FakeWebSocket.CLOSED = 3;
  FakeWebSocket.prototype.send = function (data) {
    this.sent.push(data);
  };
  FakeWebSocket.prototype.close = function () {
    this.readyState = FakeWebSocket.CLOSED;
    if (typeof this.onclose === 'function') this.onclose();
  };
  win.WebSocket = FakeWebSocket;
}

module.exports = {
  chatHtml,
  newDom,
  loadPageScripts,
  readShimJs,
  installFakeWebSocket,
};
