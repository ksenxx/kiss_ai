// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/**
 * A fake kiss-web daemon for the extension's tests.
 *
 * The production clients (`AgentClient`, `daemonHealth`) find the daemon
 * through its endpoint file and speak WebSocket to the URL it names.
 * This helper is a plain-TCP (`ws://`) RFC 6455 server on an ephemeral
 * loopback port that writes such a file, completes the local `auth`
 * handshake itself, and hands each authenticated connection to the
 * test as a line-oriented duplex: `conn.write(jsonLine)` sends one text
 * frame, `conn.on('data', buf)` delivers each incoming frame as
 * `Buffer(text + '\n')`, so tests read and write newline-delimited JSON
 * exactly as they did against the old socket.
 *
 * `createFakeDaemon(onConnection, opts)` returns a server with
 * `listen(endpointPath, cb)`, `close(cb)`, `address()`, `connections`,
 * `token`, `url` and `writeEndpoint()`.  `opts.auth` chooses how the
 * `auth` frame is answered: `'accept'` (default, `auth_ok local:true`,
 * then `onConnection`), `'remote'` (`auth_ok local:false`), `'reject'`
 * (`auth_required`) or `'silent'` (never answered — for timeout tests).
 * `opts.tokenCheck` (default true) makes a wrong token get
 * `auth_required` regardless of `opts.auth`.
 */

const crypto = require('crypto');
const fs = require('fs');
const net = require('net');
const path = require('path');
const {EventEmitter} = require('events');

const WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

/** The endpoint file path a fake daemon writes: `<dir>/<name>`. */
function fakeEndpointPath(dir, name) {
  return path.join(dir, name);
}

/** Build one unmasked server frame. */
function frame(opcode, payload) {
  const len = payload.length;
  let header;
  if (len < 126) {
    header = Buffer.from([0x80 | opcode, len]);
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x80 | opcode;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x80 | opcode;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payload]);
}

/** One accepted WebSocket connection, presented as a line duplex. */
class FakeConnection extends EventEmitter {
  constructor(socket, server) {
    super();
    this.socket = socket;
    this.server = server;
    this.authenticated = false;
    this.remoteAddress = socket.remoteAddress;
    this.destroyed = false;
    this._buf = Buffer.alloc(0);
    this._fragments = [];
    this._handshook = false;
    socket.on('data', d => this._onData(d));
    socket.on('close', () => {
      this.destroyed = true;
      this.emit('close');
    });
    socket.on('error', () => {});
  }

  /** Send `text` (one JSON line; a trailing newline is stripped) as a text frame. */
  write(text) {
    if (this.destroyed || !this._handshook) return false;
    let s = Buffer.isBuffer(text) ? text.toString('utf8') : String(text);
    if (s.endsWith('\n')) s = s.slice(0, -1);
    this.socket.write(frame(0x1, Buffer.from(s, 'utf8')));
    return true;
  }

  /** Send a raw (unmasked) frame with the given opcode. */
  writeFrame(opcode, payload) {
    if (this.destroyed) return;
    this.socket.write(frame(opcode, payload));
  }

  /** Write raw bytes to the TCP socket, bypassing framing (protocol tests). */
  writeRaw(buf) {
    if (this.destroyed) return;
    this.socket.write(buf);
  }

  /** Close the connection with a WebSocket close frame, then end the socket. */
  end() {
    if (this.destroyed) return;
    if (this._handshook) {
      const payload = Buffer.alloc(2);
      payload.writeUInt16BE(1000, 0);
      try {
        this.socket.write(frame(0x8, payload));
      } catch {}
    }
    this.socket.end();
    this.emit('end');
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.socket.destroy();
  }

  setEncoding() {}
  setNoDelay() {}

  /** Stop reading from the peer (a wedged daemon that never drains). */
  pause() {
    this.socket.pause();
  }

  resume() {
    this.socket.resume();
  }

  _onData(data) {
    this._buf = Buffer.concat([this._buf, data]);
    if (!this._handshook) {
      const end = this._buf.indexOf('\r\n\r\n');
      if (end < 0) return;
      const head = this._buf.subarray(0, end).toString('latin1');
      this._buf = this._buf.subarray(end + 4);
      const m = /sec-websocket-key:\s*(\S+)/i.exec(head);
      if (!m) {
        this.socket.end('HTTP/1.1 400 Bad Request\r\n\r\n');
        return;
      }
      const accept = crypto
        .createHash('sha1')
        .update(m[1] + WS_GUID)
        .digest('base64');
      this.socket.write(
        'HTTP/1.1 101 Switching Protocols\r\n' +
          'Upgrade: websocket\r\n' +
          'Connection: Upgrade\r\n' +
          `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
      );
      this._handshook = true;
      this.server.emit('handshake', this);
    }
    for (;;) {
      const buf = this._buf;
      if (buf.length < 2) return;
      const fin = (buf[0] & 0x80) !== 0;
      const opcode = buf[0] & 0x0f;
      const masked = (buf[1] & 0x80) !== 0;
      let len = buf[1] & 0x7f;
      let offset = 2;
      if (len === 126) {
        if (buf.length < 4) return;
        len = buf.readUInt16BE(2);
        offset = 4;
      } else if (len === 127) {
        if (buf.length < 10) return;
        len = Number(buf.readBigUInt64BE(2));
        offset = 10;
      }
      const maskLen = masked ? 4 : 0;
      if (buf.length < offset + maskLen + len) return;
      let payload = Buffer.from(
        buf.subarray(offset + maskLen, offset + maskLen + len),
      );
      if (masked) {
        const mask = buf.subarray(offset, offset + 4);
        for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
      }
      this._buf = buf.subarray(offset + maskLen + len);
      this._onFrame(fin, opcode, payload);
    }
  }

  _onFrame(fin, opcode, payload) {
    if (opcode === 0x9) {
      this.writeFrame(0xa, payload);
      return;
    }
    if (opcode === 0xa) return;
    if (opcode === 0x8) {
      if (!this.destroyed) {
        try {
          this.socket.write(frame(0x8, payload.subarray(0, 2)));
        } catch {}
        this.socket.end();
      }
      return;
    }
    if (opcode === 0x1 || opcode === 0x2 || opcode === 0x0) {
      this._fragments.push(payload);
      if (!fin) return;
      const text = Buffer.concat(this._fragments).toString('utf8');
      this._fragments = [];
      this._onMessage(text);
    }
  }

  _onMessage(text) {
    if (!this.authenticated) {
      this.server._handleAuth(this, text);
      return;
    }
    this.emit('data', Buffer.from(text + '\n', 'utf8'));
  }
}

class FakeDaemon extends EventEmitter {
  constructor(onConnection, opts) {
    super();
    this.opts = opts || {};
    this.onConnection = onConnection || (() => {});
    this.token = this.opts.token || crypto.randomBytes(16).toString('hex');
    this.connections = new Set();
    this.endpointPath = null;
    this.port = 0;
    this.url = '';
    this._server = net.createServer(socket => {
      socket.setNoDelay(true);
      const conn = new FakeConnection(socket, this);
      this.connections.add(conn);
      conn.on('close', () => this.connections.delete(conn));
      this.emit('connection', conn);
    });
    this._server.on('error', err => this.emit('error', err));
  }

  /**
   * Bind an ephemeral loopback port and publish `endpointPath`.
   * `cb(err)` follows `net.Server#listen`'s callback convention.
   */
  listen(endpointPath, cb) {
    this.endpointPath = endpointPath;
    let writeError = null;
    const publish = () => {
      if (this.url) return;
      this.port = this._server.address().port;
      this.url = `ws://127.0.0.1:${this.port}/ws`;
      try {
        this.writeEndpoint();
      } catch (err) {
        writeError = err;
      }
    };
    this._server.listen(0, '127.0.0.1', () => {
      publish();
      if (cb) cb(writeError || undefined);
    });
    // Node binds a loopback TCP listener synchronously inside listen(),
    // so the endpoint file can be published before this returns — the
    // way a Unix socket file appeared as soon as listen() was called.
    // Callers that connect right after listen() (without awaiting the
    // callback) therefore find the daemon on their first attempt.
    if (this._server.address()) publish();
    return this;
  }

  /** (Re)write the endpoint file for this daemon (default: `this.endpointPath`). */
  writeEndpoint(endpointPath) {
    const target = endpointPath || this.endpointPath;
    fs.mkdirSync(path.dirname(target), {recursive: true});
    fs.writeFileSync(
      target,
      JSON.stringify({
        url: this.url,
        token: this.token,
        ca: null,
        pid: this.opts.pid === undefined ? process.pid : this.opts.pid,
      }),
    );
  }

  address() {
    return this._server.address();
  }

  /**
   * Stop accepting connections and remove the endpoint file, like
   * `net.Server#close` on a Unix socket (libuv unlinks the path).
   * Existing connections survive; `cb` runs once they have all ended.
   * Use {@link destroyConnections} to end them.
   */
  close(cb) {
    if (this.endpointPath && !this.opts.keepEndpoint) {
      try {
        fs.unlinkSync(this.endpointPath);
      } catch {}
    }
    this._server.close(cb);
    return this;
  }

  /** Destroy every live connection. */
  destroyConnections() {
    for (const conn of this.connections) conn.destroy();
  }

  _handleAuth(conn, text) {
    let msg;
    try {
      msg = JSON.parse(text);
    } catch {
      msg = null;
    }
    const mode = this.opts.auth || 'accept';
    if (mode === 'silent') return;
    const tokenOk =
      this.opts.tokenCheck === false ||
      (msg && msg.type === 'auth' && msg.token === this.token);
    if (!tokenOk || mode === 'reject') {
      conn.write(JSON.stringify({type: 'auth_required'}));
      return;
    }
    conn.authenticated = true;
    conn.write(JSON.stringify({type: 'auth_ok', local: mode !== 'remote'}));
    this.onConnection(conn);
  }
}

/**
 * Create a fake daemon.  `onConnection(conn)` runs once a connection has
 * authenticated; see the module comment for `opts`.
 */
function createFakeDaemon(onConnection, opts) {
  return new FakeDaemon(onConnection, opts);
}

/**
 * Write an endpoint file that points at nothing listening on `port`
 * (a stale file left by a dead daemon) and return its path.
 */
function writeStaleEndpoint(endpointPath, port) {
  fs.mkdirSync(path.dirname(endpointPath), {recursive: true});
  fs.writeFileSync(
    endpointPath,
    JSON.stringify({
      url: `ws://127.0.0.1:${port}/ws`,
      token: 'stale',
      ca: null,
      pid: 0,
    }),
  );
  return endpointPath;
}

/** Bind and immediately release a loopback port so nothing listens on it. */
function freePort() {
  return new Promise(resolve => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
  });
}

module.exports = {
  createFakeDaemon,
  fakeEndpointPath,
  writeStaleEndpoint,
  freePort,
};
