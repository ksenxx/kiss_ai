// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

/**
 * Minimal RFC 6455 WebSocket client over `tls`/`net`.
 *
 * The extension ships without `node_modules` (see
 * `scripts/package-vsix.js`), so it cannot depend on `ws`, and the
 * `WebSocket` global is not available in every VS Code Node runtime.
 * This client covers exactly what talking to the kiss-web daemon
 * needs: the opening handshake with `Sec-WebSocket-Accept`
 * verification, text frames in both directions (fragmented incoming
 * messages are reassembled), ping/pong, and the closing handshake.  No
 * extensions are negotiated (permessage-deflate saves nothing on
 * loopback and costs daemon CPU), so RSV bits must be zero.
 *
 * `wss://` connections verify the server against the CA passed in
 * `ca` (the daemon's local CA from `~/.kiss/tls/ca.pem`); `ws://` is
 * accepted for tests.
 */

import * as crypto from 'crypto';
import * as net from 'net';
import * as tls from 'tls';
import {EventEmitter} from 'events';

const WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const DEFAULT_CONNECT_TIMEOUT_MS = 10_000;
const CLOSE_GRACE_MS = 1_000;
const utf8 = new TextDecoder('utf-8', {fatal: true});
/** Matches the daemon's frame limit (`web_server._MAX_LINE_BYTES`). */
const DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024;

const OP_CONTINUATION = 0x0;
const OP_TEXT = 0x1;
const OP_BINARY = 0x2;
const OP_CLOSE = 0x8;
const OP_PING = 0x9;
const OP_PONG = 0xa;

export interface WsClientOptions {
  /** `wss://host:port/path` or `ws://host:port/path`. */
  url: string;
  /** PEM certificate(s) to trust for `wss://`; undefined = system store. */
  ca?: string | Buffer;
  connectTimeoutMs?: number;
  maxMessageBytes?: number;
}

type WsReadyState = 'connecting' | 'open' | 'closing' | 'closed';

/**
 * One WebSocket connection, single-use: `connect()` once, then
 * `destroy()`; a closed instance cannot be reconnected.  Events: `open`,
 * `message` (string), `close` ({code, reason}), `error` (Error).  `error`
 * is always followed by `close`; `close` fires exactly once.
 */
export class WsClient extends EventEmitter {
  private _socket: net.Socket | null = null;
  private _state: WsReadyState = 'connecting';
  private _buf: Buffer = Buffer.alloc(0);
  private _key = '';
  private _fragments: Buffer[] = [];
  private _fragmentBytes = 0;
  private _fragmentOpcode = 0;
  private _closeSent = false;
  private _closeTimer: ReturnType<typeof setTimeout> | null = null;
  private _connectTimer: ReturnType<typeof setTimeout> | null = null;
  private _pendingClose: {code: number; reason: string} | null = null;
  private readonly _url: URL;
  private readonly _ca: string | Buffer | undefined;
  private readonly _connectTimeoutMs: number;
  private readonly _maxMessageBytes: number;

  constructor(options: WsClientOptions) {
    super();
    this._url = new URL(options.url);
    this._ca = options.ca;
    this._connectTimeoutMs =
      options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS;
    this._maxMessageBytes =
      options.maxMessageBytes ?? DEFAULT_MAX_MESSAGE_BYTES;
  }

  /** Open the TCP/TLS connection and start the WebSocket handshake. */
  connect(): void {
    // Single-use: once closed the instance stays closed.
    if (this._socket || this._state !== 'connecting') return;
    const secure = this._url.protocol === 'wss:';
    if (!secure && this._url.protocol !== 'ws:') {
      this._fail(new Error(`unsupported URL scheme ${this._url.protocol}`));
      return;
    }
    const host = this._url.hostname.replace(/^\[|\]$/g, '');
    const port = Number(this._url.port) || (secure ? 443 : 80);
    const onConnected = () => this._sendHandshake();
    const sock = secure
      ? tls.connect(
          {
            host,
            port,
            servername: net.isIP(host) ? undefined : host,
            ca: this._ca === undefined ? undefined : [this._ca],
            rejectUnauthorized: true,
          },
          onConnected,
        )
      : net.connect({host, port}, onConnected);
    this._socket = sock;
    sock.setNoDelay(true);
    sock.on('data', (data: Buffer) => this._onData(data));
    sock.on('error', (err: Error) => this._emitError(err));
    sock.on('close', () => this._onSocketClosed());
    this._connectTimer = setTimeout(() => {
      this._connectTimer = null;
      if (this._state === 'connecting') {
        this._fail(new Error('WebSocket connect timed out'));
      }
    }, this._connectTimeoutMs);
  }

  /**
   * Send one text message.  Returns false (and sends nothing) unless the
   * connection is open.
   */
  send(text: string): boolean {
    if (this._state !== 'open' || !this._socket) return false;
    this._socket.write(this._frame(OP_TEXT, Buffer.from(text, 'utf8')));
    return true;
  }

  /** Tear the connection down immediately, without a closing handshake. */
  destroy(): void {
    const sock = this._socket;
    if (sock) {
      try {
        sock.destroy();
      } catch {}
    }
    this._onSocketClosed();
  }

  /** Emit `error` only when someone listens: an unhandled `error` event throws. */
  private _emitError(err: Error): void {
    if (this.listenerCount('error') > 0) this.emit('error', err);
  }

  private _fail(err: Error): void {
    this._emitError(err);
    this.destroy();
  }

  private _sendHandshake(): void {
    if (!this._socket) return;
    this._key = crypto.randomBytes(16).toString('base64');
    const hostHeader = this._url.host;
    const target = this._url.pathname + this._url.search || '/';
    const req =
      `GET ${target} HTTP/1.1\r\n` +
      `Host: ${hostHeader}\r\n` +
      'Upgrade: websocket\r\n' +
      'Connection: Upgrade\r\n' +
      `Sec-WebSocket-Key: ${this._key}\r\n` +
      'Sec-WebSocket-Version: 13\r\n' +
      '\r\n';
    this._socket.write(req);
  }

  private _onData(data: Buffer): void {
    this._buf = this._buf.length ? Buffer.concat([this._buf, data]) : data;
    if (this._state === 'connecting') {
      const end = this._buf.indexOf('\r\n\r\n');
      if (end < 0) {
        if (this._buf.length > 64 * 1024) {
          this._fail(new Error('WebSocket handshake response too large'));
        }
        return;
      }
      const head = this._buf.subarray(0, end).toString('latin1');
      this._buf = this._buf.subarray(end + 4);
      if (!this._acceptHandshake(head)) return;
      this._state = 'open';
      if (this._connectTimer) {
        clearTimeout(this._connectTimer);
        this._connectTimer = null;
      }
      this.emit('open');
    }
    this._parseFrames();
  }

  private _acceptHandshake(head: string): boolean {
    const lines = head.split('\r\n');
    const status = lines[0] ?? '';
    const m = /^HTTP\/1\.1 (\d{3})/.exec(status);
    if (!m || m[1] !== '101') {
      this._fail(new Error(`WebSocket handshake rejected: ${status}`));
      return false;
    }
    const headers = new Map<string, string>();
    for (const line of lines.slice(1)) {
      const i = line.indexOf(':');
      if (i > 0) {
        headers.set(
          line.slice(0, i).trim().toLowerCase(),
          line.slice(i + 1).trim(),
        );
      }
    }
    const expected = crypto
      .createHash('sha1')
      .update(this._key + WS_GUID)
      .digest('base64');
    if (headers.get('sec-websocket-accept') !== expected) {
      this._fail(new Error('WebSocket handshake: bad Sec-WebSocket-Accept'));
      return false;
    }
    if ((headers.get('upgrade') ?? '').toLowerCase() !== 'websocket') {
      this._fail(new Error('WebSocket handshake: missing Upgrade header'));
      return false;
    }
    if (headers.has('sec-websocket-extensions')) {
      this._fail(new Error('WebSocket handshake: unexpected extension'));
      return false;
    }
    return true;
  }

  private _parseFrames(): void {
    while (this._socket && this._state !== 'closed') {
      const buf = this._buf;
      if (buf.length < 2) return;
      const b0 = buf[0];
      const b1 = buf[1];
      const fin = (b0 & 0x80) !== 0;
      const rsv = b0 & 0x70;
      const opcode = b0 & 0x0f;
      const masked = (b1 & 0x80) !== 0;
      let len = b1 & 0x7f;
      let offset = 2;
      // Everything the first two bytes already decide is checked here,
      // before waiting for extended-length bytes: an invalid header must
      // not leave the connection open while the parser waits for more.
      if (rsv !== 0) {
        this._protocolError(1002, 'unexpected RSV bits');
        return;
      }
      if (masked) {
        this._protocolError(1002, 'masked frame from the server');
        return;
      }
      if (opcode >= 0x8 && (!fin || len > 125)) {
        this._protocolError(1002, 'fragmented or oversized control frame');
        return;
      }
      if (len === 126) {
        if (buf.length < 4) return;
        len = buf.readUInt16BE(2);
        offset = 4;
      } else if (len === 127) {
        if (buf.length < 10) return;
        const big = buf.readBigUInt64BE(2);
        if (big > BigInt(this._maxMessageBytes)) {
          this._protocolError(1009, 'message too big');
          return;
        }
        len = Number(big);
        offset = 10;
      }
      if (len > this._maxMessageBytes) {
        this._protocolError(1009, 'message too big');
        return;
      }
      // Server frames are never masked (rejected above), so the payload
      // starts right after the length bytes.
      if (buf.length < offset + len) return;
      const payload = buf.subarray(offset, offset + len);
      this._buf = buf.subarray(offset + len);
      this._handleFrame(fin, opcode, payload);
    }
  }

  private _handleFrame(fin: boolean, opcode: number, payload: Buffer): void {
    if (this._state === 'closing' && opcode !== OP_CLOSE) {
      // After a Close went out (ours, or the answer to a protocol
      // failure) only the peer's Close matters; data frames that were
      // already in flight are dropped, not delivered.
      return;
    }
    switch (opcode) {
      case OP_TEXT:
      case OP_BINARY:
        if (this._fragments.length) {
          this._protocolError(1002, 'new message inside a fragmented one');
          return;
        }
        if (fin) {
          this._emitMessage(opcode, payload);
          return;
        }
        this._fragmentOpcode = opcode;
        this._pushFragment(payload);
        return;
      case OP_CONTINUATION:
        if (!this._fragments.length) {
          this._protocolError(1002, 'continuation without a start frame');
          return;
        }
        if (!this._pushFragment(payload)) return;
        if (fin) {
          const whole = Buffer.concat(this._fragments);
          this._fragments = [];
          this._fragmentBytes = 0;
          this._emitMessage(this._fragmentOpcode, whole);
        }
        return;
      case OP_PING:
        if (this._socket && this._state === 'open') {
          this._socket.write(this._frame(OP_PONG, payload));
        }
        return;
      case OP_PONG:
        return;
      case OP_CLOSE: {
        const code = payload.length >= 2 ? payload.readUInt16BE(0) : 1005;
        const reason =
          payload.length > 2 ? payload.subarray(2).toString('utf8') : '';
        this._pendingClose = {code, reason};
        this._finishClose(code === 1005 ? 1000 : code, '');
        return;
      }
      default:
        this._protocolError(1002, `unknown opcode ${opcode}`);
    }
  }

  private _pushFragment(payload: Buffer): boolean {
    this._fragmentBytes += payload.length;
    if (this._fragmentBytes > this._maxMessageBytes) {
      this._protocolError(1009, 'message too big');
      return false;
    }
    this._fragments.push(payload);
    return true;
  }

  private _protocolError(code: number, reason: string): void {
    this._emitError(new Error(`WebSocket protocol error: ${reason}`));
    this._pendingClose = {code, reason};
    this._finishClose(code, reason);
  }

  /**
   * Deliver a complete message.  Text must be valid UTF-8 (RFC 6455
   * §8.1: anything else closes with 1007); binary is decoded leniently
   * since the daemon only ever sends JSON text.
   */
  private _emitMessage(opcode: number, payload: Buffer): void {
    let text: string;
    if (opcode === OP_TEXT) {
      try {
        text = utf8.decode(payload);
      } catch {
        this._protocolError(1007, 'invalid UTF-8 in text message');
        return;
      }
    } else {
      text = payload.toString('utf8');
    }
    this.emit('message', text);
  }

  /**
   * Answer (or initiate) the closing handshake and let the socket drain.
   *
   * The Close frame is queued behind any message still being written
   * (a large frame under backpressure), so the socket is half-closed
   * with `end()` rather than destroyed: destroy would discard the
   * queued bytes, including the Close itself.  `CLOSE_GRACE_MS` bounds
   * the wait for the peer to finish.
   */
  private _finishClose(code: number, reason: string): void {
    const sock = this._socket;
    if (!sock) {
      this._onSocketClosed();
      return;
    }
    if (this._state === 'open' || this._state === 'closing') {
      this._state = 'closing';
      this._sendClose(code, reason);
      try {
        sock.end();
      } catch {}
      if (!this._closeTimer) {
        this._closeTimer = setTimeout(() => this.destroy(), CLOSE_GRACE_MS);
      }
      return;
    }
    this.destroy();
  }

  private _sendClose(code: number, reason: string): void {
    if (this._closeSent || !this._socket) return;
    this._closeSent = true;
    const reasonBuf = Buffer.from(reason, 'utf8').subarray(0, 123);
    const payload = Buffer.alloc(2 + reasonBuf.length);
    payload.writeUInt16BE(code, 0);
    reasonBuf.copy(payload, 2);
    try {
      this._socket.write(this._frame(OP_CLOSE, payload));
    } catch {}
  }

  /** Build one masked client frame (FIN set). */
  private _frame(opcode: number, payload: Buffer): Buffer {
    const len = payload.length;
    const headerLen = len < 126 ? 2 : len < 65_536 ? 4 : 10;
    const out = Buffer.alloc(headerLen + 4 + len);
    out[0] = 0x80 | opcode;
    if (len < 126) {
      out[1] = 0x80 | len;
    } else if (len < 65_536) {
      out[1] = 0x80 | 126;
      out.writeUInt16BE(len, 2);
    } else {
      out[1] = 0x80 | 127;
      out.writeBigUInt64BE(BigInt(len), 2);
    }
    const mask = crypto.randomBytes(4);
    mask.copy(out, headerLen);
    for (let i = 0; i < len; i++) {
      out[headerLen + 4 + i] = payload[i] ^ mask[i & 3];
    }
    return out;
  }

  private _onSocketClosed(): void {
    if (this._state === 'closed') return;
    this._state = 'closed';
    if (this._connectTimer) {
      clearTimeout(this._connectTimer);
      this._connectTimer = null;
    }
    if (this._closeTimer) {
      clearTimeout(this._closeTimer);
      this._closeTimer = null;
    }
    const sock = this._socket;
    this._socket = null;
    if (sock) {
      sock.removeAllListeners('data');
      try {
        sock.destroy();
      } catch {}
    }
    this._buf = Buffer.alloc(0);
    this._fragments = [];
    this._fragmentBytes = 0;
    const info = this._pendingClose ?? {code: 1006, reason: ''};
    this._pendingClose = null;
    this.emit('close', info);
  }
}
