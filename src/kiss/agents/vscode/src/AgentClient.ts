// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as net from 'net';
import * as path from 'path';
import {EventEmitter} from 'events';
import {StringDecoder} from 'string_decoder';
import {AgentCommand, ToWebviewMessage} from './types';
import {kissHomeDir} from './userAssets';

const MAX_LINE_BUFFER_BYTES = 32 * 1024 * 1024;

const RECONNECT_DELAY_MS = 500;

export class AgentClient extends EventEmitter {
  private _socket: net.Socket | null = null;
  private _buffer: string = '';
  private _pendingSends: string[] = [];
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _disposed: boolean = false;
  private _connecting: boolean = false;
  private _sockPath: string;

  constructor(sockPath?: string) {
    super();
    this._sockPath =
      sockPath ??
      process.env.KISS_SORCAR_SOCK ??
      path.join(kissHomeDir(), 'sorcar.sock');
  }

  connect(): void {
    if (this._socket || this._disposed || this._connecting) return;
    this._connecting = true;
    // Line-buffer state is connection-scoped: a partial line left over from
    // a dead connection must not contaminate the next connection.
    this._buffer = '';
    const sock = net.createConnection({path: this._sockPath});
    this._socket = sock;
    sock.setNoDelay(true);
    // A persistent decoder per connection keeps UTF-8 code points intact
    // even when they are split across stream chunks.
    const decoder = new StringDecoder('utf8');

    sock.on('connect', () => {
      if (this._disposed || this._socket !== sock) {
        sock.destroy();
        return;
      }
      this._connecting = false;
      this.emit('connect');
      const pending = this._pendingSends;
      this._pendingSends = [];
      for (const line of pending) sock.write(line);
    });

    sock.on('data', (data: Buffer) => {
      if (this._socket !== sock) return;
      this._handleData(decoder.write(data));
    });

    sock.on('error', err => {
      const code = (err as NodeJS.ErrnoException).code;
      if (code !== 'ENOENT' && code !== 'ECONNREFUSED') {
        console.error('[AgentClient] socket error:', err.message);
      }
    });

    sock.on('close', () => {
      if (this._socket !== sock) return;
      this._connecting = false;
      this._socket = null;
      this._buffer = '';
      this.emit('disconnect');
      if (this._disposed) return;
      this._scheduleReconnect();
    });
  }

  sendCommand(cmd: AgentCommand): void {
    const line = JSON.stringify(cmd) + '\n';
    const sock = this._socket;
    if (sock && !sock.connecting && sock.writable) {
      sock.write(line);
      return;
    }
    this._pendingSends.push(line);
    this.connect();
  }

  dispose(): void {
    this._disposed = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this._socket) {
      try {
        this._socket.end();
      } catch {}
      this._socket = null;
    }
    this.removeAllListeners();
  }

  private _scheduleReconnect(): void {
    if (this._reconnectTimer || this._disposed) return;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      if (!this._disposed) this.connect();
    }, RECONNECT_DELAY_MS);
  }

  private _handleData(chunk: string): void {
    this._buffer += chunk;
    if (this._buffer.length > MAX_LINE_BUFFER_BYTES) {
      console.error(
        '[AgentClient] line buffer exceeded limit ' +
          `(${this._buffer.length} > ${MAX_LINE_BUFFER_BYTES}); ` +
          'dropping connection.',
      );
      this._buffer = '';
      if (this._socket) this._socket.destroy();
      return;
    }
    const lines = this._buffer.split('\n');
    this._buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line) as ToWebviewMessage;
        this.emit('message', msg);
      } catch {
        console.warn(
          '[AgentClient] non-JSON line from daemon:',
          line.slice(0, 200),
        );
      }
    }
  }
}
