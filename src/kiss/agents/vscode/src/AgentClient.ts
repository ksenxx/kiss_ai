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

const RECONNECT_BASE_DELAY_MS = 500;
const RECONNECT_MAX_DELAY_MS = 15_000;
// A command queued while the daemon is unreachable is delivered to
// whichever daemon answers next.  Past this age that is a DIFFERENT
// process from the one the user was talking to -- replaying a `run` into
// it starts an agent nobody asked for -- so the frame is dropped.
const PENDING_SEND_TTL_MS = 10_000;
const MAX_PENDING_SENDS = 256;
// A connection has to last this long to count as a good one.  A daemon
// that accepts and immediately drops -- a crash loop -- must not reset
// the backoff on every accept, or it is hammered as hard as one that
// never listens at all.
const STABLE_CONNECTION_MS = 5_000;

/** Tunables, so a test can exercise the timing without waiting on it. */
export interface AgentClientOptions {
  reconnectBaseMs?: number;
  reconnectMaxMs?: number;
  pendingTtlMs?: number;
  maxPendingSends?: number;
}

/** Why a queued command was never delivered. */
export type DroppedCommandReason = 'expired' | 'overflow';

interface PendingSend {
  line: string;
  cmd: AgentCommand;
  at: number;
}

export class AgentClient extends EventEmitter {
  private _socket: net.Socket | null = null;
  private _buffer: string = '';
  private _pendingSends: PendingSend[] = [];
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _reconnectAttempts: number = 0;
  private _connectedAt: number = 0;
  private _disposed: boolean = false;
  private _connecting: boolean = false;
  private _sockPath: string;
  private _reconnectBaseMs: number;
  private _reconnectMaxMs: number;
  private _pendingTtlMs: number;
  private _maxPendingSends: number;

  constructor(sockPath?: string, options: AgentClientOptions = {}) {
    super();
    this._sockPath =
      sockPath ??
      process.env.KISS_SORCAR_SOCK ??
      path.join(kissHomeDir(), 'sorcar.sock');
    this._reconnectBaseMs = options.reconnectBaseMs ?? RECONNECT_BASE_DELAY_MS;
    this._reconnectMaxMs = options.reconnectMaxMs ?? RECONNECT_MAX_DELAY_MS;
    this._pendingTtlMs = options.pendingTtlMs ?? PENDING_SEND_TTL_MS;
    this._maxPendingSends = options.maxPendingSends ?? MAX_PENDING_SENDS;
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
      this._connectedAt = Date.now();
      this.emit('connect');
      const cutoff = Date.now() - this._pendingTtlMs;
      const pending = this._pendingSends;
      this._pendingSends = [];
      for (const item of pending) {
        if (item.at < cutoff) {
          this._announceDropped(item, 'expired');
          continue;
        }
        sock.write(item.line);
      }
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
      if (
        this._connectedAt &&
        Date.now() - this._connectedAt >= STABLE_CONNECTION_MS
      ) {
        this._reconnectAttempts = 0;
      }
      this._connectedAt = 0;
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
    this._pendingSends.push({line, cmd, at: Date.now()});
    const surplus = this._pendingSends.length - this._maxPendingSends;
    if (surplus > 0) {
      for (const item of this._pendingSends.splice(0, surplus)) {
        this._announceDropped(item, 'overflow');
      }
    }
    this.connect();
  }

  /**
   * Report a queued command the client has decided never to deliver.
   *
   * Both reasons are deliberate -- a `run` replayed into the daemon
   * that REPLACED the one it was meant for starts an agent nobody asked
   * for, and an unbounded queue is its own problem -- but neither is
   * free: the webview shows a task as running the moment it is sent,
   * so a command that quietly evaporates leaves a tab running for ever
   * with nothing behind it.  The owner of that optimistic state is
   * told, and undoes it.
   *
   * @param item The queued frame being discarded.
   * @param reason Why it is being discarded.
   */
  private _announceDropped(
    item: PendingSend,
    reason: DroppedCommandReason,
  ): void {
    this.emit('commandDropped', item.cmd, reason);
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
    this._pendingSends = [];
    this.removeAllListeners();
  }

  /**
   * Retry the connection, backing off so a daemon restart is not met by
   * a connect storm.
   *
   * Every open window runs one of these against the same socket, so a
   * fixed retry meant N windows hammered the daemon 2N times a second
   * for the whole of every outage -- exactly while it was trying to
   * bind.  The delay doubles up to a ceiling and carries jitter so the
   * windows do not re-converge on the same instant.
   */
  private _scheduleReconnect(): void {
    if (this._reconnectTimer || this._disposed) return;
    const capped = Math.min(
      this._reconnectBaseMs * 2 ** this._reconnectAttempts,
      this._reconnectMaxMs,
    );
    this._reconnectAttempts += 1;
    const delay = capped / 2 + Math.random() * (capped / 2);
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      if (!this._disposed) this.connect();
    }, delay);
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
