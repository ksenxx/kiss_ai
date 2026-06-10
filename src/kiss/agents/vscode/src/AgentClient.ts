// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Persistent Unix-domain socket client to the kiss-web daemon.
 *
 * Drop-in replacement for ``AgentProcess.sendCommand`` /
 * ``'message'`` event API, but with a single connection multiplexing
 * every tab.  The daemon (`RemoteAccessServer`) injects ``tabId`` into
 * every outgoing event, so the message listener can route by tab id
 * without any per-process bookkeeping on the extension side.
 *
 * Connection model
 * ----------------
 * - One persistent UDS connection per VS Code window (the extension's
 *   ``SorcarSidebarView`` owns it).
 * - Auto-connect on first ``sendCommand``; auto-reconnect on socket
 *   close with a 500 ms backoff until ``dispose()``.
 * - Commands sent while disconnected are queued and flushed on
 *   ``connect`` so the caller never sees write failures during
 *   reconnect.
 *
 * Wire format
 * -----------
 * Newline-delimited JSON, identical to the WSS protocol the daemon
 * already speaks to browsers.  Filesystem permissions on the socket
 * (mode 0o600, see ``RemoteAccessServer._setup_server``) gate access
 * — there is no password handshake on the UDS path.
 */

import * as net from 'net';
import * as os from 'os';
import * as path from 'path';
import {EventEmitter} from 'events';
import {AgentCommand, ToWebviewMessage} from './types';

/**
 * Cap on the per-line incoming buffer (matches ``AgentProcess``).  If
 * the daemon ever emits one huge JSON line without a newline we drop
 * the connection rather than OOM the extension host.
 */
const MAX_LINE_BUFFER_BYTES = 32 * 1024 * 1024;

/** Backoff between reconnect attempts. */
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
    // Default UDS path, matches ``_UDS_PATH`` in ``web_server.py``.
    this._sockPath =
      sockPath ?? path.join(os.homedir(), '.kiss', 'sorcar.sock');
  }

  /** Initiate a connection.  Idempotent — repeated calls are no-ops. */
  connect(): void {
    if (this._socket || this._disposed || this._connecting) return;
    this._connecting = true;
    const sock = net.createConnection({path: this._sockPath});
    this._socket = sock;
    sock.setNoDelay(true);

    sock.on('connect', () => {
      this._connecting = false;
      // Emit BEFORE flushing the queue so 'connect' listeners can send
      // connection-preamble commands (e.g. SorcarSidebarView re-syncs
      // the daemon's per-connection work_dir with setWorkDir) ahead of
      // any commands queued while disconnected.  The daemon tracks
      // work_dir per connection, so the preamble must be re-sent on
      // every reconnect, not just once per AgentClient lifetime.
      this.emit('connect');
      const pending = this._pendingSends;
      this._pendingSends = [];
      for (const line of pending) sock.write(line);
    });

    sock.on('data', (data: Buffer) => this._handleData(data.toString()));

    sock.on('error', err => {
      // ENOENT (no daemon yet) is expected during startup; suppress the
      // noise but log everything else.
      const code = (err as NodeJS.ErrnoException).code;
      if (code !== 'ENOENT' && code !== 'ECONNREFUSED') {
        console.error('[AgentClient] socket error:', err.message);
      }
    });

    sock.on('close', () => {
      this._connecting = false;
      this._socket = null;
      if (this._disposed) return;
      this._scheduleReconnect();
    });
  }

  /**
   * Send a command (any ``AgentCommand``).  If the socket is not yet
   * open the line is queued and flushed on the next ``connect``.
   */
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

  /**
   * Graceful disconnect: half-close the socket so the daemon's
   * ``_uds_handler`` exits its ``readline`` loop cleanly and arms
   * deferred disposal for all tabs on this connection.
   */
  dispose(): void {
    this._disposed = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this._socket) {
      try {
        this._socket.end();
      } catch {
        /* ignored */
      }
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
