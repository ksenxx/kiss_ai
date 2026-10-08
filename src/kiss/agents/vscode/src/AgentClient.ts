// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as fs from 'fs';
import {EventEmitter} from 'events';
import {AgentCommand, ToWebviewMessage} from './types';
import {readLocalEndpoint, sorcarEndpointPath} from './userAssets';
import {WsClient} from './wsClient';

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
const CONNECT_TIMEOUT_MS = 10_000;
/**
 * Deadline for the daemon's `auth_ok` after the WebSocket opened.  The
 * socket's own connect timer stops at the upgrade; a peer that upgrades
 * and then never answers the auth frame would otherwise hold `_ws` for
 * good and block every further connect attempt.
 */
const AUTH_TIMEOUT_MS = 10_000;
// How often to re-read a missing endpoint file.  No socket is opened
// while the file is absent -- the daemon has not published one -- so
// this is a cheap stat, not a connect storm, and it stays fixed rather
// than backing off: a daemon that has just been started publishes its
// endpoint within milliseconds and every window should notice at once.
const ENDPOINT_POLL_MS = 100;

/** Tunables, so a test can exercise the timing without waiting on it. */
interface AgentClientOptions {
  reconnectBaseMs?: number;
  reconnectMaxMs?: number;
  pendingTtlMs?: number;
  maxPendingSends?: number;
  endpointPollMs?: number;
  /** Deadline for the daemon's `auth_ok` after the socket opened (tests). */
  authTimeoutMs?: number;
}

/** Why a queued command was never delivered. */
export type DroppedCommandReason = 'expired' | 'overflow';

interface PendingSend {
  text: string;
  cmd: AgentCommand;
  at: number;
}

/**
 * The extension host's connection to the kiss-web daemon.
 *
 * Reads the daemon's endpoint file on every connect attempt (the URL
 * and local token change with each daemon start), opens the WSS
 * connection with the daemon's CA pinned, completes the local `auth`
 * handshake and only then reports `connect` and flushes queued
 * commands.  Events: `connect`, `disconnect`, `message` (parsed daemon
 * event), `commandDropped` (cmd, reason).
 */
export class AgentClient extends EventEmitter {
  private _ws: WsClient | null = null;
  private _authenticated = false;
  /**
   * Whether the current outage has been reported with `disconnect`.
   * The endpoint file is polled every 100 ms while absent; each poll is
   * a failed attempt, but only the first one is news to the listeners.
   */
  private _downAnnounced = false;
  private _pendingSends: PendingSend[] = [];
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _authTimer: ReturnType<typeof setTimeout> | null = null;
  private _reconnectAttempts: number = 0;
  private _connectedAt: number = 0;
  private _disposed: boolean = false;
  private _endpointPath: string;
  private _reconnectBaseMs: number;
  private _reconnectMaxMs: number;
  private _pendingTtlMs: number;
  private _maxPendingSends: number;
  private _endpointPollMs: number;
  private _authTimeoutMs: number;
  private _preamble: AgentCommand | null = null;

  constructor(endpointPath?: string, options: AgentClientOptions = {}) {
    super();
    this._endpointPath = endpointPath ?? sorcarEndpointPath();
    this._reconnectBaseMs = options.reconnectBaseMs ?? RECONNECT_BASE_DELAY_MS;
    this._reconnectMaxMs = options.reconnectMaxMs ?? RECONNECT_MAX_DELAY_MS;
    this._pendingTtlMs = options.pendingTtlMs ?? PENDING_SEND_TTL_MS;
    this._maxPendingSends = options.maxPendingSends ?? MAX_PENDING_SENDS;
    this._endpointPollMs = options.endpointPollMs ?? ENDPOINT_POLL_MS;
    this._authTimeoutMs = options.authTimeoutMs ?? AUTH_TIMEOUT_MS;
  }

  /** The endpoint file this client reads on every connect attempt. */
  get endpointPath(): string {
    return this._endpointPath;
  }

  connect(): void {
    // A pending reconnect timer already guarantees the next attempt;
    // starting one now (from `sendCommand`) would bypass the back-off.
    if (this._ws || this._disposed || this._reconnectTimer) return;
    const endpoint = readLocalEndpoint(this._endpointPath);
    if (!endpoint) {
      // No daemon has published an endpoint (yet).
      this._failAttempt(this._endpointPollMs);
      return;
    }
    let ca: string | undefined;
    if (endpoint.ca) {
      try {
        ca = fs.readFileSync(endpoint.ca, 'utf8');
      } catch (err) {
        console.error(
          '[AgentClient] cannot read daemon CA ' +
            `${endpoint.ca}: ${(err as Error).message}`,
        );
        this._failAttempt(this._endpointPollMs);
        return;
      }
    }
    const ws = new WsClient({
      url: endpoint.url,
      ca,
      connectTimeoutMs: CONNECT_TIMEOUT_MS,
    });
    this._ws = ws;
    this._authenticated = false;

    ws.on('open', () => {
      if (this._disposed || this._ws !== ws) {
        ws.destroy();
        return;
      }
      ws.send(JSON.stringify({type: 'auth', token: endpoint.token}));
      this._authTimer = setTimeout(() => {
        this._authTimer = null;
        if (this._ws !== ws || this._authenticated) return;
        console.error('[AgentClient] daemon did not answer the auth frame');
        ws.destroy();
      }, this._authTimeoutMs);
    });

    ws.on('message', (text: string) => {
      if (this._ws !== ws) return;
      let msg: ToWebviewMessage;
      try {
        msg = JSON.parse(text) as ToWebviewMessage;
      } catch {
        console.warn(
          '[AgentClient] non-JSON frame from daemon:',
          text.slice(0, 200),
        );
        return;
      }
      if (!this._authenticated) {
        this._handleAuthReply(ws, msg as unknown as Record<string, unknown>);
        return;
      }
      this.emit('message', msg);
    });

    ws.on('error', (err: Error) => {
      const code = (err as NodeJS.ErrnoException).code;
      if (code !== 'ECONNREFUSED') {
        console.error('[AgentClient] connection error:', err.message);
      }
    });

    ws.on('close', () => {
      if (this._ws !== ws) return;
      this._clearAuthTimer();
      this._ws = null;
      this._authenticated = false;
      if (
        this._connectedAt &&
        Date.now() - this._connectedAt >= STABLE_CONNECTION_MS
      ) {
        this._reconnectAttempts = 0;
      }
      this._connectedAt = 0;
      this._announceDown();
      if (this._disposed) return;
      this._scheduleReconnect();
    });

    ws.connect();
  }

  /**
   * Report a connect attempt that failed before a socket was opened.
   *
   * Delivered on the next tick, like a socket's connection error, so
   * `sendCommand()` never re-enters the caller's `disconnect` handler
   * from inside the call.  The UI treats it like any other failed
   * attempt (the daemon is down); the next read of the endpoint file
   * is due after `retryMs`.
   */
  private _failAttempt(retryMs: number): void {
    setImmediate(() => {
      if (this._disposed || this._ws) return;
      this._announceDown();
      if (this._reconnectTimer) return;
      this._reconnectTimer = setTimeout(() => {
        this._reconnectTimer = null;
        if (!this._disposed) this.connect();
      }, retryMs);
    });
  }

  /**
   * Consume the daemon's answer to the `auth` frame.
   *
   * `auth_ok` with `local: true` completes the connection: the
   * preamble (see `setPreamble`) goes out first — the daemon resolves
   * every later command's missing workDir against the pin it carries,
   * and the queue may hold commands that need it (a submit or a file
   * link clicked during the outage) — then queued commands are flushed
   * BEFORE `connect` is announced, because a `connect` handler
   * immediately writes fresh commands (e.g. getModels) whose replies
   * could otherwise overwrite state a queued command (e.g.
   * selectModel) was about to change.  Anything else means the token
   * in the endpoint file is stale (a daemon restarted between the read
   * and the handshake) or the peer is not our daemon: drop the
   * connection and retry, re-reading the file.
   */
  private _handleAuthReply(ws: WsClient, msg: Record<string, unknown>): void {
    if (msg.type === 'auth_ok' && msg.local === true) {
      this._clearAuthTimer();
      this._authenticated = true;
      this._downAnnounced = false;
      this._connectedAt = Date.now();
      if (this._preamble) ws.send(JSON.stringify(this._preamble));
      const cutoff = Date.now() - this._pendingTtlMs;
      const pending = this._pendingSends;
      this._pendingSends = [];
      for (const item of pending) {
        if (item.at < cutoff) {
          this._announceDropped(item, 'expired');
          continue;
        }
        ws.send(item.text);
      }
      this.emit('connect');
      return;
    }
    console.error(
      '[AgentClient] daemon did not accept the local token:',
      JSON.stringify(msg).slice(0, 200),
    );
    ws.destroy();
  }

  /**
   * Set the command written first on every (re)connect, ahead of any
   * queued frame; `null` clears it.
   *
   * The sidebar leads with `setWorkDir {.., ifUnset: true}`: on a fresh
   * install it seeds the daemon's global working directory with the
   * window's folder before the commands queued while the daemon was
   * down (a prompt submitted, a file link clicked) resolve against it;
   * once a directory is persisted the daemon ignores the seed.  The
   * preamble is written on this connection and on every reconnect; a
   * change takes effect from the next connect (send the new command
   * yourself for the current one).
   *
   * @param cmd The command to lead every connection with.
   */
  setPreamble(cmd: AgentCommand | null): void {
    this._preamble = cmd;
  }

  sendCommand(cmd: AgentCommand): void {
    const text = JSON.stringify(cmd);
    const ws = this._ws;
    if (ws && this._authenticated && ws.send(text)) return;
    this._pendingSends.push({text, cmd, at: Date.now()});
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

  /** Emit `disconnect` once per outage, on the connected->down transition. */
  private _announceDown(): void {
    if (this._downAnnounced) return;
    this._downAnnounced = true;
    this.emit('disconnect');
  }

  private _clearAuthTimer(): void {
    if (this._authTimer) {
      clearTimeout(this._authTimer);
      this._authTimer = null;
    }
  }

  dispose(): void {
    this._disposed = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this._clearAuthTimer();
    if (this._ws) {
      // Disposal is cancellation, not a graceful goodbye: a closing
      // handshake would keep the socket -- and whatever it still has
      // buffered -- alive until the daemon reads it, which a wedged
      // daemon never does.
      const ws = this._ws;
      this._ws = null;
      try {
        ws.destroy();
      } catch {}
    }
    this._authenticated = false;
    this._pendingSends = [];
    this.removeAllListeners();
  }

  /**
   * Retry the connection, backing off so a daemon restart is not met by
   * a connect storm.
   *
   * Every open window runs one of these against the same daemon, so a
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
}
