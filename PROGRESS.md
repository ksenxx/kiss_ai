# Task

Make sorcar CLI interactive mode show the user input bar at the bottom of the screen at ALL times
(like Claude Code's `/tui fullscreen`), including idle. While a task runs, user-typed messages
should be added to context (already implemented via `pending_user_messages` + steering).

Use `claude-opus-4-7` for all coding; use `gpt-5.5` (not codex) for review.

# Implementation plan (full reference)

See the architecture decision in the previous PROGRESS.md (now overwritten); summary:

**Strategy:** Make a single owner of the anchored box across the REPL via `AnchoredRepl` (context
manager) in `cli_steering.py`. Refactor `SteeringSession` to accept an externally-owned box.
`run_repl` in `cli_repl.py` dispatches to anchored mode when `supports_steering()` and rows>=MIN.

# Sessions

## Session 1 (completed)

- Explored code, set model to claude-opus-4-7. Web research done.
- Identified gap: idle REPL inline, not bottom-anchored.

## Session 2 (this)

- Updated `_InputBox` in `cli_steering.py`:
  - Added imports: `Callable`, `IDLE_TITLE`.
  - Added attributes: `history: list[str]`, `_hist_idx: int | None = None`, `_hist_saved: str`,
    `completer_fn: Callable[[str], list[str]] | None`, `_tab_candidates: list[str]`,
    `_tab_idx: int`, `_tab_origin: str`.
  - Added helpers: `_history_back()`, `_history_forward()`, `_reset_completion_state()`.
  - Changed `feed()` signature: added `on_eof=None` parameter.
  - Updated CSI swallow branch to detect Up arrow (`\x1b[A`) → `_history_back()`, Down arrow
    (`\x1b[B`) → `_history_forward()`. Other CSI sequences still swallowed.
  - Added Tab handling: cycles through `completer_fn(buf)` results.
  - Added Ctrl+D (`\x04`) handling: when buffer empty + `on_eof` set, calls `on_eof()` and returns.
  - Reset `_hist_idx` and `_reset_completion_state()` on every edit
    (Enter/Backspace/Ctrl+U/printable char) so Up/Tab restart cleanly.

## Remaining work for Session 3

### 3a. Refactor `SteeringSession.__init__` in `cli_steering.py`

Replace its `__init__` and `run` to support optional shared box. The current code:

```python
def __init__(
    self,
    agent: SorcarAgent,
    state: _RunningAgentState,
    chat_id: str,
) -> None:
    del chat_id
    self.agent = agent
    self.state = state
    self.lock = threading.RLock()
    self._real_stdout = sys.stdout
    self._real_stderr = sys.stderr
    self.box = _InputBox(self.lock, self._real_stdout)
    self._done = threading.Event()
    self._aborted = threading.Event()
    self._result = ""
    self._error: BaseException | None = None
    self._queued_count = 0
    self._answer_q: queue.Queue[str] = queue.Queue(maxsize=1)
    self._question_pending = threading.Event()
```

Replace with:

```python
def __init__(
    self,
    agent: SorcarAgent,
    state: _RunningAgentState,
    chat_id: str,
    *,
    box: _InputBox | None = None,
    lock: threading.RLock | None = None,
    real_stdout: Any = None,
    real_stderr: Any = None,
) -> None:
    del chat_id
    self.agent = agent
    self.state = state
    self.lock = lock if lock is not None else threading.RLock()
    self._real_stdout = real_stdout if real_stdout is not None else sys.stdout
    self._real_stderr = real_stderr if real_stderr is not None else sys.stderr
    self._owns_box = box is None
    self.box = box if box is not None else _InputBox(self.lock, self._real_stdout)
    self._done = threading.Event()
    self._aborted = threading.Event()
    self._result = ""
    self._error: BaseException | None = None
    self._queued_count = 0
    self._answer_q: queue.Queue[str] = queue.Queue(maxsize=1)
    self._question_pending = threading.Event()
```

### 3b. Refactor `SteeringSession.run` in `cli_steering.py`

Replace:

```python
def run(self, run_kwargs: dict[str, Any]) -> str:
    real_stdout = self._real_stdout
    real_stderr = self._real_stderr
    proxy = _StdoutProxy(real_stdout, self.lock, self.box)
    sys.stdout = cast(Any, proxy)
    sys.stderr = cast(
        Any, _StdoutProxy(real_stderr, self.lock, self.box)
    )
    self.box.start()
    worker = threading.Thread(
        target=self._worker, args=(run_kwargs,), daemon=True
    )
    worker.start()
    try:
        self._loop()
    except KeyboardInterrupt:
        self._on_abort()
    finally:
        self.box.stop()
        sys.stdout = real_stdout
        sys.stderr = real_stderr
    if self._aborted.is_set():
        self._interrupt_worker(worker)
        raise KeyboardInterrupt
    if self._error is not None:
        raise self._error
    return self._result
```

with:

```python
def run(self, run_kwargs: dict[str, Any]) -> str:
    prev_stdout = sys.stdout
    prev_stderr = sys.stderr
    prev_title = self.box.title
    prev_status = self.box.status
    if self._owns_box:
        sys.stdout = cast(
            Any, _StdoutProxy(self._real_stdout, self.lock, self.box),
        )
        sys.stderr = cast(
            Any, _StdoutProxy(self._real_stderr, self.lock, self.box),
        )
        self.box.start()
    # Whether the box is owned or shared, the in-task title/status come
    # from the steering preset so the user sees "queue follow-ups".
    with self.lock:
        self.box.title = STEER_TITLE
        self.box.status = ""
        if self.box._active:
            self.box.redraw()
    worker = threading.Thread(
        target=self._worker, args=(run_kwargs,), daemon=True,
    )
    worker.start()
    try:
        self._loop()
    except KeyboardInterrupt:
        self._on_abort()
    finally:
        if self._owns_box:
            self.box.stop()
            sys.stdout = prev_stdout
            sys.stderr = prev_stderr
        else:
            # Shared box stays drawn for the next idle read; restore
            # title/status so the next read shows the idle preset.
            with self.lock:
                self.box.title = prev_title
                self.box.status = prev_status
                if self.box._active:
                    self.box.redraw()
    if self._aborted.is_set():
        self._interrupt_worker(worker)
        raise KeyboardInterrupt
    if self._error is not None:
        raise self._error
    return self._result
```

### 3c. Add `AnchoredRepl` class at the end of `cli_steering.py` (before `run_with_steering`)

```python
class AnchoredRepl:
    """Owns the bottom-anchored input box for the whole sorcar REPL.

    The box stays pinned at the bottom of the screen for both idle reads
    (the next instruction to dispatch) and task execution (queueing
    follow-up instructions into ``state.pending_user_messages``).  The
    scroll region above scrolls agent output as usual.

    Used as a context manager so the box is torn down on exit even when
    the REPL raises:

        with AnchoredRepl(completer_fn=fn, history=h) as repl:
            line = repl.read_idle_line()
            repl.run_task(agent, state, chat_id, run_kwargs)
    """

    def __init__(
        self,
        completer_fn: Callable[[str], list[str]] | None = None,
        history: list[str] | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self._real_stdout = sys.stdout
        self._real_stderr = sys.stderr
        self.box = _InputBox(self.lock, self._real_stdout)
        self.box.completer_fn = completer_fn
        self.box.history = list(history or [])
        self._prev_stdout: Any = None
        self._prev_stderr: Any = None

    def __enter__(self) -> AnchoredRepl:
        self._prev_stdout = sys.stdout
        self._prev_stderr = sys.stderr
        sys.stdout = cast(
            Any, _StdoutProxy(self._real_stdout, self.lock, self.box),
        )
        sys.stderr = cast(
            Any, _StdoutProxy(self._real_stderr, self.lock, self.box),
        )
        self.box.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.box.stop()
        sys.stdout = self._prev_stdout
        sys.stderr = self._prev_stderr

    def read_idle_line(self) -> str | None:
        """Read one line in idle mode.

        Returns the typed line (possibly empty), or ``None`` on Ctrl+D
        (empty buffer).

        Raises:
            KeyboardInterrupt: When the user presses Ctrl+C.
        """
        with self.lock:
            self.box.title = IDLE_TITLE
            self.box.status = ""
            self.box.redraw()
        result: list[str] = []
        eof_flag: list[bool] = []
        abort_flag: list[bool] = []

        def on_submit(line: str) -> None:
            result.append(line)

        def on_abort() -> None:
            abort_flag.append(True)

        def on_eof() -> None:
            eof_flag.append(True)

        fd = sys.stdin.fileno()
        last_size = _term_size()
        while not result and not eof_flag and not abort_flag:
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
            except (InterruptedError, OSError):
                continue
            except KeyboardInterrupt:
                abort_flag.append(True)
                break
            if not ready:
                size = _term_size()
                if size != last_size:
                    last_size = size
                    self.box.redraw()
                continue
            try:
                data = os.read(fd, 4096)
            except (InterruptedError, OSError):
                continue
            if not data:
                eof_flag.append(True)
                break
            self.box.feed(data, on_submit, on_abort, on_eof)
        if abort_flag:
            raise KeyboardInterrupt
        if eof_flag:
            return None
        line = result[0]
        if line.strip() and (
            not self.box.history or self.box.history[-1] != line
        ):
            self.box.history.append(line)
        with self.lock:
            sys.stdout.write(f"\x1b[36m> {line}\x1b[0m\n")
            sys.stdout.flush()
        return line

    def run_task(
        self,
        agent: SorcarAgent,
        state: _RunningAgentState,
        chat_id: str,
        run_kwargs: dict[str, Any],
    ) -> str:
        """Run an agent task while reusing the existing anchored box.

        Mirrors :func:`run_with_steering` but threads the shared box,
        lock and proxied streams into :class:`SteeringSession` so the
        box is not torn down between tasks.
        """
        session = SteeringSession(
            agent, state, chat_id,
            box=self.box, lock=self.lock,
            real_stdout=self._real_stdout, real_stderr=self._real_stderr,
        )
        kwargs = dict(run_kwargs)
        kwargs["ask_user_question_callback"] = session.ask_user_question
        return session.run(kwargs)
```

### 3d. Modify `cli_repl.py`

Add imports at the top of `cli_repl.py`:

```python
from kiss.agents.sorcar.cli_panel import _term_size  # already imported? check
from kiss.agents.sorcar.cli_steering import (
    AnchoredRepl,
    _MIN_ROWS,
    run_with_steering,
    supports_steering,
)
```

Add helpers:

```python
def _load_history_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return [
            ln for ln in path.read_text(encoding="utf-8").splitlines() if ln
        ]
    except OSError:
        return []


def _save_history_lines(path: Path, history: list[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(history[-1000:]) + ("\n" if history else ""),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("could not save history", exc_info=True)
```

Add the anchored REPL driver:

```python
def _run_anchored_repl(
    agent: SorcarAgent, run_kwargs: dict[str, Any],
) -> None:
    work_dir = run_kwargs.get("work_dir") or str(Path(".").resolve())
    model_name = (
        run_kwargs.get("model_name", "")
        or getattr(agent, "model_name", "")
    )
    active_file = run_kwargs.get("current_editor_file") or ""
    completer = CliCompleter(work_dir, active_file)
    history_path = _history_path(work_dir)
    history = _load_history_lines(history_path)

    def completer_fn(buf: str) -> list[str]:
        try:
            return completer._build_matches(buf)
        except Exception:
            logger.debug("completion failed", exc_info=True)
            return []

    with AnchoredRepl(completer_fn=completer_fn, history=history) as repl:
        _print_welcome(work_dir, model_name)
        interrupt_armed = False
        while True:
            try:
                line = repl.read_idle_line()
            except KeyboardInterrupt:
                if interrupt_armed:
                    print("\nGoodbye.")
                    break
                interrupt_armed = True
                print("\n(Press Ctrl+C again or type /exit to quit)")
                continue
            if line is None:  # EOF / Ctrl+D
                print("\nGoodbye.")
                break
            interrupt_armed = False
            text = line.strip()
            if not text:
                continue
            if text in _EXIT_WORDS:
                break
            if text.startswith("/"):
                try:
                    if _handle_slash(agent, line, run_kwargs):
                        break
                except Exception as exc:
                    logger.debug("slash command failed", exc_info=True)
                    print(f"\n✗ Command failed: {exc}\n")
                continue
            _record_mentions(line)
            _run_one_anchored(agent, line, run_kwargs, repl)
        _save_history_lines(history_path, repl.box.history)


def _run_one_anchored(
    agent: SorcarAgent,
    prompt: str,
    run_kwargs: dict[str, Any],
    repl: AnchoredRepl,
) -> None:
    from kiss.agents.sorcar.persistence import _allocate_chat_id
    from kiss.agents.sorcar.running_agent_state import _RunningAgentState
    from typing import cast

    kwargs = dict(run_kwargs)
    kwargs["prompt_template"] = prompt
    chat_id = getattr(agent, "_chat_id", "") or _allocate_chat_id()
    agent._chat_id = chat_id  # type: ignore[attr-defined]
    agent._tab_id = chat_id  # type: ignore[attr-defined]
    state = _RunningAgentState(
        chat_id,
        getattr(agent, "model_name", "") or "",
        agent=cast(Any, agent),
    )
    state.chat_id = chat_id
    state.is_task_active = True
    _RunningAgentState.register(chat_id, state)
    start = time.time()
    try:
        result = repl.run_task(agent, state, chat_id, kwargs)
    except KeyboardInterrupt:
        print("\n⏹  Task interrupted.\n")
        return
    except Exception as exc:
        logger.debug("task failed", exc_info=True)
        print(f"\n✗ Task failed: {exc}\n")
        return
    finally:
        with _RunningAgentState._registry_lock:
            if (
                _RunningAgentState.running_agent_states.get(chat_id) is state
            ):
                state.is_task_active = False
                _RunningAgentState.unregister(chat_id)
    elapsed = time.time() - start
    verbose = bool(kwargs.get("verbose", True))
    print_outcome(agent, result, elapsed, verbose)
    if not verbose:
        print()
```

Modify `run_repl()` to dispatch:

```python
def run_repl(agent, run_kwargs):
    rows, _ = _term_size()
    if supports_steering() and rows >= _MIN_ROWS:
        _run_anchored_repl(agent, run_kwargs)
        return
    # ... rest of existing implementation unchanged ...
```

### 3e. Tests in `src/kiss/tests/agents/sorcar/test_cli_steering.py`

Add classes `TestInputBoxHistory`, `TestInputBoxCompletion`, `TestInputBoxEOF`,
`TestSteeringSessionSharedBox`. Concrete code is in earlier PROGRESS plan and the design holds.

### 3f. Verify

1. `uv run check --full`
1. `uv run pytest src/kiss/tests/agents/sorcar/test_cli_steering.py src/kiss/tests/agents/sorcar/test_cli_repl.py -v`

### 3g. Cleanup

`rm tmp/information-anchored-input.md` and commit.

# Status of files

- `cli_steering.py`: `_InputBox` extension DONE. Pending: SteeringSession refactor + AnchoredRepl.
- `cli_repl.py`: Pending all edits.
- `test_cli_steering.py`: Pending new test classes.
