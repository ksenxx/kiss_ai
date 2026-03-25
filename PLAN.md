# Plan: Optimize VSCode Extension Chat Window Loading Time

## Problem

The chat window takes ~1–2 seconds to become interactive after VSCode launches.
The root cause is a **sequential chain of blocking operations** between the
TypeScript extension, the Python backend subprocess, and the webview.

## Measured Bottlenecks (Root Cause Analysis)

### Bottleneck 1: `kiss/core/__init__.py` eagerly imports all LLM SDKs (~540ms)

**This is the #1 root cause.** `kiss/core/__init__.py` re-exports all model
classes at the package level:

```python
# kiss/core/__init__.py
from kiss.core.config import DEFAULT_CONFIG, AgentConfig, Config
from kiss.core.models import AnthropicModel, GeminiModel, Model, OpenAICompatibleModel
```

This means **any module that touches `kiss.core`** — even just
`from kiss.core import config as config_module` — triggers the full LLM SDK
import chain (~540ms). The following files all do this and are all in the
server.py startup path:

- `kiss/agents/vscode/diff_merge.py` → `from kiss.core import config as config_module`
- `kiss/core/base.py` → `from kiss.core import config as config_module`
- `kiss/core/kiss_agent.py` → `from kiss.core import config as config_module`
- `kiss/core/relentless_agent.py` → `from kiss.core import config as config_module`
- `kiss/agents/sorcar/sorcar_agent.py` → `from kiss.core import config as config_module`

The import chain is:
```
from kiss.core import config as config_module
  → kiss/core/__init__.py executes
    → from kiss.core.models import AnthropicModel, GeminiModel, ...
      → kiss/core/models/__init__.py
        → from kiss.core.models.anthropic_model import AnthropicModel  (~75ms)
        → from kiss.core.models.openai_compatible_model import ...     (~137ms)
        → from kiss.core.models.gemini_model import GeminiModel        (~268ms)
```

`model_info.py` also imports these directly, but the `__init__.py` is the
sneakier trigger because it fires from any `from kiss.core import ...`.

Measured with `python3 -X importtime`:
- `google.genai` (including `google.genai.types`): **~268ms**
- `openai` (including `openai.types`): **~137ms**
- `anthropic`: **~75ms**
- **Total**: **~540ms** (includes `pydantic` ~20ms and `pydantic_core` ~12ms)

### Bottleneck 2: `kiss/agents/__init__.py` eagerly imports `kiss.agents.kiss` (~61ms docker)

`kiss/agents/__init__.py` re-exports convenience functions from `kiss.agents.kiss`:

```python
# kiss/agents/__init__.py
from kiss.agents.kiss import (
    get_run_simple_coding_agent,
    prompt_refiner_agent,
    run_bash_task_in_sandboxed_ubuntu_latest,
)
```

`kiss.agents.kiss` imports `DockerManager` at module level:
```python
# kiss/agents/kiss.py
from kiss.docker.docker_manager import DockerManager
```

`kiss/docker/__init__.py` also eagerly imports both `DockerManager` and
`DockerTools`:
```python
# kiss/docker/__init__.py
from kiss.docker.docker_manager import DockerManager
from kiss.docker.docker_tools import DockerTools
```

The `docker` package alone costs **~61ms** standalone. In the server startup
chain, this is masked by shared dependencies (urllib3, requests) already loaded
by LLM SDKs, showing only ~6ms cumulative. **But after Fix 1 makes LLM imports
lazy, docker becomes an independent ~35ms bottleneck.**

Additionally, `kiss/agents/sorcar/sorcar_agent.py` and
`kiss/core/relentless_agent.py` both import `DockerManager` and `DockerTools`
at module level, even though Docker is never used in the VS Code extension's
normal flow.

### Bottleneck 3: `kiss/agents/sorcar/__init__.py` eagerly imports sorcar config

```python
# kiss/agents/sorcar/__init__.py
import kiss.agents.sorcar.config  # noqa: F401
```

This pulls in `pydantic` and `kiss.core.config_builder` at package load time.
`config_builder.py` in turn imports `pydantic_settings` (~13ms).

### Bottleneck 4: `kiss/core/config.py` imports pydantic at module level (~44ms)

`config.py` uses `pydantic.BaseModel` for all config classes and also eagerly
creates `DEFAULT_CONFIG = Config()` at import time, which instantiates all
nested pydantic models. Measured at **~44ms** standalone for pydantic.

`kiss/core/__init__.py` re-exports these: `from kiss.core.config import
DEFAULT_CONFIG, AgentConfig, Config` — so this always runs during startup.

Additionally, `config.py` calls `_generate_artifact_dir()` at module level,
which creates a new timestamped directory on disk (~6ms) on **every** import.
This creates filesystem garbage (empty job directories) and adds disk I/O
during startup.

### Bottleneck 5: `base.py` eagerly imports `rich` via `print_to_console.py` (~54ms)

**This was previously missing from the plan.** `kiss/core/base.py` has:
```python
from kiss.core.print_to_console import ConsolePrinter
```

`print_to_console.py` imports `rich.console`, `rich.markdown`, `rich.panel`,
`rich.syntax`, and `rich.text` — costing **~37-54ms** standalone (includes
`pygments`, `markdown_it`, `attrs`). After Fix 1 removes LLM SDK imports,
`rich` becomes the **2nd largest independent bottleneck**.

The `ConsolePrinter` is **never used by the VSCode server** — it uses its own
`VSCodePrinter`. `ConsolePrinter` is only used in `base.py`'s `set_printer()`
method when `verbose=True` and no explicit printer is provided. The VSCode
server always passes its own printer.

### Bottleneck 6: Python subprocess only starts on `resolveWebviewView`

`AgentProcess.start()` is called inside `resolveWebviewView()`, meaning the
Python subprocess doesn't begin until the webview panel is rendered. The user
sees a blank chat window during the entire ~730ms subprocess startup
(48ms `uv` overhead + ~600ms Python imports).

### Bottleneck 7: Sequential "ready" handler commands (~round-trip latency)

When the webview sends `ready`, `SorcarPanel._handleMessage` fires **5
sequential commands** to the Python backend, each requiring a stdin→stdout
round-trip:

```typescript
case 'ready':
  this.sendToWebview({ type: 'status', running: this._isRunning });
  this._agentProcess.sendCommand({ type: 'getModels' });
  this._sendWelcomeSuggestions();
  this._agentProcess.sendCommand({ type: 'getInputHistory' });
  this._agentProcess.sendCommand({ type: 'getLastSession' });
  this._sendActiveFileInfo();
```

`getLastSession` is the heaviest — it loads history, reads chat events from the
DB, calls `resume_chat`, and checks for pending merge JSON on disk.

### Bottleneck 8: `findKissProject()` repeated filesystem scans

`findKissProject()` is called independently 3+ times:
1. In `AgentProcess.start()` (on `resolveWebviewView`)
2. In `ensureDependencies()` (on `activate()`)
3. In `_getVersion()` (during HTML generation)

Each call reads `pyproject.toml` files and searches up to 10 parent
directories, plus checks 4 common home directory locations.

### Bottleneck 9: Synchronous `_getVersion()` during HTML generation

`_getHtmlContent()` calls `_getVersion()` which synchronously reads
`_version.py` from disk using `findKissProject()` + `fs.readFileSync`. This
runs on the extension host thread during `resolveWebviewView`, blocking webview
rendering.

---

## Optimization Plan (Ordered by Impact)

### Fix 1: Clean up `kiss/core/__init__.py` — remove eager model imports (saves ~540ms)

**File:** `src/kiss/core/__init__.py`

**Root cause:** This file re-exports `AnthropicModel`, `GeminiModel`,
`OpenAICompatibleModel` from `kiss.core.models`, which triggers the entire LLM
SDK import chain. Every file doing `from kiss.core import config as
config_module` pays the full ~540ms cost.

**Change:** Remove the model class re-exports from `__init__.py`. Any code
that needs model classes should import them directly from
`kiss.core.models.model_info` (via `model()` factory) or from
`kiss.core.models` explicitly.

```python
# kiss/core/__init__.py — AFTER
from kiss.core.config import DEFAULT_CONFIG, AgentConfig, Config
from kiss.core.kiss_error import KISSError

__all__ = ["AgentConfig", "Config", "DEFAULT_CONFIG", "KISSError"]
```

Any external code using `from kiss.core import AnthropicModel` etc. must be
updated to `from kiss.core.models import AnthropicModel`. Search the codebase
for all such usages and fix them.

### Fix 2: Lazy-import LLM SDKs in `kiss/core/models/__init__.py` and `model_info.py` (saves ~540ms if accessed)

**File:** `src/kiss/core/models/__init__.py`, `src/kiss/core/models/model_info.py`

Even after Fix 1 cleans up `kiss/core/__init__.py`, `model_info.py` has at
the top level:
```python
from kiss.core.models import AnthropicModel, GeminiModel, NovitaModel, OpenAICompatibleModel
```
This triggers all SDK imports at `model_info.py` import time (which happens
when `server.py` imports it for `MODEL_INFO` and `get_available_models`).

**Change:**

a) Make `kiss/core/models/__init__.py` use lazy `__getattr__`:
```python
# kiss/core/models/__init__.py — AFTER
from kiss.core.models.model import Attachment, Model

def __getattr__(name: str):
    if name == "AnthropicModel":
        from kiss.core.models.anthropic_model import AnthropicModel
        return AnthropicModel
    if name == "GeminiModel":
        from kiss.core.models.gemini_model import GeminiModel
        return GeminiModel
    # ... etc
    raise AttributeError(name)
```

b) **Critical:** Move `from kiss.core.models import ...` inside `model()` and
`_openai_compatible()` in `model_info.py`, since top-level `from X import Y`
triggers `__getattr__` immediately at import time, defeating the lazy loading:
```python
# model_info.py — move the 4 model class imports from top-level to inside model():
def model(model_name: str, ...) -> Model:
    from kiss.core.models import AnthropicModel, GeminiModel, NovitaModel, OpenAICompatibleModel
    ...

def _openai_compatible(...) -> Model:
    from kiss.core.models import OpenAICompatibleModel
    ...
```

### Fix 3: Clean up `kiss/agents/__init__.py` — remove eager imports (saves ~35ms+)

**File:** `src/kiss/agents/__init__.py`

This file eagerly imports `kiss.agents.kiss` which pulls in `DockerManager`
(~35ms standalone after Fix 1). These convenience re-exports are not needed at
server startup.

**Change:** Either make the imports lazy (via `__getattr__`) or remove them
entirely if they're only used in scripts/tests.

### Fix 4: Lazy-import docker in `sorcar_agent.py` and `relentless_agent.py` (saves ~35ms)

**Files:**
- `src/kiss/agents/sorcar/sorcar_agent.py`
- `src/kiss/core/relentless_agent.py`

Both files import `DockerManager` and/or `DockerTools` at module level. Docker
is only used when Docker mode is explicitly enabled, never during normal VS
Code chat operation.

**Change:** Move docker imports inside the methods that use them:

```python
# sorcar_agent.py — move inside _get_tools() and run() where docker is used
def _get_tools(self):
    if self.docker_manager:
        from kiss.docker.docker_tools import DockerTools
        ...

# relentless_agent.py — move inside run() where DockerManager context manager is entered
def run(self, ...):
    ...
    if self.docker_image:
        from kiss.docker.docker_manager import DockerManager
        with DockerManager(self.docker_image) as docker_mgr:
            ...
```

Also clean up `kiss/docker/__init__.py` to not eagerly import both classes.

### Fix 5: Clean up `kiss/agents/sorcar/__init__.py` — defer config import

**File:** `src/kiss/agents/sorcar/__init__.py`

Currently does `import kiss.agents.sorcar.config` which pulls in pydantic,
pydantic_settings (~13ms), and config_builder. This runs whenever
`kiss.agents.sorcar` is imported as a package (which happens on every server
startup).

**Change:** Use lazy import or remove if not needed at package init time.

### Fix 6: Lazy-import `rich` in `base.py` via deferred `ConsolePrinter` (saves ~54ms)

**File:** `src/kiss/core/base.py`

**Root cause:** `base.py` has `from kiss.core.print_to_console import
ConsolePrinter` at the top level. This pulls in `rich` (~37-54ms), which
includes `pygments`, `markdown_it`, and `attrs`. The VSCode server **never
uses `ConsolePrinter`** — it has its own `VSCodePrinter`.

**Change:** Move the import inside `set_printer()`, the only method that uses
`ConsolePrinter`:

```python
# base.py — BEFORE
from kiss.core.print_to_console import ConsolePrinter
...
class Base:
    def set_printer(self, printer=None, verbose=None):
        if printer:
            self.printer = printer
        elif verbose is not False and config_module.DEFAULT_CONFIG.agent.verbose:
            self.printer = ConsolePrinter()
        ...

# base.py — AFTER (remove top-level import)
class Base:
    def set_printer(self, printer=None, verbose=None):
        if printer:
            self.printer = printer
        elif verbose is not False and config_module.DEFAULT_CONFIG.agent.verbose:
            from kiss.core.print_to_console import ConsolePrinter
            self.printer = ConsolePrinter()
        ...
```

This is safe because the VSCode server always passes an explicit printer, so
`ConsolePrinter` is never even instantiated in the server path.

### Fix 7: Pre-start the Python subprocess during `activate()` (saves ~730ms perceived)

**File:** `src/kiss/agents/vscode/src/SorcarPanel.ts`, `src/kiss/agents/vscode/src/extension.ts`

Currently `AgentProcess.start()` is only called inside `resolveWebviewView()`.
The user stares at a blank chat window during the ~730ms subprocess boot.

**Change:** Start the agent process eagerly during `activate()`, right after
creating the `SorcarViewProvider`. Cache the `findKissProject()` result:

```typescript
// extension.ts activate():
const kissProjectPath = findKissProject();
primaryProvider = new SorcarViewProvider(extensionUri, mergeManager, kissProjectPath);
primaryProvider.preStartAgent(workDir);
```

Then in `resolveWebviewView`, skip `this._agentProcess.start()` if already
running. The subprocess will be warm by the time the webview sends `ready`.

### Fix 8: Cache `findKissProject()` result (saves ~10–30ms repeated I/O)

**File:** `src/kiss/agents/vscode/src/AgentProcess.ts`

Called 3+ times during activation (`ensureDependencies`, `_getVersion`,
`AgentProcess.start`). Cache after first successful lookup:

```typescript
let _cachedKissProject: string | null | undefined;
export function findKissProject(): string | null {
  if (_cachedKissProject !== undefined) return _cachedKissProject;
  // ... existing search logic ...
  _cachedKissProject = result;
  return result;
}
```

### Fix 9: Cache `_getVersion()` result (saves ~5ms per HTML render)

**File:** `src/kiss/agents/vscode/src/SorcarPanel.ts`

`_getVersion()` reads `_version.py` from disk every time `_getHtmlContent` is
called. Cache it as a class-level static value.

### Fix 10: Batch initial commands into single `init` command (saves round-trips)

**File:** `src/kiss/agents/vscode/server.py`

Instead of 5 separate commands on `ready`, add a single `init` command:

```python
elif cmd_type == "init":
    self._get_models()
    self._get_input_history()
    self._get_last_session()
```

```typescript
case 'ready':
  this.sendToWebview({ type: 'status', running: this._isRunning });
  this._agentProcess.sendCommand({ type: 'init' });
  this._sendWelcomeSuggestions();
  this._sendActiveFileInfo();
```

### Fix 11: Defer `_generate_artifact_dir()` in `config.py` (saves ~6ms + disk I/O)

**File:** `src/kiss/core/config.py`

`config.py` calls `_generate_artifact_dir()` at module level, which:
- Calls `time.strftime()` and `random.randint()`
- Creates a new timestamped directory on disk via `Path.mkdir(parents=True)`

This runs on **every** import, creating empty job directories as garbage.
The artifact directory is only needed when a task actually runs.

**Change:** Make `artifact_dir` lazy — use a property or callable default in
`AgentConfig` that defers directory creation until first access:

```python
# config.py — BEFORE
artifact_dir = _generate_artifact_dir()
class AgentConfig(BaseModel):
    artifact_dir: str = Field(default=artifact_dir, ...)

# config.py — AFTER (defer until first access)
class AgentConfig(BaseModel):
    artifact_dir: str = Field(default="", ...)

    def model_post_init(self, __context: Any) -> None:
        if not self.artifact_dir:
            self.artifact_dir = _generate_artifact_dir()
```

Or simpler: use `default_factory=_generate_artifact_dir` so it only runs when
`AgentConfig()` is instantiated (which still happens at `DEFAULT_CONFIG =
Config()` time). A better approach is to make `_generate_artifact_dir()` not
create the directory (just compute the path) and have the agent create it when
actually needed.

---

## Full Import Chain Diagram

```
server.py
├── kiss.agents.sorcar.persistence       (sqlite3, ~small)
├── kiss.agents.sorcar.stateful_sorcar_agent
│   ├── yaml (~5ms)
│   └── kiss.agents.sorcar.sorcar_agent
│       ├── kiss.agents.sorcar.useful_tools  (stdlib only, lightweight)
│       ├── kiss.agents.sorcar.web_use_tool  (lightweight + webbrowser ~6ms)
│       ├── kiss.core.relentless_agent
│       │   ├── kiss.core.base
│       │   │   ├── kiss.core (→ __init__.py)          ← TRIGGERS ALL LLM SDKs (~540ms)
│       │   │   │   ├── kiss.core.config (pydantic)    (~44ms)
│       │   │   │   └── kiss.core.models               ← all 4 model classes
│       │   │   │       ├── anthropic_model → anthropic (~75ms)
│       │   │   │       ├── openai_compatible_model → openai (~137ms)
│       │   │   │       └── gemini_model → google.genai (~268ms)
│       │   │   ├── kiss.core.print_to_console → rich  (~54ms) ← NEW: previously missing
│       │   │   └── kiss.core.models.model_info (also imports model classes)
│       │   ├── kiss.core.kiss_agent (→ kiss.core again, cached)
│       │   └── kiss.docker.docker_manager → docker    (~35ms after Fix 1)
│       ├── kiss.docker.docker_manager                  (already imported)
│       └── kiss.docker.docker_tools
├── kiss.agents.vscode.browser_ui
│   └── kiss.core.printer → yaml (cached)
├── kiss.agents.vscode.diff_merge
│   └── kiss.core (cached)
├── kiss.agents.vscode.helpers
│   └── kiss.core.models.model_info (cached)
└── kiss.core.models.model_info (cached)
```

Key insight: `kiss.core.__init__.py` is the chokepoint. Fixing it (Fix 1) alone
eliminates ~540ms. After that, `rich` (~54ms), `docker` (~35ms), and
`pydantic` (~44ms) are the next independent targets.

---

## Expected Impact

| Fix | Savings | Type |
|-----|---------|------|
| Fix 1: Clean `kiss/core/__init__.py` | ~540ms | Python startup (root cause) |
| Fix 2: Lazy `models/__init__.py` + `model_info.py` | safety net for Fix 1 | Python startup |
| Fix 3: Clean `kiss/agents/__init__.py` | ~35ms | Python startup |
| Fix 4: Lazy docker in sorcar/relentless | ~35ms (after Fix 1) | Python startup |
| Fix 5: Clean `sorcar/__init__.py` | ~13ms | Python startup |
| Fix 6: Lazy `rich` in `base.py` | ~54ms (after Fix 1) | Python startup |
| Fix 7: Pre-start subprocess | ~730ms perceived | Parallelization |
| Fix 8: Cache findKissProject | ~10–30ms | Repeated I/O |
| Fix 9: Cache _getVersion | ~5ms | Repeated I/O |
| Fix 10: Batch init commands | ~20–50ms | Round-trip elimination |
| Fix 11: Defer _generate_artifact_dir | ~6ms + disk cleanup | Import-time side effects |

**Fixes 1–6** reduce Python server import time from ~600ms to **<30ms**.
- After Fix 1: ~600ms → ~60ms (removes LLM SDK chain)
- After Fixes 2: safety net (no additional savings if Fix 1 is done)
- After Fix 3: ~60ms → ~25ms (removes docker via agents/__init__.py)
- After Fix 4: ensures docker stays lazy even via direct imports
- After Fix 5: ~25ms → ~12ms (removes pydantic_settings via sorcar config)
- After Fix 6: ~12ms → <10ms (removes rich)

**Fix 7** overlaps the remaining startup with VS Code panel rendering.
Combined, the chat window should appear interactive in **<100ms** after the
panel becomes visible, down from **~1–2 seconds**.

## Implementation Order

1. **Fix 1** (highest impact, root cause — `kiss/core/__init__.py`)
2. **Fix 2** (safety net, makes `kiss/core/models/__init__.py` + model_info lazy)
3. **Fix 6** (lazy `rich` — 2nd largest independent bottleneck after Fix 1)
4. **Fix 3** (clean `kiss/agents/__init__.py`)
5. **Fix 4** (lazy docker imports)
6. **Fix 5** (clean `sorcar/__init__.py`)
7. **Fix 7** (pre-start subprocess — highest perceived impact after Python fixes)
8. **Fix 10** (batch init commands)
9. **Fix 8** (cache findKissProject)
10. **Fix 9** (cache _getVersion)
11. **Fix 11** (defer _generate_artifact_dir)

## Testing Strategy

- Run existing test suite (`uv run check --full`) after each fix
- Measure import time with `python3 -X importtime` before and after Fixes 1–6
- Measure end-to-end time from `activate()` to first `ready` response
- Verify all features still work: task submission, model selection, history,
  merge view, commit message generation, autocomplete
- Grep codebase for all `from kiss.core import` to update any broken imports
