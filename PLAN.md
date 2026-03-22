# PLAN: KISS Sorcar VS Code Extension

## Overview

This plan details how to implement a VS Code extension that mirrors the Sorcar browser-based chatbot interface but runs natively inside VS Code. The extension will use `src/kiss/agents/sorcar/sorcar_agent.py` as its backend agent and provide the same UI layout, buttons, and functionalities as the browser-based Sorcar.

______________________________________________________________________

## Research Summary

### How Claude Code Extension Works

Based on research of the official Claude Code VS Code extension and open-source implementations like Claudix:

1. **Architecture**: The extension uses VS Code's Webview API to render a custom HTML/CSS/JS chat interface within a panel
1. **Communication**: Uses `postMessage` API for bidirectional communication between the webview and the extension host
1. **Backend**: The extension host (Node.js/TypeScript) spawns or communicates with a backend process (in our case, the Python sorcar_agent)
1. **Features**:
   - Chat panel with message history
   - Model selector dropdown
   - File attachment support
   - Stop/cancel functionality
   - Inline code rendering with syntax highlighting
   - Tool call visualization (expandable sections)
   - Streaming responses
   - Session/conversation history

### Sorcar Browser UI Key Components

From `chatbot_ui.py` and `sorcar.py`:

1. **Header**: Logo, status indicator (idle/running), history button
1. **Output Area**: Scrollable message display with:
   - User messages (with optional image attachments)
   - Assistant text responses (markdown rendered)
   - Thinking blocks (collapsible)
   - Tool calls (collapsible with parameters and results)
   - Result cards (success/error summary)
   - Usage statistics
1. **Input Area**:
   - Expandable textarea with ghost text autocomplete
   - Model picker dropdown (searchable with grouped models)
   - File upload button
   - Send button (circular, blue)
   - Stop button (red, appears when running)
   - Clear button
   - File chips for attachments
1. **Sidebar**: Conversation history with search
1. **Code Server Integration**: Split view with embedded code-server
1. **Merge Toolbar**: Git diff/merge controls

### Sorcar Workflow Features

Features from `sorcar.py` that need VS Code equivalents:

| Sorcar Feature | VS Code Equivalent |
|----------------|-------------------|
| Browser-based code-server | Native VS Code editor (FREE) |
| File browser in code-server | Native VS Code Explorer (FREE) |
| Terminal in code-server | Native VS Code Terminal (FREE) |
| Git integration via merge toolbar | Native VS Code Git (FREE, but need merge review) |
| Split view (chat + editor) | VS Code Panel/Sidebar layout (FREE) |
| Active file tracking | VS Code Active Editor API (FREE) |
| File autocomplete (@file mentions) | MUST IMPLEMENT |
| Conversation history | MUST IMPLEMENT |
| Model selection with cost display | MUST IMPLEMENT |
| File attachments (images, PDFs) | MUST IMPLEMENT |
| Streaming responses | MUST IMPLEMENT |
| Stop/cancel task | MUST IMPLEMENT |
| ask_user_question callback | MUST IMPLEMENT |
| wait_for_user (browser action) | MUST IMPLEMENT |
| Merge view for reviewing changes | Can use VS Code diff view or MUST IMPLEMENT |

______________________________________________________________________

## Implementation Plan

### Phase 1: Project Structure Setup

#### Directory Structure

```
src/kiss/agents/vscode/
├── package.json           # Extension manifest
├── tsconfig.json          # TypeScript config
├── src/
│   ├── extension.ts       # Extension entry point
│   ├── SorcarPanel.ts     # Webview panel manager
│   ├── AgentProcess.ts    # Python agent process manager
│   └── types.ts           # TypeScript type definitions
├── media/
│   ├── main.css           # Webview styles (ported from chatbot_ui.py)
│   └── main.js            # Webview JavaScript
├── webview/
│   └── index.html         # Webview HTML template
└── out/                   # Compiled TypeScript output
```

#### package.json Configuration

```json
{
  "name": "kiss-sorcar",
  "displayName": "KISS Sorcar",
  "description": "AI coding assistant powered by KISS Sorcar Agent",
  "version": "0.1.0",
  "publisher": "kiss",
  "engines": {"vscode": "^1.98.0"},
  "categories": ["AI", "Chat"],
  "activationEvents": ["onCommand:kissSorcar.openPanel"],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {"command": "kissSorcar.openPanel", "title": "KISS: Open Sorcar Chat"},
      {"command": "kissSorcar.newConversation", "title": "KISS: New Conversation"},
      {"command": "kissSorcar.stopTask", "title": "KISS: Stop Current Task"}
    ],
    "viewsContainers": {
      "activitybar": [{
        "id": "kissSorcar",
        "title": "KISS Sorcar",
        "icon": "media/spark.svg"
      }]
    },
    "views": {
      "kissSorcar": [{
        "type": "webview",
        "id": "kissSorcar.chatView",
        "name": "Chat"
      }]
    },
    "configuration": {
      "title": "KISS Sorcar",
      "properties": {
        "kissSorcar.defaultModel": {
          "type": "string",
          "default": "claude-opus-4-6",
          "description": "Default LLM model"
        },
        "kissSorcar.kissProjectPath": {
          "type": "string",
          "default": "",
          "description": "Path to KISS project (auto-detected if empty)"
        }
      }
    }
  }
}
```

### Phase 2: Extension Host Implementation

#### extension.ts

- Register commands: `openPanel`, `newConversation`, `stopTask`
- Create webview provider for sidebar view
- Handle activation/deactivation lifecycle
- Manage AgentProcess singleton

#### SorcarPanel.ts

Main webview panel manager:

```typescript
class SorcarPanel {
  // Webview management
  private _panel: WebviewView | WebviewPanel
  private _agentProcess: AgentProcess

  // Message handlers for webview -> extension
  handleMessage(message: WebviewMessage) {
    switch(message.type) {
      case 'submit': // Send task to agent
      case 'stop': // Stop current task
      case 'selectModel': // Change model
      case 'uploadFile': // Handle file attachment
      case 'getHistory': // Load conversation history
      case 'resumeSession': // Resume past conversation
      case 'getFiles': // File autocomplete
      case 'userAnswer': // Response to ask_user_question
      case 'userActionDone': // Response to wait_for_user
    }
  }

  // Send events to webview
  sendToWebview(event: AgentEvent) {
    this._panel.webview.postMessage(event)
  }

  // Build webview HTML with CSP and nonce
  getHtmlContent(): string
}
```

#### AgentProcess.ts

Python subprocess manager:

```typescript
class AgentProcess {
  private process: ChildProcess | null
  private eventQueue: EventEmitter

  // Discover KISS project path
  findKissProject(): string {
    // 1. Check configuration setting
    // 2. Search up from workspace folder for pyproject.toml with kiss
    // 3. Check common locations
  }

  // Start the Python backend
  start(workDir: string, kissPath: string): void {
    // Spawn: uv run python -m kiss.agents.vscode.server
    // Set up stdio/stderr handlers
    // Parse JSON events from stdout
  }

  // Send task to agent
  runTask(params: TaskParams): void {
    // Write JSON command to stdin
  }

  // Stop current task
  stop(): void {
    // Send stop signal
    // Set stop event in agent
  }

  // Cleanup
  dispose(): void
}
```

### Phase 3: Python Backend Server

Create `src/kiss/agents/vscode/server.py`:

```python
"""VS Code extension backend server for Sorcar agent."""

import json
import sys
import threading
from typing import Any

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.printer import Printer


class VSCodePrinter(Printer):
    """Printer that outputs JSON events to stdout for VS Code."""

    def __init__(self) -> None:
        super().__init__(interactive=False, verbose=False)
        self._lock = threading.Lock()

    def _emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            sys.stdout.write(json.dumps(event) + "\n")
            sys.stdout.flush()

    def print(self, text: str, type: str = "text", **kwargs: Any) -> None:
        self._emit({"type": type, "text": text, **kwargs})


def main() -> None:
    """Main entry point for VS Code backend server."""
    printer = VSCodePrinter()
    agent = SorcarAgent("Sorcar VS Code")

    # Read commands from stdin
    for line in sys.stdin:
        try:
            cmd = json.loads(line.strip())
            handle_command(cmd, agent, printer)
        except json.JSONDecodeError:
            printer._emit({"type": "error", "text": f"Invalid JSON: {line}"})


def handle_command(cmd: dict, agent: SorcarAgent, printer: VSCodePrinter) -> None:
    """Handle a command from VS Code."""
    cmd_type = cmd.get("type")

    if cmd_type == "run":
        # Run agent task
        result = agent.run(
            prompt_template=cmd.get("prompt", ""),
            model_name=cmd.get("model"),
            work_dir=cmd.get("workDir"),
            printer=printer,
            current_editor_file=cmd.get("activeFile"),
            attachments=cmd.get("attachments"),
            wait_for_user_callback=lambda instr, url: wait_for_user(printer, instr, url),
            ask_user_question_callback=lambda q: ask_user_question(printer, q),
        )
        printer._emit({"type": "result", "data": result})

    elif cmd_type == "stop":
        # Signal stop to agent
        pass

    elif cmd_type == "getModels":
        # Return available models
        pass

    elif cmd_type == "getHistory":
        # Return conversation history
        pass


if __name__ == "__main__":
    main()
```

### Phase 4: Webview UI Implementation

Port the CSS from `chatbot_ui.py` to `media/main.css`, adapting colors to support VS Code themes using CSS variables.

#### Theme Support

```css
:root {
  --vscode-bg: var(--vscode-editor-background);
  --vscode-fg: var(--vscode-editor-foreground);
  --vscode-border: var(--vscode-panel-border);
  --vscode-accent: var(--vscode-textLink-foreground);
  /* Map to Sorcar colors */
}
```

#### Key UI Components to Port

1. **Header** (`header` in chatbot_ui.py)

   - Logo with status indicator
   - History button (sidebar toggle)

1. **Output Area** (`#output`)

   - User message cards (`.user-msg`)
   - Text responses (`.txt`) with markdown rendering
   - Thinking blocks (`.think`) - collapsible
   - Tool call cards (`.tc`) - collapsible with params/results
   - Result cards (`.rc`)
   - Spinner animation

1. **Input Area** (`#input-area`)

   - Textarea with auto-resize (`#task-input`)
   - Ghost text overlay (`#ghost-overlay`)
   - Model picker dropdown (`#model-picker`)
   - File upload button (`#upload-btn`)
   - File chips (`#file-chips`)
   - Send button (`#send-btn`)
   - Stop button (`#stop-btn`)

1. **Autocomplete Panel** (`#autocomplete`)

   - File/folder suggestions
   - Keyboard navigation

1. **History Sidebar** (`#sidebar`)

   - Search input
   - Grouped history items

### Phase 5: Message Protocol

Define the JSON message protocol between extension and webview:

#### Extension → Webview Messages

```typescript
type ToWebviewMessage =
  | { type: 'text'; text: string }
  | { type: 'thinking'; text: string }
  | { type: 'tool_call'; name: string; params: Record<string, any> }
  | { type: 'tool_result'; name: string; result: string; error?: boolean }
  | { type: 'result'; success: boolean; summary: string }
  | { type: 'status'; running: boolean }
  | { type: 'models'; models: ModelInfo[] }
  | { type: 'history'; sessions: SessionInfo[] }
  | { type: 'files'; files: string[] }
  | { type: 'askUser'; question: string }
  | { type: 'waitForUser'; instruction: string; url: string }
```

#### Webview → Extension Messages

```typescript
type FromWebviewMessage =
  | { type: 'submit'; prompt: string; model: string; attachments: Attachment[] }
  | { type: 'stop' }
  | { type: 'selectModel'; model: string }
  | { type: 'getModels' }
  | { type: 'getHistory'; query?: string }
  | { type: 'resumeSession'; id: string }
  | { type: 'getFiles'; prefix: string }
  | { type: 'userAnswer'; answer: string }
  | { type: 'userActionDone' }
  | { type: 'openFile'; path: string }
```

### Phase 6: Feature Implementation Checklist

| Feature | Priority | Complexity | Notes |
|---------|----------|------------|-------|
| Basic chat UI | HIGH | Medium | Port from chatbot_ui.py |
| Send/receive messages | HIGH | Medium | JSON over stdio |
| Streaming text display | HIGH | Medium | Handle chunks |
| Model selector | HIGH | Low | Dropdown with search |
| Stop task | HIGH | Low | Signal to process |
| Tool call display | HIGH | Medium | Collapsible cards |
| File autocomplete (@mentions) | HIGH | Medium | Scan workspace |
| File attachments | MEDIUM | Medium | Use VS Code file picker |
| Conversation history | MEDIUM | Medium | Port from task_history.py |
| ask_user_question | MEDIUM | Low | Modal dialog in webview |
| wait_for_user (browser) | MEDIUM | Medium | External browser + callback |
| Merge/diff review | LOW | High | VS Code diff API or custom |
| Keyboard shortcuts | LOW | Low | Standard VS Code keybindings |

### Phase 7: KISS Project Path Discovery

The extension MUST NOT hardcode the KISS project path. Use this discovery order:

1. **Configuration Setting**: Check `kissSorcar.kissProjectPath` in VS Code settings
1. **Workspace Detection**: Search upward from workspace folder for `pyproject.toml` containing `[project]` with `name = "kiss"`
1. **Parent Directory**: Search from extension's directory upward
1. **Environment Variable**: Check `KISS_PROJECT_PATH` environment variable
1. **Common Locations**: Check `~/work/kiss`, `~/projects/kiss`, etc.
1. **Error**: Show error message if not found, with link to settings

```typescript
function findKissProject(): string | null {
  // 1. Config setting
  const configPath = vscode.workspace.getConfiguration('kissSorcar').get<string>('kissProjectPath')
  if (configPath && isValidKissProject(configPath)) return configPath

  // 2. Workspace parent search
  const workspaceFolders = vscode.workspace.workspaceFolders
  if (workspaceFolders) {
    for (const folder of workspaceFolders) {
      const found = searchUpward(folder.uri.fsPath)
      if (found) return found
    }
  }

  // 3. Extension path search
  // 4. Environment variable
  // 5. Common locations
  return null
}
```

______________________________________________________________________

## Build & Test Instructions

### Development Setup

```bash
cd src/kiss/agents/vscode
npm install
npm run compile  # or `npm run watch` for development
```

### Testing in VS Code

1. Open the vscode folder in VS Code
1. Press F5 to launch Extension Development Host
1. In the new window, run "KISS: Open Sorcar Chat" from Command Palette

### Packaging

```bash
npm run package  # Creates .vsix file
```

______________________________________________________________________

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Python process communication issues | Use well-defined JSON protocol, handle errors gracefully |
| Webview security (CSP) | Use strict CSP, nonces for scripts |
| Theme compatibility | Use VS Code CSS variables, test with multiple themes |
| Performance with large outputs | Virtual scrolling, message batching |
| Cross-platform paths | Use VS Code's URI/path APIs |
| Agent process crashes | Auto-restart, preserve state |

______________________________________________________________________

## Timeline Estimate

| Phase | Duration | Depends On |
|-------|----------|------------|
| Phase 1: Project Setup | 2 hours | - |
| Phase 2: Extension Host | 4 hours | Phase 1 |
| Phase 3: Python Backend | 3 hours | Phase 1 |
| Phase 4: Webview UI | 6 hours | Phase 2 |
| Phase 5: Message Protocol | 2 hours | Phase 2, 3 |
| Phase 6: Features | 8 hours | Phase 4, 5 |
| Phase 7: Path Discovery | 1 hour | Phase 2 |
| Testing & Polish | 4 hours | All |
| **Total** | **~30 hours** | |

______________________________________________________________________

## Open Questions

1. Should we support multiple concurrent agent sessions?
1. Should the chat persist between VS Code restarts?
1. How to handle long-running browser automation tasks?
1. Should we integrate with VS Code's built-in Git for merge review?

______________________________________________________________________

## References

- [VS Code Extension API](https://code.visualstudio.com/api)
- [VS Code Webview Guide](https://code.visualstudio.com/api/extension-guides/webview)
- [Claude Code VS Code Docs](https://code.claude.com/docs/en/vs-code)
- [Claudix (Open Source Reference)](https://github.com/Haleclipse/Claudix)
- Sorcar source: `src/kiss/agents/sorcar/`
