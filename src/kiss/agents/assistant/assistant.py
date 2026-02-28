"""Browser-based chatbot for RelentlessAgent-based agents."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import types
import webbrowser
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from kiss.agents.assistant.browser_ui import (
    BASE_CSS,
    EVENT_HANDLER_JS,
    HTML_HEAD,
    OUTPUT_CSS,
    BaseBrowserPrinter,
    find_free_port,
)
from kiss.agents.assistant.relentless_agent import RelentlessAgent
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import MODEL_INFO, get_available_models, get_most_expensive_model

_KISS_DIR = Path.home() / ".kiss"
HISTORY_FILE = _KISS_DIR / "task_history.json"
PROPOSALS_FILE = _KISS_DIR / "proposed_tasks.json"
MODEL_USAGE_FILE = _KISS_DIR / "model_usage.json"
MAX_HISTORY = 1000

SAMPLE_TASKS = [
    {"task": "run 'uv run check --clean' and fix", "result": ""},
    {
        "task": (
            "plan a trip to Yosemite over the weekend based on"
            " road closures, warnings, hotel availability"
        ),
        "result": "",
    },
    {
        "task": (
            "find the cheapest afternoon non-stop flight"
            " from SFO to NYC around March 15"
        ),
        "result": "",
    },
    {
        "task": (
            "run <<command>> in the background, monitor output,"
            " fix errors, and optimize the code iteratively"
        ),
        "result": "",
    },
    {
        "task": (
            "implement and validate results from the research"
            " paper https://arxiv.org/pdf/2505.10961 using relentless_coding_agent and kiss_agent"
        ),
        "result": "",
    },
    {
        "task": (
            "develop an automated evaluation framework for"
            " agent performance against benchmarks"
        ),
        "result": "",
    },
    {
        "task": (
            "launch a browser, research technical innovations,"
            " and compile a document incrementally"
        ),
        "result": "",
    },
    {
        "task": (
            "read all *.md files, check consistency with"
            " the code, and fix any inconsistencies"
        ),
        "result": "",
    },
    {
        "task": (
            "remove duplicate or redundant tests while"
            " ensuring coverage doesn't decrease"
        ),
        "result": "",
    },
]



def _normalize_history_entry(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict) and "task" in raw:
        return {"task": str(raw["task"]), "result": str(raw.get("result", ""))}
    return {"task": str(raw), "result": ""}


_history_cache: list[dict[str, str]] | None = None


def _load_history() -> list[dict[str, str]]:
    global _history_cache
    if _history_cache is not None:
        return _history_cache
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            if isinstance(data, list) and data:
                seen: set[str] = set()
                result: list[dict[str, str]] = []
                for t in data[:MAX_HISTORY]:
                    entry = _normalize_history_entry(t)
                    task_str = entry["task"]
                    if task_str not in seen:
                        seen.add(task_str)
                        result.append(entry)
                _history_cache = result
                return result
        except (json.JSONDecodeError, OSError):
            pass
    entries = [_normalize_history_entry(t) for t in SAMPLE_TASKS]
    _save_history(entries)
    return entries


def _save_history(entries: list[dict[str, str]]) -> None:
    global _history_cache
    _history_cache = entries[:MAX_HISTORY]
    try:
        _KISS_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(_history_cache, indent=2))
    except OSError:
        pass


def _set_latest_result(result: str) -> None:
    history = _load_history()
    if history:
        history[0]["result"] = result
        _save_history(history)


def _load_proposals() -> list[str]:
    if PROPOSALS_FILE.exists():
        try:
            data = json.loads(PROPOSALS_FILE.read_text())
            if isinstance(data, list):
                return [str(t) for t in data if isinstance(t, str) and t.strip()][:5]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_proposals(proposals: list[str]) -> None:
    try:
        _KISS_DIR.mkdir(parents=True, exist_ok=True)
        PROPOSALS_FILE.write_text(json.dumps(proposals))
    except OSError:
        pass


def _load_model_usage() -> dict[str, int]:
    if MODEL_USAGE_FILE.exists():
        try:
            data = json.loads(MODEL_USAGE_FILE.read_text())
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items() if isinstance(v, (int, float))}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _load_last_model() -> str:
    if MODEL_USAGE_FILE.exists():
        try:
            data = json.loads(MODEL_USAGE_FILE.read_text())
            if isinstance(data, dict):
                last = data.get("_last")
                if isinstance(last, str):
                    return last
        except (json.JSONDecodeError, OSError):
            pass
    return ""


def _record_model_usage(model: str) -> None:
    usage: dict[str, int | str] = dict(_load_model_usage())
    usage[model] = int(usage.get(model, 0)) + 1
    usage["_last"] = model
    try:
        _KISS_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_USAGE_FILE.write_text(json.dumps(usage))
    except OSError:
        pass


def _add_task(task: str) -> None:
    history = [e for e in _load_history() if e["task"] != task]
    history.insert(0, {"task": task, "result": ""})
    _save_history(history[:MAX_HISTORY])


def _scan_files(work_dir: str) -> list[str]:
    paths: list[str] = []
    skip = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    }
    try:
        for root, dirs, files in os.walk(work_dir):
            depth = os.path.relpath(root, work_dir).count(os.sep)
            if depth > 3:
                dirs.clear()
                continue
            dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith("."))
            for d in dirs:
                paths.append(os.path.relpath(os.path.join(root, d), work_dir) + "/")
            for name in sorted(files):
                paths.append(os.path.relpath(os.path.join(root, name), work_dir))
                if len(paths) >= 2000:
                    return paths
    except OSError:
        pass
    return paths


def _parse_diff_hunks(work_dir: str) -> dict[str, list[tuple[int, int, int, int]]]:
    result = subprocess.run(
        ["git", "diff", "-U0", "HEAD", "--no-color"],
        capture_output=True, text=True, cwd=work_dir,
    )
    hunks: dict[str, list[tuple[int, int, int, int]]] = {}
    current_file = ""
    for line in result.stdout.split("\n"):
        dm = re.match(r"^diff --git a/.* b/(.*)", line)
        if dm:
            current_file = dm.group(1)
            continue
        hm = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if hm and current_file:
            hunks.setdefault(current_file, []).append((
                int(hm.group(1)),
                int(hm.group(2)) if hm.group(2) is not None else 1,
                int(hm.group(3)),
                int(hm.group(4)) if hm.group(4) is not None else 1,
            ))
    return hunks


def _capture_untracked(work_dir: str) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, cwd=work_dir,
    )
    return {line.strip() for line in result.stdout.split("\n") if line.strip()}


def _prepare_merge_view(
    work_dir: str,
    data_dir: str,
    pre_hunks: dict[str, list[tuple[int, int, int, int]]],
    pre_untracked: set[str],
) -> dict[str, Any]:
    post_hunks = _parse_diff_hunks(work_dir)
    file_hunks: dict[str, list[dict[str, int]]] = {}
    for fname, hunks in post_hunks.items():
        pre = {(bs, bc) for bs, bc, _, _ in pre_hunks.get(fname, [])}
        filtered = [
            {"bs": bs - 1, "bc": bc, "cs": cs - 1, "cc": cc}
            for bs, bc, cs, cc in hunks
            if (bs, bc) not in pre
        ]
        if filtered:
            file_hunks[fname] = filtered
    new_files = _capture_untracked(work_dir) - pre_untracked
    for fname in new_files:
        fpath = Path(work_dir) / fname
        try:
            if not fpath.is_file() or fpath.stat().st_size > 2_000_000:
                continue
            line_count = len(fpath.read_text().splitlines())
            if line_count:
                file_hunks[fname] = [{"bs": 0, "bc": 0, "cs": 0, "cc": line_count}]
        except (OSError, UnicodeDecodeError):
            pass
    if not file_hunks:
        return {"error": "No changes"}
    merge_dir = Path(data_dir) / "merge-temp"
    if merge_dir.exists():
        shutil.rmtree(merge_dir)
    manifest_files = []
    for fname, hunks in file_hunks.items():
        current_path = Path(work_dir) / fname
        base_path = merge_dir / fname
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_result = subprocess.run(
            ["git", "show", f"HEAD:{fname}"],
            capture_output=True, text=True, cwd=work_dir,
        )
        base_path.write_text(base_result.stdout if base_result.returncode == 0 else "")
        manifest_files.append({
            "name": fname,
            "base": str(base_path),
            "current": str(current_path),
            "hunks": hunks,
        })
    manifest = Path(data_dir) / "pending-merge.json"
    manifest.write_text(json.dumps({
        "branch": "HEAD", "files": manifest_files,
    }))
    return {"status": "opened", "count": len(manifest_files)}


class _StopRequested(BaseException):
    pass


_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4", "codex", "computer-use")

_CS_SETTINGS = {
    "workbench.startupEditor": "none",
    "workbench.tips.enabled": False,
    "workbench.welcomePage.walkthroughs.openOnInstall": False,
    "security.workspace.trust.enabled": False,
    "update.showReleaseNotes": False,
    "workbench.panel.defaultLocation": "bottom",
    "editor.fontSize": 13,
    "terminal.integrated.fontSize": 13,
    "scm.inputFontSize": 13,
    "debug.console.fontSize": 13,
    "window.restoreWindows": "all",
    "workbench.editor.restoreViewState": True,
    "files.hotExit": "onExitAndWindowClose",
}

_CS_STATE_ENTRIES = [
    ("workbench.activity.pinnedViewlets2", "[]"),
    ("workbench.welcomePage.walkthroughMetadata", "[]"),
    ("coderGettingStarted/v1", "installed"),
    ("workbench.panel.pinnedPanels", "[]"),
    ("memento/gettingStartedService", '{"installed":true}'),
    ("profileAssociations", '{"workspaces":{}}'),
    ("userDataProfiles", '[]'),
    ("welcomePage.gettingStartedTabs", '[]'),
    ("workbench.welcomePage.opened", "true"),
    ("chat.setupCompleted", "true"),
    ("chat.panelVisible", "false"),
    ("workbench.panel.chat.hidden", "true"),
    ("workbench.panel.chatSidebar.hidden", "true"),
]

_CS_EXTENSION_JS = """\
const vscode=require("vscode");
const fs=require("fs");
const path=require("path");
function activate(ctx){
  function cleanup(){
    for(const g of vscode.window.tabGroups.all){
      for(const t of g.tabs){
        if(!t.input||!t.input.uri){
          vscode.window.tabGroups.close(t).then(()=>{},()=>{});
        }
      }
    }
    vscode.commands.executeCommand('workbench.action.closePanel');
    vscode.commands.executeCommand('workbench.action.closeAuxiliaryBar');
  }
  cleanup();
  setTimeout(cleanup,1500);
  setTimeout(cleanup,4000);
  setTimeout(cleanup,8000);
  var home=process.env.HOME||process.env.USERPROFILE||'';
  var ms={};
  var clFire=new vscode.EventEmitter();
  ctx.subscriptions.push(vscode.languages.registerCodeLensProvider({scheme:'file'},{
    onDidChangeCodeLenses:clFire.event,
    provideCodeLenses:function(doc){
      var s=ms[doc.uri.fsPath];
      if(!s||!s.hunks.length)return[];
      var L=[];
      for(var i=0;i<s.hunks.length;i++){
        var h=s.hunks[i];
        if(h.cc<=0)continue;
        var ln=Math.min(h.cs+h.cc,doc.lineCount-1);
        var r=new vscode.Range(ln,0,ln,0);
        var fp=doc.uri.fsPath;
        L.push(new vscode.CodeLens(r,{title:'\\u2705 Accept',
          command:'kiss.acceptChange',arguments:[fp,i]}));
        L.push(new vscode.CodeLens(r,{title:'\\u274c Reject',
          command:'kiss.rejectChange',arguments:[fp,i]}));
      }
      return L;
    }
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.acceptChange',function(fp,idx){
    var s=ms[fp];if(!s)return;
    s.hunks.splice(idx,1);
    if(!s.hunks.length)delete ms[fp];
    clFire.fire();checkAllDone();
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.rejectChange',async function(fp,idx){
    var s=ms[fp];if(!s)return;
    var h=s.hunks[idx];
    var ed=vscode.window.visibleTextEditors.find(function(e){return e.document.uri.fsPath===fp;});
    if(!ed)return;
    var baseLines=fs.readFileSync(s.basePath,'utf8').split('\\n');
    var repl=baseLines.slice(h.bs,h.bs+h.bc);
    if(h.cc>0){
      var last=Math.min(h.cs+h.cc-1,ed.document.lineCount-1);
      var rng=new vscode.Range(h.cs,0,last,ed.document.lineAt(last).text.length);
      await ed.edit(function(eb){eb.replace(rng,repl.join('\\n'));});
    }else if(h.bc>0){
      await ed.edit(function(eb){eb.insert(new vscode.Position(h.cs+1,0),repl.join('\\n')+'\\n');});
    }
    var diff=repl.length-h.cc;
    s.hunks.splice(idx,1);
    for(var i=idx;i<s.hunks.length;i++)s.hunks[i].cs+=diff;
    if(!s.hunks.length)delete ms[fp];
    clFire.fire();checkAllDone();
  }));
  function checkAllDone(){
    if(Object.keys(ms).length>0)return;
    vscode.workspace.saveAll(false).then(function(){
      vscode.window.showInformationMessage('All changes reviewed.');
    });
  }
  var mp=path.join(home,'.kiss','code-server-data','pending-merge.json');
  var op=path.join(home,'.kiss','code-server-data','pending-open.json');
  var iv=setInterval(function(){
    try{
      if(fs.existsSync(op)){
        var od=JSON.parse(fs.readFileSync(op,'utf8'));
        fs.unlinkSync(op);
        var uri=vscode.Uri.file(od.path);
        vscode.workspace.openTextDocument(uri).then(function(doc){
          vscode.window.showTextDocument(doc,{preview:false});
        });
      }
      if(!fs.existsSync(mp))return;
      var data=JSON.parse(fs.readFileSync(mp,'utf8'));
      fs.unlinkSync(mp);
      openMerge(data);
    }catch(e){}
  },800);
  ctx.subscriptions.push({dispose:function(){clearInterval(iv)}});
  async function openMerge(data){
    ms={};
    for(var f of(data.files||[])){
      var baseUri=vscode.Uri.file(f.base);
      var currentUri=vscode.Uri.file(f.current);
      await vscode.commands.executeCommand('vscode.diff',baseUri,currentUri,f.name+' (Changes)');
      ms[f.current]={basePath:f.base,hunks:(f.hunks||[]).map(function(h){
        return{cs:h.cs,cc:h.cc,bs:h.bs,bc:h.bc};
      })};
    }
    clFire.fire();
    vscode.window.showInformationMessage(
      'Reviewing '+data.files.length+' file(s) vs '+data.branch
      +'. Use \\u2705 Accept / \\u274c Reject on each change.');
  }
}
module.exports={activate};
"""


def _setup_code_server(data_dir: str) -> None:
    """Pre-configure code-server user data: settings, state DB, and cleanup extension."""
    user_dir = Path(data_dir) / "User"
    user_dir.mkdir(parents=True, exist_ok=True)

    settings_file = user_dir / "settings.json"
    try:
        existing = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    except (json.JSONDecodeError, OSError):
        existing = {}
    existing.update(_CS_SETTINGS)
    settings_file.write_text(json.dumps(existing, indent=2))

    state_db = user_dir / "globalStorage" / "state.vscdb"
    state_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(state_db)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ItemTable"
            " (key TEXT UNIQUE ON CONFLICT REPLACE, value TEXT)"
        )
        for key, value in _CS_STATE_ENTRIES:
            conn.execute(
                "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (key, value),
            )
        conn.commit()

    ws_storage = user_dir / "workspaceStorage"
    if ws_storage.exists():
        for ws_dir in ws_storage.iterdir():
            for sub in ("chatSessions", "chatEditingSessions"):
                chat_dir = ws_dir / sub
                if chat_dir.exists():
                    shutil.rmtree(chat_dir, ignore_errors=True)

    ext_dir = Path(data_dir) / "extensions" / "kiss-init"
    ext_dir.mkdir(parents=True, exist_ok=True)
    (ext_dir / "package.json").write_text(json.dumps({
        "name": "kiss-init", "version": "0.0.1", "publisher": "kiss",
        "engines": {"vscode": "^1.80.0"},
        "activationEvents": ["onStartupFinished"],
        "main": "./extension.js",
        "contributes": {
            "commands": [
                {"command": "kiss.acceptChange", "title": "Accept Change"},
                {"command": "kiss.rejectChange", "title": "Reject Change"},
            ],
        },
    }))
    (ext_dir / "extension.js").write_text(_CS_EXTENSION_JS)


def _model_vendor_order(name: str) -> int:
    if name.startswith("claude-"):
        return 0
    if name.startswith(_OPENAI_PREFIXES) and not name.startswith("openai/"):
        return 1
    if name.startswith("gemini-"):
        return 2
    if name.startswith("minimax-"):
        return 3
    if name.startswith("openrouter/"):
        return 4
    return 5


CHATBOT_CSS = r"""
body{
  font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
  background:#0a0a0c;display:block;
}
header{
  background:rgba(10,10,12,0.85);backdrop-filter:blur(24px);
  -webkit-backdrop-filter:blur(24px);
  border-bottom:1px solid rgba(255,255,255,0.06);padding:14px 24px;z-index:50;
  box-shadow:0 1px 12px rgba(0,0,0,0.3);
}
.logo{font-size:12px;color:#4da6ff;font-weight:600;letter-spacing:-0.2px}
.logo span{
  color:rgba(255,255,255,0.55);font-weight:400;font-size:12px;margin-left:10px;
  max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  display:inline-block;vertical-align:middle;
}
.status{font-size:12px;color:rgba(255,255,255,0.6)}
.dot{width:7px;height:7px;background:rgba(255,255,255,0.4)}
.dot.running{background:#22c55e}
#output{
  flex:1;overflow-y:auto;padding:32px 24px 24px;
  scroll-behavior:smooth;min-height:0;
}
.ev,.txt,.spinner,.empty-msg,.user-msg{max-width:820px;margin-left:auto;margin-right:auto}
.user-msg{
  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
  border-radius:14px;padding:14px 20px;margin:20px auto 16px;
  font-size:14.5px;line-height:1.6;color:rgba(255,255,255,0.88);
}
.txt{
  font-size:14.5px;line-height:1.75;color:rgba(255,255,255,0.82);padding:8px 14px;
  margin:6px auto;
}
.think{
  border:1px solid rgba(168,130,255,0.12);
  background:rgba(168,130,255,0.03);border-radius:10px;margin:12px auto;
  padding:12px 16px;
}
.think .lbl{color:rgba(168,130,255,0.7)}
.think .cnt{color:rgba(255,255,255,0.4)}
.tc{
  border:1px solid rgba(88,166,255,0.15);border-radius:12px;
  margin:12px auto;background:rgba(88,166,255,0.02);
  transition:border-color 0.2s,box-shadow 0.2s;
}
.tc:hover{box-shadow:0 2px 20px rgba(88,166,255,0.08);border-color:rgba(88,166,255,0.3)}
.tc-h{
  padding:10px 16px;background:rgba(88,166,255,0.03);border-radius:12px 12px 0 0;
  border-bottom:1px solid rgba(88,166,255,0.08);
}
.tc-h:hover{background:rgba(88,166,255,0.06)}
.tn{color:rgba(88,166,255,0.9);font-size:13px}
.tp{font-size:12px;color:rgba(120,180,255,0.45)}
.td{color:rgba(255,255,255,0.3)}
.tr{
  border:1px solid rgba(34,197,94,0.15);
  background:rgba(34,197,94,0.02);border-radius:10px;
}
.tr.err{
  border-color:rgba(248,81,73,0.15);
  background:rgba(248,81,73,0.02);
}
.rc{
  border:1px solid rgba(34,197,94,0.25);border-radius:14px;
  background:rgba(34,197,94,0.02);
  box-shadow:0 0 20px rgba(34,197,94,0.04);
}
.rc-h{
  padding:16px 24px;background:rgba(34,197,94,0.05);
  border-bottom:1px solid rgba(34,197,94,0.12);
}
.usage{
  border:1px solid rgba(255,255,255,0.06);background:rgba(255,255,255,0.02);
  color:rgba(255,255,255,0.3);border-radius:8px;
}
.spinner{color:rgba(255,255,255,0.35)}
.spinner::before{border-color:rgba(255,255,255,0.08);border-top-color:rgba(88,166,255,0.7)}
#input-area{
  flex-shrink:0;padding:0 24px 24px;position:relative;
  background:linear-gradient(transparent,rgba(10,10,12,0.9) 50%);
  padding-top:24px;
}
#input-container{
  max-width:820px;margin:0 auto;position:relative;
  background:rgba(255,255,255,0.035);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:16px;padding:14px 18px;
  box-shadow:0 0 0 1px rgba(255,255,255,0.02),0 8px 40px rgba(0,0,0,0.35);
  transition:border-color 0.3s,box-shadow 0.3s;
}
#input-container:focus-within{
  border-color:rgba(88,166,255,0.4);
  box-shadow:0 0 0 1px rgba(88,166,255,0.12),0 0 30px rgba(88,166,255,0.1),
    0 8px 40px rgba(0,0,0,0.35);
}
#input-wrap{position:relative;display:flex;align-items:flex-start;gap:8px}
#task-input{
  flex:1;min-width:0;background:transparent;border:none;
  color:rgba(255,255,255,0.88);font-size:15px;font-family:inherit;
  resize:none;outline:none;line-height:1.5;
  max-height:200px;min-height:24px;
  position:relative;z-index:1;
}
#task-input::placeholder{color:rgba(255,255,255,0.28)}
#task-input:disabled{opacity:0.35;cursor:not-allowed}
#ghost-overlay{
  position:absolute;top:0;left:0;right:0;bottom:0;
  pointer-events:none;user-select:none;
  font-size:15px;font-family:inherit;line-height:1.5;
  white-space:pre-wrap;word-break:break-word;overflow:hidden;
  z-index:0;
}
.gm{visibility:hidden;white-space:pre-wrap}
.gs{color:rgba(255,255,255,0.22);font-style:italic}
#input-footer{
  display:flex;justify-content:space-between;align-items:center;
  margin-top:10px;padding-top:10px;
  border-top:1px solid rgba(255,255,255,0.04);
}
#model-picker{position:relative}
#model-btn{
  background:rgba(255,255,255,0.03);color:rgba(255,255,255,0.5);
  border:1px solid rgba(255,255,255,0.08);border-radius:8px;
  padding:6px 12px;font-size:12px;font-family:inherit;
  outline:none;cursor:pointer;max-width:300px;transition:border-color 0.2s;
  display:flex;align-items:center;gap:6px;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;
}
#model-btn:hover{border-color:rgba(255,255,255,0.16);color:rgba(255,255,255,0.65)}
#model-btn svg{flex-shrink:0;opacity:0.4}
#model-dropdown{
  position:absolute;bottom:100%;left:0;min-width:320px;max-width:420px;
  background:rgba(18,18,20,0.97);backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,0.08);border-radius:12px;
  max-height:360px;display:none;z-index:15;
  box-shadow:0 -8px 32px rgba(0,0,0,0.5);overflow:hidden;
  flex-direction:column;
}
#model-dropdown.open{display:flex}
#model-search{
  width:100%;background:transparent;border:none;
  border-bottom:1px solid rgba(255,255,255,0.06);
  color:rgba(255,255,255,0.8);font-size:12px;font-family:inherit;
  padding:10px 14px;outline:none;
}
#model-search::placeholder{color:rgba(255,255,255,0.25)}
#model-list{overflow-y:auto;flex:1}
.model-item{
  padding:7px 14px;cursor:pointer;font-size:12px;
  display:flex;justify-content:space-between;align-items:center;
  border-bottom:1px solid rgba(255,255,255,0.02);transition:background 0.08s;
  color:rgba(255,255,255,0.6);
}
.model-item:hover,.model-item.sel{background:rgba(88,166,255,0.08)}
.model-item.active{color:rgba(88,166,255,0.9);font-weight:500}
.model-cost{font-size:10px;color:rgba(255,255,255,0.2);flex-shrink:0;margin-left:12px}
.model-group-hdr{
  padding:6px 14px 4px;font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:0.05em;
  color:rgba(255,255,255,0.25);
  background:rgba(18,18,20,0.97);
  border-bottom:1px solid rgba(255,255,255,0.04);
  position:sticky;top:0;z-index:1;
}
#input-actions{display:flex;gap:8px;align-items:center}
#send-btn{
  background:rgba(88,166,255,0.15);color:rgba(88,166,255,0.9);border:none;
  border-radius:50%;width:36px;height:36px;cursor:pointer;
  transition:all 0.2s;display:flex;align-items:center;justify-content:center;
  flex-shrink:0;
}
#send-btn:hover{background:rgba(88,166,255,0.3);color:#fff;box-shadow:0 0 16px rgba(88,166,255,0.2)}
#send-btn:disabled{opacity:0.2;cursor:not-allowed;box-shadow:none}
#send-btn svg{width:16px;height:16px}
#stop-btn{
  background:rgba(248,81,73,0.1);color:#f85149;
  border:1px solid rgba(248,81,73,0.15);
  border-radius:50%;width:36px;height:36px;
  cursor:pointer;transition:all 0.2s;display:none;
  align-items:center;justify-content:center;flex-shrink:0;
}
#stop-btn:hover{background:rgba(248,81,73,0.2);box-shadow:0 0 16px rgba(248,81,73,0.15)}
#stop-btn svg{width:14px;height:14px}
#clear-btn{
  background:none;color:rgba(255,255,255,0.2);border:none;
  width:24px;height:24px;cursor:pointer;flex-shrink:0;
  transition:color 0.15s;display:flex;align-items:center;justify-content:center;
  padding:0;margin-top:1px;
}
#clear-btn:hover{color:rgba(255,255,255,0.6)}
#clear-btn svg{width:14px;height:14px}
#autocomplete{
  position:absolute;bottom:100%;left:0;right:0;
  max-width:820px;margin:0 auto;
  background:rgba(18,18,20,0.97);backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,0.08);border-radius:12px;
  max-height:340px;overflow-y:auto;display:none;z-index:10;
  box-shadow:0 -8px 32px rgba(0,0,0,0.5),0 0 0 1px rgba(255,255,255,0.03);
}
.ac-section{
  padding:6px 16px 4px;font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:0.05em;
  color:rgba(255,255,255,0.2);
  background:rgba(18,18,20,0.97);
  border-bottom:1px solid rgba(255,255,255,0.03);
  position:sticky;top:0;z-index:1;
}
.ac-item{
  padding:8px 16px;cursor:pointer;font-size:13px;
  border-bottom:1px solid rgba(255,255,255,0.025);
  display:flex;align-items:center;gap:8px;transition:background 0.08s;
}
.ac-item:last-child{border-bottom:none}
.ac-item:hover,.ac-item.sel{background:rgba(88,166,255,0.08)}
.ac-icon{font-size:13px;flex-shrink:0;width:18px;text-align:center;opacity:0.4}
.ac-text{
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  color:rgba(255,255,255,0.6);flex:1;min-width:0;
}
.ac-hl{color:rgba(88,166,255,0.95);font-weight:600}
.ac-hint{
  font-size:10px;color:rgba(255,255,255,0.2);
  background:rgba(255,255,255,0.04);
  padding:2px 6px;border-radius:4px;margin-left:auto;flex-shrink:0;
  font-family:'SF Mono','Fira Code',monospace;
}
.ac-footer{
  padding:5px 16px;font-size:10px;color:rgba(255,255,255,0.15);
  border-top:1px solid rgba(255,255,255,0.04);
  display:flex;gap:14px;background:rgba(255,255,255,0.01);
  position:sticky;bottom:0;
}
.ac-footer kbd{
  background:rgba(255,255,255,0.06);color:rgba(255,255,255,0.25);
  padding:1px 5px;border-radius:3px;font-size:9px;
  font-family:'SF Mono','Fira Code',monospace;
}
#welcome{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:100%;padding:40px 20px;text-align:center;max-width:820px;margin:0 auto;
}
#welcome h2{
  font-size:28px;font-weight:700;color:rgba(255,255,255,0.92);
  margin-bottom:8px;letter-spacing:-0.5px;
  animation:fadeUp 0.5s ease;
}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
#welcome p{color:rgba(255,255,255,0.4);font-size:14px;margin-bottom:36px;
  animation:fadeUp 0.5s ease 0.1s both}
#suggestions{animation:fadeUp 0.5s ease 0.2s both;
  display:grid;grid-template-columns:1fr 1fr;gap:12px;width:100%;max-width:760px;
}
.suggestion-chip{
  background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
  border-radius:12px;padding:14px 18px;cursor:pointer;text-align:left;
  font-size:13px;color:rgba(255,255,255,0.7);line-height:1.5;
  transition:all 0.2s ease;
}
.suggestion-chip:hover{
  background:rgba(255,255,255,0.055);border-color:rgba(255,255,255,0.14);
  color:rgba(255,255,255,0.9);transform:translateY(-2px);
  box-shadow:0 4px 24px rgba(0,0,0,0.35),0 0 0 1px rgba(255,255,255,0.05);
}
.suggestion-chip:active{transform:translateY(0);transition-duration:0.05s}
.chip-label{
  font-size:10px;font-weight:600;text-transform:uppercase;
  letter-spacing:0.04em;margin-bottom:5px;display:block;
}
.chip-label.recent{color:rgba(88,166,255,0.7)}
.chip-label.suggested{color:rgba(188,140,255,0.7)}
#sidebar{
  position:fixed;right:0;top:0;bottom:0;width:340px;
  background:rgba(12,12,14,0.95);backdrop-filter:blur(24px);
  border-left:1px solid rgba(255,255,255,0.06);
  transform:translateX(100%);transition:transform 0.3s cubic-bezier(0.4,0,0.2,1);
  z-index:200;overflow-y:auto;padding:24px;
}
#sidebar.open{transform:translateX(0)}
#sidebar-overlay{
  position:fixed;inset:0;background:rgba(0,0,0,0.4);
  z-index:199;opacity:0;pointer-events:none;transition:opacity 0.3s;
}
#sidebar-overlay.open{opacity:1;pointer-events:auto}
#history-btn,#proposals-btn{
  background:none;border:1px solid rgba(255,255,255,0.15);border-radius:8px;
  color:rgba(255,255,255,0.6);font-size:12px;cursor:pointer;
  padding:5px 12px;transition:all 0.15s;display:flex;align-items:center;gap:6px;
}
#history-btn:hover,#proposals-btn:hover{color:rgba(255,255,255,0.85);border-color:rgba(255,255,255,0.3)}
#history-btn svg,#proposals-btn svg{opacity:0.85}
#sidebar-close{
  position:absolute;top:16px;right:16px;background:none;border:none;
  color:rgba(255,255,255,0.3);font-size:20px;cursor:pointer;
  padding:4px 8px;border-radius:6px;transition:all 0.15s;
}
#sidebar-close:hover{color:rgba(255,255,255,0.7);background:rgba(255,255,255,0.05)}
.sidebar-section{margin-bottom:28px}
.sidebar-hdr{
  font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:0.06em;color:rgba(255,255,255,0.25);margin-bottom:12px;
}
.sidebar-item{
  padding:10px 14px;background:rgba(255,255,255,0.02);
  border:1px solid rgba(255,255,255,0.04);border-radius:10px;
  margin-bottom:6px;cursor:pointer;font-size:13px;
  color:rgba(255,255,255,0.5);transition:all 0.15s;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.sidebar-item:hover{
  border-color:rgba(255,255,255,0.1);background:rgba(255,255,255,0.04);
  color:rgba(255,255,255,0.8);
}
.sidebar-empty{color:rgba(255,255,255,0.2);font-size:13px;padding:8px 0}
#history-search{
  width:100%;background:rgba(255,255,255,0.04);
  border:1px solid rgba(255,255,255,0.08);border-radius:8px;
  color:rgba(255,255,255,0.8);font-size:12px;font-family:inherit;
  padding:8px 12px;outline:none;margin-bottom:12px;
  transition:border-color 0.2s;
}
#history-search:focus{border-color:rgba(88,166,255,0.4)}
#history-search::placeholder{color:rgba(255,255,255,0.25)}
.followup-bar{
  max-width:820px;margin:16px auto 8px;padding:12px 18px;
  background:rgba(188,140,255,0.04);
  border:1px solid rgba(188,140,255,0.15);border-radius:12px;
  cursor:pointer;transition:all 0.2s;display:flex;align-items:center;gap:10px;
}
.followup-bar:hover{
  background:rgba(188,140,255,0.08);
  border-color:rgba(188,140,255,0.3);
  transform:translateY(-1px);
  box-shadow:0 4px 20px rgba(188,140,255,0.08);
}
.fu-label{
  font-size:10px;font-weight:600;text-transform:uppercase;
  letter-spacing:0.04em;color:rgba(188,140,255,0.7);
  white-space:nowrap;flex-shrink:0;
}
.fu-text{
  font-size:13.5px;color:rgba(255,255,255,0.7);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.llm-panel{
  border:1px solid rgba(88,166,255,0.15);border-radius:10px;
  margin:8px auto;max-height:350px;overflow-y:auto;
  padding:12px 16px;background:rgba(88,166,255,0.03);
  max-width:820px;
}
.llm-panel .txt{font-size:10px;line-height:1.5;color:rgba(255,255,255,0.6)}
.llm-panel .think .cnt{font-size:10px}
.bash-panel{
  max-width:820px;margin-left:auto;margin-right:auto;
  background:rgba(0,0,0,0.5);color:rgba(255,255,255,0.55);
  border-color:rgba(255,255,255,0.05);
}
#split-container{display:flex;height:100vh;width:100vw;overflow:hidden}
#editor-panel{position:relative;overflow:hidden}
#editor-panel iframe{
  width:125%;height:125%;border:none;
  transform:scale(0.8);transform-origin:0 0;
}
#editor-fallback{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:100%;background:#1e1e1e;color:rgba(255,255,255,0.7);
  font-size:14px;text-align:center;padding:40px;gap:4px;
}
#editor-fallback h3{color:rgba(255,255,255,0.9);margin-bottom:12px;font-size:20px}
#editor-fallback code{
  background:rgba(255,255,255,0.08);padding:8px 16px;border-radius:8px;
  display:block;margin:8px 0;font-size:13px;color:rgba(255,255,255,0.8);
}
#editor-fallback p{margin:4px 0;color:rgba(255,255,255,0.5);font-size:13px}
#divider{
  width:6px;flex-shrink:0;cursor:col-resize;
  background:rgba(255,255,255,0.06);position:relative;z-index:10;
  transition:background 0.15s;
}
#divider:hover,#divider.active{background:rgba(88,166,255,0.5)}
#divider::after{
  content:'';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:2px;height:40px;background:rgba(255,255,255,0.2);border-radius:1px;
}
#assistant-panel{
  display:flex;flex-direction:column;overflow:hidden;min-width:300px;
  background:#0a0a0c;position:relative;
}
.tp[data-path]{cursor:pointer;text-decoration:underline dotted}
.tp[data-path]:hover{color:rgba(120,180,255,0.8);text-decoration:underline solid}
.logo span{max-width:150px}
#assistant-panel{font-size:11px}
#assistant-panel header{padding:8px 12px}
#assistant-panel .logo{font-size:11px}
#assistant-panel .logo span{display:none}
#assistant-panel .status{font-size:11px}
#assistant-panel #history-btn,#assistant-panel #proposals-btn{
  font-size:0;padding:5px;border-radius:6px;gap:0;
}
#assistant-panel #history-btn svg,#assistant-panel #proposals-btn svg{width:13px;height:13px}
#assistant-panel #welcome{padding:20px 14px}
#assistant-panel #welcome h2{font-size:17px;margin-bottom:3px;letter-spacing:-0.3px}
#assistant-panel #welcome p{font-size:11px;margin-bottom:14px}
#assistant-panel #suggestions{grid-template-columns:1fr;gap:5px;max-width:100%}
#assistant-panel .suggestion-chip{
  padding:7px 11px;border-radius:8px;font-size:11px;line-height:1.35;
}
#assistant-panel .chip-label{font-size:9px;margin-bottom:2px}
#assistant-panel #output{padding:14px 12px 12px}
#assistant-panel .ev,#assistant-panel .txt,#assistant-panel .spinner,
#assistant-panel .empty-msg,#assistant-panel .user-msg,
#assistant-panel .llm-panel,#assistant-panel .bash-panel,
#assistant-panel .followup-bar{max-width:none}
#assistant-panel .user-msg{
  font-size:11px;padding:10px 14px;margin:12px 0 10px;border-radius:10px;
}
#assistant-panel .txt{font-size:11px;padding:4px 10px}
#assistant-panel .tn{font-size:11px}
#assistant-panel .tp{font-size:11px}
#assistant-panel .tc{margin:8px 0;border-radius:8px}
#assistant-panel .tc-h{padding:7px 10px;border-radius:8px 8px 0 0}
#assistant-panel .tc-b{padding:6px 10px;max-height:200px;font-size:11px}
#assistant-panel .tr{padding:5px 10px;max-height:150px;font-size:11px}
#assistant-panel .think{padding:8px 12px;margin:8px 0;border-radius:8px}
#assistant-panel .think .cnt{font-size:10px}
#assistant-panel .rc{border-radius:10px}
#assistant-panel .rc-h{padding:10px 14px}
#assistant-panel .rc-body{padding:10px 14px;max-height:250px;font-size:11px}
#assistant-panel #input-area{padding:0 12px 12px;padding-top:10px}
#assistant-panel #input-container{padding:8px 10px;border-radius:10px}
#assistant-panel #input-wrap{gap:4px}
#assistant-panel #task-input,#assistant-panel #ghost-overlay{font-size:11px}
#assistant-panel #input-footer{margin-top:5px;padding-top:5px}
#assistant-panel #model-btn{font-size:11px;padding:4px 8px;border-radius:6px}
#assistant-panel #model-search{font-size:11px;padding:7px 10px}
#assistant-panel .model-item{font-size:11px;padding:5px 10px}
#assistant-panel .model-cost{font-size:9px}
#assistant-panel .model-group-hdr{font-size:9px;padding:4px 10px 3px}
#assistant-panel #send-btn{width:28px;height:28px}
#assistant-panel #send-btn svg{width:12px;height:12px}
#assistant-panel #stop-btn{width:28px;height:28px}
#assistant-panel #stop-btn svg{width:11px;height:11px}
#assistant-panel #clear-btn{width:18px;height:18px}
#assistant-panel #clear-btn svg{width:11px;height:11px}
#assistant-panel #history-search{font-size:11px;padding:6px 10px}
#assistant-panel .sidebar-hdr{font-size:10px}
#assistant-panel .sidebar-item{font-size:11px;padding:7px 10px;border-radius:8px;margin-bottom:4px}
#assistant-panel .sidebar-empty{font-size:11px}
#assistant-panel #sidebar-close{font-size:17px}
#assistant-panel .ac-item{font-size:11px;padding:6px 12px}
#assistant-panel .ac-icon{font-size:11px}
#assistant-panel .ac-section{font-size:9px;padding:4px 12px 3px}
#assistant-panel .ac-hint{font-size:9px}
#assistant-panel .ac-footer{font-size:9px;padding:4px 12px}
#assistant-panel .ac-footer kbd{font-size:9px}
#assistant-panel .fu-label{font-size:9px}
#assistant-panel .fu-text{font-size:11px}
#assistant-panel .followup-bar{padding:8px 12px;margin:10px 0 6px;border-radius:8px}
#assistant-panel .llm-panel{padding:8px 10px;margin:6px 0;border-radius:8px}
#assistant-panel .llm-panel .txt{font-size:10px}
#assistant-panel .llm-panel .think .cnt{font-size:10px}
#assistant-panel .bash-panel{max-height:200px;font-size:10px}
#assistant-panel .prompt-h{font-size:11px;padding:6px 12px}
#assistant-panel .prompt-body{font-size:11px;padding:8px 12px}
#assistant-panel .rc-h{
  padding:10px 14px;flex-direction:column;align-items:flex-start;gap:6px;
}
#assistant-panel .rc-h h3{font-size:11px;margin-bottom:2px}
#assistant-panel .rs{
  font-size:10px;gap:0;width:100%;
  display:grid;grid-template-columns:repeat(3,1fr);
}
#assistant-panel .rs b{display:block;font-size:11px}
#assistant-panel .td{font-size:11px}
#assistant-panel .sys{font-size:11px}
#assistant-panel .spinner{font-size:11px}
#assistant-panel .empty-msg{font-size:11px}
#assistant-panel .rl{font-size:10px}
#assistant-panel .usage{font-size:11px}
"""

CHATBOT_JS = r"""
var O=document.getElementById('output');
var D=document.getElementById('dot');
var ST=document.getElementById('stxt');
var inp=document.getElementById('task-input');
var btn=document.getElementById('send-btn');
var stopBtn=document.getElementById('stop-btn');
var clearBtn=document.getElementById('clear-btn');
var ac=document.getElementById('autocomplete');
var rl=document.getElementById('recent-list');
var pl=document.getElementById('proposed-list');
var histSearch=document.getElementById('history-search');
var allTasks=[];
var modelBtn=document.getElementById('model-btn');
var modelLabel=document.getElementById('model-label');
var modelDD=document.getElementById('model-dropdown');
var modelSearch=document.getElementById('model-search');
var modelList=document.getElementById('model-list');
var allModels=[],selectedModel='',modelDDIdx=-1;
var sidebar=document.getElementById('sidebar');
var sidebarOverlay=document.getElementById('sidebar-overlay');
var suggestionsEl=document.getElementById('suggestions');
var running=false,_scrollLock=false;
var scrollRaf=0,state={thinkEl:null,txtEl:null,bashPanel:null};
var acIdx=-1,t0=null,timerIv=null,evtSrc=null;
var acTimer=null,histIdx=-1,histCache=[];
var lastToolName='',llmPanel=null,pendingPanel=false;
var llmPanelState={thinkEl:null,txtEl:null,bashPanel:null};
var ghostEl=document.getElementById('ghost-overlay');
var ghostSuggest='',ghostTimer2=null,ghostAbort=null;
var ghostCache={q:'',s:''};
inp.addEventListener('input',function(){
  this.style.height='auto';
  this.style.height=Math.min(this.scrollHeight,200)+'px';
  histIdx=-1;
  clearGhost();
  if(getAtCtx()){
    if(acTimer)clearTimeout(acTimer);
    acTimer=setTimeout(fetchAC,150);
  } else {
    hideAC();
    if(ghostTimer2)clearTimeout(ghostTimer2);
    ghostTimer2=setTimeout(fetchGhost,500);
  }
});
var sidebarHistSec=document.getElementById('sidebar-history-sec');
var sidebarPropSec=document.getElementById('sidebar-proposals-sec');
function toggleSidebar(mode){
  if(mode){
    sidebarHistSec.style.display=mode==='proposals'?'none':'';
    sidebarPropSec.style.display=mode==='history'?'none':'';
    if(sidebar.classList.contains('open'))return;
  }
  sidebar.classList.toggle('open');
  sidebarOverlay.classList.toggle('open');
}
O.addEventListener('wheel',function(e){
  if(running&&e.deltaY<0)_scrollLock=true;
});
O.addEventListener('scroll',function(){
  if(_scrollLock){
    var atBottom=O.scrollTop+O.clientHeight>=O.scrollHeight-150;
    if(atBottom)_scrollLock=false;
  }
});
function sb(){
  if(!_scrollLock&&!scrollRaf){scrollRaf=requestAnimationFrame(function(){
    O.scrollTo({top:O.scrollHeight,behavior:'instant'});scrollRaf=0;
  });}
}
new MutationObserver(function(){sb()}).observe(O,{childList:true,subtree:true,characterData:true});
function startTimer(){
  t0=Date.now();
  if(timerIv)clearInterval(timerIv);
  timerIv=setInterval(function(){
    var s=Math.floor((Date.now()-t0)/1000);
    var m=Math.floor(s/60);
    ST.textContent='Running '+(m>0?m+'m ':'')+s%60+'s';
  },1000);
}
function stopTimer(){if(timerIv){clearInterval(timerIv);timerIv=null;}}
function removeSpinner(){
  var sp=document.getElementById('wait-spinner');
  if(sp)sp.remove();
}
function showSpinner(msg){
  removeSpinner();
  var sp=mkEl('div','spinner');
  sp.id='wait-spinner';
  sp.textContent=msg||'Waiting ...';
  O.appendChild(sp);sb();
}
function setReady(label){
  running=false;D.classList.remove('running');
  stopTimer();removeSpinner();
  ST.textContent=label||'Ready';
  inp.disabled=false;
  btn.style.display='';
  stopBtn.style.display='none';
  inp.focus();
}
function connectSSE(){
  if(evtSrc)evtSrc.close();
  evtSrc=new EventSource('/events');
  evtSrc.onopen=function(){};
  evtSrc.onmessage=function(e){
    var ev;try{ev=JSON.parse(e.data);}catch(x){return;}
    try{handleEvent(ev);}catch(err){console.error('Event error:',err,ev);}
  };
  evtSrc.onerror=function(){};
}
function handleEvent(ev){
  var t=ev.type;
  if(t==='thinking_start'||t==='thinking_delta'||t==='text_delta'
    ||t==='tool_call'||t==='tool_result'||t==='system_output'
    ||t==='task_done'||t==='task_error'||t==='task_stopped')removeSpinner();
  switch(t){
  case'tasks_updated':loadTasks();loadWelcome();break;
  case'proposed_updated':loadProposed();loadWelcome();break;
  case'clear':
    O.innerHTML='';state.thinkEl=null;state.txtEl=null;state.bashPanel=null;
    llmPanel=null;llmPanelState={thinkEl:null,txtEl:null,bashPanel:null};lastToolName='';pendingPanel=false;
    _scrollLock=false;showSpinner();break;
  case'task_done':{
    var el=t0?Math.floor((Date.now()-t0)/1000):0;
    var em=Math.floor(el/60);
    setReady('Done ('+(em>0?em+'m ':'')+el%60+'s)');
    loadTasks();break}
  case'followup_suggestion':{
    var fu=mkEl('div','followup-bar');
    fu.title=ev.text;
    fu.innerHTML='<span class="fu-label">Suggested next</span>'
      +'<span class="fu-text">'+esc(ev.text)+'</span>';
    fu.addEventListener('click',function(){
      inp.value=ev.text;inp.focus();
    });
    O.appendChild(fu);sb();break}
  case'task_error':{
    var err=mkEl('div','ev tr err');
    err.innerHTML='<div class="rl fail">ERROR</div>'+esc(ev.text||'Unknown error');
    O.appendChild(err);
    setReady('Error');loadTasks();loadProposed();break}
  case'task_stopped':{
    var stEl=mkEl('div','ev tr err');
    stEl.innerHTML='<div class="rl fail">STOPPED</div>Agent execution stopped by user';
    O.appendChild(stEl);
    setReady('Stopped');loadTasks();loadProposed();break}
  default:{
    if(t==='tool_call'){
      lastToolName=ev.name||'';
      llmPanel=null;llmPanelState={thinkEl:null,txtEl:null,bashPanel:null};pendingPanel=false;
    }
    if(t==='tool_result'&&lastToolName!=='finish'){pendingPanel=true;}
    if(pendingPanel&&(t==='thinking_start'||t==='text_delta')){
      llmPanel=mkEl('div','llm-panel');
      O.appendChild(llmPanel);
      llmPanelState={thinkEl:null,txtEl:null,bashPanel:null};pendingPanel=false;
    }
    var target=O,tState=state;
    if(llmPanel&&(t==='thinking_start'||t==='thinking_delta'||t==='thinking_end'
      ||t==='text_delta'||t==='text_end')){
      target=llmPanel;tState=llmPanelState;
    }
    handleOutputEvent(ev,target,tState);
    if(target===llmPanel)llmPanel.scrollTop=llmPanel.scrollHeight;
    if(running&&(t==='tool_call'||t==='tool_result'||t==='thinking_end'
      ||t==='text_end'||(t==='system_output'&&!state.bashPanel)))showSpinner();
  }}
  sb();
}
function loadModels(){
  fetch('/models').then(function(r){return r.json();})
  .then(function(d){
    allModels=d.models;
    selectedModel=d.selected;
    modelLabel.textContent=selectedModel;
    renderModelList('');
  }).catch(function(){});
}
function modelVendor(name){
  if(name.startsWith('claude-'))return'Anthropic';
  if(/^(gpt|o[134]|codex|computer-use)/.test(name)&&!name.startsWith('openai/'))return'OpenAI';
  if(name.startsWith('gemini-'))return'Gemini';
  if(name.startsWith('minimax-'))return'MiniMax';
  if(name.startsWith('openrouter/'))return'OpenRouter';
  return'Together AI';
}
function renderModelItem(m){
  var d=mkEl('div','model-item'+(m.name===selectedModel?' active':''));
  var price='$'+m.inp.toFixed(2)+' / $'+m.out.toFixed(2);
  d.innerHTML='<span>'+esc(m.name)+'</span><span class="model-cost">'+price+'</span>';
  d.addEventListener('click',function(){selectModel(m.name)});
  return d;
}
function renderModelList(q){
  modelList.innerHTML='';modelDDIdx=-1;
  var ql=q.toLowerCase();
  var used=[],rest=[];
  allModels.forEach(function(m){
    if(ql&&m.name.toLowerCase().indexOf(ql)<0)return;
    if(m.uses>0)used.push(m);else rest.push(m);
  });
  used.sort(function(a,b){return b.uses-a.uses});
  if(used.length){
    var hdr=mkEl('div','model-group-hdr');
    hdr.textContent='Recently Used';
    modelList.appendChild(hdr);
    used.forEach(function(m){modelList.appendChild(renderModelItem(m))});
  }
  var lastVendor='';
  rest.forEach(function(m){
    var v=modelVendor(m.name);
    if(v!==lastVendor){
      var hdr=mkEl('div','model-group-hdr');
      hdr.textContent=v;
      modelList.appendChild(hdr);
      lastVendor=v;
    }
    modelList.appendChild(renderModelItem(m));
  });
}
function selectModel(name){
  selectedModel=name;
  modelLabel.textContent=name;
  closeModelDD();
  renderModelList('');
}
function toggleModelDD(){
  if(modelDD.classList.contains('open')){closeModelDD();return}
  modelDD.classList.add('open');
  modelSearch.value='';
  renderModelList('');
  modelSearch.focus();
}
function closeModelDD(){
  modelDD.classList.remove('open');
  modelSearch.value='';
  modelDDIdx=-1;
}
modelSearch.addEventListener('input',function(){renderModelList(this.value)});
modelSearch.addEventListener('keydown',function(e){
  var items=modelList.querySelectorAll('.model-item');
  if(e.key==='ArrowDown'){e.preventDefault();modelDDIdx=Math.min(modelDDIdx+1,items.length-1);updateModelSel(items);return}
  if(e.key==='ArrowUp'){e.preventDefault();modelDDIdx=Math.max(modelDDIdx-1,-1);updateModelSel(items);return}
  if(e.key==='Enter'){e.preventDefault();var ti=modelDDIdx>=0?modelDDIdx:0;
  if(items[ti])items[ti].click();return}
  if(e.key==='Escape'){e.preventDefault();closeModelDD();return}
});
function updateModelSel(items){
  items.forEach(function(it,i){it.classList.toggle('sel',i===modelDDIdx)});
  if(modelDDIdx>=0)items[modelDDIdx].scrollIntoView({block:'nearest'});
}
document.addEventListener('click',function(e){
  if(!document.getElementById('model-picker').contains(e.target))closeModelDD();
  if(!ac.contains(e.target)&&e.target!==inp)hideAC();
});
function submitTask(){
  var task=inp.value.trim();
  if(!task||running)return;
  var fileMatch=task.match(/^@(\S+)$/);
  if(fileMatch){openInEditor(fileMatch[1]);inp.value='';return}
  running=true;inp.disabled=true;
  btn.style.display='none';
  stopBtn.style.display='inline-flex';
  D.classList.add('running');hideAC();startTimer();
  inp.style.height='auto';
  fetch('/run',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task:task,model:selectedModel})
  }).then(function(r){
    if(!r.ok){r.json().then(function(d){setReady('Error');alert(d.error||'Failed')});return;}
    inp.value='';loadModels();
  }).catch(function(){setReady('Error');alert('Network error')});
}
btn.addEventListener('click',submitTask);
stopBtn.addEventListener('click',function(){fetch('/stop',{method:'POST'}).catch(function(){})});
clearBtn.addEventListener('click',function(){
  if(running)return;
  O.innerHTML='<div id="welcome"><h2>What can I help you with?</h2>'
    +'<p>Describe a task and the agent will work on it</p>'
    +'<div id="suggestions"></div></div>';
  suggestionsEl=document.getElementById('suggestions');
  state={thinkEl:null,txtEl:null,bashPanel:null};
  llmPanel=null;llmPanelState={thinkEl:null,txtEl:null,bashPanel:null};
  lastToolName='';pendingPanel=false;_scrollLock=false;
  loadWelcome();inp.value='';inp.focus();
});
inp.addEventListener('keydown',function(e){
  if(ac.style.display==='block'){
    var items=ac.querySelectorAll('.ac-item');
    if(e.key==='ArrowDown'){e.preventDefault();acIdx=Math.min(acIdx+1,items.length-1);updateACSel(items);return}
    if(e.key==='ArrowUp'){e.preventDefault();acIdx=Math.max(acIdx-1,-1);updateACSel(items);return}
    if(e.key==='Tab'){e.preventDefault();var ti=acIdx>=0?acIdx:0;
      if(items[ti])items[ti].click();return}
    if(e.key==='Enter'&&acIdx>=0){e.preventDefault();items[acIdx].click();return}
    if(e.key==='Escape'){hideAC();return}
  }
  if(ghostSuggest){
    if(e.key==='Tab'){e.preventDefault();acceptGhost();return}
    if(e.key==='ArrowRight'&&inp.selectionStart===inp.value.length){e.preventDefault();acceptGhost();return}
    if(e.key==='Escape'){clearGhost();return}
  }
  if(e.key==='ArrowUp'&&ac.style.display!=='block'&&(!inp.value.trim()||histIdx>=0)){
    e.preventDefault();cycleHistory(1);return;
  }
  if(e.key==='ArrowDown'&&histIdx>=0&&ac.style.display!=='block'){
    e.preventDefault();cycleHistory(-1);return;
  }
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submitTask()}
});
function getAtCtx(){
  var val=inp.value,pos=inp.selectionStart||0;
  var before=val.substring(0,pos);
  var m=before.match(/@([^\s]*)$/);
  return m?{start:before.length-m[0].length,query:m[1]}:null;
}
function fetchAC(){
  var atCtx=getAtCtx();
  if(!atCtx){hideAC();return}
  fetch('/suggestions?mode=files&q='+encodeURIComponent(atCtx.query))
    .then(function(r){return r.json()}).then(renderAC).catch(function(){hideAC()});
}
function hlMatch(text,query){
  if(!query)return esc(text);
  var idx=text.toLowerCase().indexOf(query.toLowerCase());
  if(idx<0)return esc(text);
  return esc(text.substring(0,idx))
    +'<strong class="ac-hl">'+esc(text.substring(idx,idx+query.length))+'</strong>'
    +esc(text.substring(idx+query.length));
}
function renderAC(data){
  if(!data.length){hideAC();return}
  ac.innerHTML='';acIdx=-1;
  var atCtx=getAtCtx();
  var searchQ=atCtx?atCtx.query:'';
  var groups={},order=['dir','file'];
  var labels={dir:'Directories',file:'Files'};
  var icons={dir:'\uD83D\uDCC1',file:'\uD83D\uDCCB'};
  data.forEach(function(item){
    if(!groups[item.type])groups[item.type]=[];
    groups[item.type].push(item);
  });
  var isFirst=true;
  order.forEach(function(type){
    var g=groups[type];if(!g)return;
    var hdr=mkEl('div','ac-section');
    hdr.textContent=labels[type]||type;
    ac.appendChild(hdr);
    g.forEach(function(item){
      var d=mkEl('div','ac-item');
      d.innerHTML='<span class="ac-icon">'+(icons[item.type]||'')+'</span>'
        +'<span class="ac-text">'+hlMatch(item.text,searchQ)+'</span>';
      if(isFirst){d.innerHTML+='<span class="ac-hint">Tab</span>';isFirst=false}
      d.addEventListener('click',function(){selectAC(item)});
      ac.appendChild(d);
    });
  });
  var footer=mkEl('div','ac-footer');
  footer.innerHTML='<span><kbd>\u2191\u2193</kbd> navigate</span>'
    +'<span><kbd>Tab</kbd> accept</span>'
    +'<span><kbd>Esc</kbd> dismiss</span>';
  ac.appendChild(footer);
  ac.style.display='block';
}
function selectAC(item){
  var atCtx=getAtCtx();
  if(atCtx){
    var before=inp.value.substring(0,atCtx.start);
    var after=inp.value.substring(inp.selectionStart||inp.value.length);
    inp.value=before+'@'+item.text+' '+after;
    var np=before.length+1+item.text.length+1;
    inp.setSelectionRange(np,np);
  }
  hideAC();inp.focus();
}
function hideAC(){ac.style.display='none';acIdx=-1}
function updateACSel(items){
  items.forEach(function(it,i){it.classList.toggle('sel',i===acIdx)});
  if(acIdx>=0)items[acIdx].scrollIntoView({block:'nearest'});
}
function clearGhost(){ghostSuggest='';ghostEl.innerHTML=''}
function updateGhost(){
  if(!ghostSuggest){ghostEl.innerHTML='';return}
  ghostEl.innerHTML='<span class="gm">'+esc(inp.value)+'</span>'
    +'<span class="gs">'+esc(ghostSuggest)+'</span>';
}
function acceptGhost(){
  inp.value+=ghostSuggest;
  clearGhost();inp.focus();
}
function fetchGhost(){
  var q=inp.value;
  if(!q.trim()||q.trim().length<3){clearGhost();return}
  if(ghostCache.q&&q.startsWith(ghostCache.q)&&ghostCache.s){
    var extra=q.substring(ghostCache.q.length);
    if(ghostCache.s.startsWith(extra)){
      ghostSuggest=ghostCache.s.substring(extra.length);
      if(ghostSuggest){updateGhost();return}
    }
  }
  if(ghostAbort)ghostAbort.abort();
  ghostAbort=new AbortController();
  fetch('/complete?q='+encodeURIComponent(q),{signal:ghostAbort.signal})
    .then(function(r){return r.json()})
    .then(function(d){
      if(d.suggestion&&inp.value===q){
        ghostSuggest=d.suggestion;
        ghostCache={q:q,s:d.suggestion};
        updateGhost();
      }
    }).catch(function(){});
}
function cycleHistory(dir){
  if(!histCache.length){
    fetch('/tasks').then(function(r){return r.json()}).then(function(tasks){
      histCache=tasks.map(function(t){return typeof t==='string'?t:(t.task||'')});
      doHistCycle(dir);
    });return;
  }
  doHistCycle(dir);
}
function doHistCycle(dir){
  histIdx+=dir;
  if(histIdx<0){histIdx=-1;inp.value='';return}
  if(histIdx>=histCache.length){histIdx=histCache.length-1;return}
  inp.value=histCache[histIdx];
}
function loadTasks(){
  fetch('/tasks').then(function(r){return r.json()}).then(function(tasks){
    allTasks=tasks;renderTasks('');histSearch.value='';
  }).catch(function(){});
}
function renderTasks(q){
  rl.innerHTML='';
  var ql=q.toLowerCase(),filtered=[];
  allTasks.forEach(function(t){
    var txt=typeof t==='string'?t:(t.task||'');
    if(!ql||txt.toLowerCase().indexOf(ql)>=0)filtered.push(txt);
  });
  if(!filtered.length){rl.innerHTML='<div class="sidebar-empty">'
    +(ql?'No matches':'No recent tasks')+'</div>';return}
  filtered.forEach(function(taskText){
    var d=mkEl('div','sidebar-item');
    d.textContent=taskText;d.title=taskText;
    d.addEventListener('click',function(){inp.value=taskText;inp.focus();toggleSidebar()});
    rl.appendChild(d);
  });
}
histSearch.addEventListener('input',function(){renderTasks(this.value)});
function loadProposed(){
  fetch('/proposed_tasks').then(function(r){return r.json()}).then(function(tasks){
    pl.innerHTML='';
    if(!tasks.length){pl.innerHTML='<div class="sidebar-empty">No suggestions yet</div>';return}
    tasks.forEach(function(t){
      var d=mkEl('div','sidebar-item');
      d.textContent=t;d.title=t;
      d.addEventListener('click',function(){inp.value=t;inp.focus();toggleSidebar()});
      pl.appendChild(d);
    });
  }).catch(function(){});
}
function loadWelcome(){
  if(!suggestionsEl)return;
  Promise.all([
    fetch('/tasks').then(function(r){return r.json()}).catch(function(){return []}),
    fetch('/proposed_tasks').then(function(r){return r.json()}).catch(function(){return []})
  ]).then(function(res){
    var tasks=res[0],proposed=res[1];
    suggestionsEl.innerHTML='';
    var items=[];
    proposed.slice(0,3).forEach(function(t){items.push({text:t,type:'suggested'})});
    tasks.slice(0,3).forEach(function(t){
      items.push({text:typeof t==='string'?t:(t.task||''),type:'recent'});
    });
    items.slice(0,6).forEach(function(item){
      var chip=mkEl('div','suggestion-chip');
      chip.title=item.text;
      chip.innerHTML='<span class="chip-label '+item.type+'">'
        +(item.type==='recent'?'Recent':'Suggested')+'</span>'
        +esc(item.text);
      chip.addEventListener('click',function(){inp.value=item.text;inp.focus()});
      suggestionsEl.appendChild(chip);
    });
  });
}
var divider=document.getElementById('divider');
var editorPanel=document.getElementById('editor-panel');
var assistantPanel=document.getElementById('assistant-panel');
var splitContainer=document.getElementById('split-container');
var isDragging=false;
if(divider){
  divider.addEventListener('mousedown',function(e){
    isDragging=true;divider.classList.add('active');
    document.body.style.cursor='col-resize';
    document.body.style.userSelect='none';
    var frame=document.getElementById('code-server-frame');
    if(frame)frame.style.pointerEvents='none';
    e.preventDefault();
  });
  document.addEventListener('mousemove',function(e){
    if(!isDragging)return;
    var rect=splitContainer.getBoundingClientRect();
    var x=e.clientX-rect.left;
    var pct=Math.max(15,Math.min(85,(x/rect.width)*100));
    editorPanel.style.width=pct+'%';
    editorPanel.style.flex='none';
    assistantPanel.style.flex='1';
  });
  document.addEventListener('mouseup',function(){
    if(!isDragging)return;
    isDragging=false;divider.classList.remove('active');
    document.body.style.cursor='';
    document.body.style.userSelect='';
    var frame=document.getElementById('code-server-frame');
    if(frame)frame.style.pointerEvents='';
  });
}
function openInEditor(path){
  fetch('/open-file',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:path})}).catch(function(){});
}
document.addEventListener('click',function(e){
  var el=e.target.closest('[data-path]');
  if(el&&el.dataset.path){openInEditor(el.dataset.path);}
});
connectSSE();loadModels();loadTasks();loadProposed();loadWelcome();inp.focus();
"""


def _build_html(title: str, subtitle: str, code_server_url: str = "", work_dir: str = "") -> str:
    font_import = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');\n"
    css = font_import + BASE_CSS + OUTPUT_CSS + CHATBOT_CSS

    if code_server_url:
        import urllib.parse
        wd_enc = urllib.parse.quote(work_dir, safe="")
        editor_content = (
            f'<iframe id="code-server-frame"'
            f' src="{code_server_url}/?folder={wd_enc}"'
            f' data-base-url="{code_server_url}"'
            f' data-work-dir="{work_dir}"></iframe>'
        )
    else:
        editor_content = (
            '<div id="editor-fallback">'
            '<h3>VS Code Editor</h3>'
            '<p>code-server is not installed. Install it to enable the embedded editor:</p>'
            '<code>curl -fsSL https://code-server.dev/install.sh | sh</code>'
            '<p>Or via Homebrew:</p>'
            '<code>brew install code-server</code>'
            '<p style="margin-top:16px;font-size:12px;opacity:0.5">'
            'Restart the assistant after installation.</p>'
            '</div>'
        )

    return HTML_HEAD.format(title=title, css=css) + f"""<body>
<div id="split-container">
  <div id="editor-panel" style="width:80%;flex-shrink:0">
    {editor_content}
  </div>
  <div id="divider"></div>
  <div id="assistant-panel">
    <div id="sidebar-overlay" onclick="toggleSidebar()"></div>
    <div id="sidebar">
      <button id="sidebar-close" onclick="toggleSidebar()">&times;</button>
      <div class="sidebar-section" id="sidebar-history-sec">
        <div class="sidebar-hdr">Recent Tasks</div>
        <input type="text" id="history-search" placeholder="Search history\u2026"
          autocomplete="off"/>
        <div id="recent-list"></div>
      </div>
      <div class="sidebar-section" id="sidebar-proposals-sec">
        <div class="sidebar-hdr">Suggested Tasks</div>
        <div id="proposed-list"></div>
      </div>
    </div>
    <header>
      <div class="logo">{title}<span>{subtitle}</span></div>
      <div style="display:flex;align-items:center;gap:10px;flex-shrink:0">
        <div class="status"><div class="dot" id="dot"></div><span id="stxt">Ready</span></div>
        <button id="history-btn" onclick="toggleSidebar('history')" title="Task history">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          History
        </button>
        <button id="proposals-btn" onclick="toggleSidebar('proposals')" title="Suggested tasks">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/>
            <path d="M2 12l10 5 10-5"/>
          </svg>
          Proposals
        </button>
      </div>
    </header>
    <div id="output">
      <div id="welcome">
        <h2>What can I help you with?</h2>
        <p>Describe a task and the agent will work on it</p>
        <div id="suggestions"></div>
      </div>
    </div>
    <div id="input-area">
      <div id="autocomplete"></div>
      <div id="input-container">
        <div id="input-wrap">
          <div id="ghost-overlay"></div>
          <textarea id="task-input" placeholder="Ask anything\u2026" rows="1"
            autocomplete="off"></textarea>
          <button id="clear-btn" title="Clear chat"><svg viewBox="0 0 24 24" fill="none"
            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
            ><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6"
              x2="18" y2="18"/></svg></button>
        </div>
        <div id="input-footer">
          <div id="model-picker">
            <button type="button" id="model-btn" onclick="toggleModelDD()">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2"><path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z"/></svg>
              <span id="model-label">Loading\u2026</span>
            </button>
            <div id="model-dropdown">
              <input type="text" id="model-search"
                placeholder="Search models\u2026" autocomplete="off"/>
              <div id="model-list"></div>
            </div>
          </div>
          <div id="input-actions">
            <button id="send-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
              ><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"
              /></svg></button>
            <button id="stop-btn"><svg viewBox="0 0 24 24" fill="currentColor"
              ><rect x="6" y="6" width="12" height="12" rx="2"/></svg></button>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
{EVENT_HANDLER_JS}
{CHATBOT_JS}
</script>
</body>
</html>"""


def run_chatbot(
    agent_factory: Callable[[str], RelentlessAgent],
    title: str = "KISS Assistant",
    subtitle: str = "Interactive Agent",
    work_dir: str | None = None,
    default_model: str = "claude-opus-4-6",
    agent_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run a browser-based chatbot UI for any RelentlessAgent-based agent.

    Args:
        agent_factory: Callable that takes a name string and returns a RelentlessAgent instance.
        title: Title displayed in the browser UI header.
        subtitle: Subtitle displayed in the browser UI header.
        work_dir: Working directory for the agent. Defaults to current directory.
        default_model: Default LLM model name for the model selector.
        agent_kwargs: Additional keyword arguments passed to agent.run().
    """
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
    from starlette.routing import Route

    printer = BaseBrowserPrinter()
    running = False
    running_lock = threading.Lock()
    shutting_down = threading.Event()
    actual_work_dir = work_dir or os.getcwd()
    file_cache: list[str] = _scan_files(actual_work_dir)
    agent_thread: threading.Thread | None = None
    proposed_tasks: list[str] = _load_proposals()
    proposed_lock = threading.Lock()
    selected_model = (
        _load_last_model() or default_model or get_most_expensive_model() or "claude-opus-4-6"
    )

    cs_proc: subprocess.Popen[bytes] | None = None
    code_server_url = ""
    cs_data_dir = str(_KISS_DIR / "code-server-data")
    cs_binary = shutil.which("code-server")
    if cs_binary:
        _setup_code_server(cs_data_dir)
        cs_port = 13338
        port_in_use = False
        try:
            with socket.create_connection(("127.0.0.1", cs_port), timeout=0.5):
                port_in_use = True
        except (ConnectionRefusedError, OSError):
            pass
        if port_in_use:
            code_server_url = f"http://127.0.0.1:{cs_port}"
            print(f"Reusing existing code-server at {code_server_url}")
        else:
            cs_proc = subprocess.Popen(
                [
                    cs_binary, "--port", str(cs_port), "--auth", "none",
                    "--bind-addr", f"127.0.0.1:{cs_port}", "--disable-telemetry",
                    "--user-data-dir", cs_data_dir,
                    "--extensions-dir", str(Path(cs_data_dir) / "extensions"),
                    "--disable-getting-started-override",
                    "--disable-workspace-trust",
                    actual_work_dir,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for _ in range(30):
                try:
                    with socket.create_connection(("127.0.0.1", cs_port), timeout=0.5):
                        code_server_url = f"http://127.0.0.1:{cs_port}"
                        break
                except (ConnectionRefusedError, OSError):
                    time.sleep(0.5)
            if code_server_url:
                print(f"code-server running at {code_server_url}")
            else:
                print("Warning: code-server failed to start")

    html_page = _build_html(title, subtitle, code_server_url, actual_work_dir)
    shutdown_timer: threading.Timer | None = None

    def refresh_file_cache() -> None:
        nonlocal file_cache
        file_cache = _scan_files(actual_work_dir)

    def refresh_proposed_tasks() -> None:
        nonlocal proposed_tasks
        history = _load_history()
        if not history:
            with proposed_lock:
                proposed_tasks = []
            printer.broadcast({"type": "proposed_updated"})
            return
        task_list = "\n".join(f"- {e['task']}" for e in history[:20])
        agent = KISSAgent("Task Proposer")
        try:
            result = agent.run(
                model_name="gemini-2.0-flash",
                prompt_template=(
                    "Based on these past tasks a developer has worked on, suggest 5 new "
                    "tasks they might want to do next. Tasks should be natural follow-ups, "
                    "related improvements, or complementary work.\n\n"
                    "Past tasks:\n{task_list}\n\n"
                    "Return ONLY a JSON array of 5 short task description strings. "
                    'Example: ["Add unit tests for X", "Refactor Y module"]'
                ),
                arguments={"task_list": task_list},
                is_agentic=False,
            )
            start = result.index("[")
            end = result.index("]", start) + 1
            proposals = json.loads(result[start:end])
            proposals = [str(p) for p in proposals if isinstance(p, str) and p.strip()][:5]
        except Exception:
            proposals = []
        with proposed_lock:
            proposed_tasks = proposals
        _save_proposals(proposals)
        printer.broadcast({"type": "proposed_updated"})

    def generate_followup(task: str, result: str) -> None:
        try:
            agent = KISSAgent("Followup Proposer")
            raw = agent.run(
                model_name="gemini-2.0-flash",
                prompt_template=(
                    "A developer just completed this task:\n"
                    "Task: {task}\n"
                    "Result summary: {result}\n\n"
                    "Suggest ONE short, concrete follow-up task they "
                    "might want to do next. Return ONLY the task "
                    "description as a single plain-text sentence."
                ),
                arguments={
                    "task": task,
                    "result": result[:500],
                },
                is_agentic=False,
            )
            suggestion = raw.strip().strip('"').strip("'")
            if suggestion:
                printer.broadcast({
                    "type": "followup_suggestion",
                    "text": suggestion,
                })
        except Exception:
            pass

    def run_agent_thread(task: str, model_name: str) -> None:
        nonlocal running, agent_thread
        pre_hunks: dict[str, list[tuple[int, int, int, int]]] = {}
        pre_untracked: set[str] = set()
        try:
            _add_task(task)
            printer.broadcast({"type": "tasks_updated"})
            printer.broadcast({"type": "clear"})
            pre_hunks = _parse_diff_hunks(actual_work_dir)
            pre_untracked = _capture_untracked(actual_work_dir)
            agent = agent_factory("Chatbot")
            result = agent.run(
                prompt_template=task,
                work_dir=actual_work_dir,
                printer=printer,
                model_name=model_name,
                **(agent_kwargs or {}),
            )
            _set_latest_result(result or "")
            printer.broadcast({"type": "task_done"})
            threading.Thread(
                target=generate_followup,
                args=(task, result or ""),
                daemon=True,
            ).start()
        except _StopRequested:
            _set_latest_result("(stopped)")
            printer.broadcast({"type": "task_stopped"})
        except Exception as e:
            _set_latest_result(f"(error: {e})")
            printer.broadcast({"type": "task_error", "text": str(e)})
        finally:
            with running_lock:
                running = False
                agent_thread = None
            try:
                _prepare_merge_view(
                    actual_work_dir, cs_data_dir, pre_hunks, pre_untracked,
                )
            except Exception:
                pass
            refresh_file_cache()
            try:
                refresh_proposed_tasks()
            except Exception:
                pass

    def stop_agent() -> bool:
        nonlocal agent_thread
        with running_lock:
            thread = agent_thread
        if thread is None or not thread.is_alive():
            return False
        import ctypes

        tid = thread.ident
        if tid is None:
            return False
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid),
            ctypes.py_object(_StopRequested),
        )
        return True

    def _cleanup() -> None:
        stop_agent()
        if cs_proc:
            cs_proc.terminate()
            try:
                cs_proc.wait()
            except Exception:
                cs_proc.kill()

    def _do_shutdown() -> None:
        with printer._lock:
            if printer._clients:
                return
        _cleanup()
        os._exit(0)

    def _schedule_shutdown() -> None:
        nonlocal shutdown_timer
        with printer._lock:
            if printer._clients:
                return
        if shutdown_timer is not None:
            shutdown_timer.cancel()
        shutdown_timer = threading.Timer(1.0, _do_shutdown)
        shutdown_timer.daemon = True
        shutdown_timer.start()

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(html_page)

    async def events(request: Request) -> StreamingResponse:
        cq = printer.add_client()

        async def generate() -> AsyncGenerator[str]:
            try:
                while not shutting_down.is_set():
                    try:
                        event = cq.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                printer.remove_client(cq)
                _schedule_shutdown()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def run_task(request: Request) -> JSONResponse:
        nonlocal running, agent_thread, selected_model
        with running_lock:
            if running:
                return JSONResponse({"error": "Agent is already running"}, status_code=409)
            running = True
        body = await request.json()
        task = body.get("task", "").strip()
        model = body.get("model", "").strip() or selected_model
        selected_model = model
        if not task:
            with running_lock:
                running = False
            return JSONResponse({"error": "Empty task"}, status_code=400)
        _record_model_usage(model)
        t = threading.Thread(target=run_agent_thread, args=(task, model), daemon=True)
        with running_lock:
            agent_thread = t
        t.start()
        return JSONResponse({"status": "started"})

    async def stop_task(request: Request) -> JSONResponse:
        if stop_agent():
            return JSONResponse({"status": "stopping"})
        return JSONResponse({"error": "No running task"}, status_code=404)

    async def suggestions(request: Request) -> JSONResponse:
        query = request.query_params.get("q", "").strip()
        mode = request.query_params.get("mode", "general")
        if mode == "files":
            q = query.lower()
            results: list[dict[str, str]] = []
            for path in file_cache:
                if not q or q in path.lower():
                    ptype = "dir" if path.endswith("/") else "file"
                    results.append({"type": ptype, "text": path})
                    if len(results) >= 20:
                        break
            return JSONResponse(results)
        if not query:
            return JSONResponse([])
        q_lower = query.lower()
        results = []
        for entry in _load_history():
            task = entry["task"]
            if q_lower in task.lower():
                results.append({"type": "task", "text": task})
                if len(results) >= 5:
                    break
        with proposed_lock:
            for t in proposed_tasks:
                if q_lower in t.lower():
                    results.append({"type": "suggested", "text": t})
        words = query.split()
        last_word = words[-1].lower() if words else q_lower
        if last_word and len(last_word) >= 2:
            count = 0
            for path in file_cache:
                if last_word in path.lower():
                    results.append({"type": "file", "text": path})
                    count += 1
                    if count >= 8:
                        break
        return JSONResponse(results)

    async def tasks(request: Request) -> JSONResponse:
        return JSONResponse(_load_history())

    async def proposed_tasks_endpoint(request: Request) -> JSONResponse:
        with proposed_lock:
            return JSONResponse(list(proposed_tasks))

    async def complete(request: Request) -> JSONResponse:
        raw_query = request.query_params.get("q", "")
        query = raw_query.strip()
        if not query or len(query) < 3:
            return JSONResponse({"suggestion": ""})

        def _generate() -> str:
            history = _load_history()
            task_list = "\n".join(f"- {e['task']}" for e in history[:20])
            agent = KISSAgent("Autocomplete")
            try:
                result = agent.run(
                    model_name="gemini-2.0-flash",
                    prompt_template=(
                        "You are an inline autocomplete engine for a coding assistant. "
                        "Given the user's partial input and their past task history, "
                        "predict what they want to type and return ONLY the remaining "
                        "text to complete their input. Do NOT repeat the text they already typed. "
                        "Keep the completion concise and natural.\n\n"
                        "Past tasks:\n{task_list}\n\n"
                        'Partial input: "{query}"\n\n'
                        "Return ONLY the completion text (the part after what they typed). "
                        "If no good completion, return empty string."
                    ),
                    arguments={"task_list": task_list, "query": query},
                    is_agentic=False,
                )
                s = result.strip().strip('"').strip("'")
                if s.lower().startswith(query.lower()):
                    s = s[len(query):]
                if s and not s[0].isspace() and raw_query and not raw_query[-1].isspace():
                    s = " " + s
                return s
            except Exception:
                return ""

        suggestion = await asyncio.to_thread(_generate)
        return JSONResponse({"suggestion": suggestion})

    async def models_endpoint(request: Request) -> JSONResponse:
        usage = _load_model_usage()
        models_list: list[dict[str, Any]] = []
        for name in get_available_models():
            info = MODEL_INFO.get(name)
            if info and info.is_function_calling_supported:
                models_list.append({
                    "name": name,
                    "inp": info.input_price_per_1M,
                    "out": info.output_price_per_1M,
                    "uses": usage.get(name, 0),
                })
        models_list.sort(key=lambda m: (
            _model_vendor_order(str(m["name"])),
            -(float(m["inp"]) + float(m["out"])),
        ))
        return JSONResponse({"models": models_list, "selected": selected_model})


    async def open_file(request: Request) -> JSONResponse:
        body = await request.json()
        rel = body.get("path", "").strip()
        if not rel:
            return JSONResponse({"error": "No path"}, status_code=400)
        full = rel if rel.startswith("/") else os.path.join(actual_work_dir, rel)
        if not os.path.isfile(full):
            return JSONResponse({"error": "File not found"}, status_code=404)
        pending = os.path.join(cs_data_dir, "pending-open.json")
        with open(pending, "w") as f:
            json.dump({"path": full}, f)
        return JSONResponse({"status": "ok"})

    app = Starlette(routes=[
        Route("/", index),
        Route("/events", events),
        Route("/run", run_task, methods=["POST"]),
        Route("/stop", stop_task, methods=["POST"]),
        Route("/open-file", open_file, methods=["POST"]),
        Route("/suggestions", suggestions),
        Route("/complete", complete),
        Route("/tasks", tasks),
        Route("/proposed_tasks", proposed_tasks_endpoint),
        Route("/models", models_endpoint),
    ])

    threading.Thread(target=refresh_proposed_tasks, daemon=True).start()

    import atexit
    atexit.register(_cleanup)

    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"{title} running at {url}")
    print(f"Work directory: {actual_work_dir}")
    webbrowser.open(url)
    import logging
    logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    _orig_handle_exit = server.handle_exit

    def _on_exit(sig: int, frame: types.FrameType | None) -> None:
        shutting_down.set()
        _orig_handle_exit(sig, frame)

    server.handle_exit = _on_exit  # type: ignore[method-assign]
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    _cleanup()
    os._exit(0)


def main() -> None:
    """Launch the KISS chatbot UI in assistant or coding mode based on KISS_MODE env var."""
    from kiss._version import __version__
    from kiss.agents.assistant.assistant_agent import AssistantAgent
    from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

    work_dir = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else os.getcwd()

    mode = os.environ.get("KISS_MODE", "assistant").lower()
    if mode == "assistant":
        run_chatbot(
            agent_factory=AssistantAgent,
            title=f"KISS Assistant: {__version__}",
            subtitle=f"Working Directory: {work_dir}",
            work_dir=work_dir,
            agent_kwargs={"headless": False},
        )
    else:
        run_chatbot(
            agent_factory=RelentlessCodingAgent,
            title=f"KISS Coding Assistant: {__version__}",
            subtitle=f"Working Directory: {work_dir}",
            work_dir=work_dir,
        )


if __name__ == "__main__":
    main()
