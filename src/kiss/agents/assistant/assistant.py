"""Browser-based chatbot for RelentlessAgent-based agents."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import webbrowser
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from kiss.agents.assistant.relentless_agent import RelentlessAgent
from kiss.core.browser_ui import (
    BASE_CSS,
    EVENT_HANDLER_JS,
    HTML_HEAD,
    OUTPUT_CSS,
    BaseBrowserPrinter,
    find_free_port,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import MODEL_INFO, get_available_models

_KISS_DIR = Path.home() / ".kiss"
HISTORY_FILE = _KISS_DIR / "task_history.json"
PROPOSALS_FILE = _KISS_DIR / "proposed_tasks.json"
MAX_HISTORY = 1000


def _normalize_history_entry(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict) and "task" in raw:
        return {"task": str(raw["task"]), "result": str(raw.get("result", ""))}
    return {"task": str(raw), "result": ""}


def _load_history() -> list[dict[str, str]]:
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            if isinstance(data, list):
                seen: set[str] = set()
                result: list[dict[str, str]] = []
                for t in data[:MAX_HISTORY]:
                    entry = _normalize_history_entry(t)
                    task_str = entry["task"]
                    if task_str not in seen:
                        seen.add(task_str)
                        result.append(entry)
                return result
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(entries: list[dict[str, str]]) -> None:
    try:
        _KISS_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(entries[:MAX_HISTORY]))
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


def _find_semantic_duplicates(new_task: str, existing_tasks: list[str]) -> list[int]:
    if not existing_tasks:
        return []
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(existing_tasks))
    agent = KISSAgent("Task Deduplicator")
    result = agent.run(
        model_name="gemini-2.0-flash",
        prompt_template=(
            "Which existing tasks are semantically the same as the new task? "
            '"Same" means they ask for essentially the same work, just worded differently.\n\n'
            "New task: {new_task}\n\n"
            "Existing tasks:\n{existing_tasks}\n\n"
            "Return ONLY a JSON array of indices of duplicate tasks. "
            "If none are duplicates, return []. Examples: [2, 5] or []"
        ),
        arguments={"new_task": new_task, "existing_tasks": numbered},
        is_agentic=False,
    )
    try:
        start = result.index("[")
        end = result.index("]", start) + 1
        indices = json.loads(result[start:end])
        return [i for i in indices if isinstance(i, int) and 0 <= i < len(existing_tasks)]
    except (ValueError, json.JSONDecodeError):
        return []


def _add_task(task: str) -> None:
    history = _load_history()
    task_strings = [e["task"] for e in history]
    if task in task_strings:
        idx = next(i for i, e in enumerate(history) if e["task"] == task)
        history.pop(idx)
    else:
        try:
            duplicates = _find_semantic_duplicates(task, task_strings)
            for idx in sorted(duplicates, reverse=True):
                history.pop(idx)
        except Exception:
            pass
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
            for name in sorted(files):
                paths.append(os.path.relpath(os.path.join(root, name), work_dir))
                if len(paths) >= 1000:
                    return paths
    except OSError:
        pass
    return paths


class _StopRequested(BaseException):
    pass


CHATBOT_CSS = r"""
#output{
  flex:2;overflow-y:auto;padding:16px 24px;
  scroll-behavior:smooth;border-bottom:1px solid var(--border);min-height:0;
}
#input-area{
  flex-shrink:0;padding:12px 24px;background:var(--surface);
  border-bottom:1px solid var(--border);position:relative;
}
#input-row{display:flex;gap:10px;align-items:center}
#task-input{
  flex:1;background:var(--bg);border:1px solid var(--border);border-radius:8px;
  padding:10px 14px;color:var(--text);font-size:14px;font-family:inherit;outline:none;
  transition:border-color .2s;
}
#task-input:focus{border-color:var(--accent)}
#task-input:disabled{opacity:.5;cursor:not-allowed}
#send-btn{
  background:var(--accent);color:#fff;border:none;border-radius:8px;
  padding:10px 20px;font-size:14px;font-weight:600;cursor:pointer;
  transition:background .2s;white-space:nowrap;
}
#send-btn:hover{background:#79b8ff}
#send-btn:disabled{opacity:.5;cursor:not-allowed}
#stop-btn{
  background:var(--red);color:#fff;border:none;border-radius:8px;
  padding:10px 20px;font-size:14px;font-weight:600;cursor:pointer;
  transition:background .2s;white-space:nowrap;display:none;
}
#stop-btn:hover{background:#f9706a}
#autocomplete{
  position:absolute;bottom:100%;left:24px;right:24px;
  background:var(--surface2);border:1px solid var(--border);border-radius:8px;
  max-height:220px;overflow-y:auto;display:none;z-index:10;
  box-shadow:0 -4px 16px rgba(0,0,0,.4);
}
.ac-item{
  padding:8px 14px;cursor:pointer;font-size:13px;
  border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;
}
.ac-item:last-child{border-bottom:none}
.ac-item:hover,.ac-item.sel{background:rgba(88,166,255,.1)}
.ac-type{
  font-size:10px;font-weight:600;text-transform:uppercase;
  padding:2px 6px;border-radius:3px;flex-shrink:0;
}
.ac-type.task{background:rgba(188,140,255,.15);color:var(--purple)}
.ac-type.file{background:rgba(63,185,80,.15);color:var(--green)}
.ac-text{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#bottom-panel{flex:1;display:flex;gap:1px;background:var(--border);min-height:0}
.panel-col{flex:1;overflow-y:auto;padding:10px 16px;background:var(--bg);min-height:0}
.panel-hdr{
  font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;color:var(--dim);margin-bottom:8px;padding:0 2px;
}
.task-item{
  padding:8px 12px;background:var(--surface);border:1px solid var(--border);
  border-radius:6px;margin-bottom:6px;cursor:pointer;font-size:13px;
  transition:all .15s;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.task-item:hover{border-color:var(--accent);background:var(--surface2)}
.proposed-item{
  padding:8px 12px;background:var(--surface);border:1px solid var(--border);
  border-radius:6px;margin-bottom:6px;cursor:pointer;font-size:13px;
  transition:all .15s;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.proposed-item:hover{border-color:var(--purple);background:var(--surface2)}
.no-tasks{color:var(--dim);font-size:13px;padding:8px 2px}
.loading-proposals{color:var(--dim);font-size:12px;font-style:italic;padding:8px 2px}
#model-select{
  background:var(--bg);color:var(--text);
  border:1px solid var(--border);border-radius:8px;
  padding:8px 10px;font-size:13px;font-family:inherit;
  outline:none;max-width:260px;flex-shrink:0;cursor:pointer;
}
#model-select:focus{border-color:var(--accent)}
#model-select option.no-fc{color:var(--yellow)}
"""

CHATBOT_JS = r"""
var O=document.getElementById('output');
var D=document.getElementById('dot');
var ST=document.getElementById('stxt');
var inp=document.getElementById('task-input');
var btn=document.getElementById('send-btn');
var stopBtn=document.getElementById('stop-btn');
var ac=document.getElementById('autocomplete');
var rl=document.getElementById('recent-list');
var pl=document.getElementById('proposed-list');
var modelSel=document.getElementById('model-select');
var running=false,autoScroll=true;
var scrollRaf=0,state={thinkEl:null,txtEl:null};
var acIdx=-1,t0=null,timerIv=null,evtSrc=null;
O.addEventListener('scroll',function(){
  autoScroll=O.scrollTop+O.clientHeight>=O.scrollHeight-60;
});
function sb(){
  if(!scrollRaf){scrollRaf=requestAnimationFrame(function(){
    if(autoScroll)O.scrollTop=O.scrollHeight;scrollRaf=0;
  });}
}
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
function showSpinner(){
  removeSpinner();
  var sp=mkEl('div','spinner');
  sp.id='wait-spinner';
  sp.textContent='Waiting for agent output\u2026';
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
  evtSrc.onmessage=function(e){
    var ev;try{ev=JSON.parse(e.data);}catch(x){return;}
    handleEvent(ev);
  };
  evtSrc.onerror=function(){};
}
function handleEvent(ev){
  var t=ev.type;
  if(t==='thinking_start'||t==='text_delta'||t==='tool_call'
    ||t==='task_done'||t==='task_error'||t==='task_stopped')removeSpinner();
  switch(t){
  case'tasks_updated':loadTasks();break;
  case'proposed_updated':loadProposed();break;
  case'clear':
    O.innerHTML='';state.thinkEl=null;state.txtEl=null;
    autoScroll=true;showSpinner();break;
  case'task_done':{
    var el=t0?Math.floor((Date.now()-t0)/1000):0;
    var em=Math.floor(el/60);
    setReady('Done ('+(em>0?em+'m ':'')+el%60+'s)');
    loadTasks();loadProposed();break}
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
  default:
    handleOutputEvent(ev,O,state);
    if(t==='tool_result'&&running)showSpinner();
    if(t==='thinking_end'&&running)showSpinner();
  }
  sb();
}
function loadModels(){
  fetch('/models').then(function(r){return r.json();})
  .then(function(d){
    modelSel.innerHTML='';
    d.models.forEach(function(m){
      var o=document.createElement('option');
      o.value=m.name;
      o.textContent=m.name+(m.fc?'':' [no FC]');
      if(!m.fc)o.className='no-fc';
      if(m.name===d.selected)o.selected=true;
      modelSel.appendChild(o);
    });
  }).catch(function(){});
}
function submitTask(){
  var task=inp.value.trim();
  if(!task||running)return;
  running=true;inp.disabled=true;
  btn.style.display='none';
  stopBtn.style.display='inline-block';
  D.classList.add('running');hideAC();startTimer();
  fetch('/run',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task:task,model:modelSel.value})
  }).then(function(r){
    if(!r.ok){r.json().then(function(d){setReady('Error');alert(d.error||'Failed')});return;}
    inp.value='';
  }).catch(function(){setReady('Error');alert('Network error')});
}
btn.addEventListener('click',submitTask);
stopBtn.addEventListener('click',function(){fetch('/stop',{method:'POST'}).catch(function(){})});
inp.addEventListener('keydown',function(e){
  if(ac.style.display==='block'){
    var items=ac.querySelectorAll('.ac-item');
    if(e.key==='ArrowDown'){e.preventDefault();acIdx=Math.min(acIdx+1,items.length-1);updateACSel(items);return}
    if(e.key==='ArrowUp'){e.preventDefault();acIdx=Math.max(acIdx-1,-1);updateACSel(items);return}
    if(e.key==='Enter'&&acIdx>=0){e.preventDefault();items[acIdx].click();return}
    if(e.key==='Escape'){hideAC();return}
  }
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submitTask()}
});
var acTimer=null;
inp.addEventListener('input',function(){if(acTimer)clearTimeout(acTimer);acTimer=setTimeout(fetchAC,200)});
function fetchAC(){
  var q=inp.value.trim();
  if(!q){hideAC();return}
  fetch('/suggestions?q='+encodeURIComponent(q))
  .then(function(r){return r.json()})
  .then(function(data){
    if(!data.length){hideAC();return}
    ac.innerHTML='';acIdx=-1;
    data.forEach(function(item){
      var d=mkEl('div','ac-item');
      d.innerHTML='<span class="ac-type '+item.type+'">'+item.type+'</span>'
        +'<span class="ac-text">'+esc(item.text)+'</span>';
      d.addEventListener('click',function(){
        if(item.type==='task'){inp.value=item.text}
        else{var p=inp.value.split(/\s/);p[p.length-1]=item.text;inp.value=p.join(' ')}
        hideAC();inp.focus();
      });
      ac.appendChild(d);
    });
    ac.style.display='block';
  }).catch(function(){hideAC()});
}
function hideAC(){ac.style.display='none';acIdx=-1}
function updateACSel(items){
  items.forEach(function(it,i){it.classList.toggle('sel',i===acIdx)});
  if(acIdx>=0)items[acIdx].scrollIntoView({block:'nearest'});
}
document.addEventListener('click',function(e){if(!ac.contains(e.target)&&e.target!==inp)hideAC()});
function loadTasks(){
  fetch('/tasks').then(function(r){return r.json()}).then(function(tasks){
    rl.innerHTML='';
    if(!tasks.length){rl.innerHTML='<div class="no-tasks">No recent tasks yet</div>';return}
    tasks.forEach(function(t){
      var taskText=typeof t==='string'?t:(t.task||'');
      var resultText=typeof t==='string'?'':(t.result||'');
      var d=mkEl('div','task-item');
      d.textContent=taskText;
      d.title=resultText?taskText+'\n\nResult: '+resultText:taskText;
      d.addEventListener('click',function(){inp.value=taskText;inp.focus()});
      rl.appendChild(d);
    });
  }).catch(function(){});
}
function loadProposed(){
  pl.innerHTML='<div class="loading-proposals">Loading suggestions\u2026</div>';
  fetch('/proposed_tasks').then(function(r){return r.json()}).then(function(tasks){
    pl.innerHTML='';
    if(!tasks.length){pl.innerHTML='<div class="no-tasks">No suggestions yet</div>';return}
    tasks.forEach(function(t){
      var d=mkEl('div','proposed-item');
      d.textContent=t;d.title=t;
      d.addEventListener('click',function(){inp.value=t;inp.focus()});
      pl.appendChild(d);
    });
  }).catch(function(){pl.innerHTML='<div class="no-tasks">Could not load suggestions</div>'});
}
connectSSE();loadModels();loadTasks();loadProposed();inp.focus();
"""


def _build_html(title: str, subtitle: str) -> str:
    css = BASE_CSS + OUTPUT_CSS + CHATBOT_CSS
    return HTML_HEAD.format(title=title, css=css) + f"""<body>
<header>
  <div class="logo">{title}<span>{subtitle}</span></div>
  <div class="status"><div class="dot" id="dot"></div><span id="stxt">Ready</span></div>
</header>
<div id="output">
  <div class="empty-msg">Enter a task below to start the agent.<br>
    The output will appear here.</div>
</div>
<div id="input-area">
  <div id="autocomplete"></div>
  <div id="input-row">
    <select id="model-select"></select>
    <input type="text" id="task-input"
      placeholder="Describe your task..." autocomplete="off">
    <button id="send-btn">Send</button>
    <button id="stop-btn">Stop</button>
  </div>
</div>
<div id="bottom-panel">
  <div class="panel-col">
    <div class="panel-hdr">Recent Tasks</div>
    <div id="recent-list"></div>
  </div>
  <div class="panel-col">
    <div class="panel-hdr">Proposed Tasks</div>
    <div id="proposed-list"></div>
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
    default_model: str = "claude-sonnet-4-5",
    agent_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run a browser-based chatbot for any RelentlessAgent-based agent."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
    from starlette.routing import Route

    printer = BaseBrowserPrinter()
    running = False
    running_lock = threading.Lock()
    actual_work_dir = work_dir or os.getcwd()
    file_cache: list[str] = _scan_files(actual_work_dir)
    agent_thread: threading.Thread | None = None
    proposed_tasks: list[str] = _load_proposals()
    proposed_lock = threading.Lock()
    selected_model = default_model
    extra_kwargs = agent_kwargs or {}
    html_page = _build_html(title, subtitle)

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

    def run_agent_thread(task: str, model_name: str) -> None:
        nonlocal running, agent_thread
        try:
            _add_task(task)
            printer.broadcast({"type": "tasks_updated"})
            printer.broadcast({"type": "clear"})
            agent = agent_factory("Chatbot")
            result = agent.run(
                prompt_template=task,
                work_dir=actual_work_dir,
                printer=printer,
                model_name=model_name,
                **extra_kwargs,
            )
            _set_latest_result(result or "")
            printer.broadcast({"type": "task_done"})
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

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(html_page)

    async def events(request: Request) -> StreamingResponse:
        cq = printer.add_client()

        async def generate() -> AsyncGenerator[str]:
            try:
                while True:
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
        if not query:
            return JSONResponse([])
        q_lower = query.lower()
        results: list[dict[str, str]] = []
        for entry in _load_history():
            task = entry["task"]
            if q_lower in task.lower():
                results.append({"type": "task", "text": task})
                if len(results) >= 5:
                    break
        last_word = query.split()[-1].lower() if query.split() else q_lower
        if last_word:
            count = 0
            for path in file_cache:
                if last_word in path.lower():
                    results.append({"type": "file", "text": path})
                    count += 1
                    if count >= 10:
                        break
        return JSONResponse(results)

    async def tasks(request: Request) -> JSONResponse:
        return JSONResponse(_load_history())

    async def proposed_tasks_endpoint(request: Request) -> JSONResponse:
        with proposed_lock:
            return JSONResponse(list(proposed_tasks))

    async def models_endpoint(request: Request) -> JSONResponse:
        available = get_available_models()
        models_list = []
        for name in available:
            info = MODEL_INFO.get(name)
            if info:
                models_list.append({"name": name, "fc": info.is_function_calling_supported})
        return JSONResponse({"models": models_list, "selected": selected_model})

    app = Starlette(routes=[
        Route("/", index),
        Route("/events", events),
        Route("/run", run_task, methods=["POST"]),
        Route("/stop", stop_task, methods=["POST"]),
        Route("/suggestions", suggestions),
        Route("/tasks", tasks),
        Route("/proposed_tasks", proposed_tasks_endpoint),
        Route("/models", models_endpoint),
    ])

    threading.Thread(target=refresh_proposed_tasks, daemon=True).start()

    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"{title} running at {url}")
    print(f"Work directory: {actual_work_dir}")
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main() -> None:
    from kiss.agents.assistant.assistant_agent import AssistantAgent
    from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

    work_dir = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else os.getcwd()

    mode = os.environ.get("KISS_MODE", "coding").lower()
    if mode == "assistant":
        run_chatbot(
            agent_factory=AssistantAgent,
            title="KISS Assistant",
            subtitle="Code + Browser Agent",
            work_dir=work_dir,
            agent_kwargs={"headless": False},
        )
    else:
        run_chatbot(
            agent_factory=RelentlessCodingAgent,
            title="KISS Coding Agent",
            subtitle="Relentless Coder",
            work_dir=work_dir,
        )


if __name__ == "__main__":
    main()
