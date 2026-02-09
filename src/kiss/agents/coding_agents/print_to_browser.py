"""Browser output streaming for Claude Coding Agent via SSE."""

import asyncio
import json
import queue
import socket
import threading
import time
import webbrowser
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

_LANG_MAP = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "rb": "ruby",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "h": "c",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "xml": "xml",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "md": "markdown",
}
_MAX_RESULT_LEN = 3000


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KISS Agent</title>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0d1117;--surface:#161b22;--surface2:#1c2128;--border:#30363d;
  --text:#e6edf3;--dim:#8b949e;--accent:#58a6ff;--green:#3fb950;
  --red:#f85149;--yellow:#d29922;--cyan:#79c0ff;--purple:#bc8cff;
}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.6;
  height:100vh;display:flex;flex-direction:column;
}
header{
  background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 100%);
  border-bottom:1px solid var(--border);padding:14px 28px;
  display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
}
.logo{font-size:18px;font-weight:700;color:var(--accent);letter-spacing:-.3px}
.logo span{color:var(--dim);font-weight:400;font-size:14px;margin-left:8px}
.status{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--dim)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.dot.done{animation:none;background:var(--dim)}
main{flex:1;overflow-y:auto;padding:20px 28px;scroll-behavior:smooth}
.ev{margin-bottom:6px;animation:fadeIn .15s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:translateY(0)}}
.think{
  border-left:3px solid var(--cyan);padding:10px 16px;margin:10px 0;
  background:rgba(121,192,255,.04);border-radius:0 8px 8px 0;
  max-height:200px;overflow-y:auto;
}
.think .lbl{
  font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;color:var(--cyan);margin-bottom:4px;
  display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none;
}
.think .lbl .arrow{transition:transform .2s;display:inline-block}
.think .lbl .arrow.collapsed{transform:rotate(-90deg)}
.think .cnt{
  font-size:13px;color:var(--dim);font-style:italic;
  white-space:pre-wrap;word-break:break-word;
}
.think .cnt.hidden{display:none}
.txt{font-size:14px;white-space:pre-wrap;word-break:break-word;padding:2px 0;line-height:1.7}
.tc{
  border:1px solid var(--border);border-radius:8px;margin:10px 0;
  overflow:hidden;background:var(--surface);
  transition:box-shadow .2s;
}
.tc:hover{box-shadow:0 2px 12px rgba(0,0,0,.3)}
.tc-h{
  padding:9px 14px;background:var(--surface2);
  display:flex;align-items:center;gap:10px;
  cursor:pointer;user-select:none;
}
.tc-h:hover{background:rgba(48,54,61,.8)}
.tc-h .chv{color:var(--dim);transition:transform .2s;font-size:11px;flex-shrink:0}
.tc-h .chv.open{transform:rotate(90deg)}
.tn{font-weight:600;font-size:13px;color:var(--accent)}
.tp{font-size:12px;color:var(--cyan);font-family:'SF Mono','Fira Code',monospace}
.td{font-size:12px;color:var(--dim);font-style:italic}
.tc-b{
  padding:10px 14px;max-height:300px;overflow-y:auto;
  font-family:'SF Mono','Fira Code',monospace;font-size:12px;line-height:1.5;
}
.tc-b.hide{display:none}
.tc-b pre{margin:4px 0;white-space:pre-wrap;word-break:break-word}
.diff-old{color:var(--red);background:rgba(248,81,73,.08);
  padding:2px 6px;border-radius:3px;display:block;margin:2px 0}
.diff-new{color:var(--green);background:rgba(63,185,80,.08);
  padding:2px 6px;border-radius:3px;display:block;margin:2px 0}
.extra{color:var(--dim);margin:2px 0}
.tr{
  border-left:3px solid var(--green);padding:8px 14px;margin:6px 0;
  border-radius:0 8px 8px 0;font-family:'SF Mono','Fira Code',monospace;
  font-size:12px;max-height:200px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-word;background:rgba(63,185,80,.04);
}
.tr.err{border-left-color:var(--red);background:rgba(248,81,73,.04)}
.tr .rl{font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:4px}
.tr .rl.ok{color:var(--green)}
.tr .rl.fail{color:var(--red)}
.rc{border:2px solid var(--green);border-radius:10px;
  margin:20px 0;overflow:hidden;background:var(--surface)}
.rc-h{
  padding:14px 20px;background:rgba(63,185,80,.08);
  display:flex;align-items:center;justify-content:space-between;
}
.rc-h h3{color:var(--green);font-size:15px;font-weight:600}
.rs{font-size:12px;color:var(--dim);display:flex;gap:18px}
.rs b{color:var(--text);font-weight:500}
.rc-body{
  padding:16px 20px;font-size:14px;max-height:400px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-word;line-height:1.7;
}
.sys{
  font-size:13px;color:var(--dim);font-family:'SF Mono','Fira Code',monospace;
  white-space:pre-wrap;word-break:break-word;padding:2px 0;
}
.usage{
  border:1px solid var(--border);border-radius:6px;margin:8px 0;
  padding:6px 12px;background:var(--surface);
  font-size:9px;color:var(--dim);line-height:1.5;opacity:0.7;
  white-space:pre-wrap;word-break:break-word;
}
footer{
  background:var(--surface);border-top:1px solid var(--border);
  padding:8px 28px;font-size:12px;color:var(--dim);flex-shrink:0;
  display:flex;justify-content:space-between;
}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--dim)}
code{font-family:'SF Mono','Fira Code',monospace}
.hljs{background:var(--surface2)!important;border-radius:6px;padding:10px!important}
</style>
</head>
<body>
<header>
  <div class="logo">KISS Agent<span>Live Stream</span></div>
  <div class="status"><div class="dot" id="dot"></div><span id="stxt">Running</span></div>
</header>
<main id="out"></main>
<footer>
  <span id="evcnt">Events: 0</span>
  <span id="elapsed">Elapsed: 0s</span>
</footer>
<script>
const O=document.getElementById('out'),D=document.getElementById('dot'),
  ST=document.getElementById('stxt'),EC=document.getElementById('evcnt'),
  EL=document.getElementById('elapsed');
let ec=0,auto=true,thinkEl=null,txtEl=null;
const t0=Date.now();
O.addEventListener('scroll',()=>{auto=O.scrollTop+O.clientHeight>=O.scrollHeight-60});
function sb(){if(auto)O.scrollTop=O.scrollHeight}
function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML}
setInterval(()=>{
  const s=Math.floor((Date.now()-t0)/1000),m=Math.floor(s/60);
  EL.textContent='Elapsed: '+(m>0?m+'m ':'')+s%60+'s';
},1000);
function toggleTC(el){
  const b=el.nextElementSibling,c=el.querySelector('.chv');
  b.classList.toggle('hide');c.classList.toggle('open');
}
function toggleThink(el){
  const cnt=el.parentElement.querySelector('.cnt');
  const arrow=el.querySelector('.arrow');
  cnt.classList.toggle('hidden');
  arrow.classList.toggle('collapsed');
}
const src=new EventSource('/events');
src.onmessage=function(e){
  let ev;try{ev=JSON.parse(e.data)}catch{return}
  ec++;EC.textContent='Events: '+ec;
  switch(ev.type){
    case'thinking_start':
      thinkEl=document.createElement('div');
      thinkEl.className='ev think';
      thinkEl.innerHTML='<div class="lbl" onclick="toggleThink(this)">'
        +'<span class="arrow">▾</span> Thinking</div>'
        +'<div class="cnt"></div>';
      O.appendChild(thinkEl);break;
    case'thinking_delta':
      if(thinkEl){const c=thinkEl.querySelector('.cnt');
        c.textContent+=ev.text||'';
        thinkEl.scrollTop=thinkEl.scrollHeight}break;
    case'thinking_end':
      if(thinkEl){
        const lbl=thinkEl.querySelector('.lbl');
        lbl.innerHTML='<span class="arrow collapsed">▾</span> Thinking (click to expand)';
        thinkEl.querySelector('.cnt').classList.add('hidden');
      }
      thinkEl=null;break;
    case'text_delta':
      if(!txtEl){txtEl=document.createElement('div');txtEl.className='ev txt';O.appendChild(txtEl)}
      txtEl.textContent+=ev.text||'';break;
    case'text_end':txtEl=null;break;
    case'tool_call':{
      const c=document.createElement('div');c.className='ev tc';
      let h='<span class="chv open">▶</span><span class="tn">'+esc(ev.name)+'</span>';
      if(ev.path)h+='<span class="tp">'+esc(ev.path)+'</span>';
      if(ev.description)h+='<span class="td">'+esc(ev.description)+'</span>';
      let b='';
      if(ev.command)b+='<pre><code class="language-bash">'+esc(ev.command)+'</code></pre>';
      if(ev.content)b+='<pre><code class="'
        +(ev.lang?'language-'+esc(ev.lang):'')
        +'">'+esc(ev.content)+'</code></pre>';
      if(ev.old_string!==undefined)b+='<div class="diff-old">- '+esc(ev.old_string)+'</div>';
      if(ev.new_string!==undefined)b+='<div class="diff-new">+ '+esc(ev.new_string)+'</div>';
      if(ev.extras)for(const[k,v]of Object.entries(ev.extras))
        b+='<div class="extra">'+esc(k)+': '+esc(v)+'</div>';
      c.innerHTML='<div class="tc-h" onclick="toggleTC(this)">'+h+'</div>'
        +'<div class="tc-b'+(b?'':' hide')+'">'
        +(b||'<em style="color:var(--dim)">No arguments</em>')
        +'</div>';
      O.appendChild(c);
      c.querySelectorAll('pre code').forEach(bl=>hljs.highlightElement(bl));
      break}
    case'tool_result':{
      const r=document.createElement('div');
      r.className='ev tr'+(ev.is_error?' err':'');
      const lb=ev.is_error?'FAILED':'OK',lc=ev.is_error?'fail':'ok';
      r.innerHTML='<div class="rl '+lc+'">'+lb+'</div>'+esc(ev.content);
      O.appendChild(r);break}
    case'system_output':{
      const s=document.createElement('div');s.className='ev sys';
      s.textContent=ev.text||'';O.appendChild(s);break}
    case'result':{
      const c=document.createElement('div');c.className='ev rc';
      c.innerHTML='<div class="rc-h"><h3>Result</h3><div class="rs">'
        +'Steps: <b>'+(ev.step_count||0)+'</b>'
        +'Tokens: <b>'+(ev.total_tokens||0)+'</b>'
        +'Cost: <b>'+(ev.cost||'N/A')+'</b>'
        +'</div></div><div class="rc-body">'+esc(ev.text||'(no result)')+'</div>';
      O.appendChild(c);break}
    case'usage_info':{
      const u=document.createElement('div');u.className='ev usage';
      u.textContent=ev.text||'';O.appendChild(u);break}
    case'done':
      D.classList.add('done');ST.textContent='Completed';src.close();break;
  }
  sb();
};
src.onerror=function(){D.classList.add('done');ST.textContent='Disconnected'};
</script>
</body>
</html>"""


class BrowserPrinter:
    """Handles all browser output for Claude Coding Agent via SSE.

    API:
        start(open_browser=True) -> None:
            Start the uvicorn server and optionally open browser.

        stop() -> None:
            Send done event and stop server.

        print_stream_event(event) -> str:
            Handle a streaming event and send to browser.
            Returns extracted text content for token callbacks.

        print_message(message, **context) -> None:
            Send a complete message to browser.
    """

    def __init__(self) -> None:
        self._clients: list[queue.Queue[dict[str, Any]]] = []
        self._clients_lock = threading.Lock()
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""
        self._port = 0
        self._server_thread: threading.Thread | None = None
        self._server: Any = None

    def start(self, open_browser: bool = True) -> None:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import HTMLResponse, StreamingResponse
        from starlette.routing import Route

        printer = self

        async def index(request: Request) -> HTMLResponse:
            return HTMLResponse(_HTML_PAGE)

        async def events(request: Request) -> StreamingResponse:
            client_q: queue.Queue[dict[str, Any]] = queue.Queue()
            with printer._clients_lock:
                printer._clients.append(client_q)

            async def generate() -> AsyncGenerator[str]:
                try:
                    while True:
                        try:
                            event = client_q.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.05)
                            continue
                        yield f"data: {json.dumps(event)}\n\n"
                        if event.get("type") == "done":
                            break
                except asyncio.CancelledError:
                    pass
                finally:
                    with printer._clients_lock:
                        if client_q in printer._clients:
                            printer._clients.remove(client_q)

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        app = Starlette(routes=[Route("/", index), Route("/events", events)])
        self._port = _find_free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(target=self._server.run, daemon=True)
        self._server_thread.start()
        time.sleep(0.5)
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{self._port}")

    def stop(self) -> None:
        self._broadcast({"type": "done"})
        time.sleep(0.3)
        if self._server:
            self._server.should_exit = True

    def reset(self) -> None:
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""

    def _broadcast(self, event: dict[str, Any]) -> None:
        with self._clients_lock:
            for cq in self._clients:
                cq.put(event)

    @staticmethod
    def _lang_for_path(path: str) -> str:
        ext = Path(path).suffix.lstrip(".")
        return _LANG_MAP.get(ext, ext or "text")

    def _format_tool_call(self, name: str, tool_input: dict[str, Any]) -> None:
        file_path = str(tool_input.get("file_path") or tool_input.get("path") or "")
        lang = self._lang_for_path(file_path) if file_path else "text"
        event: dict[str, Any] = {"type": "tool_call", "name": name}
        if file_path:
            event["path"] = file_path
            event["lang"] = lang
        desc = tool_input.get("description")
        if desc:
            event["description"] = str(desc)
        command = tool_input.get("command")
        if command:
            event["command"] = str(command)
        content = tool_input.get("content")
        if content:
            event["content"] = str(content)
        old_string = tool_input.get("old_string")
        new_string = tool_input.get("new_string")
        if old_string is not None:
            event["old_string"] = str(old_string)
        if new_string is not None:
            event["new_string"] = str(new_string)
        skip = {
            "file_path",
            "path",
            "content",
            "command",
            "old_string",
            "new_string",
            "description",
        }
        extras: dict[str, str] = {}
        for k, v in tool_input.items():
            if k not in skip:
                val = str(v)
                if len(val) > 200:
                    val = val[:200] + "..."
                extras[k] = val
        if extras:
            event["extras"] = extras
        self._broadcast(event)

    def _print_tool_result(self, content: str, is_error: bool) -> None:
        display = content
        if len(display) > _MAX_RESULT_LEN:
            half = _MAX_RESULT_LEN // 2
            display = display[:half] + "\n... (truncated) ...\n" + display[-half:]
        self._broadcast({"type": "tool_result", "content": display, "is_error": is_error})

    def print_stream_event(self, event: Any) -> str:
        evt = event.event
        evt_type = evt.get("type", "")
        text = ""

        if evt_type == "content_block_start":
            block = evt.get("content_block", {})
            block_type = block.get("type", "")
            self._current_block_type = block_type
            if block_type == "thinking":
                self._broadcast({"type": "thinking_start"})
            elif block_type == "tool_use":
                self._tool_name = block.get("name", "?")
                self._tool_json_buffer = ""

        elif evt_type == "content_block_delta":
            delta = evt.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "thinking_delta":
                text = delta.get("thinking", "")
                if text:
                    self._broadcast({"type": "thinking_delta", "text": text})
            elif delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    self._broadcast({"type": "text_delta", "text": text})
            elif delta_type == "input_json_delta":
                self._tool_json_buffer += delta.get("partial_json", "")

        elif evt_type == "content_block_stop":
            block_type = self._current_block_type
            if block_type == "thinking":
                self._broadcast({"type": "thinking_end"})
            elif block_type == "tool_use":
                try:
                    tool_input = json.loads(self._tool_json_buffer)
                except (json.JSONDecodeError, ValueError):
                    tool_input = {"_raw": self._tool_json_buffer}
                self._format_tool_call(self._tool_name, tool_input)
            else:
                self._broadcast({"type": "text_end"})
            self._current_block_type = ""

        return text

    def print_message(
        self,
        message: Any,
        step_count: int = 0,
        budget_used: float = 0.0,
        total_tokens_used: int = 0,
    ) -> None:
        if hasattr(message, "subtype") and hasattr(message, "data"):
            self._print_system(message)
        elif hasattr(message, "result"):
            self._print_result(message, step_count, budget_used, total_tokens_used)
        elif hasattr(message, "content"):
            self._print_tool_results(message)

    def _print_system(self, message: Any) -> None:
        if message.subtype == "tool_output":
            text = message.data.get("content", "")
            if text:
                self._broadcast({"type": "system_output", "text": text})

    def _print_result(
        self, message: Any, step_count: int, budget_used: float, total_tokens_used: int
    ) -> None:
        cost_str = f"${budget_used:.4f}" if budget_used else "N/A"
        self._broadcast(
            {
                "type": "result",
                "text": message.result or "(no result)",
                "step_count": step_count,
                "total_tokens": total_tokens_used,
                "cost": cost_str,
            }
        )

    def print_usage_info(self, usage_info: str) -> None:
        self._broadcast({"type": "usage_info", "text": usage_info.strip()})

    def _print_tool_results(self, message: Any) -> None:
        for block in message.content:
            if hasattr(block, "is_error") and hasattr(block, "content"):
                content = block.content if isinstance(block.content, str) else str(block.content)
                self._print_tool_result(content, bool(block.is_error))
