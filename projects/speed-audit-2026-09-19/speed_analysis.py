"""Wall-clock speed audit of Sorcar tasks from ~/.kiss/sorcar.db.

Usage (see README.md in this directory):
  1. sqlite3 ~/.kiss/sorcar.db "VACUUM INTO '/tmp/speed7.db'"
  2. build /tmp/speed7d.db (derived event table) with the SQL in README.md
  3. python3 speed_analysis.py   -> writes /tmp/speed7a.db (tables steps, calls, tasks)

Per step: llm_time = usage_info("Steps: N") ts - previous boundary (prompt ts or last tool_result ts);
per tool call: dur = tool_result ts - tool_call ts.  Live "Tokens: ..." usage_info events emitted while
run_parallel is waiting are NOT step boundaries.
"""
import sqlite3, json, collections, statistics, re, sys
con = sqlite3.connect('/tmp/speed7d.db')
con.row_factory = sqlite3.Row
th = {r['id']: dict(r) for r in con.execute('select * from th')}
kids = collections.defaultdict(list)
for t in th.values():
    if t['parent_task_id']:
        kids[t['parent_task_id']].append(t['id'])

def root_of(tid):
    seen = set()
    while th.get(tid, {}).get('parent_task_id') and tid not in seen:
        seen.add(tid)
        tid = th[tid]['parent_task_id']
    return tid

# per-task step segmentation
steps_rows = []   # (task_id, step_no, llm_time, out_chars, ctx_tokens, tool_time, ntools, tools_str)
calls_rows = []   # (task_id, step_no, tool_name, dur, is_error, snippet)
task_rows = {}
cur = con.execute('select task_id, seq, ts, type, name, tool_name, is_error, text_len, content_len, total_steps, total_tokens, snippet from ev order by task_id, seq')
groups = collections.defaultdict(list)
for r in cur:
    groups[r['task_id']].append(r)

ctx_re = re.compile(r'Context: ([\d,]+)/')
for tid, evs in groups.items():
    first_ts = evs[0]['ts']; last_ts = evs[-1]['ts']
    prompt_ts = None; settings_ts = None
    boundary = None
    out_chars = 0
    step_no = 0
    pending = []  # list of (name, ts, snippet)
    step_tool_first = None; step_tool_last = None; step_tools = []
    llm_total = 0.0; tool_total = 0.0; tool_by = collections.Counter(); ntool_calls = 0
    result_ts = None; done_ts = None
    for e in evs:
        t = e['type']
        if t == 'task_settings': settings_ts = e['ts']
        elif t == 'prompt':
            if prompt_ts is None: prompt_ts = e['ts']
            if boundary is None: boundary = e['ts']
        elif t in ('thinking_delta', 'text_delta'):
            out_chars += (e['text_len'] or 0)
        elif t == 'usage_info':
            if not (e['snippet'] or '').startswith('Steps:'):
                continue
            # close previous step's tools
            if step_no > 0:
                ttime = (step_tool_last - step_tool_first) if (step_tool_first is not None and step_tool_last is not None) else 0.0
                steps_rows[-1][5] = ttime; steps_rows[-1][6] = len(step_tools); steps_rows[-1][7] = ','.join(step_tools)
                if step_tool_last is not None: boundary = step_tool_last
            step_no += 1
            b = boundary if boundary is not None else first_ts
            llm = e['ts'] - b
            m = ctx_re.search(e['snippet'] or '')
            ctx = int(m.group(1).replace(',', '')) if m else None
            steps_rows.append([tid, step_no, llm, out_chars, ctx, 0.0, 0, ''])
            llm_total += llm
            out_chars = 0
            boundary = e['ts']
            step_tool_first = None; step_tool_last = None; step_tools = []; pending = []
        elif t == 'tool_call':
            pending.append((e['name'], e['ts'], e['snippet']))
            if step_tool_first is None: step_tool_first = e['ts']
            step_tools.append(e['name'] or '?')
        elif t == 'tool_result':
            # match FIFO by tool_name if possible
            idx = None
            for i, p in enumerate(pending):
                if p[0] == e['tool_name']: idx = i; break
            if idx is None and pending: idx = 0
            if idx is not None:
                name, cts, snip = pending.pop(idx)
                dur = e['ts'] - cts
                calls_rows.append((tid, step_no, name, dur, e['is_error'], snip, e['snippet'], e['content_len']))
                tool_total += dur; tool_by[name] += dur; ntool_calls += 1
            step_tool_last = e['ts']
        elif t == 'result': result_ts = e['ts']
        elif t == 'task_done': done_ts = e['ts']
    if step_no > 0 and step_tool_first is not None and step_tool_last is not None:
        ttime = step_tool_last - step_tool_first
        steps_rows[-1][5] = ttime; steps_rows[-1][6] = len(step_tools); steps_rows[-1][7] = ','.join(step_tools)
    t = th[tid]
    task_rows[tid] = dict(id=tid, wall=last_ts - first_ts, first_ts=first_ts, last_ts=last_ts, prompt_ts=prompt_ts, settings_ts=settings_ts,
                          startup=(prompt_ts - first_ts) if prompt_ts else None,
                          llm=llm_total, tool=tool_total, tool_by=dict(tool_by), ntools=ntool_calls, steps=step_no,
                          result_ts=result_ts, done_ts=done_ts, model=t['model'], parent=t['parent_task_id'], cost=t['cost'],
                          task=t['task'], root=root_of(tid), max_budget=t['max_budget'], start_ts=t['start_ts'], end_ts=t['end_ts'])

out = sqlite3.connect('/tmp/speed7a.db')
out.executescript('''
drop table if exists steps; drop table if exists calls; drop table if exists tasks;
create table steps(task_id, step_no, llm_time, out_chars, ctx, tool_time, ntools, tools);
create table calls(task_id, step_no, tool, dur, is_error, call_snip, res_snip, res_len);
create table tasks(id, wall, first_ts, last_ts, prompt_ts, startup, llm, tool, ntools, steps, result_ts, done_ts, model, parent, cost, task, root, max_budget, start_ts, end_ts, tool_by);
''')
out.executemany('insert into steps values (?,?,?,?,?,?,?,?)', steps_rows)
out.executemany('insert into calls values (?,?,?,?,?,?,?,?)', calls_rows)
out.executemany('insert into tasks values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                [(v['id'], v['wall'], v['first_ts'], v['last_ts'], v['prompt_ts'], v['startup'], v['llm'], v['tool'], v['ntools'], v['steps'], v['result_ts'], v['done_ts'], v['model'], v['parent'], v['cost'], v['task'], v['root'], v['max_budget'], v['start_ts'], v['end_ts'], json.dumps(v['tool_by'])) for v in task_rows.values()])
out.commit()
print('tasks', len(task_rows), 'steps', len(steps_rows), 'calls', len(calls_rows))
