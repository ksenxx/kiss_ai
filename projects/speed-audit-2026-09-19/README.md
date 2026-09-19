# Sorcar speed audit (7 days ending 2026-09-19 05:41 UTC)

Reproduce:

```bash
sqlite3 ~/.kiss/sorcar.db "VACUUM INTO '/tmp/speed7.db'"
sqlite3 /tmp/speed7.db "
attach '/tmp/speed7d.db' as d;
create table d.th as select id, timestamp, task, result, chat_id, model, work_dir, tokens, cost, steps,
  is_parallel, is_worktree, start_ts, end_ts, parent_task_id, max_budget
  from task_history where timestamp > strftime('%s','now')-7*86400;
create table d.ev as select e.task_id, e.seq, e.timestamp ts,
  json_extract(e.event_json,'$.type') type, json_extract(e.event_json,'$.name') name,
  json_extract(e.event_json,'$.tool_name') tool_name, json_extract(e.event_json,'$.is_error') is_error,
  json_extract(e.event_json,'$.callId') call_id, length(json_extract(e.event_json,'$.text')) text_len,
  length(json_extract(e.event_json,'$.content')) content_len, json_extract(e.event_json,'$.total_steps') total_steps,
  json_extract(e.event_json,'$.total_tokens') total_tokens, json_extract(e.event_json,'$.cost') cost_str,
  case when json_extract(e.event_json,'$.type')='usage_info' then json_extract(e.event_json,'$.text')
       when json_extract(e.event_json,'$.type')='tool_call' then substr(e.event_json,1,600)
       when json_extract(e.event_json,'$.type')='tool_result' then substr(json_extract(e.event_json,'$.content'),1,300)
       when json_extract(e.event_json,'$.type') in ('result','task_done','task_stopped','followup_suggestion') then substr(e.event_json,1,400)
       else null end snippet
  from events e where e.task_id in (select id from d.th);
create index d.idx_ev on d.ev(task_id, seq);"
sqlite3 /tmp/speed7.db "attach '/tmp/speed7a.db' as a; create table a.tc_len as select e.task_id, e.seq, e.timestamp ts,
  json_extract(e.event_json,'$.name') name, length(e.event_json) l from events e
  where e.task_id in (select id from a.tasks) and json_extract(e.event_json,'$.type')='tool_call';"  # after step 3
python3 projects/speed-audit-2026-09-19/speed_analysis.py
```

Findings are in FINDINGS.md.
