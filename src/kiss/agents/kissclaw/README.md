# KISSClaw

**A lightweight personal AI assistant framework built on KISSAgent.**

KISSClaw is a Python clone of [NanoClaw](https://github.com/nicholasgriffintn/NanoClaw), providing message routing, per-group isolation, scheduled tasks, and agent execution. It acts as an orchestration layer that connects messaging channels (e.g., console, WhatsApp) to an LLM-powered agent, managing conversations across multiple groups with independent context.

______________________________________________________________________

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     KissClawOrchestrator                        │
│  (orchestrator.py — central coordinator)                        │
│                                                                 │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐  │
│  │ Channel   │   │ GroupQueue│   │    Task   │   │    IPC    │  │
│  │ (console, │   │  (group_  │   │ Scheduler │   │  Watcher  │  │
│  │  etc.)    │   │ queue.py) │   │  (task_   │   │ (ipc.py)  │  │
│  │           │   │           │   │ sched.py) │   │           │  │
│  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘  │
│        │               │               │               │        │
│        ▼               ▼               ▼               ▼        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  KissClawDB (db.py)                      │   │
│  │        SQLite: messages, groups, tasks, state            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                │
│                                ▼                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            AgentRunner (agent_runner.py)                 │   │
│  │     Formats prompt → calls KISSAgent or mock fn          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                │
│                                ▼                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Router (router.py)                             │   │
│  │  format_messages() — XML inbound                         │   │
│  │  format_outbound() — strip <internal> tags               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. `KissClawOrchestrator` — *orchestrator.py*

The central coordinator. It wires together every subsystem and runs the main message loop.

**Responsibilities:**

- Registers groups and manages their per-group directories
- Polls for new messages via a background thread (`poll_interval`, default 2s)
- Enqueues messages into the `GroupQueue` for concurrent processing
- Delegates agent invocation to `AgentRunner`
- Sends replies back through the `Channel`
- Saves/loads persistent state (message cursors, sessions) through the DB
- Recovers pending messages on startup

**Key methods:**
| Method | Purpose |
|---|---|
| `start()` / `stop()` | Lifecycle — boots all subsystems |
| `register_group(jid, group)` | Registers a chat group for monitoring |
| `inject_message(msg)` | Programmatically injects a message (testing / API use) |
| `poll_messages_once()` | One poll cycle — fetches new messages, enqueues processing |
| `recover_pending_messages()` | Re-enqueues any unprocessed messages after restart |

### 2. `KissClawDB` — *db.py*

A SQLite-backed persistence layer (in-memory by default for testing).

**Tables:**
| Table | Purpose |
|---|---|
| `messages` | All incoming messages (indexed by timestamp) |
| `chats` | Chat metadata (JID, name, last message time) |
| `registered_groups` | Groups the bot monitors (JID → name, folder, trigger) |
| `scheduled_tasks` | Cron/interval/one-shot scheduled tasks |
| `task_run_logs` | Execution history for scheduled tasks |
| `router_state` | Key-value store for message cursors |
| `sessions` | Per-group session IDs |

### 3. `GroupQueue` — *group_queue.py*

A per-group message queue with global concurrency limiting.

**Design:**

- Each group gets its own logical queue so messages are processed in order
- A global semaphore (`max_concurrent`, default 5) limits how many groups run agents simultaneously
- Work items are dispatched to background threads — `enqueue_message_check()` returns immediately
- Automatic retry with exponential backoff (up to 5 retries, base delay 5s)
- Deduplication: the same task ID won't be enqueued twice

**Flow:**

```
enqueue_message_check("group_jid")
  │
  ├── slot available? ──yes──► spawn worker thread → _process_messages_fn(jid)
  │                                                      │
  │                                                      ├── success → reset retries, drain next
  │                                                      └── failure → schedule_retry (exp backoff)
  │
  └── at capacity? ──yes──► add to _waiting list (FIFO)
                             (drained when a slot frees)
```

### 4. `AgentRunner` — *agent_runner.py*

Builds a prompt and invokes the LLM agent.

**Prompt construction:**

1. Loads per-group memory from `{groups_dir}/{folder}/MEMORY.md`
1. Fills the `AGENT_SYSTEM_PROMPT` template with assistant name, group name, memory, and XML-formatted messages
1. Calls either:
   - A user-provided `agent_fn(prompt) → str` (for testing/custom agents), or
   - `KISSAgent.run(...)` with the configured model (default: `claude-sonnet-4-5`)

**Returns:** `AgentOutput(status, result, error)`

### 5. `Router` — *router.py*

Handles message formatting in both directions.

- **`format_messages(messages)`** — Converts a list of `Message` objects into XML:
  ```xml
  <messages>
    <message sender="Alice" timestamp="2024-01-01T00:00:01Z">Hello!</message>
    <message sender="Bob" timestamp="2024-01-01T00:00:02Z">Hi there</message>
  </messages>
  ```
- **`format_outbound(raw_text)`** — Strips `<internal>...</internal>` reasoning tags before sending to the user
- **`escape_xml(s)`** — Escapes `<`, `>`, `&`, `"` for safe XML embedding

### 6. `TaskScheduler` — *task_scheduler.py*

Polls for due scheduled tasks and executes them.

**Schedule types:**
| Type | `schedule_value` | Behavior |
|---|---|---|
| `cron` | Cron expression (e.g., `0 9 * * *`) | Repeats on cron schedule (requires `croniter`) |
| `interval` | Milliseconds (e.g., `86400000`) | Repeats every N ms |
| `once` | ISO datetime | Runs once, then status → `completed` |

**Execution flow:**

1. `poll_once()` queries `get_due_tasks(now)`
1. For each due task, creates a synthetic `Message` from the task prompt
1. Runs the agent via `run_agent()`
1. Sends the result to the task's `chat_jid`
1. Computes `next_run` (or marks `completed` for one-shot tasks)
1. Logs the run in `task_run_logs`

### 7. `IpcWatcher` — *ipc.py*

Watches the filesystem for inter-process communication files — allowing agents or external scripts to send messages and manage tasks.

**Directory structure:**

```
{data_dir}/ipc/
  {group_folder}/
    messages/          ← JSON files to send messages
    tasks/             ← JSON files to schedule/pause/resume/cancel tasks
  errors/              ← Failed files moved here
```

**Security model:**

- The **main** group can send messages to any group and register new groups
- Non-main groups can only send messages to their own JID
- Unauthorized cross-group messages are blocked and logged

**Supported IPC commands:**
| `type` | Fields | Effect |
|---|---|---|
| `message` | `chatJid`, `text` | Sends a message |
| `schedule_task` | `targetJid`, `prompt`, `schedule_type`, `schedule_value` | Creates a scheduled task |
| `pause_task` | `taskId` | Pauses a task |
| `resume_task` | `taskId` | Resumes a task |
| `cancel_task` | `taskId` | Deletes a task |
| `register_group` | `jid`, `name`, `folder`, `trigger` | Registers a new group (main only) |

### 8. `Channel` — *channels/base.py* & *channels/console.py*

Abstract interface for messaging backends.

**`Channel` (ABC):**

- `connect()` / `disconnect()` — Lifecycle
- `send_message(jid, text)` — Send outbound message
- `is_connected()` / `owns_jid(jid)` — Status checks
- Callbacks: `set_on_message()`, `set_on_chat_metadata()`

**`ConsoleChannel`** — An in-memory implementation for testing. Messages can be injected programmatically and sent messages are recorded in a list.

### 9. `Types` — *types.py*

Plain dataclasses used throughout:

| Type | Purpose |
|---|---|
| `Message` | A chat message (id, chat_jid, sender, content, timestamp, ...) |
| `RegisteredGroup` | A monitored group (name, folder, trigger, requires_trigger) |
| `ScheduledTask` | A scheduled task definition |
| `TaskRunLog` | Execution log entry for a task run |
| `AgentOutput` | Result of an agent invocation (status + result/error) |
| `ChatInfo` | Chat metadata (jid, name, last_message_time) |

### 10. `KissClawConfig` — *config.py*

Configuration dataclass with sensible defaults:

| Field | Default | Description |
|---|---|---|
| `assistant_name` | `"Andy"` | Bot name, used in trigger pattern |
| `trigger_pattern` | `^@Andy\b` | Regex to detect trigger in non-main groups |
| `poll_interval` | `2.0s` | Message polling frequency |
| `scheduler_poll_interval` | `60.0s` | Task scheduler polling frequency |
| `ipc_poll_interval` | `1.0s` | IPC file watcher frequency |
| `max_concurrent_agents` | `5` | Max simultaneous agent invocations |
| `model_name` | `"claude-sonnet-4-5"` | LLM model for KISSAgent |
| `max_steps` | `15` | Max agent steps per invocation |
| `max_budget` | `10.0` | Max budget per invocation |
| `idle_timeout` | `1800.0s` | 30 min idle timeout |

______________________________________________________________________

## Pseudocode

```
ORCHESTRATOR.start():
    load_state(last_timestamp, last_agent_timestamp, sessions) from DB
    create directories: data_dir, groups_dir, store_dir
    channel.connect()
    scheduler.start()                              // background thread: poll for due tasks
    ipc_watcher.start()                            // background thread: watch for IPC files
    recover_pending_messages()                      // re-enqueue any unprocessed messages
    spawn message_loop thread                       // polls every poll_interval (2s)

ORCHESTRATOR.poll_messages_once():
    messages, new_ts = DB.get_new_messages(registered_jids, since=last_timestamp)
    if no messages: return 0
    last_timestamp = new_ts; save_state()
    group messages by chat_jid
    for each (chat_jid, group_msgs):
        if non-main group AND requires_trigger:
            if no message matches trigger_pattern (e.g. "@Andy"): skip
        group_queue.enqueue_message_check(chat_jid)
    return count of enqueued groups

GROUP_QUEUE.enqueue_message_check(jid):
    if group already active: mark pending_messages = true; return
    if active_count >= max_concurrent: add to waiting list (FIFO); return
    acquire slot (active_count++); spawn worker thread → worker_messages(jid)

GROUP_QUEUE.worker_messages(jid):
    success = orchestrator.process_group_messages(jid)
    if success: reset retry_count
    else:       schedule_retry(jid)                // exponential backoff, max 5 retries
    release_and_drain(jid)                         // free slot, start next waiting group

GROUP_QUEUE.release_and_drain(jid):
    state.active = false; active_count--
    if group has pending_tasks:  pop and dispatch next task
    elif group has pending_msgs: dispatch message processing
    else: pop next group from global waiting list and dispatch

ORCHESTRATOR.process_group_messages(chat_jid):
    group = registered_groups[chat_jid]
    missed = DB.get_messages_since(chat_jid, last_agent_timestamp[chat_jid])
    if no missed messages: return true
    if non-main AND requires_trigger AND no trigger found: return true
    formatted_xml = ROUTER.format_messages(missed)
    save cursor: last_agent_timestamp[chat_jid] = missed[-1].timestamp
    output = AGENT_RUNNER.run_agent(group_name, group_folder, formatted_xml)
    if error: rollback cursor; return false
    if output.result:
        text = ROUTER.format_outbound(output.result)   // strip <internal> tags
        channel.send_message(chat_jid, text)
    return true

AGENT_RUNNER.run_agent(group_name, group_folder, formatted_messages):
    memory = read("{groups_dir}/{group_folder}/MEMORY.md") or ""
    prompt = SYSTEM_PROMPT.format(assistant_name, group_name, memory, formatted_messages)
    if agent_fn provided: return agent_fn(prompt)       // mock/custom agent
    else: return KISSAgent.run(model=model_name, prompt=prompt)

ROUTER.format_messages(messages):                       // list[Message] → XML string
    return "<messages>" + for each m: "<message sender=.. ts=..>{content}</message>" + "</messages>"

ROUTER.format_outbound(raw_text):                       // strip internal reasoning
    return strip_regex("<internal>...</internal>", raw_text).strip()

TASK_SCHEDULER.poll_once():                             // runs every scheduler_poll_interval (60s)
    due_tasks = DB.get_due_tasks(now)
    for each task in due_tasks:
        if task.status != "active": skip
        synthetic_msg = Message(sender="scheduler", content=task.prompt)
        output = AGENT_RUNNER.run_agent(group_name, group_folder, format([synthetic_msg]))
        if output.result: send_message(task.chat_jid, format_outbound(result))
        DB.log_task_run(task_id, duration, status, result)
        next_run = compute_next_run(task.schedule_type, task.schedule_value)
        DB.update_task_after_run(task_id, next_run)     // once-tasks → status="completed"

IPC_WATCHER.poll_once():                                // runs every ipc_poll_interval (1s)
    for each group_dir in {data_dir}/ipc/:
        is_main = (group_dir.name == main_group_folder)
        for each JSON file in group_dir/messages/:
            if authorized(is_main, source_group, target_jid):
                send_message(target_jid, text)
            else: log "blocked unauthorized message"
            delete file (or move to errors/ on failure)
        for each JSON file in group_dir/tasks/:
            match data.type:
                "schedule_task"  → validate, compute next_run, DB.create_task()
                "pause_task"     → DB.update_task(status="paused")
                "resume_task"    → DB.update_task(status="active")
                "cancel_task"    → DB.delete_task()
                "register_group" → DB.set_registered_group() [main only]
            delete file (or move to errors/ on failure)
```

______________________________________________________________________

## Message Flow

```
 User sends "Hello"        Trigger check           Agent invocation
 to group chat              (non-main groups        
      │                      need @Andy)                  │
      ▼                         │                         ▼
 ┌─────────┐    poll      ┌─────────┐  enqueue   ┌──────────────┐
 │ Channel  │───────────►│Orchestr- │──────────►│  GroupQueue   │
 │          │            │  ator    │           │ (concurrent)  │
 └─────────┘            └─────────┘           └──────┬───────┘
                              │                       │
                              │                       ▼
                              │              ┌──────────────┐
                              │              │ AgentRunner   │
                              │              │  (KISSAgent)  │
                              │              └──────┬───────┘
                              │                     │
                              ▼                     ▼
                        ┌─────────┐         ┌──────────────┐
                        │   DB    │◄────────│   Router     │
                        │ (state) │         │ (format msg) │
                        └─────────┘         └──────┬───────┘
                                                    │
                                              strip <internal>
                                                    │
                                                    ▼
                                              Channel.send_message()
                                              → User sees reply
```

______________________________________________________________________

## Sample Session

Below is a complete programmatic session showing how to set up KISSClaw, register groups, send messages, and receive responses:

```python
import time
from kiss.agents.kissclaw import KissClawOrchestrator
from kiss.agents.kissclaw.config import KissClawConfig
from kiss.agents.kissclaw.db import KissClawDB
from kiss.agents.kissclaw.channels.console import ConsoleChannel
from kiss.agents.kissclaw.types import Message, RegisteredGroup

# --- 1. Configure ---
config = KissClawConfig(
    assistant_name="Andy",
    poll_interval=0.1,       # fast polling for demo
    model_name="claude-sonnet-4-5",
)

# --- 2. Set up components ---
db = KissClawDB()                    # in-memory DB (pass a path for persistence)
channel = ConsoleChannel()           # in-memory channel for testing

# Mock agent function (replace with None to use real KISSAgent)
def mock_agent(prompt):
    if "weather" in prompt.lower():
        return "<internal>User asked about weather</internal>It's sunny and 72°F today!"
    return "Hello! How can I help you?"

# --- 3. Create orchestrator ---
orch = KissClawOrchestrator(
    config=config,
    db=db,
    channel=channel,
    agent_fn=mock_agent,  # remove this to use real LLM
)

# --- 4. Register groups ---
# Main group: no trigger needed, bot responds to all messages
orch.register_group("main-chat-001", RegisteredGroup(
    name="Main",
    folder="main",
    trigger="@Andy",
    added_at="2024-01-01T00:00:00Z",
    requires_trigger=False,
))

# Family group: requires "@Andy" trigger to respond
orch.register_group("family-chat-002", RegisteredGroup(
    name="Family",
    folder="family",
    trigger="@Andy",
    added_at="2024-01-01T00:00:00Z",
    requires_trigger=True,
))

# --- 5. Load state and start ---
orch.load_state()

# --- 6. Inject messages ---
# Message to main group (no trigger needed)
orch.inject_message(Message(
    id="msg-001",
    chat_jid="main-chat-001",
    sender="user123",
    sender_name="Alice",
    content="What's the weather like?",
    timestamp="2024-06-15T10:00:01Z",
))

# Message to family group WITHOUT trigger (will be ignored)
orch.inject_message(Message(
    id="msg-002",
    chat_jid="family-chat-002",
    sender="user456",
    sender_name="Bob",
    content="Anyone home?",
    timestamp="2024-06-15T10:00:02Z",
))

# Message to family group WITH trigger (will be processed)
orch.inject_message(Message(
    id="msg-003",
    chat_jid="family-chat-002",
    sender="user456",
    sender_name="Bob",
    content="@Andy what's for dinner?",
    timestamp="2024-06-15T10:00:03Z",
))

# --- 7. Poll and process ---
count = orch.poll_messages_once()
print(f"Enqueued {count} group(s) for processing")
# Output: Enqueued 2 group(s) for processing
#   (main-chat-001 and family-chat-002; the no-trigger message to family was filtered)

time.sleep(0.5)  # wait for async processing

# --- 8. Check responses ---
for jid, text in channel.get_sent_messages():
    print(f"  → [{jid}] {text}")
# Output:
#   → [main-chat-001] It's sunny and 72°F today!
#   → [family-chat-002] Hello! How can I help you?
#
# Note: the <internal> tag was stripped from the weather response!

# --- 9. Cleanup ---
orch.stop()
db.close()
```

### Using the Real KISSAgent (no mock)

```python
# Just omit agent_fn — the orchestrator will use KISSAgent automatically:
orch = KissClawOrchestrator(
    config=KissClawConfig(
        assistant_name="Andy",
        model_name="claude-sonnet-4-5",
        max_steps=15,
        max_budget=10.0,
    ),
    db=KissClawDB("./kissclaw_data/kissclaw.db"),
    channel=ConsoleChannel(),
    # no agent_fn → uses KISSAgent
)
```

### Running as a Long-Lived Service

```python
orch = KissClawOrchestrator(config=config, db=db, channel=channel)
orch.register_group("main-jid", main_group)
orch.start()  # starts message loop, scheduler, and IPC watcher in background threads

# ... runs until stopped ...

orch.stop()
```

### Scheduling Tasks via IPC

Write a JSON file to schedule a recurring task:

```bash
# Create IPC directory
mkdir -p kissclaw_data/ipc/main/tasks

# Schedule a daily summary task
cat > kissclaw_data/ipc/main/tasks/daily-summary.json << 'EOF'
{
    "type": "schedule_task",
    "targetJid": "main-chat-001",
    "prompt": "Give me a summary of today's important news",
    "schedule_type": "interval",
    "schedule_value": "86400000",
    "context_mode": "isolated"
}
EOF
# The IPC watcher will pick this up within 1 second (ipc_poll_interval)
```

______________________________________________________________________

## Per-Group Isolation

Each registered group gets:

- Its own **folder** under `{groups_dir}/{folder}/`
- An optional **`MEMORY.md`** file that is injected into every agent prompt for that group
- Independent **message cursors** — the agent processes each group's messages separately
- **Trigger filtering** — non-main groups require `@{assistant_name}` to activate the agent

This ensures conversations in one group never leak context to another.

______________________________________________________________________

## File Layout

```
kissclaw_data/
├── groups/
│   ├── main/
│   │   └── MEMORY.md          ← persistent memory for main group
│   └── family/
│       └── MEMORY.md          ← persistent memory for family group
├── ipc/
│   ├── main/
│   │   ├── messages/          ← outbound message requests
│   │   └── tasks/             ← task scheduling commands
│   └── errors/                ← failed IPC files
└── store/                     ← general storage
```
