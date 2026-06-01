# Agentic Workflow System — KISS Framework + Obsidian Knowledge Graph

> A comprehensive blueprint for building a planner-driven multi-agent workflow using the **KISS Sorcar framework** for orchestration, with an **Obsidian vault** as the unified knowledge database and graph view for all research outputs.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Schematic](#2-architecture-schematic)
3. [Why KISS Framework](#3-why-kiss-framework)
4. [Agent Taxonomy & Responsibilities](#4-agent-taxonomy--responsibilities)
5. [Planner Agent — Deep Dive](#5-planner-agent--deep-dive)
6. [Workflow Execution Patterns](#6-workflow-execution-patterns)
7. [Obsidian Vault as Knowledge Database](#7-obsidian-vault-as-knowledge-database)
8. [Implementation Guide](#8-implementation-guide)
9. [Data Schemas](#9-data-schemas)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Vault Organizer Agent — Token Optimization](#11-vault-organizer-agent--token-optimization)
12. [MCP Server Integration — Evaluation & Recommendation](#12-mcp-server-integration--evaluation--recommendation)
13. [Agent Definitions, Model Selection & Token Optimization](#13-agent-definitions-model-selection--token-optimization)
14. [Synthesizer Web App Generation](#14-synthesizer-web-app-generation)
15. [Research Agent MCP Integration](#15-research-agent-mcp-integration)
16. [Agent-Specific Add-Ons Registry](#16-agent-specific-add-ons-registry)

---

## 1. System Overview

The system follows an **Orchestrator-Workers** pattern built entirely on the **KISS Sorcar framework**, with **MCP (Model Context Protocol)** as the universal integration layer between agents and all external systems. A central **Planner Agent** (a `KISSAgent`) receives incoming tasks, decomposes them into categorized subtasks, delegates each to specialized **Sub-Agents** (also `KISSAgent` instances). All agents interact with external resources — the Obsidian vault, academic databases, web search engines, and document analysis tools — exclusively through **MCP servers**, creating a standardized, token-efficient, and reproducible data pipeline. Every artifact is written as a **Markdown file into an Obsidian vault** — creating an interconnected knowledge graph with bidirectional links, tags, and metadata.

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
│                  (task description / query)                       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │     PLANNER AGENT      │
              │     (KISSAgent)        │
              │                        │
              │  1. Categorize task     │
              │  2. Decompose subtasks  │
              │  3. Assign agents       │
              │  4. Define dependencies │
              │  5. Set success criteria│
              └────────┬───────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ Research  │  │ Writing  │  │ Analysis │  ... (N KISSAgents)
   │ KISSAgent │  │ KISSAgent│  │ KISSAgent│
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │              │              │
        ▼              ▼              ▼
   ┌──────────────────────────────────────────────────────────┐
   │              MCP INTEGRATION LAYER                       │
   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
   │  │  MCPVault    │ │ RivalSearch  │ │ PapersFlow /     │ │
   │  │  (vault I/O, │ │ (web, social,│ │ OpenAlex /       │ │
   │  │  search,     │ │ news, data-  │ │ paper-fetch      │ │
   │  │  frontmatter,│ │ sets, OCR)   │ │ (academic DBs)   │ │
   │  │  tags)       │ │              │ │                  │ │
   │  └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘ │
   └─────────┼────────────────┼──────────────────┼───────────┘
             │                │                  │
             ▼                ▼                  ▼
   ┌──────────────────────────────────────┐   External
   │        OBSIDIAN VAULT                │   APIs &
   │   (Markdown files with [[links]],   │   Databases
   │    YAML frontmatter, #tags)          │   (240M+
   │                                      │   scholarly
   │   📊 Graph View = Knowledge Map     │   works,
   │   🔍 Search = Full-text + tags      │   474M+
   │   🔗 Links = Automatic lineage      │   papers)
   └──────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │   SYNTHESIZER AGENT    │
              │     (KISSAgent)        │
              │                        │
              │  Reads vault via MCP    │
              │  Follows [[links]]      │
              │  Produces final report  │
              │  + interactive web app  │
              └────────────────────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │     FINAL OUTPUT       │
              │  (saved to vault +     │
              │   web app + returned   │
              │   to user)             │
              └────────────────────────┘
```

---

## 2. Architecture Schematic

### 2.1 Four-Layer Architecture (KISS + MCP + Obsidian)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER (KISS Framework)                      │
│                                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │   Planner    │  │   Router     │  │  Evaluator   │  │  Synthesizer  │   │
│  │  KISSAgent   │  │  KISSAgent   │  │  KISSAgent   │  │  KISSAgent    │   │
│  │              │  │  (classify   │  │  (quality    │  │  (combine     │   │
│  │  decompose   │  │   & route)   │  │   gate)      │  │   outputs)    │   │
│  │  & assign    │  │              │  │              │  │               │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────────┘   │
│                                                                             │
│  Parallel execution via RelentlessAgent._run_tasks_parallel()              │
│  Auto-continuation via RelentlessAgent.perform_task()                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                    EXECUTION LAYER (KISS KISSAgents)                        │
│                                                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │  Research   │  │  Writing   │  │  Code      │  │  Analysis  │  ...      │
│  │  KISSAgent  │  │  KISSAgent │  │  KISSAgent │  │  KISSAgent │           │
│  │  Opus 4.7   │  │  Sonnet 4.6│  │  Opus 4.7  │  │  Flash 2.5 │           │
│  │             │  │            │  │            │  │            │           │
│  │  Tools:     │  │  Tools:    │  │  Tools:    │  │  Tools:    │           │
│  │  - MCP*     │  │  - MCP*    │  │  - Bash    │  │  - MCP*    │           │
│  │  - WebUse   │  │  - Read    │  │  - Read    │  │  - Bash    │           │
│  │  - Read     │  │  - Write   │  │  - Write   │  │  - Read    │           │
│  │  - Write    │  │  - Edit    │  │  - Edit    │  │  - Write   │           │
│  │  - Bash     │  │            │  │            │  │            │           │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘           │
│         │               │               │               │                   │
│  * MCP = agent calls MCP tools instead of raw file I/O / web scraping      │
│                                                                             │
├─────────┼───────────────┼───────────────┼───────────────┼───────────────────┤
│         ▼               ▼               ▼               ▼                   │
│                    MCP INTEGRATION LAYER (Protocol Bridge)                   │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐       │
│  │  MCPVault         │  │  RivalSearchMCP   │  │  PapersFlow /       │       │
│  │  (vault I/O)      │  │  (web + academic) │  │  OpenAlex / Fetch   │       │
│  │                   │  │                   │  │  (academic DBs)     │       │
│  │  14 tools:        │  │  10 tools:        │  │  39+ tools:         │       │
│  │  read, write,     │  │  5 search engines │  │  474M+ papers       │       │
│  │  search (BM25),   │  │  9 social media   │  │  240M+ works        │       │
│  │  frontmatter,     │  │  5 news sources   │  │  DOI→PDF resolve    │       │
│  │  tags, batch,     │  │  5 academic DBs   │  │  citation graphs    │       │
│  │  vault stats      │  │  4 dataset hubs   │  │  journal presets    │       │
│  │                   │  │  + OCR, GitHub     │  │  trend analysis     │       │
│  └────────┬──────────┘  └────────┬──────────┘  └─────────┬───────────┘       │
│           │                      │                       │                   │
│  Protocol: MCP (Stdio transport for local, Streamable HTTP for remote)      │
│  All servers MIT licensed · No API keys required for Tier 1                 │
│                                                                             │
├───────────┼──────────────────────┼───────────────────────┼───────────────────┤
│           ▼                      ▼                       ▼                   │
│                    PERSISTENCE LAYER (Obsidian Vault + External APIs)        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    OBSIDIAN VAULT                                 │       │
│  │                                                                  │       │
│  │  📁 runs/           — Per-run execution artifacts                │       │
│  │  📁 research/       — Research outputs by topic                  │       │
│  │  📁 analysis/       — Analysis and comparison docs               │       │
│  │  📁 reports/        — Final synthesized reports                  │       │
│  │  📁 agents/         — Agent run logs and metadata                │       │
│  │  📁 templates/      — Reusable templates                        │       │
│  │  📁 sources/        — Source material and references             │       │
│  │                                                                  │       │
│  │  Features:                                                       │       │
│  │  • [[Wikilinks]] for bidirectional linking                      │       │
│  │  • YAML frontmatter for structured metadata                     │       │
│  │  • #tags for categorization                                     │       │
│  │  • Graph View for visual knowledge map                          │       │
│  │  • Full-text search across all outputs                          │       │
│  │  • Dataview plugin for SQL-like queries over frontmatter        │       │
│  │  • Canvas for visual workflow planning                          │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │              EXTERNAL APIs (accessed via MCP servers)             │       │
│  │                                                                  │       │
│  │  OpenAlex (240M+ works, CC0) · CrossRef (150M+, CC0)           │       │
│  │  PubMed (36M+, public domain) · arXiv (2.4M+, CC0)             │       │
│  │  Semantic Scholar (200M+) · Unpaywall (40M+ OA articles)        │       │
│  │  Europe PMC (40M+) · Google/Bing/DuckDuckGo (web search)       │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key architectural change:** No agent directly reads/writes the Obsidian vault filesystem or
scrapes websites. All external interactions pass through the **MCP Integration Layer**, which
provides structured JSON responses, BM25-ranked search, selective field retrieval, and built-in
40-60% token savings compared to raw file I/O. The MCP layer is the **single source of truth**
for all data access — enforcing consistency, token efficiency, and reproducibility across agents.

### 2.2 Data Flow Sequence (MCP-Integrated)

```
USER ──► Planner KISSAgent
              │
              ├──► [1] Categorize (research? code? writing? analysis? mixed?)
              ├──► [2] Decompose into subtask DAG
              ├──► [3] For each subtask:
              │         ├── Select KISSAgent type + MCP servers to attach
              │         ├── Define input context via MCPVault search/read
              │         ├── Set expected output format (MD with frontmatter)
              │         └── Set success criteria
              │
              ├──► [4] Execute (parallel via run_parallel, sequential per DAG)
              │         │
              │         ├── Independent subtasks → _run_tasks_parallel()
              │         └── Dependent subtasks → sequential KISSAgent.run()
              │         │
              │         │  ALL agent ↔ data interactions go through MCP:
              │         │
              │         │  Research Agent:
              │         │    ├── RivalSearchMCP.web_search() → structured JSON
              │         │    ├── PapersFlow.search_papers() → DOIs, abstracts
              │         │    ├── OpenAlex.search_works() → curated results
              │         │    ├── MCPVault.search_notes() → existing knowledge
              │         │    ├── WebUseTool (fallback for uncovered sources)
              │         │    └── MCPVault.write_note() → save to vault
              │         │
              │         │  Writing/Analysis/Code Agents:
              │         │    ├── MCPVault.read_note() / search_notes() → context
              │         │    ├── MCPVault.get_frontmatter() → metadata-only
              │         │    └── MCPVault.write_note() → save output
              │
              ├──► [5] Each agent writes output via MCPVault.write_note():
              │         ├── MD file with YAML frontmatter (metadata)
              │         ├── [[Wikilinks]] to related notes (lineage)
              │         ├── #tags for categorization
              │         └── Backlinks auto-created by Obsidian
              │
              ├──► [6] Evaluator KISSAgent reads output via MCPVault, checks criteria
              │         ├── PASS → MCPVault.update_frontmatter(status: complete)
              │         └── FAIL → MCPVault.update_frontmatter(status: failed), re-run (max 3)
              │
              └──► [7] Synthesizer KISSAgent reads all via MCPVault.search_notes()
                        ├── Summary MD written via MCPVault.write_note()
                        └── Web app generated via Quartz 5 build
```

---

## 3. Why KISS Framework

### 3.1 KISS Agent Architecture Maps Perfectly

The KISS framework provides everything needed for this agentic workflow:

| KISS Component | Role in Workflow |
|---|---|
| **`KISSAgent`** | Base for all agents — Planner, Research, Writing, Code, Analysis, Review, Synthesizer |
| **`KISSAgent.run()`** | Execute a single agent with model, prompt, tools, and budget control |
| **`RelentlessAgent`** | Auto-continuation for long tasks (planner orchestration loop) |
| **`RelentlessAgent._run_tasks_parallel()`** | Parallel execution of independent subtasks |
| **`UsefulTools`** (Read, Write, Edit, Bash) | Local file I/O tools (supplementary to MCP vault access) |
| **`WebUseTool`** | Browser automation for research agents (fallback when MCP doesn't cover a source) |
| **MCP Client Integration** | Every agent connects to MCP servers for vault I/O (MCPVault), research (RivalSearch, PapersFlow, OpenAlex), and document analysis — replacing raw file I/O with structured, token-efficient API calls |
| **`finish()`** | Structured result return with success/failure/continue status |
| **`prompt_refiner_agent()`** | Iterative prompt improvement based on agent trajectory |
| **Built-in trajectory saving** | Full audit trail in YAML — every agent run is recorded |

### 3.2 How Each KISS Component Is Used

```python
from kiss.core.kiss_agent import KISSAgent
from kiss.core.relentless_agent import RelentlessAgent

# 1. Planner Agent — uses RelentlessAgent for auto-continuation
class PlannerAgent(RelentlessAgent):
    """Orchestrates the entire workflow with auto-continuation."""
    
    def plan_and_execute(self, task: str, vault_path: str):
        tools = [self.categorize, self.decompose, self.delegate, self.synthesize]
        self.run(
            model_name="claude-sonnet-4-20250514",
            prompt_template=PLANNER_PROMPT,
            arguments={"task": task, "vault_path": vault_path},
            tools=tools,
            max_steps=100,
            max_budget=50.0,
            work_dir=vault_path,
        )

# 2. Worker Agents — use KISSAgent for single-run tasks
def run_research_agent(topic: str, vault_path: str) -> str:
    agent = KISSAgent("Research Agent")
    return agent.run(
        model_name="claude-sonnet-4-20250514",
        prompt_template=RESEARCH_PROMPT,
        arguments={"topic": topic, "vault_path": vault_path},
        tools=[Read, Write, Edit, Bash, go_to_url, get_page_content],
        max_steps=50,
        max_budget=5.0,
    )

# 3. Parallel execution — independent research tasks
results = planner._run_tasks_parallel(
    tasks=[
        "Research agentic AI frameworks and write to vault",
        "Research vector databases and write to vault",
        "Research workflow orchestration patterns and write to vault",
    ],
    max_workers=3,
)

# 4. Non-agentic evaluation — single-shot quality check  
evaluator = KISSAgent("Evaluator")
evaluation = evaluator.run(
    model_name="claude-sonnet-4-20250514",
    prompt_template=EVAL_PROMPT,
    arguments={"output_file": "research/frameworks.md"},
    is_agentic=False,  # Single generation, no tool loop
)
```

### 3.3 KISS vs Other Frameworks

| Feature | KISS Framework | LangGraph | CrewAI |
|---|---|---|---|
| **Agent definition** | Python class, 1 line | State graph + nodes | YAML config |
| **Tool attachment** | Pass list of callables | Bind to node | Declare in agent YAML |
| **Parallel execution** | Built-in `_run_tasks_parallel()` | Send API | Limited |
| **Auto-continuation** | Built-in `RelentlessAgent` | Manual checkpoint | None |
| **Budget control** | Built-in per-agent | Manual | None |
| **Trajectory logging** | Built-in YAML save | LangSmith integration | Basic |
| **Model switching** | Any model via `model_name` | LangChain models | LiteLLM |
| **Complexity** | ~500 LOC core | ~5000 LOC | ~10000 LOC |
| **Dependencies** | Minimal (yaml, model SDK) | Heavy (langchain) | Heavy |

**KISS wins**: Simplest possible agent framework. Each agent is a Python function call. No graphs, no YAML configs, no ceremony. Just `KISSAgent.run(model, prompt, tools)`.

---

## 4. Agent Taxonomy & Responsibilities

### 4.1 Orchestration Agents

| Agent | KISS Class | Role | Input | Output |
|---|---|---|---|---|
| **Planner** | `RelentlessAgent` | Categorize, decompose, orchestrate | User task | Execution plan JSON |
| **Router** | `KISSAgent` (non-agentic) | Classify task type | Ambiguous input | Category + workflow |
| **Evaluator** | `KISSAgent` (non-agentic) | Quality gate each output | Agent output + criteria | Pass/Fail + feedback |
| **Synthesizer** | `KISSAgent` | Combine all outputs | All vault files for run | Final report in vault |

### 4.2 Execution Agents (Workers)

| Agent | KISS Class | Tools | Vault Output |
|---|---|---|---|
| **Research Agent** | `KISSAgent` | WebUseTool, Read, Write, Bash | `research/{topic}.md` |
| **Writing Agent** | `KISSAgent` | Read, Write, Edit | `reports/{document}.md` |
| **Code Agent** | `KISSAgent` | Bash, Read, Write, Edit | `code/{module}.md` + code files |
| **Data Agent** | `KISSAgent` | Bash, Read, Write | `analysis/{dataset}.md` |
| **Review Agent** | `KISSAgent` | Read, Write | `reviews/{review}.md` |
| **Summary Agent** | `KISSAgent` (non-agentic) | — | `summaries/{summary}.md` |
| **Vault Organizer** | `KISSAgent` | MCP Tools (MCPVault), Read, Write, Edit, Bash | `_maintenance/{report}.md` |

### 4.3 Agent Definition in KISS

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class AgentDefinition:
    """Definition for a workflow agent backed by KISSAgent."""
    name: str                              # e.g., "research_agent"
    role: str                              # Human-readable role
    system_prompt: str                     # Behavioral instructions
    prompt_template: str                   # Task prompt with {placeholders}
    tools: list[Callable]                  # List of tool functions
    model_name: str = "claude-sonnet-4-20250514"
    is_agentic: bool = True                # False = single generation
    max_steps: int = 50                    # Per-run step limit
    max_budget: float = 5.0                # Per-run budget in USD
    max_retries: int = 3                   # Retries on evaluator rejection
    vault_output_folder: str = "research"  # Where in vault to write output
    tags: list[str] = field(default_factory=list)
```

---

## 5. Planner Agent — Deep Dive

### Phase 1: Task Categorization

```python
from enum import Enum

class TaskCategory(str, Enum):
    RESEARCH = "research"
    WRITING = "writing"
    CODE = "code"
    ANALYSIS = "analysis"
    REVIEW = "review"
    MIXED = "mixed"
```

### Phase 2: Task Decomposition (DAG)

```python
from pydantic import BaseModel
from datetime import datetime

class SubTask(BaseModel):
    id: str
    title: str
    description: str
    category: TaskCategory
    assigned_agent: str
    dependencies: list[str]           # IDs of prerequisite subtasks
    expected_output: str
    success_criteria: list[str]
    priority: int                     # 1 = highest
    estimated_tokens: int
    vault_output_path: str            # Where in vault to write result
    vault_links: list[str]            # [[Links]] to related vault notes

class ExecutionPlan(BaseModel):
    task_id: str
    original_task: str
    category: TaskCategory
    subtasks: list[SubTask]
    execution_order: list[list[str]]  # Groups (parallel within, sequential between)
    total_estimated_tokens: int
    vault_run_folder: str             # runs/{task_id}/
    created_at: datetime
```

### Phase 3: Context Assembly from Obsidian Vault

For each subtask, the planner assembles context by:
1. Reading the original user request
2. Reading outputs from completed dependency subtasks (via vault `[[links]]`)
3. Searching the vault for related past research (via tags and backlinks)
4. Passing relevant vault file contents as context to the agent

```python
def assemble_context(subtask: SubTask, vault_path: str) -> str:
    """Read dependency outputs and related vault notes for context."""
    context_parts = []
    
    # 1. Read outputs from completed dependencies
    for dep_id in subtask.dependencies:
        dep_file = f"{vault_path}/runs/{subtask.task_id}/{dep_id}.md"
        if os.path.exists(dep_file):
            content = Path(dep_file).read_text()
            context_parts.append(f"## Output from {dep_id}\n{content}")
    
    # 2. Search vault for related notes by tag
    for tag in subtask.vault_links:
        # Grep vault for notes containing this tag
        related = find_vault_notes_by_tag(vault_path, tag)
        for note_path in related[:5]:  # Top 5 related
            content = Path(note_path).read_text()
            context_parts.append(f"## Related: {note_path}\n{content}")
    
    return "\n\n---\n\n".join(context_parts)
```

### Phase 4: Execution via KISS

```python
def execute_plan(plan: ExecutionPlan, vault_path: str):
    """Execute the plan using KISS agents."""
    
    for group in plan.execution_order:
        if len(group) == 1:
            # Sequential: run single agent
            subtask = get_subtask(plan, group[0])
            run_agent_for_subtask(subtask, vault_path)
        else:
            # Parallel: use _run_tasks_parallel
            task_descriptions = []
            for subtask_id in group:
                subtask = get_subtask(plan, subtask_id)
                context = assemble_context(subtask, vault_path)
                task_descriptions.append(
                    f"{subtask.description}\n\nContext:\n{context}\n\n"
                    f"Write output to: {vault_path}/{subtask.vault_output_path}"
                )
            
            planner._run_tasks_parallel(
                tasks=task_descriptions,
                max_workers=len(group),
            )
```

### Phase 5: Synthesis

```python
def synthesize(plan: ExecutionPlan, vault_path: str) -> str:
    """Combine all subtask outputs into final report."""
    synthesizer = KISSAgent("Synthesizer")
    
    # Collect all output files
    all_outputs = []
    for subtask in plan.subtasks:
        output_path = f"{vault_path}/{subtask.vault_output_path}"
        if os.path.exists(output_path):
            all_outputs.append(Path(output_path).read_text())
    
    combined_context = "\n\n---\n\n".join(all_outputs)
    
    return synthesizer.run(
        model_name="claude-sonnet-4-20250514",
        prompt_template=SYNTHESIS_PROMPT,
        arguments={
            "task": plan.original_task,
            "all_outputs": combined_context,
            "vault_path": vault_path,
        },
        tools=[Read, Write, Edit],
        max_steps=30,
        max_budget=10.0,
    )
```

### Planner System Prompt

```
You are a Planner Agent in the KISS Sorcar framework. Your job is to take a
user's task and create a detailed execution plan, then orchestrate its execution
using KISS sub-agents.

All outputs are written to an Obsidian vault at {vault_path}. Every output
file must:
1. Have YAML frontmatter with metadata (task_id, agent, status, tags, date)
2. Use [[wikilinks]] to link to related notes and dependencies
3. Use #tags for categorization
4. Follow the vault folder structure

For each task you receive:
1. CATEGORIZE: Determine if this is research, writing, code, analysis,
   review, or a mix.
2. DECOMPOSE: Break it into the smallest meaningful subtasks that can be
   independently executed by a KISSAgent.
3. ASSIGN: For each subtask, select the best-suited agent type.
4. ORDER: Determine which subtasks can run in parallel (use
   _run_tasks_parallel) and which have dependencies (sequential).
5. CRITERIA: Define clear, measurable success criteria for each subtask.
6. LINK: Define which vault notes each subtask should link to.

Output your plan as a structured JSON ExecutionPlan.

Rules:
- Maximize parallelism: independent subtasks → _run_tasks_parallel()
- Each subtask = one KISSAgent.run() call
- Include a final synthesis subtask that reads all outputs and produces
  a [[linked]] report
- Be specific in subtask descriptions — KISSAgents work best with clear
  instructions
- Every output file must link back to the run index note
```

---

## 6. Workflow Execution Patterns (KISS Implementation)

### 6.1 Orchestrator-Workers (Primary)

```python
# Planner (RelentlessAgent) dynamically creates KISSAgent workers
planner = RelentlessAgent("Planner")
planner.run(
    model_name="claude-sonnet-4-20250514",
    prompt_template="{task}",
    arguments={"task": user_task},
    tools=[create_agent, run_agent, run_parallel, read_vault, write_vault],
    work_dir=vault_path,
)
```

### 6.2 Prompt Chaining

```python
# Sequential KISSAgent calls, each reading previous output from vault
outline = KISSAgent("Outliner").run(...)   # writes outline.md
draft = KISSAgent("Drafter").run(...)      # reads [[outline]], writes draft.md  
edited = KISSAgent("Editor").run(...)      # reads [[draft]], writes final.md
```

### 6.3 Parallelization

```python
# Built-in KISS parallel execution
results = planner._run_tasks_parallel(
    tasks=[
        "Research topic A, write to vault/research/topic_a.md",
        "Research topic B, write to vault/research/topic_b.md",
        "Research topic C, write to vault/research/topic_c.md",
    ],
    max_workers=3,
)
```

### 6.4 Evaluator-Optimizer

```python
# KISSAgent evaluator with retry loop
for attempt in range(max_retries):
    output = worker_agent.run(...)
    
    evaluation = KISSAgent("Evaluator").run(
        model_name="claude-sonnet-4-20250514",
        prompt_template=EVAL_PROMPT,
        arguments={"output": output, "criteria": criteria},
        is_agentic=False,  # Single-shot evaluation
    )
    
    if "PASS" in evaluation:
        break
    # Else: use prompt_refiner_agent() to improve and retry
```

---

## 7. Obsidian Vault as Knowledge Database

### 7.1 Why Obsidian Instead of SQLite + ChromaDB

| Feature | Obsidian Vault | SQLite + ChromaDB |
|---|---|---|
| **Human readability** | ✅ Open any note in Obsidian, VS Code, or browser | ❌ Requires queries |
| **Visual knowledge graph** | ✅ Built-in Graph View shows all connections | ❌ Need custom visualization |
| **Bidirectional links** | ✅ `[[wikilinks]]` + automatic backlinks | ❌ Manual foreign keys |
| **Search** | ✅ Full-text + tag + property search built-in | ⚠️ Separate text + vector search |
| **Metadata queries** | ✅ Dataview plugin = SQL over frontmatter | ✅ SQL queries |
| **Version control** | ✅ Git-native (plain text MD files) | ⚠️ Binary DB files |
| **Plugins** | ✅ 1800+ community plugins (Dataview, Templater, Canvas, etc.) | ❌ Custom code only |
| **Collaboration** | ✅ Obsidian Publish / Sync, or just Git | ❌ Custom sharing |
| **Setup** | ✅ Zero config — just a folder of .md files | ⚠️ Schema, migrations |
| **Portability** | ✅ Standard Markdown, works anywhere | ❌ Vendor-locked |
| **Agent integration** | ✅ Agents just Read/Write .md files (KISS tools already do this) | ⚠️ Need DB drivers |
| **Knowledge graph** | ✅ Automatic from `[[links]]` and tags | ❌ Must build manually |

**Key insight**: The KISS framework already has `Read`, `Write`, and `Edit` tools that work with files. An Obsidian vault is just a folder of Markdown files. **Zero additional tooling needed** — agents write Markdown with `[[links]]` and Obsidian renders the graph.

### 7.2 Vault Folder Structure

```
vault/                              ← This IS the Obsidian vault root
├── .obsidian/                      ← Obsidian config (themes, plugins, graph settings)
│   ├── app.json
│   ├── appearance.json
│   ├── graph.json                  ← Graph View settings (colors, groups, filters)
│   ├── plugins/
│   │   ├── dataview/               ← SQL-like queries over frontmatter
│   │   └── templater/              ← Template engine for consistent output
│   └── community-plugins.json
│
├── 📋 _index.md                    ← Master index with Dataview tables
├── 📋 _MOC.md                      ← Map of Content (manual curated overview)
│
├── 📁 runs/                        ← One folder per task execution
│   ├── 2025-05-27_agentic-research/
│   │   ├── _run-index.md           ← Run metadata, plan, links to all outputs
│   │   ├── st_01-research-frameworks.md
│   │   ├── st_02-research-databases.md
│   │   ├── st_03-analysis-comparison.md
│   │   ├── st_04-draft-report.md
│   │   ├── st_05-review.md
│   │   └── final-report.md         ← Synthesized output
│   └── 2025-05-28_code-review/
│       └── ...
│
├── 📁 research/                    ← Persistent research notes (cross-run)
│   ├── ai-agents/
│   │   ├── frameworks-overview.md
│   │   ├── langraph.md
│   │   ├── crewai.md
│   │   └── kiss-framework.md
│   ├── databases/
│   │   ├── vector-databases.md
│   │   └── obsidian-as-database.md
│   └── ...
│
├── 📁 analysis/                    ← Comparisons, evaluations
│   ├── framework-comparison-2025.md
│   └── ...
│
├── 📁 reports/                     ← Final deliverables
│   ├── agentic-ai-state-of-art.md
│   └── ...
│
├── 📁 agents/                      ← Agent run logs
│   ├── agent-log-2025-05-27-research-01.md
│   └── ...
│
├── 📁 sources/                     ← External source references
│   ├── anthropic-building-agents.md
│   ├── openai-agents-sdk.md
│   └── ...
│
└── 📁 templates/                   ← Obsidian/Templater templates
    ├── research-note.md
    ├── run-index.md
    ├── agent-log.md
    └── source-reference.md
```

### 7.3 YAML Frontmatter Schema (Structured Metadata)

Every file in the vault has YAML frontmatter that Obsidian indexes and Dataview can query:

#### Research Note Frontmatter

```yaml
---
type: research
task_id: run_20250527_001
subtask_id: st_01
agent: research_agent
model: claude-sonnet-4-20250514
status: completed          # pending | running | completed | failed
attempt: 1
created: 2025-05-27T12:01:00
completed: 2025-05-27T12:03:45
tokens_used: 7500
cost_usd: 0.023
evaluator_score: 0.92
tags:
  - research
  - ai-agents
  - frameworks
  - 2025
sources:
  - "[[anthropic-building-agents]]"
  - "[[crewai-docs]]"
depends_on:
  - "[[st_00-plan]]"
feeds_into:
  - "[[st_03-analysis-comparison]]"
  - "[[st_04-draft-report]]"
---
```

#### Run Index Frontmatter

```yaml
---
type: run-index
task_id: run_20250527_001
title: "Research the state of agentic AI workflows"
category: mixed
status: completed
created: 2025-05-27T12:00:00
completed: 2025-05-27T12:15:00
total_tokens: 33000
total_cost_usd: 0.14
subtask_count: 5
tags:
  - run
  - agentic-ai
  - research
---
```

### 7.4 Wikilinks for Knowledge Graph

The **power** of Obsidian is in `[[wikilinks]]`. Each agent output links to:

1. **Dependencies**: `Depends on: [[st_01-research-frameworks]]`
2. **Sources**: `Based on: [[anthropic-building-agents]]`
3. **Related research**: `See also: [[crewai]], [[langraph]]`
4. **Run index**: `Part of: [[2025-05-27_agentic-research/_run-index]]`
5. **Topics**: `Topics: [[ai-agents]], [[workflow-orchestration]]`

Obsidian automatically creates **backlinks** for all of these, so:
- Opening `[[crewai]]` shows every research note that references it
- The **Graph View** shows the entire web of connections
- Clicking through links lets you trace the full lineage of any conclusion

Example output file:

```markdown
---
type: research
task_id: run_20250527_001
subtask_id: st_01
agent: research_agent
status: completed
tags: [research, ai-agents, frameworks]
---

# Research: Agentic AI Frameworks 2025

> Part of run: [[2025-05-27_agentic-research/_run-index]]
> Depends on: (none — first subtask)
> Feeds into: [[st_03-analysis-comparison]], [[st_04-draft-report]]

## Frameworks Investigated

### 1. [[KISS Framework]]
- **Architecture**: KISSAgent + RelentlessAgent for auto-continuation
- **Key feature**: Minimal code, maximum capability
- See: [[kiss-framework]] for detailed analysis

### 2. [[LangGraph]]  
- **Architecture**: State graph with nodes and edges
- **Key feature**: Send API for dynamic parallelism
- Source: [[langraph-docs]]

### 3. [[CrewAI]]
- **Architecture**: YAML-configured crews with hierarchical planning
- **Key feature**: Built-in planning agent
- Source: [[crewai-docs]]

## Sources
- [[anthropic-building-agents]] — Anthropic's guide to agentic patterns
- [[openai-agents-sdk]] — OpenAI's agent SDK documentation

#research #ai-agents #frameworks #2025
```

### 7.5 Graph View Configuration

Obsidian's Graph View renders all `[[links]]` as a visual network. Configure it for the workflow:

```json
// .obsidian/graph.json
{
  "collapse-filter": false,
  "search": "",
  "showTags": true,
  "showAttachments": false,
  "hideUnresolved": false,
  "showOrphans": false,
  "collapse-color-groups": false,
  "colorGroups": [
    { "query": "tag:#run", "color": { "a": 1, "rgb": 2605547 } },
    { "query": "tag:#research", "color": { "a": 1, "rgb": 52326 } },
    { "query": "tag:#analysis", "color": { "a": 1, "rgb": 16750848 } },
    { "query": "tag:#report", "color": { "a": 1, "rgb": 10040115 } },
    { "query": "tag:#source", "color": { "a": 1, "rgb": 8421504 } },
    { "query": "tag:#agent-log", "color": { "a": 1, "rgb": 16776960 } }
  ],
  "collapse-display": false,
  "showArrow": true,
  "textFadeMultiplier": 0,
  "nodeSizeMultiplier": 1,
  "lineSizeMultiplier": 1,
  "collapse-forces": true,
  "centerStrength": 0.5,
  "repelStrength": 10,
  "linkStrength": 1,
  "linkDistance": 250
}
```

**Color coding in Graph View:**
- 🔵 **Blue** = Research notes
- 🟠 **Orange** = Analysis documents
- 🟢 **Green** = Run indices
- 🟣 **Purple** = Final reports
- ⚪ **Gray** = Source references
- 🟡 **Yellow** = Agent logs

### 7.6 Dataview Queries (SQL-like over Frontmatter)

Install the **Dataview** community plugin to query structured frontmatter:

#### All Research Notes for a Run

````markdown
```dataview
TABLE agent, status, tokens_used, cost_usd, evaluator_score
FROM "runs/2025-05-27_agentic-research"
WHERE type = "research"
SORT created ASC
```
````

#### All Runs by Cost

````markdown
```dataview
TABLE title, category, total_cost_usd, subtask_count, status
FROM "runs"
WHERE type = "run-index"
SORT total_cost_usd DESC
```
````

#### All Notes Linking to a Topic

````markdown
```dataview
LIST
FROM [[ai-agents]]
SORT created DESC
```
````

#### Failed Subtasks Across All Runs

````markdown
```dataview
TABLE task_id, agent, attempt, evaluator_score
FROM "runs"
WHERE status = "failed"
SORT created DESC
```
````

### 7.7 Obsidian Templates (Templater Plugin)

#### Research Note Template (`templates/research-note.md`)

```markdown
---
type: research
task_id: <% tp.frontmatter.task_id %>
subtask_id: <% tp.frontmatter.subtask_id %>
agent: research_agent
model: claude-sonnet-4-20250514
status: pending
attempt: 1
created: <% tp.date.now("YYYY-MM-DDTHH:mm:ss") %>
completed: 
tokens_used: 
cost_usd: 
evaluator_score: 
tags: [research]
sources: []
depends_on: []
feeds_into: []
---

# Research: <% tp.frontmatter.title %>

> Part of run: [[<% tp.frontmatter.run_folder %>/_run-index]]

## Findings

## Sources

## Related Notes
```

#### Run Index Template (`templates/run-index.md`)

```markdown
---
type: run-index
task_id: <% tp.frontmatter.task_id %>
title: "<% tp.frontmatter.title %>"
category: 
status: pending
created: <% tp.date.now("YYYY-MM-DDTHH:mm:ss") %>
completed: 
total_tokens: 0
total_cost_usd: 0
subtask_count: 0
tags: [run]
---

# Run: <% tp.frontmatter.title %>

## Execution Plan

## Subtasks

| ID | Title | Agent | Status | Links |
|----|-------|-------|--------|-------|

## Results

## Timeline
```

### 7.8 Agent-to-Vault Integration

Since KISS agents already have `Read`, `Write`, and `Edit` tools, writing to the Obsidian vault requires **zero additional tooling**:

```python
def write_to_vault(file_path: str, content: str) -> str:
    """Write a markdown file to the Obsidian vault.
    
    The file should include:
    - YAML frontmatter with metadata
    - [[Wikilinks]] to related notes
    - #tags for categorization
    
    Args:
        file_path: Path relative to vault root (e.g., "research/topic.md")
        content: Full markdown content including frontmatter
    
    Returns:
        Confirmation message
    """
    full_path = os.path.join(VAULT_PATH, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    Path(full_path).write_text(content)
    return f"Written to vault: {file_path}"

def read_from_vault(file_path: str) -> str:
    """Read a markdown file from the Obsidian vault.
    
    Args:
        file_path: Path relative to vault root
    
    Returns:
        File contents
    """
    full_path = os.path.join(VAULT_PATH, file_path)
    return Path(full_path).read_text()

def search_vault(query: str) -> str:
    """Search the vault for files matching a query (tag, text, or frontmatter).
    
    Args:
        query: Search string (e.g., "#research", "framework", "type: analysis")
    
    Returns:
        List of matching file paths with excerpts
    """
    results = []
    vault = Path(VAULT_PATH)
    for md_file in vault.rglob("*.md"):
        content = md_file.read_text()
        if query.lower() in content.lower():
            # Extract first 200 chars after match
            idx = content.lower().index(query.lower())
            excerpt = content[max(0, idx-50):idx+200]
            rel_path = md_file.relative_to(vault)
            results.append(f"- {rel_path}: ...{excerpt}...")
    return "\n".join(results[:20]) if results else "No matches found"

def list_vault_links(file_path: str) -> str:
    """List all [[wikilinks]] in a vault file.
    
    Args:
        file_path: Path relative to vault root
    
    Returns:
        List of linked notes
    """
    import re
    content = read_from_vault(file_path)
    links = re.findall(r'\[\[([^\]]+)\]\]', content)
    return "\n".join(f"- [[{link}]]" for link in links)
```

---

## 8. Implementation Guide

### 8.1 Project Structure

```
workflow/
├── AGENTIC_WORKFLOW_PLAN.md          ← This file
├── pyproject.toml
├── src/
│   └── workflow/
│       ├── __init__.py
│       ├── planner.py                ← PlannerAgent (RelentlessAgent subclass)
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── research.py           ← Research agent definition + prompt
│       │   ├── writing.py            ← Writing agent
│       │   ├── code.py               ← Code agent
│       │   ├── analysis.py           ← Analysis agent
│       │   ├── review.py             ← Review agent
│       │   ├── evaluator.py          ← Evaluator agent
│       │   └── synthesizer.py        ← Synthesizer agent
│       ├── vault/
│       │   ├── __init__.py
│       │   ├── vault_tools.py        ← write_to_vault, read_from_vault, search_vault
│       │   └── vault_setup.py        ← Initialize vault structure + Obsidian config
│       ├── models.py                 ← Pydantic models (SubTask, ExecutionPlan)
│       └── cli.py                    ← CLI entry point
│
├── vault/                            ← The Obsidian vault (open this folder in Obsidian)
│   ├── .obsidian/
│   ├── _index.md
│   ├── templates/
│   ├── runs/
│   ├── research/
│   ├── analysis/
│   ├── reports/
│   ├── agents/
│   └── sources/
│
└── tests/
    └── ...
```

### 8.2 Core Implementation Files

#### `planner.py` — The Brain

```python
"""Planner agent that orchestrates the multi-agent workflow."""

from kiss.core.relentless_agent import RelentlessAgent
from kiss.core.kiss_agent import KISSAgent
from workflow.vault.vault_tools import write_to_vault, read_from_vault, search_vault

PLANNER_SYSTEM_PROMPT = """
You are the Planner Agent for an agentic workflow system built on the KISS framework.
You manage an Obsidian vault at {vault_path} as your knowledge database.

Your capabilities:
1. plan_task(task) — Analyze and decompose a task into subtasks
2. run_agent(agent_type, subtask_description) — Run a KISSAgent worker
3. run_parallel(tasks) — Run multiple independent tasks in parallel
4. write_to_vault(path, content) — Write output to the Obsidian vault
5. read_from_vault(path) — Read from the vault
6. search_vault(query) — Search the vault for context
7. finish(result) — Return the final result

Every output must be written to the vault with proper frontmatter, [[links]], and #tags.
"""

class PlannerAgent(RelentlessAgent):
    """Main orchestrator agent."""
    
    def execute(self, task: str, vault_path: str) -> str:
        """Plan, delegate, and synthesize a complex task.
        
        Args:
            task: The user's task description
            vault_path: Path to the Obsidian vault
        
        Returns:
            YAML result with success status and summary
        """
        return self.run(
            model_name="claude-sonnet-4-20250514",
            prompt_template=task,
            system_prompt=PLANNER_SYSTEM_PROMPT.format(vault_path=vault_path),
            tools=[
                write_to_vault,
                read_from_vault,
                search_vault,
                self._run_sub_agent,
            ],
            max_steps=100,
            max_budget=50.0,
            work_dir=vault_path,
        )
    
    def _run_sub_agent(self, agent_type: str, task: str, output_path: str) -> str:
        """Run a specialized KISSAgent for a subtask.
        
        Args:
            agent_type: Type of agent (research, writing, code, analysis, review)
            task: Detailed task description for the agent
            output_path: Where in the vault to write output
        
        Returns:
            The agent's result
        """
        agent = KISSAgent(f"{agent_type.title()} Agent")
        return agent.run(
            model_name="claude-sonnet-4-20250514",
            prompt_template=AGENT_PROMPTS[agent_type],
            arguments={"task": task, "output_path": output_path},
            tools=AGENT_TOOLS[agent_type],
            max_steps=50,
            max_budget=5.0,
        )
```

#### `vault_tools.py` — Vault I/O

```python
"""Tools for reading/writing to the Obsidian vault."""

import os
import re
from pathlib import Path

VAULT_PATH = os.environ.get("WORKFLOW_VAULT_PATH", "./vault")

def write_to_vault(file_path: str, content: str) -> str:
    """Write a markdown file to the Obsidian vault.
    
    The content should include YAML frontmatter and [[wikilinks]].
    
    Args:
        file_path: Relative path within vault (e.g., "research/topic.md")
        content: Full markdown with frontmatter
    
    Returns:
        Confirmation string
    """
    full = os.path.join(VAULT_PATH, file_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    Path(full).write_text(content, encoding="utf-8")
    return f"✅ Written to vault: {file_path}"

def read_from_vault(file_path: str) -> str:
    """Read a file from the Obsidian vault.
    
    Args:
        file_path: Relative path within vault
    
    Returns:
        File contents as string
    """
    full = os.path.join(VAULT_PATH, file_path)
    return Path(full).read_text(encoding="utf-8")

def search_vault(query: str, max_results: int = 20) -> str:
    """Search vault files by content, tag, or frontmatter field.
    
    Args:
        query: Search string
        max_results: Maximum results to return
    
    Returns:
        Formatted list of matching files with excerpts
    """
    results = []
    vault = Path(VAULT_PATH)
    for md in vault.rglob("*.md"):
        if md.parts[0] == ".obsidian":
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            continue
        if query.lower() in text.lower():
            rel = md.relative_to(vault)
            idx = text.lower().index(query.lower())
            excerpt = text[max(0, idx-80):idx+200].replace("\n", " ")
            results.append(f"📄 {rel}\n   ...{excerpt}...")
    return "\n".join(results[:max_results]) or "No matches found."

def get_backlinks(file_path: str) -> str:
    """Find all vault files that link to this file.
    
    Args:
        file_path: Relative path within vault
    
    Returns:
        List of files containing [[links]] to this note
    """
    stem = Path(file_path).stem
    results = []
    vault = Path(VAULT_PATH)
    for md in vault.rglob("*.md"):
        if str(md.relative_to(vault)) == file_path:
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            continue
        if f"[[{stem}]]" in text or f"[[{file_path}]]" in text:
            results.append(f"← {md.relative_to(vault)}")
    return "\n".join(results) or "No backlinks found."
```

#### `vault_setup.py` — Initialize Vault

```python
"""Initialize the Obsidian vault with proper structure and configuration."""

import json
import os
from pathlib import Path

def initialize_vault(vault_path: str) -> None:
    """Create the vault folder structure and Obsidian configuration.
    
    Args:
        vault_path: Root path for the Obsidian vault
    """
    vault = Path(vault_path)
    
    # Create folder structure
    folders = [
        ".obsidian/plugins/dataview",
        "runs",
        "research",
        "analysis",
        "reports",
        "agents",
        "sources",
        "templates",
    ]
    for folder in folders:
        (vault / folder).mkdir(parents=True, exist_ok=True)
    
    # Obsidian app config
    (vault / ".obsidian/app.json").write_text(json.dumps({
        "useMarkdownLinks": False,  # Use [[wikilinks]]
        "newLinkFormat": "shortest",
        "showFrontmatter": True,
        "defaultViewMode": "preview",
    }, indent=2))
    
    # Graph view config with color groups
    (vault / ".obsidian/graph.json").write_text(json.dumps({
        "collapse-filter": False,
        "search": "",
        "showTags": True,
        "showAttachments": False,
        "hideUnresolved": False,
        "showOrphans": False,
        "colorGroups": [
            {"query": "tag:#run", "color": {"a": 1, "rgb": 2605547}},
            {"query": "tag:#research", "color": {"a": 1, "rgb": 52326}},
            {"query": "tag:#analysis", "color": {"a": 1, "rgb": 16750848}},
            {"query": "tag:#report", "color": {"a": 1, "rgb": 10040115}},
            {"query": "tag:#source", "color": {"a": 1, "rgb": 8421504}},
            {"query": "tag:#agent-log", "color": {"a": 1, "rgb": 16776960}},
        ],
        "showArrow": True,
        "nodeSizeMultiplier": 1.2,
        "lineSizeMultiplier": 1,
        "linkDistance": 250,
        "repelStrength": 10,
        "centerStrength": 0.5,
    }, indent=2))
    
    # Enable community plugins
    (vault / ".obsidian/community-plugins.json").write_text(
        json.dumps(["dataview", "templater-obsidian"], indent=2)
    )
    
    # Dataview plugin config
    (vault / ".obsidian/plugins/dataview/data.json").write_text(json.dumps({
        "renderNullAs": "—",
        "taskCompletionTracking": False,
        "warnOnEmptyResult": True,
        "enableDataviewJs": False,
        "enableInlineDataview": True,
    }, indent=2))
    
    # Master index
    (vault / "_index.md").write_text("""---
type: index
tags: [index, MOC]
---

# 🧠 Agentic Workflow Knowledge Base

## Recent Runs

```dataview
TABLE title, category, status, total_cost_usd
FROM "runs"
WHERE type = "run-index"
SORT created DESC
LIMIT 10
```

## All Research Notes

```dataview
TABLE agent, status, tags
FROM "research"
WHERE type = "research"
SORT created DESC
```

## Knowledge Graph

Open the **Graph View** (Ctrl/Cmd + G) to see all connections between notes.

## Folders

- 📁 [[runs/]] — Task execution runs
- 📁 [[research/]] — Research outputs
- 📁 [[analysis/]] — Analysis documents  
- 📁 [[reports/]] — Final reports
- 📁 [[sources/]] — Source references
- 📁 [[agents/]] — Agent run logs
""")
    
    # Templates
    (vault / "templates/research-note.md").write_text("""---
type: research
task_id: 
subtask_id: 
agent: research_agent
model: claude-sonnet-4-20250514
status: pending
attempt: 1
created: {{date}}
completed: 
tokens_used: 
cost_usd: 
evaluator_score: 
tags: [research]
sources: []
depends_on: []
feeds_into: []
---

# Research: {{title}}

> Part of run: [[{{run_index}}]]

## Findings

## Sources

## Related Notes
""")
    
    (vault / "templates/run-index.md").write_text("""---
type: run-index
task_id: 
title: ""
category: 
status: pending
created: {{date}}
completed: 
total_tokens: 0
total_cost_usd: 0
subtask_count: 0
tags: [run]
---

# Run: {{title}}

## Execution Plan

## Subtasks

| ID | Title | Agent | Status | Output |
|----|-------|-------|--------|--------|

## Results

## Timeline
""")
```

---

## 9. Data Schemas

### 9.1 Execution Plan (Planner Output)

```json
{
  "task_id": "run_20250527_001",
  "original_task": "Research the current state of agentic AI workflows and write a report",
  "category": "mixed",
  "vault_run_folder": "runs/2025-05-27_agentic-research",
  "subtasks": [
    {
      "id": "st_01",
      "title": "Web Research — Agentic Frameworks",
      "description": "Search for top 10 agentic AI frameworks in 2025...",
      "category": "research",
      "assigned_agent": "research_agent",
      "dependencies": [],
      "expected_output": "Markdown with structured findings per framework",
      "success_criteria": ["Covers 8+ frameworks", "Each has name, stars, features", "Sources cited"],
      "priority": 1,
      "estimated_tokens": 8000,
      "vault_output_path": "runs/2025-05-27_agentic-research/st_01-research-frameworks.md",
      "vault_links": ["[[ai-agents]]", "[[frameworks]]"]
    },
    {
      "id": "st_02",
      "title": "Web Research — Knowledge Management",
      "description": "Research Obsidian as a knowledge management tool for AI workflows...",
      "category": "research",
      "assigned_agent": "research_agent",
      "dependencies": [],
      "expected_output": "Markdown comparing knowledge management approaches",
      "success_criteria": ["Covers Obsidian vault structure", "Compares alternatives"],
      "priority": 1,
      "estimated_tokens": 6000,
      "vault_output_path": "runs/2025-05-27_agentic-research/st_02-research-knowledge.md",
      "vault_links": ["[[obsidian]]", "[[knowledge-management]]"]
    },
    {
      "id": "st_03",
      "title": "Analysis — Compare and Recommend",
      "description": "Using [[st_01-research-frameworks]] and [[st_02-research-knowledge]], create comparison...",
      "category": "analysis",
      "assigned_agent": "analysis_agent",
      "dependencies": ["st_01", "st_02"],
      "expected_output": "Analysis with tables and clear recommendation",
      "success_criteria": ["Comparison matrix", "Clear recommendation with justification"],
      "priority": 2,
      "estimated_tokens": 5000,
      "vault_output_path": "runs/2025-05-27_agentic-research/st_03-analysis.md",
      "vault_links": ["[[st_01-research-frameworks]]", "[[st_02-research-knowledge]]"]
    },
    {
      "id": "st_04",
      "title": "Write Final Report",
      "description": "Synthesize all research and analysis into comprehensive report...",
      "category": "writing",
      "assigned_agent": "writing_agent",
      "dependencies": ["st_01", "st_02", "st_03"],
      "expected_output": "Polished report with TOC, tables, citations",
      "success_criteria": ["Clear structure", "All claims sourced", "Professional tone"],
      "priority": 3,
      "estimated_tokens": 10000,
      "vault_output_path": "reports/agentic-ai-state-of-art.md",
      "vault_links": ["[[st_01-research-frameworks]]", "[[st_02-research-knowledge]]", "[[st_03-analysis]]"]
    }
  ],
  "execution_order": [
    ["st_01", "st_02"],
    ["st_03"],
    ["st_04"]
  ],
  "total_estimated_tokens": 29000,
  "created_at": "2025-05-27T12:00:00Z"
}
```

### 9.2 How the Graph View Looks

After a run completes, opening Obsidian's Graph View (Ctrl/Cmd + G) shows:

```
                    ┌─────────────────┐
                    │   _run-index    │ (green - run)
                    └───┬───┬───┬────┘
                        │   │   │
           ┌────────────┘   │   └────────────┐
           ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ st_01        │ │ st_02        │ │ st_03        │  (blue - research/analysis)
    │ frameworks   │ │ knowledge    │ │ analysis     │
    └──┬───┬───┬──┘ └──┬───┬──────┘ └──────┬───────┘
       │   │   │       │   │               │
       ▼   ▼   ▼       ▼   ▼               ▼
   ┌────┐┌────┐┌────┐┌────┐┌────┐    ┌──────────────┐
   │kiss││lang││crew││obsi││alte│    │ final report │  (purple - report)
   │    ││grap││ai  ││dian││rnat│    └──────────────┘
   └────┘└────┘└────┘└────┘└────┘
   (gray - source references)
```

Each node is clickable. Hovering shows preview. Filtering by tag shows subgraphs. This is **automatic** — no code needed beyond writing proper `[[links]]`.

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

```
1. Set up project with pyproject.toml + KISS dependency
2. Implement vault_setup.py — initialize Obsidian vault structure
3. Implement vault_tools.py — read, write, search, backlinks
4. Create Obsidian templates for all note types
5. Build base agent definitions with prompts and tool lists
6. Implement PlannerAgent (RelentlessAgent subclass) with task decomposition
```

### Phase 2: Core Agents (Week 3-4)

```
1. Research Agent — WebUseTool + vault writing with [[links]]
2. Writing Agent — Read vault context, produce linked documents
3. Code Agent — Bash + file tools, output to vault
4. Evaluator Agent — Non-agentic quality gate
5. Synthesizer Agent — Read all subtask outputs, produce final report
6. Wire up execution DAG with parallel + sequential execution
```

### Phase 3: Integration (Week 5-6)

```
1. End-to-end: user input → plan → execute → synthesize → vault output
2. Context retrieval from vault (search past research by tags/links)
3. Retry loop with evaluator feedback + prompt_refiner_agent()
4. Run index auto-generation with Dataview queries
5. Graph View configuration and testing
6. Install Dataview + Templater plugins, verify queries work
```

### Phase 4: Polish (Week 7-8)

```
1. CLI entry point (kiss-workflow run "task description")
2. Budget management across all sub-agents
3. Error handling and graceful degradation
4. Vault maintenance tools (orphan detection, broken link check)
5. Documentation
6. Integration tests
```

---

## 11. Vault Organizer Agent — Token Optimization

### 11.1 Purpose

The **Vault Organizer Agent** is a specialized maintenance agent that periodically optimizes the Obsidian vault structure for **minimal token consumption** when other agents read from it. As the vault grows, raw notes accumulate noise — verbose prose, redundant content, broken links, and unstructured data. Without optimization, agents waste tokens loading irrelevant context, and **context rot** (accuracy degradation as token count grows) undermines output quality.

The Vault Organizer solves this by acting as a librarian: it indexes, summarizes, deduplicates, and restructures the vault so every token an agent reads is maximally informative.

### 11.2 Core Operations

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     VAULT ORGANIZER AGENT                                │
│                      (KISSAgent + MCP Tools)                             │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  1. INDEX     │  │  2. SUMMARIZE│  │ 3. DEDUPLICATE│  │ 4. ARCHIVE  │ │
│  │              │  │              │  │              │  │             │ │
│  │ Build topic  │  │ Compress     │  │ Merge near-  │  │ Move stale  │ │
│  │ indexes &    │  │ verbose notes│  │ duplicate     │  │ notes to    │ │
│  │ MOCs from    │  │ into compact │  │ notes, unify  │  │ archive/,   │ │
│  │ frontmatter  │  │ summaries    │  │ overlapping   │  │ update      │ │
│  │ + tags       │  │ w/ key facts │  │ content       │  │ links       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ 5. LINK      │  │ 6. FRONTMATTER│  │ 7. REPORT   │                   │
│  │   AUDIT      │  │   NORMALIZE  │  │              │                   │
│  │              │  │              │  │ Generate     │                   │
│  │ Find broken  │  │ Ensure all   │  │ optimization │                   │
│  │ [[links]],   │  │ notes have   │  │ report with  │                   │
│  │ orphan notes,│  │ consistent   │  │ token savings│                   │
│  │ circular refs│  │ YAML schema  │  │ metrics      │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Token Optimization Strategies

#### Strategy 1: Progressive Disclosure Indexing

Instead of agents reading full notes, the organizer creates **three-tier access**:

```
Tier 1: INDEX (frontmatter catalog)    ~50 tokens/note
  → agent reads metadata only, decides which notes are relevant

Tier 2: SUMMARY (compressed key facts)  ~200 tokens/note
  → agent reads compact summary, confirms relevance

Tier 3: FULL CONTENT (original note)    ~2000+ tokens/note
  → agent reads full note only when truly needed
```

The organizer generates `_indexes/` files that catalog all notes by topic:

```markdown
---
type: index
topic: machine-learning
updated: 2025-05-28
note_count: 47
---
# Machine Learning Index

| Note | Tags | Date | Summary |
|------|------|------|---------|
| [[ml-transformers]] | #deep-learning #nlp | 2025-05-15 | Transformer architecture comparison: GPT vs BERT vs T5 |
| [[rl-policy-gradient]] | #reinforcement-learning | 2025-05-10 | Policy gradient methods with variance reduction |
| ... | ... | ... | ... |
```

#### Strategy 2: Extractive Summarization

For each verbose research note, the organizer adds a `_summary` field to frontmatter:

```yaml
---
title: "Deep Dive: Transformer Architectures"
tags: [deep-learning, nlp, transformers]
token_count: 4200
_summary: >
  Compares GPT (decoder-only, autoregressive), BERT (encoder-only, MLM),
  and T5 (encoder-decoder, text-to-text). Key finding: T5 outperforms on
  structured tasks; GPT dominates generation. All benefit from scaling.
  Recommended: use T5 for classification, GPT for generation.
_summary_tokens: 85
---
```

Agents can read `get_frontmatter` (via MCP) to get the summary without loading the full note — **98% token savings** on large notes.

#### Strategy 3: Deduplication & Merging

The organizer detects near-duplicate content by:
1. Comparing frontmatter tags and titles for overlap
2. Using MCP `search_notes` to find notes with similar key phrases
3. Merging overlapping content into canonical notes with `[[redirects]]`

```markdown
---
title: "MERGED: API Authentication Methods"
merged_from:
  - "[[oauth2-notes]]"
  - "[[api-auth-research]]"
  - "[[jwt-tokens-explained]]"
tags: [api, authentication, oauth, jwt]
---
# API Authentication Methods (Consolidated)
<!-- Merged from 3 separate notes to reduce redundancy -->
...
```

#### Strategy 4: Map of Content (MOC) Generation

The organizer auto-generates MOC notes that serve as navigable entry points:

```markdown
---
type: moc
domain: "artificial-intelligence"
child_notes: 47
total_tokens: 94000
moc_tokens: 350
token_savings_ratio: "268:1"
---
# 🗺️ Artificial Intelligence — Map of Content

## Core Concepts
- [[neural-networks-fundamentals]] — Backprop, activation functions, architectures
- [[transformer-architectures]] — GPT vs BERT vs T5 comparison

## Applied ML
- [[ml-for-code]] — Code generation, bug detection, refactoring
- [[rl-agents]] — Policy gradient, Q-learning, multi-agent systems

## Tools & Frameworks
- [[pytorch-guide]] — Training loops, custom datasets, distributed
- [[langchain-vs-llamaindex]] — RAG framework comparison
```

A MOC gives agents a **268:1 token savings ratio** — they read 350 tokens instead of 94,000 to understand the entire domain.

#### Strategy 5: Archive Stale Content

Notes older than a configurable threshold (e.g., 90 days since last modification or last reference) are moved to `archive/`:

```python
# Pseudo-logic for the organizer
for note in vault_notes:
    if note.last_modified > 90_days_ago and note.incoming_links == 0:
        move_to_archive(note)
        update_all_references(note, f"archive/{note.path}")
```

Archived notes are excluded from default searches and indexes but remain accessible.

### 11.4 Agent Definition

```python
vault_organizer = AgentDefinition(
    name="vault_organizer",
    role="Vault Organizer — Token Optimization Specialist",
    system_prompt="""You are the Vault Organizer Agent. Your mission is to optimize
the Obsidian vault for minimal token consumption by other agents.

You have access to the vault via MCP tools (MCPVault server). Use these tools to:
1. Read vault structure and note metadata efficiently (get_vault_stats, list_directory, get_frontmatter)
2. Search for related/duplicate content (search_notes with BM25 ranking)
3. Read notes that need processing (read_note, read_multiple_notes)
4. Write optimized versions (write_note, patch_note, update_frontmatter)
5. Generate indexes and MOCs (write_note)
6. Manage tags consistently (manage_tags)

OPTIMIZATION PRINCIPLES:
- Never delete content — archive or merge instead
- Always preserve [[wikilinks]] integrity when moving/merging notes
- Add _summary to frontmatter for notes exceeding 500 tokens
- Generate topic indexes in _indexes/ folder
- Generate MOCs in _mocs/ folder
- Track token savings in _maintenance/ reports
- Normalize all frontmatter to consistent schema
""",
    prompt_template="""Optimize the vault at {vault_path} for token efficiency.

Current vault stats: {vault_stats}
Last optimization: {last_run_date}

Focus areas this run:
{focus_areas}

Generate an optimization report at _maintenance/optimization-{run_id}.md
""",
    tools=["mcp_vault_tools"],  # MCPVault MCP server tools
    model_name="claude-sonnet-4-20250514",
    is_agentic=True,
    max_steps=100,
    max_budget=10.0,
    max_retries=1,
    vault_output_folder="_maintenance",
    tags=["maintenance", "optimization", "token-efficiency"],
)
```

### 11.5 Scheduling

The Vault Organizer runs as a **periodic maintenance task**, not during regular workflow execution:

| Trigger | Frequency | Scope |
|---------|-----------|-------|
| After every workflow run | Automatic | Light pass — add _summary to new notes only |
| Daily scheduled | Cron / manual | Medium pass — regenerate indexes, check for duplicates |
| Weekly scheduled | Cron / manual | Full pass — deduplication, MOC generation, archival, full report |
| Manual trigger | On-demand | User-specified focus area (e.g., "optimize research/ folder") |

### 11.6 Optimization Report Output

Each run produces a report in `_maintenance/`:

```markdown
---
type: maintenance-report
agent: vault_organizer
run_id: "opt-2025-05-28-001"
timestamp: 2025-05-28T14:30:00Z
duration_seconds: 120
---
# Vault Optimization Report — 2025-05-28

## Token Savings Summary
| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| Avg tokens/note retrieval | 2,400 | 180 | **92.5%** |
| Total vault tokens | 450,000 | 450,000 | 0% (content preserved) |
| Indexed access tokens | — | 12,000 | New: covers all 188 notes |
| MOC coverage | 0% | 85% | 16 MOCs generated |

## Actions Taken
- ✅ Added _summary to 23 new research notes
- ✅ Generated 3 new topic indexes (_indexes/ai.md, _indexes/databases.md, _indexes/security.md)
- ✅ Created 2 new MOCs (_mocs/machine-learning.md, _mocs/system-design.md)
- ✅ Merged 4 duplicate notes into 2 canonical notes
- ✅ Archived 7 stale notes (>90 days, 0 incoming links)
- ✅ Fixed 3 broken [[wikilinks]]
- ✅ Normalized frontmatter on 12 notes missing required fields

## Vault Health
- Total notes: 188 (↓4 from merges, ↓7 from archival, ↑3 new indexes)
- Orphan notes: 2 (down from 9)
- Missing frontmatter: 0 (down from 12)
- Coverage: 100% notes have _summary
```

---

## 12. MCP Server Integration — Evaluation & Recommendation

### 12.1 Evaluation Summary

**Should an MCP server be used for Obsidian vault access in this agentic workflow?**

> **Verdict: ✅ STRONG YES** — based on research across 10 authoritative sources including Anthropic's official documentation, the MCP specification, GitHub ecosystem analysis, and community validation.

### 12.2 What is MCP?

The **Model Context Protocol (MCP)** is an open standard created by Anthropic for connecting AI applications to external data sources and tools — described as "USB-C for AI." It provides a standardized client-server protocol with three primitives:

- **Tools** — Executable functions (model-controlled): `search_notes`, `write_note`, `manage_tags`
- **Resources** — Passive data sources (app-controlled): `obsidian://vault/notes/{path}`
- **Prompts** — Reusable templates (user-controlled): `summarize-research`, `create-moc`

### 12.3 Why MCP for This Workflow

| Benefit | How It Applies |
|---------|---------------|
| **Token efficiency** | MCP tools enforce selective retrieval — agents call `search_notes` or `get_frontmatter` instead of loading everything. This directly combats **context rot** (accuracy degradation as token count grows). |
| **Standardized interface** | All agents interact with the vault through the same protocol. No custom file I/O code per agent. |
| **Built-in search** | BM25-ranked full-text search returns relevant excerpts, not entire files. |
| **Metadata-first access** | `get_frontmatter` and `get_notes_info` let agents read metadata without content. |
| **Batch operations** | `read_multiple_notes` reduces tool call overhead for multi-note operations. |
| **Portable** | Works with Claude, ChatGPT, Cursor, VS Code, and future MCP-compatible tools. |
| **Anthropic endorsed** | Anthropic explicitly recommends MCP for tool integration in agents. |

### 12.4 Recommended MCP Server: MCPVault

**Selected: [`@bitbonsai/mcpvault`](https://github.com/bitbonsai/mcpvault)** (1.3k ⭐, 180 commits, MIT License)

#### Why MCPVault over alternatives:

| Criteria | MCPVault (bitbonsai) | mcp-obsidian (MarkusPfundstein) |
|----------|---------------------|-------------------------------|
| **Stars** | 1.3k | 3.8k |
| **License** | MIT ✅ | MIT ✅ |
| **Language** | TypeScript | Python |
| **Obsidian app required?** | ❌ No — filesystem-based | ✅ Yes — requires REST API plugin |
| **Token optimization** | ✅ Built-in (40-60% smaller responses, minified fields) | ❌ None |
| **Tools count** | 14 (comprehensive) | 7 (basic) |
| **Search** | BM25 with relevance reranking | Basic text search |
| **Batch reads** | ✅ `read_multiple_notes` | ❌ |
| **Frontmatter access** | ✅ `get_frontmatter`, `update_frontmatter` | ❌ |
| **Tag management** | ✅ `manage_tags` (add/remove/list) | ❌ |
| **Vault stats** | ✅ `get_vault_stats`, `get_notes_info` | ❌ |
| **Safety** | Path traversal prevention, confirmPath for deletes, YAML validation | Basic |
| **Headless operation** | ✅ Works without Obsidian running | ❌ Requires Obsidian desktop |
| **Security** | SECURITY.md, CodeQL scanning, Dependabot, SLSA provenance | Basic |
| **Last updated** | 2 days ago (very active) | 2 weeks ago |

**MCPVault wins** because:
1. **No Obsidian app dependency** — critical for headless/server agentic workflows
2. **Built-in token optimization** — 40-60% smaller responses by default
3. **Comprehensive 14-tool API** — covers all vault operations our agents need
4. **Superior security** — CodeQL, SLSA provenance, path traversal prevention, YAML validation
5. **MIT License** — fully permissive for commercial and private use

#### Licensing Details

| Component | License | Permissions |
|-----------|---------|-------------|
| MCPVault (`@bitbonsai/mcpvault`) | **MIT** | Use, copy, modify, merge, publish, distribute, sublicense, sell — no restrictions |
| MCP SDK (`@modelcontextprotocol/sdk`) | **MIT** | Same as above |
| mcp-obsidian (alternative) | **MIT** | Same as above |
| MCP Protocol specification | **Open standard** (Linux Foundation) | Freely implementable |

All components are **MIT-licensed** — the most permissive open-source license. No copyleft obligations, no attribution requirements beyond including the license text.

### 12.5 MCPVault Tool Reference

The 14 tools exposed by MCPVault, grouped by function:

#### File Operations
| Tool | Description | Token Impact |
|------|-------------|-------------|
| `read_note` | Read note with parsed frontmatter | Returns full content |
| `write_note` | Write note (overwrite/append/prepend) | — |
| `patch_note` | Replace exact string in note | Minimal response |
| `delete_note` | Delete with confirmation (trash modes) | Minimal response |
| `move_note` | Move/rename a note | Minimal response |
| `move_file` | Move any file type | Minimal response |

#### Discovery & Search
| Tool | Description | Token Impact |
|------|-------------|-------------|
| `list_directory` | List files/dirs in path | Compact: `{dirs: [...], files: [...]}` |
| `search_notes` | BM25 full-text search | Returns excerpts (21 chars), not full content |
| `get_vault_stats` | Vault-level statistics | Compact summary |

#### Metadata Operations
| Tool | Description | Token Impact |
|------|-------------|-------------|
| `get_frontmatter` | Read only YAML frontmatter | **No content loaded** — key for token savings |
| `update_frontmatter` | Modify frontmatter fields | Minimal response |
| `get_notes_info` | Batch metadata for multiple notes | **No content loaded** |

#### Batch & Tag Operations
| Tool | Description | Token Impact |
|------|-------------|-------------|
| `read_multiple_notes` | Batch read multiple notes | Reduces tool call overhead |
| `manage_tags` | Add/remove/list tags | Minimal response |

### 12.6 MCP Integration Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    KISS AGENT WORKFLOW                                    │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────────────┐ │
│  │ Planner  │  │ Research │  │ Writing  │  │ Vault Organizer         │ │
│  │ Agent    │  │ Agent    │  │ Agent    │  │ Agent                   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬─────────────────┘ │
│       │              │              │                │                    │
│       └──────────────┴──────────────┴────────────────┘                   │
│                              │                                           │
│                    MCP Client (JSON-RPC 2.0)                             │
│                              │                                           │
└──────────────────────────────┼───────────────────────────────────────────┘
                               │ stdio transport
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    MCPVault SERVER                                        │
│                    (@bitbonsai/mcpvault)                                  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ 14 Tools: read_note, write_note, search_notes, get_frontmatter, │    │
│  │ patch_note, list_directory, manage_tags, get_vault_stats, ...    │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│             Token-Optimized Response Layer                               │
│        (minified field names, compact JSON, 40-60% savings)              │
│                              │                                           │
└──────────────────────────────┼───────────────────────────────────────────┘
                               │ filesystem access
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    OBSIDIAN VAULT (filesystem)                            │
│                                                                          │
│  📁 research/        📁 _indexes/       📁 _mocs/                       │
│  📁 reports/         📁 _maintenance/   📁 archive/                      │
│  📁 analysis/        📁 runs/           📁 sources/                      │
│                                                                          │
│     .md files with [[wikilinks]], YAML frontmatter, #tags                │
│     ↕ Obsidian Graph View renders the knowledge web                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 12.7 Setup & Configuration

#### Install MCPVault

```bash
# No global install needed — use npx
npx @bitbonsai/mcpvault@latest /path/to/obsidian/vault
```

#### Configure for KISS Agents

The MCP server runs as a subprocess alongside the KISS agent workflow:

```python
import subprocess
import json

def start_mcp_server(vault_path: str) -> subprocess.Popen:
    """Start MCPVault as an MCP server subprocess (stdio transport)."""
    return subprocess.Popen(
        ["npx", "@bitbonsai/mcpvault@latest", vault_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

# MCP server configuration for KISS agents
MCP_CONFIG = {
    "mcpServers": {
        "obsidian": {
            "command": "npx",
            "args": ["@bitbonsai/mcpvault@latest", "/path/to/workflow-vault"],
        }
    }
}
```

#### KISS Agent Tool Wrapper

```python
def mcp_search_notes(query: str, limit: int = 10) -> list[dict]:
    """Search vault notes via MCP (BM25 ranked, token-optimized).

    Returns compact results with fields: p (path), t (title),
    ex (excerpt), mc (match_count), ln (line_number).
    """
    return mcp_call("search_notes", {
        "query": query,
        "limit": limit,
        "searchContent": True,
        "prettyPrint": False,
    })

def mcp_get_frontmatter(path: str) -> dict:
    """Read only frontmatter metadata — no content loaded.

    Key for token optimization: ~50 tokens instead of ~2000+.
    """
    return mcp_call("get_frontmatter", {"path": path, "prettyPrint": False})

def mcp_read_note(path: str) -> dict:
    """Read full note content + parsed frontmatter.

    Use only when full content is needed (Tier 3 access).
    """
    return mcp_call("read_note", {"path": path, "prettyPrint": False})

def mcp_write_note(path: str, content: str, frontmatter: dict = None,
                   mode: str = "overwrite") -> dict:
    """Write a note to the vault with optional frontmatter."""
    args = {"path": path, "content": content, "mode": mode}
    if frontmatter:
        args["frontmatter"] = frontmatter
    return mcp_call("write_note", args)
```

### 12.8 MCP vs Direct File I/O — Full Comparison

| Aspect | MCP Server (MCPVault) | Direct File I/O (Read/Write/Edit) |
|--------|----------------------|----------------------------------|
| **Token efficiency** | Built-in: minified fields, excerpts, 40-60% savings | Manual: must build optimization layer |
| **Search** | BM25 ranked results with excerpts | Grep/regex — unranked, returns full lines |
| **Frontmatter access** | Parsed, queryable, separate from content | Manual YAML parsing needed |
| **Batch operations** | `read_multiple_notes`, `get_notes_info` | Manual loop over files |
| **Standardization** | MCP protocol — same tools work in Claude, ChatGPT, Cursor | Custom per-framework |
| **Portability** | Any MCP-compatible client | KISS framework only |
| **Safety** | Path traversal prevention, YAML validation, CodeQL | Must implement manually |
| **Extra dependency** | Node.js runtime + MCPVault npm package | None |
| **Process overhead** | +1 subprocess (MCPVault server) | Zero |
| **Latency** | ~5-20ms per tool call (stdio transport) | ~1-5ms per file operation |
| **Complexity** | MCP client setup required | Zero setup |

**When to use MCP**: Agentic workflows with multiple agents reading from vault, token-sensitive operations, need for ranked search, portability across AI tools.

**When to use direct I/O**: Simple single-agent scripts, performance-critical batch processing, environments without Node.js.

**Recommendation for this workflow**: Use **MCP as the primary vault interface** for all agents. Fall back to direct file I/O only for bulk operations where MCP overhead matters (e.g., initial vault setup, mass migration).

### 12.9 Research Sources

This evaluation was based on research across 10 authoritative sources:

| # | Source | Key Insight |
|---|--------|-------------|
| 1 | [MCP Introduction](https://modelcontextprotocol.io/docs/getting-started/intro) | MCP is "USB-C for AI" — standardized tool integration |
| 2 | [MCP Architecture](https://modelcontextprotocol.io/docs/learn/architecture) | Client-Server with Tools, Resources, Prompts primitives |
| 3 | [MCP Server Concepts](https://modelcontextprotocol.io/docs/learn/server-concepts) | Resources enable on-demand retrieval vs bulk loading |
| 4 | [GitHub Search: Obsidian MCP](https://github.com/search?q=obsidian+mcp+server) | 614+ repos — mature ecosystem |
| 5 | [mcp-obsidian (3.8k ⭐)](https://github.com/MarkusPfundstein/mcp-obsidian) | Top Python server, requires Obsidian REST API plugin |
| 6 | [MCPVault (1.3k ⭐)](https://github.com/bitbonsai/mcpvault) | Filesystem-based, token-optimized, 14 tools, MIT |
| 7 | [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | Anthropic recommends MCP for agent tool integration |
| 8 | [Reddit r/ObsidianMD](https://www.reddit.com/r/ObsidianMD/) | Active community using MCP+Obsidian for "second brain" agents |
| 9 | [MCP Registry](https://modelcontextprotocol.io/registry/about) | Official registry backed by Anthropic, GitHub, Microsoft |
| 10 | [Anthropic: Context Windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) | Context rot = accuracy degrades with more tokens; curate context |

---

## Summary

This architecture provides:

- **🧠 KISS-Powered Orchestration** — Planner (RelentlessAgent) + Worker (KISSAgent) agents with built-in parallel execution, auto-continuation, and budget control
- **📊 Obsidian Graph View** — Every research output, analysis, and report is a linked Markdown note; the Graph View auto-renders the entire knowledge web
- **🔗 Zero-Config Knowledge Graph** — `[[Wikilinks]]` + YAML frontmatter + `#tags` = automatic bidirectional linking, no database schema needed
- **🔍 Queryable Metadata** — Dataview plugin enables SQL-like queries over frontmatter (cost tracking, status, agent performance)
- **📝 Human-Readable Outputs** — Plain Markdown files, viewable in Obsidian, VS Code, GitHub, or any text editor
- **🔄 Quality Gates** — Evaluator agents ensure output quality with feedback loops
- **📈 Full Lineage** — Every note links to its dependencies, sources, and run index — trace any conclusion back to its origin
- **⚡ Minimal Code** — KISS agents are Python function calls; vault is just a folder of `.md` files. No databases, no migrations, no Docker for storage

- **🗄️ MCP Server Integration** — MCPVault provides standardized, token-optimized vault access for all agents via the Model Context Protocol (14 tools, BM25 search, 40-60% smaller responses)
- **🧹 Vault Organizer Agent** — Periodic maintenance agent that indexes, summarizes, deduplicates, and archives vault content for minimal token consumption (92%+ savings via progressive disclosure)

**Stack:**
```
KISS Framework (KISSAgent + RelentlessAgent)  — Agent orchestration
+ MCPVault (@bitbonsai/mcpvault, MIT)         — MCP server for vault access (token-optimized)
+ Obsidian Vault (Markdown + [[links]] + YAML frontmatter)  — Knowledge database + graph
+ Dataview Plugin  — SQL-like queries over frontmatter
+ Templater Plugin  — Consistent output formatting
+ Claude Sonnet 4.6 / Gemini 3.5 Flash (primary LLMs)  — Agent models (multi-provider)
+ Claude Haiku 4.5 / Gemini 2.5 Flash-Lite (budget LLMs)  — Classification & utility agents
+ Node.js (≥20.0.0)  — MCPVault runtime
```

---

## 13. Agent Definitions, Model Selection & Token Optimization

> Research-backed strategies for maximizing agent performance while minimizing token cost.
> Based on 12 authoritative sources: Anthropic docs (prompt caching, effort parameter, compaction, context windows, batch processing, context editing, manage tool context, models overview, structured outputs), Anthropic Engineering Blog (context engineering), and Google Gemini API docs (models, pricing).

---

### 13.1 Multi-Provider Model Comparison

Each agent in the workflow should use the **cheapest model that meets its quality threshold**. The two providers compared are **Anthropic Claude** and **Google Gemini**.

#### Current Model Landscape (as of June 2026)

| Model | Provider | Input $/MTok | Output $/MTok | Context | Max Output | Latency | Best For |
|-------|----------|-------------|---------------|---------|------------|---------|----------|
| **Claude Opus 4.7** | Anthropic | $5.00 | $25.00 | 1M | 128k | Moderate | Complex reasoning, agentic coding |
| **Claude Sonnet 4.6** | Anthropic | $3.00 | $15.00 | 1M | 64k | Fast | Best balance intelligence + speed |
| **Claude Haiku 4.5** | Anthropic | $1.00 | $5.00 | 200k | 64k | Fastest | High-volume, classification |
| **Gemini 3.5 Flash** | Google | $1.50 | $9.00 | 1M | — | Fast | Frontier agentic + coding |
| **Gemini 3.1 Pro** | Google | $2.00 | $12.00 | — | — | Moderate | Advanced intelligence, vibe-coding |
| **Gemini 3 Flash** | Google | $0.50 | $3.00 | — | — | Fast | Frontier at fraction of cost |
| **Gemini 2.5 Pro** | Google | $1.25 | $10.00 | 1M | — | Moderate | Coding + complex reasoning |
| **Gemini 2.5 Flash** | Google | $0.30 | $2.50 | 1M | — | Fast | Price-performance king |
| **Gemini 2.5 Flash-Lite** | Google | $0.10 | $0.40 | — | — | Fastest | Ultra-cheap, high-volume |
| **Gemini 3.1 Flash-Lite** | Google | $0.25 | $1.50 | — | — | Fastest | High-volume agentic |

#### Provider Feature Comparison

| Feature | Anthropic Claude | Google Gemini |
|---------|-----------------|--------------|
| **Prompt/Context Caching** | ✅ 90% savings (cache reads at 10% base) | ✅ Context caching (90% savings) |
| **Batch API** | ✅ 50% discount | ✅ 50% discount |
| **Effort Parameter** | ✅ `low`/`medium`/`high`/`xhigh`/`max` | ❌ Not available (use thinking budgets) |
| **Context Editing** | ✅ Tool result clearing, thinking clearing | ❌ Not available |
| **Compaction** | ✅ Server-side + client-side | ❌ Not native (manual only) |
| **Structured Outputs** | ✅ JSON schema + strict tool use | ✅ JSON schema |
| **Tool Search** | ✅ Dynamic tool discovery | ❌ Not available |
| **Extended Thinking** | ✅ Sonnet 4.6, Haiku 4.5 | ✅ Thinking budgets (Gemini 2.5+) |
| **Free Tier** | ❌ No | ✅ Yes (rate-limited) |
| **Flex/Priority Tiers** | ✅ Priority tier | ✅ Flex (cheapest) + Priority |
| **Built-in Web Search** | ✅ Server tool | ✅ Google Search grounding (5k free/month) |
| **KISS Framework Native** | ✅ Primary LLM | ⚠️ Requires OpenAI-compat endpoint adapter |

#### Challenge: Claude vs Gemini — Which Provider?

**Argument for Claude-primary:**
- KISS Framework is built for Anthropic Claude natively — tool use, caching, effort, compaction all work out of the box
- Effort parameter gives fine-grained token control per agent (no Gemini equivalent)
- Context editing (tool result clearing) is critical for long-running agents — Gemini lacks this
- Batch API + prompt caching discounts stack for up to 95% savings on eligible agents

**Argument for Gemini-primary:**
- Gemini 2.5 Flash-Lite at $0.10/$0.40 is **10x cheaper** than Claude Haiku at $1/$5 for bulk classification
- Gemini 3 Flash at $0.50/$3.00 offers frontier performance at Haiku-like prices
- Free tier enables zero-cost development and testing
- Google Search grounding is built-in (5,000 free prompts/month) — no separate web search tool needed
- Flex inference mode provides additional cost savings for non-latency-sensitive tasks

**Verdict: Hybrid Strategy (Claude-primary, Gemini for budget agents)**

Use Claude for agents that benefit from its unique features (effort, context editing, compaction, deep KISS integration). Use Gemini for high-volume, cost-sensitive agents where raw price dominates.

---

### 13.2 Full Agent Definitions with Skills Instructions

Each agent receives a **Skills Instruction** — a structured prompt section injected into the system prompt that defines its persona, constraints, output format, and domain expertise. Skills instructions are referenced in the agent's `system_prompt` field and are part of the prompt sent to the sub-agent.

#### Updated AgentDefinition with Skills and Optimization Parameters

```python
from dataclasses import dataclass, field
from typing import Callable, Literal

@dataclass
class AgentDefinition:
    """Complete definition for a workflow agent with skills and optimization config."""
    
    # Identity
    name: str                                    # e.g., "research_agent"
    role: str                                    # Human-readable role
    
    # Prompt
    system_prompt: str                           # Behavioral instructions
    skills_instruction: str                      # Domain expertise prompt (injected into system_prompt)
    prompt_template: str                         # Task prompt with {placeholders}
    
    # Tools
    tools: list[Callable]                        # List of tool functions
    
    # Model Selection
    model_name: str = "claude-sonnet-4-6"        # Primary model
    fallback_model: str | None = None            # Fallback if primary unavailable
    provider: Literal["anthropic", "gemini"] = "anthropic"
    
    # Execution Config
    is_agentic: bool = True                      # False = single generation (non-agentic)
    max_steps: int = 50                          # Per-run step limit
    max_budget: float = 5.0                      # Per-run budget in USD
    max_retries: int = 3                         # Retries on evaluator rejection
    
    # Token Optimization
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    enable_prompt_caching: bool = True           # Cache system prompt + tools
    enable_context_editing: bool = False         # Tool result clearing
    context_edit_trigger_tokens: int = 100000    # When to start clearing
    context_edit_keep_tool_uses: int = 3         # Recent tool uses to keep
    enable_compaction: bool = False              # Client-side compaction
    compaction_threshold_tokens: int = 50000     # When to compact
    compaction_summary_model: str = "claude-haiku-4-5"  # Cheap model for summaries
    enable_structured_output: bool = False       # JSON schema for output
    output_schema: dict | None = None            # JSON schema definition
    batch_eligible: bool = False                 # Can run via Batch API (50% off)
    enable_thinking_clearing: bool = False       # Clear old thinking blocks
    thinking_keep_turns: int = 2                 # Recent thinking turns to keep
    
    # Vault
    vault_output_folder: str = "research"        # Where in vault to write output
    tags: list[str] = field(default_factory=list)
```

---

#### Agent 1: Planner Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  PLANNER AGENT                                                      │
│                                                                     │
│  Model:    Claude Sonnet 4.6 ($3/$15/MTok)                         │
│  Effort:   high                                                     │
│  Class:    RelentlessAgent (auto-continuation)                      │
│  Agentic:  Yes (multi-step orchestration)                           │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Prompt caching (system prompt + tool defs = stable prefix)     │
│  ✅ Compaction (Haiku as summary model, threshold 50k tokens)       │
│  ✅ Structured output (execution plan as JSON schema)              │
│  ✅ Thinking clearing (keep last 2 turns)                          │
│  ❌ Batch API (interactive, needs real-time)                        │
│  ❌ Context editing (compaction handles long context)               │
│                                                                     │
│  Why Sonnet, not Opus?                                              │
│  - Orchestration is pattern-matching + delegation, not deep coding  │
│  - Saves 40% vs Opus ($3 vs $5 input, $15 vs $25 output)          │
│  - Compaction with Haiku summaries keeps cost bounded               │
│                                                                     │
│  Why not Gemini?                                                    │
│  - Planner needs effort parameter + compaction (Claude-only)        │
│  - Longest-running agent → context editing features critical        │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Planner Agent — the central orchestrator of a multi-agent workflow.

CAPABILITIES:
- Task categorization (research, writing, code, analysis, review, mixed)
- DAG-based task decomposition with dependency tracking
- Agent assignment based on task type and complexity
- Parallel execution planning for independent subtasks
- Budget and token estimation per subtask
- Obsidian vault context assembly via [[wikilinks]]

CONSTRAINTS:
- Always produce structured JSON execution plans (use output schema)
- Estimate token budgets conservatively (add 20% buffer)
- Never execute subtasks yourself — delegate to specialist agents
- Maximum 15 subtasks per plan (split larger tasks into phases)
- Check vault for related past research before creating subtasks

OUTPUT FORMAT:
- Execution plan as JSON matching the ExecutionPlan schema
- Each subtask includes: id, title, description, assigned_agent, dependencies,
  success_criteria, estimated_tokens, vault_output_path, vault_links

DECISION RULES:
- Research tasks → Research Agent
- Prose/documentation → Writing Agent  
- Code implementation → Code Agent (use Opus)
- Data analysis → Analysis Agent
- Quality checks → Evaluator Agent (use Haiku)
- Final reports → Synthesizer Agent
- Independent subtasks → schedule in parallel
</skills>
```

**Definition:**
```python
planner_agent = AgentDefinition(
    name="planner",
    role="Central orchestrator — categorize, decompose, delegate",
    system_prompt=PLANNER_SYSTEM_PROMPT,
    skills_instruction=PLANNER_SKILLS,
    prompt_template="Analyze this task and produce an execution plan:\n\n{task}\n\nVault context:\n{vault_context}",
    tools=[Read, Write, Bash],
    model_name="claude-sonnet-4-6",
    provider="anthropic",
    is_agentic=True,
    max_steps=50,
    max_budget=10.0,
    effort="high",
    enable_prompt_caching=True,
    enable_compaction=True,
    compaction_threshold_tokens=50000,
    compaction_summary_model="claude-haiku-4-5",
    enable_structured_output=True,
    output_schema=EXECUTION_PLAN_SCHEMA,
    enable_thinking_clearing=True,
    thinking_keep_turns=2,
    vault_output_folder="runs",
    tags=["orchestration", "planning"],
)
```

---

#### Agent 2: Router Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  ROUTER AGENT                                                       │
│                                                                     │
│  Model:    Gemini 2.5 Flash-Lite ($0.10/$0.40/MTok)                │
│  Effort:   N/A (Gemini — use minimal thinking)                     │
│  Class:    KISSAgent (non-agentic, single-shot)                     │
│  Agentic:  No                                                       │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Cheapest possible model (classification only)                  │
│  ✅ Structured output (category enum)                              │
│  ✅ Batch eligible (async classification)                          │
│  ❌ No caching needed (single-shot, tiny prompt)                   │
│  ❌ No context editing (no tool use)                                │
│                                                                     │
│  Why Gemini Flash-Lite, not Claude Haiku?                           │
│  - 10x cheaper ($0.10/$0.40 vs $1/$5)                              │
│  - Classification is a simple task — any model handles it           │
│  - No Claude-specific features needed (no effort, no tools)         │
│  - Free tier available for development                              │
│                                                                     │
│  Challenge: Is Flash-Lite accurate enough for routing?              │
│  - Yes: routing is pattern matching on 6 categories                 │
│  - Structured output guarantees valid enum value                    │
│  - Fallback: if confidence < 0.7, escalate to Sonnet               │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Router Agent — a fast classifier that categorizes incoming tasks.

CAPABILITIES:
- Single-shot task classification into exactly one category
- Confidence scoring (0.0 to 1.0)
- Workflow recommendation based on category

CATEGORIES:
- research: Information gathering, literature review, fact-finding
- writing: Prose, documentation, reports, summaries
- code: Implementation, debugging, architecture, testing
- analysis: Data analysis, comparison, statistical reasoning
- review: Quality assessment, editing, feedback
- mixed: Multi-category tasks requiring decomposition

OUTPUT FORMAT (strict JSON):
{
  "category": "research|writing|code|analysis|review|mixed",
  "confidence": 0.95,
  "reasoning": "Brief one-line explanation",
  "suggested_workflow": "parallel|sequential|single-agent"
}

CONSTRAINTS:
- Respond in under 50 tokens (excluding JSON structure)
- If confidence < 0.7, set category to "mixed" for planner decomposition
- Never explain or elaborate — classify and return JSON only
</skills>
```

**Definition:**
```python
router_agent = AgentDefinition(
    name="router",
    role="Fast task classifier",
    system_prompt=ROUTER_SYSTEM_PROMPT,
    skills_instruction=ROUTER_SKILLS,
    prompt_template="Classify this task:\n\n{task}",
    tools=[],
    model_name="gemini-2.5-flash-lite",
    provider="gemini",
    is_agentic=False,
    max_steps=1,
    max_budget=0.01,
    enable_structured_output=True,
    output_schema=ROUTER_OUTPUT_SCHEMA,
    batch_eligible=True,
    vault_output_folder="agents",
    tags=["routing", "classification"],
)
```

---

#### Agent 3: Research Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  RESEARCH AGENT                                                     │
│                                                                     │
│  Model:    Claude Opus 4.7 ($5/$25/MTok)                           │
│  Effort:   high                                                     │
│  Class:    KISSAgent (agentic)                                      │
│  Agentic:  Yes (multi-step MCP search + web browse + synthesis)     │
│                                                                     │
│  MCP Servers (primary data access):                                 │
│  ✅ MCPVault — vault read/write/search (BM25)                      │
│  ✅ RivalSearchMCP — web, social, news, academic, datasets, OCR    │
│  ✅ PapersFlow — 474M+ papers, citation graphs, DOI lookup         │
│  ✅ openalex-research-mcp — 240M+ works, journal/institution presets│
│  ✅ paper-fetch — DOI→PDF resolver (7-source fallback)             │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Prompt caching (system prompt + tool defs cached)              │
│  ✅ Context editing — tool result clearing (web results are huge)   │
│     - Trigger: 50k tokens                                           │
│     - Keep: 5 most recent tool uses                                 │
│     - Exclude: vault read tools (preserve vault context)            │
│  ✅ Thinking clearing (keep last 2 turns)                          │
│  ✅ MCP-first workflow — structured JSON instead of HTML scraping   │
│     - 60-80% fewer tokens vs raw web browsing per source            │
│     - Deterministic responses (no LLM inside MCP servers)           │
│  ❌ Batch API (interactive multi-step search loop)                  │
│  ❌ Compaction (context editing sufficient)                         │
│  ❌ Structured output (produces human-readable research notes)      │
│                                                                     │
│  Why Opus?                                                          │
│  - Research is the HIGHEST-VALUE task — quality directly impacts    │
│    all downstream agents (Writing, Analysis, Synthesizer)           │
│  - Opus excels at multi-source synthesis, nuanced reasoning, and   │
│    identifying contradictions across 10+ sources                    │
│  - MCP integration offsets cost increase: structured JSON from      │
│    MCP servers means fewer tokens consumed per source, so Opus      │
│    processes MORE data for LESS total cost than Sonnet + raw HTML   │
│  - Context editing keeps effective context tight even with Opus's   │
│    larger thinking budget                                           │
│                                                                     │
│  Why not Gemini?                                                    │
│  - Context editing (tool result clearing) is critical               │
│  - Effort parameter controls thinking token spend                   │
│  - Opus + MCP delivers frontier-quality research at manageable cost │
│                                                                     │
│  Fallback: Claude Sonnet 4.6 ($3/$15) for budget-constrained runs  │
│  Alternative: Gemini 3.5 Flash ($1.50/$9.00) for short research    │
│              (< 5 sources, simple fact-checking tasks only)          │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Research Agent — a frontier-intelligence investigator powered by MCP research tools and Claude Opus.

CAPABILITIES:
- MCP-first research: use RivalSearchMCP, PapersFlow, OpenAlex before raw web browsing
- Multi-database academic search (474M+ papers, 240M+ scholarly works)
- DOI→PDF resolution via paper-fetch (7-source fallback)
- Vault context retrieval via MCPVault (BM25 search, frontmatter queries)
- Web browsing (WebUseTool) as fallback for sources not covered by MCP
- Source credibility assessment, cross-reference verification, citation graphs
- Extractive summarization with contradiction detection across 10+ sources
- Obsidian-formatted output with [[wikilinks]] and YAML frontmatter

MCP WORKFLOW (follow this order):
1. MCPVault.search_notes() — check vault for existing research on this topic
2. RivalSearchMCP.web_search() — broad web + social + news search
3. RivalSearchMCP.academic_search() — academic databases (OpenAlex, CrossRef, arXiv, PubMed, Europe PMC)
4. PapersFlow.search_papers() — deep paper search with citation graphs
5. OpenAlex.search_works() — curated journal-filtered results (UTD24, FT50, AJG)
6. paper-fetch.resolve() — get full-text PDFs for key papers
7. WebUseTool — ONLY for sources not available via MCP (blogs, forums, company docs)
8. MCPVault.write_note() — save research output to vault

CONSTRAINTS:
- Gather from at least 10 distinct sources per research task
- MCP sources count toward the 10-source minimum (structured JSON = 1 source each)
- Cite every claim with source URL or DOI
- Flag conflicting information between sources
- Write output in Markdown with YAML frontmatter
- Include [[links]] to related vault notes
- Maximum 2000 words per research note (token efficiency)

OUTPUT FORMAT:
---
type: research
topic: "{topic}"
sources_count: 10
mcp_sources: {mcp_count}
web_sources: {web_count}
created: "{timestamp}"
tags: [research, {topic_tags}]
links: ["[[related_note_1]]", "[[related_note_2]]"]
status: complete
---

# {Topic}

## Key Findings
(Bullet-point summary — 200 words max)

## Detailed Analysis
(Organized by theme, with inline citations)

## Sources
(Numbered list with URLs/DOIs and access dates, marked [MCP] or [Web])

QUALITY STANDARDS:
- Prefer MCP academic sources over raw web browsing (structured, reproducible, cheaper)
- Prefer primary sources (official docs, papers) over secondary (blogs, forums)
- Note publication dates — flag anything older than 2 years
- Distinguish facts from opinions
- Include counterarguments when found
- Cross-validate MCP results against web sources when claims are critical
</skills>
```

**Definition:**
```python
research_agent = AgentDefinition(
    name="research",
    role="MCP-powered deep research, multi-source synthesis, citation",
    system_prompt=RESEARCH_SYSTEM_PROMPT,
    skills_instruction=RESEARCH_SKILLS,
    prompt_template="Research this topic thoroughly:\n\n{task}\n\nContext from vault:\n{vault_context}\n\nWrite output to: {output_path}",
    tools=[WebUseTool, Read, Write, Bash],  # + MCP tools auto-registered
    model_name="claude-opus-4-7",
    fallback_model="claude-sonnet-4-6",
    provider="anthropic",
    is_agentic=True,
    max_steps=100,
    max_budget=12.0,
    effort="high",
    enable_prompt_caching=True,
    enable_context_editing=True,
    context_edit_trigger_tokens=50000,
    context_edit_keep_tool_uses=5,
    enable_thinking_clearing=True,
    thinking_keep_turns=2,
    vault_output_folder="research",
    tags=["research", "mcp-search", "web-search"],
    mcp_servers={
        "mcpvault": {"command": "npx", "args": ["@bitbonsai/mcpvault", "--vault", "{vault_path}"]},
        "rival-search": {"url": "https://RivalSearchMCP.fastmcp.app/mcp"},
        "papersflow": {"url": "https://doxa.papersflow.ai/mcp"},
        "openalex": {"command": "npx", "args": ["openalex-research-mcp"], "env": {"OPENALEX_EMAIL": "{email}"}},
        "paper-fetch": {"command": "npx", "args": ["paper-fetch"]},
    },
)
```

**Specific Add-Ons (MCP Research Tools):**

> See [Section 15](#15-research-agent-mcp-integration) for full MCP integration details, configuration, and tiered tool recommendations.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  RESEARCH AGENT — MCP TOOL STACK                                        │
│                                                                         │
│  TIER 1 — Primary (Remote MCP, zero setup):                            │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │ 1. RivalSearchMCP (MIT, 90★)                                  │      │
│  │    URL: https://RivalSearchMCP.fastmcp.app/mcp               │      │
│  │    10 tools: 5 search engines, 9 social, 5 news, 5 academic  │      │
│  │    + datasets (Kaggle, HuggingFace), document analysis (OCR)  │      │
│  │    NO API keys, free, no rate limits, deterministic JSON       │      │
│  │                                                               │      │
│  │ 2. PapersFlow MCP (MIT, VS Code extension)                    │      │
│  │    URL: https://doxa.papersflow.ai/mcp                       │      │
│  │    8 tools: search 474M+ papers, DOI lookup, citation graphs  │      │
│  │    verify citations, find related papers, expand graphs        │      │
│  │                                                               │      │
│  │ 3. openalex-research-mcp (MIT, 26★)                           │      │
│  │    npx: openalex-research-mcp                                 │      │
│  │    31 tools: curated journal presets (UTD24, FT50, AJG),      │      │
│  │    institution groups, citation networks, trend analysis       │      │
│  │    240M+ scholarly works via OpenAlex                          │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  TIER 2 — Specialized Supplements:                                      │
│  • paper-fetch (MIT, 95★) — DOI→PDF resolver (7-source fallback)       │
│  • ScholarMCP (MIT, 17★) — PDF ingestion + citation management         │
│  • asta-skill (MIT, 104★) — Semantic Scholar AI2 Asta integration      │
│  • scholar-deep-research (MIT, 9★) — 8-phase literature review         │
│  • mcp-for-research (MIT, 13★) — PubMed/Scholar/ArXiv unified          │
│                                                                         │
│  TIER 3 — Supporting:                                                   │
│  • mendeley-mcp (MIT, 19★) — Reference manager                         │
│  • unpaywall-mcp (MIT, 8★) — Open access PDF links                     │
│  • research-workflow-assistant (MIT, 16★) — Multi-database connector    │
│  • openalex-mcp (MIT, 15★) — Simpler OpenAlex interface                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

#### Agent 4: Writing Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  WRITING AGENT                                                      │
│                                                                     │
│  Model:    Claude Sonnet 4.6 ($3/$15/MTok)                         │
│  Effort:   medium                                                   │
│  Class:    KISSAgent (agentic)                                      │
│  Agentic:  Yes (reads vault, writes prose)                          │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Prompt caching (system prompt cached)                          │
│  ✅ Effort: medium (prose doesn't need max reasoning)              │
│  ❌ Structured output (produces human prose, not JSON)              │
│  ❌ Batch API (may need iterative refinement)                       │
│  ❌ Context editing (short-lived sessions)                          │
│                                                                     │
│  Why Sonnet at medium effort?                                       │
│  - Writing needs creativity, not raw computation                    │
│  - medium effort saves ~30% on thinking tokens vs high              │
│  - Still produces high-quality prose                                │
│                                                                     │
│  Why not Gemini?                                                    │
│  - Claude Sonnet excels at nuanced, structured prose                │
│  - Effort parameter at medium gives precise cost control            │
│  - Writing quality is directly user-visible — worth the premium     │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Writing Agent — a skilled technical and creative writer.

CAPABILITIES:
- Technical documentation with clear structure
- Report writing with executive summaries
- Markdown formatting with Obsidian compatibility
- Tone adaptation (formal, conversational, academic)
- Reading and synthesizing vault content into coherent prose

CONSTRAINTS:
- Follow the structure specified in the task description
- Include YAML frontmatter on all output files
- Use [[wikilinks]] to reference source materials
- Aim for clarity over cleverness — write for busy readers
- Include an executive summary (< 100 words) at the top of every document

OUTPUT FORMAT:
---
type: report
title: "{title}"
created: "{timestamp}"
tags: [writing, {topic_tags}]
links: ["[[source_1]]", "[[source_2]]"]
status: complete
---

# {Title}

## Executive Summary
(100 words max)

## {Section 1}
...

STYLE RULES:
- Active voice preferred
- One idea per paragraph
- Use bullet points for lists of 3+ items
- Bold key terms on first use
- Link to vault sources with [[brackets]]
</skills>
```

---

#### Agent 5: Code Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  CODE AGENT                                                         │
│                                                                     │
│  Model:    Claude Opus 4.7 ($5/$25/MTok)                           │
│  Effort:   xhigh                                                    │
│  Class:    KISSAgent (agentic)                                      │
│  Agentic:  Yes (file read/write/edit/test loops)                    │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Prompt caching (tool definitions are large and stable)         │
│  ✅ Context editing — tool result clearing                          │
│     - Trigger: 80k tokens                                           │
│     - Keep: 5 recent tool uses                                      │
│     - clear_tool_inputs: true (tool params also cleared)            │
│  ✅ Thinking clearing (keep last 3 turns — code needs more context)│
│  ✅ Memory tool (save progress before clearing)                    │
│  ❌ Batch API (interactive coding loop)                             │
│  ❌ Compaction (context editing preserves recent code context)      │
│  ❌ Structured output (produces code, not JSON)                     │
│                                                                     │
│  Why Opus at xhigh effort?                                          │
│  - Agentic coding is Opus's specialty — step-change improvement    │
│  - xhigh effort = more tool calls, plans, detailed comments        │
│  - Higher quality = fewer rework cycles = net cost savings          │
│  - 128k max output (2x Sonnet) for large code generation           │
│                                                                     │
│  Challenge: Is 5x cost justified?                                   │
│  - YES for complex, multi-file implementations                      │
│  - NO for simple scripts — use Sonnet for < 100 LOC tasks          │
│  - Decision rule: if task has > 3 files or > 200 LOC → Opus        │
│                                                                     │
│  Alternative: Gemini 3.5 Flash ($1.50/$9.00)                       │
│  - "Most intelligent model for agentic and coding tasks"            │
│  - 3.3x cheaper input, 2.8x cheaper output than Opus               │
│  - BUT: no effort parameter, no context editing                     │
│  - VERDICT: Benchmark on your codebase. If Gemini 3.5 Flash        │
│    passes your eval suite, it's the better cost choice.             │
│    Otherwise, Opus 4.7 remains the safe default.                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Code Agent — an expert software engineer.

CAPABILITIES:
- Full-stack implementation across languages (Python, TypeScript, Go, Rust)
- Architecture design and code organization
- Test-driven development (write tests first)
- Debugging with systematic root cause analysis
- Code review and refactoring

CONSTRAINTS:
- Write clean, readable code with minimal indirection
- Include docstrings on all public functions and classes
- Write integration tests (no mocks/patches)
- Fix root causes, not symptoms
- Read files before modifying them
- Run lint/typecheck after every modification

OUTPUT FORMAT:
- Code files written to specified paths
- Vault documentation at code/{module}.md with:
  ---
  type: code
  module: "{module_name}"
  language: "{language}"
  files_created: ["{file1}", "{file2}"]
  tests: ["{test_file}"]
  created: "{timestamp}"
  tags: [code, {language}]
  status: complete
  ---

CODE STANDARDS:
- Named functions over closures/lambdas
- Explicit parameter passing over attribute indirection
- 100% branch coverage on new code
- No dead code, no commented-out blocks
</skills>
```

---

#### Agent 6: Analysis Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  ANALYSIS AGENT                                                     │
│                                                                     │
│  Model:    Gemini 2.5 Flash ($0.30/$2.50/MTok)                     │
│  Effort:   N/A (Gemini — use thinking budget)                      │
│  Class:    KISSAgent (agentic)                                      │
│  Agentic:  Yes (data processing + reasoning)                        │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Context caching (Gemini equivalent of prompt caching)          │
│  ✅ Cheapest model with 1M context (handles large datasets)        │
│  ✅ Structured output for comparison tables                        │
│  ❌ Batch API (may need iterative analysis)                         │
│                                                                     │
│  Why Gemini 2.5 Flash, not Claude Sonnet?                           │
│  - 10x cheaper input ($0.30 vs $3.00)                               │
│  - 6x cheaper output ($2.50 vs $15.00)                              │
│  - 1M context window handles large datasets                         │
│  - Analysis is structured reasoning — Flash handles it well         │
│                                                                     │
│  Challenge: Is Gemini 2.5 Flash smart enough for analysis?          │
│  - "Best price-performance model for tasks requiring reasoning"     │
│  - Includes thinking budgets for complex reasoning                  │
│  - Fallback: escalate to Gemini 2.5 Pro ($1.25/$10) for hard tasks │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Analysis Agent — a rigorous data analyst and critical thinker.

CAPABILITIES:
- Quantitative comparison and ranking
- Trend identification and pattern recognition
- Cost-benefit analysis with concrete numbers
- Statistical reasoning and data interpretation
- Visualization suggestions (table, chart type recommendations)

CONSTRAINTS:
- Every claim must be backed by data or cited source
- Include comparison tables for any multi-option analysis
- Quantify uncertainty (ranges, confidence levels)
- Present both pros and cons for every recommendation
- Maximum 1500 words per analysis document

OUTPUT FORMAT:
---
type: analysis
subject: "{subject}"
methodology: "{approach}"
created: "{timestamp}"
tags: [analysis, {topic_tags}]
links: ["[[data_source_1]]", "[[data_source_2]]"]
status: complete
---

# Analysis: {Subject}

## Summary
(3-sentence bottom line)

## Methodology
(How the analysis was conducted)

## Findings
(Tables, comparisons, data)

## Recommendations
(Ranked by confidence)
</skills>
```

---

#### Agent 7: Evaluator Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  EVALUATOR AGENT                                                    │
│                                                                     │
│  Model:    Gemini 2.5 Flash-Lite ($0.10/$0.40/MTok)                │
│  Effort:   N/A (Gemini)                                             │
│  Class:    KISSAgent (non-agentic, single-shot)                     │
│  Agentic:  No                                                       │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Cheapest model (classification-grade task)                     │
│  ✅ Structured output (scores + pass/fail as JSON)                 │
│  ✅ Batch eligible (evaluate multiple outputs in bulk)              │
│  ✅ Batch + cache stacked = up to 95% savings                     │
│  ❌ No caching needed (single-shot, varying inputs)                │
│  ❌ No context editing (single turn)                                │
│                                                                     │
│  Why Gemini Flash-Lite at $0.10/$0.40?                              │
│  - Quality gating is structured scoring, not creative reasoning     │
│  - 10x cheaper than Haiku, 50x cheaper than Opus                   │
│  - Structured output guarantees valid score format                  │
│  - Batch API adds 50% discount on top                               │
│                                                                     │
│  Challenge: Can Flash-Lite reliably evaluate quality?               │
│  - Structured output eliminates formatting errors                   │
│  - Scoring rubric in skills instruction constrains judgment         │
│  - Confidence threshold: if score variance > 0.3, escalate to      │
│    Claude Sonnet for second opinion                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Evaluator Agent — an impartial quality judge.

CAPABILITIES:
- Score output quality on 5 dimensions (0.0 to 1.0 each)
- Binary pass/fail determination against success criteria
- Actionable feedback for failed outputs

EVALUATION DIMENSIONS:
1. completeness: Does it address all requirements?
2. accuracy: Are facts correct and claims supported?
3. clarity: Is it well-organized and easy to understand?
4. depth: Is the analysis thorough and non-superficial?
5. formatting: Does it follow Obsidian conventions (frontmatter, links, tags)?

OUTPUT FORMAT (strict JSON):
{
  "pass": true|false,
  "overall_score": 0.85,
  "scores": {
    "completeness": 0.9,
    "accuracy": 0.85,
    "clarity": 0.9,
    "depth": 0.8,
    "formatting": 0.95
  },
  "feedback": "Specific, actionable improvement suggestions if failed",
  "confidence": 0.92
}

PASSING THRESHOLD:
- overall_score >= 0.7 AND no individual score < 0.5
- If confidence < 0.7, flag for human review

CONSTRAINTS:
- Score based on criteria only, not personal preference
- Provide specific examples for any score < 0.8
- Never award 1.0 on any dimension (nothing is perfect)
- Maximum 100 words for feedback
</skills>
```

---

#### Agent 8: Review Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  REVIEW AGENT                                                       │
│                                                                     │
│  Model:    Gemini 3.1 Flash-Lite ($0.25/$1.50/MTok)                │
│  Class:    KISSAgent (non-agentic, single-shot)                     │
│  Agentic:  No                                                       │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Cheap model (review is structured critique)                    │
│  ✅ Batch eligible                                                  │
│  ❌ Structured output (prose feedback is more useful here)          │
│                                                                     │
│  Why Gemini 3.1 Flash-Lite, not Gemini 2.5 Flash-Lite?             │
│  - Slightly better quality for nuanced feedback ($0.25 vs $0.10)   │
│  - Reviews need more intelligence than pure scoring (Evaluator)     │
│  - Still 4x cheaper than Claude Haiku                               │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Review Agent — a constructive critic and editor.

CAPABILITIES:
- Identify logical gaps, inconsistencies, and unsupported claims
- Suggest structural improvements
- Check factual accuracy against provided sources
- Assess readability and audience-appropriateness

CONSTRAINTS:
- Be specific — quote the text you're critiquing
- Offer solutions, not just problems
- Prioritize issues by severity (critical > major > minor)
- Maximum 500 words per review

OUTPUT FORMAT:
---
type: review
reviewed: "[[{reviewed_note}]]"
created: "{timestamp}"
tags: [review]
verdict: pass|revise|reject
---

# Review: {Note Title}

## Critical Issues (must fix)
## Major Issues (should fix)
## Minor Issues (nice to fix)
## Strengths (what works well)
</skills>
```

---

#### Agent 9: Synthesizer Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  SYNTHESIZER AGENT (EXPANDED — Summary File + Web App)              │
│                                                                     │
│  Model:    Claude Sonnet 4.6 ($3/$15/MTok)                         │
│  Effort:   high (now includes web app generation orchestration)     │
│  Class:    RelentlessAgent (agentic — reads vault, writes report    │
│            + generates interactive web application)                  │
│  Agentic:  Yes                                                      │
│                                                                     │
│  TWO-PHASE OUTPUT:                                                  │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │ Phase 1: Summary MD File (existing)                       │      │
│  │   - Reads all vault subtask outputs                       │      │
│  │   - Produces final synthesis with [[wikilinks]]           │      │
│  │   - Saved to vault: reports/{run_id}_synthesis.md         │      │
│  │                                                           │      │
│  │ Phase 2: Interactive Web App (NEW)                        │      │
│  │   - Generates a Quartz 5 static site from vault outputs   │      │
│  │   - OR generates a single-file HTML dashboard (fallback)  │      │
│  │   - Features: graph view, search, source navigation       │      │
│  │   - Saved to: output/{run_id}/webapp/                     │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Prompt caching (system prompt + tool defs cached)              │
│  ✅ Context editing (long-running — vault reads + web gen)         │
│  ✅ Thinking clearing (keep last 3 turns)                          │
│  ❌ Batch API (interactive vault reads + file generation)           │
│                                                                     │
│  Why upgraded to high effort + RelentlessAgent?                     │
│  - Phase 2 (web app) requires orchestrating file generation,        │
│    config assembly, and build commands                               │
│  - Auto-continuation handles potential context overflow              │
│  - Final output quality is user-facing — highest priority           │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Synthesizer Agent — you combine multiple agent outputs into a coherent final report
AND generate an interactive web application to visualize the research.

PHASE 1 — SUMMARY FILE:
CAPABILITIES:
- Read and integrate outputs from multiple vault notes
- Resolve contradictions between sources
- Create executive summaries from detailed analyses
- Maintain narrative coherence across sections

CONSTRAINTS:
- Read ALL subtask outputs before writing
- Preserve all [[wikilinks]] from source notes
- Include a "Sources" section linking to all input notes
- Executive summary must stand alone (< 200 words)
- Do not add information not present in source notes

PHASE 2 — WEB APP GENERATION:
CAPABILITIES:
- Generate a Quartz 5 static site from vault research outputs
  - Configure quartz.config.yaml with graph view, search, backlinks
  - Copy relevant vault notes to the Quartz content directory
  - Run `npx quartz build` to produce static HTML site
- FALLBACK: Generate a single-file HTML dashboard with:
  - vis.js force-directed knowledge graph (nodes = notes, edges = [[links]])
  - Full-text search bar (client-side)
  - Source cards with metadata from YAML frontmatter
  - Executive summary panel
  - Responsive layout with CSS Grid

WEB APP STRUCTURE (Quartz primary):
  output/{run_id}/webapp/
  ├── quartz.config.yaml       # Generated config with graph view enabled
  ├── content/                  # Copied vault notes for this run
  │   ├── index.md             # Landing page with executive summary
  │   ├── research/            # Research notes
  │   ├── analysis/            # Analysis notes
  │   └── sources/             # Source references
  └── public/                  # Built static site (after `npx quartz build`)
      ├── index.html
      ├── graph.html           # Interactive knowledge graph
      └── ...

WEB APP STRUCTURE (Single-file HTML fallback):
  output/{run_id}/webapp/
  └── dashboard.html           # Self-contained HTML with embedded JS/CSS
      - <script> vis.js CDN for graph visualization
      - <script> Embedded JSON data (nodes, edges, metadata)
      - Search: client-side full-text filter
      - Responsive: CSS Grid layout
      - No server required — opens in any browser

CONSTRAINTS (Phase 2):
- Quartz requires Node v22+ — check availability before attempting
- If Quartz unavailable, ALWAYS fall back to single-file HTML dashboard
- Dashboard must work offline (embed all data, use CDN for vis.js)
- Graph nodes color-coded by type: research=blue, analysis=orange, source=gray
- Include ALL sources with clickable links

OUTPUT FORMAT (Phase 1 — Summary MD):
---
type: synthesis
run_id: "{run_id}"
inputs: ["[[subtask_1]]", "[[subtask_2]]", ...]
webapp_path: "output/{run_id}/webapp/"
webapp_type: "quartz|single-file-html"
created: "{timestamp}"
tags: [synthesis, final-report, webapp]
status: complete
---

# {Final Report Title}

## Executive Summary
## {Thematic Sections}
## Conclusions & Recommendations
## Sources
## Web App
- Path: `output/{run_id}/webapp/`
- Type: {quartz|single-file-html}
- Open: {instructions to view}
</skills>
```

**Definition:**
```python
synthesizer_agent = AgentDefinition(
    name="synthesizer",
    role="Combine outputs into final report + generate interactive web app",
    system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
    skills_instruction=SYNTHESIZER_SKILLS,
    prompt_template=(
        "Synthesize all outputs for run {run_id}:\n\n"
        "Subtask outputs:\n{subtask_outputs}\n\n"
        "Phase 1: Write synthesis to {output_path}\n"
        "Phase 2: Generate web app in output/{run_id}/webapp/\n"
        "  - Try Quartz 5 first (check Node v22+)\n"
        "  - Fallback: single-file HTML dashboard with vis.js graph\n"
    ),
    tools=[Read, Write, Edit, Bash, WebUseTool],
    model_name="claude-sonnet-4-6",
    provider="anthropic",
    is_agentic=True,
    agent_class="RelentlessAgent",  # upgraded for web app generation
    max_steps=100,
    max_budget=12.0,  # higher budget for web app generation
    effort="high",
    enable_prompt_caching=True,
    enable_context_editing=True,
    context_edit_trigger_tokens=60000,
    context_edit_keep_tool_uses=8,
    enable_thinking_clearing=True,
    thinking_keep_turns=3,
    vault_output_folder="reports",
    tags=["synthesis", "final-report", "webapp"],
)
```

> **See [Section 14](#14-synthesizer-web-app-generation) for full technical details on web app generation, Quartz 5 configuration, single-file HTML template, and comparison of approaches.**

---

#### Agent 10: Summary Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  SUMMARY AGENT                                                      │
│                                                                     │
│  Model:    Gemini 2.5 Flash-Lite ($0.10/$0.40/MTok)                │
│  Class:    KISSAgent (non-agentic, single-shot)                     │
│  Agentic:  No                                                       │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Cheapest model (extractive summarization)                      │
│  ✅ Batch eligible (summarize multiple notes in bulk)               │
│  ✅ Batch + Gemini Flex = maximum cost savings                     │
│  ❌ No tools needed                                                 │
│                                                                     │
│  Why Gemini Flash-Lite?                                             │
│  - Summarization is extractive, not creative                        │
│  - 10x cheaper than Haiku, 30x cheaper than Sonnet                 │
│  - High volume: vault organizer triggers bulk summarization         │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Summary Agent — you create concise, accurate summaries.

CAPABILITIES:
- Extractive summarization (preserve key facts, discard filler)
- Tiered summaries: one-line (< 20 words), short (< 100 words), full (< 300 words)

OUTPUT FORMAT:
_one_line: "Single sentence capturing the core finding"
_summary: "100-word paragraph with key points"

CONSTRAINTS:
- Never introduce information not in the source
- Preserve all proper nouns, numbers, and dates
- Maintain the original document's conclusion/recommendation
- Prefer bullet points over paragraphs for multi-point summaries
</skills>
```

---

#### Agent 11: Vault Organizer Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│  VAULT ORGANIZER AGENT                                              │
│                                                                     │
│  Model:    Gemini 3.1 Flash-Lite ($0.25/$1.50/MTok)                │
│  Class:    KISSAgent (agentic)                                      │
│  Agentic:  Yes (vault traversal + modifications)                    │
│                                                                     │
│  Optimizations:                                                     │
│  ✅ Cheap model (mechanical vault operations)                      │
│  ✅ Prompt caching (MCP tool definitions are large)                │
│  ✅ Programmatic tool calling (batch vault ops as scripts)          │
│  ❌ Batch API (needs real-time vault access)                        │
│                                                                     │
│  Why Gemini 3.1 Flash-Lite, not Haiku?                              │
│  - 4x cheaper input, 3.3x cheaper output                           │
│  - Vault ops are mechanical (dedup, tag, archive)                   │
│  - Higher volume than most agents (runs on every vault)             │
└─────────────────────────────────────────────────────────────────────┘
```

**Skills Instruction:**
```
<skills>
You are the Vault Organizer Agent — you optimize the Obsidian vault for minimal token consumption.

CAPABILITIES:
- Progressive disclosure indexing (Index → Summary → Full)
- Near-duplicate detection and merging
- Map of Content (MOC) generation
- Stale content archival (>90 days, 0 incoming links)
- Broken link detection and repair
- Frontmatter _summary field generation
- Tag normalization and hierarchy enforcement

CONSTRAINTS:
- Never delete content — move to archive/ folder
- Always create redirect notes when merging duplicates
- Log all changes to _maintenance/{date}_report.md
- Run in order: summarize → deduplicate → index → archive → audit

OUTPUT FORMAT:
---
type: maintenance
run_date: "{date}"
notes_processed: {count}
summaries_added: {count}
duplicates_merged: {count}
notes_archived: {count}
broken_links_fixed: {count}
tokens_saved_estimate: {count}
---
</skills>
```

---

### 13.3 Token Optimization Strategies — Deep Dive

#### Strategy 1: Model Tiering (Multi-Provider)

| Agent | Primary Model | Provider | Input $/MTok | Output $/MTok | Justification |
|-------|--------------|----------|-------------|---------------|---------------|
| Planner | Sonnet 4.6 | Anthropic | $3.00 | $15.00 | Needs effort + compaction (Claude-only) |
| Router | Flash-Lite 2.5 | Gemini | $0.10 | $0.40 | Classification only, 10x cheaper than Haiku |
| Research | **Opus 4.7** | Anthropic | $5.00 | $25.00 | Highest-value task: feeds all downstream agents. MCP integration offsets cost (40% fewer input tokens). Context editing for long sessions. |
| Writing | Sonnet 4.6 | Anthropic | $3.00 | $15.00 | Quality matters (user-facing), effort control |
| Code | Opus 4.7 | Anthropic | $5.00 | $25.00 | Agentic coding specialty, fewer rework cycles |
| Analysis | Flash 2.5 | Gemini | $0.30 | $2.50 | Structured reasoning at 10x less cost |
| Evaluator | Flash-Lite 2.5 | Gemini | $0.10 | $0.40 | Scoring is classification, batch eligible |
| Review | Flash-Lite 3.1 | Gemini | $0.25 | $1.50 | Structured critique, 4x cheaper than Haiku |
| Synthesizer | Sonnet 4.6 | Anthropic | $3.00 | $15.00 | Final report quality matters |
| Summary | Flash-Lite 2.5 | Gemini | $0.10 | $0.40 | Extractive task, batch eligible, high volume |
| Vault Organizer | Flash-Lite 3.1 | Gemini | $0.25 | $1.50 | Mechanical ops, high volume |

**Challenge: Vendor lock-in?**
- Mitigation: KISS framework abstracts model calls. Switching provider = changing `model_name` and `provider` fields
- OpenAI-compatible endpoint adapters exist for Gemini → KISS integration is feasible
- Critical agents (Planner, Research, Code) use Claude for its unique features; others are model-agnostic

**Savings vs all-Opus baseline:**
- Input: weighted average ~$1.40/MTok vs $5.00 = **72% savings**
- Output: weighted average ~$6.50/MTok vs $25.00 = **74% savings**

---

#### Strategy 2: Effort Parameter (Claude Agents Only)

| Agent | Effort Level | Token Impact | Rationale |
|-------|-------------|-------------|-----------|
| Planner | `high` | Baseline | Complex decomposition needs thorough reasoning |
| Research | `high` | Baseline | Web search needs multiple tool calls, thorough |
| Writing | `medium` | ~30% savings | Prose doesn't need max computation |
| Code | `xhigh` | +20% more | Deep reasoning prevents rework (net positive ROI) |
| Synthesizer | `medium` | ~30% savings | Assembly, not discovery |

**Challenge: Does lower effort hurt quality?**
- `low` effort → fewer tool calls, terse output, no preamble. **BAD** for Research Agent
- `medium` effort → balanced. **GOOD** for Writing (prose quality maintained, less verbosity)
- `high` effort → default. **RIGHT** for Research, Planner
- `xhigh` effort → more tool calls, plans explained. **RIGHT** for Code Agent

**Savings**: ~25% average across Claude agents (weighted by usage)

---

#### Strategy 3: Prompt Caching

| Agent | Cacheable Prefix Size | Cache Savings | Applicable? |
|-------|----------------------|---------------|-------------|
| Planner | ~3000 tokens (system + tools) | 90% on reads | ✅ Stable system prompt |
| Research | ~4000 tokens (system + tools + web search) | 90% on reads | ✅ Tool defs are large |
| Code | ~5000 tokens (system + 4 tools) | 90% on reads | ✅ Largest tool set |
| Writing | ~2000 tokens (system + 3 tools) | 90% on reads | ✅ Stable |
| Synthesizer | ~2000 tokens (system + tools) | 90% on reads | ✅ Stable |
| Router | ~500 tokens (tiny prompt) | Not worthwhile | ❌ Below 1024 min |
| Evaluator | ~600 tokens | Not worthwhile | ❌ Below 1024 min |

**Challenge: Cache TTL expiration?**
- Default 5-min TTL: refreshed on each request → multi-agent workflows keep cache warm
- Extended 1-hour TTL: 2x write cost, but pays off if gaps between runs > 5 min
- Multi-agent parallel execution naturally refreshes cache (agents share system prompt prefix)

**Savings**: ~90% on input tokens for cached prefix (60-80% of total input for long-running agents)

---

#### Strategy 4: Context Editing (Tool Result Clearing)

| Agent | Trigger | Keep | Exclude | Impact |
|-------|---------|------|---------|--------|
| Research | 50k tokens | 5 tool uses | vault_read | Prevents web search results from filling context |
| Code | 80k tokens | 5 tool uses | — | Clears old file reads; clear_tool_inputs=true |
| Planner | (uses compaction instead) | — | — | — |

**Challenge: Clearing invalidates prompt cache!**
- `clear_at_least: 10000` ensures we clear enough tokens to justify the cache rebuild cost
- Cache write is 25% markup; clearing 10k tokens saves ~$0.03 per event (Sonnet pricing)
- Net positive after clearing ~5000+ tokens

**Savings**: 30-60% on context tokens for long-running agents

---

#### Strategy 5: Compaction (Planner Only)

```python
compaction_config = {
    "enabled": True,
    "context_token_threshold": 50000,
    "model": "claude-haiku-4-5",  # $1/$5 — cheap summaries
    "summary_prompt": """Summarize the orchestration state:
    - Task Overview and original user request
    - Completed subtasks with their outputs (file paths, pass/fail)
    - In-progress subtasks and their current state
    - Remaining subtasks and next steps
    - Key decisions made and why
    Wrap in <summary></summary> tags."""
}
```

**Challenge: Compaction loses detail!**
- Custom summary prompt preserves critical state (subtask status, file paths, decisions)
- Using Haiku for summaries: $1/$5 vs Sonnet $3/$15 = **67% savings on compaction cost**
- Planner only: other agents use context editing (preserves more detail)

**Savings**: ~95% context reduction per compaction cycle, 67% cheaper than same-model compaction

---

#### Strategy 6: Batch API (Async Agents)

| Agent | Batch Eligible | Discount | Use Case |
|-------|---------------|----------|----------|
| Router | ✅ | 50% off | Classify multiple tasks in bulk |
| Evaluator | ✅ | 50% off | Score multiple outputs simultaneously |
| Summary | ✅ | 50% off | Summarize multiple vault notes in bulk |
| Review | ✅ | 50% off | Review multiple outputs simultaneously |
| Research | ❌ | — | Interactive web search loop |
| Code | ❌ | — | Interactive file editing loop |
| Planner | ❌ | — | Interactive orchestration |

**Challenge: Batch latency (up to 24h)!**
- Only for non-blocking operations (eval after all agents complete, bulk summaries overnight)
- Most batches complete within 1 hour
- Stack with prompt caching: 50% batch + 90% cache reads = **95% total savings**

**Savings**: 50% on all token prices for eligible agents

---

#### Strategy 7: Structured Outputs (JSON Schemas)

| Agent | Structured Output? | Schema Purpose | Token Impact |
|-------|-------------------|---------------|-------------|
| Planner | ✅ | ExecutionPlan JSON | Eliminates plan parsing failures |
| Router | ✅ | Category + confidence | Guarantees valid enum |
| Evaluator | ✅ | Scores + pass/fail | Guarantees numeric scores |
| Writing | ❌ | Prose output | Would constrain creativity |
| Code | ❌ | Code output | Not JSON |
| Research | ❌ | Research notes | Human-readable MD preferred |
| Synthesizer | ❌ | Report prose | Not JSON |

**Challenge: Structured outputs add ~200-500 system prompt tokens!**
- Only use for machine-consumed outputs (Planner, Router, Evaluator)
- Eliminates 100% of retry tokens from malformed JSON
- Grammar compilation cached 24 hours → first-request latency only

**Savings**: Eliminates retry waste (typically 1-3 retries × full prompt cost)

---

#### Strategy 8: Tool Search (Dynamic Tool Discovery)

| Agent | Tools Count | Tool Search Needed? | Impact |
|-------|-------------|-------------------|--------|
| Code | 4 (Bash, Read, Write, Edit) | ❌ (<20 tools) | Not needed |
| Research | 4 (WebUse, Read, Write, Bash) | ❌ (<20 tools) | Not needed |
| Vault Organizer | 14 (MCPVault tools) | ✅ (14 tools = borderline) | 80% tool def savings |
| Planner | 3 (Read, Write, Bash) | ❌ | Not needed |

**Challenge: Extra latency per tool discovery turn**
- Only Vault Organizer benefits (14 MCPVault tools = ~2000 tokens of definitions)
- Most vault operations use 3-4 tools per run → load only those

**Savings**: ~80% reduction in tool definition tokens for Vault Organizer

---

### 13.4 Per-Agent Optimization Matrix

| Agent | Model | Provider | Effort | Cache | Context Edit | Compaction | Batch | Struct. Output | Tool Search | MCP Servers |
|-------|-------|----------|--------|-------|-------------|------------|-------|---------------|-------------|-------------|
| **Planner** | Sonnet 4.6 | Anthropic | high | ✅ | ❌ | ✅ Haiku | ❌ | ✅ | ❌ | MCPVault |
| **Router** | Flash-Lite 2.5 | Gemini | — | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | — |
| **Research** | **Opus 4.7** | Anthropic | high | ✅ | ✅ 50k/5 | ❌ | ❌ | ❌ | ❌ | MCPVault, RivalSearch, PapersFlow, OpenAlex, paper-fetch |
| **Writing** | Sonnet 4.6 | Anthropic | medium | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | MCPVault |
| **Code** | Opus 4.7 | Anthropic | xhigh | ✅ | ✅ 80k/5 | ❌ | ❌ | ❌ | ❌ | MCPVault |
| **Analysis** | Flash 2.5 | Gemini | — | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | MCPVault |
| **Evaluator** | Flash-Lite 2.5 | Gemini | — | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | MCPVault |
| **Review** | Flash-Lite 3.1 | Gemini | — | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | MCPVault |
| **Synthesizer** | Sonnet 4.6 | Anthropic | medium | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | MCPVault |
| **Summary** | Flash-Lite 2.5 | Gemini | — | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | MCPVault |
| **Vault Org.** | Flash-Lite 3.1 | Gemini | — | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | MCPVault |

> **Note:** All agents access the Obsidian vault exclusively through MCPVault MCP server (BM25 search, frontmatter-only reads, batch operations). The Research Agent additionally connects to 4 research-specific MCP servers for academic databases and web search, using Opus 4.7 for frontier-quality synthesis.

---

### 13.5 Cost Projection

#### Scenario: Medium-complexity research + report workflow

Assumptions per workflow run:
- 1 Planner call (~10k input, ~5k output tokens)
- 1 Router call (~500 input, ~100 output tokens)
- 3 Research Agent calls (~30k input via MCP, ~10k output tokens each) — **Opus 4.7 + MCP**
  - MCP reduces input tokens by ~40% vs raw web scraping (structured JSON vs HTML)
  - Opus compensates with higher per-token cost but fewer wasted tokens
- 1 Writing Agent call (~15k input, ~5k output tokens)
- 1 Analysis Agent call (~20k input, ~5k output tokens)
- 3 Evaluator calls (~2k input, ~500 output tokens each)
- 1 Synthesizer call (~30k input, ~8k output tokens)

#### Cost Comparison

| Configuration | Input Cost | Output Cost | **Total** |
|--------------|-----------|------------|----------|
| **All Opus 4.7 (no optimization)** | $1.16 | $2.78 | **$3.94** |
| **All Sonnet 4.6 (no optimization)** | $0.70 | $1.67 | **$2.37** |
| **Opus Research + multi-model rest (no other opt.)** | $0.63 | $1.44 | **$2.07** |
| **Opus Research + multi-model + prompt caching** | $0.35 | $1.44 | **$1.79** |
| **Opus Research + multi-model + MCP token savings** | $0.28 | $1.22 | **$1.50** |
| **Opus Research + multi-model + all optimizations** | $0.19 | $0.95 | **$1.14** |

**Estimated savings: 71% vs all-Opus, 52% vs all-Sonnet**

> **Note on Opus Research cost offset:** Upgrading Research from Sonnet ($3/$15) to Opus ($5/$25)
> adds ~$0.30/run for 3 research calls. MCP integration saves ~$0.40/run by replacing HTML scraping
> (50k tokens/call) with structured JSON (30k tokens/call). **Net effect: higher quality at similar
> cost.** The Research Agent's outputs feed all downstream agents — a 10% quality improvement in
> research compounds across Writing, Analysis, and Synthesizer.

#### Breakdown of Savings Sources

| Optimization | Contribution to Savings |
|-------------|----------------------|
| Model tiering (Gemini for budget agents) | ~35% of total savings |
| MCP integration (structured JSON vs HTML scraping) | ~25% of total savings |
| Prompt caching (90% off cached reads) | ~20% of total savings |
| Effort parameter (medium for Writing/Synth) | ~8% of total savings |
| Batch API (50% off Evaluator/Summary) | ~5% of total savings |
| Context editing (30-60% on Research/Code) | ~4% of total savings |
| Structured outputs (eliminates retries) | ~3% of total savings |

---

### 13.6 Self-Challenge Summary

| Idea | Challenge | Resolution |
|------|-----------|-----------|
| Opus 4.7 for Research | Too expensive? +67% cost vs Sonnet | MCP integration reduces input tokens by ~40% (JSON vs HTML), offsetting Opus markup. Net cost increase is ~$0.10/run, but research quality improvement compounds across all downstream agents. Fallback to Sonnet for budget-constrained runs. |
| MCP as universal layer | Adds complexity? Another abstraction? | MCP is an open standard (Linux Foundation), supported by all major LLM providers. MCPVault is 14 tools, MIT licensed, zero config. The abstraction SAVES complexity — agents don't need custom file I/O code. Token savings (40-60%) pay for the integration cost. |
| All vault I/O via MCP | What if MCPVault is down? | MCPVault runs locally via npx — no network dependency. Stdio transport. Fallback: direct Read/Write/Edit tools still available on every agent. MCP is primary, not exclusive. |
| Gemini for budget agents | Quality degradation? | Structured outputs constrain format; benchmarks show Flash-Lite handles classification well. Fallback to Claude for confidence < 0.7. |
| Effort `low` for Evaluator | Misses nuances? | Evaluator uses structured JSON output — format is guaranteed. Scoring rubric in skills instruction constrains judgment. Escalation path exists. |
| Opus for Code + Research | Two Opus agents — doubling cost? | Different justifications: Code needs code generation intelligence, Research needs synthesis intelligence. Together they handle the two highest-value tasks. MCP token savings offset Research Opus cost. Budget agents (Router, Evaluator, Summary, Review, Vault Org.) use Gemini Flash-Lite at $0.10-$0.25/MTok to compensate. |
| Compaction loses detail | Planner forgets subtask status? | Custom summary prompt preserves critical state. Combined with memory tool for external persistence. |
| Context editing + cache | Clearing breaks cache? | `clear_at_least` parameter ensures minimum tokens cleared to justify cache rebuild. Net positive for > 5000 tokens cleared. |
| Multi-provider complexity | Operational burden? | KISS framework abstracts model selection. Per-agent config, not per-call decisions. Gemini via OpenAI-compat endpoint. |
| Batch API latency | Blocks workflow? | Only for non-blocking post-processing (evaluation, summarization). Most complete within 1 hour. |
| Structured output overhead | Extra system prompt tokens? | ~200-500 tokens. Eliminates 1-3 retries × full prompt cost. Net savings for any agent called more than once. |
| 5 MCP servers for Research | Too many tool definitions? | Tool search can limit exposed tools per query. Most calls only use 2-3 servers per research task. Prompt caching covers tool definitions (cached after first call). |

---

### 13.7 Research Sources

1. Anthropic — [Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — 90% savings on cache reads, 5-min/1-hour TTL
2. Anthropic — [Effort Parameter](https://platform.claude.com/docs/en/build-with-claude/effort) — `low` to `max` token control per agent
3. Anthropic — [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) — Server-side context summarization
4. Anthropic — [Context Windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) — 1M context, context rot, awareness
5. Anthropic — [Batch Processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing) — 50% discount, stacks with caching
6. Anthropic Engineering Blog — [Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Sub-agent architectures, progressive disclosure
7. Anthropic — [Models Overview](https://platform.claude.com/docs/en/about-claude/models/overview) — Opus/Sonnet/Haiku pricing and capabilities
8. Anthropic — [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) — JSON schema, strict tool use
9. Anthropic — [Context Editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) — Tool result clearing, thinking clearing
10. Anthropic — [Manage Tool Context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context) — 4 composable approaches
11. Google — [Gemini Models](https://ai.google.dev/gemini-api/docs/models) — Gemini 3.5 Flash, 3.1 Pro, 2.5 Pro/Flash/Flash-Lite
12. Google — [Gemini Pricing](https://ai.google.dev/gemini-api/docs/pricing) — Per-model pricing, batch, flex, priority tiers


---

## 14. Synthesizer Web App Generation

The Synthesizer Agent (Agent 9) now has a **dual-output** responsibility: producing both the traditional summary Markdown file with `[[wikilinks]]` AND an interactive web application that visualizes all research findings.

### 14.1 Architecture — Two-Phase Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SYNTHESIZER AGENT — DUAL OUTPUT PIPELINE                                    │
│                                                                              │
│  Phase 1: Summary MD                   Phase 2: Web App                      │
│  ┌─────────────────────────┐           ┌─────────────────────────┐           │
│  │ 1. Read all subtask     │           │ 1. Check Node v22+      │           │
│  │    outputs from vault   │           │    ├─ YES → Quartz 5    │           │
│  │ 2. Resolve contradicts  │           │    └─ NO  → Single HTML │           │
│  │ 3. Write executive      │           │ 2. Copy vault notes to  │           │
│  │    summary              │           │    content directory     │           │
│  │ 4. Write thematic       │           │ 3. Generate config      │           │
│  │    sections             │           │ 4. Build static site    │           │
│  │ 5. Link all sources     │           │ 5. Verify output        │           │
│  └──────────┬──────────────┘           └──────────┬──────────────┘           │
│             │                                      │                          │
│             ▼                                      ▼                          │
│  reports/{run_id}_synthesis.md         output/{run_id}/webapp/                │
│  (vault note with [[wikilinks]])       (browseable static site)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 14.2 Primary Approach: Quartz 5

**Why Quartz 5?** Based on research across 10 tools (Quartz, Astro, MkDocs Material, Streamlit, Obsidian Publish, Obsidian Digital Garden, ObsidianHTML, D3.js, vis.js, vis-network):

| Tool | Graph View | Obsidian Native | CLI Build | License | Server Req | Verdict |
|------|-----------|-----------------|-----------|---------|------------|---------|
| **Quartz 5** | ✅ D3.js | ✅ Wikilinks, backlinks | ✅ `npx quartz build` | MIT | None (static) | **PRIMARY** |
| Obsidian Digital Garden | ✅ | ✅ | ❌ Needs app | MIT | Vercel | Good but not headless |
| ObsidianHTML | ✅ | ✅ | ✅ `obsidianhtml` | GPL-3.0 | None | GPL restrictive |
| Astro | ❌ (manual) | ❌ | ✅ | MIT | None | Over-engineered |
| MkDocs Material | ❌ | ❌ | ✅ `mkdocs build` | MIT | None | No graph, no wikilinks |
| Streamlit | ❌ (manual) | ❌ | ❌ | Apache-2.0 | Python server | Dynamic, not static |
| Obsidian Publish | ✅ | ✅ | ❌ Needs app | Proprietary | Cloud ($8/mo) | Not automatable |

**Quartz 5 wins because:**
1. **Natively reads Obsidian vault** — wikilinks, backlinks, YAML frontmatter all work out-of-box
2. **Built-in interactive graph view** — D3.js force-directed graph with local + global views
3. **CLI-based build** — `npx quartz build` (no GUI, fully automatable by agents)
4. **Static output** — no server needed, opens in any browser, deploy anywhere
5. **MIT License** — no restrictions
6. **Full-text search** — built-in client-side search across all notes

**Quartz Configuration (generated by Synthesizer):**
```yaml
# quartz.config.yaml — auto-generated by Synthesizer Agent
baseUrl: "./"
ignorePatterns: ["private", "templates", ".obsidian"]
theme:
  typography:
    header: "Inter"
    body: "Inter"
    code: "JetBrains Mono"
  colors:
    lightMode:
      light: "#faf8f8"
      dark: "#2b2b2b"
      primary: "#284b63"
      secondary: "#526980"
plugins:
  transformers:
    - name: "FrontMatter"
    - name: "CreatedModifiedDate"
      options:
        priority: ["frontmatter", "filesystem"]
    - name: "SyntaxHighlighting"
    - name: "ObsidianFlavoredMarkdown"
      options:
        enableInHtmlEmbed: true
        enableCheckbox: true
    - name: "TableOfContents"
    - name: "CrawlLinks"
  filters:
    - name: "RemoveDrafts"
  emitters:
    - name: "ContentPage"
    - name: "FolderPage"
    - name: "TagPage"
    - name: "ContentIndex"  # Required for graph view
    - name: "AliasRedirects"
    - name: "ComponentResources"
    - name: "NotFoundPage"
    - name: "Static"
    - name: "CNAME"
```

**Graph View Plugin Config:**
```yaml
# Added via: npm install github:quartz-community/graph --legacy-peer-deps
plugins:
  emitters:
    - name: "Graph"
      options:
        localGraph:
          drag: true
          zoom: true
          depth: 1
          scale: 1.1
          repelForce: 0.5
          centerForce: 0.3
          linkDistance: 30
          fontSize: 0.6
          opacityScale: 1
          showTags: true
          focusOnHover: true
        globalGraph:
          drag: true
          zoom: true
          depth: -1
          scale: 0.9
          repelForce: 0.5
          centerForce: 0.3
          linkDistance: 30
          fontSize: 0.6
          opacityScale: 1
          showTags: true
          focusOnHover: true
```

### 14.3 Fallback Approach: Single-File HTML Dashboard

When Node v22+ is unavailable, the Synthesizer generates a **self-contained HTML file** with embedded data and vis.js for graph visualization:

```
┌─────────────────────────────────────────────────────────────┐
│  SINGLE-FILE HTML DASHBOARD                                  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Header: Report Title + Date + Run ID               │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Executive Summary Panel (collapsible)              │    │
│  ├─────────────┬───────────────────────────────────────┤    │
│  │  Knowledge  │  Source Details Panel                  │    │
│  │  Graph      │  - Title, URL, date                   │    │
│  │  (vis.js)   │  - Key findings (bullet list)         │    │
│  │             │  - Tags, frontmatter metadata          │    │
│  │  Nodes:     │  - Full content (expandable)           │    │
│  │  🔵 research│                                        │    │
│  │  🟠 analysis│  [Click a node in the graph to view   │    │
│  │  🟣 report  │   its details here]                    │    │
│  │  ⚪ source  │                                        │    │
│  ├─────────────┴───────────────────────────────────────┤    │
│  │  Search Bar (client-side full-text filter)          │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Source Table (sortable by date, type, relevance)   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Tech: vis-network CDN + vanilla JS + CSS Grid              │
│  Size: ~50KB (data) + ~300KB (vis.js CDN cached)            │
│  Works: Any browser, no server, offline-capable              │
└─────────────────────────────────────────────────────────────┘
```

**Why vis.js over D3.js for the fallback?**
- vis-network is higher-level — requires ~50 lines of JS vs ~200+ for D3 force graphs
- Built-in physics simulation, clustering, zoom, drag, hover — no manual wiring
- Fewer tokens to generate (agent cost optimization)
- Still interactive and visually comparable to D3

### 14.4 Build Script (generated by Synthesizer Agent)

```bash
#!/bin/bash
# build_webapp.sh — generated by Synthesizer Agent
# Usage: ./build_webapp.sh <run_id> <vault_path>

RUN_ID="${1:?Usage: build_webapp.sh <run_id> <vault_path>}"
VAULT_PATH="${2:?Usage: build_webapp.sh <run_id> <vault_path>}"
OUTPUT_DIR="output/${RUN_ID}/webapp"

# Check for Node v22+
NODE_VERSION=$(node --version 2>/dev/null | sed 's/v//' | cut -d. -f1)
if [ -n "$NODE_VERSION" ] && [ "$NODE_VERSION" -ge 22 ]; then
    echo "✅ Node v${NODE_VERSION} found — using Quartz 5"
    
    # Initialize Quartz
    mkdir -p "${OUTPUT_DIR}"
    cd "${OUTPUT_DIR}"
    npx quartz create --directory content --strategy copy
    
    # Copy vault notes for this run
    cp -r "${VAULT_PATH}/research/" content/research/ 2>/dev/null
    cp -r "${VAULT_PATH}/analysis/" content/analysis/ 2>/dev/null
    cp -r "${VAULT_PATH}/reports/" content/reports/ 2>/dev/null
    cp -r "${VAULT_PATH}/sources/" content/sources/ 2>/dev/null
    
    # Install graph view plugin
    npm install github:quartz-community/graph --legacy-peer-deps
    
    # Apply generated config
    # (quartz.config.yaml already written by Synthesizer)
    
    # Build
    npx quartz build
    
    echo "✅ Web app built at: ${OUTPUT_DIR}/public/"
    echo "   Open: ${OUTPUT_DIR}/public/index.html"
else
    echo "⚠️  Node v22+ not found — falling back to single-file HTML"
    # Synthesizer generates dashboard.html directly via Write()
    echo "   Open: ${OUTPUT_DIR}/dashboard.html"
fi
```

---

## 15. Research Agent MCP Integration

The Research Agent (Agent 3) is enhanced with **MCP (Model Context Protocol) servers** that provide structured, deterministic access to academic databases, web search, and document analysis — replacing or supplementing raw web browsing.

### 15.1 Why MCP for Research?

| Aspect | Raw Web Browsing | MCP Research Tools |
|--------|------------------|-------------------|
| **Token cost** | High — full HTML pages loaded | Low — structured JSON responses |
| **Reliability** | Fragile — UI changes break scraping | Stable — API-backed, structured |
| **Speed** | Slow — page load + render + parse | Fast — direct API calls |
| **Coverage** | Any website | 240M+ scholarly works, 474M+ papers |
| **Citation quality** | Manual extraction | Built-in DOI, metadata, BibTeX |
| **Reproducibility** | Non-deterministic | Deterministic JSON (no LLM in server) |
| **Rate limits** | Varies by site | Generous (100K+ req/day on OpenAlex) |

### 15.2 Recommended MCP Stack

**Minimal setup (2 remote MCPs, zero local install):**

```json
{
  "mcpServers": {
    "rival-search": {
      "url": "https://RivalSearchMCP.fastmcp.app/mcp",
      "description": "Web + social + academic + news + datasets (10 tools)"
    },
    "papersflow": {
      "url": "https://doxa.papersflow.ai/mcp",
      "description": "474M+ papers, citation graphs, DOI lookup (8 tools)"
    }
  }
}
```

**Full research stack (add as needed):**

```json
{
  "mcpServers": {
    "rival-search": {
      "url": "https://RivalSearchMCP.fastmcp.app/mcp"
    },
    "papersflow": {
      "url": "https://doxa.papersflow.ai/mcp"
    },
    "openalex": {
      "command": "npx",
      "args": ["openalex-research-mcp"],
      "env": { "OPENALEX_EMAIL": "your@email.com" }
    },
    "paper-fetch": {
      "command": "npx",
      "args": ["paper-fetch"],
      "description": "DOI → PDF resolver with 7-source fallback"
    },
    "scholar-mcp": {
      "command": "npx",
      "args": ["scholar-mcp"],
      "description": "PDF ingestion + citation management"
    }
  }
}
```

### 15.3 MCP Tool Reference for Research Agent

| MCP Server | Tools | Data Sources | Best For |
|------------|-------|-------------|----------|
| **RivalSearchMCP** | 10 | 5 search engines, 9 social, 5 news, 5 academic DBs, 4 dataset hubs, GitHub, OCR | **General research** — broadest coverage |
| **PapersFlow** | 8 (free) + 2 (auth) | 474M+ papers | **Academic search** — citation graphs, DOI lookup |
| **openalex-research-mcp** | 31 | OpenAlex (240M+ works) | **Curated academic** — journal/institution presets |
| **paper-fetch** | 1 | Unpaywall, S2, arXiv, PMC, bioRxiv, publishers, Sci-Hub | **PDF retrieval** — DOI to full-text |
| **ScholarMCP** | 10 | OpenAlex, Crossref, S2, Google Scholar | **Writing workflows** — citation management |
| **asta-skill** | 5 | Semantic Scholar (AI2 Asta) | **AI/ML papers** — body text search |
| **scholar-deep-research** | 1 (8-phase) | Federated | **Automated lit reviews** — full pipeline |
| **mcp-for-research** | 5 | PubMed, Scholar, ArXiv | **Biomedical** — full-text from PMC |

### 15.4 Integration with KISS Framework

```python
# Research Agent with MCP tools — KISS integration (Opus 4.7 + full MCP stack)
research_agent = AgentDefinition(
    name="research",
    role="MCP-powered deep research, multi-source synthesis, citation",
    model_name="claude-opus-4-7",         # Frontier model for highest-value task
    fallback_model="claude-sonnet-4-6",   # Budget fallback
    provider="anthropic",
    effort="high",
    is_agentic=True,
    max_steps=100,
    max_budget=12.0,
    tools=[
        WebUseTool,       # Fallback web browsing (when MCP doesn't cover a source)
        Read, Write, Bash,
        # MCP tools auto-discovered from mcpServers config:
        # MCPVault:
        #   mcpvault_read_note(), mcpvault_write_note(), mcpvault_search_notes()
        #   mcpvault_get_frontmatter(), mcpvault_update_frontmatter()
        #   mcpvault_get_vault_stats(), mcpvault_manage_tags()
        # RivalSearchMCP:
        #   rival_search_web(), rival_search_academic()
        #   rival_search_social(), rival_search_news()
        #   rival_search_datasets(), rival_search_document_analysis()
        # PapersFlow:
        #   papersflow_search(), papersflow_citation_graph()
        #   papersflow_doi_lookup(), papersflow_related_papers()
        # OpenAlex:
        #   openalex_search_works(), openalex_journal_preset()
        #   openalex_citation_network(), openalex_trend_analysis()
        # paper-fetch:
        #   paper_fetch_resolve()
    ],
    mcp_servers={
        "mcpvault": {"command": "npx", "args": ["@bitbonsai/mcpvault", "--vault", "{vault_path}"]},
        "rival-search": {"url": "https://RivalSearchMCP.fastmcp.app/mcp"},
        "papersflow": {"url": "https://doxa.papersflow.ai/mcp"},
        "openalex": {"command": "npx", "args": ["openalex-research-mcp"]},
        "paper-fetch": {"command": "npx", "args": ["paper-fetch"]},
    },
    enable_prompt_caching=True,
    enable_context_editing=True,
    context_edit_trigger_tokens=50000,
    context_edit_keep_tool_uses=5,
    enable_thinking_clearing=True,
    thinking_keep_turns=2,
)
```

### 15.5 Research Workflow with MCP

```
Research Task → Research Agent
    │
    ├── [1] Query RivalSearchMCP for broad web + academic results
    │       → Structured JSON: titles, URLs, snippets, quality scores
    │
    ├── [2] Query PapersFlow for deep paper search
    │       → DOIs, abstracts, citation counts, related papers
    │
    ├── [3] Query OpenAlex for curated journal-filtered results
    │       → Works from UTD24/FT50/AJG journals only
    │
    ├── [4] For each promising paper: paper-fetch to get PDF
    │       → Full-text for detailed analysis
    │
    ├── [5] WebUseTool for non-academic sources (blogs, forums, docs)
    │       → Traditional web scraping (when MCP doesn't cover it)
    │
    └── [6] Synthesize all results into vault note with [[wikilinks]]
            → Markdown with YAML frontmatter, citations, source links
```

### 15.6 All Underlying APIs — License & Cost

| API | License | Cost | Records |
|-----|---------|------|---------|
| OpenAlex | CC0 (public domain) | Free | 240M+ works |
| CrossRef | CC0 metadata | Free | 150M+ records |
| PubMed/NCBI | Public domain (US gov) | Free | 36M+ citations |
| arXiv | CC0 metadata | Free | 2.4M+ preprints |
| Europe PMC | Open | Free | 40M+ articles |
| Semantic Scholar | Free API | Free (key for higher limits) | 200M+ papers |
| Unpaywall | Free non-commercial | Free | 40M+ OA articles |

---

## 16. Agent-Specific Add-Ons Registry

Every agent definition includes a **Specific Add-Ons** category — tools, MCP servers, extensions, and skill packs tailored to that agent's role.

### 16.1 Add-Ons Master Table

| Agent | MCP Servers | CLI Tools | VS Code Extensions | Skill Packs | APIs |
|-------|-------------|-----------|-------------------|-------------|------|
| **1. Planner** | — | — | — | `orchestration-patterns` | — |
| **2. Router** | — | — | — | `task-classifier` | — |
| **3. Research** | RivalSearchMCP, PapersFlow, OpenAlex MCP, paper-fetch, ScholarMCP, asta-skill, scholar-deep-research, mcp-for-research | — | PapersFlow.papersflow | `web-research-10-site`, `academic-search` | OpenAlex, CrossRef, PubMed, arXiv, Semantic Scholar |
| **4. Writing** | — | `vale` (prose linter) | `streetsidesoftware.code-spell-checker` | `technical-writing`, `markdown-style` | — |
| **5. Code** | `@anthropic/code-mcp` | `uv`, `npm`, `cargo` | — | `code-review`, `test-generation` | GitHub API |
| **6. Analysis** | — | `jq`, `csvkit` | — | `data-analysis`, `statistical-reasoning` | — |
| **7. Evaluator** | — | — | — | `quality-gate-rubric` | — |
| **8. Review** | — | `markdownlint` | — | `review-checklist` | — |
| **9. Synthesizer** | MCPVault (`@bitbonsai/mcpvault`) | `npx quartz`, `node` | — | `report-assembly`, `webapp-generation` | — |
| **10. Summary** | — | — | — | `extractive-summary` | — |
| **11. Vault Organizer** | MCPVault (`@bitbonsai/mcpvault`) | `rg` (ripgrep) | — | `vault-maintenance`, `deduplication` | — |

### 16.2 Detailed Add-Ons Per Agent

#### Agent 1: Planner — Add-Ons
```
Skill Packs:
  - orchestration-patterns: DAG decomposition templates, parallel vs sequential
    decision rules, budget estimation heuristics

Notes: Planner is a pure orchestrator — minimal external tools needed.
       Relies on vault reads (via MCPVault or Read tool) for context.
```

#### Agent 2: Router — Add-Ons
```
Skill Packs:
  - task-classifier: Classification taxonomy (research/writing/code/analysis/
    review/mixed), routing rules, confidence thresholds

Notes: Router is non-agentic (single-shot classification).
       No external tools — classification is done entirely in-context.
```

#### Agent 3: Research — Add-Ons
```
MCP Servers (Tier 1 — always active):
  - RivalSearchMCP:     https://RivalSearchMCP.fastmcp.app/mcp
  - PapersFlow MCP:     https://doxa.papersflow.ai/mcp
  - openalex-research:  npx openalex-research-mcp

MCP Servers (Tier 2 — activated per task):
  - paper-fetch:           npx paper-fetch (DOI→PDF)
  - ScholarMCP:            npx scholar-mcp (PDF ingestion)
  - asta-skill:            Semantic Scholar via AI2 Asta
  - scholar-deep-research: 8-phase automated literature review
  - mcp-for-research:      npx scholarly-research-mcp

MCP Servers (Tier 3 — optional):
  - mendeley-mcp:                Reference manager
  - unpaywall-mcp:               Open access links
  - research-workflow-assistant:  VS Code + GitHub Copilot connector
  - openalex-mcp:                Simpler OpenAlex interface

VS Code Extensions:
  - PapersFlow.papersflow: Paper search + citation graphs in editor

Skill Packs:
  - web-research-10-site: Enforces 10-site minimum, counter tracking,
    information file template, synthesis-after-10 rule
  - academic-search: Journal quality filters, citation chasing,
    systematic review methodology

APIs (via MCP, no direct access needed):
  - OpenAlex (240M+ works, CC0)
  - CrossRef (150M+ records, CC0)
  - PubMed (36M+ citations, public domain)
  - arXiv (2.4M+ preprints, CC0)
  - Semantic Scholar (200M+ papers, free API)
  - Europe PMC (40M+ articles, open)
```

#### Agent 4: Writing — Add-Ons
```
CLI Tools:
  - vale: Prose linter with configurable style rules
    (Microsoft, Google, Joblint, write-good)
    Install: brew install vale
    Usage: vale --config=.vale.ini <file.md>

VS Code Extensions:
  - streetsidesoftware.code-spell-checker: Real-time spell checking

Skill Packs:
  - technical-writing: Tone guidelines, active voice enforcement,
    jargon avoidance, heading hierarchy rules
  - markdown-style: Obsidian-specific formatting (callouts, wikilinks,
    YAML frontmatter templates)
```

#### Agent 5: Code — Add-Ons
```
MCP Servers:
  - @anthropic/code-mcp: Code search, file operations, repo analysis
    (if available in KISS runtime)

CLI Tools:
  - uv: Python package manager and runner
  - npm: Node.js package manager (for Quartz, MCP servers)
  - cargo: Rust package manager (if Rust tasks arise)

Skill Packs:
  - code-review: Security checklist, performance patterns,
    error handling best practices
  - test-generation: Integration test patterns, E2E test templates,
    coverage-driven test planning

APIs:
  - GitHub API: Repository operations, PR creation, issue tracking
```

#### Agent 6: Analysis — Add-Ons
```
CLI Tools:
  - jq: JSON processor for structured data analysis
  - csvkit: CSV processing suite (csvstat, csvsql, csvjoin)

Skill Packs:
  - data-analysis: Statistical methods selection, visualization
    recommendations, data quality assessment
  - statistical-reasoning: Hypothesis testing frameworks,
    confidence intervals, correlation vs causation guidelines
```

#### Agent 7: Evaluator — Add-Ons
```
Skill Packs:
  - quality-gate-rubric: Scoring criteria (0-100) with dimensions:
    completeness, accuracy, coherence, source quality, formatting.
    Pass threshold: 70. Auto-retry trigger: < 50.

Notes: Evaluator is non-agentic. No external tools needed.
       Uses structured output (JSON rubric scores).
```

#### Agent 8: Review — Add-Ons
```
CLI Tools:
  - markdownlint: Markdown linting for formatting consistency
    Install: npm install -g markdownlint-cli
    Usage: markdownlint <file.md>

Skill Packs:
  - review-checklist: Human-readable checklist covering:
    frontmatter completeness, [[wikilink]] validity,
    source citation format, word count compliance, tag coverage
```

#### Agent 9: Synthesizer — Add-Ons
```
MCP Servers:
  - MCPVault (@bitbonsai/mcpvault): Read vault notes via MCP
    (14 tools: read, search, frontmatter, tags, batch ops)
    Enables structured vault access without raw file I/O

CLI Tools:
  - npx quartz: Quartz 5 static site generator (primary web app)
  - node: Node.js runtime (required for Quartz, v22+)

Skill Packs:
  - report-assembly: Multi-source synthesis patterns, contradiction
    resolution, executive summary templates
  - webapp-generation: Quartz 5 configuration, single-file HTML
    dashboard templates, vis.js graph setup, CSS Grid layouts

Notes: Synthesizer upgraded to RelentlessAgent with high effort
       to handle the dual-output (MD + web app) pipeline.
```

#### Agent 10: Summary — Add-Ons
```
Skill Packs:
  - extractive-summary: Extractive summarization rules, key-sentence
    selection, frontmatter _summary field generation, token budget
    targeting (200 words / ~300 tokens per summary)

Notes: Summary is non-agentic. No external tools needed.
       Single-shot summarization with structured output.
```

#### Agent 11: Vault Organizer — Add-Ons
```
MCP Servers:
  - MCPVault (@bitbonsai/mcpvault): Primary vault access tool
    (read, write, search BM25, frontmatter ops, tag management,
    batch operations, vault statistics — 14 tools total)

CLI Tools:
  - rg (ripgrep): Fast full-text search for broken link detection,
    orphan note discovery, pattern matching across vault
    Install: brew install ripgrep

Skill Packs:
  - vault-maintenance: MOC generation templates, archive policies,
    link audit rules, stale content detection thresholds
  - deduplication: Near-duplicate detection via BM25 scoring,
    merge strategies, redirect note templates

Notes: Vault Organizer runs on schedule (post-workflow, daily, weekly).
       MCPVault is the primary interface — avoids raw file I/O
       for better token efficiency (40-60% savings via compact JSON).
```

### 16.3 Add-On Configuration in AgentDefinition

```python
@dataclass
class AgentAddOns:
    """Specific add-ons for an agent — tools, MCP servers, extensions, skills."""
    
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    """MCP server configurations: {"name": {"url": "..."} or {"command": "...", "args": [...]}}"""
    
    cli_tools: list[str] = field(default_factory=list)
    """CLI tools the agent may invoke via Bash: ["vale", "jq", "npx quartz"]"""
    
    vscode_extensions: list[str] = field(default_factory=list)
    """VS Code extensions relevant to this agent's domain."""
    
    skill_packs: list[str] = field(default_factory=list)
    """Named skill instruction packs injected into the system prompt."""
    
    apis: list[str] = field(default_factory=list)
    """External APIs this agent accesses (directly or via MCP)."""


# Example: Research Agent with full add-ons
research_addons = AgentAddOns(
    mcp_servers={
        "rival-search": {"url": "https://RivalSearchMCP.fastmcp.app/mcp"},
        "papersflow": {"url": "https://doxa.papersflow.ai/mcp"},
        "openalex": {"command": "npx", "args": ["openalex-research-mcp"]},
        "paper-fetch": {"command": "npx", "args": ["paper-fetch"]},
        "scholar-mcp": {"command": "npx", "args": ["scholar-mcp"]},
    },
    cli_tools=[],
    vscode_extensions=["PapersFlow.papersflow"],
    skill_packs=["web-research-10-site", "academic-search"],
    apis=["OpenAlex", "CrossRef", "PubMed", "arXiv", "Semantic Scholar", "Europe PMC"],
)

research_agent = AgentDefinition(
    name="research",
    # ... existing config ...
    add_ons=research_addons,
)

# Example: Synthesizer Agent with web app add-ons
synthesizer_addons = AgentAddOns(
    mcp_servers={
        "mcpvault": {"command": "npx", "args": ["@bitbonsai/mcpvault"]},
    },
    cli_tools=["npx quartz", "node"],
    vscode_extensions=[],
    skill_packs=["report-assembly", "webapp-generation"],
    apis=[],
)

synthesizer_agent = AgentDefinition(
    name="synthesizer",
    # ... existing config ...
    add_ons=synthesizer_addons,
)
```
