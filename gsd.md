# Plan: Incorporating GSD Ideas into KISS Sorcar

## Executive Summary

After deep analysis of [GSD (Get Shit Done)](https://github.com/gsd-build/get-shit-done) — a 48k-star meta-prompting and context engineering system — this document extracts its most powerful ideas and proposes concrete ways to incorporate them into KISS Sorcar's agent system. The focus is on ideas that will make the Sorcar agent **get tasks done perfectly**, not on adding enterprise ceremony.

GSD's core insight: **the agent's quality is determined by the context it receives, not by the model's raw ability.** Everything GSD does — structured planning docs, fresh-context execution, quality gates, revision loops, deep work rules — serves one goal: giving the AI exactly the right context at the right time.

---

## Part 1: GSD's Best Ideas (Ranked by Impact for Sorcar)

### 1. Quality Gates (Pre-flight, Revision, Escalation, Abort)

**What GSD does:** Every validation checkpoint in every workflow is classified into one of 4 gate types:
- **Pre-flight gate** — Check preconditions before starting (does the file exist? is the config valid?). Cheap, deterministic, prevents wasted work.
- **Revision gate** — After producing output, check quality and loop back with specific feedback if insufficient. Bounded by iteration cap (max 3). Includes stall detection (if issue count doesn't decrease, escalate).
- **Escalation gate** — When automated resolution fails, surface the issue to the human with clear options.
- **Abort gate** — Stop immediately when continuing would cause damage (context window critically low, error state).

**What Sorcar does today:** The `RelentlessAgent` has continuation logic (re-spawn when steps/context exhausted) and a `finish()` tool, but no structured quality gates. The agent can silently produce broken work without self-checking.

**Proposed change:** Add a lightweight gate system to the agent loop.

### 2. Fresh Context Per Sub-Task (Context Rot Prevention)

**What GSD does:** When executing a multi-plan phase, each plan runs in a **fresh 200k-token context window**. The orchestrator spawns a new subagent for each plan, giving it only the files it needs (`<read_first>` list). This prevents "context rot" — the quality degradation as accumulated conversation fills the window.

**What Sorcar does today:** `RelentlessAgent` has continuation sessions (new sub-session when steps exhaust), which partially addresses this. But within a single session, context accumulates freely. For complex tasks, the agent often degrades in quality as it fills its window.

**Proposed change:** For complex tasks, decompose into sub-tasks and execute each in a fresh agent session.

### 3. Deep Work Rules (Concrete Plans, Not Vague Instructions)

**What GSD does:** Every task must include:
- `<read_first>` — Files the executor MUST read before modifying anything (always includes the file being modified + source of truth files)
- `<acceptance_criteria>` — Verifiable conditions provable with grep/test/CLI output. NEVER subjective ("looks correct"). ALWAYS exact strings, patterns, values.
- `<action>` — Concrete values, not references. NEVER "align X with Y" without specifying the exact target state.

**What Sorcar does today:** The system prompt (SYSTEM.md) has good rules about code style and testing, but the agent often makes vague plans internally and then executes them shallowly.

**Proposed change:** Inject "deep work" discipline into the system prompt for complex tasks.

### 4. Discuss → Research → Plan → Execute → Verify Pipeline

**What GSD does:** A strict 5-phase pipeline for every significant piece of work:
1. **Discuss** — Identify gray areas, capture user preferences (CONTEXT.md)
2. **Research** — Investigate domain, patterns, dependencies (RESEARCH.md)
3. **Plan** — Create atomic task plans with frontmatter, acceptance criteria (PLAN.md)
4. **Execute** — Run plans in dependency-ordered waves, fresh context each
5. **Verify** — Check deliverables against criteria, run tests, human UAT

**What Sorcar does today:** The agent receives a task and immediately starts executing. There's no structured planning or verification phase.

**Proposed change:** For complex tasks (multi-file changes, architectural work), inject a structured planning phase into the agent's workflow.

### 5. Revision Loops with Stall Detection

**What GSD does:** After the planner produces plans, a plan-checker reviews them. If issues found, loop back to planner with specific feedback (max 3 iterations). **Stall detection:** if issue count doesn't decrease between iterations, escalate to human instead of looping forever.

**What Sorcar does today:** The agent runs tests and may retry on failure, but there's no structured self-review or bounded revision.

**Proposed change:** After the agent makes changes, add a self-review step with bounded retries.

### 6. Requirements Traceability

**What GSD does:** Every requirement gets an ID. Plans must reference which requirements they address. A coverage gate after planning verifies that every phase requirement appears in at least one plan.

**What Sorcar does today:** The task description is the only "requirement" — there's no traceability to check if all parts were addressed.

**Proposed change:** For complex tasks, decompose the task into checkable sub-requirements and verify coverage.

### 7. Wave-Based Parallel Execution

**What GSD does:** Plans are grouped into "waves" based on dependencies. Independent plans run in parallel within a wave. Waves execute sequentially.

**What Sorcar does today:** Everything is sequential within a single agent.

**Proposed change:** This is less applicable to Sorcar's single-agent model, but the dependency-ordering concept is valuable for task decomposition.

### 8. Self-Improvement Loop (LESSONS.md)

**What GSD does:** The system maintains state across sessions (STATE.md, PROJECT.md). GSD's codebase-mapping spawns parallel agents to analyze the codebase's stack, architecture, conventions.

**What Sorcar does today:** Already has LESSONS.md (read at start, updated at end). This is aligned with GSD's philosophy.

**No change needed** — Sorcar already does this well. Could be enhanced with richer project state tracking.

---

## Part 2: Concrete Implementation Plan

### Phase 1: Pre-flight Gates in System Prompt (Low effort, High impact)

Add pre-flight validation rules to SYSTEM.md that the agent must follow before executing:

```markdown
## Pre-flight Checks (MANDATORY before modifying code)
- Read every file you're about to modify FIRST — understand current state before changing
- Run existing tests FIRST — know the baseline before you change anything
- If the task mentions files that don't exist, STOP and report — don't guess
- If the task requires understanding architecture, read relevant source files FIRST
```

**Where:** Add to `SYSTEM.md` under a new `## Quality Gates` section.

**Why:** The #1 reason agents produce broken code is modifying files without reading them first. This single change would prevent the largest class of errors.

### Phase 2: Self-Verification After Execution (Medium effort, High impact)

Add a verification step to the agent's workflow. After the agent believes it's done, before calling `finish(success=True)`, it must:

1. Re-read every file it modified
2. Run tests (`uv run pytest -v`)
3. Run linters (`uv run check --full`)
4. Verify acceptance criteria are met (grep for expected patterns)

**Where:** Add to `SYSTEM.md`:

```markdown
## Post-Execution Verification (MANDATORY before finish)
Before calling finish(success=True), you MUST:
1. Run tests and fix any failures
2. Run linters/typecheckers and fix any errors
3. Re-read every file you modified — verify the changes are correct and complete
4. Verify the task requirements are fully satisfied — check each sub-requirement
5. If ANY verification fails, fix it before finishing
```

### Phase 3: Deep Work Rules in System Prompt (Low effort, High impact)

Add GSD's "anti-shallow execution" rules:

```markdown
## Deep Work Rules
- Before modifying a file, ALWAYS read it first — never assume its current state
- When the task says "align X with Y", determine the EXACT target values first
- Every code change must have a verifiable check: a test that passes, a grep that matches,
  a command that succeeds
- Never make a change based on assumption — always verify the current state first
- When making multiple related changes, list them all upfront, then execute systematically
```

**Where:** Add to `SYSTEM.md`.

### Phase 4: Structured Planning for Complex Tasks (Medium effort, High impact)

For tasks that involve multi-file changes, add a planning step. Modify the system prompt:

```markdown
## Task Planning (for tasks involving 3+ files or architectural changes)
Before writing any code, create a brief plan:
1. List all files that need to change and WHY
2. For each file, describe the EXACT change (not "update to match" — the actual values)
3. Identify dependencies between changes (what must happen first)
4. Define how you'll verify each change worked
Then execute the plan in dependency order.
```

**Where:** Add to `SYSTEM.md`.

### Phase 5: Bounded Revision Loop (Medium effort, Medium impact)

Enhance `RelentlessAgent.perform_task()` to support bounded self-revision:

**Concept:** After the agent finishes a task, if tests fail, re-run with specific error context (up to 3 retries). Track whether the error count is decreasing — if stalled after 2 retries, stop and report rather than spinning.

The `RelentlessAgent` already has continuation logic. Enhance it:
- Track test failure count across continuations
- If failure count doesn't decrease after a continuation, add an explicit prompt: "The same errors persist. Step back and rethink the approach."
- After 3 failed retries of the same error, report the issue rather than retrying

**Where:** `src/kiss/core/relentless_agent.py`, modify continuation prompt logic.

### Phase 6: Sub-Task Decomposition with Fresh Context (High effort, High impact)

For very complex tasks (e.g., "refactor the entire test suite"), add the ability to decompose into sub-tasks and execute each in a fresh agent session:

**Concept:**
1. The orchestrator agent reads the task and produces a list of sub-tasks
2. Each sub-task runs in a fresh `SorcarAgent` session with minimal context (only the sub-task description + relevant files)
3. Results from each sub-task are collected and verified
4. A final verification pass checks the overall result

**Where:** Create a new `OrchestratorAgent` class that spawns sub-`SorcarAgent` instances. This mirrors GSD's phase-executor spawning plan-executors.

**Implementation sketch:**
```python
class OrchestratorAgent(SorcarAgent):
    """Decomposes complex tasks into sub-tasks with fresh context each."""

    def perform_task(self, tools, attachments=None):
        # 1. Plan: decompose task into sub-tasks
        # 2. For each sub-task, spawn a fresh SorcarAgent
        # 3. Collect results
        # 4. Verify overall result
```

This is the most impactful but also most complex change. It directly addresses context rot.

### Phase 7: Project State Tracking (Low effort, Medium impact)

Add lightweight project state tracking inspired by GSD's `STATE.md`:

**Concept:** Maintain a `.kiss/project_state.md` file in the work directory that tracks:
- What the agent has done (completed tasks with summaries)
- Known issues/blockers
- Architecture decisions made
- Files recently modified

The agent reads this at the start of each task for context, and updates it at the end.

**Where:** `src/kiss/agents/sorcar/persistence.py` — extend with project state read/write.

---

## Part 3: Priority Order

| Priority | Phase | Effort | Impact | Description |
|----------|-------|--------|--------|-------------|
| **P0** | 1 | Low | High | Pre-flight gates in system prompt |
| **P0** | 3 | Low | High | Deep work rules in system prompt |
| **P0** | 2 | Medium | High | Self-verification before finish |
| **P1** | 4 | Medium | High | Structured planning for complex tasks |
| **P1** | 5 | Medium | Medium | Bounded revision loop with stall detection |
| **P2** | 6 | High | High | Sub-task decomposition with fresh context |
| **P2** | 7 | Low | Medium | Project state tracking |

**P0 items (do first):** Pure prompt engineering. No code changes required. Immediately improves agent quality for every task.

**P1 items (do second):** Minor prompt + small code changes. Improve agent quality for complex tasks.

**P2 items (do later):** Significant architecture changes. Transform how the agent handles large tasks.

---

## Part 4: What NOT to Adopt from GSD

GSD has many features that are **not appropriate** for KISS Sorcar:

1. **Enterprise workflow ceremony** — GSD has milestones, roadmaps, sprint-like phases. Sorcar tasks are typically single-task-in, result-out. Don't add project management overhead.

2. **60+ slash commands** — GSD has dozens of commands (discuss-phase, plan-phase, execute-phase, verify-work, ship, etc.). Sorcar's simplicity (one task → one result) is a strength. Don't fragment the workflow into many commands.

3. **Multiple specialized agent types** — GSD has researcher, planner, plan-checker, executor, verifier as separate agents. For Sorcar, the single-agent-with-tools model is simpler and more reliable. The *concepts* (research, plan, execute, verify) should be phases within the same agent, not separate agents.

4. **Heavy state management** — GSD maintains PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md, CONTEXT.md, RESEARCH.md, PLAN.md, SUMMARY.md, VERIFICATION.md, UAT.md per phase. This is overkill for Sorcar's task model.

5. **UI/security/schema-specific gates** — GSD has UI-SPEC gates, security threat model gates, schema push detection. These are domain-specific and don't apply to Sorcar's general-purpose agent.

6. **Wave-based parallel execution** — Sorcar is a single-agent system. Parallelism adds complexity without clear benefit for most tasks.

---

## Part 5: GSD Ideas Already Present in KISS

These GSD ideas are already implemented in KISS, confirming good alignment:

| GSD Concept | KISS Equivalent |
|-------------|----------------|
| Self-improvement (STATE.md) | LESSONS.md read/write loop |
| Continuation on context exhaustion | RelentlessAgent continuation sessions |
| Fresh context on continuation | New sub-session with progress summary |
| Atomic commits | Agent can commit per logical change |
| Project-specific rules | SORCAR.md loaded into system prompt |
| Git worktree isolation | WorktreeSorcarAgent |
| Chat session history | StatefulSorcarAgent with chat_id |

---

## Summary

The highest-impact changes from GSD are **all prompt engineering** — no code changes required:

1. **Pre-flight gates**: Read before write, test before change
2. **Deep work rules**: Concrete values, not vague references
3. **Self-verification**: Check your work before declaring success

These three prompt additions, inspired by GSD's quality gate taxonomy and deep work rules, would immediately and significantly improve Sorcar's task completion quality. The more complex changes (sub-task decomposition, revision loops) can follow later as the agent matures.
