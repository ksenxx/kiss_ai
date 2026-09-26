---
title: Grok Bot "Teach a task" and how to replicate it
uuid: 02d28356-c10c-4844-b919-d857e8b045ca
summary: How Grok Bot's learn-by-demonstration works (records up to 10 min, produces
  a draft skill, then review, test, routine) and a verified Playwright recorder approach
  for Sorcar.
created: '2026-09-26T19:52:53Z'
updated: '2026-09-26T19:52:53Z'
---
Grok Bot (SpaceXAI with Cursor, 2026). Sources: https://x.ai/bot, https://docs.x.ai/grok-bot/skills-routines-and-automations
- Teach a task: records visible computer interaction for up to 10 min (no mic audio) and produces a DRAFT skill (when to use, inputs, steps, validation, output, approvals). The human reviews it and adds decision rules, failure handling and approval boundaries, then runs a test on a safe example. A routine (schedule or trigger) can be attached afterwards.
- The output is instructions that an agent follows with its browser tools, not a macro replay.

Replication (verified 2026-09-26, playwright python in the repo venv):
- context.expose_binding("__teachEvent", cb) + context.add_init_script(recorder.js) with capture-phase click/change/keydown listeners, recording role + accessible name, test id, and nearby text. Use e.composedPath()[0] for shadow DOM. Redact password/cc/otp values.
- In the sync API, callbacks fire only while Playwright calls run, so the wait loop must use page.wait_for_timeout rather than a blocking input().
- A Tab keydown event arrives before the change event; merge them during normalization.
- Full write-up: reports/teach-an-agent-browser-workflows.html
- Sorcar hooks: WebUseTool._context (web_use_tool.py), persistent profile ~/.kiss/browser_profile; skills go in ~/.kiss/skills/<name>/SKILL.md (skills.py discover_skills).
