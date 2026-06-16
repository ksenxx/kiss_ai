# PROGRESS — Task 4 (still-broken close+reopen running tab)

## Task

Bug NOT fixed after 3 prior fixes (af798ff2, c34c3727, 89617d34). User reports:
when a task runs in a tab and the tab is closed while running, then reopened while running:

- user input via textbox is ignored DURING the task
- and AFTER the task ends, NO new task can be started in the tab
- tab behaves DIFFERENTLY from a tab that loads the task

Must reproduce with an integration test, then fix.

## Plan

1. Read SorcarSidebarView.ts + main.js to understand reopen/restore flow.
1. Read server.py \_replay_session, \_reattach_running_chat, \_cmd_resume_session.
1. Write an end-to-end integration test against REAL daemon + REAL extension + jsdom-real-main.js
   that reproduces: start task, close tab DURING run, reopen tab DURING run, type input -> assert
   it reaches the agent, wait for task end, type new input -> assert it triggers a new task.
1. Identify the layer that still drops; fix root cause.
