- Use ./src/kiss/agents/third_party_agents/govee.py to take action on home lights.

## Memory / Pending user requests

- (Noted on 2026-07) Speaker #1 plans to ask in a couple of weeks to implement the "Vision & Physical-World Control" feature set discussed earlier: (1) camera capture tool (OpenCV/ffmpeg, webcam/RTSP), (2) Home Assistant integration tool (ha_get_states/ha_call_service via REST + long-lived token), (3) scheduled camera monitors via cron tasks (see tasks.json) with messaging alerts, (4) closed-loop see→act→verify skills saved to a skill library, (5) ROS 2 bridge for robotics, plus safety guardrails (confirmation tiers, allow-lists, audit log with justifying camera frame, auto-revert dead-man switches). Note: govee.py already provides light control — reuse it for the actuation layer.

