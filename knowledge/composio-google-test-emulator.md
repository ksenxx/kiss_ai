---
title: Testing KISS Google agents via a local Composio emulator
uuid: c38e8439-7308-4bd1-a154-cab83c16b45b
summary: 'How KISS Google Workspace agents (Composio-brokered) are tested end-to-end:
  composio_test_utils.py fake Composio v3.1 server with forwarding proxy.'
created: '2026-09-26T20:44:20Z'
updated: '2026-09-26T20:44:20Z'
---
# Testing Composio-brokered Google agents (KISS)

- Google agents (gmail, gcal, gdocs, gdrive, gsheets, googlechat) auth via `_composio_google.py`; tools from `_google_workspace_utils.make_google_auth_tools(agent, service, label, on_connected)`.
- Tests: `src/kiss/tests/agents/third_party_agents/composio_test_utils.py`
  - `start_fake_composio(monkeypatch)` sets `COMPOSIO_BASE_URL` + `COMPOSIO_API_KEY` (the Composio SDK honors the base URL env var).
  - Emulated endpoints: GET/POST `/api/v3.1/auth_configs`, GET `/api/v3.1/connected_accounts` (list, used by `link`), POST `.../connected_accounts/link`, GET/DELETE `.../connected_accounts/{id}`, POST `/api/v3.1/tools/execute/proxy`.
  - The proxy really forwards to `endpoint`, injecting `Authorization: Bearer <server.token>`; `server.upstream_overrides` reroutes hard-wired Google origins (e.g. `https://gmail.googleapis.com`) for googleapiclient services; non-JSON/non-text answers come back as `binary_data` URLs.
  - `connect(server, service)` runs real start/finish; `reset_state(service)` forgets the connection and saved key.
- The SDK models use `model_construct` (no strict validation), so minimal JSON responses work.
- `finish_connect` on a pending account waits ~5 s (SDK wait timeout), so keep pending tests few.
- Composio proxy supports only JSON bodies in KISS's wrapper, so Drive uploads return an error (the raw API has `binary_body`, unused).
