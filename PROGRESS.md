# Task: Make API keys in settings panel secret by default with eye toggle

## Steps done

1. Located the settings panel markup in `src/kiss/agents/vscode/media/chat.html`
   (API key inputs `cfg-key-*` + `cfg-custom-api-key` were `type="text"`) and the
   existing remote-password eye-toggle pattern (`setupPasswordToggle` in
   `media/main.js`, `.config-password-wrap`/`.config-password-toggle` CSS in
   `media/main.css` — CSS reused as-is, no CSS changes needed).

1. `media/chat.html`: changed all 7 API-key inputs (GEMINI, OPENAI, ANTHROPIC,
   TOGETHER, OPENROUTER, MINIMAX, custom) to
   `type="password" autocomplete="off"` so keys are masked even before JS runs.

1. `media/main.js`:

   - `setupPasswordToggle(toggleId, inputId, secretName)` gained an optional
     `secretName` noun so aria-label/title read "Show/Hide API key" for keys.
   - Added `setupSecretInput(inputId)`: forces `type="password"`, wraps the
     input in `.config-password-wrap`, clones the eye/eye-off button from
     `#cfg-remote-password-toggle` (SVG markup stays in one place), appends it,
     and wires it via `setupPasswordToggle(btn.id, inputId, 'API key')`.
   - In `setupEventListeners()` (right after the existing remote-password
     toggle setup): `[...7 input ids...].forEach(setupSecretInput);`

1. Wrote `src/kiss/tests/agents/vscode/test_api_key_secret_inputs.py` pinning
   served HTML (`_build_html()`) + main.js wiring (repo pattern, same style as
   test_password_persistence.py).

1. Verification (all green):

   - `uv run pytest test_api_key_secret_inputs.py test_password_persistence.py test_web_extension_parity.py` → 21 passed.
   - Compiled the extension (`npx tsc -p .`) to fix a pre-existing
     missing-`out/` failure in the extension check, then
     `uv run check --full` → all checks passed (mdformat fix applied to
     PROGRESS.md).
   - Staged the new test file in git.

## Status: COMPLETE
