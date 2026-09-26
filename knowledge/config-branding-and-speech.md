---
title: Product branding (brand.json) and talk speech synthesis (gpt-audio-1.5)
uuid: 1bdc6ee4-12c3-403f-93fd-284c7fafcf34
summary: brand.py loads media/brand.json with per-key fallback and render_brand placeholders;
  speech_synthesis.py synthesize_talk_audio via gpt-audio-1.5 MP3 for the talk tool
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Product branding and talk speech synthesis

## brand.py
Every user-visible name ("KISS Sorcar", the "KISS:" command prefix, the identity sentence in
`SYSTEM.md`) comes from `src/kiss/agents/vscode/media/brand.json` (`BRAND_FILE`). It lives in the
extension's `media/` directory because that is the one place reachable by the Python package, the
extension host (`src/brand.ts`) and the shared `chat.html` template.

- `load_brand(path)`: starts from `DEFAULT_BRAND` (`product_name`, `short_name`, `tagline`,
  `identity`, `extension_description`) and overrides key by key with non-empty string values from the
  file. Missing/malformed files fall back entirely, so a partial `brand.json` is enough.
- Module constants `BRAND`, `PRODUCT_NAME`, `SHORT_NAME` are computed at import.
- `render_brand(text)` fills `{{PRODUCT_NAME}}`, `{{SHORT_NAME}}`, `{{TAGLINE}}`, `{{IDENTITY}}` in the
  prompt files (`SYSTEM.md`, `SYSTEM_LITE.md`); unknown `{{...}}` tokens are left alone.
- Custom distributions re-brand by placing their files in the git-ignored `.brand/` directory at the
  checkout root; `install.sh` swaps them in for the extension build ("Brand overlay"), so checked-in
  files always carry the stock brand. Do not hard-code the product name in new code; import from
  `kiss.core.brand`.

## speech_synthesis.py
Server-side TTS for the `talk` tool, so every client plays the same natural voice instead of the
browser's Web Speech API (that robotic fallback was removed; browsers without audio stay silent and
the kiss-web daemon falls back to the system TTS command in `kiss.server.talk_player`).

- `synthesize_talk_audio(text, language="", emotion="", model="gpt-audio-1.5", voice="cedar",
  usage_out=None)` returns `(audio_b64, "audio/mpeg")` or `None` (empty text or any failure; it never
  raises, printing the error to stderr).
- Runs a single-shot, non-agentic `TalkSynthesisAgent` (a `KISSAgent` whose `_save` is a no-op so no
  trajectory is written per utterance) with `model_config` `modalities=["text","audio"]`,
  `audio={"voice", "format": "mp3"}` and a timeout. The MP3 comes back as the model's
  `last_audio_data`; no direct OpenAI SDK call.
- Prompt framing matters: `TTS_SYSTEM_PROMPT` casts the model as a text-to-speech engine reading a
  `Script:` block word for word; without it `gpt-audio-mini` answers the text instead of reading it.
  `emotion` is woven into the tone; `language` is appended as a hint.
- `usage_out` receives `budget_used` and `total_tokens_used` even on failure, so the caller can charge
  the spend (audio output is expensive) to its own task.
- Timeout: `KISS_VOICE_AUDIO_TIMEOUT` env override, default `DEFAULT_AUDIO_TIMEOUT_SECONDS = 60.0`;
  junk, non-finite or non-positive values fall back to the default.

## Sources
- `src/kiss/core/brand.py` (`BRAND_FILE`, `DEFAULT_BRAND`, `load_brand`, `render_brand`, `PRODUCT_NAME`)
- `src/kiss/core/speech_synthesis.py` (`synthesize_talk_audio`, `TalkSynthesisAgent`, `TTS_SYSTEM_PROMPT`, `audio_timeout_seconds`, `DEFAULT_TTS_MODEL`, `DEFAULT_TTS_VOICE`)
