---
title: Prompt attachments across providers (Attachment, images, PDF, audio, video,
  HEIC/HEIF transcoding)
uuid: b3c74269-1af0-4659-9ab6-475cdcba2a3a
summary: Attachment class and per-provider handling of images, PDF, audio, video;
  Whisper transcription for Anthropic; OpenAI format limits; HEIC/HEIF photos transcoded
  to JPEG.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Prompt attachments across providers

## `Attachment` (`src/kiss/core/models/model.py`)
Dataclass `(data: bytes, mime_type: str)`, passed to `Model.initialize(prompt, attachments)`.
- `Attachment.from_file(path)`: MIME from `mimetypes` (fallback suffix map); unsupported types raise
  `ValueError`.
- `Attachment.from_bytes(data, mime_type)`: if the bytes are HEIF (`is_heif`), transcode to JPEG.
- `SUPPORTED_MIME_TYPES`: JPEG, PNG, GIF, WEBP, PDF, audio (mpeg, wav, x-wav, ogg, webm, flac,
  aac, mp4), video (mp4, webm, ogg, mpeg, quicktime).

## Per-provider handling
| Adapter | Images | PDF | Audio | Video |
|---|---|---|---|---|
| Anthropic (`_attachments_to_blocks`) | `image` block | `document` block | transcribed with OpenAI Whisper (`transcribe_audio`, needs `OPENAI_API_KEY`) into a text block, else skipped | skipped with warning |
| OpenAI-compatible v1/v2 | PNG, JPEG, WEBP, GIF only (`OPENAI_INPUT_IMAGE_MIME_TYPES`) | file part | mp3/wav only (`OPENAI_INPUT_AUDIO_FORMATS`) as `input_audio` | dropped |
| Gemini | inline parts; accepts HEIF natively | inline | inline | inline |
| `cc/*`, `codex/*` | not supported, ignored with warning | - | - | - |
| Decisions | dropped with warning (text only) | - | - | - |

Both OpenAI transports apply the **same** format set on purpose: whether a turn goes through Chat
Completions or Responses is an internal routing decision and must not change what the model sees.

## Tool-result attachments
Tools may return binary payloads (screenshots, audio). `Model.tool_result_text_and_attachments`
splits them out. Anthropic puts image blocks directly inside the `tool_result`; other adapters
append a follow-up user message introduced by `TOOL_RESULT_ATTACHMENT_NOTE`; CLI models append a
note saying the attachments could not be shown (`CLITextModel._deliver_tool_result_attachments`),
so the model does not reason about a picture it never received.

## HEIC/HEIF (`src/kiss/core/models/heif.py`)
iPhones store photos as HEIC by default. OpenAI and Anthropic reject `image/heic`, Gemini accepts it.
- `is_heif(data)`: checks the ISO-BMFF header (`ftyp` at bytes 4..8 and a HEIF major brand such as
  `heic`, `heix`, `hevc`, `mif1`, `msf1` in `_HEIF_BRANDS`). The header is authoritative because iOS
  sometimes sends an empty MIME type and browsers disagree on `image/heic` vs `image/heif`.
- `heif_to_jpeg(data)`: tries host converters in order (`_CONVERTERS`): `sips` (macOS),
  `heif-convert -q 88` (libheif), `magick`, `ffmpeg`; 60 s timeout each (`_CONVERT_TIMEOUT_S`);
  output path cleared before each attempt so a truncated file from a failed converter is not
  mistaken for success. Returns `None` if none works.
- No decoder -> the attachment keeps the HEIF bytes labelled `image/heic`, which still works on
  Gemini. No native image-codec dependency is added to KISS.

## Sources
- `src/kiss/core/models/model.py` (`Attachment`, `Attachment.from_file`, `Attachment.from_bytes`, `SUPPORTED_MIME_TYPES`, `transcribe_audio`, `TOOL_RESULT_ATTACHMENT_NOTE`, `CLITextModel._deliver_tool_result_attachments`)
- `src/kiss/core/models/heif.py` (`is_heif`, `heif_to_jpeg`, `HEIF_MIME_TYPES`, `HEIF_SUFFIXES`, `_CONVERTERS`)
- `src/kiss/core/models/anthropic_model.py` (`_attachments_to_blocks`)
- `src/kiss/core/models/openai_compatible_model.py` (`OPENAI_INPUT_IMAGE_MIME_TYPES`, `OPENAI_INPUT_AUDIO_FORMATS`, `_attachments_to_content_parts`)
- `src/kiss/core/models/gemini_model.py` (`_media_block_to_part`)
