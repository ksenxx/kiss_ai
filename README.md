<div align="center">

![KISS Framework](assets/KISS-Sorcar.png)

# When Simplicity Becomes Your Superpower: Meet KISS Sorcar, a General-purpose and Software engineering AI Assistant and IDE

[![Version](https://img.shields.io/badge/version-2026.5.4-blue?style=flat-square)](https://pypi.org/project/kiss-agent-framework/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue?style=flat-square)](https://www.python.org/)
[![Website](https://img.shields.io/badge/website-kisssorcar.github.io-1976d2?style=flat-square)](https://kisssorcar.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.23822-b31b1b?style=flat-square)](https://arxiv.org/abs/2604.23822)

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

**Website:** [https://kisssorcar.github.io/](https://kisssorcar.github.io/) · **Paper:** [arXiv:2604.23822](https://arxiv.org/abs/2604.23822)

</div>

______________________________________________________________________

KISS stands for ["Keep it Simple, Stupid"](https://en.wikipedia.org/wiki/KISS_principle) which is a well-known software engineering principle.

<details>
<summary><strong>Table of Contents</strong></summary>

- [Introduction to KISS Sorcar](#introduction-to-kiss-sorcar)
- [Full Installation](#full-installation)
- [KISS Sorcar Extension Installation](#kiss-sorcar-extension-installation)
- [CLI Interface](#cli-interface)
- [Messaging & Third-Party Agents](#-messaging--third-party-agents)
- [Models Supported](#-models-supported)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Authors](#%EF%B8%8F-authors)

</details>

# Introduction to KISS Sorcar

![KISS Sorcar](assets/KISS-Sorcar-UI.png)
**KISS Sorcar** (named after [P. C. Sorcar, the legendary Bengali magician](https://en.wikipedia.org/wiki/P._C._Sorcar), evoking the idea of an agent that performs feats that appear magical yet are grounded in disciplined engineering) is a **general-purpose assistant** and **integrated development environment** (IDE) built on top of the **KISS Agent Framework**, a stupidly-simple agentic framework. It **codes really well** and **works pretty fast**. It can do research for you without much AI slop using the internet. The agent can **run relentlessly for hours**. KISS Sorcar is implemented as a **Visual Studio Code extension** that runs **locally**. It has **full browser** support (using open-source Chromium browser and Playwright), **multimodal** support, **Docker container** support, and OpenClaw like features (whose functionality will be posted later in the social media). The good part is that KISS Sorcar is **completely free** and **open-source**; all one needs is a model API key from a major LLM provider, such as Anthropic (highly recommended). A paper on KISS Sorcar can be found at [papers/kisssorcar/kiss_sorcar.pdf](papers/kisssorcar/kiss_sorcar.pdf).

**KISS Sorcar scored 62.2% on Terminal Bench 2.0, beating both Cursor agent (61.7%) and Claude Code (58%).**

> **Built on time-tested, robust software engineering principles.** Both KISS Sorcar and the KISS Agent Framework are grounded in disciplined engineering practice. These principles are encoded directly into the agent's system prompt, enabling KISS Sorcar to write code that is **simple, elegant, maintainable, and bug-free**.

An old video on KISS Sorcar can be found at [https://www.youtube.com/watch?v=xnYxWvRqACE](https://www.youtube.com/watch?v=xnYxWvRqACE). We **no longer** recommend to explicitly create a plan in KISS Sorcar. See the paper for details.

<scriptsize>Note that **Sorcar** also means government in Bengali.</scriptsize>

## Full Installation

```
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

## KISS Sorcar Extension Installation

To Install KISS Sorcar, open Visual Studio Code, search for "KISS Sorcar" in the extension marketplace, install, and relaunch VS Code. Press ESC if you don't have a specific API key, but you must provide at least one API key.

You can also manually download the extension from [src/kiss/agents/vscode/kiss-sorcar.vsix](src/kiss/agents/vscode/kiss-sorcar.vsix).

## CLI Interface

If you do not want to use the KISS Sorcar IDE, you can open a terminal and use sorcar as a normal shell command. Some examples are:

```
sorcar -t "What is 2435*234"

sorcar -n -t --use-chat "What is 2435*234?" # to start in a new chat session in sorcar use -n

sorcar -m "claude-sonnet-4-6" -t "What is 2435*234?" # to use a specific model

echo "Can you find the cheapest non-stop flight from SFO to JFK on June 15 by consulting various websites?" > prompt
sorcar -f prompt # use contents of a file to send task

sorcar -t 'Can you send the message "Hello from Sorcar!" to ksen via the desktop slack app?'

sorcar -t 'Can you show me the detailed step-by-step workflow of gepa.py?'
```

## 🤖 Models Supported

**561 models** across 5 providers (OpenAI, Anthropic, Gemini, Together AI, OpenRouter) with built-in pricing, context lengths, and capability flags.

**Generation Models** (text generation with function calling support):

- **OpenAI**: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gpt-4.5-preview, gpt-4-turbo, gpt-4, gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro, gpt-5.1, gpt-5.2, gpt-5.2-pro, gpt-5.3-chat-latest, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-5.4-pro, gpt-5.5, gpt-5.5-pro
- **OpenAI (Codex)**: gpt-5-codex, gpt-5.1-codex, gpt-5.1-codex-max, gpt-5.1-codex-mini, gpt-5.2-codex, gpt-5.3-codex, codex-mini-latest
- **OpenAI (Reasoning)**: o1, o1-mini, o1-pro, o3, o3-mini, o3-mini-high, o3-pro, o3-deep-research, o4-mini, o4-mini-high, o4-mini-deep-research
- **OpenAI (Image & Audio)**: gpt-image-1, gpt-image-1-mini, gpt-image-1.5, gpt-image-2, gpt-audio, gpt-audio-mini, gpt-realtime, gpt-realtime-mini, computer-use-preview
- **OpenAI (Open Source)**: openai/gpt-oss-20b, openai/gpt-oss-120b
- **Anthropic**: claude-opus-4-7, claude-opus-4-6, claude-opus-4-5, claude-opus-4-1, claude-opus-4, claude-sonnet-4-6, claude-sonnet-4-5, claude-sonnet-4, claude-haiku-4-5
- **Anthropic (Legacy)**: claude-3-5-haiku
- **Anthropic (Claude Code)**: cc/haiku, cc/opus, cc/sonnet
- **Gemini**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-image, gemini-2.0-flash, gemini-2.0-flash-lite
- **Gemini (Preview)**: gemini-3-pro-preview, gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3.1-flash-lite-preview, gemini-3.1-flash-tts-preview, gemini-2.5-flash-lite
- **Gemini (Open Models)**: google/gemma-4-31B-it, google/gemma-3n-E4B-it, google/gemma-2-27b-it
- **Together AI (Llama)**: Llama-4-Scout/Maverick (with function calling), Llama-3.x series (generation only)
- **Together AI (Qwen)**: Qwen2.5-72B/14B/7B-Instruct, Qwen2.5-VL-72B, Qwen2.5-Coder-32B, Qwen2-VL-72B, QwQ-32B, Qwen3-235B series, Qwen3-Coder-480B, Qwen3-Coder-Next, Qwen3-Next-80B, Qwen3-VL-32B/8B, Qwen3.5-397B/9B, Qwen3.6-Plus (with function calling)
- **Together AI (DeepSeek)**: DeepSeek-R1, DeepSeek-R1-0528, DeepSeek-R1-Distill-Llama-70B, DeepSeek-R1-Distill-Qwen-1.5B/14B, DeepSeek-V3-0324, DeepSeek-V3.1, DeepSeek-V4-Pro, deepseek-coder-33b-instruct (with function calling)
- **Together AI (Kimi/Moonshot)**: Kimi-K2-Instruct, Kimi-K2-Instruct-0905, Kimi-K2-Thinking, Kimi-K2.5, Kimi-K2.6
- **Together AI (Mistral)**: Ministral-3-14B, Mistral-7B-v0.1/v0.2/v0.3, Mistral-Small-24B, Mixtral-8x7B
- **Together AI (Z.AI)**: GLM-5, GLM-5.1, GLM-4.5-Air, GLM-4.6, GLM-4.7
- **Together AI (MiniMax)**: MiniMax-M2.5, MiniMax-M2.7, minimax-m2.5-lightning
- **Together AI (DeepCogito)**: cogito-v1-preview (llama-70B/8B, qwen-14B/32B), cogito-v2-1-671b
- **Together AI (NVIDIA)**: Llama-3.1-Nemotron-70B, Nemotron-Nano-9B-v2
- **Together AI (Other)**: arcee-ai/trinity-mini, essentialai/rnj-1-instruct
- **OpenRouter**: Access to 338+ models from 55+ providers via unified API:
  - OpenAI (gpt-3.5-turbo through gpt-5.5, codex variants, o1/o3/o4-mini, gpt-oss, gpt-audio)
  - Anthropic (claude-3-haiku through claude-opus-4.7 with 1M context)
  - Google (gemini-2.0-flash through gemini-3.1-pro-preview, gemma-2/3/3n/4)
  - Meta Llama (llama-3-8b through llama-4-maverick/scout, llama-guard-3/4)
  - DeepSeek (deepseek-chat/v3/v3.1/v3.2/v3.2-speciale/v4-flash/v4-pro, deepseek-r1 variants)
  - Qwen (qwen-2.5 through qwen3.6, qwen3-coder variants, qwq-32b, qwen3-vl series)
  - Amazon Nova (nova-micro/lite/pro, nova-2-lite, nova-premier)
  - Cohere (command-r, command-r-plus, command-a, command-r7b)
  - X.AI Grok (grok-3/3-mini, grok-4/4-fast, grok-4.1-fast, grok-4.20/4.20-multi-agent, grok-code-fast-1)
  - MiniMax (minimax-01, minimax-m1, minimax-m2/m2.1/m2.5/m2.7/m2-her)
  - ByteDance Seed (seed-1.6, seed-1.6-flash, seed-2.0-lite, seed-2.0-mini)
  - MoonshotAI (kimi-k2, kimi-k2-thinking, kimi-k2.5, kimi-k2.6)
  - Mistral (codestral, devstral/devstral-medium/devstral-small, mistral-large/medium/small, mixtral series, ministral-3b/8b/14b, pixtral, voxtral)
  - NVIDIA (llama-3.1-nemotron-70b, llama-3.3-nemotron-super-49b, nemotron-nano-9b-v2/12b-v2-vl, nemotron-3-nano-30b/super-120b)
  - Z.AI/GLM (glm-4-32b through glm-5.1, glm-5v-turbo)
  - AllenAI (olmo-3-32b-think, olmo-3.1-32b-instruct)
  - Perplexity (sonar, sonar-pro, sonar-pro-search, sonar-deep-research, sonar-reasoning-pro)
  - NousResearch (hermes-2-pro, hermes-3/4-llama series, hermes-4-70b/405b)
  - Baidu ERNIE (ernie-4.5 series including VL and thinking variants)
  - Xiaomi (mimo-v2-flash/omni/pro, mimo-v2.5/v2.5-pro)
  - Reka AI (reka-edge, reka-flash-3)
  - InclusionAI (ling-2.6-flash)
  - Arcee AI (coder-large, maestro-reasoning, spotlight, trinity-large-preview/thinking, trinity-mini, virtuoso-large)
  - And 25+ more providers (ai21, aion-labs, alfredpros, alibaba, alpindale, anthracite-org, bytedance, deepcogito, essentialai, gryphe, ibm-granite, inception, inflection, kwaipilot, liquid, mancer, morph, nex-agi, prime-intellect, relace, sao10k, stepfun, switchpoint, tencent, thedrummer, tngtech, upstage, writer, etc.)
  - Dynamic latest-model aliases: ~anthropic/claude-{haiku,sonnet,opus}-latest, ~google/gemini-{flash,pro}-latest, ~moonshotai/kimi-latest, ~openai/gpt-{latest,mini-latest}

**Embedding Models** (for RAG and semantic search):

- **OpenAI**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Google**: text-embedding-004, gemini-embedding-001, gemini-embedding-2, gemini-embedding-2-preview
- **Together AI**: BAAI/bge-base-en-v1.5, intfloat/multilingual-e5-large-instruct

Each model in `MODEL_INFO` includes capability flags:

- `is_function_calling_supported`: Whether the model reliably supports tool/function calling
- `is_generation_supported`: Whether the model supports text generation
- `is_embedding_supported`: Whether the model is an embedding model

## 🤗 Contributing

Contributions in the form of issues are welcome! KISS Sorcar should be able to take care of them.

## 📄 License

Apache-2.0

## 📚 Citation

If you use KISS Sorcar in your research, please cite:

```bibtex
@misc{sen2026kisssorcar,
  title         = {KISS Sorcar: A Stupidly-Simple General-Purpose and Software Engineering AI Assistant},
  author        = {Sen, Koushik},
  year          = {2026},
  eprint        = {2604.23822},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2604.23822}
}
```
