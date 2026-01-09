# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Model information: pricing and context lengths for supported LLM providers.

FLAKY MODEL MARKERS:
- fc=False: Model has unreliable function calling (use for non-agentic tasks only)
- Models with comments like "FLAKY" have inconsistent behavior
- Models with comments like "SLOW" may timeout on some requests
"""

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini3_model import Gemini3Model
from kiss.core.models.model import Model
from kiss.core.models.openai_model import OpenAIModel
from kiss.core.models.openrouter_model import OpenRouterModel
from kiss.core.models.together_model import TogetherModel


class ModelInfo:
    def __init__(
        self,
        context_length: int,
        input_price_per_million: float,
        output_price_per_million: float,
        is_function_calling_supported: bool,
        is_embedding_supported: bool,
        is_generation_supported: bool,
    ):
        self.context_length = context_length
        self.input_price_per_1M = input_price_per_million
        self.output_price_per_1M = output_price_per_million
        self.is_function_calling_supported = is_function_calling_supported
        self.is_embedding_supported = is_embedding_supported
        self.is_generation_supported = is_generation_supported


def _mi(
    ctx: int,
    inp: float,
    out: float,
    fc: bool = True,
    emb: bool = False,
    gen: bool = True,
) -> ModelInfo:
    """Helper to create ModelInfo with shorter syntax.

    Args:
        ctx: context_length
        inp: input_price_per_million
        out: output_price_per_million
        fc: is_function_calling_supported (default True for generation models)
        emb: is_embedding_supported (default False)
        gen: is_generation_supported (default True)
    """
    return ModelInfo(ctx, inp, out, fc, emb, gen)


def _emb(ctx: int, inp: float) -> ModelInfo:
    """Helper to create embedding-only ModelInfo."""
    return ModelInfo(ctx, inp, 0.0, False, True, False)


MODEL_INFO: dict[str, ModelInfo] = {
    # ==========================================================================
    # OpenAI models - GPT-4.1 series (current)
    # ==========================================================================
    "gpt-4.1": _mi(128000, 2.00, 8.00),
    "gpt-4.1-mini": _mi(128000, 0.40, 1.60),
    "gpt-4.1-nano": _mi(128000, 0.10, 0.40),
    # OpenAI models - GPT-4o series (legacy)
    "gpt-4o": _mi(128000, 2.50, 10.00),
    "gpt-4o-mini": _mi(128000, 0.15, 0.60),
    "gpt-4-turbo": _mi(128000, 10.00, 30.00),
    "gpt-4": _mi(8192, 30.00, 60.00),
    # GPT-5 models
    "gpt-5.1": _mi(400000, 1.25, 10.00),
    "gpt-5.2": _mi(400000, 1.75, 14.00),
    # ==========================================================================
    # OpenAI Embedding models
    # ==========================================================================
    "text-embedding-3-small": _emb(8191, 0.02),  # 1536 dimensions
    "text-embedding-3-large": _emb(8191, 0.13),  # 3072 dimensions
    "text-embedding-ada-002": _emb(8191, 0.10),  # 1536 dimensions (legacy)
    # ==========================================================================
    # Anthropic models (Note: Anthropic does not provide embedding API)
    # ==========================================================================
    "claude-opus-4-5": _mi(200000, 5.00, 25.00),
    "claude-opus-4-1": _mi(200000, 15.00, 75.00),
    "claude-sonnet-4-5": _mi(200000, 3.00, 15.00),
    "claude-haiku-4-5": _mi(200000, 1.00, 5.00),
    # ==========================================================================
    # Google Gemini models
    # ==========================================================================
    # Gemini 2.5 models
    "gemini-2.5-pro": _mi(1048576, 1.25, 10.00),
    "gemini-2.5-flash": _mi(1048576, 0.075, 0.30),
    "gemini-2.5-flash-lite": _mi(1048576, 0.0375, 0.15, fc=False),  # Poor tool use
    # Gemini 3 models
    "gemini-3-pro-preview": _mi(1000000, 2.00, 12.00),
    "gemini-3-flash-preview": _mi(1000000, 0.50, 3.00),
    # ==========================================================================
    # Google Gemini Embedding models
    # ==========================================================================
    "text-embedding-004": _emb(2048, 0.00),  # 768 dimensions, free tier
    # ==========================================================================
    # Together AI models - Llama series
    # ==========================================================================
    # Models with inconsistent/unreliable function calling (fc=False)
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": _mi(128000, 0.13, 0.40, fc=False),
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": _mi(128000, 1.79, 1.79, fc=False),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": _mi(128000, 0.35, 0.40, fc=False),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": _mi(128000, 0.09, 0.09, fc=False),
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": _mi(131072, 0.06, 0.06, fc=False),
    # Llama 4 models with function calling support
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": _mi(10000000, 0.20, 0.60),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": _mi(10000000, 0.11, 0.34),
    # ==========================================================================
    # Together AI models - Qwen series
    # ==========================================================================
    "Qwen/Qwen2.5-72B-Instruct-Turbo": _mi(131072, 1.20, 1.20),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": _mi(32768, 0.30, 0.30, fc=False),
    # Qwen3 series (function calling supported)
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": _mi(262144, 0.90, 0.90),
    "Qwen/Qwen3-235B-A22B-fp8-tput": _mi(40960, 0.90, 0.90, fc=False),
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": _mi(262144, 0.90, 0.90),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": _mi(262144, 0.30, 0.30),
    # ==========================================================================
    # Together AI models - Mistral series (most have unreliable function calling)
    # ==========================================================================
    "mistralai/Mixtral-8x7B-Instruct-v0.1": _mi(32768, 0.50, 0.50, fc=False),
    "mistralai/Mistral-7B-Instruct-v0.3": _mi(32768, 0.20, 0.20, fc=False),
    "mistralai/Mistral-7B-Instruct-v0.2": _mi(32768, 0.20, 0.20, fc=False),
    "mistralai/Mistral-Small-24B-Instruct-2501": _mi(32768, 0.80, 0.80, fc=False),
    "mistralai/Ministral-3-14B-Instruct-2512": _mi(262144, 0.30, 0.30, fc=False),
    # ==========================================================================
    # Together AI models - DeepSeek series
    # ==========================================================================
    "deepseek-ai/DeepSeek-R1": _mi(64000, 0.55, 2.19),
    "deepseek-ai/DeepSeek-R1-0528-tput": _mi(64000, 0.55, 2.19),
    "deepseek-ai/DeepSeek-V3": _mi(64000, 0.27, 1.10, fc=False),  # API issues
    "deepseek-ai/DeepSeek-V3.1": _mi(64000, 0.27, 1.10),
    # ==========================================================================
    # Together AI models - Other providers
    # ==========================================================================
    "google/gemma-3n-E4B-it": _mi(32768, 0.10, 0.10, fc=False),
    "moonshotai/Kimi-K2-Instruct-0905": _mi(262144, 0.60, 2.50),
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": _mi(131072, 0.15, 0.15),
    "zai-org/GLM-4.5-Air-FP8": _mi(131072, 0.10, 0.30),
    "zai-org/GLM-4.6": _mi(202752, 0.10, 0.30),
    "openai/gpt-oss-120b": _mi(131072, 0.50, 0.50, fc=False),
    "openai/gpt-oss-20b": _mi(131072, 0.20, 0.20, fc=False),
    # ==========================================================================
    # Together AI Embedding models
    # ==========================================================================
    "BAAI/bge-large-en-v1.5": _emb(512, 0.02),  # 1024 dimensions
    "BAAI/bge-base-en-v1.5": _emb(512, 0.01),  # 768 dimensions
    "togethercomputer/m2-bert-80M-32k-retrieval": _emb(32768, 0.008),  # 768 dim, 32k ctx
    "intfloat/multilingual-e5-large-instruct": _emb(514, 0.016),  # 1024 dimensions
    "Alibaba-NLP/gte-modernbert-base": _emb(8192, 0.008),  # 768 dimensions
    # ==========================================================================
    # OpenRouter models - AI21
    # ==========================================================================
    "openrouter/ai21/jamba-large-1.7": _mi(256000, 2.00, 8.00),
    "openrouter/ai21/jamba-mini-1.7": _mi(256000, 0.20, 0.40),
    # ==========================================================================
    # OpenRouter models - Alibaba
    # ==========================================================================
    "openrouter/alibaba/tongyi-deepresearch-30b-a3b": _mi(131072, 0.09, 0.40),
    # ==========================================================================
    # OpenRouter models - AllenAI
    # ==========================================================================
    "openrouter/allenai/olmo-3-7b-instruct": _mi(65536, 0.10, 0.20),
    # ==========================================================================
    # OpenRouter models - Amazon Nova
    # ==========================================================================
    "openrouter/amazon/nova-micro-v1": _mi(128000, 0.04, 0.14),
    "openrouter/amazon/nova-lite-v1": _mi(300000, 0.06, 0.24),
    "openrouter/amazon/nova-pro-v1": _mi(300000, 0.80, 3.20),
    "openrouter/amazon/nova-2-lite-v1": _mi(1000000, 0.30, 2.50),
    "openrouter/amazon/nova-premier-v1": _mi(1000000, 2.50, 12.50),
    # ==========================================================================
    # OpenRouter models - Anthropic
    # ==========================================================================
    "openrouter/anthropic/claude-3-haiku": _mi(200000, 0.25, 1.25),
    "openrouter/anthropic/claude-3.5-haiku": _mi(200000, 0.80, 4.00),
    "openrouter/anthropic/claude-3.5-sonnet": _mi(200000, 6.00, 30.00),
    "openrouter/anthropic/claude-3.7-sonnet": _mi(200000, 3.00, 15.00),
    "openrouter/anthropic/claude-sonnet-4": _mi(1000000, 3.00, 15.00),
    "openrouter/anthropic/claude-sonnet-4.5": _mi(1000000, 3.00, 15.00),
    "openrouter/anthropic/claude-haiku-4.5": _mi(200000, 1.00, 5.00),
    "openrouter/anthropic/claude-opus-4": _mi(200000, 15.00, 75.00),
    "openrouter/anthropic/claude-opus-4.1": _mi(200000, 15.00, 75.00),
    "openrouter/anthropic/claude-opus-4.5": _mi(200000, 5.00, 25.00),
    # ==========================================================================
    # OpenRouter models - Arcee AI
    # FLAKY: virtuoso-large returns 503 errors frequently
    # ==========================================================================
    "openrouter/arcee-ai/trinity-mini": _mi(131072, 0.04, 0.15),
    # ==========================================================================
    # OpenRouter models - Baidu ERNIE
    # FLAKY: ernie-4.5-21b-a3b has unreliable function calling (exceeds step limit)
    # ==========================================================================
    "openrouter/baidu/ernie-4.5-21b-a3b": _mi(120000, 0.07, 0.28, fc=False),
    # ==========================================================================
    # OpenRouter models - ByteDance Seed
    # ==========================================================================
    "openrouter/bytedance-seed/seed-1.6": _mi(262144, 0.25, 2.00),
    "openrouter/bytedance-seed/seed-1.6-flash": _mi(262144, 0.07, 0.30),
    # ==========================================================================
    # OpenRouter models - Cohere
    # ==========================================================================
    "openrouter/cohere/command-r-08-2024": _mi(128000, 0.15, 0.60),
    "openrouter/cohere/command-r-plus-08-2024": _mi(128000, 2.50, 10.00),
    "openrouter/cohere/command-a": _mi(256000, 2.50, 10.00, fc=False),
    # ==========================================================================
    # OpenRouter models - Deep Cogito
    # ==========================================================================
    "openrouter/deepcogito/cogito-v2-preview-llama-70b": _mi(32768, 0.88, 0.88),
    # ==========================================================================
    # OpenRouter models - DeepSeek
    # FLAKY: deepseek-chat can timeout; use deepseek-chat-v3.1 instead
    # ==========================================================================
    "openrouter/deepseek/deepseek-chat": _mi(163840, 0.30, 1.20),  # FLAKY: can timeout
    "openrouter/deepseek/deepseek-chat-v3-0324": _mi(163840, 0.19, 0.87),
    "openrouter/deepseek/deepseek-chat-v3.1": _mi(32768, 0.15, 0.75),
    "openrouter/deepseek/deepseek-r1": _mi(163840, 0.70, 2.40),  # SLOW: thinking model
    "openrouter/deepseek/deepseek-r1-0528": _mi(163840, 0.40, 1.75),  # SLOW: thinking
    "openrouter/deepseek/deepseek-r1-distill-llama-70b": _mi(131072, 0.03, 0.11),
    "openrouter/deepseek/deepseek-v3.1-terminus": _mi(163840, 0.21, 0.79),
    "openrouter/deepseek/deepseek-v3.2": _mi(163840, 0.25, 0.38),
    "openrouter/deepseek/deepseek-v3.2-exp": _mi(163840, 0.21, 0.32),
    "openrouter/deepseek/deepseek-prover-v2": _mi(163840, 0.50, 2.18, fc=False),
    # ==========================================================================
    # OpenRouter models - Google
    # ==========================================================================
    "openrouter/google/gemini-2.0-flash-001": _mi(1048576, 0.10, 0.40),
    "openrouter/google/gemini-2.0-flash-lite-001": _mi(1048576, 0.07, 0.30),
    "openrouter/google/gemini-2.5-flash": _mi(1048576, 0.30, 2.50),
    "openrouter/google/gemini-2.5-flash-lite": _mi(1048576, 0.10, 0.40),
    "openrouter/google/gemini-2.5-pro": _mi(1048576, 1.25, 10.00),
    "openrouter/google/gemini-2.5-pro-preview": _mi(1048576, 1.25, 10.00),
    "openrouter/google/gemini-3-flash-preview": _mi(1048576, 0.50, 3.00),
    "openrouter/google/gemini-3-pro-preview": _mi(1048576, 2.00, 12.00),
    "openrouter/google/gemma-3-27b-it": _mi(131072, 0.04, 0.06),
    "openrouter/google/gemma-3-12b-it": _mi(131072, 0.03, 0.10, fc=False),
    "openrouter/google/gemma-3-4b-it": _mi(96000, 0.02, 0.07, fc=False),
    "openrouter/google/gemma-3n-e4b-it": _mi(32768, 0.02, 0.04, fc=False),
    # ==========================================================================
    # OpenRouter models - Inception
    # ==========================================================================
    "openrouter/inception/mercury": _mi(128000, 0.25, 1.00),
    "openrouter/inception/mercury-coder": _mi(128000, 0.25, 1.00),
    # ==========================================================================
    # OpenRouter models - Kwaipilot
    # ==========================================================================
    "openrouter/kwaipilot/kat-coder-pro": _mi(256000, 0.21, 0.83),
    # ==========================================================================
    # OpenRouter models - Meta Llama
    # ==========================================================================
    "openrouter/meta-llama/llama-3-70b-instruct": _mi(8192, 0.30, 0.40),
    "openrouter/meta-llama/llama-3-8b-instruct": _mi(8192, 0.03, 0.06),
    "openrouter/meta-llama/llama-3.1-405b-instruct": _mi(10000, 3.50, 3.50),
    "openrouter/meta-llama/llama-3.1-70b-instruct": _mi(131072, 0.40, 0.40),
    "openrouter/meta-llama/llama-3.1-8b-instruct": _mi(16384, 0.02, 0.05),
    "openrouter/meta-llama/llama-3.2-3b-instruct": _mi(131072, 0.02, 0.02),
    "openrouter/meta-llama/llama-3.3-70b-instruct": _mi(131072, 0.10, 0.32),
    "openrouter/meta-llama/llama-4-maverick": _mi(1048576, 0.15, 0.60),
    "openrouter/meta-llama/llama-4-scout": _mi(327680, 0.08, 0.30),
    # ==========================================================================
    # OpenRouter models - Microsoft
    # ==========================================================================
    "openrouter/microsoft/phi-4": _mi(16384, 0.06, 0.14, fc=False),
    "openrouter/microsoft/phi-4-reasoning-plus": _mi(32768, 0.07, 0.35, fc=False),
    # ==========================================================================
    # OpenRouter models - MiniMax
    # ==========================================================================
    "openrouter/minimax/minimax-m1": _mi(1000000, 0.40, 2.20),
    "openrouter/minimax/minimax-m2": _mi(196608, 0.20, 1.00),
    "openrouter/minimax/minimax-m2.1": _mi(196608, 0.12, 0.48),
    # ==========================================================================
    # OpenRouter models - Mistral
    # ==========================================================================
    "openrouter/mistralai/codestral-2508": _mi(256000, 0.30, 0.90),
    "openrouter/mistralai/devstral-2512": _mi(262144, 0.05, 0.22),
    "openrouter/mistralai/devstral-medium": _mi(131072, 0.40, 2.00),
    "openrouter/mistralai/devstral-small": _mi(128000, 0.07, 0.28),
    "openrouter/mistralai/ministral-14b-2512": _mi(262144, 0.20, 0.20),
    "openrouter/mistralai/ministral-3b": _mi(131072, 0.04, 0.04),
    "openrouter/mistralai/ministral-8b": _mi(131072, 0.10, 0.10),
    "openrouter/mistralai/mistral-7b-instruct": _mi(32768, 0.03, 0.05),
    "openrouter/mistralai/mistral-large": _mi(128000, 2.00, 6.00),
    "openrouter/mistralai/mistral-large-2411": _mi(131072, 2.00, 6.00),
    "openrouter/mistralai/mistral-large-2512": _mi(262144, 0.50, 1.50),
    "openrouter/mistralai/mistral-medium-3": _mi(131072, 0.40, 2.00),
    "openrouter/mistralai/mistral-medium-3.1": _mi(131072, 0.40, 2.00),
    "openrouter/mistralai/mistral-nemo": _mi(131072, 0.02, 0.04),
    "openrouter/mistralai/mistral-saba": _mi(32768, 0.20, 0.60),
    "openrouter/mistralai/mistral-small-24b-instruct-2501": _mi(32768, 0.03, 0.11),
    "openrouter/mistralai/mistral-small-3.1-24b-instruct": _mi(131072, 0.03, 0.11),
    "openrouter/mistralai/mistral-small-3.2-24b-instruct": _mi(131072, 0.06, 0.18),
    "openrouter/mistralai/mistral-tiny": _mi(32768, 0.25, 0.25),
    "openrouter/mistralai/mixtral-8x22b-instruct": _mi(65536, 2.00, 6.00),
    "openrouter/mistralai/mixtral-8x7b-instruct": _mi(32768, 0.54, 0.54),
    "openrouter/mistralai/pixtral-12b": _mi(32768, 0.10, 0.10),
    "openrouter/mistralai/pixtral-large-2411": _mi(131072, 2.00, 6.00),
    # ==========================================================================
    # OpenRouter models - MoonshotAI
    # ==========================================================================
    "openrouter/moonshotai/kimi-k2": _mi(131072, 0.50, 2.40),
    "openrouter/moonshotai/kimi-k2-0905": _mi(262144, 0.39, 1.90),
    "openrouter/moonshotai/kimi-k2-thinking": _mi(262144, 0.32, 0.48),  # SLOW: thinking
    # ==========================================================================
    # OpenRouter models - Nous Research
    # ==========================================================================
    "openrouter/nousresearch/deephermes-3-mistral-24b-preview": _mi(32768, 0.02, 0.10),
    "openrouter/nousresearch/hermes-4-70b": _mi(131072, 0.11, 0.38),
    # ==========================================================================
    # OpenRouter models - NVIDIA
    # ==========================================================================
    "openrouter/nvidia/llama-3.1-nemotron-70b-instruct": _mi(131072, 1.20, 1.20),
    "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5": _mi(131072, 0.10, 0.40),
    "openrouter/nvidia/nemotron-3-nano-30b-a3b": _mi(262144, 0.06, 0.24),
    "openrouter/nvidia/nemotron-nano-9b-v2": _mi(131072, 0.04, 0.16),
    # ==========================================================================
    # OpenRouter models - OpenAI
    # ==========================================================================
    "openrouter/openai/gpt-3.5-turbo": _mi(16385, 0.50, 1.50),
    "openrouter/openai/gpt-3.5-turbo-16k": _mi(16385, 3.00, 4.00),
    "openrouter/openai/gpt-4": _mi(8191, 30.00, 60.00),
    "openrouter/openai/gpt-4-turbo": _mi(128000, 10.00, 30.00),
    "openrouter/openai/gpt-4.1": _mi(1047576, 2.00, 8.00),
    "openrouter/openai/gpt-4.1-mini": _mi(1047576, 0.40, 1.60),
    "openrouter/openai/gpt-4.1-nano": _mi(1047576, 0.10, 0.40),
    "openrouter/openai/gpt-4o": _mi(128000, 2.50, 10.00),
    "openrouter/openai/gpt-4o-mini": _mi(128000, 0.15, 0.60),
    "openrouter/openai/gpt-5": _mi(400000, 1.25, 10.00),
    "openrouter/openai/gpt-5-mini": _mi(400000, 0.25, 2.00),
    "openrouter/openai/gpt-5-nano": _mi(400000, 0.05, 0.40),
    "openrouter/openai/gpt-5.1": _mi(400000, 1.25, 10.00),
    "openrouter/openai/gpt-5.2": _mi(400000, 1.75, 14.00),
    "openrouter/openai/codex-mini": _mi(200000, 1.50, 6.00),
    "openrouter/openai/o1": _mi(200000, 15.00, 60.00),  # SLOW: reasoning model
    "openrouter/openai/o3": _mi(200000, 2.00, 8.00),
    "openrouter/openai/o3-mini": _mi(200000, 1.10, 4.40),
    "openrouter/openai/o3-mini-high": _mi(200000, 1.10, 4.40),
    "openrouter/openai/o3-pro": _mi(200000, 20.00, 80.00),  # SLOW: reasoning model
    "openrouter/openai/o4-mini": _mi(200000, 1.10, 4.40),
    "openrouter/openai/o4-mini-high": _mi(200000, 1.10, 4.40),
    "openrouter/openai/gpt-oss-120b": _mi(131072, 0.02, 0.10),
    "openrouter/openai/gpt-oss-20b": _mi(131072, 0.02, 0.06),
    # ==========================================================================
    # OpenRouter models - Perplexity
    # ==========================================================================
    "openrouter/perplexity/sonar": _mi(127072, 1.00, 1.00, fc=False),
    "openrouter/perplexity/sonar-pro": _mi(200000, 3.00, 15.00, fc=False),
    # ==========================================================================
    # OpenRouter models - Prime Intellect
    # ==========================================================================
    "openrouter/prime-intellect/intellect-3": _mi(131072, 0.20, 1.10),
    # ==========================================================================
    # OpenRouter models - Qwen
    # ==========================================================================
    "openrouter/qwen/qwen-2.5-72b-instruct": _mi(32768, 0.12, 0.39),
    "openrouter/qwen/qwen-2.5-7b-instruct": _mi(32768, 0.04, 0.10, fc=False),
    "openrouter/qwen/qwen-2.5-coder-32b-instruct": _mi(32768, 0.03, 0.11, fc=False),
    "openrouter/qwen/qwen-max": _mi(32768, 1.60, 6.40),
    "openrouter/qwen/qwen-plus": _mi(131072, 0.40, 1.20),
    "openrouter/qwen/qwen-turbo": _mi(1000000, 0.05, 0.20),
    "openrouter/qwen/qwen3-8b": _mi(128000, 0.04, 0.14),
    "openrouter/qwen/qwen3-14b": _mi(40960, 0.05, 0.22),
    "openrouter/qwen/qwen3-30b-a3b": _mi(40960, 0.06, 0.22),
    "openrouter/qwen/qwen3-32b": _mi(40960, 0.08, 0.24),
    "openrouter/qwen/qwen3-235b-a22b": _mi(40960, 0.18, 0.54),
    "openrouter/qwen/qwen3-235b-a22b-2507": _mi(262144, 0.07, 0.46),
    "openrouter/qwen/qwen3-coder": _mi(262144, 0.22, 0.95),
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct": _mi(160000, 0.07, 0.27),
    "openrouter/qwen/qwen3-coder-flash": _mi(128000, 0.30, 1.50),
    "openrouter/qwen/qwen3-coder-plus": _mi(128000, 1.00, 5.00),
    "openrouter/qwen/qwen3-max": _mi(256000, 1.20, 6.00),
    "openrouter/qwen/qwen3-next-80b-a3b-instruct": _mi(262144, 0.06, 0.60),
    "openrouter/qwen/qwq-32b": _mi(32768, 0.15, 0.40),  # SLOW: reasoning model
    # ==========================================================================
    # OpenRouter models - StepFun
    # ==========================================================================
    "openrouter/stepfun-ai/step3": _mi(65536, 0.57, 1.42),
    # ==========================================================================
    # OpenRouter models - X.AI Grok
    # ==========================================================================
    "openrouter/x-ai/grok-3": _mi(131072, 3.00, 15.00),
    "openrouter/x-ai/grok-3-beta": _mi(131072, 3.00, 15.00),
    "openrouter/x-ai/grok-3-mini": _mi(131072, 0.30, 0.50),
    "openrouter/x-ai/grok-3-mini-beta": _mi(131072, 0.30, 0.50),
    "openrouter/x-ai/grok-4": _mi(256000, 3.00, 15.00),
    "openrouter/x-ai/grok-4-fast": _mi(2000000, 0.20, 0.50),
    "openrouter/x-ai/grok-4.1-fast": _mi(2000000, 0.20, 0.50),
    "openrouter/x-ai/grok-code-fast-1": _mi(256000, 0.20, 1.50),
    # ==========================================================================
    # OpenRouter models - Z.AI GLM
    # ==========================================================================
    "openrouter/z-ai/glm-4-32b": _mi(128000, 0.10, 0.10),
    "openrouter/z-ai/glm-4.5": _mi(131072, 0.35, 1.55),
    "openrouter/z-ai/glm-4.5-air": _mi(131072, 0.05, 0.22),
    "openrouter/z-ai/glm-4.5v": _mi(65536, 0.60, 1.80),
    "openrouter/z-ai/glm-4.6": _mi(202752, 0.35, 1.50),
    "openrouter/z-ai/glm-4.6v": _mi(131072, 0.30, 0.90),
    "openrouter/z-ai/glm-4.7": _mi(202752, 0.16, 0.80),
}

# ==========================================================================
# FLAKY MODEL REGISTRY
# Models that have been tested and found to have reliability issues
# ==========================================================================
FLAKY_MODELS: dict[str, str] = {
    # Models with 503 errors (service unavailable)
    "openrouter/arcee-ai/virtuoso-large": "503 Service Unavailable errors",
    # Models with unreliable function calling
    "openrouter/baidu/ernie-4.5-21b-a3b": "Exceeds step limit in agentic mode",
    # Models that timeout frequently
    "openrouter/deepseek/deepseek-chat": "Can timeout on requests",
    # Slow thinking/reasoning models (not flaky, just slow)
    "openrouter/deepseek/deepseek-r1": "Slow: thinking model",
    "openrouter/deepseek/deepseek-r1-0528": "Slow: thinking model",
    "openrouter/openai/o1": "Slow: reasoning model",
    "openrouter/openai/o3-pro": "Slow: reasoning model",
    "openrouter/qwen/qwq-32b": "Slow: reasoning model",
    "openrouter/moonshotai/kimi-k2-thinking": "Slow: thinking model",
}


def is_model_flaky(model_name: str) -> bool:
    """Check if a model is known to be flaky."""
    return model_name in FLAKY_MODELS


def get_flaky_reason(model_name: str) -> str:
    """Get the reason why a model is flaky."""
    return FLAKY_MODELS.get(model_name, "")


def model(model_name: str) -> Model:
    """Get a model instance based on model name prefix."""
    # OpenRouter models (strip "openrouter/" prefix for API calls)
    if model_name.startswith("openrouter/"):
        # Create OpenRouter model with the model ID after "openrouter/"
        openrouter_model_id = model_name[len("openrouter/") :]
        m = OpenRouterModel(openrouter_model_id)
        return m
    # Google Gemini models (generation and embedding) - check before OpenAI
    elif model_name.startswith("gemini") or model_name == "text-embedding-004":
        return Gemini3Model(model_name)
    # OpenAI models (generation and embedding)
    elif model_name.startswith(("gpt", "text-embedding")) and not model_name.startswith(
        "openai/gpt-oss"
    ):
        return OpenAIModel(model_name)
    elif model_name.startswith("claude"):
        return AnthropicModel(model_name)
    # Together AI models (generation and embedding)
    elif model_name.startswith(
        (
            "meta-llama/",
            "Qwen/",
            "mistralai/",
            "deepseek-ai/",
            "google/gemma",
            "moonshotai/",
            "nvidia/",
            "zai-org/",
            "openai/gpt-oss",
            # Together AI embedding models
            "BAAI/",
            "togethercomputer/",
            "intfloat/",
            "Alibaba-NLP/",
        )
    ):
        return TogetherModel(model_name)
    else:
        raise KISSError(f"Unknown model name: {model_name}")


def calculate_cost(
    model_name: str, num_input_tokens: int, num_output_tokens: int
) -> float:
    """Calculates the cost in USD for the given token counts.

    Args:
        model_name: Name of the model.
        num_input_tokens: Number of input tokens.
        num_output_tokens: Number of output tokens.

    Returns:
        float: Cost in USD, or 0.0 if pricing is not available for the model.
    """
    info = MODEL_INFO.get(model_name)
    if info is None:
        return 0.0
    input_cost: float = (num_input_tokens / 1_000_000) * info.input_price_per_1M
    output_cost: float = (num_output_tokens / 1_000_000) * info.output_price_per_1M
    return input_cost + output_cost


def get_max_context_length(model_name: str) -> int:
    """Returns the maximum context length supported by the model.

    Args:
        model_name: Name of the model.
    Returns:
        int: Maximum context length in tokens.
    """
    return MODEL_INFO[model_name].context_length
