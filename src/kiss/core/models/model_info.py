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

from typing import Any

from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Model, ThinkingCallback, TokenCallback


class ModelInfo:
    """Container for model metadata including pricing and capabilities."""

    def __init__(
        self,
        context_length: int,
        input_price_per_million: float,
        output_price_per_million: float,
        is_function_calling_supported: bool,
        is_embedding_supported: bool,
        is_generation_supported: bool,
        cache_read_price_per_million: float | None = None,
        cache_write_price_per_million: float | None = None,
    ):
        self.context_length = context_length
        self.input_price_per_1M = input_price_per_million
        self.output_price_per_1M = output_price_per_million
        self.is_function_calling_supported = is_function_calling_supported
        self.is_embedding_supported = is_embedding_supported
        self.is_generation_supported = is_generation_supported
        self.cache_read_price_per_1M = cache_read_price_per_million
        self.cache_write_price_per_1M = cache_write_price_per_million


def _mi(
    ctx: int,
    inp: float,
    out: float,
    fc: bool = True,
    emb: bool = False,
    gen: bool = True,
    cr: float | None = None,
    cw: float | None = None,
) -> ModelInfo:
    """Helper to create ModelInfo with shorter syntax.

    Args:
        ctx: context_length
        inp: input_price_per_million
        out: output_price_per_million
        fc: is_function_calling_supported (default True for generation models)
        emb: is_embedding_supported (default False)
        gen: is_generation_supported (default True)
        cr: cache_read_price_per_million (None = use input price)
        cw: cache_write_price_per_million (None = use input price)
    """
    return ModelInfo(ctx, inp, out, fc, emb, gen, cr, cw)


def _emb(ctx: int, inp: float) -> ModelInfo:
    """Helper to create embedding-only ModelInfo.

    Args:
        ctx: Maximum context length in tokens.
        inp: Input price per million tokens.

    Returns:
        ModelInfo: A ModelInfo configured for embedding models.
    """
    return ModelInfo(ctx, inp, 0.0, False, True, False)


_OPENAI_PREFIXES = ("gpt", "text-embedding", "o1", "o3", "o4", "codex", "computer-use")
_TOGETHER_PREFIXES = (
    "meta-llama/",
    "Qwen/",
    "MiniMaxAI/",
    "mistralai/",
    "deepseek-ai/",
    "deepcogito/",
    "google/gemma",
    "moonshotai/",
    "nvidia/",
    "zai-org/",
    "openai/gpt-oss",
    "arcee-ai/",
    "essentialai/",
    "BAAI/",
    "intfloat/",
)


def _openai_compatible(
    model_name: str,
    base_url: str,
    api_key: str,
    model_config: dict[str, Any] | None,
    token_callback: TokenCallback | None,
    thinking_callback: ThinkingCallback | None = None,
) -> Model:
    from kiss.core.models import OpenAICompatibleModel

    if OpenAICompatibleModel is None:  # pragma: no cover – openai always installed
        raise KISSError("OpenAI SDK not installed. Install 'openai' to use this model.")
    return OpenAICompatibleModel(  # type: ignore[no-any-return]
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        model_config=model_config,
        token_callback=token_callback,
        thinking_callback=thinking_callback,
    )


MODEL_INFO: dict[str, ModelInfo] = {
    "arcee-ai/trinity-mini": _mi(128000, 0.04, 0.15, fc=False),
    "BAAI/bge-base-en-v1.5": _emb(512, 0.01),
    "cc/haiku": _mi(200000, 0.00, 0.00),
    "cc/opus": _mi(200000, 0.00, 0.00),
    "cc/sonnet": _mi(200000, 0.00, 0.00),
    "claude-haiku-4-5": _mi(200000, 1.00, 5.00),
    "claude-haiku-4-5-20251001": _mi(200000, 1.00, 5.00),
    "claude-opus-4": _mi(200000, 15.00, 75.00),
    "claude-opus-4-1": _mi(200000, 15.00, 75.00),
    "claude-opus-4-1-20250805": _mi(200000, 15.00, 75.00),
    "claude-opus-4-20250514": _mi(200000, 15.00, 75.00),
    "claude-opus-4-5": _mi(200000, 5.00, 25.00),
    "claude-opus-4-5-20251101": _mi(200000, 5.00, 25.00),
    "claude-opus-4-6": _mi(200000, 5.00, 25.00),
    "claude-opus-4-7": _mi(200000, 5.00, 25.00),
    "claude-sonnet-4": _mi(200000, 3.00, 15.00),
    "claude-sonnet-4-20250514": _mi(200000, 3.00, 15.00),
    "claude-sonnet-4-5": _mi(200000, 3.00, 15.00),
    "claude-sonnet-4-5-20250929": _mi(200000, 3.00, 15.00),
    "claude-sonnet-4-6": _mi(200000, 3.00, 15.00),
    "computer-use-preview": _mi(128000, 3.00, 12.00),
    "computer-use-preview-2025-03-11": _mi(128000, 3.00, 12.00),
    "deepcogito/cogito-v1-preview-llama-70B": _mi(131072, 0.00, 0.00),  # NEW: needs pricing
    "deepcogito/cogito-v1-preview-llama-70B-Turbo": _mi(131072, 0.00, 0.00),  # NEW: needs pricing
    "deepcogito/cogito-v1-preview-llama-8B": _mi(131072, 0.00, 0.00),  # NEW: needs pricing
    "deepcogito/cogito-v1-preview-qwen-14B": _mi(131072, 0.00, 0.00),  # NEW: needs pricing
    "deepcogito/cogito-v1-preview-qwen-32B": _mi(131072, 0.00, 0.00),  # NEW: needs pricing
    "deepcogito/cogito-v2-1-671b": _mi(163840, 1.25, 1.25, fc=False),
    "deepseek-ai/deepseek-coder-33b-instruct": _mi(16384, 0.80, 0.80),  # NEW
    "deepseek-ai/DeepSeek-R1": _mi(163840, 3.00, 7.00, fc=False),
    "deepseek-ai/DeepSeek-R1-0528": _mi(163840, 3.00, 7.00),
    "deepseek-ai/DeepSeek-R1-0528-tput": _mi(163840, 3.00, 7.00),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": _mi(131072, 2.00, 2.00),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": _mi(131072, 0.18, 0.18),  # NEW
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": _mi(131072, 1.60, 1.60),  # NEW
    "deepseek-ai/DeepSeek-V3-0324": _mi(163840, 1.25, 1.25),
    "deepseek-ai/DeepSeek-V3.1": _mi(131072, 0.60, 1.70, fc=False),
    "deepseek-ai/DeepSeek-V4-Pro": _mi(512000, 2.10, 4.40),  # NEW
    "essentialai/rnj-1-instruct": _mi(32768, 0.15, 0.15, fc=False),
    "gemini-2.0-flash": _mi(1048576, 0.10, 0.40),
    "gemini-2.0-flash-001": _mi(1048576, 0.10, 0.40),
    "gemini-2.0-flash-lite": _mi(1048576, 0.075, 0.30),
    "gemini-2.0-flash-lite-001": _mi(1048576, 0.075, 0.30),
    "gemini-2.5-flash": _mi(1048576, 0.30, 2.50),
    "gemini-2.5-flash-image": _mi(32768, 0.30, 2.50),
    "gemini-2.5-flash-lite": _mi(1048576, 0.10, 0.40, fc=False),
    "gemini-2.5-pro": _mi(1048576, 1.25, 10.00),
    "gemini-3-flash-preview": _mi(1048576, 0.50, 3.00),
    "gemini-3-pro-preview": _mi(1048576, 2.00, 12.00),
    "gemini-3.1-flash-lite-preview": _mi(1048576, 0.25, 1.50),
    "gemini-3.1-flash-tts-preview": _mi(8192, 0.00, 0.00),  # NEW: needs pricing
    "gemini-3.1-pro-preview": _mi(1048576, 2.00, 12.00),
    "gemini-embedding-001": _emb(2048, 0.15),
    "gemini-embedding-2": _mi(8192, 0.00, 0.00),  # NEW: needs pricing
    "gemini-embedding-2-preview": _emb(8192, 0.00),
    "google/gemma-2-27b-it": _mi(8192, 0.80, 0.80),
    "google/gemma-3n-E4B-it": _mi(32768, 0.06, 0.12, fc=False),
    "google/gemma-4-31B-it": _mi(262144, 0.20, 0.50),
    "gpt-3.5-turbo": _mi(16385, 0.50, 1.50),
    "gpt-3.5-turbo-0125": _mi(16385, 0.50, 1.50),
    "gpt-3.5-turbo-1106": _mi(16385, 1.00, 2.00),
    "gpt-3.5-turbo-16k": _mi(16385, 3.00, 4.00),
    "gpt-4": _mi(8192, 30.00, 60.00),
    "gpt-4-0613": _mi(8192, 30.00, 60.00),
    "gpt-4-turbo": _mi(128000, 10.00, 30.00),
    "gpt-4-turbo-2024-04-09": _mi(128000, 10.00, 30.00),
    "gpt-4.1": _mi(128000, 2.00, 8.00),
    "gpt-4.1-2025-04-14": _mi(1047576, 2.00, 8.00),
    "gpt-4.1-mini": _mi(128000, 0.40, 1.60),
    "gpt-4.1-mini-2025-04-14": _mi(1047576, 0.40, 1.60),
    "gpt-4.1-nano": _mi(128000, 0.10, 0.40, fc=False),
    "gpt-4.1-nano-2025-04-14": _mi(1047576, 0.10, 0.40),
    "gpt-4o": _mi(128000, 2.50, 10.00),
    "gpt-4o-2024-05-13": _mi(128000, 5.00, 15.00),
    "gpt-4o-2024-08-06": _mi(128000, 2.50, 10.00),
    "gpt-4o-2024-11-20": _mi(128000, 2.50, 10.00),
    "gpt-4o-mini": _mi(128000, 0.15, 0.60),
    "gpt-4o-mini-2024-07-18": _mi(128000, 0.15, 0.60),
    "gpt-4o-mini-search-preview": _mi(128000, 0.15, 0.60),
    "gpt-4o-mini-search-preview-2025-03-11": _mi(128000, 0.15, 0.60),
    "gpt-4o-search-preview": _mi(128000, 2.50, 10.00),
    "gpt-4o-search-preview-2025-03-11": _mi(128000, 2.50, 10.00),
    "gpt-5": _mi(400000, 1.25, 10.00),
    "gpt-5-2025-08-07": _mi(400000, 1.25, 10.00),
    "gpt-5-chat-latest": _mi(400000, 1.25, 10.00),
    "gpt-5-codex": _mi(400000, 1.25, 10.00),
    "gpt-5-mini": _mi(400000, 0.25, 2.00),
    "gpt-5-mini-2025-08-07": _mi(400000, 0.25, 2.00),
    "gpt-5-nano": _mi(400000, 0.05, 0.40, fc=False),
    "gpt-5-nano-2025-08-07": _mi(400000, 0.05, 0.40),
    "gpt-5-pro": _mi(400000, 15.00, 120.00),
    "gpt-5-pro-2025-10-06": _mi(400000, 15.00, 120.00),
    "gpt-5.1": _mi(400000, 1.25, 10.00),
    "gpt-5.1-2025-11-13": _mi(400000, 1.25, 10.00),
    "gpt-5.1-chat-latest": _mi(400000, 1.25, 10.00),
    "gpt-5.1-codex": _mi(400000, 1.25, 10.00),
    "gpt-5.1-codex-max": _mi(400000, 1.25, 10.00),
    "gpt-5.1-codex-mini": _mi(400000, 0.25, 2.00),
    "gpt-5.2": _mi(400000, 1.75, 14.00),
    "gpt-5.2-2025-12-11": _mi(400000, 1.75, 14.00),
    "gpt-5.2-chat-latest": _mi(400000, 1.75, 14.00),
    "gpt-5.2-codex": _mi(400000, 1.75, 14.00),
    "gpt-5.2-pro": _mi(400000, 21.00, 168.00),
    "gpt-5.2-pro-2025-12-11": _mi(400000, 21.00, 168.00),
    "gpt-5.3-chat-latest": _mi(400000, 1.75, 14.00),
    "gpt-5.3-codex": _mi(400000, 1.75, 14.00),
    "gpt-5.4": _mi(1050000, 2.50, 15.00),
    "gpt-5.4-2026-03-05": _mi(1050000, 2.50, 15.00),
    "gpt-5.4-mini": _mi(400000, 0.75, 4.50),
    "gpt-5.4-mini-2026-03-17": _mi(400000, 0.75, 4.50),
    "gpt-5.4-nano": _mi(400000, 0.20, 1.25),
    "gpt-5.4-nano-2026-03-17": _mi(400000, 0.20, 1.25),
    "gpt-5.4-pro": _mi(1050000, 30.00, 180.00),
    "gpt-5.4-pro-2026-03-05": _mi(1050000, 30.00, 180.00),
    "gpt-5.5": _mi(1050000, 5.00, 30.00),  # NEW
    "gpt-5.5-2026-04-23": _mi(1050000, 5.00, 30.00),  # NEW
    "gpt-5.5-pro": _mi(1050000, 30.00, 180.00),  # NEW
    "gpt-5.5-pro-2026-04-23": _mi(1050000, 30.00, 180.00),  # NEW
    "gpt-image-1": _mi(32768, 5.00, 40.00, fc=False),
    "gpt-image-1-mini": _mi(32768, 1.50, 12.00, fc=False),
    "gpt-image-1.5": _mi(32768, 5.00, 40.00, fc=False),
    "gpt-image-2": _mi(32768, 5.00, 30.00, fc=False),
    "gpt-image-2-2026-04-21": _mi(32768, 5.00, 30.00, fc=False),
    "intfloat/multilingual-e5-large-instruct": _emb(514, 0.02),
    "meta-llama/Llama-3-70b-chat-hf": _mi(8192, 0.88, 0.88, fc=False),
    "meta-llama/Llama-3-8b-chat-hf": _mi(8192, 0.20, 0.20),
    "meta-llama/Llama-3.1-405B-Instruct": _mi(4096, 3.50, 3.50),
    "meta-llama/Llama-3.2-1B-Instruct": _mi(131072, 0.06, 0.06),
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": _mi(131072, 0.06, 0.06, fc=False),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": _mi(131072, 0.88, 0.88, fc=False),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-test": _mi(131072, 0.88, 0.88),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": _mi(1048576, 0.27, 0.85),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": _mi(1048576, 0.18, 0.59),
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": _mi(8192, 0.88, 0.88),
    "meta-llama/Meta-Llama-3-8B-Instruct": _mi(8192, 0.20, 0.20),
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite": _mi(8192, 0.10, 0.10, fc=False),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference": _mi(8192, 0.90, 0.90),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": _mi(131072, 0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference": _mi(16384, 0.20, 0.20),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": _mi(131072, 0.18, 0.18, fc=False),
    "minimax-m2.5": _mi(1000000, 0.15, 1.20),
    "minimax-m2.5-lightning": _mi(1000000, 0.30, 2.40),
    "MiniMaxAI/MiniMax-M2.5": _mi(196608, 0.30, 1.20),
    "MiniMaxAI/MiniMax-M2.7": _mi(196608, 0.30, 1.20),
    "mistralai/Ministral-3-14B-Instruct-2512": _mi(262144, 0.20, 0.20, fc=False),
    "mistralai/Mistral-7B-Instruct-v0.1": _mi(32768, 0.20, 0.20),
    "mistralai/Mistral-7B-Instruct-v0.2": _mi(32768, 0.20, 0.20, fc=False),
    "mistralai/Mistral-7B-Instruct-v0.3": _mi(32768, 0.20, 0.20, fc=False),
    "mistralai/Mistral-Small-24B-Instruct-2501": _mi(32768, 0.10, 0.30, fc=False),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": _mi(32768, 0.60, 0.60, fc=False),
    "moonshotai/Kimi-K2-Instruct": _mi(128000, 1.00, 3.00, fc=False),
    "moonshotai/Kimi-K2-Instruct-0905": _mi(262144, 1.00, 3.00, fc=False),
    "moonshotai/Kimi-K2-Thinking": _mi(262144, 1.20, 4.00, fc=False),
    "moonshotai/Kimi-K2.5": _mi(262144, 0.50, 2.80),
    "moonshotai/Kimi-K2.6": _mi(262144, 1.20, 4.50),  # NEW
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": _mi(32768, 0.88, 0.88),
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": _mi(131072, 0.06, 0.25),
    "o1": _mi(200000, 15.00, 60.00, fc=False),
    "o1-2024-12-17": _mi(200000, 15.00, 60.00),
    "o1-pro": _mi(200000, 150.00, 600.00, fc=False),
    "o1-pro-2025-03-19": _mi(200000, 150.00, 600.00),
    "o3": _mi(200000, 2.00, 8.00),
    "o3-2025-04-16": _mi(200000, 2.00, 8.00),
    "o3-deep-research": _mi(200000, 10.00, 40.00, fc=False),
    "o3-deep-research-2025-06-26": _mi(200000, 10.00, 40.00),
    "o3-mini": _mi(200000, 1.10, 4.40, fc=False),
    "o3-mini-2025-01-31": _mi(200000, 1.10, 4.40),
    "o3-pro": _mi(200000, 20.00, 80.00, fc=False),
    "o3-pro-2025-06-10": _mi(200000, 20.00, 80.00),
    "o4-mini": _mi(200000, 1.10, 4.40),
    "o4-mini-2025-04-16": _mi(200000, 1.10, 4.40),
    "o4-mini-deep-research": _mi(200000, 2.00, 8.00, fc=False),
    "o4-mini-deep-research-2025-06-26": _mi(200000, 2.00, 8.00),
    "openai/gpt-oss-120b": _mi(131072, 0.15, 0.60, fc=False),
    "openai/gpt-oss-20b": _mi(131072, 0.05, 0.20, fc=False),
    "openrouter/ai21/jamba-large-1.7": _mi(256000, 2.00, 8.00, fc=False),
    "openrouter/aion-labs/aion-1.0": _mi(131072, 4.00, 8.00, fc=False),
    "openrouter/aion-labs/aion-1.0-mini": _mi(131072, 0.70, 1.40, fc=False),
    "openrouter/aion-labs/aion-2.0": _mi(131072, 0.80, 1.60),
    "openrouter/aion-labs/aion-rp-llama-3.1-8b": _mi(32768, 0.80, 1.60, fc=False),
    "openrouter/alfredpros/codellama-7b-instruct-solidity": _mi(4096, 0.80, 1.20, fc=False),
    "openrouter/alibaba/tongyi-deepresearch-30b-a3b": _mi(131072, 0.09, 0.45, fc=False),
    "openrouter/allenai/olmo-3-32b-think": _mi(65536, 0.15, 0.50, fc=False),
    "openrouter/allenai/olmo-3.1-32b-instruct": _mi(65536, 0.20, 0.60, fc=False),
    "openrouter/alpindale/goliath-120b": _mi(6144, 3.75, 7.50, fc=False),
    "openrouter/amazon/nova-2-lite-v1": _mi(1000000, 0.30, 2.50, fc=False),
    "openrouter/amazon/nova-lite-v1": _mi(300000, 0.06, 0.24),
    "openrouter/amazon/nova-micro-v1": _mi(128000, 0.04, 0.14, fc=False),
    "openrouter/amazon/nova-premier-v1": _mi(1000000, 2.50, 12.50),
    "openrouter/amazon/nova-pro-v1": _mi(300000, 0.80, 3.20),
    "openrouter/anthracite-org/magnum-v4-72b": _mi(16384, 3.00, 5.00, fc=False),
    "openrouter/anthropic/claude-3-haiku": _mi(200000, 0.25, 1.25),
    "openrouter/anthropic/claude-3.5-haiku": _mi(200000, 0.80, 4.00),
    "openrouter/anthropic/claude-3.7-sonnet": _mi(200000, 3.00, 15.00),
    "openrouter/anthropic/claude-3.7-sonnet:thinking": _mi(200000, 3.00, 15.00),
    "openrouter/anthropic/claude-haiku-4.5": _mi(200000, 1.00, 5.00),
    "openrouter/anthropic/claude-opus-4": _mi(200000, 15.00, 75.00),
    "openrouter/anthropic/claude-opus-4.1": _mi(200000, 15.00, 75.00),
    "openrouter/anthropic/claude-opus-4.5": _mi(200000, 5.00, 25.00),
    "openrouter/anthropic/claude-opus-4.6": _mi(1000000, 5.00, 25.00),
    "openrouter/anthropic/claude-opus-4.6-fast": _mi(1000000, 30.00, 150.00),
    "openrouter/anthropic/claude-opus-4.7": _mi(1000000, 5.00, 25.00),
    "openrouter/anthropic/claude-sonnet-4": _mi(1000000, 3.00, 15.00),
    "openrouter/anthropic/claude-sonnet-4.5": _mi(1000000, 3.00, 15.00),
    "openrouter/anthropic/claude-sonnet-4.6": _mi(1000000, 3.00, 15.00),
    "openrouter/arcee-ai/coder-large": _mi(32768, 0.50, 0.80, fc=False),
    "openrouter/arcee-ai/maestro-reasoning": _mi(131072, 0.90, 3.30, fc=False),
    "openrouter/arcee-ai/spotlight": _mi(131072, 0.18, 0.18, fc=False),
    "openrouter/arcee-ai/trinity-large-preview": _mi(131000, 0.15, 0.45),  # NEW
    "openrouter/arcee-ai/trinity-large-thinking": _mi(262144, 0.22, 0.85),
    "openrouter/arcee-ai/trinity-mini": _mi(131072, 0.045, 0.15, fc=False),
    "openrouter/arcee-ai/virtuoso-large": _mi(131072, 0.75, 1.20),
    "openrouter/baidu/ernie-4.5-21b-a3b": _mi(120000, 0.07, 0.28, fc=False),
    "openrouter/baidu/ernie-4.5-21b-a3b-thinking": _mi(131072, 0.07, 0.28, fc=False),
    "openrouter/baidu/ernie-4.5-300b-a47b": _mi(123000, 0.28, 1.10, fc=False),
    "openrouter/baidu/ernie-4.5-vl-28b-a3b": _mi(30000, 0.14, 0.56),
    "openrouter/baidu/ernie-4.5-vl-424b-a47b": _mi(123000, 0.42, 1.25, fc=False),
    "openrouter/bytedance-seed/seed-1.6": _mi(262144, 0.25, 2.00),
    "openrouter/bytedance-seed/seed-1.6-flash": _mi(262144, 0.075, 0.30),
    "openrouter/bytedance-seed/seed-2.0-lite": _mi(262144, 0.25, 2.00),
    "openrouter/bytedance-seed/seed-2.0-mini": _mi(262144, 0.10, 0.40),
    "openrouter/bytedance/ui-tars-1.5-7b": _mi(128000, 0.10, 0.20, fc=False),
    "openrouter/cohere/command-a": _mi(256000, 2.50, 10.00, fc=False),
    "openrouter/cohere/command-r-08-2024": _mi(128000, 0.15, 0.60, fc=False),
    "openrouter/cohere/command-r-plus-08-2024": _mi(128000, 2.50, 10.00, fc=False, gen=False),
    "openrouter/cohere/command-r7b-12-2024": _mi(128000, 0.04, 0.15, fc=False),
    "openrouter/deepcogito/cogito-v2.1-671b": _mi(128000, 1.25, 1.25, fc=False),
    "openrouter/deepseek/deepseek-chat": _mi(163840, 0.32, 0.89),
    "openrouter/deepseek/deepseek-chat-v3-0324": _mi(163840, 0.20, 0.77, fc=False),
    "openrouter/deepseek/deepseek-chat-v3.1": _mi(32768, 0.15, 0.75, fc=False, gen=False),
    "openrouter/deepseek/deepseek-r1": _mi(64000, 0.70, 2.50),
    "openrouter/deepseek/deepseek-r1-0528": _mi(163840, 0.50, 2.15),
    "openrouter/deepseek/deepseek-r1-distill-llama-70b": _mi(131072, 0.70, 0.80),
    "openrouter/deepseek/deepseek-r1-distill-qwen-32b": _mi(32768, 0.29, 0.29),
    "openrouter/deepseek/deepseek-v3.1-terminus": _mi(163840, 0.21, 0.79, fc=False, gen=False),
    "openrouter/deepseek/deepseek-v3.2": _mi(131072, 0.252, 0.378),
    "openrouter/deepseek/deepseek-v3.2-exp": _mi(163840, 0.27, 0.41),
    "openrouter/deepseek/deepseek-v3.2-speciale": _mi(163840, 0.40, 1.20),
    "openrouter/deepseek/deepseek-v4-flash": _mi(1048576, 0.14, 0.28),  # NEW
    "openrouter/deepseek/deepseek-v4-pro": _mi(1048576, 0.435, 0.87, fc=False),  # NEW
    "openrouter/essentialai/rnj-1-instruct": _mi(32768, 0.15, 0.15, fc=False),
    "openrouter/google/gemini-2.0-flash-001": _mi(1000000, 0.10, 0.40),
    "openrouter/google/gemini-2.0-flash-lite-001": _mi(1048576, 0.075, 0.30),
    "openrouter/google/gemini-2.5-flash": _mi(1048576, 0.30, 2.50),
    "openrouter/google/gemini-2.5-flash-image": _mi(32768, 0.30, 2.50, fc=False, gen=False),
    "openrouter/google/gemini-2.5-flash-lite": _mi(1048576, 0.10, 0.40, fc=False),
    "openrouter/google/gemini-2.5-flash-lite-preview-09-2025": _mi(1048576, 0.10, 0.40, fc=False),
    "openrouter/google/gemini-2.5-pro": _mi(1048576, 1.25, 10.00),
    "openrouter/google/gemini-2.5-pro-preview": _mi(1048576, 1.25, 10.00),
    "openrouter/google/gemini-2.5-pro-preview-05-06": _mi(1048576, 1.25, 10.00, fc=False),
    "openrouter/google/gemini-3-flash-preview": _mi(1048576, 0.50, 3.00),
    "openrouter/google/gemini-3-pro-image-preview": _mi(65536, 2.00, 12.00, fc=False, gen=False),
    "openrouter/google/gemini-3.1-flash-image-preview": _mi(65536, 0.50, 3.00),
    "openrouter/google/gemini-3.1-flash-lite-preview": _mi(1048576, 0.25, 1.50),
    "openrouter/google/gemini-3.1-pro-preview": _mi(1048576, 2.00, 12.00),
    "openrouter/google/gemini-3.1-pro-preview-customtools": _mi(1048576, 2.00, 12.00),
    "openrouter/google/gemma-2-27b-it": _mi(8192, 0.65, 0.65, fc=False),
    "openrouter/google/gemma-3-12b-it": _mi(131072, 0.04, 0.13, fc=False),
    "openrouter/google/gemma-3-27b-it": _mi(131072, 0.08, 0.16, fc=False),
    "openrouter/google/gemma-3-4b-it": _mi(131072, 0.04, 0.08, fc=False),
    "openrouter/google/gemma-3n-e4b-it": _mi(32768, 0.06, 0.12, fc=False),
    "openrouter/google/gemma-4-26b-a4b-it": _mi(262144, 0.06, 0.33),
    "openrouter/google/gemma-4-31b-it": _mi(262144, 0.13, 0.38),
    "openrouter/google/lyria-3-clip-preview": _mi(1048576, 0.00, 0.00),
    "openrouter/google/lyria-3-pro-preview": _mi(1048576, 0.00, 0.00),
    "openrouter/gryphe/mythomax-l2-13b": _mi(4096, 0.06, 0.06, fc=False),
    "openrouter/ibm-granite/granite-4.0-h-micro": _mi(131000, 0.02, 0.11, fc=False),
    "openrouter/ibm-granite/granite-4.1-8b": _mi(131072, 0.05, 0.10),  # NEW
    "openrouter/inception/mercury-2": _mi(128000, 0.25, 0.75),
    "openrouter/inclusionai/ling-2.6-flash": _mi(262144, 0.08, 0.24),  # NEW
    "openrouter/inflection/inflection-3-pi": _mi(8000, 2.50, 10.00, fc=False),
    "openrouter/inflection/inflection-3-productivity": _mi(8000, 2.50, 10.00, fc=False),
    "openrouter/kwaipilot/kat-coder-pro-v2": _mi(256000, 0.30, 1.20),
    "openrouter/liquid/lfm-2-24b-a2b": _mi(32768, 0.03, 0.12),
    "openrouter/mancer/weaver": _mi(8000, 0.75, 1.00, fc=False),
    "openrouter/meta-llama/llama-3-70b-instruct": _mi(8192, 0.51, 0.74, fc=False),
    "openrouter/meta-llama/llama-3-8b-instruct": _mi(8192, 0.03, 0.04, fc=False),
    "openrouter/meta-llama/llama-3.1-70b-instruct": _mi(131072, 0.40, 0.40, fc=False),
    "openrouter/meta-llama/llama-3.1-8b-instruct": _mi(16384, 0.02, 0.05, fc=False),
    "openrouter/meta-llama/llama-3.2-11b-vision-instruct": _mi(131072, 0.245, 0.245, fc=False),
    "openrouter/meta-llama/llama-3.2-1b-instruct": _mi(60000, 0.03, 0.20, fc=False),
    "openrouter/meta-llama/llama-3.2-3b-instruct": _mi(80000, 0.051, 0.34, fc=False),
    "openrouter/meta-llama/llama-3.3-70b-instruct": _mi(131072, 0.10, 0.32, fc=False, gen=False),
    "openrouter/meta-llama/llama-4-maverick": _mi(1048576, 0.15, 0.60, fc=False),
    "openrouter/meta-llama/llama-4-scout": _mi(327680, 0.08, 0.30, fc=False),
    "openrouter/meta-llama/llama-guard-3-8b": _mi(131072, 0.48, 0.03, fc=False),
    "openrouter/meta-llama/llama-guard-4-12b": _mi(163840, 0.18, 0.18, fc=False),
    "openrouter/microsoft/phi-4": _mi(16384, 0.065, 0.14, fc=False),
    "openrouter/microsoft/wizardlm-2-8x22b": _mi(65535, 0.62, 0.62, fc=False),
    "openrouter/minimax/minimax-01": _mi(1000192, 0.20, 1.10, fc=False),
    "openrouter/minimax/minimax-m1": _mi(1000000, 0.40, 2.20, fc=False),
    "openrouter/minimax/minimax-m2": _mi(196608, 0.255, 1.00, fc=False),
    "openrouter/minimax/minimax-m2-her": _mi(65536, 0.30, 1.20, fc=False),
    "openrouter/minimax/minimax-m2.1": _mi(196608, 0.29, 0.95),
    "openrouter/minimax/minimax-m2.5": _mi(196608, 0.15, 1.15),
    "openrouter/minimax/minimax-m2.7": _mi(196608, 0.30, 1.20),
    "openrouter/mistralai/codestral-2508": _mi(256000, 0.30, 0.90, fc=False),
    "openrouter/mistralai/devstral-2512": _mi(262144, 0.40, 2.00, fc=False),
    "openrouter/mistralai/devstral-medium": _mi(131072, 0.40, 2.00, fc=False, gen=False),
    "openrouter/mistralai/devstral-small": _mi(131072, 0.10, 0.30, fc=False, gen=False),
    "openrouter/mistralai/ministral-14b-2512": _mi(262144, 0.20, 0.20, fc=False),
    "openrouter/mistralai/ministral-3b-2512": _mi(131072, 0.10, 0.10, fc=False),
    "openrouter/mistralai/ministral-8b-2512": _mi(262144, 0.15, 0.15, fc=False),
    "openrouter/mistralai/mistral-7b-instruct-v0.1": _mi(2824, 0.11, 0.19, fc=False),
    "openrouter/mistralai/mistral-large": _mi(128000, 2.00, 6.00, fc=False),
    "openrouter/mistralai/mistral-large-2407": _mi(131072, 2.00, 6.00),
    "openrouter/mistralai/mistral-large-2411": _mi(131072, 2.00, 6.00, fc=False),
    "openrouter/mistralai/mistral-large-2512": _mi(262144, 0.50, 1.50, fc=False),
    "openrouter/mistralai/mistral-medium-3": _mi(131072, 0.40, 2.00, fc=False),
    "openrouter/mistralai/mistral-medium-3.1": _mi(131072, 0.40, 2.00, fc=False),
    "openrouter/mistralai/mistral-nemo": _mi(131072, 0.02, 0.03, fc=False),
    "openrouter/mistralai/mistral-saba": _mi(32768, 0.20, 0.60, fc=False),
    "openrouter/mistralai/mistral-small-24b-instruct-2501": _mi(32768, 0.05, 0.08, fc=False),
    "openrouter/mistralai/mistral-small-2603": _mi(262144, 0.15, 0.60),
    "openrouter/mistralai/mistral-small-3.1-24b-instruct": _mi(128000, 0.35, 0.56, fc=False),
    "openrouter/mistralai/mistral-small-3.2-24b-instruct": _mi(128000, 0.075, 0.20, fc=False),
    "openrouter/mistralai/mixtral-8x22b-instruct": _mi(65536, 2.00, 6.00, fc=False),
    "openrouter/mistralai/mixtral-8x7b-instruct": _mi(32768, 0.54, 0.54, fc=False),
    "openrouter/mistralai/pixtral-large-2411": _mi(131072, 2.00, 6.00, fc=False),
    "openrouter/mistralai/voxtral-small-24b-2507": _mi(32000, 0.10, 0.30, fc=False),
    "openrouter/moonshotai/kimi-k2": _mi(131072, 0.57, 2.30, fc=False),
    "openrouter/moonshotai/kimi-k2-0905": _mi(262144, 0.40, 2.00, fc=False),
    "openrouter/moonshotai/kimi-k2-thinking": _mi(262144, 0.60, 2.50, fc=False),
    "openrouter/moonshotai/kimi-k2.5": _mi(262144, 0.44, 2.00),
    "openrouter/moonshotai/kimi-k2.6": _mi(262142, 0.74, 3.49),
    "openrouter/morph/morph-v3-fast": _mi(81920, 0.80, 1.20, fc=False),
    "openrouter/morph/morph-v3-large": _mi(262144, 0.90, 1.90, fc=False),
    "openrouter/nex-agi/deepseek-v3.1-nex-n1": _mi(131072, 0.135, 0.50, fc=False),
    "openrouter/nousresearch/hermes-2-pro-llama-3-8b": _mi(8192, 0.14, 0.14, fc=False),
    "openrouter/nousresearch/hermes-3-llama-3.1-405b": _mi(131072, 1.00, 1.00, fc=False),
    "openrouter/nousresearch/hermes-3-llama-3.1-70b": _mi(131072, 0.30, 0.30, fc=False),
    "openrouter/nousresearch/hermes-4-405b": _mi(131072, 1.00, 3.00, fc=False),
    "openrouter/nousresearch/hermes-4-70b": _mi(131072, 0.13, 0.40, fc=False),
    "openrouter/nvidia/llama-3.1-nemotron-70b-instruct": _mi(131072, 1.20, 1.20, fc=False),
    "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5": _mi(131072, 0.10, 0.40),
    "openrouter/nvidia/nemotron-3-nano-30b-a3b": _mi(262144, 0.05, 0.20, fc=False),
    "openrouter/nvidia/nemotron-3-super-120b-a12b": _mi(262144, 0.09, 0.45),
    "openrouter/nvidia/nemotron-nano-12b-v2-vl": _mi(131072, 0.20, 0.60, fc=False),
    "openrouter/nvidia/nemotron-nano-9b-v2": _mi(131072, 0.04, 0.16),
    "openrouter/openai/gpt-3.5-turbo": _mi(16385, 0.50, 1.50),
    "openrouter/openai/gpt-3.5-turbo-0613": _mi(4095, 1.00, 2.00),
    "openrouter/openai/gpt-3.5-turbo-16k": _mi(16385, 3.00, 4.00),
    "openrouter/openai/gpt-3.5-turbo-instruct": _mi(4095, 1.50, 2.00, fc=False),
    "openrouter/openai/gpt-4": _mi(8191, 30.00, 60.00),
    "openrouter/openai/gpt-4-0314": _mi(8191, 30.00, 60.00),
    "openrouter/openai/gpt-4-1106-preview": _mi(128000, 10.00, 30.00),
    "openrouter/openai/gpt-4-turbo": _mi(128000, 10.00, 30.00),
    "openrouter/openai/gpt-4-turbo-preview": _mi(128000, 10.00, 30.00),
    "openrouter/openai/gpt-4.1": _mi(1047576, 2.00, 8.00),
    "openrouter/openai/gpt-4.1-mini": _mi(1047576, 0.40, 1.60),
    "openrouter/openai/gpt-4.1-nano": _mi(1047576, 0.10, 0.40, fc=False),
    "openrouter/openai/gpt-4o": _mi(128000, 2.50, 10.00),
    "openrouter/openai/gpt-4o-2024-05-13": _mi(128000, 5.00, 15.00),
    "openrouter/openai/gpt-4o-2024-08-06": _mi(128000, 2.50, 10.00),
    "openrouter/openai/gpt-4o-2024-11-20": _mi(128000, 2.50, 10.00),
    "openrouter/openai/gpt-4o-audio-preview": _mi(128000, 2.50, 10.00, fc=False),
    "openrouter/openai/gpt-4o-mini": _mi(128000, 0.15, 0.60),
    "openrouter/openai/gpt-4o-mini-2024-07-18": _mi(128000, 0.15, 0.60),
    "openrouter/openai/gpt-4o-mini-search-preview": _mi(128000, 0.15, 0.60, fc=False),
    "openrouter/openai/gpt-4o-search-preview": _mi(128000, 2.50, 10.00, fc=False),
    "openrouter/openai/gpt-4o:extended": _mi(128000, 6.00, 18.00),
    "openrouter/openai/gpt-5": _mi(400000, 1.25, 10.00),
    "openrouter/openai/gpt-5-chat": _mi(128000, 1.25, 10.00, fc=False),
    "openrouter/openai/gpt-5-codex": _mi(400000, 1.25, 10.00),
    "openrouter/openai/gpt-5-image": _mi(400000, 10.00, 10.00, fc=False, gen=False),
    "openrouter/openai/gpt-5-image-mini": _mi(400000, 2.50, 2.00, fc=False, gen=False),
    "openrouter/openai/gpt-5-mini": _mi(400000, 0.25, 2.00),
    "openrouter/openai/gpt-5-nano": _mi(400000, 0.05, 0.40, fc=False),
    "openrouter/openai/gpt-5-pro": _mi(400000, 15.00, 120.00),
    "openrouter/openai/gpt-5.1": _mi(400000, 1.25, 10.00),
    "openrouter/openai/gpt-5.1-chat": _mi(128000, 1.25, 10.00),
    "openrouter/openai/gpt-5.1-codex": _mi(400000, 1.25, 10.00),
    "openrouter/openai/gpt-5.1-codex-max": _mi(400000, 1.25, 10.00),
    "openrouter/openai/gpt-5.1-codex-mini": _mi(400000, 0.25, 2.00),
    "openrouter/openai/gpt-5.2": _mi(400000, 1.75, 14.00),
    "openrouter/openai/gpt-5.2-chat": _mi(128000, 1.75, 14.00),
    "openrouter/openai/gpt-5.2-codex": _mi(400000, 1.75, 14.00),
    "openrouter/openai/gpt-5.2-pro": _mi(400000, 21.00, 168.00),
    "openrouter/openai/gpt-5.3-chat": _mi(128000, 1.75, 14.00),
    "openrouter/openai/gpt-5.3-codex": _mi(400000, 1.75, 14.00),
    "openrouter/openai/gpt-5.4": _mi(1050000, 2.50, 15.00),
    "openrouter/openai/gpt-5.4-image-2": _mi(272000, 8.00, 15.00, fc=False),
    "openrouter/openai/gpt-5.4-mini": _mi(400000, 0.75, 4.50),
    "openrouter/openai/gpt-5.4-nano": _mi(400000, 0.20, 1.25),
    "openrouter/openai/gpt-5.4-pro": _mi(1050000, 30.00, 180.00),
    "openrouter/openai/gpt-5.5": _mi(1050000, 5.00, 30.00),  # NEW
    "openrouter/openai/gpt-5.5-pro": _mi(1050000, 30.00, 180.00),  # NEW
    "openrouter/openai/gpt-audio": _mi(128000, 2.50, 10.00, fc=False),
    "openrouter/openai/gpt-audio-mini": _mi(128000, 0.60, 2.40, fc=False),
    "openrouter/openai/gpt-oss-120b": _mi(131072, 0.039, 0.18, fc=False),
    "openrouter/openai/gpt-oss-20b": _mi(131072, 0.03, 0.14, fc=False),
    "openrouter/openai/gpt-oss-safeguard-20b": _mi(131072, 0.075, 0.30, fc=False),
    "openrouter/openai/o1": _mi(200000, 15.00, 60.00, fc=False),
    "openrouter/openai/o1-pro": _mi(200000, 150.00, 600.00, fc=False),
    "openrouter/openai/o3": _mi(200000, 2.00, 8.00),
    "openrouter/openai/o3-deep-research": _mi(200000, 10.00, 40.00, fc=False),
    "openrouter/openai/o3-mini": _mi(200000, 1.10, 4.40, fc=False),
    "openrouter/openai/o3-mini-high": _mi(200000, 1.10, 4.40),
    "openrouter/openai/o3-pro": _mi(200000, 20.00, 80.00, fc=False),
    "openrouter/openai/o4-mini": _mi(200000, 1.10, 4.40),
    "openrouter/openai/o4-mini-deep-research": _mi(200000, 2.00, 8.00, fc=False),
    "openrouter/openai/o4-mini-high": _mi(200000, 1.10, 4.40),
    "openrouter/perplexity/sonar": _mi(127072, 1.00, 1.00, fc=False),
    "openrouter/perplexity/sonar-deep-research": _mi(128000, 2.00, 8.00, fc=False),
    "openrouter/perplexity/sonar-pro": _mi(200000, 3.00, 15.00, fc=False),
    "openrouter/perplexity/sonar-pro-search": _mi(200000, 3.00, 15.00, fc=False),
    "openrouter/perplexity/sonar-reasoning-pro": _mi(128000, 2.00, 8.00, fc=False),
    "openrouter/prime-intellect/intellect-3": _mi(131072, 0.20, 1.10, fc=False),
    "openrouter/qwen/qwen-2.5-72b-instruct": _mi(32768, 0.36, 0.40),
    "openrouter/qwen/qwen-2.5-7b-instruct": _mi(32768, 0.04, 0.10, fc=False),
    "openrouter/qwen/qwen-2.5-coder-32b-instruct": _mi(32768, 0.66, 1.00, fc=False),
    "openrouter/qwen/qwen-max": _mi(32768, 1.04, 4.16, fc=False),
    "openrouter/qwen/qwen-plus": _mi(1000000, 0.26, 0.78),
    "openrouter/qwen/qwen-plus-2025-07-28": _mi(1000000, 0.26, 0.78),
    "openrouter/qwen/qwen-plus-2025-07-28:thinking": _mi(1000000, 0.40, 4.00),
    "openrouter/qwen/qwen-turbo": _mi(131072, 0.033, 0.13),
    "openrouter/qwen/qwen-vl-max": _mi(131072, 0.52, 2.08),
    "openrouter/qwen/qwen-vl-plus": _mi(131072, 0.137, 0.41, fc=False),
    "openrouter/qwen/qwen2.5-vl-72b-instruct": _mi(32000, 0.25, 0.75, fc=False),
    "openrouter/qwen/qwen3-14b": _mi(40960, 0.06, 0.24, fc=False),
    "openrouter/qwen/qwen3-235b-a22b": _mi(131072, 0.455, 1.82, fc=False, gen=False),
    "openrouter/qwen/qwen3-235b-a22b-2507": _mi(262144, 0.07, 0.10, fc=False),
    "openrouter/qwen/qwen3-235b-a22b-thinking-2507": _mi(131072, 0.15, 1.495),
    "openrouter/qwen/qwen3-30b-a3b": _mi(40960, 0.08, 0.28, fc=False),
    "openrouter/qwen/qwen3-30b-a3b-instruct-2507": _mi(262144, 0.09, 0.30),
    "openrouter/qwen/qwen3-30b-a3b-thinking-2507": _mi(131072, 0.08, 0.40),
    "openrouter/qwen/qwen3-32b": _mi(40960, 0.08, 0.24, fc=False),
    "openrouter/qwen/qwen3-8b": _mi(40960, 0.05, 0.40, fc=False),
    "openrouter/qwen/qwen3-coder": _mi(262144, 0.22, 1.80),
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct": _mi(160000, 0.07, 0.27),
    "openrouter/qwen/qwen3-coder-flash": _mi(1000000, 0.195, 0.975),
    "openrouter/qwen/qwen3-coder-next": _mi(262144, 0.12, 0.80),
    "openrouter/qwen/qwen3-coder-plus": _mi(1000000, 0.65, 3.25, fc=False),
    "openrouter/qwen/qwen3-max": _mi(262144, 0.78, 3.90),
    "openrouter/qwen/qwen3-max-thinking": _mi(262144, 0.78, 3.90, fc=False),
    "openrouter/qwen/qwen3-next-80b-a3b-instruct": _mi(262144, 0.09, 1.10),
    "openrouter/qwen/qwen3-next-80b-a3b-thinking": _mi(131072, 0.098, 0.78),
    "openrouter/qwen/qwen3-vl-235b-a22b-instruct": _mi(262144, 0.20, 0.88),
    "openrouter/qwen/qwen3-vl-235b-a22b-thinking": _mi(131072, 0.26, 2.60, fc=False),
    "openrouter/qwen/qwen3-vl-30b-a3b-instruct": _mi(131072, 0.13, 0.52),
    "openrouter/qwen/qwen3-vl-30b-a3b-thinking": _mi(131072, 0.13, 1.56),
    "openrouter/qwen/qwen3-vl-32b-instruct": _mi(131072, 0.104, 0.416),
    "openrouter/qwen/qwen3-vl-8b-instruct": _mi(131072, 0.08, 0.50),
    "openrouter/qwen/qwen3-vl-8b-thinking": _mi(131072, 0.117, 1.365),
    "openrouter/qwen/qwen3.5-122b-a10b": _mi(262144, 0.26, 2.08),
    "openrouter/qwen/qwen3.5-27b": _mi(262144, 0.195, 1.56),
    "openrouter/qwen/qwen3.5-35b-a3b": _mi(262144, 0.163, 1.30),
    "openrouter/qwen/qwen3.5-397b-a17b": _mi(262144, 0.39, 2.34),
    "openrouter/qwen/qwen3.5-9b": _mi(262144, 0.10, 0.15),
    "openrouter/qwen/qwen3.5-flash-02-23": _mi(1000000, 0.065, 0.26),
    "openrouter/qwen/qwen3.5-plus-02-15": _mi(1000000, 0.26, 1.56),
    "openrouter/qwen/qwen3.5-plus-20260420": _mi(1000000, 0.40, 2.40),  # NEW
    "openrouter/qwen/qwen3.6-27b": _mi(256000, 0.325, 3.25),  # NEW
    "openrouter/qwen/qwen3.6-35b-a3b": _mi(262144, 0.161, 0.965, fc=False),  # NEW
    "openrouter/qwen/qwen3.6-flash": _mi(1000000, 0.25, 1.50),  # NEW
    "openrouter/qwen/qwen3.6-max-preview": _mi(262144, 1.04, 6.24),  # NEW
    "openrouter/qwen/qwen3.6-plus": _mi(1000000, 0.325, 1.95),
    "openrouter/rekaai/reka-edge": _mi(16384, 0.10, 0.10),
    "openrouter/rekaai/reka-flash-3": _mi(65536, 0.10, 0.20),
    "openrouter/relace/relace-apply-3": _mi(256000, 0.85, 1.25, fc=False),
    "openrouter/relace/relace-search": _mi(256000, 1.00, 3.00, fc=False),
    "openrouter/sao10k/l3-euryale-70b": _mi(8192, 1.48, 1.48),
    "openrouter/sao10k/l3-lunaris-8b": _mi(8192, 0.04, 0.05, fc=False),
    "openrouter/sao10k/l3.1-70b-hanami-x1": _mi(16000, 3.00, 3.00, fc=False),
    "openrouter/sao10k/l3.1-euryale-70b": _mi(131072, 0.85, 0.85),
    "openrouter/sao10k/l3.3-euryale-70b": _mi(131072, 0.65, 0.75, fc=False),
    "openrouter/stepfun/step-3.5-flash": _mi(262144, 0.10, 0.30),
    "openrouter/switchpoint/router": _mi(131072, 0.85, 3.40, fc=False),
    "openrouter/tencent/hunyuan-a13b-instruct": _mi(131072, 0.14, 0.57, fc=False),
    "openrouter/thedrummer/cydonia-24b-v4.1": _mi(131072, 0.30, 0.50, fc=False),
    "openrouter/thedrummer/rocinante-12b": _mi(32768, 0.17, 0.43),
    "openrouter/thedrummer/skyfall-36b-v2": _mi(32768, 0.55, 0.80, fc=False),
    "openrouter/thedrummer/unslopnemo-12b": _mi(32768, 0.40, 0.40),
    "openrouter/tngtech/deepseek-r1t2-chimera": _mi(163840, 0.30, 1.10),
    "openrouter/undi95/remm-slerp-l2-13b": _mi(6144, 0.45, 0.65, fc=False),
    "openrouter/upstage/solar-pro-3": _mi(128000, 0.15, 0.60, fc=False),
    "openrouter/writer/palmyra-x5": _mi(1040000, 0.60, 6.00, fc=False),
    "openrouter/x-ai/grok-3": _mi(131072, 3.00, 15.00),
    "openrouter/x-ai/grok-3-beta": _mi(131072, 3.00, 15.00),
    "openrouter/x-ai/grok-3-mini": _mi(131072, 0.30, 0.50),
    "openrouter/x-ai/grok-3-mini-beta": _mi(131072, 0.30, 0.50),
    "openrouter/x-ai/grok-4": _mi(256000, 3.00, 15.00),
    "openrouter/x-ai/grok-4-fast": _mi(2000000, 0.20, 0.50),
    "openrouter/x-ai/grok-4.1-fast": _mi(2000000, 0.20, 0.50),
    "openrouter/x-ai/grok-4.20": _mi(2000000, 1.25, 2.50),
    "openrouter/x-ai/grok-4.20-multi-agent": _mi(2000000, 2.00, 6.00),
    "openrouter/x-ai/grok-4.3": _mi(1000000, 1.25, 2.50),  # NEW
    "openrouter/x-ai/grok-code-fast-1": _mi(256000, 0.20, 1.50),
    "openrouter/xiaomi/mimo-v2-flash": _mi(262144, 0.09, 0.29, fc=False),
    "openrouter/xiaomi/mimo-v2-omni": _mi(262144, 0.40, 2.00),
    "openrouter/xiaomi/mimo-v2-pro": _mi(1048576, 1.00, 3.00),
    "openrouter/xiaomi/mimo-v2.5": _mi(1048576, 0.40, 2.00),  # NEW
    "openrouter/xiaomi/mimo-v2.5-pro": _mi(1048576, 1.00, 3.00),  # NEW
    "openrouter/z-ai/glm-4-32b": _mi(128000, 0.10, 0.10),
    "openrouter/z-ai/glm-4.5": _mi(131072, 0.60, 2.20),
    "openrouter/z-ai/glm-4.5-air": _mi(131072, 0.13, 0.85),
    "openrouter/z-ai/glm-4.5v": _mi(65536, 0.60, 1.80, fc=False),
    "openrouter/z-ai/glm-4.6": _mi(204800, 0.39, 1.90, fc=False),
    "openrouter/z-ai/glm-4.6v": _mi(131072, 0.30, 0.90),
    "openrouter/z-ai/glm-4.7": _mi(202752, 0.38, 1.74),
    "openrouter/z-ai/glm-4.7-flash": _mi(202752, 0.06, 0.40),
    "openrouter/z-ai/glm-5": _mi(202752, 0.60, 2.08),
    "openrouter/z-ai/glm-5-turbo": _mi(202752, 1.20, 4.00),
    "openrouter/z-ai/glm-5.1": _mi(202752, 1.05, 3.50),
    "openrouter/z-ai/glm-5v-turbo": _mi(202752, 1.20, 4.00),
    "openrouter/~anthropic/claude-haiku-latest": _mi(200000, 1.00, 5.00),  # NEW
    "openrouter/~anthropic/claude-opus-latest": _mi(1000000, 5.00, 25.00),
    "openrouter/~anthropic/claude-sonnet-latest": _mi(1000000, 3.00, 15.00),  # NEW
    "openrouter/~google/gemini-flash-latest": _mi(1048576, 0.50, 3.00),  # NEW
    "openrouter/~google/gemini-pro-latest": _mi(1048576, 2.00, 12.00),  # NEW
    "openrouter/~moonshotai/kimi-latest": _mi(262142, 0.74, 3.49),  # NEW
    "openrouter/~openai/gpt-latest": _mi(1050000, 5.00, 30.00),  # NEW
    "openrouter/~openai/gpt-mini-latest": _mi(400000, 0.75, 4.50),  # NEW
    "Qwen/Qwen2-1.5B-Instruct": _mi(32768, 0.02, 0.02),  # NEW
    "Qwen/Qwen2-VL-72B-Instruct": _mi(32768, 1.20, 1.20),
    "Qwen/Qwen2.5-14B-Instruct": _mi(32768, 0.80, 0.80),
    "Qwen/Qwen2.5-72B-Instruct": _mi(32768, 1.20, 1.20),
    "Qwen/Qwen2.5-72B-Instruct-Turbo": _mi(131072, 1.20, 1.20),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": _mi(32768, 0.30, 0.30, fc=False),
    "Qwen/Qwen2.5-Coder-32B-Instruct": _mi(16384, 0.80, 0.80),  # NEW
    "Qwen/Qwen2.5-VL-72B-Instruct": _mi(32768, 1.95, 8.00),
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": _mi(262144, 0.20, 0.60, fc=False),
    "Qwen/Qwen3-235B-A22B-Thinking-2507": _mi(262144, 0.65, 3.00, fc=False),
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": _mi(262144, 2.00, 2.00),
    "Qwen/Qwen3-Coder-Next-FP8": _mi(262144, 0.50, 1.20),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": _mi(262144, 0.15, 1.50),
    "Qwen/Qwen3-Next-80B-A3B-Thinking": _mi(262144, 0.15, 1.50),
    "Qwen/Qwen3-VL-32B-Instruct": _mi(262144, 0.50, 1.50),
    "Qwen/Qwen3-VL-8B-Instruct": _mi(262144, 0.18, 0.68, fc=False),
    "Qwen/Qwen3.5-397B-A17B": _mi(262144, 0.60, 3.60, fc=False),
    "Qwen/Qwen3.5-9B": _mi(262144, 0.10, 0.15),
    "Qwen/Qwen3.6-Plus": _mi(1000000, 0.50, 3.00),  # NEW
    "Qwen/QwQ-32B": _mi(131072, 1.20, 1.20),  # NEW
    "text-embedding-3-large": _emb(8191, 0.13),
    "text-embedding-3-small": _emb(8191, 0.02),
    "text-embedding-ada-002": _emb(8191, 0.10),
    "zai-org/GLM-4.5-Air-FP8": _mi(131072, 0.20, 1.10),
    "zai-org/GLM-4.6": _mi(202752, 0.60, 2.20),
    "zai-org/GLM-4.7": _mi(202752, 0.45, 2.00),
    "zai-org/GLM-5": _mi(202752, 1.00, 3.20),
    "zai-org/GLM-5.1": _mi(202752, 1.40, 4.40),
}

for _name, _info in MODEL_INFO.items():
    if _info.cache_read_price_per_1M is not None:  # pragma: no cover – no models set explicit cr
        continue
    if _name.startswith(("claude-", "openrouter/anthropic/")):
        _info.cache_read_price_per_1M = _info.input_price_per_1M * 0.1
        _info.cache_write_price_per_1M = _info.input_price_per_1M * 1.25
    elif (
        _name.startswith(_OPENAI_PREFIXES)
        and not _name.startswith(("text-embedding", "openai/"))
        and _info.is_generation_supported
    ):
        _info.cache_read_price_per_1M = _info.input_price_per_1M * 0.5

FLAKY_MODELS: dict[str, str] = {
    "openrouter/baidu/ernie-4.5-21b-a3b": "Ignores function calling tools",
}


def is_model_flaky(model_name: str) -> bool:
    """Check if a model is known to be flaky.

    Args:
        model_name: The name of the model to check.

    Returns:
        bool: True if the model is known to have reliability issues.
    """
    return model_name in FLAKY_MODELS


def get_flaky_reason(model_name: str) -> str:
    """Get the reason why a model is flaky.

    Args:
        model_name: The name of the model to check.

    Returns:
        str: The reason for flakiness, or empty string if not flaky.
    """
    return FLAKY_MODELS.get(model_name, "")


def _strip_provider_prefix(model_name: str) -> str:
    """Strip harbor-style provider prefixes that duplicate KISS's own routing.

    Harbor (and other frameworks) pass model names as ``provider/model``
    (e.g. ``openai/gpt-5.4``, ``anthropic/claude-opus-4-6``,
    ``google/gemini-2.5-pro``).  KISS already routes by the model name
    itself (``gpt-*`` → OpenAI, ``claude-*`` → Anthropic, etc.), so the
    provider prefix is redundant and must be stripped.

    Prefixes that KISS uses for its own routing (``openrouter/``,
    ``openai/gpt-oss``, ``meta-llama/``, etc.) are NOT stripped — they
    are already handled by the ``model()`` dispatch chain.

    Args:
        model_name: Model name, possibly with a ``provider/`` prefix.

    Returns:
        The model name with redundant provider prefix stripped.
    """
    strip_prefixes = ("openai/", "anthropic/", "google/")
    for prefix in strip_prefixes:
        if model_name.startswith(prefix):
            bare = model_name[len(prefix):]
            if bare.startswith(_OPENAI_PREFIXES) and not bare.startswith("gpt-oss"):
                return bare
            if bare.startswith("claude-"):
                return bare
            if bare.startswith("gemini-"):
                return bare
    return model_name


def model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
    token_callback: TokenCallback | None = None,
    thinking_callback: ThinkingCallback | None = None,
) -> Model:
    """Get a model instance based on model name prefix.

    Args:
        model_name: The name of the model (with provider prefix if applicable).
            Accepts harbor-style ``provider/model`` names (e.g.
            ``openai/gpt-5.4``, ``anthropic/claude-opus-4-6``) — the
            redundant provider prefix is stripped automatically.
        model_config: Optional dictionary of model configuration parameters.
            If it contains "base_url", routing is bypassed and an OpenAICompatibleModel
            is built with that base_url and optional "api_key".
        token_callback: Optional callback invoked with each streamed text token.
        thinking_callback: Optional callback invoked with ``True`` when a
            thinking block starts and ``False`` when it ends.

    Returns:
        Model: An appropriate Model instance for the specified model.

    Raises:
        KISSError: If the model name is not recognized.
    """
    model_name = _strip_provider_prefix(model_name)
    if model_config and "base_url" in model_config:
        base_url = model_config["base_url"]
        api_key = model_config.get("api_key", "")
        filtered = {k: v for k, v in model_config.items() if k not in ("base_url", "api_key")}
        return _openai_compatible(
            model_name,
            base_url,
            api_key,
            filtered or None,
            token_callback,
            thinking_callback,
        )
    keys = config_module.DEFAULT_CONFIG
    if model_name.startswith("openrouter/"):
        return _openai_compatible(
            model_name,
            "https://openrouter.ai/api/v1",
            keys.OPENROUTER_API_KEY,
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name == "text-embedding-004":
        from kiss.core.models import GeminiModel

        if GeminiModel is None:  # pragma: no cover – google-genai always installed
            raise KISSError(
                "Google GenAI SDK not installed. Install 'google-genai' to use Gemini models."
            )
        return GeminiModel(  # type: ignore[no-any-return]
            model_name=model_name,
            api_key=keys.GEMINI_API_KEY,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    if model_name.startswith(_OPENAI_PREFIXES) and not model_name.startswith("openai/gpt-oss"):
        return _openai_compatible(
            model_name,
            "https://api.openai.com/v1",
            keys.OPENAI_API_KEY,
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name.startswith(_TOGETHER_PREFIXES):
        return _openai_compatible(
            model_name,
            "https://api.together.xyz/v1",
            keys.TOGETHER_API_KEY,
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name.startswith("claude-"):
        from kiss.core.models import AnthropicModel

        if AnthropicModel is None:  # pragma: no cover – anthropic always installed
            raise KISSError(
                "Anthropic SDK not installed. Install 'anthropic' to use Claude models."
            )
        return AnthropicModel(  # type: ignore[no-any-return]
            model_name=model_name,
            api_key=keys.ANTHROPIC_API_KEY,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    if model_name.startswith("gemini-"):
        from kiss.core.models import GeminiModel

        if GeminiModel is None:  # pragma: no cover – google-genai always installed
            raise KISSError(
                "Google GenAI SDK not installed. Install 'google-genai' to use Gemini models."
            )
        return GeminiModel(  # type: ignore[no-any-return]
            model_name=model_name,
            api_key=keys.GEMINI_API_KEY,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    if model_name.startswith("minimax-"):
        return _openai_compatible(
            model_name,
            "https://api.minimax.chat/v1",
            keys.MINIMAX_API_KEY,
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name.startswith("cc/"):
        from kiss.core.models import ClaudeCodeModel

        if ClaudeCodeModel is None:  # pragma: no cover – always available
            raise KISSError("ClaudeCodeModel could not be loaded.")
        return ClaudeCodeModel(  # type: ignore[no-any-return]
            model_name=model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    raise KISSError(f"Unknown model name: {model_name}")


def get_available_models() -> list[str]:
    """Return model names for which an API key is configured and generation is supported.

    Returns:
        list[str]: Sorted list of model name strings that have a configured API key
            and support text generation.
    """
    keys = config_module.DEFAULT_CONFIG
    prefix_to_key = {
        "openrouter/": keys.OPENROUTER_API_KEY,
        "claude-": keys.ANTHROPIC_API_KEY,
        "gemini-": keys.GEMINI_API_KEY,
        "minimax-": keys.MINIMAX_API_KEY,
    }
    import shutil

    has_claude_cli = shutil.which("claude") is not None
    result = []
    for name, info in MODEL_INFO.items():
        if not info.is_generation_supported:
            continue
        if name.startswith("cc/"):
            if has_claude_cli:
                result.append(name)
            continue
        api_key = ""
        for prefix, key in prefix_to_key.items():
            if name.startswith(prefix):
                api_key = key
                break
        if not api_key:
            if name == "text-embedding-004":  # pragma: no cover – embedding model filtered above
                api_key = keys.GEMINI_API_KEY
            elif name.startswith(_OPENAI_PREFIXES) and not name.startswith("openai/gpt-oss"):
                api_key = keys.OPENAI_API_KEY
            elif name.startswith(_TOGETHER_PREFIXES):
                api_key = keys.TOGETHER_API_KEY
        if api_key:
            result.append(name)
    return sorted(result)


def get_fast_model() -> str:
    """Return a cheap/fast model based on which API keys are available.

    Priority: Anthropic → OpenAI → Gemini → OpenRouter → Together → Claude Code CLI.

    Returns:
        A fast model name for the first available provider.
    """
    import shutil

    keys = config_module.DEFAULT_CONFIG
    if keys.ANTHROPIC_API_KEY:
        return "claude-haiku-4-5"
    if keys.OPENAI_API_KEY:
        return "gpt-4o"
    if keys.GEMINI_API_KEY:
        return "gemini-2.0-flash"
    if keys.OPENROUTER_API_KEY:
        return "openrouter/anthropic/claude-haiku-4.5"
    if keys.TOGETHER_API_KEY:
        return "deepseek-ai/DeepSeek-R1-0528"
    if shutil.which("claude") is not None:
        return "cc/haiku"
    return "No model"


def get_default_model() -> str:
    """Return the best default model based on which API keys are configured.

    Priority order: Anthropic > OpenAI > Gemini > OpenRouter > Together AI > Claude Code CLI.
    Falls back to ``"claude-opus-4-6"`` if no keys are set.
    """
    import shutil

    keys = config_module.DEFAULT_CONFIG
    if keys.ANTHROPIC_API_KEY:
        return "claude-opus-4-7"
    if keys.OPENAI_API_KEY:
        return "gpt-5.5"
    if keys.GEMINI_API_KEY:
        return "gemini-3.1-pro-preview"
    if keys.OPENROUTER_API_KEY:
        return "openrouter/anthropic/claude-opus-4.7"
    if keys.TOGETHER_API_KEY:
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    if shutil.which("claude") is not None:
        return "cc/opus"
    return "No model"


def get_most_expensive_model(fc_only: bool = True) -> str:
    best_name, best_price = "", -1.0
    for name in get_available_models():
        info = MODEL_INFO[name]
        if fc_only and not info.is_function_calling_supported:
            continue
        price = info.input_price_per_1M + info.output_price_per_1M
        if price > best_price:
            best_price = price
            best_name = name
    return best_name


def calculate_cost(
    model_name: str,
    num_input_tokens: int,
    num_output_tokens: int,
    num_cache_read_tokens: int = 0,
    num_cache_write_tokens: int = 0,
) -> float:
    """Calculates the cost in USD for the given token counts.

    Args:
        model_name: Name of the model (with or without provider prefix).
        num_input_tokens: Number of non-cached input tokens.
        num_output_tokens: Number of output tokens.
        num_cache_read_tokens: Number of tokens read from cache.
        num_cache_write_tokens: Number of tokens written to cache.

    Returns:
        float: Cost in USD, or 0.0 if pricing is not available for the model.
    """
    info = MODEL_INFO.get(model_name)
    info = MODEL_INFO.get(model_name) or MODEL_INFO.get(_strip_provider_prefix(model_name))
    if info is None:
        return 0.0
    cr_price = (
        info.cache_read_price_per_1M
        if info.cache_read_price_per_1M is not None
        else info.input_price_per_1M
    )
    cw_price = (
        info.cache_write_price_per_1M
        if info.cache_write_price_per_1M is not None
        else info.input_price_per_1M
    )
    return (
        num_input_tokens * info.input_price_per_1M
        + num_output_tokens * info.output_price_per_1M
        + num_cache_read_tokens * cr_price
        + num_cache_write_tokens * cw_price
    ) / 1_000_000


def get_max_context_length(model_name: str) -> int:
    """Returns the maximum context length supported by the model.

    Args:
        model_name: Name of the model (with or without provider prefix).
    Returns:
        int: Maximum context length in tokens.
    """
    stripped = _strip_provider_prefix(model_name)
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name].context_length
    if stripped in MODEL_INFO:
        return MODEL_INFO[stripped].context_length
    raise KeyError(f"Model '{model_name}' not found in MODEL_INFO")
