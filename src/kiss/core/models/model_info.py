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
from kiss.core.models.model import Model, TokenCallback

try:
    from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
except ImportError:
    OpenAICompatibleModel = None  # type: ignore[assignment,misc]

try:
    from kiss.core.models.anthropic_model import AnthropicModel
except ImportError:
    AnthropicModel = None  # type: ignore[assignment,misc]

try:
    from kiss.core.models.gemini_model import GeminiModel
except ImportError:
    GeminiModel = None  # type: ignore[assignment,misc]


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
    "meta-llama/", "Qwen/", "MiniMaxAI/", "mistralai/", "deepseek-ai/", "deepcogito/",
    "google/gemma", "moonshotai/", "nvidia/", "zai-org/", "openai/gpt-oss",
    "arcee-ai/", "essentialai/", "BAAI/", "intfloat/",
)


def _openai_compatible(
    model_name: str,
    base_url: str,
    api_key: str,
    model_config: dict[str, Any] | None,
    token_callback: TokenCallback | None,
) -> Model:
    if OpenAICompatibleModel is None:
        raise KISSError("OpenAI SDK not installed. Install 'openai' to use this model.")
    return OpenAICompatibleModel(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        model_config=model_config,
        token_callback=token_callback,
    )


MODEL_INFO: dict[str, ModelInfo] = {
    "arcee-ai/trinity-mini": _mi(128000, 0.04, 0.15, fc=False),
    "BAAI/bge-base-en-v1.5": _emb(512, 0.01),  # 768 dimensions
    "claude-3-5-haiku": _mi(200000, 0.80, 4.00),  # Alias for claude-3-5-haiku-20241022
    "claude-3-5-haiku-20241022": _mi(200000, 0.80, 4.00),  # Deprecated, retiring Feb 19, 2026
    "claude-3-haiku-20240307": _mi(200000, 0.25, 1.25),
    "claude-haiku-4-5": _mi(200000, 1.00, 5.00),
    "claude-haiku-4-5-20251001": _mi(200000, 1.00, 5.00),  # Snapshot version
    "claude-opus-4": _mi(200000, 15.00, 75.00),
    "claude-opus-4-1": _mi(200000, 15.00, 75.00),
    "claude-opus-4-1-20250805": _mi(200000, 15.00, 75.00),  # Snapshot version
    "claude-opus-4-20250514": _mi(200000, 15.00, 75.00),  # Snapshot version
    "claude-opus-4-5": _mi(200000, 5.00, 25.00),  # 1M beta via header
    "claude-opus-4-5-20251101": _mi(200000, 5.00, 25.00),  # Snapshot version
    "claude-opus-4-6": _mi(200000, 5.00, 25.00),  # 1M beta via header
    "claude-sonnet-4": _mi(200000, 3.00, 15.00),  # 1M beta via header
    "claude-sonnet-4-20250514": _mi(200000, 3.00, 15.00),  # Snapshot version
    "claude-sonnet-4-5": _mi(200000, 3.00, 15.00),  # 1M beta via header
    "claude-sonnet-4-5-20250929": _mi(200000, 3.00, 15.00),  # Snapshot version
    "claude-sonnet-4-6": _mi(200000, 3.00, 15.00),  # 1M beta via header
    "codex-mini-latest": _mi(200000, 1.50, 6.00),
    "computer-use-preview": _mi(128000, 3.00, 12.00),
    "deepcogito/cogito-v2-1-671b": _mi(163840, 1.25, 1.25, fc=False),
    "deepseek-ai/DeepSeek-R1": _mi(163840, 3.00, 7.00, fc=False),  # Serves R1-0528
    "deepseek-ai/DeepSeek-V3.1": _mi(131072, 0.60, 1.70, fc=False),
    "essentialai/rnj-1-instruct": _mi(32768, 0.15, 0.15, fc=False),
    "gemini-1.5-flash": _mi(1048576, 0.075, 0.30),
    "gemini-1.5-pro": _mi(2097152, 1.25, 5.00),  # 2M context window
    "gemini-2.0-flash": _mi(1048576, 0.10, 0.40),
    "gemini-2.0-flash-lite": _mi(1048576, 0.075, 0.30),
    "gemini-2.5-flash": _mi(1048576, 0.30, 2.50),
    "gemini-2.5-flash-lite": _mi(1048576, 0.10, 0.40, fc=False),  # Poor tool use
    "gemini-2.5-flash-lite-preview-09-2025": _mi(1048576, 0.10, 0.40, fc=False),
    "gemini-2.5-flash-preview-09-2025": _mi(1048576, 0.30, 2.50),
    "gemini-2.5-pro": _mi(1048576, 1.25, 10.00),
    "gemini-3-flash-preview": _mi(1048576, 0.50, 3.00),
    "gemini-3-pro-preview": _mi(1048576, 2.00, 12.00),
    "gemini-3.1-pro-preview": _mi(1048576, 2.50, 15.00),  # 1M context, 64k output
    "gemini-embedding-001": _emb(2048, 0.15),  # Newest embedding model
    "google/gemma-3n-E4B-it": _mi(32768, 0.02, 0.04, fc=False),
    "gpt-4": _mi(8192, 30.00, 60.00),
    "gpt-4-turbo": _mi(128000, 10.00, 30.00),
    "gpt-4.1": _mi(128000, 2.00, 8.00),
    "gpt-4.1-mini": _mi(128000, 0.40, 1.60),
    "gpt-4.1-nano": _mi(128000, 0.10, 0.40, fc=False),
    "gpt-4.5-preview": _mi(128000, 2.50, 10.00),  # Deprecated
    "gpt-4o": _mi(128000, 2.50, 10.00),
    "gpt-4o-2024-05-13": _mi(128000, 5.00, 15.00),
    "gpt-4o-mini": _mi(128000, 0.15, 0.60),
    "gpt-4o-mini-audio-preview": _mi(128000, 0.15, 0.60, fc=False),
    "gpt-4o-mini-realtime-preview": _mi(128000, 0.60, 2.40, fc=False),
    "gpt-4o-realtime-preview": _mi(128000, 5.00, 20.00, fc=False),
    "gpt-4o-transcribe-diarize": _mi(128000, 2.50, 10.00, fc=False),
    "gpt-5": _mi(400000, 1.25, 10.00),
    "gpt-5-chat-latest": _mi(400000, 1.25, 10.00),
    "gpt-5-codex": _mi(400000, 1.25, 10.00),
    "gpt-5-mini": _mi(400000, 0.25, 2.00),
    "gpt-5-nano": _mi(400000, 0.05, 0.40, fc=False),
    "gpt-5-pro": _mi(400000, 15.00, 120.00),
    "gpt-5-search-api": _mi(400000, 1.25, 10.00),
    "gpt-5.1": _mi(400000, 1.25, 10.00),
    "gpt-5.1-chat-latest": _mi(400000, 1.25, 10.00),
    "gpt-5.1-codex": _mi(400000, 1.25, 10.00),
    "gpt-5.1-codex-max": _mi(400000, 1.25, 10.00),
    "gpt-5.1-codex-mini": _mi(400000, 0.25, 2.00),
    "gpt-5.2": _mi(400000, 1.75, 14.00),
    "gpt-5.2-chat-latest": _mi(400000, 1.75, 14.00),
    "gpt-5.2-codex": _mi(400000, 1.75, 14.00),
    "gpt-5.2-pro": _mi(400000, 21.00, 168.00),
    "gpt-audio": _mi(128000, 2.50, 10.00, fc=False),
    "gpt-audio-mini": _mi(128000, 0.60, 2.40, fc=False),
    "gpt-realtime": _mi(128000, 4.00, 16.00, fc=False),
    "gpt-realtime-mini": _mi(128000, 0.60, 2.40, fc=False),
    "intfloat/multilingual-e5-large-instruct": _emb(514, 0.02),  # 1024 dimensions
    "meta-llama/Llama-3-70b-chat-hf": _mi(8192, 0.88, 0.88, fc=False),
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": _mi(131072, 0.06, 0.06, fc=False),  # Dep 2026-03
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": _mi(131072, 0.88, 0.88, fc=False),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": _mi(1048576, 0.27, 0.85),
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite": _mi(8192, 0.10, 0.10, fc=False),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": _mi(131072, 0.88, 0.88),  # NEW
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": _mi(131072, 0.18, 0.18, fc=False),  # Dep 2026-03
    "minimax-m2.5": _mi(1000000, 0.15, 1.20),  # Standard: 50 TPS
    "minimax-m2.5-lightning": _mi(1000000, 0.30, 2.40),  # Lightning: 100 TPS
    "MiniMaxAI/MiniMax-M2.5": _mi(196608, 0.30, 1.20),
    "mistralai/Ministral-3-14B-Instruct-2512": _mi(262144, 0.20, 0.20, fc=False),
    "mistralai/Mistral-7B-Instruct-v0.2": _mi(32768, 0.20, 0.20, fc=False),
    "mistralai/Mistral-7B-Instruct-v0.3": _mi(32768, 0.20, 0.20, fc=False),  # NEW
    "mistralai/Mistral-Small-24B-Instruct-2501": _mi(32768, 0.10, 0.30, fc=False),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": _mi(32768, 0.60, 0.60, fc=False),  # NEW
    "moonshotai/Kimi-K2-Instruct": _mi(128000, 1.00, 3.00, fc=False),
    "moonshotai/Kimi-K2-Instruct-0905": _mi(262144, 1.00, 3.00, fc=False),  # Dep 2026-03
    "moonshotai/Kimi-K2-Thinking": _mi(262144, 1.20, 4.00, fc=False),  # Dep 2026-03
    "moonshotai/Kimi-K2.5": _mi(262144, 0.50, 2.80),
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": _mi(131072, 0.06, 0.25),  # NEW
    "o1": _mi(200000, 15.00, 60.00, fc=False),  # SLOW: reasoning model
    "o1-mini": _mi(128000, 1.10, 4.40, fc=False),
    "o1-pro": _mi(200000, 150.00, 600.00, fc=False),
    "o3": _mi(200000, 2.00, 8.00),
    "o3-deep-research": _mi(200000, 10.00, 40.00, fc=False),
    "o3-mini": _mi(200000, 1.10, 4.40, fc=False),
    "o3-mini-high": _mi(200000, 1.10, 4.40),
    "o3-pro": _mi(200000, 20.00, 80.00, fc=False),
    "o4-mini": _mi(200000, 1.10, 4.40),
    "o4-mini-deep-research": _mi(200000, 2.00, 8.00, fc=False),
    "o4-mini-high": _mi(200000, 1.10, 4.40),
    "openai/gpt-oss-120b": _mi(131072, 0.15, 0.60, fc=False),
    "openai/gpt-oss-20b": _mi(131072, 0.05, 0.20, fc=False),
    "openrouter/ai21/jamba-large-1.7": _mi(256000, 2.00, 8.00, fc=False),
    "openrouter/ai21/jamba-mini-1.7": _mi(256000, 0.20, 0.40, fc=False),
    "openrouter/aion-labs/aion-1.0": _mi(131072, 4.00, 8.00, fc=False),
    "openrouter/aion-labs/aion-1.0-mini": _mi(131072, 0.70, 1.40, fc=False),
    "openrouter/aion-labs/aion-rp-llama-3.1-8b": _mi(32768, 0.80, 1.60, fc=False),
    "openrouter/alfredpros/codellama-7b-instruct-solidity": _mi(4096, 0.80, 1.20, fc=False),
    "openrouter/alibaba/tongyi-deepresearch-30b-a3b": _mi(131072, 0.09, 0.45, fc=False),
    "openrouter/allenai/molmo-2-8b": _mi(36864, 0.20, 0.20, fc=False),  # Vision model
    "openrouter/allenai/olmo-2-0325-32b-instruct": _mi(128000, 0.05, 0.20, fc=False),
    "openrouter/allenai/olmo-3-32b-think": _mi(65536, 0.15, 0.50, fc=False),
    "openrouter/allenai/olmo-3-7b-instruct": _mi(65536, 0.10, 0.20, fc=False),
    "openrouter/allenai/olmo-3-7b-think": _mi(65536, 0.12, 0.20, fc=False),
    "openrouter/allenai/olmo-3.1-32b-instruct": _mi(65536, 0.20, 0.60, fc=False),
    "openrouter/allenai/olmo-3.1-32b-think": _mi(65536, 0.15, 0.50, fc=False),
    "openrouter/alpindale/goliath-120b": _mi(6144, 3.75, 7.50, fc=False),
    "openrouter/amazon/nova-2-lite-v1": _mi(1000000, 0.30, 2.50, fc=False),  # Unreliable FC
    "openrouter/amazon/nova-lite-v1": _mi(300000, 0.06, 0.24),
    "openrouter/amazon/nova-micro-v1": _mi(128000, 0.04, 0.14, fc=False),  # Unreliable FC
    "openrouter/amazon/nova-premier-v1": _mi(1000000, 2.50, 12.50),
    "openrouter/amazon/nova-pro-v1": _mi(300000, 0.80, 3.20),
    "openrouter/anthracite-org/magnum-v4-72b": _mi(16384, 3.00, 5.00, fc=False),
    "openrouter/anthropic/claude-3-haiku": _mi(200000, 0.25, 1.25),
    "openrouter/anthropic/claude-3.5-haiku": _mi(200000, 0.80, 4.00),  # Deprecated
    "openrouter/anthropic/claude-3.5-sonnet": _mi(200000, 6.00, 30.00),  # NEW
    "openrouter/anthropic/claude-3.7-sonnet": _mi(200000, 3.00, 15.00),  # NEW
    "openrouter/anthropic/claude-3.7-sonnet:thinking": _mi(200000, 3.00, 15.00),  # Deprecated
    "openrouter/anthropic/claude-haiku-4.5": _mi(200000, 1.00, 5.00),
    "openrouter/anthropic/claude-opus-4": _mi(200000, 15.00, 75.00),
    "openrouter/anthropic/claude-opus-4.1": _mi(200000, 15.00, 75.00),
    "openrouter/anthropic/claude-opus-4.5": _mi(200000, 5.00, 25.00),
    "openrouter/anthropic/claude-opus-4.6": _mi(1000000, 5.00, 25.00),  # 1M context
    "openrouter/anthropic/claude-sonnet-4": _mi(1000000, 3.00, 15.00),
    "openrouter/anthropic/claude-sonnet-4.5": _mi(1000000, 3.00, 15.00),
    "openrouter/anthropic/claude-sonnet-4.6": _mi(1000000, 3.00, 15.00),
    "openrouter/arcee-ai/coder-large": _mi(32768, 0.50, 0.80, fc=False),
    "openrouter/arcee-ai/maestro-reasoning": _mi(131072, 0.90, 3.30, fc=False),
    "openrouter/arcee-ai/spotlight": _mi(131072, 0.18, 0.18, fc=False),
    "openrouter/arcee-ai/trinity-large-preview": _mi(131000, 0.00, 0.00, fc=False),  # Free preview
    "openrouter/arcee-ai/trinity-mini": _mi(131072, 0.045, 0.15, fc=False),
    "openrouter/aurora-alpha": _mi(128000, 0.00, 0.00, fc=False),  # Free cloaked reasoning
    "openrouter/baidu/ernie-4.5-21b-a3b": _mi(120000, 0.07, 0.28, fc=False),
    "openrouter/baidu/ernie-4.5-21b-a3b-thinking": _mi(131072, 0.07, 0.28, fc=False),
    "openrouter/baidu/ernie-4.5-300b-a47b": _mi(123000, 0.28, 1.10, fc=False),
    "openrouter/baidu/ernie-4.5-vl-28b-a3b": _mi(30000, 0.14, 0.56),  # Vision model
    "openrouter/baidu/ernie-4.5-vl-424b-a47b": _mi(123000, 0.42, 1.25, fc=False),  # Vision model
    "openrouter/bytedance-seed/seed-1.6": _mi(262144, 0.25, 2.00),
    "openrouter/bytedance-seed/seed-1.6-flash": _mi(262144, 0.075, 0.30),
    "openrouter/bytedance-seed/seed-2.0": _mi(262144, 0.30, 2.50),
    "openrouter/bytedance/ui-tars-1.5-7b": _mi(128000, 0.10, 0.20, fc=False),
    "openrouter/cohere/command-a": _mi(256000, 2.50, 10.00, fc=False),
    "openrouter/cohere/command-r-08-2024": _mi(128000, 0.15, 0.60, fc=False),
    "openrouter/cohere/command-r-plus-08-2024": _mi(128000, 2.50, 10.00, fc=False, gen=False),
    "openrouter/cohere/command-r7b-12-2024": _mi(128000, 0.04, 0.15, fc=False),
    "openrouter/deepcogito/cogito-v2-preview-llama-70b": _mi(32768, 0.88, 0.88, fc=False),
    "openrouter/deepcogito/cogito-v2.1-671b": _mi(128000, 1.25, 1.25, fc=False),
    "openrouter/deepseek/deepseek-chat": _mi(163840, 0.32, 0.89),  # FLAKY: can timeout
    "openrouter/deepseek/deepseek-chat-v3-0324": _mi(163840, 0.19, 0.87, fc=False),
    "openrouter/deepseek/deepseek-chat-v3.1": _mi(32768, 0.15, 0.75, fc=False, gen=False),
    "openrouter/deepseek/deepseek-coder-v2": _mi(131072, 0.14, 0.28, fc=False),  # Code generation
    "openrouter/deepseek/deepseek-prover-v2": _mi(163840, 0.50, 2.18, fc=False),  # Math proofs
    "openrouter/deepseek/deepseek-r1": _mi(64000, 0.70, 2.50),  # Reasoning model
    "openrouter/deepseek/deepseek-r1-0528": _mi(163840, 0.40, 1.75),  # Reasoning model
    "openrouter/deepseek/deepseek-r1-distill-llama-70b": _mi(131072, 0.70, 0.80),
    "openrouter/deepseek/deepseek-r1-distill-llama-8b": _mi(131072, 0.01, 0.03),
    "openrouter/deepseek/deepseek-r1-distill-qwen-1.5b": _mi(64000, 0.005, 0.01),
    "openrouter/deepseek/deepseek-r1-distill-qwen-14b": _mi(131072, 0.02, 0.05),
    "openrouter/deepseek/deepseek-r1-distill-qwen-32b": _mi(32768, 0.29, 0.29),
    "openrouter/deepseek/deepseek-r1-distill-qwen-7b": _mi(64000, 0.01, 0.02),
    "openrouter/deepseek/deepseek-r1-turbo": _mi(163840, 0.35, 1.40),  # Faster reasoning
    "openrouter/deepseek/deepseek-v3": _mi(163840, 0.27, 1.10),  # Base V3 model
    "openrouter/deepseek/deepseek-v3.1-terminus": _mi(163840, 0.21, 0.79, fc=False, gen=False),
    "openrouter/deepseek/deepseek-v3.2": _mi(163840, 0.26, 0.38),
    "openrouter/deepseek/deepseek-v3.2-exp": _mi(163840, 0.27, 0.41),
    "openrouter/deepseek/deepseek-v3.2-speciale": _mi(163840, 0.40, 1.20),  # Specialized variant
    "openrouter/eleutherai/llemma_7b": _mi(4096, 0.80, 1.20, fc=False),
    "openrouter/essentialai/rnj-1-instruct": _mi(32768, 0.15, 0.15, fc=False),
    "openrouter/google/gemini-2.0-flash-001": _mi(1048576, 0.10, 0.40),
    "openrouter/google/gemini-2.0-flash-lite-001": _mi(1048576, 0.075, 0.30),
    "openrouter/google/gemini-2.5-flash": _mi(1048576, 0.30, 2.50),
    "openrouter/google/gemini-2.5-flash-image": _mi(32768, 0.30, 2.50, fc=False, gen=False),
    "openrouter/google/gemini-2.5-flash-lite": _mi(1048576, 0.10, 0.40, fc=False),  # Unreliable FC
    "openrouter/google/gemini-2.5-flash-lite-preview-09-2025": _mi(1048576, 0.10, 0.40, fc=False),
    "openrouter/google/gemini-2.5-flash-preview-09-2025": _mi(1048576, 0.30, 2.50, fc=False),
    "openrouter/google/gemini-2.5-pro": _mi(1048576, 1.25, 10.00),
    "openrouter/google/gemini-2.5-pro-preview": _mi(1048576, 1.25, 10.00),
    "openrouter/google/gemini-2.5-pro-preview-05-06": _mi(1048576, 1.25, 10.00, fc=False),
    "openrouter/google/gemini-3-flash-preview": _mi(1048576, 0.50, 3.00),
    "openrouter/google/gemini-3-pro-image-preview": _mi(65536, 2.00, 12.00, fc=False, gen=False),
    "openrouter/google/gemini-3-pro-preview": _mi(1048576, 2.00, 12.00),
    "openrouter/google/gemini-3.1-pro-preview": _mi(1048576, 2.00, 12.00),  # Latest Feb 2026
    "openrouter/google/gemma-2-27b-it": _mi(8192, 0.65, 0.65, fc=False),
    "openrouter/google/gemma-2-9b-it": _mi(8192, 0.03, 0.09, fc=False),
    "openrouter/google/gemma-3-12b-it": _mi(131072, 0.04, 0.13, fc=False),
    "openrouter/google/gemma-3-27b-it": _mi(128000, 0.04, 0.15, fc=False),
    "openrouter/google/gemma-3-4b-it": _mi(131072, 0.04, 0.08, fc=False),
    "openrouter/google/gemma-3n-e4b-it": _mi(32768, 0.02, 0.04, fc=False),
    "openrouter/gryphe/mythomax-l2-13b": _mi(4096, 0.06, 0.06, fc=False),
    "openrouter/ibm-granite/granite-4.0-h-micro": _mi(131000, 0.02, 0.11, fc=False),
    "openrouter/inception/mercury": _mi(128000, 0.25, 1.00, fc=False, gen=False),  # Returns empty
    "openrouter/inception/mercury-coder": _mi(128000, 0.25, 1.00, fc=False, gen=False),
    "openrouter/inflection/inflection-3-pi": _mi(8000, 2.50, 10.00, fc=False),
    "openrouter/inflection/inflection-3-productivity": _mi(8000, 2.50, 10.00, fc=False),
    "openrouter/kwaipilot/kat-coder-pro": _mi(256000, 0.207, 0.828),
    "openrouter/liquid/lfm-2.2-6b": _mi(32768, 0.01, 0.02, fc=False),
    "openrouter/liquid/lfm-2.5-1.2b-instruct": _mi(32768, 0.00, 0.00, fc=False),  # Free
    "openrouter/liquid/lfm-2.5-1.2b-thinking": _mi(32768, 0.00, 0.00, fc=False),  # Free
    "openrouter/liquid/lfm2-8b-a1b": _mi(32768, 0.01, 0.02, fc=False),
    "openrouter/mancer/weaver": _mi(8000, 0.75, 1.00, fc=False),
    "openrouter/meituan/longcat-flash-chat": _mi(131072, 0.20, 0.80, fc=False),
    "openrouter/meta-llama/llama-3-70b-instruct": _mi(8192, 0.51, 0.74, fc=False),
    "openrouter/meta-llama/llama-3-8b-instruct": _mi(8192, 0.03, 0.04, fc=False),
    "openrouter/meta-llama/llama-3.1-405b": _mi(32768, 4.00, 4.00, fc=False),
    "openrouter/meta-llama/llama-3.1-405b-instruct": _mi(131000, 4.00, 4.00, fc=False, gen=False),
    "openrouter/meta-llama/llama-3.1-70b-instruct": _mi(131072, 0.40, 0.40, fc=False),
    "openrouter/meta-llama/llama-3.1-8b-instruct": _mi(16384, 0.02, 0.05, fc=False),
    "openrouter/meta-llama/llama-3.2-11b-vision-instruct": _mi(131072, 0.05, 0.05, fc=False),
    "openrouter/meta-llama/llama-3.2-1b-instruct": _mi(60000, 0.03, 0.20, fc=False),
    "openrouter/meta-llama/llama-3.2-3b-instruct": _mi(131072, 0.02, 0.02, fc=False),
    "openrouter/meta-llama/llama-3.3-70b-instruct": _mi(131072, 0.10, 0.32, fc=False, gen=False),
    "openrouter/meta-llama/llama-4-maverick": _mi(1048576, 0.15, 0.60, fc=False),
    "openrouter/meta-llama/llama-4-scout": _mi(327680, 0.08, 0.30, fc=False),
    "openrouter/meta-llama/llama-guard-2-8b": _mi(8192, 0.20, 0.20, fc=False),  # Safety model
    "openrouter/meta-llama/llama-guard-3-8b": _mi(131072, 0.02, 0.06, fc=False),  # Safety model
    "openrouter/meta-llama/llama-guard-4-12b": _mi(163840, 0.18, 0.18, fc=False),  # Safety model
    "openrouter/microsoft/phi-4": _mi(16384, 0.06, 0.14, fc=False),
    "openrouter/microsoft/phi-4-reasoning-plus": _mi(32768, 0.07, 0.35, fc=False),
    "openrouter/microsoft/wizardlm-2-8x22b": _mi(65535, 0.62, 0.62, fc=False),
    "openrouter/minimax/minimax-01": _mi(1000192, 0.20, 1.10, fc=False),
    "openrouter/minimax/minimax-m1": _mi(1000000, 0.40, 2.20, fc=False),  # Unreliable FC
    "openrouter/minimax/minimax-m2": _mi(196608, 0.255, 1.00, fc=False),
    "openrouter/minimax/minimax-m2-her": _mi(65536, 0.30, 1.20, fc=False),  # Roleplay model
    "openrouter/minimax/minimax-m2.1": _mi(196608, 0.27, 0.95),
    "openrouter/minimax/minimax-m2.5": _mi(196608, 0.30, 1.10),
    "openrouter/mistralai/codestral-2508": _mi(256000, 0.30, 0.90, fc=False),
    "openrouter/mistralai/devstral-2512": _mi(262144, 0.40, 2.00, fc=False),
    "openrouter/mistralai/devstral-medium": _mi(131072, 0.40, 2.00, fc=False, gen=False),
    "openrouter/mistralai/devstral-small": _mi(131072, 0.10, 0.30, fc=False, gen=False),
    "openrouter/mistralai/ministral-14b-2512": _mi(262144, 0.20, 0.20, fc=False),
    "openrouter/mistralai/ministral-3b": _mi(131072, 0.04, 0.04, fc=False),
    "openrouter/mistralai/ministral-3b-2512": _mi(131072, 0.10, 0.10, fc=False),
    "openrouter/mistralai/ministral-8b": _mi(131072, 0.10, 0.10, fc=False),
    "openrouter/mistralai/ministral-8b-2512": _mi(262144, 0.15, 0.15, fc=False),
    "openrouter/mistralai/mistral-7b-instruct": _mi(32768, 0.20, 0.20, fc=False),
    "openrouter/mistralai/mistral-7b-instruct-v0.1": _mi(2824, 0.11, 0.19, fc=False),
    "openrouter/mistralai/mistral-7b-instruct-v0.2": _mi(32768, 0.20, 0.20, fc=False),
    "openrouter/mistralai/mistral-7b-instruct-v0.3": _mi(32768, 0.20, 0.20, fc=False),
    "openrouter/mistralai/mistral-large": _mi(128000, 2.00, 6.00, fc=False),
    "openrouter/mistralai/mistral-large-2407": _mi(131072, 2.00, 6.00),
    "openrouter/mistralai/mistral-large-2411": _mi(131072, 2.00, 6.00, fc=False),
    "openrouter/mistralai/mistral-large-2512": _mi(262144, 0.50, 1.50, fc=False),
    "openrouter/mistralai/mistral-medium-3": _mi(131072, 0.40, 2.00, fc=False),
    "openrouter/mistralai/mistral-medium-3.1": _mi(131072, 0.40, 2.00, fc=False),
    "openrouter/mistralai/mistral-nemo": _mi(131072, 0.02, 0.04, fc=False),
    "openrouter/mistralai/mistral-saba": _mi(32768, 0.20, 0.60, fc=False),
    "openrouter/mistralai/mistral-small-24b-instruct-2501": _mi(32768, 0.05, 0.08, fc=False),
    "openrouter/mistralai/mistral-small-3.1-24b-instruct": _mi(128000, 0.35, 0.56, fc=False),
    "openrouter/mistralai/mistral-small-3.2-24b-instruct": _mi(131072, 0.06, 0.18, fc=False),
    "openrouter/mistralai/mistral-small-creative": _mi(32768, 0.10, 0.30, fc=False),
    "openrouter/mistralai/mistral-tiny": _mi(32768, 0.25, 0.25, fc=False),
    "openrouter/mistralai/mixtral-8x22b-instruct": _mi(65536, 2.00, 6.00, fc=False),
    "openrouter/mistralai/mixtral-8x7b-instruct": _mi(32768, 0.54, 0.54, fc=False),
    "openrouter/mistralai/pixtral-12b": _mi(32768, 0.10, 0.10, fc=False),
    "openrouter/mistralai/pixtral-large-2411": _mi(131072, 2.00, 6.00, fc=False),
    "openrouter/mistralai/voxtral-small-24b-2507": _mi(32000, 0.10, 0.30, fc=False),  # Audio input
    "openrouter/moonshotai/kimi-dev-72b": _mi(131072, 0.29, 1.15, fc=False),
    "openrouter/moonshotai/kimi-k2": _mi(131072, 0.50, 2.40, fc=False),  # Unreliable FC
    "openrouter/moonshotai/kimi-k2-0905": _mi(131072, 0.40, 2.00, fc=False),  # Unreliable FC
    "openrouter/moonshotai/kimi-k2-thinking": _mi(131072, 0.47, 2.00, fc=False),  # SLOW
    "openrouter/moonshotai/kimi-k2.5": _mi(262144, 0.45, 2.20),  # Multimodal + agentic
    "openrouter/morph/morph-v3-fast": _mi(81920, 0.80, 1.20, fc=False),
    "openrouter/morph/morph-v3-large": _mi(262144, 0.90, 1.90, fc=False),
    "openrouter/neversleep/llama-3.1-lumimaid-8b": _mi(32768, 0.09, 0.60, fc=False),
    "openrouter/neversleep/noromaid-20b": _mi(4096, 1.00, 1.75, fc=False),
    "openrouter/nex-agi/deepseek-v3.1-nex-n1": _mi(131072, 0.27, 1.00, fc=False),
    "openrouter/nousresearch/deephermes-3-mistral-24b-preview": _mi(
    32768, 0.02, 0.10, fc=False, gen=False
    ),
    "openrouter/nousresearch/hermes-2-pro-llama-3-8b": _mi(8192, 0.14, 0.14, fc=False),
    "openrouter/nousresearch/hermes-3-llama-3.1-405b": _mi(131072, 1.00, 1.00, fc=False),
    "openrouter/nousresearch/hermes-3-llama-3.1-70b": _mi(65536, 0.30, 0.30, fc=False),
    "openrouter/nousresearch/hermes-4-405b": _mi(131072, 1.00, 3.00, fc=False),
    "openrouter/nousresearch/hermes-4-70b": _mi(131072, 0.13, 0.40, fc=False),
    "openrouter/nvidia/llama-3.1-nemotron-70b-instruct": _mi(131072, 1.20, 1.20, fc=False),
    "openrouter/nvidia/llama-3.1-nemotron-ultra-253b-v1": _mi(131072, 0.60, 1.80, fc=False),
    "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5": _mi(131072, 0.10, 0.40),
    "openrouter/nvidia/nemotron-3-nano-30b-a3b": _mi(262144, 0.05, 0.20, fc=False),  # Unreliable FC
    "openrouter/nvidia/nemotron-nano-12b-v2-vl": _mi(131072, 0.07, 0.20, fc=False),  # Vision model
    "openrouter/nvidia/nemotron-nano-9b-v2": _mi(131072, 0.04, 0.16),
    "openrouter/openai/chatgpt-4o-latest": _mi(128000, 5.00, 15.00, fc=False),
    "openrouter/openai/codex-mini-latest": _mi(200000, 1.50, 6.00),
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
    "openrouter/openai/gpt-4o-audio-preview": _mi(128000, 2.50, 10.00, fc=False),  # Audio model
    "openrouter/openai/gpt-4o-mini": _mi(128000, 0.15, 0.60),
    "openrouter/openai/gpt-4o-mini-2024-07-18": _mi(128000, 0.15, 0.60),
    "openrouter/openai/gpt-4o-mini-search-preview": _mi(128000, 0.15, 0.60, fc=False),
    "openrouter/openai/gpt-4o-search-preview": _mi(128000, 2.50, 10.00, fc=False),
    "openrouter/openai/gpt-4o:extended": _mi(128000, 6.00, 18.00),
    "openrouter/openai/gpt-5": _mi(400000, 1.25, 10.00),
    "openrouter/openai/gpt-5-chat": _mi(128000, 1.25, 10.00, fc=False),
    "openrouter/openai/gpt-5-codex": _mi(400000, 1.25, 10.00),
    "openrouter/openai/gpt-5-image": _mi(400000, 10.00, 10.00, fc=False, gen=False),  # Image gen
    "openrouter/openai/gpt-5-image-mini": _mi(400000, 2.50, 2.00, fc=False, gen=False),  # Image gen
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
    "openrouter/openai/gpt-audio": _mi(128000, 2.50, 10.00, fc=False),  # Audio model
    "openrouter/openai/gpt-audio-mini": _mi(128000, 0.60, 2.40, fc=False),  # Audio model
    "openrouter/openai/gpt-oss-120b": _mi(131072, 0.039, 0.19, fc=False),
    "openrouter/openai/gpt-oss-20b": _mi(131072, 0.03, 0.14, fc=False),
    "openrouter/openai/gpt-oss-safeguard-20b": _mi(131072, 0.075, 0.30, fc=False),  # Safety model
    "openrouter/openai/o1": _mi(200000, 15.00, 60.00, fc=False),  # SLOW: reasoning model
    "openrouter/openai/o1-mini": _mi(128000, 1.10, 4.40, fc=False),
    "openrouter/openai/o1-pro": _mi(200000, 150.00, 600.00, fc=False),
    "openrouter/openai/o3": _mi(200000, 2.00, 8.00),
    "openrouter/openai/o3-deep-research": _mi(200000, 10.00, 40.00, fc=False),
    "openrouter/openai/o3-mini": _mi(200000, 1.10, 4.40, fc=False),  # Unreliable FC
    "openrouter/openai/o3-mini-high": _mi(200000, 1.10, 4.40),
    "openrouter/openai/o3-pro": _mi(200000, 20.00, 80.00, fc=False),
    "openrouter/openai/o4-mini": _mi(200000, 1.10, 4.40),
    "openrouter/openai/o4-mini-deep-research": _mi(200000, 2.00, 8.00, fc=False),
    "openrouter/openai/o4-mini-high": _mi(200000, 1.10, 4.40),
    "openrouter/opengvlab/internvl3-78b": _mi(32768, 0.15, 0.60, fc=False),
    "openrouter/perplexity/sonar": _mi(127072, 1.00, 1.00, fc=False),
    "openrouter/perplexity/sonar-deep-research": _mi(128000, 2.00, 8.00, fc=False),
    "openrouter/perplexity/sonar-pro": _mi(200000, 3.00, 15.00, fc=False),
    "openrouter/perplexity/sonar-pro-search": _mi(200000, 3.00, 15.00, fc=False),  # Agentic search
    "openrouter/perplexity/sonar-reasoning-pro": _mi(128000, 2.00, 8.00, fc=False),
    "openrouter/prime-intellect/intellect-3": _mi(131072, 0.20, 1.10, fc=False),
    "openrouter/qwen/qwen-2.5-72b-instruct": _mi(32768, 0.12, 0.39),
    "openrouter/qwen/qwen-2.5-7b-instruct": _mi(32768, 0.04, 0.10, fc=False),
    "openrouter/qwen/qwen-2.5-coder-32b-instruct": _mi(32768, 0.20, 0.20, fc=False),
    "openrouter/qwen/qwen-2.5-vl-7b-instruct": _mi(32768, 0.20, 0.20, fc=False),  # Vision model
    "openrouter/qwen/qwen-max": _mi(32768, 1.60, 6.40, fc=False),
    "openrouter/qwen/qwen-plus": _mi(1000000, 0.40, 1.20),
    "openrouter/qwen/qwen-plus-2025-07-28": _mi(1000000, 0.40, 1.20),
    "openrouter/qwen/qwen-plus-2025-07-28:thinking": _mi(1000000, 0.40, 4.00),  # SLOW: thinking
    "openrouter/qwen/qwen-turbo": _mi(131072, 0.05, 0.20),
    "openrouter/qwen/qwen-vl-max": _mi(131072, 0.80, 3.20),  # Vision model
    "openrouter/qwen/qwen-vl-plus": _mi(131072, 0.21, 0.63, fc=False),  # Vision model
    "openrouter/qwen/qwen2.5-coder-7b-instruct": _mi(32768, 0.03, 0.09, fc=False),
    "openrouter/qwen/qwen2.5-vl-32b-instruct": _mi(128000, 0.20, 0.60, fc=False),  # Vision model
    "openrouter/qwen/qwen2.5-vl-72b-instruct": _mi(32768, 0.80, 0.80, fc=False),  # Vision model
    "openrouter/qwen/qwen3-14b": _mi(40960, 0.06, 0.24, fc=False),
    "openrouter/qwen/qwen3-235b-a22b": _mi(131072, 0.455, 1.82, fc=False, gen=False),
    "openrouter/qwen/qwen3-235b-a22b-2507": _mi(262144, 0.07, 0.10, fc=False),
    "openrouter/qwen/qwen3-235b-a22b-thinking-2507": _mi(131072, 0.00, 0.00),  # SLOW: thinking
    "openrouter/qwen/qwen3-30b-a3b": _mi(40960, 0.08, 0.28, fc=False),
    "openrouter/qwen/qwen3-30b-a3b-instruct-2507": _mi(262144, 0.09, 0.30),
    "openrouter/qwen/qwen3-30b-a3b-thinking-2507": _mi(32768, 0.05, 0.34),  # SLOW: thinking
    "openrouter/qwen/qwen3-32b": _mi(40960, 0.08, 0.24, fc=False),
    "openrouter/qwen/qwen3-8b": _mi(32000, 0.05, 0.40, fc=False),
    "openrouter/qwen/qwen3-coder": _mi(262144, 0.22, 1.00),
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct": _mi(160000, 0.07, 0.27),
    "openrouter/qwen/qwen3-coder-flash": _mi(1000000, 0.30, 1.50),
    "openrouter/qwen/qwen3-coder-next": _mi(262144, 0.12, 0.75),  # Latest agentic coding model
    "openrouter/qwen/qwen3-coder-plus": _mi(1000000, 1.00, 5.00, fc=False),
    "openrouter/qwen/qwen3-max": _mi(262144, 1.20, 6.00),
    "openrouter/qwen/qwen3-max-thinking": _mi(262144, 1.20, 6.00, fc=False),  # SLOW: thinking
    "openrouter/qwen/qwen3-next-80b-a3b-instruct": _mi(262144, 0.09, 1.10),
    "openrouter/qwen/qwen3-next-80b-a3b-thinking": _mi(128000, 0.15, 1.20),  # SLOW: thinking
    "openrouter/qwen/qwen3-vl-235b-a22b-instruct": _mi(262144, 0.20, 0.88),  # Vision model
    "openrouter/qwen/qwen3-vl-235b-a22b-thinking": _mi(131072, 0.00, 0.00, fc=False),
    "openrouter/qwen/qwen3-vl-30b-a3b-instruct": _mi(131072, 0.13, 0.52),  # Vision model
    "openrouter/qwen/qwen3-vl-30b-a3b-thinking": _mi(131072, 0.00, 0.00),  # SLOW: thinking
    "openrouter/qwen/qwen3-vl-32b-instruct": _mi(131072, 0.104, 0.416, fc=False),  # Vision model
    "openrouter/qwen/qwen3-vl-8b-instruct": _mi(131072, 0.08, 0.50),  # Vision model
    "openrouter/qwen/qwen3-vl-8b-thinking": _mi(131072, 0.117, 1.365),  # SLOW: thinking
    "openrouter/qwen/qwen3.5-397b-a17b": _mi(262144, 0.55, 3.50),  # NEW
    "openrouter/qwen/qwen3.5-plus-02-15": _mi(1000000, 0.40, 2.40),  # NEW
    "openrouter/qwen/qwq-32b": _mi(32768, 0.15, 0.40, fc=False),  # SLOW: reasoning model
    "openrouter/raifle/sorcererlm-8x22b": _mi(16000, 4.50, 4.50, fc=False),
    "openrouter/relace/relace-apply-3": _mi(256000, 0.85, 1.25, fc=False),
    "openrouter/relace/relace-search": _mi(256000, 1.00, 3.00, fc=False),  # Agentic search
    "openrouter/sao10k/l3-euryale-70b": _mi(8192, 1.48, 1.48),
    "openrouter/sao10k/l3-lunaris-8b": _mi(8192, 0.04, 0.05, fc=False),
    "openrouter/sao10k/l3.1-70b-hanami-x1": _mi(16000, 3.00, 3.00, fc=False),
    "openrouter/sao10k/l3.1-euryale-70b": _mi(32768, 0.65, 0.75),
    "openrouter/sao10k/l3.3-euryale-70b": _mi(131072, 0.65, 0.75, fc=False),
    "openrouter/stepfun-ai/step-3.5-flash": _mi(256000, 0.10, 0.30, fc=False),
    "openrouter/stepfun-ai/step3": _mi(65536, 0.57, 1.42, fc=False),
    "openrouter/stepfun/step-3.5-flash": _mi(256000, 0.10, 0.30),  # NEW
    "openrouter/switchpoint/router": _mi(131072, 0.85, 3.40, fc=False),
    "openrouter/tencent/hunyuan-a13b-instruct": _mi(131072, 0.14, 0.57, fc=False),
    "openrouter/thedrummer/cydonia-24b-v4.1": _mi(131072, 0.30, 0.50, fc=False),
    "openrouter/thedrummer/rocinante-12b": _mi(32768, 0.17, 0.43),
    "openrouter/thedrummer/skyfall-36b-v2": _mi(32768, 0.55, 0.80, fc=False),
    "openrouter/thedrummer/unslopnemo-12b": _mi(32768, 0.40, 0.40),
    "openrouter/tngtech/deepseek-r1t-chimera": _mi(163840, 0.30, 1.20, fc=False),
    "openrouter/tngtech/deepseek-r1t2-chimera": _mi(163840, 0.25, 0.85),
    "openrouter/tngtech/tng-r1t-chimera": _mi(163840, 0.25, 0.85, fc=False),  # Creative model
    "openrouter/undi95/remm-slerp-l2-13b": _mi(6144, 0.45, 0.65, fc=False),
    "openrouter/upstage/solar-pro-3": _mi(128000, 0.00, 0.00, fc=False),  # Free
    "openrouter/writer/palmyra-x5": _mi(1040000, 0.60, 6.00, fc=False),
    "openrouter/x-ai/grok-3": _mi(131072, 3.00, 15.00),
    "openrouter/x-ai/grok-3-beta": _mi(131072, 3.00, 15.00),
    "openrouter/x-ai/grok-3-mini": _mi(131072, 0.30, 0.50),
    "openrouter/x-ai/grok-3-mini-beta": _mi(131072, 0.30, 0.50),
    "openrouter/x-ai/grok-4": _mi(256000, 3.00, 15.00),
    "openrouter/x-ai/grok-4-fast": _mi(2000000, 0.20, 0.50),
    "openrouter/x-ai/grok-4.1-fast": _mi(2000000, 0.20, 0.50),
    "openrouter/x-ai/grok-code-fast-1": _mi(256000, 0.20, 1.50),
    "openrouter/xiaomi/mimo-v2-flash": _mi(262144, 0.09, 0.29, fc=False),
    "openrouter/z-ai/glm-4-32b": _mi(128000, 0.10, 0.10),
    "openrouter/z-ai/glm-4.5": _mi(131000, 0.55, 2.00),
    "openrouter/z-ai/glm-4.5-air": _mi(131072, 0.13, 0.85),
    "openrouter/z-ai/glm-4.5v": _mi(65536, 0.60, 1.80, fc=False),
    "openrouter/z-ai/glm-4.6": _mi(202752, 0.35, 1.71, fc=False),  # Unreliable FC
    "openrouter/z-ai/glm-4.6v": _mi(131072, 0.30, 0.90),
    "openrouter/z-ai/glm-4.7": _mi(202752, 0.38, 1.70),
    "openrouter/z-ai/glm-4.7-flash": _mi(202752, 0.06, 0.40),
    "openrouter/z-ai/glm-5": _mi(204800, 0.95, 2.55),  # 744B MoE - FC verified
    "Qwen/Qwen2.5-7B-Instruct-Turbo": _mi(32768, 0.30, 0.30, fc=False),
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": _mi(262144, 0.20, 0.60, fc=False),
    "Qwen/Qwen3-235B-A22B-Thinking-2507": _mi(262144, 0.65, 3.00, fc=False),  # Dep 2026-03
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": _mi(262144, 2.00, 2.00),
    "Qwen/Qwen3-Coder-Next-FP8": _mi(262144, 0.50, 1.20),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": _mi(262144, 0.15, 1.50),
    "Qwen/Qwen3-Next-80B-A3B-Thinking": _mi(262144, 0.15, 1.50),  # NEW
    "Qwen/Qwen3-VL-32B-Instruct": _mi(262144, 0.50, 1.50),  # NEW
    "Qwen/Qwen3-VL-8B-Instruct": _mi(262144, 0.18, 0.68, fc=False),  # Vision model
    "Qwen/Qwen3.5-397B-A17B": _mi(262144, 0.60, 3.60, fc=False),
    "text-embedding-004": _emb(2048, 0.00),  # 768 dimensions, free tier
    "text-embedding-3-large": _emb(8191, 0.13),  # 3072 dimensions
    "text-embedding-3-small": _emb(8191, 0.02),  # 1536 dimensions
    "text-embedding-ada-002": _emb(8191, 0.10),  # 1536 dimensions (legacy)
    "zai-org/GLM-4.5-Air-FP8": _mi(131072, 0.20, 1.10),
    "zai-org/GLM-4.6": _mi(202752, 0.60, 2.20),  # NEW
    "zai-org/GLM-4.7": _mi(202752, 0.45, 2.00),
    "zai-org/GLM-5": _mi(202752, 1.00, 3.20),
    # ==========================================================================
    # Auto-discovered models (verify pricing and capabilities)
    # ==========================================================================
    "openrouter/arcee-ai/virtuoso-large": _mi(131072, 0.75, 1.20),  # NEW
    "meta-llama/Llama-3.1-405B-Instruct": _mi(4096, 3.50, 3.50),  # NEW
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference": _mi(8192, 0.90, 0.90),  # NEW
    "meta-llama/Llama-3.2-1B-Instruct": _mi(131072, 0.06, 0.06),  # NEW
    "Qwen/Qwen2.5-14B-Instruct": _mi(32768, 0.80, 0.80),  # NEW
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference": _mi(16384, 0.20, 0.20),  # NEW
    "meta-llama/Meta-Llama-3-8B-Instruct": _mi(8192, 0.20, 0.20),  # NEW
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": _mi(1048576, 0.18, 0.59),  # NEW
    "Qwen/Qwen2.5-72B-Instruct": _mi(32768, 1.20, 1.20),  # NEW
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": _mi(131072, 2.00, 2.00),  # NEW
    "gemini-2.0-flash-001": _mi(1048576, 0.00, 0.00),  # NEW: needs pricing
    "gemini-2.0-flash-lite-001": _mi(1048576, 0.00, 0.00),  # NEW: needs pricing
    "gemini-2.5-flash-image": _mi(32768, 0.00, 0.00),  # NEW: needs pricing
}

# Populate cache pricing for known providers.
# Anthropic: cache_read = 10% of input, cache_write = 125% of input.
# OpenAI: cache_read = 50% of input (automatic caching, no write premium).
for _name, _info in MODEL_INFO.items():
    if _info.cache_read_price_per_1M is not None:
        continue
    if _name.startswith("claude-"):
        _info.cache_read_price_per_1M = _info.input_price_per_1M * 0.1
        _info.cache_write_price_per_1M = _info.input_price_per_1M * 1.25
    elif (
        _name.startswith(_OPENAI_PREFIXES)
        and not _name.startswith(("text-embedding", "openai/"))
        and _info.is_generation_supported
    ):
        _info.cache_read_price_per_1M = _info.input_price_per_1M * 0.5

# ==========================================================================
# FLAKY MODEL REGISTRY
# Models that have been tested and found to have reliability issues
# ==========================================================================
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


def model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
    token_callback: TokenCallback | None = None,
) -> Model:
    """Get a model instance based on model name prefix.

    Args:
        model_name: The name of the model (with provider prefix if applicable).
        model_config: Optional dictionary of model configuration parameters.
            If it contains "base_url", routing is bypassed and an OpenAICompatibleModel
            is built with that base_url and optional "api_key".
        token_callback: Optional async callback invoked with each streamed text token.

    Returns:
        Model: An appropriate Model instance for the specified model.

    Raises:
        KISSError: If the model name is not recognized.
    """
    if model_config and "base_url" in model_config:
        base_url = model_config["base_url"]
        api_key = model_config.get("api_key", "")
        filtered = {k: v for k, v in model_config.items() if k not in ("base_url", "api_key")}
        return _openai_compatible(
            model_name, base_url, api_key, filtered or None, token_callback,
        )
    keys = config_module.DEFAULT_CONFIG.agent.api_keys
    if model_name.startswith("openrouter/"):
        return _openai_compatible(
            model_name, "https://openrouter.ai/api/v1",
            keys.OPENROUTER_API_KEY, model_config, token_callback,
        )
    if model_name == "text-embedding-004":
        if GeminiModel is None:
            raise KISSError(
                "Google GenAI SDK not installed. Install 'google-genai' to use Gemini models."
            )
        return GeminiModel(
            model_name=model_name, api_key=keys.GEMINI_API_KEY,
            model_config=model_config, token_callback=token_callback,
        )
    if model_name.startswith(_OPENAI_PREFIXES) and not model_name.startswith("openai/gpt-oss"):
        return _openai_compatible(
            model_name, "https://api.openai.com/v1",
            keys.OPENAI_API_KEY, model_config, token_callback,
        )
    if model_name.startswith(_TOGETHER_PREFIXES):
        return _openai_compatible(
            model_name, "https://api.together.xyz/v1",
            keys.TOGETHER_API_KEY, model_config, token_callback,
        )
    if model_name.startswith("claude-"):
        if AnthropicModel is None:
            raise KISSError(
                "Anthropic SDK not installed. Install 'anthropic' to use Claude models."
            )
        return AnthropicModel(
            model_name=model_name, api_key=keys.ANTHROPIC_API_KEY,
            model_config=model_config, token_callback=token_callback,
        )
    if model_name.startswith("gemini-"):
        if GeminiModel is None:
            raise KISSError(
                "Google GenAI SDK not installed. Install 'google-genai' to use Gemini models."
            )
        return GeminiModel(
            model_name=model_name, api_key=keys.GEMINI_API_KEY,
            model_config=model_config, token_callback=token_callback,
        )
    if model_name.startswith("minimax-"):
        return _openai_compatible(
            model_name, "https://api.minimax.chat/v1",
            keys.MINIMAX_API_KEY, model_config, token_callback,
        )
    raise KISSError(f"Unknown model name: {model_name}")


def get_available_models() -> list[str]:
    """Return model names for which an API key is configured and generation is supported.

    Returns:
        list[str]: Sorted list of model name strings that have a configured API key
            and support text generation.
    """
    keys = config_module.DEFAULT_CONFIG.agent.api_keys
    prefix_to_key = {
        "openrouter/": keys.OPENROUTER_API_KEY,
        "claude-": keys.ANTHROPIC_API_KEY,
        "gemini-": keys.GEMINI_API_KEY,
        "minimax-": keys.MINIMAX_API_KEY,
    }
    result = []
    for name, info in MODEL_INFO.items():
        if not info.is_generation_supported:
            continue
        api_key = ""
        for prefix, key in prefix_to_key.items():
            if name.startswith(prefix):
                api_key = key
                break
        if not api_key:
            if name == "text-embedding-004":
                api_key = keys.GEMINI_API_KEY
            elif name.startswith(_OPENAI_PREFIXES) and not name.startswith("openai/gpt-oss"):
                api_key = keys.OPENAI_API_KEY
            elif name.startswith(_TOGETHER_PREFIXES):
                api_key = keys.TOGETHER_API_KEY
        if api_key:
            result.append(name)
    return sorted(result)


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
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name].context_length
    raise KeyError(f"Model '{model_name}' not found in MODEL_INFO")
