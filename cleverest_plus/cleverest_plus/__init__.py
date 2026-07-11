"""Cleverest++ hardened research prototype (integrated pipeline)."""

from .benchmark import Issue, approximate_reached, load_benchmark, load_issue
from .campaign import CampaignConfig, run_campaign
from .core import (
    Candidate,
    CandidateError,
    CrashSignature,
    Execution,
    Outcome,
    classify_differential,
    coverage_fingerprint,
    crash_signature,
    execute,
    materialize_argv,
    select_diverse,
    validate_argv,
    validate_format,
)
from .dispatcher import PortfolioMember, allocate_budget, run_issue
from .llm import ChatMessage, LLMClient, LLMError, LLMResponse, openai_client, openrouter_client
from .loop import IterationRecord, TrialResult, run_trial, summarize_trials
from .prompt import AttemptRecord, CommitContext, build_messages

__all__ = [
    "AttemptRecord", "Candidate", "CandidateError", "CampaignConfig", "ChatMessage",
    "CommitContext", "CrashSignature", "Execution", "IterationRecord", "Issue",
    "LLMClient", "LLMError", "LLMResponse", "Outcome", "PortfolioMember",
    "TrialResult",
    "allocate_budget", "approximate_reached", "build_messages",
    "classify_differential", "coverage_fingerprint", "crash_signature",
    "execute", "load_benchmark", "load_issue", "materialize_argv", "openai_client",
    "openrouter_client", "run_campaign", "run_issue", "run_trial",
    "select_diverse", "summarize_trials", "validate_argv", "validate_format",
]
