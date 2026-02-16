"""KISSClaw - A lightweight personal AI assistant built on KISSAgent.

A Python clone of NanoClaw: message routing, per-group isolation,
scheduled tasks, and agent execution via KISSAgent.
"""

from kiss.agents.kissclaw.orchestrator import KissClawOrchestrator

__all__ = ["KissClawOrchestrator"]
