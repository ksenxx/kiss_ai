"""Configuration for KISSClaw."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KissClawConfig:
    assistant_name: str = "Andy"
    trigger_pattern: str = ""  # auto-built from assistant_name
    poll_interval: float = 2.0  # seconds
    scheduler_poll_interval: float = 60.0  # seconds
    ipc_poll_interval: float = 1.0  # seconds
    idle_timeout: float = 1800.0  # 30 min
    max_concurrent_agents: int = 5
    max_retries: int = 5
    base_retry_delay: float = 5.0  # seconds
    data_dir: str = ""
    groups_dir: str = ""
    store_dir: str = ""
    main_group_folder: str = "main"
    model_name: str = "claude-sonnet-4-5"
    max_steps: int = 15
    max_budget: float = 10.0
    timezone: str = ""
    group_memories: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import re
        if not self.trigger_pattern:
            escaped = re.escape(self.assistant_name)
            self.trigger_pattern = rf"^@{escaped}\b"
        if not self.data_dir:
            self.data_dir = str(Path.cwd() / "kissclaw_data")
        if not self.groups_dir:
            self.groups_dir = str(Path(self.data_dir) / "groups")
        if not self.store_dir:
            self.store_dir = str(Path(self.data_dir) / "store")
        if not self.timezone:
            try:
                import datetime
                self.timezone = str(datetime.datetime.now().astimezone().tzinfo)
            except Exception:
                self.timezone = "UTC"
