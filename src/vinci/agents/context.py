from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RunContext:
    """Per‑run metadata that the agent loop reads and mutates."""

    run_id: str = ""  # filled in by Agent.run()
    deadline: float | None = None  # None → no hard time limit
    state: Dict[str, Any] = {}  # mutable dictionary shared by all loop code

    def __post_init__(self) -> None:
        self.run_id = str(uuid.uuid4()) if not self.run_id else self.run_id
        self.state = {} if self.state is None else self.state
