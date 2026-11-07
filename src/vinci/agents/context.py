from __future__ import annotations
import threading
from typing import Any
import uuid
from dataclasses import dataclass, field


@dataclass
class RunContext:
    """Per‑run metadata shared between the agent loop and any workers/threads.

    All mutable data should live inside ``state``; the other fields are read‑only
    helpers that the loop (and later threads) inspect frequently.
    """

    run_id: str = ""  # filled automatically by __post_init__
    cv = None
    deadline: float | None = None  # None → no hard time limit
    state: dict[str, Any] = None  # mutable dictionary – the "blackboard"
    memory: dict[str, Any] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_id = str(uuid.uuid4()) if not self.run_id else self.run_id
        self.state = {} if self.state is None else self.state
        self.cv = threading.Condition()
