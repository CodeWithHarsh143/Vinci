import asyncio
from dataclasses import dataclass
import json
import time
from vinci.llm.client import LLMClient, CallResponse, ToolCallRequest
from vinci.tools.base import ToolResult
from vinci.tools.registry import ToolsRegistry
from .context import RunContext
import queue
import threading

MAX_RESULT_CHAR: int = 5000


@dataclass
class AgentResult:
    success: bool
    total_iterations: int
    total_tool_calls: int
    error: str | None = None
    final_answer: str | None = None
    duration_ms: float = 0.0


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolsRegistry,
        max_iterations: int = 30,
        time_limit: float = 10 * 60,
    ):
        self.llm = llm
        self.registry = registry
        self.context = RunContext()
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.result_queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------
    # Helper / utility methods (unchanged from the single‑threaded version)
    # ------------------------------------------------------------------
    def serialize(self, result: ToolResult) -> str:
        if result.success is True:
            return json.dumps({"content": result.data})
        return json.dumps({"content": {"error": result.error}})

    def assistant_formatted_tool_call(
        self, tool_calls: list[ToolCallRequest]
    ) -> list[dict]:
        formatted_calls = []
        for tc in tool_calls:
            entry: dict = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            if tc.thought_signature:
                entry["extra_content"] = {
                    "google": {"thought_signature": tc.thought_signature}
                }
            formatted_calls.append(entry)
        return formatted_calls

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    async def _loop(self, task: str) -> None:
        """Iterate until a final answer, cancellation or a timeout."""
        start = time.perf_counter()
        messages: list = [{"role": "user", "content": task}]

        iterations: int = 0
        tool_calls: int = 0

        while iterations < self.max_iterations:
            # ------------------- cancellation check -----------------
            # .wait(0.5) blocks up to 0.5 s; we then explicitly test the flag.
            self._cancel_event.wait(0.5)
            if self._cancel_event.is_set():
                self.context.state["error"] = "cancelled"
                break

            # ------------------- deadline checks -----------------
            elapsed = time.perf_counter() - start
            if elapsed > self.time_limit:
                self.context.state["error"] = "Timeout Error"
                break
            if self.context.deadline is not None and elapsed >= self.context.deadline:
                self.context.state["error"] = "Deadline exceeded"
                break

            # ------------------- progress counters -----------------
            iterations += 1
            self.context.state["iterations"] = iterations

            # ------------------- ask the LLM -----------------
            response: CallResponse = await self.llm.call(
                messages, self.registry.list_schemas()
            )

            # ------------------- no tool call → final answer -----
            if len(response.tool_calls) == 0:
                self.context.state["final_answer"] = response.content
                break  # <‑‑ correct: break stays inside the while

            # ------------------- append assistant message ---------
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": self.assistant_formatted_tool_call(
                        response.tool_calls
                    ),
                }
            )

            # ------------------- execute each tool -----------------
            for tool in response.tool_calls:
                result = await self.registry.execute(
                    tool_name=tool.name, args=tool.arguments
                )
                content = self.serialize(result)
                if len(content) > MAX_RESULT_CHAR:
                    content = content[:MAX_RESULT_CHAR] + "..truncated"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "name": tool.name,
                        "content": content,
                    }
                )
                tool_calls += 1
                self.context.state["tool_calls"] = tool_calls

        # ---------- post‑loop: build the final AgentResult ----------
        error = self.context.state.get("error")
        final_answer = self.context.state.get("final_answer")
        if not error and final_answer is None:
            # no explicit error and no final answer → iterations exhausted
            self.context.state["error"] = "Maximum iterations exceeded"

        duration_ms = (time.perf_counter() - start) * 1000

        if error:
            self.result_queue.put(
                AgentResult(
                    success=False,
                    error=error,
                    total_iterations=iterations,
                    total_tool_calls=tool_calls,
                    duration_ms=duration_ms,
                    final_answer=final_answer,
                )
            )
        else:
            self.result_queue.put(
                AgentResult(
                    success=True,
                    total_iterations=iterations,
                    total_tool_calls=tool_calls,
                    duration_ms=duration_ms,
                    final_answer=final_answer,
                )
            )

    # ------------------------------------------------------------------
    # Wrapper that runs the coroutine in a *new* asyncio event loop
    # ------------------------------------------------------------------
    def _call_loop(self, task: str) -> None:
        asyncio.run(self._loop(task))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def join(self) -> None:
        """Block until the internal worker thread finishes."""
        if self._thread is not None:
            self._thread.join()

    def cancel(self) -> None:
        """Ask the running loop to stop at its next safe point."""
        self._cancel_event.set()

    def run(self, task: str, timeout_sec: float | None = None) -> None:
        """Start the agent loop.

        The method returns only after the loop finishes (or is cancelled).
        """
        # --------checking if agent is on work ------------------------------
        if self._thread is not None:
            raise RuntimeError("Agent has already been started")

        # ---- deadline -------------------------------------------------------
        if timeout_sec is not None:
            self.context.deadline = time.monotonic() + timeout_sec

        # ---- initialise shared state (run‑context “blackboard”) ------------
        self.context.state.setdefault("error", None)
        self.context.state.setdefault("final_answer", None)
        self.context.state.setdefault("iterations", 0)
        self.context.state.setdefault("tool_calls", 0)

        self._thread = threading.Thread(
            target=self._call_loop, args=(task,), daemon=True
        )
        self._thread.start()
