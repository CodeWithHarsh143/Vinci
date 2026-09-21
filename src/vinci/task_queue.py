from dataclasses import dataclass
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import Any
import asyncio
import time


@dataclass
class Task:
    id: UUID
    name: str
    payload: dict
    created_at: datetime


@dataclass
class TaskResult:
    task_id: UUID
    status: str
    output: Any
    error: str | None
    duration_ms: float


class TaskQueue:
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.queue = asyncio.Queue()
        self.results: dict[UUID, TaskResult] = {}
        self.events: dict[UUID, asyncio.Event] = {}
        self.workers = []

    async def worker(self):
        while True:
            task: Task = await self.queue.get()
            start = time.perf_counter()
            try:
                result = await self.process(task)

                end = time.perf_counter()
                self.results[task.id] = TaskResult(
                    task_id=task.id,
                    status="success",
                    output=result,
                    error=None,
                    duration_ms=(end - start) * 1000,
                )

            except Exception as e:
                end = time.perf_counter()
                self.results[task.id] = TaskResult(
                    task_id=task.id,
                    status="failed",
                    output=None,
                    error=str(e),
                    duration_ms=(end - start) * 1000,
                )
            finally:
                self.events[task.id].set()
                self.queue.task_done()

    def start(self):
        self.workers = [
            asyncio.create_task(self.worker()) for _ in range(self.max_workers)
        ]

    async def submit(self, task: Task, timeout: float = 10) -> TaskResult:
        event = asyncio.Event()
        self.events[task.id] = event
        self.results[task.id] = TaskResult(
            task_id=task.id, status="pending", output=None, error=None, duration_ms=0.00
        )
        await self.queue.put(task)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task.id,
                status="timeout",
                output=None,
                error="Timeout Error",
                duration_ms=timeout,
            )

        return self.results[task.id]

    async def process(self, task: Task) -> dict:
        if task.payload.get("failed"):
            raise ValueError("Task failed intentionally")
        await asyncio.sleep(3)
        return {"Result": task.payload}

    async def shutdown(self):
        await self.queue.join()
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
