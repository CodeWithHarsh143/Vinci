from datetime import datetime, timezone
import time
from uuid import uuid4
import pytest
from vinci.task_queue import TaskQueue, Task
import asyncio
import pytest_asyncio


@pytest_asyncio.fixture
async def taskQueue():
    queue = TaskQueue()
    queue.start()
    yield queue
    await queue.shutdown()


def create_task():
    return Task(
        id=uuid4(),
        name="test",
        payload={"key": "value"},
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_success_task(taskQueue):
    task = Task(
        id=uuid4(),
        name="sucess_test",
        payload={"failed": False},
        created_at=datetime.now(timezone.utc),
    )
    result = await taskQueue.submit(task)
    assert result.status == "success"


@pytest.mark.asyncio
async def test_failed_task(taskQueue):
    task = Task(
        id=uuid4(),
        name="failed_task",
        payload={"failed": True},
        created_at=datetime.now(timezone.utc),
    )
    result = await taskQueue.submit(task)
    assert result.status == "failed"


@pytest.mark.asyncio
async def test_concurrent_exexution(taskQueue):
    tasks = [create_task() for _ in range(5)]
    start = time.perf_counter()
    await asyncio.gather(*(taskQueue.submit(task) for task in tasks))
    end = time.perf_counter()
    assert (end - start) < 5


@pytest.mark.asyncio
async def test_timeout(taskQueue):
    task = create_task()
    result = await taskQueue.submit(task, timeout=1)
    assert result.status == "timeout"
