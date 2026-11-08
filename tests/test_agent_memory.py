"""Tests for the Vinci agent's save_state / load_state functionality.

These tests verify that the :class:`vinci.agents.base.Agent` can persist its
internal state (``RunContext.memory`` and ``RunContext.history``) to a JSON file
and later restore it.

"""

import pytest
import json
from vinci.agents.base import Agent
from vinci.llm.client import CallResponse


# ----------------------------------------------------------------------
# Test 1 – save and load a basic state
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_save_state_basic():
    """Storing and retrieving the agent’s state works."""

    class FakeLLM:
        async def call(self, messages, tools):
            return CallResponse(
                content="Task completed successfully",
                tool_calls=[],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            raise AssertionError("No tool should be executed")

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    agent.run("Do something")
    agent.join()
    saved_path = agent.save_state()  # created a json file for this session
    assert saved_path is not None  # insuring if path(or file ) exists
    assert saved_path.endswith(".json")  # insuring its json file

    # Load the state back into the same agent
    success = agent.load_state(saved_path)
    assert success is True

    # The agent’s state is restored from the saved file (memory/history reflect what was saved)
    assert agent.context.memory == {}
    # History is restored from the file, not cleared
    assert len(agent.context.history) > 0

    # Also verify the file contents are valid JSON
    with open(saved_path, "r") as f:
        data = json.load(f)
    # checking if these keys exits in the json or not
    assert "run_id" in data
    assert "memory" in data
    assert "history" in data


# ----------------------------------------------------------------------
# 2️⃣  Test saving and loading with custom memory / history values
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_save_load_restore():
    """Storing and loading state preserves custom data added to memory/history."""

    class FakeLLM:
        async def call(self, messages, tools):
            return CallResponse(
                content="Task completed successfully",
                tool_calls=[],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            raise AssertionError("No tool should be executed")

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    agent.run("Do something")
    agent.join()

    # Store custom data before saving
    agent.context.memory["custom_key"] = "custom_value"
    agent.context.history.append({"role": "user", "content": "test query"})
    agent.context.history.append({"role": "assistant", "content": "test answer"})

    # Save
    saved_path = agent.save_state()
    assert saved_path is not None

    # Load into a fresh agent
    new_agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    loaded = new_agent.load_state(saved_path)
    assert loaded is True

    # Verify the restored data (history is restored entirely from the saved file)
    assert new_agent.context.memory.get("custom_key") == "custom_value"
    # History contains all entries from the saved file, including the original "Do something"
    assert len(new_agent.context.history) == 3


# ----------------------------------------------------------------------
# 3️⃣  Test saving and loading after multiple runs (cycle test)
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_save_load_cycle():
    """Running the agent multiple times and saving after each run produces
    independent state files."""

    class FakeLLM:
        async def call(self, messages, tools):
            return CallResponse(
                content="Task completed successfully",
                tool_calls=[],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            raise AssertionError("No tool should be executed")

    # First run
    agent1 = Agent(llm=FakeLLM(), registry=FakeRegistry())
    agent1.run("first task")
    agent1.join()
    saved1 = agent1.save_state()
    assert saved1 is not None

    # Second run
    agent2 = Agent(llm=FakeLLM(), registry=FakeRegistry())
    agent2.run("second task")
    agent2.join()
    saved2 = agent2.save_state()
    assert saved2 is not None
    assert saved2 != saved1  # different run_id → different file name

    # Load the first state into a fresh agent and verify its data
    loaded1 = agent1.load_state(saved1)
    assert loaded1 is True
    assert saved1 is not None  # sanity

    # Load the second state into a fresh agent and verify
    loaded2 = agent2.load_state(saved2)
    assert loaded2 is True
