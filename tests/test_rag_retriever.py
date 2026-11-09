"""Hermetic tests for the agent's ``@search:`` path.

``@search: <query>`` must route through the retriever and answer from the
retrieved snippets without executing any tools. These tests use fake
LLM/registry/retriever implementations and never touch the network.
"""

from vinci.agents.base import Agent
from vinci.llm.client import CallResponse

FAKE_ANSWER = "login validates the user and returns a token"


class FakeLLM:
    async def call(self, messages, tools):
        return CallResponse(content=FAKE_ANSWER, tool_calls=[])


class FakeRegistry:
    def list_schemas(self):
        return []

    async def execute(self, tool_name, args):
        raise AssertionError("No tool should be executed")


class FakeRetriever:
    def __init__(self, snippets: list[dict]) -> None:
        self.snippets = snippets
        self.queries: list[str] = []

    def query(self, query: str) -> list[dict]:
        self.queries.append(query)
        return self.snippets


def _make_agent(retriever: FakeRetriever) -> Agent:
    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    agent.retriever = retriever
    return agent


def test_search_path_returns_rag_answer():
    retriever = FakeRetriever(
        [{"path": "auth.py", "excerpt": "def login(user): return token", "score": 1.0}]
    )
    agent = _make_agent(retriever)

    agent.run("@search: how does login work?")
    agent.join()
    result = agent.result_queue.get(timeout=5)

    assert result.success is True
    assert result.final_answer == FAKE_ANSWER
    assert retriever.queries == ["how does login work?"]


def test_search_path_no_snippets_fails():
    retriever = FakeRetriever([])
    agent = _make_agent(retriever)

    agent.run("@search: how does login work?")
    agent.join()
    result = agent.result_queue.get(timeout=5)

    assert result.success is False
    assert result.error == "no snippets obtained"
