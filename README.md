# Vinci

An AI engineering platform built progressively, version by version. Each version solves one engineering problem and proves it with tests before the next begins. RAG, code review, memory, and evaluation are built as subsystems of this one project — not separate repos.

---

## Foundation — Complete

Agent foundation: LLM reasoning connected to real tool execution, proven end-to-end (user → agent → real tools → real LLM → answer) with 54 passing tests.

- [x] Async TaskQueue (workers, timeouts, graceful shutdown)
- [x] LLM client (`generate`, `generate_structured`, `chat`, tool calling)
- [x] Structured output with Pydantic validation + retry
- [x] `BaseTool` + `ToolResult` abstraction
- [x] `ReadFileTool` (path validation, size limits, chunked reads)
- [x] `SearchCodeTool` (ripgrep subprocess, exit-code handling)
- [x] `ToolsRegistry` (registration, OpenAI-compatible schemas, dispatch)
- [x] Agent loop (tool calls → execution → results → repeat)
- [x] Termination (final answer / max iterations / time limit)
- [x] Provider quirks handled (e.g. Gemini thought-signature echo)
- [x] 54 testcases with pytest
- [x] End-to-end demo with real API

---

## V0.1 — Production Agent Core (completed)

### Why this version?

The loop works, but failures are ad-hoc. Before RAG or code review get built on top, the runtime needs defined execution semantics — what happens on cancel, timeout, and tool failure — plus the logging and state to debug it.

### Build Checklist

- [x] Execution context
- [x] Cancellation propagation
- [x] Timeout propagation
- [x] Structured errors
- [x] Agent state management
- [x] Configuration
- [x] Structured logging
- [x] Integration tests
- [x] Failure handling

### Expected Outcome

An agent that fails predictably, shuts down cleanly, and explains what happened — ready for V0.2 Advanced AI Engineering to build on.

---

## V0.2 — Advanced AI Engineering (current)

### Why this version?

The runtime is solid, but Vinci still talks to a single hardcoded LLM provider. Production AI applications need provider portability, measurable token and cost usage, graceful model failure handling, and control over what enters the context window. This version builds that application layer.

Roadmap: `roadmap.txt` · Design: `docs/design/v0.2-provider-abstraction.md`

### Build Checklist

- [ ] Provider abstraction (`LLMProvider` protocol, error taxonomy)
- [ ] Multiple LLM providers (Gemini, OpenAI, Fake for tests)
- [ ] Model selection (model catalog: context window, pricing, capabilities)
- [x] Structured outputs (Pydantic validation + retry — landed early in the foundation)
- [ ] Token counting (usage metadata on every response, aggregated per run)
- [ ] Cost tracking (pricing table × usage)
- [ ] Fallback models (retryable vs fatal error classification)
- [ ] Context-window management (token-aware history trimming)
- [ ] Streaming responses
- [ ] Prompt management (templates, versioning)
- [ ] Conversation management (session history beyond a single run)

### Acceptance Criteria

- Providers can be switched without changing agent logic.
- Streaming works correctly.
- Token and cost usage is measurable.
- Model failures are handled gracefully.

### Expected Outcome

A flexible AI application layer: provider-portable, measurable, and failure-tolerant — ready for V0.3 Production RAG.

---

## Later Versions

See `roadmap.txt` for the full plan and detailed acceptance criteria.

- **V0.3 — Production RAG**: ingestion, chunking, embeddings, vector search, hybrid retrieval, reranking, citations, retrieval evaluation
- **V0.4 — Code Intelligence**: repository indexing, semantic code retrieval, multi-file context, code review agent
- **V0.5 — Evaluation & Observability**: execution traces, token/cost/latency metrics, agent and RAG evaluation
- **V0.6 — Persistent Task Execution**: PostgreSQL-backed tasks, retries, idempotency, crash recovery
- **V0.7 — Distributed Task Processing**: worker processes, acknowledgment, visibility timeouts, dead-letter queue, horizontal scaling
- **V0.8 — Event-Driven Architecture**: pub/sub, event persistence, replay, consumer groups
- **V0.9 — Production Infrastructure**: FastAPI, PostgreSQL, Redis, JWT auth, rate limiting, Docker, deployment
- **V1.0 — Production Vinci**: complete platform with documentation, tests, and reproducible deployment

A fully‑featured AI engineering platform where RAG and code review are built on top of a stable, configurable agent core, with measurable quality and performance guarantees.
