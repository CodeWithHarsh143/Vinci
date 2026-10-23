# Vinci

An AI engineering platform built progressively, version by version. Each version solves one engineering problem and proves it with tests before the next begins. RAG, code review, memory, and evaluation are built as subsystems of this one project — not separate repos.

---

## Foundation — Complete ✅ (M0–M3)

Agent foundation: LLM reasoning connected to real tool execution, proven end-to-end (user → agent → real tools → real LLM → answer) with 45 passing tests.

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
- [x] 45 testcases with pytest 
- [x] End-to-end demo with real API

---

## V0.1 — Production Agent Core ✅ (completed)

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

An agent that fails predictably, shuts down cleanly, and explains what happened — ready for RAG (V0.2) to build on.

---

## V0.2 — AI Engineering 🚧 (next)

### Why this version?

The runtime is solid, but advanced AI capabilities (retrieval‑augmented generation, code review, deterministic evaluation) require a defined pipeline and quality gates.

### Build Checklist

- [ ] RAG pipeline (retriever → reranker → answer generation)
- [ ] Code review subsystem (automated static analysis + LLM findings)
- [ ] Memory management (persistent context, summarisation)
- [ ] Evaluation harness (reward models, success rate, failure modes)
- [ ] Prompt engineering framework (few‑shot, chain‑of‑thought, tool use)
- [ ] Human‑in‑the‑loop feedback loop
- [ ] Performance benchmarking (latency, cost, token usage)

### Expected Outcome

A fully‑featured AI engineering platform where RAG and code review are built on top of a stable, configurable agent core, with measurable quality and performance guarantees.