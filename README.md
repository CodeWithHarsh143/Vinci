# Vinci

An AI engineering platform built progressively, version by version. Each version solves one engineering problem and proves it with tests before the next begins. RAG, code review, memory, and evaluation are built as subsystems of this one project — not separate repos.

---

## Foundation-Complete

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
- [x] End-to-end demo with real API

---

## V0.1 — Production Agent Core  (current)

### Why this version?

The loop works, but failures are ad-hoc. Before RAG or code review get built on top, the runtime needs defined execution semantics — what happens on cancel, timeout, and tool failure — plus the logging and state to debug it.

### Build Checklist

- [ ] Execution context
- [ ] Cancellation propagation
- [ ] Timeout propagation
- [ ] Structured errors
- [ ] Agent state management
- [ ] Configuration
- [ ] Structured logging
- [ ] Integration tests
- [ ] Failure handling

### Expected Outcome

An agent that fails predictably, shuts down cleanly, and explains what happened — ready for RAG (V0.2) to build on.
