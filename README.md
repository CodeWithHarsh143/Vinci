# Vinci

An AI engineering platform built progressively, version by version. Each version solves one engineering problem and proves it with tests before the next begins.

---

## V0.1 — Agent Foundation (current)

### Why this version?

Vinci needs a reliable execution loop — LLM reasoning connected to real tool execution with timeouts, retries, and cancellation — before any advanced AI capability is added.

### Built

- [x] Async TaskQueue (workers, timeouts, graceful shutdown)
- [x] LLM client (`generate`, `generate_structured`, `chat`)
- [x] Structured output with Pydantic validation + retry
- [x] Tool calling (`CallResponse`, `ToolCallRequest`)
- [x] `BaseTool` + `ToolResult` abstraction
- [x] `ReadFileTool` (path validation, size limits, chunked reads)
- [x] `SearchCodeTool` (ripgrep subprocess, exit-code handling)
- [x] `ToolsRegistry` (registration, schema generation, dispatch)
- [x] Agent loop (tool calls → execution → results → repeat)
- [x] Termination (final answer / max iterations / time limit)
- [x] 43 passing tests


- [x] Fix `llm.call` vs `llm.chat` mismatch
- [x] Add missing `durations_ms` on success path
- [x] Rename `tool_id` → `tool_call_id`
- [x] Convert assistant messages to OpenAI-compatible dictionaries
### Cleanup before exit
- [ ] End-to-end demo (user → agent → real tool → real LLM → answer)

### Next

V0.2 hardens the loop into production-grade execution semantics: cancellation, timeouts, structured errors, logging, and state.

---

## V0.2 — Production Agent Core (next)

### Why is this needed?

The loop works, but failures are ad-hoc. Vinci needs defined execution semantics — what happens on cancel, timeout, and tool failure — before growing new capabilities.

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

An agent that fails predictably, shuts down cleanly, and explains what happened.
