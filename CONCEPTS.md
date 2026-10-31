# Concepts learned

## Day 1: Calling an LLM reliably (`llm.py`, `smoke_test.py`)

- **Concept:** An LLM call has a cost measured in tokens (prompt, output, thinking). Streaming shows the answer as it arrives, so the first token matters more than the total time (TTFT). Temporary errors (503, 429, 504) are retried with waits that grow (1s, 2s, 4s) plus a little randomness. Permanent errors (400, 404) fail immediately. Every call has a timeout.
- **What breaks without it:** the program hangs forever on a stuck request, crashes when the server is busy, or wastes time retrying errors that can never succeed.
- **In LangChain:** `init_chat_model` with `timeout` (seconds) and `max_retries`, plus `.stream()` for streaming.

## Day 2: The tool-calling loop (`agent_loop.py`)

- **Concept:** The model never runs anything. It asks for a tool, my code runs it, and the result goes back to the model. This repeats until the model gives a final answer. A step limit stops it. A failing tool returns its error as text instead of crashing the loop. Text from the model is never run with `eval`.
- **What breaks without it:** infinite or expensive loops, crashes when a tool fails, and security holes from running code the model wrote.
- **In LangChain:** `@tool` and `create_agent`; `recursion_limit` is the step cap.

## Day 3: Structured output and validation (`structured.py`)

- **Concept:** Ask the model for fields in a fixed JSON shape and validate them with Pydantic (right types, valid ranges). If validation fails, retry and send back the model's own answer plus the exact error. Validation checks the shape of the output, not whether it is true, so inputs are checked too (for example, reject empty text before calling the model). Temperature controls randomness: on a clear snippet the output was identical across 5 runs at 0.0 and 1.0.
- **What breaks without it:** malformed, out-of-range or empty data reaches the rest of the program silently.
- **In LangChain:** structured output (not used yet; we wrote the validation by hand with Pydantic).

## Day 4: LangChain agents and tool-error middleware (`agent_lc.py`)

- **Concept:** LangChain's `create_agent` runs the tool loop for me, built on LangGraph. Messages, tools and the model each have a standard shape. By default, an error inside a tool crashes the whole run. Middleware (`@wrap_tool_call`) wraps every tool call: it receives the model's request, calls `handler(request)` to run the real tool, and if that fails it returns a `ToolMessage` with the error and the same `tool_call_id`, so the model can read it. I write that once instead of putting `try/except` in each tool.
- **What breaks without it:** one failing tool (for example 1/0) crashes the whole agent.
- **In LangChain:** `init_chat_model` (with `timeout` in seconds and `max_retries`), `@tool`, `create_agent`, `recursion_limit`, `@wrap_tool_call`, `ToolMessage`.

## Day 5: Seeing what the agent does (`cost_log.py`, Langfuse tracing in `agent_lc.py`)

- **Concept:** A trace is the full record of one request. It contains steps; a generation is a step that is a model call, with its tokens, time and cost. Cost is tokens times price: input costs $0.30 per million tokens and output costs $2.50, so output is about 8 times more expensive. Every model call re-sends the whole conversation, so the input grows each step (148, 207, then 245 tokens in my first trace). I keep a local cost log (one JSON line per model call) and send traces to Langfuse. Tracing is optional: with the Langfuse keys the agent sends a trace, without them it still runs. `flush()` sends waiting data before a short program exits.  Example: 245 input tokens + 80 output tokens cost 245/1,000,000 × 0.30 + 80/1,000,000 × 2.50 = $0.000274.
- **What breaks without it:** I can't tell why a request was slow or expensive. "40 seconds" means nothing until I can see it was three model calls of about 12, 12 and 17 seconds, and that the tools took no time.
- **In LangChain / Langfuse:** `CallbackHandler`, `config={"callbacks": [handler]}`, `get_client().flush()`, and the secrets `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.
- **First baseline:** 3 model calls, 2 tool calls, 727 tokens (600 in, 127 out), 40.8 s, $0.000498, from one run (see `RESULTS.md`).
