# Concepts learned

## Day 1: Calling an LLM reliably (`llm.py`, `smoke_test.py`)

- **Concept:** An LLM call has a cost measured in tokens (prompt, output, thinking). Streaming shows the answer as it arrives, so the first token matters more than the total time (TTFT). Temporary errors (503, 429, 504) are retried with waits that grow (1s, 2s, 4s) plus a little randomness. Permanent errors (400, 404) fail immediately. Every call has a timeout.
- **What breaks without it:** the program hangs forever on a stuck request, crashes when the server is busy, or wastes time retrying errors that can never succeed.

## Day 2: The tool-calling loop (`agent_loop.py`)

- **Concept:** The model never runs anything. It asks for a tool, my code runs it, and the result goes back to the model. This repeats until the model gives a final answer. A step limit stops it. A failing tool returns its error as text instead of crashing the loop. Text from the model is never run with `eval`.
- **What breaks without it:** infinite or expensive loops, crashes when a tool fails, and security holes from running code the model wrote.

## Day 3: Structured output and validation (`structured.py`)

- **Concept:** Ask the model for fields in a fixed JSON shape and validate them with Pydantic (right types, valid ranges). If validation fails, retry and send back the model's own answer plus the exact error. Validation checks the shape of the output, not whether it is true, so inputs are checked too (for example, reject empty text before calling the model). Temperature controls randomness: on a clear snippet the output was identical across 5 runs at 0.0 and 1.0.
- **What breaks without it:** malformed, out-of-range or empty data reaches the rest of the program silently.

## Day 4: LangChain agents and tool-error middleware (`agent_lc.py`)

- **Concept:** LangChain's `create_agent` runs the tool loop for me, built on LangGraph. Messages, tools and the model each have a standard shape. By default, an error inside a tool crashes the whole run. Middleware (`@wrap_tool_call`) wraps every tool call: it receives the model's request, calls `handler(request)` to run the real tool, and if that fails it returns a `ToolMessage` with the error and the same `tool_call_id`, so the model can read it. I write that once instead of putting `try/except` in each tool.
- **What breaks without it:** one failing tool (for example 1/0) crashes the whole agent.
- **In LangChain:** `init_chat_model` (with `timeout` in seconds and `max_retries`), `@tool`, `create_agent`, `recursion_limit`, `@wrap_tool_call`, `ToolMessage`.