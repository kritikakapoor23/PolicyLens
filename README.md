# PolicyLens

Cited question answering over health-insurance policy documents, with evaluation, cost tracking and tracing built in from day one.

## Why this exists

Insurance policies are long PDFs full of conditions: waiting periods, exclusions, sub-limits, co-payments. A plain chatbot doesn't know a specific policy, makes things up, and can't point to where an answer came from. PolicyLens answers from the documents only and shows its source.

## How it will work

1. **Retrieve:** find the passages in the policy documents most relevant to the question.
2. **Answer:** give the model only those passages and ask it to answer from them.
3. **Cite:** return the answer with the page and clause it came from, and say "not found" when the documents don't answer it.

A second project, ClaimCheck, will build an agent on top of this to pre-check claim scenarios.

## Status (work in progress)

- [x] Project setup: Python 3.12, `uv`, pinned dependencies
- [x] Streaming LLM wrapper with timing, token counts, retries and clean errors (`src/policylens/llm.py`)
- [x] Hand-written tool-calling loop with a calculator and a stub policy search (`src/policylens/agent_loop.py`)
- [ ] Document ingestion, chunking and vector search
- [ ] Evaluation set and metrics
- [ ] API and deployment

## Run it

Requires a Gemini API key in the `GEMINI_API_KEY` environment variable (never commit it).

```bash
uv sync
uv run python -m policylens.llm
uv run python -m policylens.agent_loop
```

## Stack

Python 3.12, `uv`, Google `google-genai` SDK. LangChain and LangGraph arrive in Week 6.

## Results

Measured numbers will be added here as they are collected.