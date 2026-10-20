# PolicyLens

Cited question answering over health-insurance policy documents.

## Why this exists

Insurance policies are long PDFs full of conditions: waiting periods, exclusions, sub-limits, co-payments. A plain chatbot doesn't know a specific policy, makes things up, and can't show where an answer came from. PolicyLens answers from the documents only and shows its source.

## How it will work

1. **Retrieve:** find the passages in the policy documents most relevant to the question.
2. **Answer:** give the model only those passages and ask it to answer from them.
3. **Cite:** return the answer with the page and clause it came from, and say "not found" when the documents don't answer it.

## Built so far

- Project setup with pinned dependencies (Python 3.12, `uv`)
- Streaming LLM wrapper: timing, token counts, retries, clean errors (`src/policylens/llm.py`)
- Hand-written tool-calling loop with a calculator and a stub policy search (`src/policylens/agent_loop.py`)

## Next

Load real policy documents and search them (replacing the stub policy search).

## Run it

Needs a Gemini API key in the `GEMINI_API_KEY` environment variable. Never commit it.

```bash
uv sync
uv run python -m policylens.llm
uv run python -m policylens.agent_loop
```