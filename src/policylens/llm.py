import os
import random
import sys
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors, types

from policylens.cost_log import log_call

MODEL = os.environ.get("POLICYLENS_MODEL", "gemini-3.5-flash-lite")

TIMEOUT_MS = 60000
MAX_ATTEMPTS = 3
RETRYABLE_CODE = {429, 500, 503, 504}


class LLMError(Exception):
    """Raised when a model call fails in a way we can report cleanly."""


@dataclass
class Result:
    text: str
    ttft_s: float
    total_s: float
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int
    total_tokens: int


def _get_client() -> genai.Client:
    if not os.environ.get("GEMINI_API_KEY"):
        raise LLMError("GEMINI_API_KEY is not set, add and restart")
    return genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))


def _stream_once(client: genai.Client, prompt: str) -> Result:
    start = time.perf_counter()
    ttft = None
    parts: list[str] = []
    usage = None
    for chunk in client.models.generate_content_stream(model=MODEL, contents=prompt):
        if chunk.text:
            if ttft is None:
                ttft = time.perf_counter() - start
            print(chunk.text, end="", flush=True)
            parts.append(chunk.text)
        if chunk.usage_metadata:
            usage = chunk.usage_metadata
    print()
    total = time.perf_counter() - start
    return Result(
        text="".join(parts),
        ttft_s=ttft if ttft is not None else total,
        total_s=total,
        prompt_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        thinking_tokens=(usage.thoughts_token_count or 0) if usage else 0,
        total_tokens=(usage.total_token_count or 0) if usage else 0,
    )


def stream_answer(prompt: str, client: genai.Client | None = None) -> Result:
    client = client or _get_client()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _stream_once(client, prompt)
        except errors.APIError as exc:
            if exc.code not in RETRYABLE_CODE:
                raise LLMError(f"{exc.code}: {exc.message}") from exc
            if attempt == MAX_ATTEMPTS:
                raise LLMError(
                    f"{exc.code} after {MAX_ATTEMPTS} attempts: {exc.message}"
                ) from exc
            wait = 2 ** (attempt - 1) + random.random()
            print(
                f"[retry {attempt}/{MAX_ATTEMPTS}] error {exc.code}; waiting {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)

    raise LLMError("unreachable")


def main() -> None:
    prompt = "In one sentence, what is a health insurance waiting period?"
    try:
        r = stream_answer(prompt)
        log_call("smoke", MODEL, r)
    except LLMError as exc:
        sys.exit(f"Error: {exc}")
    print(
        f"TTFT {r.ttft_s:.2f}s | total {r.total_s:.2f}s | token in/out/think/total "
        f"{r.prompt_tokens}/{r.output_tokens}/{r.thinking_tokens}/{r.total_tokens}"
    )


if __name__ == "__main__":
    main()
