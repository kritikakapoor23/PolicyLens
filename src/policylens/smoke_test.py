import os
import sys

from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"
TIMEOUT_MS = 60_000


def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY is not set. Add it as a Codespaces secret and restart.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    response = client.models.generate_content(
        model=MODEL,
        contents="In one sentence, what is a health insurance waiting period?",
    )

    print(response.text)
    usage = response.usage_metadata
    print(f"prompt tokens:   {usage.prompt_token_count}")
    print(f"output tokens:   {usage.candidates_token_count}")
    print(f"thinking tokens: {usage.thoughts_token_count or 0}")
    print(f"total tokens:    {usage.total_token_count}")


if __name__ == "__main__":
    main()