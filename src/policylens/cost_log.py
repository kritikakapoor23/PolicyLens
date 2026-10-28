import json
import time
from dataclasses import asdict
from pathlib import Path

LOG_PATH = Path("logs/llm_calls.jsonl")

# PLACEHOLDER prices in USD per 1 million tokens. Replace with the official prices.
PRICES = {"gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50}}


def estimate_cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> float:
    price = PRICES[model]
    return (
        prompt_tokens / 1000000 * price["input"]
        + output_tokens / 1000000 * price["output"]
    )


def log_call(label: str, model: str, result) -> None:
    numbers = {key: value for key, value in asdict(result).items() if key != "text"}
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "model": model,
        **numbers,
        "cost_usd": estimate_cost_usd(
            model, result.prompt_tokens, result.output_tokens + result.thinking_tokens
        ),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def summarize() -> None:
    if not LOG_PATH.exists():
        print("No log yet.")
        return
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line]
    if not records:
        print("No log yet.")
        return
    total_cost = sum(r["cost_usd"] for r in records)
    avg_latency = sum(r["total_s"] for r in records) / len(records)
    print(
        f"{len(records)} calls | total cost ${total_cost: .6f} | avg total time {avg_latency: .2f}s"
    )


if __name__ == "__main__":
    summarize()
