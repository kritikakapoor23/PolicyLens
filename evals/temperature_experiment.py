from collections import Counter

from policylens.llm import LLMError
from policylens.structured import extract_terms

TEXT = ("Pre-existing conditions are covered after a waiting period of 36 months." "A co-payment of 20% applies to every claim for insuredd persons aged 60 and above")
RUNS = 5
TEMPERATURE = [0.0, 1.0]

def main() -> None:
    for temp in TEMPERATURE:
        results = []
        failures = 0
        for _ in range(RUNS):
            try:
                results.append(extract_terms(TEXT, temperature=temp).model_dump_json())
            except LLMError:
                failures += 1
        counts = Counter(results)
        print(f"temperature {temp}: {len(counts)} distinct output(s), {failures} failure(s) in {RUNS} runs")
        for output, n in counts.most_common():
            print(f" {n}x {output}")

if __name__ == "__main__":
    main()