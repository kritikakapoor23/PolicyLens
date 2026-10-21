import sys
 
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

from policylens.llm import MODEL, LLMError, _get_client

MAX_VALIDATION_ATTEMPTS = 3

class PolicyTerms(BaseModel):
    waiting_period_months: int | None = Field(
        default=None, ge=0, description="Waiting period in months, null if not stated"
    )
    copayment_percent: float | None = Field(
        default=None, ge=0, le=100, description="Co-payment as a percentage of the claim, null if not stated"
    )
    copayment_min_age: int | None = Field(
        default=None, ge=0, description="Age from which the co-payment applies, null if not stated"
    )

def extract_terms(text: str,temperature: float | None = None, client=None) -> PolicyTerms:
    client = client or _get_client()
    config = types.GenerateContentConfig(response_mime_type = "application/json", response_schema = PolicyTerms, temperature = temperature,)
    prompt = f"Extract the policy terms from the text below. Use null for anything not stated. \n\nTEXT:\n{text}"
    contents = prompt
    last_error = ""
    for attempt in range(1, MAX_VALIDATION_ATTEMPTS+1):
        try:
            response = client.models.generate_content(model = MODEL, contents = contents, config = config)
        except errors.APIError as exc:
            raise LLMError(f"{exc.code}: {exc.message}") from exc
        try:
            return PolicyTerms.model_validate_json(response.text)
        except ValidationError as exc:
            last_error = str(exc)
            print(f"[validation attempt {attempt}/{MAX_VALIDATION_ATTEMPTS}] {exc.error_count()} problem(s)", file = sys.stderr)
            contents = (f"{prompt}\n\nYour previous answer was: \n{response.text}\n\n" f"It failed validation:\n{last_error}\nReturn corrected JSON only")
    raise LLMError(f"no valid output after {MAX_VALIDATION_ATTEMPTS} attempts: {last_error}")

def main() -> None:
    text = ("Pre-existing conditions are covered after a waiting period of 36 months" "A co-payment of 20% applies to every claim for insured persons aged 60 and above")
    try:
        print(extract_terms(text).model_dump_json(indent=2))
    except LLMError as exc:
        sys.exit(f"Error:{exc}")

if __name__ == "__main__":
    main()