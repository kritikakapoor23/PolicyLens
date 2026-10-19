"""Hand-written tool-calling loop (no framework)."""

import ast
import operator
import sys

from google.genai import errors, types

from policylens.llm import MODEL, LLMError, _get_client

MAX_STEPS = 5

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '0.2 * 120000'. Supports + - * / and parentheses."""
    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree.body))


def search_policy(query: str) -> str:
    """Search the insurance policy documents and return the most relevant passage."""
    return "Co-payment: 20% of every admissible claim applies to insured persons aged 60 and above."


TOOLS = {"calculator": calculator, "search_policy": search_policy}


def _run_tool(name: str, args: dict | None) -> dict:
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool '{name}'. Available: {', '.join(TOOLS)}"}
    try:
        return {"result": tool(**(args or {}))}
    except Exception as exc:  # noqa: BLE001 - tool failures go back to the model as text
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_agent(question: str, client=None) -> str:
    client = client or _get_client()
    config = types.GenerateContentConfig(
        tools=list(TOOLS.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    for step in range(1, MAX_STEPS + 1):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )
        calls = response.function_calls
        if not calls:
            return response.text
        contents.append(response.candidates[0].content)
        parts = []
        for call in calls:
            print(f"[step {step}] model wants {call.name}({call.args})")
            result = _run_tool(call.name, call.args)
            print(f"[step {step}] tool returned {result}")
            parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )
        contents.append(types.Content(role="user", parts=parts))
    raise LLMError(f"no final answer after {MAX_STEPS} steps")


def main() -> None:
    question = "According to the policy, what is the co-payment on an admissible claim of Rs 120000?"
    try:
        print(run_agent(question))
    except (LLMError, errors.APIError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
