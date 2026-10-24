import os
import sys
from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest

from policylens.agent_loop import calculator, search_policy
from policylens.llm import MODEL

calculator_tool = tool(calculator)
search_policy_tool = tool(search_policy)


@wrap_tool_call
def handle_tool_errors(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]
) -> ToolMessage:
    try:
        return handler(request)
    except Exception as exc:  # noqa: BLE001 - any tool failure goes back to the model
        return ToolMessage(
            content=f"Tool error: {type(exc).__name__}:{exc}",
            tool_call_id=request.tool_call["id"],
        )


def build_agent(model=None):
    if model is None:
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("Error: GEMINI_API_KEY is not set.")
        model = init_chat_model(
            f"google_genai:{MODEL}",
            api_key=os.environ["GEMINI_API_KEY"],
            timeout=60,
            max_retries=3,
        )
    return create_agent(
        model=model,
        tools=[calculator_tool, search_policy_tool],
        system_prompt="Answer using the tools. Use calculator for any arithmetic.",
        middleware=[handle_tool_errors],
    )


def run(question: str, agent=None) -> dict:
    agent = agent or build_agent()
    return agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 11},
    )


def main() -> None:
    question = "According to the policy, what is the co-payment on an admissible claim of Rs 120000?"
    result = run(question)
    for message in result["messages"]:
        calls = getattr(message, "tool_calls", None)
        print(
            f"[{message.type}] {message.text if message.text else ''} {calls if calls else ''}"
        )
    print("\nFINAL:", result["messages"][-1].text)


if __name__ == "__main__":
    main()
