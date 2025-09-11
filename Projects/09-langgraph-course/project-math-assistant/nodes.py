from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

SYSTEM_PROMPT = """
You are a helpful assistant to solve some math problems.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    response = llm.invoke([
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *state["messages"]
    ])

    return {"messages": [response]}


tool_node = ToolNode(tools)