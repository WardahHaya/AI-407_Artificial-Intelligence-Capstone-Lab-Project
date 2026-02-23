# agent/graph.py
# LangGraph ReAct Agent for Buraq
# Builds the reasoning loop: agent_node → router → tool_node → loop back

import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from agent.tools import ALL_TOOLS

load_dotenv()

# ══════════════════════════════════════════
#  STEP 1 — Agent State
#  This is the memory passed between every node
#  add_messages appends new messages instead of
#  overwriting — preserves full conversation history
# ══════════════════════════════════════════
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ══════════════════════════════════════════
#  STEP 2 — LLM with Tools Bound
#  Binding tools lets the LLM emit tool_call
#  objects instead of plain text when it wants
#  to take an action
# ══════════════════════════════════════════
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ══════════════════════════════════════════
#  STEP 3 — System Prompt
#  This tells the LLM who it is and how to behave
# ══════════════════════════════════════════
SYSTEM_PROMPT = """You are Buraq, an intelligent Gmail assistant.
You help the user manage their email without opening Gmail.

You can:
- Read and summarise emails
- Search emails by keyword, sender, or date
- Draft professional emails using AI
- Send emails after user review and approval
- Generate daily email digests
- Detect spam and urgent emails
- Check for replies
- Search past emails semantically using the knowledge base

Critical rules:
1. NEVER send an email without explicit user approval first
2. Always use draft_email first, then send_reviewed_email only
   after user says 'send it' or 'yes send'
3. Be concise, helpful, and conversational
4. If unsure what the user wants, ask for clarification
"""


# ══════════════════════════════════════════
#  STEP 4 — Agent Node
#  The brain of the loop
#  Takes current state, calls LLM, returns next step
#  LLM either calls a tool or gives final answer
# ══════════════════════════════════════════
def agent_node(state: AgentState) -> AgentState:
    messages_with_system = (
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    response = llm_with_tools.invoke(messages_with_system)
    return {"messages": [response]}


# ══════════════════════════════════════════
#  STEP 5 — Tool Node
#  Executes whatever tool the LLM picked
#  Returns the result as a ToolMessage
# ══════════════════════════════════════════
tool_node = ToolNode(tools=ALL_TOOLS)


# ══════════════════════════════════════════
#  STEP 6 — Conditional Router
#  The logic gate that controls the loop
#  If LLM made tool calls → go to tool_node
#  If LLM gave final answer → go to END
# ══════════════════════════════════════════
def router(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


# ══════════════════════════════════════════
#  STEP 7 — Build the Graph
#  Wire everything together
# ══════════════════════════════════════════
def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Entry point
    graph.set_entry_point("agent")

    # Conditional edge from agent
    graph.add_conditional_edges(
        source="agent",
        path=router,
        path_map={
            "tools": "tools",
            "__end__": END,
        }
    )

    # After tool runs, always go back to agent
    graph.add_edge("tools", "agent")

    return graph.compile()


# Compiled agent — imported by app.py
buraq_agent = build_graph()


# ══════════════════════════════════════════
#  Public chat function
# ══════════════════════════════════════════
def chat(user_message: str, history: list = None) -> str:
    """
    Main entry point to talk to Buraq.
    Maintains conversation history for multi-turn context.
    """
    if history is None:
        history = []

    input_messages = history + [HumanMessage(content=user_message)]
    result = buraq_agent.invoke({"messages": input_messages})
    return result["messages"][-1].content


# ══════════════════════════════════════════
#  Quick CLI test
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("\nBuraq AI — Intelligent Gmail Agent")
    print("Fly Above Your Inbox")
    print("=" * 40)
    print("Type 'exit' to quit\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\nBuraq: thinking...\n")
        response = chat(user_input, history)
        print(f"Buraq: {response}\n")
        print("-" * 40)

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))