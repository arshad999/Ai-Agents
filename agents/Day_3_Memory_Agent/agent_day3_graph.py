from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from graph_logger import GraphLogger
from lc_tools import multiply,string_length


logger = GraphLogger()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [multiply, string_length]
tool_node = ToolNode(tools)

# -------- State --------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# -------- Reason Node --------
def reason(state: AgentState):
    print("\nMESSAGES TO LLM", state["messages"])
    logger.log("MESSAGES TO LLM", [m.content for m in state["messages"]])

    response = llm.bind_tools(tools).invoke(state["messages"])

    logger.log("LLM RESPONSE", response)
    return {"messages": [response]}

# -------- Graph --------
graph = StateGraph(AgentState)

graph.add_node("reason", reason)
graph.add_node("tools", tool_node)

graph.set_entry_point("reason")

# Conditional routing
graph.add_conditional_edges(
    "reason",
    tools_condition,  # if tool call → go tools
)

graph.add_edge("tools", "reason")  # after tool → think again
graph.add_edge("reason", END)      # if no tool → finish

app = graph.compile()

# 🔥 MEMORY STORE (persistent in runtime)
memory_state = {"messages": []}


# -------- CHAT LOOP --------
while True:
    user = input("\nYou: ")
    if user == "exit":
        break
    # add new user message to memory
    memory_state["messages"].append(HumanMessage(content=user))

    # invoke graph with full history
    memory_state = app.invoke(memory_state)

    # last message is assistant reply
    reply = memory_state["messages"][-1].content

    logger.log("FINAL ANSWER", reply)
    print("Agent:", reply)

