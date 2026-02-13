from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from lc_tools import multiply, string_length
from rag_tool import search_docs

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [multiply, string_length, search_docs]
tool_node = ToolNode(tools)

class AgentState(TypedDict):
    # messages: List[BaseMessage]
    messages: Annotated[List[BaseMessage], add_messages]

def reason(state: AgentState):
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

graph = StateGraph(AgentState)

graph.add_node("reason", reason)
graph.add_node("tools", tool_node)

graph.set_entry_point("reason")

graph.add_conditional_edges(
    "reason",
    tools_condition,
    {"tools": "tools", "__end__": "__end__"},
)

graph.add_edge("tools", "reason")

app = graph.compile()

memory = {"messages": []}

while True:
    user = input("\nYou: ")
    if user == "exit":
        break
    memory["messages"].append(HumanMessage(content=user))
    memory = app.invoke(memory)
    print("Agent:", memory["messages"][-1].content)
