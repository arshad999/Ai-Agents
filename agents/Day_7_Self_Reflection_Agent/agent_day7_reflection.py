from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition    
from reflection_schema import Critique
from lc_tools import multiply, string_length
from graph_logger import GraphLogger

logger = GraphLogger()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [multiply, string_length]
tool_node = ToolNode(tools)


# ----- STATE -----
class AgentState(TypedDict):

    messages: Annotated[List[BaseMessage], add_messages]
    attempts: int


# ----- SOLVER -----
def solve(state: AgentState):
    logger.log("SOLVE NODE", f"Current Message Count: {len(state['messages'])}")
    response = llm.bind_tools(tools).invoke(state["messages"])
    logger.log("SOLVE RESPONSE", response.content)
    return {"messages": state["messages"] + [response]}


# ----- CRITIC -----
def critique(state: AgentState):
    logger.log("CRITIQUE NODE", "Analyzing response...")

    reviewer = llm.with_structured_output(Critique)

    last_answer = state["messages"][-1].content
    question = state["messages"][0].content

    result = reviewer.invoke(
        f"Question: {question}\nAnswer: {last_answer}\n"
        f"Is the answer correct? If not explain why."
    )
    logger.log("CRITIQUE RESULT", {"correct": result.correct, "feedback": result.feedback})

    return {
        "messages": state["messages"] + [
            AIMessage(content=f"CRITIQUE: {result.feedback}")
        ],
        "attempts": state["attempts"] + (0 if result.correct else 1),
        "is_correct": result.correct
    }


# ----- ROUTER -----
def reflection_router(state: AgentState):
    if state.get("is_correct", False) or state["attempts"] >= 2:
        logger.log("ROUTER DECISION", "END (Correct or Max Attempts Reached)")
        return "end"
    logger.log("ROUTER DECISION", "RETRY (Incorrect output)")
    return "retry"


# ----- GRAPH -----
graph = StateGraph(AgentState)

graph.add_node("solve", solve)
graph.add_node("tools", tool_node)
graph.add_node("critique", critique)

graph.set_entry_point("solve")

graph.add_conditional_edges("solve", tools_condition, {"tools": "tools", "__end__": "critique"})
graph.add_edge("tools", "solve")

graph.add_conditional_edges("critique", reflection_router, {"retry": "solve", "end": END})

app = graph.compile()


# ----- RUN -----
while True:
    user = input("\nYou: ")
    if user == "exit":
        break
    result = app.invoke({
        "messages": [HumanMessage(content=user)],
        "attempts": 0
    })

    print("\nAgent:", result["messages"][-1].content)
