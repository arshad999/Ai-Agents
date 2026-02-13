from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from plan_schema import Plan
from lc_tools import multiply, string_length
from graph_logger import GraphLogger

logger = GraphLogger()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [multiply, string_length]
tool_node = ToolNode(tools)


# ----- STATE -----
class AgentState(TypedDict):
   
    messages: Annotated[List[BaseMessage], add_messages]
    plan: List[str]
    current_step: int


# ----- PLAN NODE -----
def create_plan(state: AgentState):
    planner = llm.with_structured_output(Plan)

    user_goal = state["messages"][-1].content
    logger.log("Goal", user_goal)

    plan = planner.invoke(
        f"Create a minimal step-by-step plan to solve: {user_goal}. "
        f"Use available tools: multiply, string_length"
    )

    logger.log("Generated Plan", plan.steps)

    return {
        "plan": plan.steps,
        "current_step": 0
    }


# ----- EXECUTE STEP -----
def execute_step(state: AgentState):

    if state["current_step"] >= len(state["plan"]):
        return {"messages": state["messages"]}


    step = state["plan"][state["current_step"]]
    logger.log("Execute Step", step)

    response = llm.bind_tools(tools).invoke(
        state["messages"] + [HumanMessage(content=f"Execute step: {step}")]
    )
    if response.content:
        logger.log("Step Result", response.content)
    elif response.tool_calls:
        logger.log("Step Result", f"Tool Calls: {response.tool_calls}")
    else:
        logger.log("Step Result", "No Content")

    return {
        "messages": state["messages"] + [response],
        "current_step": state["current_step"] + 1
    }




# ----- SHOULD CONTINUE -----
def should_continue(state: AgentState):
    if state["messages"] and hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls:
        return "continue"
    if state["current_step"] >= len(state["plan"]):
        return "end"
    return "continue"


# ----- GRAPH -----
graph = StateGraph(AgentState)

graph.add_node("planner", create_plan)
graph.add_node("execute", execute_step)
graph.add_node("tools", tool_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "execute")

graph.add_conditional_edges(
    "execute",
    should_continue,
    {"continue": "tools", "end": END}
)

graph.add_edge("tools", "execute")

app = graph.compile()


# ----- RUN -----
while True:
    user = input("\nYou: ")
    if user == "exit":
        break
    result = app.invoke({
        "messages": [HumanMessage(content=user)],
        "plan": [],
        "current_step": 0
    })

    print("\nAgent:", result["messages"][-1].content)
    logger.log("Final Answer", result["messages"][-1].content)
