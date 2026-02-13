from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from schemas import UserIntent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---- State ----
class AgentState(TypedDict):
    messages: List[BaseMessage]
    structured_output: dict


# ---- Structured Node ----
def understand_user(state: AgentState):

    structured_llm = llm.with_structured_output(UserIntent)

    result = structured_llm.invoke(state["messages"])

    return {
        "messages": state["messages"],
        "structured_output": result.model_dump()
    }


# ---- Graph ----
graph = StateGraph(AgentState)
graph.add_node("understand", understand_user)
graph.set_entry_point("understand")

app = graph.compile()


# ---- Run ----
while True:
    user = input("\nYou: ")
    if user == "exit":
        break
    result = app.invoke({
        "messages": [HumanMessage(content=user)]
    })

    print("\nStructured Output:")
    print(result["structured_output"])
