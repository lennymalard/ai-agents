from langgraph.graph import StateGraph, END
from langchain.tools import tool
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from IPython.display import Image, display
import logging

logging.basicConfig(level=logging.INFO)

import requests

@tool
def duckduckgo_search(query: str) -> str:
    """
    A search engine.
    """
    url = f"https://api.duckduckgo.com"
    params = {
        "q": query,
        "format": "json"
    }
    response = requests.get(url, params=params)
    return response.json()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
        self.model = model.bind_tools(tools)
        self.system = system
        self.tools = {t.name: t for t in tools}
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_ollama)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.action_exists,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def call_ollama(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool in tool_calls:
            logging.info(f"Calling {tool}")
            result = self.tools[tool["name"]].invoke(tool["args"])
            results.append(ToolMessage(tool_call_id=tool["id"], name=tool["name"], content=str(result)))
        return {"messages": results}

    def action_exists(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        return len(tool_calls) > 0

system = """
You are a highly efficient search assistant. 
When given a query, retrieve accurate, up-to-date information from reliable sources. 
Summarize results clearly and concisely, provide relevant links when possible, and highlight key points. 
If the query is ambiguous, ask clarifying questions before searching.
"""

tools = [duckduckgo_search]

model = ChatOllama(model="PetrosStav/gemma3-tools:27b", temperature=0.5)
agent = Agent(model=model, tools=tools, system=system)

dot_source = agent.graph.get_graph().draw_mermaid_png()
with open("react_agent_graph.png", "wb") as f:
    f.write(dot_source)

messages = [HumanMessage("What is python ? Use duckduckgo_search.")]
print(agent.graph.invoke({"messages": messages})["messages"][-1].content)