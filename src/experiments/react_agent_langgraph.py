from langgraph.graph import StateGraph, END
from langchain.tools import tool
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessageChunk
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
import logging
import requests
from ddgs import DDGS
from bs4 import BeautifulSoup
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)

API_KEY_PATH = "../api_keys/openrouter.txt"
API_KEY = open(API_KEY_PATH, "r").readline()

"""@tool
def duckduckgo_search(query: str) -> str:
    '''
    A search engine.
    '''
    url = f"https://api.duckduckgo.com"
    params = {
        "q": query,
        "format": "json"
    }
    response = requests.get(url, params=params)
    return response.json()"""

ddg = DDGS()

def ddg_search(query: str, max_results=3):
    return ddg.text(query, max_results=max_results)

def scrape_website(url: str):
    if not url:
        raise ValueError("The URL is missing.")
    headers = {'User-Agent': 'Mozilla/5.0'}
    html = requests.get(url, headers=headers).text
    return BeautifulSoup(html, parser="lxml", features="lxml")

def extract_text(soup):
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return soup.get_text(separator='\n', strip=True)

@tool
def search_tool(query: str, max_results: int = 3):
    """
    A web search engine.
    Returns a list of tuples: (url, snippet).
    Use this to find relevant links, but DO NOT use the snippet as the final answer.
    """
    results = ddg_search(query=query, max_results=max_results)
    urls = [(result["href"], result["body"]) for result in results]
    return urls


@tool
def parse_tool(urls: list[str]):
    """
    A website parser.
    Takes a list of URLs (strings), scrapes them, and returns the full text content.
    Use this only on URLs that looked relevant in the search step.
    """
    soups = [scrape_website(url) for url in urls]
    texts = {url: extract_text(soup) for url, soup in zip(urls, soups)}
    return str(texts)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
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
        self.graph = graph.compile(checkpointer=checkpointer)

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

def search_agent(agent):
    while True:
        try:
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            is_searching = False
            human_message = input("\nEnter your query (q or quit to leave the engine): ") # TODO only put it on start or end node 
            print()
            if human_message.lower() in ("q", "quit"):
                break
            messages = [HumanMessage(human_message)]
            for message, metadata in agent.graph.stream({"messages": messages}, config, stream_mode="messages"):
                if message.content and isinstance(message, AIMessageChunk) and metadata["langgraph_node"] == "llm":
                    print(message.content, end="", flush=True)
                elif isinstance(message, AIMessageChunk) and message.tool_call_chunks:
                    if not is_searching:
                        print("Consulting Search Engine...\n\n", end="", flush=True)
                        is_searching = True
                elif metadata["langgraph_node"] == "action":
                    is_searching = False
            print("\n")
        except KeyboardInterrupt:
            break

system = f"""
You are a rigorous research agent. Your goal is to answer the user's FULL request.

**TEMPORAL OVERRIDE:**
Override internal knowledge with today's date: {datetime.now()}. Rely EXCLUSIVELY on `search_tool` for real-time facts.

**TOOLS:**
1. `search_tool(query)`: Get snippets.
2. `parse_tool(url)`: Get full text.

**CRITICAL PROTOCOL: SILENT ACCUMULATION**
If the user asks for multiple things:
1. **DO NOT generate a text summary** after finding the first item.
2. Instead, **immediately call the search tool for the SECOND item.**
3. Keep gathering data until you have ALL parts of the request.
4. ONLY when you have data for *Weather* AND *Stock* (or all requested parts) do you generate the final response.

**ALGORITHM:**
1. **Analyze**: Break request into parts (e.g., [Part A, Part B]).
2. **Execute Part A**: Search -> Select -> Parse.
3. **Check Status**:
   - Have I finished Part A? YES.
   - Do I have data for Part B? NO.
   - **ACTION**: Call `search_tool` for Part B immediately. (Do not output text like "Here is the weather...").
4. **Execute Part B**: Search -> Select -> Parse.
5. **Finalize**: Now that Part A and Part B are in context, write the combined answer.

**CONSTRAINT:**
- If you output a final text answer while a checklist item is still missing, YOU HAVE FAILED.
- Minimize "chatter". Just call the tools.
"""

tools = [search_tool, parse_tool]
memory = MemorySaver()

model = ChatOllama(model="qwen3:30b", temperature=0.5)
agent = Agent(model=model, tools=tools, system=system, checkpointer=memory)

dot_source = agent.graph.get_graph().draw_mermaid_png()
with open("react_agent_graph.png", "wb") as f:
    f.write(dot_source)

"""messages = [HumanMessage("Quel est la valeur du S&P500 d'hier ?")]
for message, metadata in agent.graph.stream({"messages": messages}, {"configurable": {"thread_id": "1"}}, stream_mode="messages"):
    if message.content and isinstance(message, AIMessageChunk) and metadata["langgraph_node"] == "llm":
        print(message.content, end="", flush=True)"""

search_agent(agent)