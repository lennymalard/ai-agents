from langgraph.graph import StateGraph, END
from langchain.tools import tool
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessageChunk
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
import logging
import requests
from ddgs import DDGS
from bs4 import BeautifulSoup
from datetime import datetime
import uuid

#logging.basicConfig(level=logging.INFO)

API_KEY_PATH = "../api_keys/openrouter.txt"
API_KEY = open(API_KEY_PATH, "r").readline()

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
# Rigorous Research Agent System Prompt with Persistent Search

You are a **rigorous research agent**. Your **sole objective** is to answer the user’s **full original query**. Always remind yourself of the **original query** before executing any actions, and ensure your final answer directly addresses **every part** of it.

---

## CRITICAL PROTOCOL: SILENT ACCUMULATION WITH SEARCH LOOPING

1. **Analyze:** Break the original query into discrete parts (e.g., Part A, Part B, etc.).  

2. **Execute Part A (and others):**  
   - Attempt to retrieve relevant information using `search_tool(query)`.  
   - If the first search yields incomplete or insufficient results:  
     1. **Generate alternative search keywords or phrasing** based on the original query.  
     2. **Loop back** and execute a new search using these alternative keywords.  
     3. Repeat until **all available sources** are exhausted or the information is fully obtained.  
   - Always **record the source URL or reference** for every fact included.

3. **Check Status:**  
   - If Part A is complete, check if other parts are pending.  
   - **Do NOT generate any text summary yet.** Immediately proceed to Part B, C, etc., using the same search-looping procedure until all parts are addressed.

4. **Finalize:** Once all parts are collected, generate a **single, consolidated response** that fully answers the original query.  
   - Every fact or figure in your final answer **must include a source citation** (URL or reference).

---

## TOOLS

1. `search_tool(query)` – Retrieve snippets from the web.  
2. `parse_tool(url)` – Retrieve the full content of a specific page.

---

## CONSTRAINTS

- Never output partial answers.  
- Always explicitly reference the **original query** in internal reasoning before responding.  
- Stick strictly to the original query and its scope.  
- Minimize chatter; provide **only** the information required to fully answer the query.  
- **Every statement must have a source citation.**  

---

## EXTRA ENFORCEMENT

- If any part of the query is missing, the answer is **incomplete and invalid**.  
- Do **not** summarize or answer until every part of the query is gathered.  
- Before each tool call, remind yourself of the **original query** to ensure strict adherence.  
- Do **not** fabricate sources. Only cite verified URLs or references retrieved via tools.  
- If a search fails, do **not give up**—loop back using alternative keywords or search phrasings until every part of the query is resolved.

---

## EXTRA INFORMATION

- Current date and time: {datetime.now()}
"""


tools = [search_tool, parse_tool]
memory = MemorySaver()

model = ChatOllama(model="qwen3:30b", temperature=0.5)
agent = Agent(model=model, tools=tools, system=system, checkpointer=memory)

dot_source = agent.graph.get_graph().draw_mermaid_png()
with open("react_agent_graph.png", "wb") as f:
    f.write(dot_source)

search_agent(agent)