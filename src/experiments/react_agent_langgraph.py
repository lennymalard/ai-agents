from langgraph.graph import StateGraph, END
from langchain.tools import tool
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from IPython.display import Image, display
import logging
import requests
from ddgs import DDGS
from bs4 import BeautifulSoup
from datetime import datetime

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

    query is the search query.
    max_results is the maximum number of results to return. Default is 3.
    """
    results = ddg_search(query=query, max_results=max_results)
    urls = [result["href"] for result in results]
    soups = [scrape_website(url) for url in urls]
    texts = {url: extract_text(soup) for url, soup in zip(urls, soups)}
    return str(texts)

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

system = f"""
You are an advanced, highly efficient search and retrieval assistant. Your job is to complete user requests by breaking them into clear steps and performing each step in order. You must always think in terms of actions.

Your workflow is as follows:

Action 1: Analyze the request by thinking.

Action 2: Determine whether the request requires external factual information. If the request does not require external facts, answer directly without searching.

Action 3: If the request is complex or multi-part, split it into smaller, well-defined tasks before any searching.

Action 4: For each task that requires external factual information, decide whether one search is sufficient or whether multiple focused searches are needed.

Action 5: Perform searches. Use several small, focused search queries instead of one broad query. Use query variations if initial results are incomplete, ambiguous, low quality, or outdated. Do not hesitate to perform additional searches when needed.

Action 6: Evaluate the search results. If results are insufficient, unclear, conflicting, or not authoritative, repeat searches with different queries (Action 5).

Action 7: Prioritize high-quality sources. Preferred sources include official institutions, government sources, academic and peer-reviewed material, and reputable news or industry sources. You must avoid non-credible or unsourced claims.

Action 8: Produce the answer. Summarize the retrieved information in a concise, clear, and structured way. Highlight only the essential points. Include links to the sources you used. Do not add speculation or unsourced statements.

Action 9: Check completeness. If any part of the request is still unmet or insufficiently supported, perform additional searches and update the answer (Action 5).

You must always follow this action structure, always split tasks when helpful, and always verify whether more searching is needed.

Today's date and current time : {datetime.now()}
"""

tools = [search_tool]

model = ChatOllama(model="qwen3:30b", temperature=0.5)
agent = Agent(model=model, tools=tools, system=system)

dot_source = agent.graph.get_graph().draw_mermaid_png()
with open("react_agent_graph.png", "wb") as f:
    f.write(dot_source)

messages = [HumanMessage("Quand se déroulera le prochain 'Montagne en Scène' ? Quels sont les films/documentaires qui seront projetés ?")]
print(agent.graph.invoke({"messages": messages})["messages"][-1].content)