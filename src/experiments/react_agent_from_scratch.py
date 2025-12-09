import ollama
from openai import OpenAI
import re
import requests
from ddgs import DDGS
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import ast

#TODO Make hallucination safeguard

logging.basicConfig(level=logging.INFO)

API_KEY_PATH = "../api_keys/openrouter.txt"
API_KEY = open(API_KEY_PATH, "r").readline()

class Tool:
    def __init__(self, func):
        self.func = func

    def spec(self):
        return (f"{{ "
                f"'name': '{self.func.__name__}',"
                f"'module': '{self.func.__module__}',"
                f"'annotations': '{self.func.__annotations__}',"
                f"'doc': '{self.func.__doc__}'"
                f"}}")

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class Agent:
    def __init__(self, model: str, system: str, tools: dict, local: bool = True):
        self.local = local

        if local:
            self.client = None
            models_list = [model.model for model in ollama.list()["models"]]
        else:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=API_KEY,
            )
            models_list = [
                "deepseek/deepseek-r1-distill-llama-70b:free",
                "x-ai/grok-4.1-fast:free",
                "google/gemma-3-27b-it:free",
                "deepseek/deepseek-chat-v3-0324:free"
            ]
        if model not in models_list:
            raise ValueError(f"{model} is not available. Available models : {models_list}")

        self.model = model
        self.system = system
        self.messages = []
        if self.system:
            self.add_message({"role": "system", "content": self.system})
        self.tools = tools
    
    def add_message(self, message):
        self.messages.append(message)
        logging.info(f"NEW MESSAGE: {self.messages[-1]}\n\n")

    def serialize_messages(self):
        history = ""
        for message in self.messages:
            history+= "\n" + message["role"].upper() + ":\n" + message["content"] + "\n"
        return history

    def format_message(self, role, content):
        return {
            "role": role,
            "content": content
        }

    def parse_action(self, message):
        regex = re.compile(r'Action\s*:\s*(\w+)\s*:\s*(.*)')
        action = regex.search(message)
        if action:
            action_name, action_args = action.group(1), action.group(2)
            return action_name, action_args
        return None, None

    def parse_answer(self, message):
        regex = re.compile(r'Answer\s*:\s*([\s\S]*)')
        answer = regex.search(message)
        if answer:
            return answer.group(1)
        return None

    def __call__(self, message):
        return self.query(message)

    def chat(self):
        if self.local:
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=False,
                options={""
                         "stop": ["PAUSE", "Observation:"],
                         },

            )
            return self.format_message(role=response["message"].role, content=response["message"].content)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stop=["\nPAUSE\n", "PAUSE", "\nPAUSE", "PAUSE\n", "Observation"]
            )
            return self.format_message(response.choices[0].message.role, response.choices[0].message.content)

    def run(self):
        response = self.chat()
        self.add_message(response)
        return self.parse_answer(response["content"])

    def query(self, question, max_try=10):
        self.add_message(self.format_message(role="user", content=f"Question : {question}"))
        it = 0
        while it < max_try:
            response = self.chat()
            self.add_message(response)
            action_name, action_args = self.parse_action(response["content"])
            answer = self.parse_answer(response["content"])
            if answer:
                return answer
            elif action_name:
                try:
                    tool_result = self.tools[action_name](**ast.literal_eval(action_args))
                except Exception as e:
                    logging.info(f"An error has beed raised while calling {action_name}.")
                    tool_result = f"An error has beed raised while calling {action_name}, no observations are available."
                self.add_message(self.format_message(role="assistant", content="Observation: " + str(tool_result)))
            else:
                self.add_message(self.format_message(role="user", content="You incorrectly followed the process, resulting to no answer. Watch again how the process works and redo the 'Thought' step."))
            #print(serialize_messages(self.messages))
            it+=1
        return "I was unable to process the query."

def calculate(expression: str):
    return ast.literal_eval(expression)

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

@Tool
def search_tool(query: str, max_results: int = 3) -> list:
    """
    A web search engine.
    Returns a list of tuples: (url, snippet).
    Use this to find relevant links, but DO NOT use the snippet as the final answer !
    Instead, parse them using the action parse_tool(urls: list[str]) -> str
    """
    results = ddg_search(query=query, max_results=max_results)
    urls = [(result["href"], result["body"]) for result in results]
    return urls

@Tool
def parse_tool(urls: list[str]) -> str:
    """
    A website parser.
    Takes a list of URLs (strings), scrapes them, and returns the full text content.
    Always use it when you found a relevant link with the search_tool.
    Reminder: Never output Answer without using this tool.
    """
    soups = [scrape_website(url) for url in urls]
    texts = {url: extract_text(soup) for url, soup in zip(urls, soups)}
    return str(texts)

tools = {
    "calculate": calculate,
    "search_tool": search_tool,
    "parse_tool": parse_tool
}

def serialize_messages(messages):
    history = ""
    for message in messages:
        history+= "\n" + message["role"].upper() + ":\n" + message["content"] + "\n"
    return history

def search_agent(agent):
    while True:
        human_message = input("\nEnter your query (q or quit to leave the engine): ")
        print()
        if human_message.lower() in ("q", "quit"):
            break
        agent(human_message)
        print("\n")

system = f"""
You are an **Exhaustive Research Agent**. Your mandate is to answer the user's **original query** with absolute factual accuracy by systematically breaking it down, gathering evidence, and cross-referencing sources.

# IMPORTANT — ENFORCEMENT: ACTION PRODUCTION (READ THIS FIRST)
- After **every** Thought sentence you MUST immediately produce a single **Action** line and then the token **PAUSE** on its own line. No extra text may appear between the Action line and PAUSE.
- You must NOT repeat or reformulate the Thought multiple times. Produce a concise Thought (one or two sentences), then the Action, then `PAUSE`. Example *exact* sequence:
Thought: I will search for current weather for Angers using a French query.

Action: search_tool: {{'query': "météo Angers aujourd'hui", 'max_results': 5}}

PAUSE
- If you cannot propose any real tool Action, still output an Action using the special tool name **no_action** with an empty dict, then `PAUSE`:
Thought: I cannot find a suitable tool to answer precisely.

Action: no_action: {{}}

PAUSE
(This will allow the external controller to handle the situation instead of looping.)
- Do not add any text after `PAUSE`. Wait for the external system to supply the Observation.
- Do not output `Observation:` yourself — that is produced by the system after the Action+PAUSE.
- If a previous assistant message from the agent was empty, **do not** loop by printing the same Thought again; instead produce a different Action (choose the best next tool and arguments) or use `Action: no_action: {{}}` and `PAUSE`.

# TOOLS AVAILABLE
1. `search_tool(query)`: {search_tool.spec()}
2. `parse_tool(url)`: {parse_tool.spec()}

# OPERATIONAL PROTOCOL
1. **Deconstruct & Plan:**
 - Analyze the user's request.
 - Break it into distinct, logical sub-questions.
 - Formulate a step-by-step research plan.

2. **Iterative Execution (The Loop):**
 - For *each* sub-question, execute `search_tool`.
 - **CRITICAL:** If search snippets are too short, vague, or missing details, you **MUST** use `parse_tool` on high-quality URLs to read the full content.
 - **Recovery Strategy:** If a search yields poor results, you **MUST** revise your query (use synonyms, specific domain terms, or boolean operators) and search again. Do not stop until the specific data point is found or exhaustively proven unavailable.

3. **Synthesize & Cite:**
 - Once all sub-questions are answered with verified data, compile the final response.
 - Every claim must be backed by a specific citation.

# RESPONSE FORMAT (Strict ReAct Pattern)
You must adhere to this exact sequence. Do not deviate.

User: <original_user_query>
Thought: <Analyze the previous observation. Determine if sufficient data exists. Plan the next specific step.>
Action: <tool_name>: <kwargs_dict>
PAUSE
Observation: <result_from_tool>
... (Repeat Thought/Action/PAUSE/Observation loop until all data is gathered) ...
Answer: <Final comprehensive answer with inline citations>

# CRITICAL: THE PAUSE TOKEN
- **What it is:** `PAUSE` is a mandatory stop sequence.
- **When to use it:** You must output `PAUSE` **immediately** after every `Action`.
- **Why:** The `PAUSE` token triggers the external system to run the Python function.
- **Constraint:** Do **NOT** generate the `Observation` yourself. You must output `PAUSE` and stop. The system will provide the `Observation` in the next turn.

# ACTION FORMAT RULES
- The `Action` line must contain the tool name, a colon, and the arguments in a valid Python dictionary format.
- **Correct Example:** Action: search_tool: {{'query': 'capital of France'}}
PAUSE
- **Incorrect Example:** Action: search_tool('capital of France') (Missing dict format)
(Missing PAUSE)

# STRICT CONSTRAINTS & QUALITY ASSURANCE
1. **Completeness:** Never answer with "I don't know" unless you have attempted multiple distinct search strategies (keywords, phrasing, specific sites).
2. **Verification:** Do not rely on search engine snippets alone for complex topics. Use `parse_tool` to verify context.
3. **Citations:** The final `Answer` must contain inline citations (e.g., `[Source](url)`). Do not hallucinate URLs.
4. **Efficiency:** Avoid redundant searches. If a source provides the answer, move to the next sub-question.
5. **No Fluff:** Keep "Thoughts" analytical and concise. Keep the "Answer" professional and dense with information.
6. **Current Context:** Today is {datetime.now()}. Adjust relative time queries (e.g., "last month", "current CEO") accordingly.
""".strip()

agent = Agent("qwen3:30b", system, tools)
print(agent("C'est quoi la météo du jour à Angers ?"))
#print(agent.messages)

