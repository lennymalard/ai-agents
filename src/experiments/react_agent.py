import ollama
from openai import OpenAI
import re

API_KEY_PATH = "../api_keys/openrouter.txt"
API_KEY = open(API_KEY_PATH, "r").readline()

def calculate(expression: str):
    return eval(expression)

ACTIONS_DICT = {
    "calculate": calculate
}

def serialize_messages(messages):
    history = ""
    for message in messages:
        history+= "\n" + message["role"].upper() + ":\n" + message["content"] + "\n"
    return history

class Agent:
    def __init__(self, model: str, system: str, tools: list, local: bool =True):
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
            self.messages.append({"role": "system", "content": self.system})
        self.tools = tools

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
                options={"stop": ["PAUSE"]}
            )
            return self.format_message(role=response["message"].role, content=response["message"].content)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stop=["\nPAUSE\n", "PAUSE", "\nPAUSE", "PAUSE\n"]
            )
            return self.format_message(response.choices[0].message.role, response.choices[0].message.content)

    def run(self):
        response = self.chat()
        self.messages.append(response)
        return self.parse_answer(response["content"])

    def query(self, question, max_try=5):
        self.messages.append(self.format_message(role="user", content=f"Question : {question}"))
        try_i = 0
        while try_i < max_try:
            try_i+=1
            observation = "An error occurred. No observations are currently available."
            assistant_message = self.chat()["content"]
            self.messages.append(self.format_message(role="assistant", content=assistant_message))
            answer = self.parse_answer(assistant_message)
            if answer:
                return answer
            else:
                action_name, action_args = self.parse_action(assistant_message)
                if action_name and action_args:
                    try:
                        action_result = ACTIONS_DICT[action_name](action_args)
                        observation = action_result
                    except KeyError:
                        pass
                self.messages.append(self.format_message(role="user", content=f"Observation : {observation}"))
        return self.run()

system_prompt = """
You are Noé, a smart assistant.

Sequence:
1. **Thought**: Analyze.
2. **Action** (Optional): "Action : calculate : [expression]" -> PAUSE.
3. **Observation**: Result.
4. **Answer**: Final response.

Tools:
- calculate: Python math expression. Output: Number.

IMPORTANT RULES:
1. **Distinguish Facts from Results**: If you calculate a value (e.g., 365 * 2 = 730), your Answer must clearly state that 730 is the *result*, not the original fact.
   - BAD: "There are 730 days in a year."
   - GOOD: "There are 365 days in a year, so multiplied by 2, it makes 730."
2. **Conversational**: If no tool is needed, go straight from "Thought : " to "Answer : ".
3. **Stop**: Stop generating after "Answer :". NEVER generate after "PAUSE" ! It would break the system.
4. **Last tokens**: Always the tasks by "Answer :" when done.
5. **Split tasks**: Whenever you need to use a tool (or multiple tools), possibly more than once, always loop back from Observation to Thought to initiate a new task.

EXAMPLES:

User: Combien de jour y a t il dans une année ? Multiplie cette valeur par 2.
Thought: I need to know days in a year (365) and multiply by 2.
Action: calculate: 365 * 2
PAUSE
Observation: 730
Answer: Il y a 365 jours dans une année. Si on multiplie ce nombre par 2, on obtient 730.

User: Hello!
Thought: Polite greeting.
Answer: Hello! How can I help?
""".strip()

actions_list = [calculate]
agent = Agent("llama3.2:latest", system_prompt, actions_list)

print(agent("Prends la circonférence de la Terre (équateur) et la circonférence de la Lune ; multiplie la circonférence terrestre par 7, ajoute la circonférence lunaire multipliée par 12, soustrais la distance moyenne Terre–Lune, puis divise le tout par 1000."))
print(serialize_messages(agent.messages))