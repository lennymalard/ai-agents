from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """
    Adds a and b.
    """
    return a + b

tools_list = [add]

llm = ChatOllama(model="llama3.2", temperature=0.5)
llm_with_tools = llm.bind_tools(tools_list)
prompt = ChatPromptTemplate.from_template("Could you add {a} and {b} ?")
chain = prompt | llm_with_tools

response = chain.invoke({"a": 5, "b": 5})

if response.tool_calls:
    for call in response.tool_calls:
        tool_name = call['name']
        tool_args = call['args']

        print(f"Tool : {tool_name}")
        print(f"Args: {tool_args}")

        tool = next(t for t in tools_list if t.name == tool_name)

        print(f"The result is {tool.invoke(tool_args)}.")
