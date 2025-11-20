from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2", temperature=5)
prompt = ChatPromptTemplate.from_messages([("human", "Hello, how are you ?")])
chain = prompt | llm

response = chain.invoke({})

print(response.content)
