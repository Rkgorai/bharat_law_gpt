import os
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

llm = ChatGroq(model_name="llama-3.1-8b-instant")
agent = create_agent(model=llm, tools=[add])

print(agent.invoke({"messages": [("user", "what is 2 plus 2?")]}))
