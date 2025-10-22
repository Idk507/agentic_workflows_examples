"""
LangChain-based agent example. Save as src/langchain_agent.py
This file shows how to wire LangChain tools and an AgentExecutor. It is illustrative; depending on your LangChain version some APIs may differ.
"""
from typing import Dict, Any

try:
    from langchain import OpenAI
    from langchain.agents import initialize_agent, Tool
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import PydanticOutputParser
except Exception:
    # Keep import-time safe for environments without langchain
    OpenAI = None

from pydantic import BaseModel

class Recommendation(BaseModel):
    title: str
    summary: str
    priority: int

# Example LangChain wiring
def build_langchain_agent(api_key: str = None):
    if OpenAI is None:
        raise RuntimeError('LangChain is not installed or import failed; install requirements.')
    # configure client - this example assumes OpenAI key in env, or pass key
    llm = OpenAI(temperature=0)

    # simple tool: search (could be vector DB wrapper)
    def web_search_tool(query: str) -> str:
        # Replace with a real search or vector DB call
        return f"Search results for: {query} (stub)"

    tools = [
        Tool(name='web_search', func=web_search_tool, description='Search the web or knowledge base')
    ]

    # initialize agent (uses ReAct / zero-shot react description under the hood)
    agent = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=True)
    return agent

# Example run
def demo_langchain_agent():
    agent = build_langchain_agent()
    out = agent.run('Find three recommendations for improving model deployment pipelines')
    print('Agent output:\n', out)

if __name__ == '__main__':
    demo_langchain_agent()
