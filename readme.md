Overview
========
This repo contains three concrete, runnable examples implementing the three items you asked for:

- README (below)
- requirements.txt (below)
- src/react_agent.py # Minimal ReAct agent with tools: vector search (FAISS-like), safe python executor
- src/sequential_chain_pydantic.py # SequentialChain example (outline->draft->polish) with Pydantic parser and unit tests
- src/langchain_agent.py # LangChain-based agent, demonstrating function-calling & tools (provider-agnostic)
- tests/test_sequential_chain.py # pytest tests for structured outputs
- run_examples.sh # simple run script

- 
1) A ReAct-style agent (react_agent.py) that can use a vector-search tool and a safe Python execution tool.
- Provider-agnostic LLM adapter (you plug your own client). The agent runs a reasoning loop and calls tools.
- Safe Python executor uses a subprocess sandbox with timeouts and restricted globals. IMPORTANT: sandboxing is limited; review before using in production.


2) A SequentialChain + Pydantic output parser (sequential_chain_pydantic.py)
- Demonstrates breaking generation into outline -> draft -> polish, and parsing structured JSON into a Pydantic model.
- Includes pytest tests validating parser behavior.


3) A LangChain implementation (langchain_agent.py)
- Shows LangChain Tools, AgentExecutor (ReAct style), Pydantic output parser, and native function-calling style where applicable.
- Provider-agnostic: configure environment variables for OpenAI/Azure/other backed ChatModels supported by LangChain.


How to use
==========
1. Create a Python 3.10+ virtualenv and install dependencies:


python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


2. Configure your LLM provider env vars (for LangChain example):
- OPENAI_API_KEY or AZURE_OPENAI_KEY, etc. (Follow LangChain docs for provider setup.)


3. Run tests:
pytest -q


4. Run examples:
bash run_examples.sh


Caveats and security
====================
- The provided "safe" Python executor is a minimal sandbox. Running untrusted code is inherently risky. For production, use proper sandboxing (containers, gVisor, Firecracker, or remote execution with strict resource & syscall policies).
- The vector store here is an in-memory, simplified FAISS-like stub for demo. Swap with real vector DB (Weaviate, Pinecone, Milvus) in production.


------------------
# FILE: requirements.txt


llama-cpp-python>=0.1.0 # optional small LLMs if you want an offline model
langchain>=0.0.280 # adjust if your environment has different versions
pydantic>=1.10.7
pytest>=7.0
sentence-transformers>=2.2.2
scikit-learn>=1.3.0
faiss-cpu; platform_system != 'Windows'
openai


# Note: remove or adapt packages you don't need. faiss-cpu may need system deps.
