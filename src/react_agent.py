"""
Minimal ReAct-style agent implementation (provider-agnostic).
"""
from __future__ import annotations
import time
import json
import subprocess
import shlex
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, List, Optional

# Simple LLM adapter interface - implement .generate(prompt) -> str
class LLMAdapter:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


# Simple in-memory vector store using sentence-transformers embeddings
class SimpleVectorStore:
    def __init__(self, encoder):
        self.encoder = encoder
        self.docs: List[Dict[str, Any]] = []

    def add(self, id: str, text: str, meta: Optional[Dict[str, Any]] = None):
        vec = self.encoder.encode([text])[0]
        self.docs.append({"id": id, "text": text, "vec": vec, "meta": meta or {}})

    def search(self, query: str, k: int = 3):
        qv = self.encoder.encode([query])[0]
        # naive cosine similarity
        def dot(a,b):
            return sum(x*y for x,y in zip(a,b))
        def norm(a):
            return sum(x*x for x in a) ** 0.5
        scores = []
        for d in self.docs:
            score = dot(qv, d['vec']) / (norm(qv)*norm(d['vec']) + 1e-12)
            scores.append((score, d))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scores[:k]]


# Safe python execution - minimal sandbox using subprocess; do NOT trust for production
def safe_python_execute(code: str, timeout: int = 3) -> str:
    """
    Executes code in a subprocess with resource limits. Returns stdout+stderr.
    NOTE: This is a minimal example and not secure for untrusted code! See README for production advice.
    """
    wrapped = f"""
import sys
import json
# minimal wrapper: execute user code and print JSON with 'output' or 'error'
try:
    locals_dict = {}
    globals_dict = {'__builtins__': {}}
    # we'll allow a very small safe set
    safe_builtins = {'range': range, 'len': len, 'print': print}
    globals_dict['__builtins__'] = safe_builtins
    exec(compile({code!r}, '<user_code>', 'exec'), globals_dict, locals_dict)
    # attempt to capture a variable named 'result' if user sets it
    result = locals_dict.get('result', None)
    print(json.dumps({'output': str(result)}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"""
    # run in subprocess
    try:
        proc = subprocess.run(["python3", "-c", wrapped], capture_output=True, text=True, timeout=timeout)
        out = proc.stdout.strip() or proc.stderr.strip()
        return out
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "timeout"})


@dataclass
class ToolResult:
    tool: str
    input: str
    observation: str


class ReActAgent:
    def __init__(self, llm: LLMAdapter, tools: Dict[str, Callable[[str], str]], max_steps: int = 6):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def build_prompt(self, question: str, scratchpad: List[ToolResult]) -> str:
        prompt = "You are an agent that decides actions and returns either a final answer or an action.\n"
        prompt += "Available tools: " + ", ".join(self.tools.keys()) + "\n"
        prompt += "When you want to use a tool, respond with JSON:\n{\n  \"action\": <tool_name>,\n  \"input\": <string>\n}\n"
        prompt += "If you want to finish, respond with JSON: {\"final_answer\": <string>}\n\n"
        prompt += f"Question: {question}\n\n"
        if scratchpad:
            prompt += "Scratchpad:\n"
            for s in scratchpad:
                prompt += f"Tool: {s.tool}\nInput: {s.input}\nObservation: {s.observation}\n---\n"
        prompt += "Agent:"
        return prompt

    def parse_model_output(self, text: str) -> Dict[str, Any]:
        # naive: find first JSON in the text
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"final_answer": text.strip()}
        try:
            obj = json.loads(m.group(0))
            return obj
        except Exception:
            return {"final_answer": text.strip()}

    def run(self, question: str) -> Dict[str, Any]:
        scratchpad: List[ToolResult] = []
        for step in range(self.max_steps):
            prompt = self.build_prompt(question, scratchpad)
            out = self.llm.generate(prompt)
            parsed = self.parse_model_output(out)
            if 'final_answer' in parsed:
                return {"answer": parsed['final_answer'], "trace": [asdict(s) for s in scratchpad]}
            if 'action' in parsed and parsed['action'] in self.tools:
                tool_name = parsed['action']
                tool_input = parsed.get('input', '')
                obs = self.tools[tool_name](tool_input)
                tr = ToolResult(tool=tool_name, input=tool_input, observation=str(obs))
                scratchpad.append(tr)
                continue
            else:
                # Model didn't follow the structure; treat as final
                return {"answer": out.strip(), "trace": [asdict(s) for s in scratchpad]}
        return {"answer": "max steps reached", "trace": [asdict(s) for s in scratchpad]}


# small example LLM adapter using a deterministic local stub
class EchoLLM(LLMAdapter):
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # VERY naive heuristic: if prompt contains 'python' or 'execute', ask to call python
        if 'python' in prompt.lower() and 'safe' in prompt.lower():
            return json.dumps({"action": "python_exec", "input": "result = 1+1"})
        if 'population' in prompt.lower():
            return json.dumps({"action": "vector_search", "input": "population Bangalore"})
        return json.dumps({"final_answer": "I could not find an action. Here's a default answer."})


# Demo runner
def demo_react_agent():
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    vs = SimpleVectorStore(encoder)
    vs.add('doc1', 'Bangalore population is about 12 million as of 2023.')
    vs.add('doc2', 'Delhi population 30 million in 2023')

    tools = {
        'vector_search': lambda q: json.dumps([d['text'] for d in vs.search(q, k=2)]),
        'python_exec': lambda code: safe_python_execute(code)
    }
    llm = EchoLLM()
    agent = ReActAgent(llm=llm, tools=tools)
    q = 'Find population of Bangalore and compute 2+2 using python (safe)'
    res = agent.run(q)
    print('RESULT:', res)


if __name__ == '__main__':
    demo_react_agent()
