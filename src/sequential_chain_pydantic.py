"""
SequentialChain example with Pydantic parsing.
Save as src/sequential_chain_pydantic.py
"""
from pydantic import BaseModel, ValidationError
from typing import List

# A simple LLM adapter interface as before
class LLMAdapter:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class DummyLLM(LLMAdapter):
    def generate(self, prompt: str) -> str:
        # Extremely simplified deterministic behavior for testing
        if 'Outline' in prompt:
            return '- Intro\n- Background\n- Methods\n- Results\n- Conclusion'
        if 'Draft' in prompt:
            return 'This is a draft based on the outline: ' + prompt[:80]
        if 'Polish' in prompt:
            return 'Polished: ' + prompt[:120]
        # For structured JSON
        if 'Product:' in prompt:
            return '{"name": "ACME Charger", "price_usd": 29.99, "description": "Portable charger", "tags": ["electronics","battery"]}'
        return 'OK'

# Pydantic model
class ProductSpec(BaseModel):
    name: str
    price_usd: float
    description: str
    tags: List[str]

# Chain steps
def generate_outline(topic: str, llm: LLMAdapter) -> str:
    prompt = f"Create an outline for: {topic}\n\nOutline:"
    return llm.generate(prompt)

def draft_from_outline(outline: str, llm: LLMAdapter) -> str:
    prompt = f"Draft a 600-word article based on this outline:\n{outline}\n\nDraft:"
    return llm.generate(prompt)

def polish_draft(draft: str, llm: LLMAdapter) -> str:
    prompt = f"Polish this draft for clarity and concision:\n{draft}\n\nPolish:"
    return llm.generate(prompt)

# Example structured extraction
def extract_product_spec(text: str, llm: LLMAdapter) -> ProductSpec:
    # ask model to produce JSON (here we simulate)
    prompt = f"Product: {text}\nReturn JSON matching schema"
    raw = llm.generate(prompt)
    try:
        return ProductSpec.parse_raw(raw)
    except ValidationError as e:
        # in practice, try repair flows: ask model to fix output
        raise

# Simple runner
def demo_sequential_chain():
    llm = DummyLLM()
    topic = 'Data versioning best practices'
    outline = generate_outline(topic, llm)
    draft = draft_from_outline(outline, llm)
    polished = polish_draft(draft, llm)
    print('Outline:\n', outline)
    print('Polished snippet:\n', polished[:200])
    # Structured extraction demo
    prod = extract_product_spec('ACME portable charger 20000mAh', llm)
    print('Parsed product:', prod)

if __name__ == '__main__':
    demo_sequential_chain()
