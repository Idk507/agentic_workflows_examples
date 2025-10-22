"""
Pytest tests for the sequential chain and Pydantic parser
Save as tests/test_sequential_chain.py
"""
from src.sequential_chain_pydantic import DummyLLM, generate_outline, draft_from_outline, polish_draft, extract_product_spec, ProductSpec


def test_outline_draft_polish_flow():
    llm = DummyLLM()
    topic = 'Test topic'
    outline = generate_outline(topic, llm)
    assert 'Intro' in outline
    draft = draft_from_outline(outline, llm)
    assert 'draft' in draft.lower()
    polished = polish_draft(draft, llm)
    assert 'polished' in polished.lower()


def test_product_spec_parsing():
    llm = DummyLLM()
    prod = extract_product_spec('ACME portable charger 20000mAh', llm)
    assert isinstance(prod, ProductSpec)
    assert prod.price_usd == 29.99
    assert 'electronics' in prod.tags

