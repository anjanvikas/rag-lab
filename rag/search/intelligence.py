"""
LLM-based intelligence for search: HyDE (Hypothetical Document Embeddings) and Query Classification.
"""
from __future__ import annotations
import os
import anthropic

def _get_async_client(api_key: str | None = None) -> anthropic.AsyncAnthropic:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    return anthropic.AsyncAnthropic(api_key=key)

async def generate_hyde_document(query: str, api_key: str | None = None) -> str:
    """
    Generate a hypothetical document (HyDE) that answers the query.
    Used to embed the hypothetical answer instead of the query for better dense retrieval.
    """
    client = _get_async_client(api_key)
    system = (
        "You are an expert AI research assistant. "
        "Given a user's technical query about machine learning or artificial intelligence, "
        "write a highly technical, factual, and detailed hypothetical excerpt from a research paper "
        "that perfectly answers the query. Write exactly 2 paragraphs. "
        "Do not include introductory or concluding remarks, just the hypothetical paper text."
    )
    
    try:
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            temperature=0.3,
            system=system,
            messages=[{"role": "user", "content": query}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"HyDE generation failed: {e}")
        return query


async def classify_query(query: str, api_key: str | None = None) -> str:
    """
    Classify a query into EXACT, SEMANTIC, or HYBRID to determine the optimal alpha weighting.
    - EXACT: acronyms, specific paper titles, author names, exact technical terms.
    - SEMANTIC: "how to", "explain", conceptual questions, broad topics.
    - HYBRID: mix of specific terms and conceptual questions.
    Returns: 'EXACT', 'SEMANTIC', or 'HYBRID'.
    """
    client = _get_async_client(api_key)
    system = (
        "Classify the given search query into exactly one of three categories: EXACT, SEMANTIC, or HYBRID.\n\n"
        "- EXACT: The user is looking for a specific paper, author, acronym (e.g., 'BERT', 'GPT-3'), year, or exact phrase.\n"
        "- SEMANTIC: The user is asking a conceptual question (e.g., 'How does attention work?', 'explain diffusion models').\n"
        "- HYBRID: A mix of both (e.g., 'How does the attention mechanism in BERT work?').\n\n"
        "Return ONLY the category name. No other text."
    )
    
    try:
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": query}],
        )
        classification = response.content[0].text.strip().upper()
        if classification in ["EXACT", "SEMANTIC", "HYBRID"]:
            return classification
        return "HYBRID"
    except Exception as e:
        print(f"Query classification failed: {e}")
        return "HYBRID"
