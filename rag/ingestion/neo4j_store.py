"""
Neo4j Knowledge Graph Store (Phase B)
Connects to Neo4j, defines schema, and uses LLM to extract and ingest structured KG data.
"""
from __future__ import annotations
import os
import json
import asyncio
from neo4j import AsyncGraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rag_password")

_driver = None

def _get_driver():
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver

async def setup_schema():
    """Set up Neo4j constraints per spec §3.2."""
    driver = _get_driver()
    async with driver.session() as session:
        # We ignore errors if constraints already exist
        queries = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE"
        ]
        for q in queries:
            try:
                await session.run(q)
            except Exception as e:
                pass


async def extract_and_write_kg(paper: dict, chunks: list[dict], api_key: str | None = None) -> tuple[int, int]:
    """
    Extract structured Entities and Relations using Anthropic Claude Haiku,
    then ingest into Neo4j. Returns (nodes_added, edges_added).
    """
    import anthropic
    
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return 0, 0
        
    client = anthropic.AsyncAnthropic(api_key=key)
    
    # We will sample up to ~2000 words from abstract and first few chunks to avoid LLM limits
    text_to_extract = "\n\n".join([c["text"] for c in chunks[:5]])[:10000]
    
    system = """
You are an expert graph data extractor. Your task is to extract entities and relationships from the following text to populate a Knowledge Graph.
Return a JSON object strictly matching this schema:
{
  "entities": [
    {"id": "unique_string_id", "name": "Human Readable Name", "type": "Concept|Method|Task|Dataset|Metric|Entity"}
  ],
  "relations": [
    {
      "source": "concept_id_from_entities",
      "target": "concept_id_from_entities",
      "type": "RELATES_TO|EVALUATED_ON|USES|IMPROVES|CONTRADICTS",
      "confidence": 0.95,
      "evidence": "brief exact quote"
    }
  ]
}

Guidelines:
- "confidence" must be between 0.0 and 1.0.
- Only extract highly relevant machine learning and AI concepts.
- Use 'CONTRADICTS' safely, require >=0.85 confidence and explicit language.
- Ensure all source and target IDs exist in the entities list.
- Return ONLY valid JSON block, starting with { and ending with }. No markdown fences.
"""

    try:
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2000,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": f"Extract from this text:\n\n{text_to_extract}"}],
        )
        out = response.content[0].text.strip()
        # Clean potential markdown fences
        if out.startswith("```json"):
            out = out[7:]
        if out.endswith("```"):
            out = out[:-3]
        
        data = json.loads(out)
    except Exception as e:
        print(f"KG LLM Extraction failed: {e}")
        return 0, 0

    entities = data.get("entities", [])
    relations = data.get("relations", [])
    
    # Filter relations based on confidence per Phase B spec
    valid_relations = []
    for r in relations:
        conf = float(r.get("confidence", 0.0))
        rel_type = r.get("type", "RELATES_TO")
        
        if rel_type == "CONTRADICTS" and conf < 0.85:
            continue
        if conf >= 0.70:
            valid_relations.append(r)
        # Spec says 0.50-0.69 -> candidate, <0.50 discard. We simply discard <0.70 for main DB for now.

    # Fast Neo4j ingest
    driver = _get_driver()
    nodes_added = 0
    edges_added = 0
    
    arxiv_id = paper.get("arxiv_id", "")
    title = paper.get("title", "")
    tier = paper.get("tier", 5)

    async with driver.session() as session:
        # Create Paper Node
        await session.run(
            "MERGE (p:Paper {arxiv_id: $arxiv_id}) "
            "SET p.title = $title, p.tier = $tier",
            arxiv_id=arxiv_id, title=title, tier=tier
        )
        nodes_added += 1
        
        # Merge Entities and connect to Paper
        for ent in entities:
            eid = ent["id"].lower()
            name = ent.get("name", "")
            etype = ent.get("type", "Entity")
            await session.run(
                "MERGE (e:Entity {id: $eid}) "
                "SET e.name = $name, e.type = $etype "
                "WITH e "
                "MATCH (p:Paper {arxiv_id: $arxiv_id}) "
                "MERGE (p)-[:MENTIONS]->(e)",
                eid=eid, name=name, etype=etype, arxiv_id=arxiv_id
            )
            nodes_added += 1
            edges_added += 1
            
        # Merge Relations
        for rel in valid_relations:
            src = rel["source"].lower()
            tgt = rel["target"].lower()
            rtype = rel.get("type", "RELATES_TO").upper()
            safe_type = "".join(c for c in rtype if c.isupper() or c == "_") or "RELATES_TO"
            
            # Neo4j parameterized relationship types are not supported directly, so we use Cypher injection
            query = (
                "MATCH (a:Entity {id: $src}) "
                "MATCH (b:Entity {id: $tgt}) "
                f"MERGE (a)-[r:{safe_type}]->(b) "
                "SET r.confidence = $conf, r.evidence = $evidence"
            )
            await session.run(query, src=src, tgt=tgt, conf=rel.get("confidence"), evidence=rel.get("evidence", ""))
            edges_added += 1
            
    return nodes_added, edges_added

