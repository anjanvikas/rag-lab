import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.ingestion.fetch import fetch_paper, chunk_paper
from rag.ingestion.qdrant_store import embed_and_upsert, setup_collection
from rag.ingestion.log import update_ingestion_log

# The "Golden 100" ArXiv IDs provided by the user
SEMINAL_PAPERS = [
    # Transformers & NLP
    "1706.03762", "1810.04805", "1901.07291", "2005.14165", 
    "1910.10683", "1907.11692", "1909.11942", "1910.01108", "2101.03961",
    
    # Deep Learning & CNNs
    "1404.5997", "1409.1556", "1409.4842", "1502.03167", "1207.0580", 
    "1512.03385", "1608.06993", "1704.04861", "1905.11946", "1311.2901",
    
    # Computer Vision
    "1311.2524", "1504.08083", "1506.01497", "1703.06870", "1506.02640", 
    "1804.02767", "2004.10934", "1505.04597", "1411.4038", "2010.11929",
    
    # Sequence Models
    "1409.3215", "1409.0473", "1508.04025", "1506.03134", "1802.05365", "1901.02860",
    
    # Generative
    "1406.2661", "1411.1784", "1703.10593", "1812.04948", "1701.07875", 
    "1312.6114", "1804.03599", "1605.08803", "2006.11239", "2112.10752",
    
    # Reinforcement Learning
    "1312.5602", "1509.06461", "1712.01815", "1602.01783", "1707.06347", 
    "1801.01290", "2106.01345",
    
    # Representation
    "2002.05709", "1911.05722", "2006.07733", "2103.00020", "2104.14294", "1707.07012",
    
    # Multimodal
    "2102.12092", "2204.14198", "2102.05918", "2201.12086", "2302.14045", 
    "2209.06794", "2304.02643",
    
    # Modern LLM & Scaling
    "2001.08361", "2203.15556", "2203.02155", "2106.09685", 
    "2205.14135", "2001.04451", "2009.14794", "2302.13971", "2005.11401",

    # RAG / Retrieval / Knowledge Systems
    "2004.05150", "2007.14062", "2112.04426", "2203.08913", "2205.05131",
    "2002.08909", "2007.01282", "2208.03299", "2004.12832", "2004.04906",

    # Diffusion & Generative Advances
    "2102.09672", "2011.13456", "2205.11487",

    # Reinforcement Learning (Advanced)
    "2010.02193", "1911.08265", "1802.01561",

    # Representation / Embeddings
    "1908.10084", "2104.08691", "2110.06864",

    # Efficiency / Systems
    "1910.02054", "1909.08053", "2309.06180"
]

STATUS_FILE = Path("ingestion_status.json")

def load_status():
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {"done": [], "failed": {}}

def save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)

async def ingest_paper_task(arxiv_id, semaphore, status):
    if arxiv_id in status["done"]:
        print(f"⏩ Skipping {arxiv_id} (already done)")
        return

    async with semaphore:
        print(f"📡 Processing {arxiv_id}...")
        try:
            # 1. Fetch
            paper = await fetch_paper(arxiv_id)
            await update_ingestion_log(arxiv_id, "in_progress", title=paper["title"])
            
            # 2. Chunk
            chunks = chunk_paper(paper)
            for c in chunks:
                c["is_seminal"] = True
            
            # 3. Embed & Upsert
            count = await embed_and_upsert(chunks, version="seminal_v1")
            
            # 4. Success
            status["done"].append(arxiv_id)
            save_status(status)
            await update_ingestion_log(arxiv_id, "done", chunks=count, title=paper["title"])
            print(f"✅ Finished {arxiv_id}: {paper['title'][:50]}... ({count} chunks)")
            
        except Exception as e:
            print(f"❌ Failed {arxiv_id}: {e}")
            status["failed"][arxiv_id] = str(e)
            save_status(status)
            await update_ingestion_log(arxiv_id, "failed", error=str(e))

async def main():
    setup_collection()
    status = load_status()
    
    # Concurrency limit to avoid ArXiv rate limits or memory issues
    semaphore = asyncio.Semaphore(3)
    
    tasks = [ingest_paper_task(aid, semaphore, status) for aid in SEMINAL_PAPERS]
    await asyncio.gather(*tasks)
    
    print("\n" + "="*50)
    print(f"Ingestion Complete!")
    print(f"Done:   {len(status['done'])}")
    print(f"Failed: {len(status['failed'])}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
