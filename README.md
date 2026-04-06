# ArXiv Research Intel: Advanced RAG Knowledge Platform

An advanced Retrieval-Augmented Generation (RAG) platform to query, explore, and synthesize insights from ArXiv Machine Learning and AI research papers. Built with a focus on deep semantic understanding, multi-modal retrieval (Hybrid Vector + Knowledge Graph), and a professional-grade streaming user experience.

## ✨ Features

- **Hybrid Search Pipeline:** Merges Dense embeddings (`BAAI/bge-small-en-v1.5`) and Sparse vectors (Corpus-fitted `BM25`) via Reciprocal Rank Fusion (RRF) in Qdrant for maximum recall.
- **Knowledge Graph Exploration:** Leverages Neo4j to map interconnectivity between papers, authors, and methodologies. Users can toggle between semantic vector search and explicit multi-hop graph retrieval.
- **Live Streaming UI:** Built with React & Vite. Answers are streamed block-by-block using Server-Sent Events (SSE). Includes a "Thinking" UI that visually tracks the internal AI pipeline (Rewrite → Retrieve → Rerank → Generate), complete with proper Markdown rendering and dynamic citation badges.
- **Intelligent Generation Pipeline:**
  - **Query Rewriting:** Uses Anthropic Claude (Haiku) to parse intent and inject temporal awareness before querying the database.
  - **Cross-Encoder Reranking:** Precision-sorts the retrieved chunks locally (`ms-marco-MiniLM-L-6-v2`) before finalizing the LLM prompt.
  - **HyDE:** Hypothetical Document Embeddings for abstract queries.
- **Asynchronous Ingestion Engine:** Uses FastAPI with Redis/ARQ for background task queueing to chunk, embed, and index papers out-of-band, providing live status monitoring to the frontend.
- **Observability & Evaluations:** Integrated with OpenTelemetry (OTLP), Phoenix, and Langfuse to trace LLM calls and measure answer relevancy/faithfulness.

## 🛠️ Tech Stack

**Frontend:** React 19, TypeScript, Vite, Zustand, React-Markdown, Vanilla CSS (Flexbox architecture)
**Backend Orchestrator:** FastAPI (Python), Uvicorn
**Databases / Stores:**
- **Vector DB:** Qdrant
- **Graph DB:** Neo4j
- **Cache & Message Broker:** Redis (with ARQ for tasks)
- **Local Embedded State:** SQLite (for User Auth & API Key encryption)
**AI & Models:** Anthropic Claude (Sonnet/Haiku), FastEmbed (`BAAI/bge-small`), Pinecone-Text (`BM25Encoder`), HuggingFace Transformers (Cross-Encoder)
**Observability:** OpenTelemetry, Arize Phoenix, Langfuse

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js & npm (for frontend building)
- Docker (for Qdrant & Neo4j)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/anjanvikas/rag-lab.git
cd rag-lab
```

2. **Set up Environment Variables:**
```bash
cp .env.example .env
# Fill in your ANTHROPIC_API_KEY, Database URLs, and other necessary values in .env
```

3. **Install Backend Dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. **Install Frontend Dependencies:**
```bash
cd frontend
npm install
npm run build
cd ..
```

5. **Start the Infrastructure (Vector DB, Redis, Graph DB):**
```bash
docker-compose up -d
```

6. **Run the API Backend:**
```bash
uvicorn main:app --reload
```
Navigate to `http://localhost:8000/` to access the chat UI!

## 🔧 Running Ingestion & Tuning
- To ingest the initial dataset of canonical papers: `python scripts/ingest_golden_100.py`
- To fit the custom BM25 sparse encoder to your dataset: `python scripts/tune_bm25.py`
