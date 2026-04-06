#!/usr/bin/env python3
"""
One-time Qdrant collection setup.
Run ONCE before starting the worker:
    python scripts/setup_qdrant.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rag.ingestion.qdrant_store import setup_collection

if __name__ == "__main__":
    print("Setting up Qdrant collection...")
    setup_collection()
    print("Done! You can now start the worker and run ingestion.")
