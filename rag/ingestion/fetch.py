"""
Fetch papers from the arXiv API and chunk them.
Phase A: abstract-only chunks (chunk_type='abstract').
Phase B will add PDF full-body extraction via PyMuPDF.
"""
from __future__ import annotations
import asyncio, re, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime

ARXIV_API = "https://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"

# Tier mapping from arXiv ID
TIER_MAP: dict[str, int] = {
    # Tier 1
    "1706.03762": 1, "1810.04805": 1, "2005.14165": 1, "2001.08361": 1,
    "2201.11903": 1, "2212.08073": 1, "2303.08774": 1, "2307.09288": 1,
    "2401.04088": 1, "2412.19437": 1,
    # Tier 2
    "2004.04906": 2, "2005.11401": 2, "1702.08734": 2, "2004.12832": 2,
    "2212.10496": 2, "2401.18059": 2, "2404.16130": 2,
    # Tier 3
    "1902.10197": 3, "2110.15256": 3, "2002.00388": 3, "2306.08302": 3, "2402.07630": 3,
    # Tier 4
    "2203.02155": 4, "2210.03629": 4, "2302.04761": 4, "2308.03688": 4,
    "2310.03714": 4, "2305.18290": 4, "2406.04692": 4,
    # Tier 5
    "2103.00020": 5, "2112.10752": 5, "2204.14198": 5, "2309.17421": 5,
}


def _base_id(arxiv_id: str) -> str:
    """Strip version suffix: '1706.03762v3' → '1706.03762'."""
    return re.sub(r"v\d+$", "", arxiv_id.split("/")[-1].strip())


async def fetch_paper(arxiv_id: str) -> dict:
    """Fetch paper metadata from arXiv API."""
    base = _base_id(arxiv_id)
    url = f"{ARXIV_API}?id_list={base}&max_results=1"

    def _do_fetch() -> str:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode("utf-8")

    loop = asyncio.get_event_loop()
    xml_data = await loop.run_in_executor(None, _do_fetch)

    root = ET.fromstring(xml_data)
    entry = root.find(f"{NS}entry")
    if entry is None:
        raise ValueError(f"arXiv ID {base} not found")

    title = (entry.findtext(f"{NS}title") or "").replace("\n", " ").strip()
    abstract = (entry.findtext(f"{NS}summary") or "").replace("\n", " ").strip()
    published = (entry.findtext(f"{NS}published") or "")[:10]
    year = published[:4] if published else str(datetime.now().year)

    authors = [
        a.findtext(f"{NS}name", "").strip()
        for a in entry.findall(f"{NS}author")
    ]

    cats: list[str] = []
    for c in entry.findall("{http://arxiv.org/schemas/atom}primary_category"):
        cats.append(c.get("term", ""))
    if not cats:
        for c in entry.findall(f"{NS}category"):
            cats.append(c.get("term", ""))

    result = {
        "arxiv_id": base,
        "title": title,
        "abstract": abstract,
        "authors": authors[:10],
        "year": year,
        "venue": cats[0] if cats else "arXiv",
        "tier": TIER_MAP.get(base, 0),
        "body": ""
    }
    
    # Download and extract PDF body
    pdf_url = f"https://arxiv.org/pdf/{base}.pdf"
    import tempfile
    import os
    try:
        def _download_pdf():
            req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()

        pdf_bytes = await loop.run_in_executor(None, _download_pdf)
        
        def _extract_pdf():
            # Try PyMuPDF primary
            try:
                import fitz
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
                return text
            except Exception as e:
                print(f"PyMuPDF failed: {e}. Trying pdfplumber...")
                import pdfplumber
                import io
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    return "\n".join([page.extract_text() or "" for page in pdf.pages])

        body_text = await loop.run_in_executor(None, _extract_pdf)
        result["body"] = body_text.strip()
    except Exception as e:
        print(f"PDF fetch/extract failed for {base}: {e}")

    return result


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200, separators: list[str] | None = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        final_chunks = []
        
        # Get the current separator
        separator = ""
        new_separators = []
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break
        
        # Split by separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks
        current_doc = []
        total_len = 0
        
        for s in splits:
            if total_len + len(s) + (len(separator) if current_doc else 0) <= self.chunk_size:
                current_doc.append(s)
                total_len += len(s) + (len(separator) if len(current_doc) > 1 else 0)
            else:
                if current_doc:
                    doc_text = separator.join(current_doc).strip()
                    if doc_text:
                        if len(doc_text) > self.chunk_size:
                            final_chunks.extend(self._recursive_split(doc_text, new_separators))
                        else:
                            final_chunks.append(doc_text)
                    
                    # Handle overlap
                    while current_doc and (total_len > self.chunk_overlap or total_len > self.chunk_size):
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + (len(separator) if current_doc else 0)
                
                current_doc.append(s)
                total_len += len(s) + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc_text = separator.join(current_doc).strip()
            if doc_text:
                if len(doc_text) > self.chunk_size:
                    final_chunks.extend(self._recursive_split(doc_text, new_separators))
                else:
                    final_chunks.append(doc_text)

        return final_chunks


def chunk_paper(paper: dict) -> list[dict]:
    """
    Produce chunks from a paper dict using RecursiveCharacterTextSplitter.
    Includes a metadata header in each chunk text for BM25/Dense boost.
    """
    chunks: list[dict] = []
    base_meta = {
        "paper_id": f"arxiv:{paper['arxiv_id']}",
        "arxiv_id": paper["arxiv_id"],
        "title":    paper["title"],
        "authors":  ", ".join(paper["authors"][:5]),
        "year":     paper["year"],
        "tier":     paper["tier"],
        "venue":    paper.get("venue", "arXiv"),
    }

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # 1. Abstract
    abstract = paper.get("abstract", "").strip()
    if abstract:
        header = f"Title: {paper['title']}\nSection: Abstract\n\n"
        abstract_splits = splitter.split_text(abstract)
        for i, text in enumerate(abstract_splits):
            chunks.append({
                **base_meta,
                "text":        header + text,
                "chunk_type":  "abstract",
                "section":     "Abstract",
                "chunk_index": len(chunks),
            })

    # 2. Body (with section awareness)
    body = paper.get("body", "").strip()
    if body:
        # Strip references
        ref_match = re.search(r'\nReferences\n|\nBibliography\n', body, re.IGNORECASE)
        if ref_match:
            body = body[:ref_match.start()]

        # Basic section splitter
        sections = re.split(r'\n(Introduction|Methods|Related Work|Results|Discussion|Conclusion)\n', body, flags=re.IGNORECASE)
        
        current_section = "Main Body"
        for i in range(0, len(sections)):
            part = sections[i].strip()
            if not part: continue
            
            # If this is a header match
            if part.lower() in ("introduction", "methods", "related work", "results", "discussion", "conclusion"):
                current_section = part.capitalize()
                continue
            
            header = f"Title: {paper['title']}\nSection: {current_section}\n\n"
            splits = splitter.split_text(part)
            for text in splits:
                if len(text) < 100: continue
                chunks.append({
                    **base_meta,
                    "text":        header + text,
                    "chunk_type":  "body",
                    "section":     current_section,
                    "chunk_index": len(chunks),
                })

    return chunks
