#!/usr/bin/env python3
"""
Build FAISS index from literature memory for recommendations.

This script reads papers from journal_club_memory.json and builds a FAISS
vector index for semantic search and recommendations.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from core.research_agent_adaptive import CachedSentenceTransformerEmbeddings

log = logging.getLogger("journal_club.build_faiss")

MEMORY_PATH = Path("cache/journal_club_memory.json")
FAISS_INDEX_PATH = Path("cache/faiss_index")


def load_papers_from_memory(memory_path: Path) -> List[dict]:
    """Load papers from journal club memory."""
    if not memory_path.exists():
        log.error(f"Memory file not found: {memory_path}")
        return []
    
    with open(memory_path) as f:
        memory = json.load(f)
    
    papers = memory.get("papers", [])
    log.info(f"Loaded {len(papers)} papers from memory")
    return papers


def papers_to_documents(papers: List[dict]) -> List[Document]:
    """Convert papers to LangChain documents."""
    documents = []
    
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        doi = paper.get("doi", "")
        year = paper.get("year", "")
        topic = paper.get("topic_name", "")
        domain = paper.get("domain", "")
        
        if not title or not abstract:
            continue
        
        # Create document content
        content = f"{title}\n{abstract}"
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "year": year,
                "topic": topic,
                "domain": domain,
                "source": "journal_club_memory",
            }
        )
        documents.append(doc)
    
    log.info(f"Converted {len(documents)} papers to documents")
    return documents


def build_faiss_index(documents: List[Document], index_path: Path) -> None:
    """Build FAISS index from documents."""
    if not documents:
        log.error("No documents to index")
        return
    
    log.info(f"Building FAISS index at {index_path}")
    
    # Create embeddings
    embeddings = CachedSentenceTransformerEmbeddings()
    
    # Create FAISS index
    db = FAISS.from_documents(documents, embeddings)
    
    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    db.save_local(str(index_path))
    
    log.info(f"FAISS index saved to {index_path}")


def main():
    """Main function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    log.info("=== Building FAISS Index ===")
    
    # Load papers
    papers = load_papers_from_memory(MEMORY_PATH)
    if not papers:
        log.error("No papers found in memory")
        return
    
    # Convert to documents
    documents = papers_to_documents(papers)
    if not documents:
        log.error("No documents created from papers")
        return
    
    # Build FAISS index
    build_faiss_index(documents, FAISS_INDEX_PATH)
    
    log.info("=== FAISS Index Build Complete ===")


if __name__ == "__main__":
    main()
