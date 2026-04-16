"""
Stage 3b: BM25 Sparse Index

Builds a sparse keyword-based index for exact-match retrieval
of financial terms, regulation numbers, and specific figures.
Used alongside dense retrieval in the hybrid search strategy.
"""

import re
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi

from src.chunking.hierarchical_chunker import Chunk

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 sparse retrieval index for exact keyword matching.
    
    Particularly valuable for:
    - SEBI regulation numbers (e.g., "Regulation 33")
    - Exact financial figures (e.g., "₹1,200 crore")
    - Specific entity names in the prospectus
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def build_index(self, chunks: List[Chunk]):
        """
        Build the BM25 index from a list of chunks.
        
        Args:
            chunks: List of Chunk objects (only child and table chunks are indexed).
        """
        retrievable = [c for c in chunks if c.chunk_type in ("child", "table")]
        
        self.chunks = []
        self.tokenized_corpus = []

        for chunk in retrievable:
            tokens = self._tokenize(chunk.text)
            self.tokenized_corpus.append(tokens)
            self.chunks.append({
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id or "",
                "company_name": chunk.metadata.get("company_name", ""),
                "section_title": chunk.metadata.get("section_title", ""),
                "page_start": chunk.metadata.get("page_start", 0),
                "chunk_type": chunk.chunk_type,
                "filename": chunk.metadata.get("filename", ""),
            })

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index built with {len(self.chunks)} documents")

    def search(
        self,
        query: str,
        top_k: int = 10,
        company_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search the BM25 index.
        
        Args:
            query: Search query string.
            top_k: Number of results to return.
            company_filter: Optional company name filter.
            
        Returns:
            List of result dicts with text, score, and metadata.
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built yet")
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Create scored results
        scored_results = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            result = self.chunks[i].copy()
            result["score"] = float(score)

            # Apply company filter
            if company_filter and result["company_name"] != company_filter:
                continue

            scored_results.append(result)

        # Sort by score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)

        return scored_results[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Preserves financial terms and numbers while lowercasing
        and removing common stop words.
        """
        # Lowercase and split
        text = text.lower()
        # Keep alphanumeric, dots (for numbers), hyphens, and currency symbols
        tokens = re.findall(r'[a-z0-9₹$€¥]+(?:\.[a-z0-9]+)*', text)
        
        # Remove very common stop words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'are', 'was',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'shall',
            'this', 'that', 'these', 'those', 'it', 'its',
        }
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

        return tokens

    def save(self, path: str):
        """Save the BM25 index to disk."""
        data = {
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: str):
        """Load a BM25 index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index loaded from {path} ({len(self.chunks)} documents)")
