"""
Stage 4: Hybrid Retrieval

Combines dense vector search (BGE embeddings) with sparse BM25 keyword matching
using Reciprocal Rank Fusion (RRF). Includes cross-encoder reranking and
Lost-in-the-Middle mitigation for optimal context placement.
"""

import logging
from typing import List, Dict, Optional, Tuple
from sentence_transformers import CrossEncoder

from src.indexing.embedder import VectorStoreManager
from src.indexing.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval engine combining dense + sparse search with reranking.
    
    Pipeline:
    1. Query rewriting (decompose complex questions)
    2. Dense search via BGE embeddings + Qdrant
    3. Sparse search via BM25
    4. Reciprocal Rank Fusion to merge results
    5. Cross-encoder reranking
    6. Lost-in-the-Middle reordering
    7. Parent chunk expansion for context
    """

    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        vector_store: VectorStoreManager,
        bm25_index: BM25Index,
        use_reranker: bool = True,
    ):
        """
        Args:
            vector_store: VectorStoreManager for dense search.
            bm25_index: BM25Index for sparse search.
            use_reranker: Whether to use cross-encoder reranking.
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.use_reranker = use_reranker
        self.reranker = None

        if use_reranker:
            try:
                logger.info(f"Loading reranker: {self.RERANKER_MODEL}")
                self.reranker = CrossEncoder(self.RERANKER_MODEL)
                logger.info("Reranker loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Proceeding without reranking.")
                self.use_reranker = False

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        company_filter: Optional[str] = None,
        expand_parents: bool = True,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> Dict:
        """
        Execute the full hybrid retrieval pipeline.
        
        Args:
            query: User query string.
            top_k: Number of final results to return.
            company_filter: Optional company name filter.
            expand_parents: Whether to expand child chunks to include parent context.
            dense_weight: Weight for dense retrieval in RRF fusion.
            sparse_weight: Weight for sparse retrieval in RRF fusion.
            
        Returns:
            Dict containing:
                - 'chunks': Final ranked list of retrieved passages
                - 'pipeline_info': Metadata about the retrieval pipeline steps
        """
        pipeline_info = {"stages": []}

        # Stage 1: Sub-queries (for complex questions)
        sub_queries = self._rewrite_query(query)
        pipeline_info["stages"].append({
            "name": "Query Rewriting",
            "input": query,
            "output": sub_queries,
        })

        # Stage 2 & 3: Dense + Sparse Search
        all_dense_results = []
        all_sparse_results = []

        for sq in sub_queries:
            dense = self.vector_store.search(sq, top_k=top_k * 2, company_filter=company_filter)
            sparse = self.bm25_index.search(sq, top_k=top_k * 2, company_filter=company_filter)
            all_dense_results.extend(dense)
            all_sparse_results.extend(sparse)

        pipeline_info["stages"].append({
            "name": "Dense Search (BGE)",
            "results_count": len(all_dense_results),
        })
        pipeline_info["stages"].append({
            "name": "Sparse Search (BM25)",
            "results_count": len(all_sparse_results),
        })

        # Stage 4: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            all_dense_results, all_sparse_results,
            dense_weight, sparse_weight
        )
        pipeline_info["stages"].append({
            "name": "Reciprocal Rank Fusion",
            "results_count": len(fused),
        })

        # Stage 5: Cross-encoder reranking
        if self.use_reranker and self.reranker and fused:
            fused = self._rerank(query, fused, top_k=top_k * 2)
            pipeline_info["stages"].append({
                "name": "Cross-Encoder Reranking",
                "model": self.RERANKER_MODEL,
                "results_count": len(fused),
            })

        # Take top_k results
        fused = fused[:top_k]

        # Stage 6: Lost-in-the-Middle reordering
        if len(fused) > 2:
            fused = self._mitigate_lost_in_middle(fused)
            pipeline_info["stages"].append({
                "name": "Lost-in-the-Middle Reordering",
                "strategy": "Primacy-Recency placement",
            })

        # Stage 7: Parent chunk expansion
        if expand_parents:
            for result in fused:
                parent_id = result.get("parent_id", "")
                if parent_id:
                    parent = self.vector_store.get_parent_chunk(parent_id)
                    if parent:
                        result["parent_text"] = parent.text
                        result["parent_section"] = parent.metadata.get("section_title", "")

            pipeline_info["stages"].append({
                "name": "Parent Chunk Expansion",
                "expanded_count": sum(1 for r in fused if "parent_text" in r),
            })

        return {
            "chunks": fused,
            "pipeline_info": pipeline_info,
        }

    def _rewrite_query(self, query: str) -> List[str]:
        """
        Decompose complex queries into sub-queries.
        
        For the prototype, uses rule-based decomposition.
        In production, this would use the LLM for query rewriting.
        """
        sub_queries = [query]  # Always include the original

        # Detect compound questions (e.g., "What are the risks and revenue?")
        if " and " in query.lower() and "?" in query:
            parts = query.split(" and ")
            if len(parts) == 2:
                # Only decompose if both parts are substantial
                if all(len(p.strip()) > 10 for p in parts):
                    sub_queries.extend([p.strip().rstrip("?") + "?" for p in parts])

        # Detect comparative questions
        if any(kw in query.lower() for kw in ["compare", "difference between", "versus", "vs"]):
            # Keep as single query — comparative questions need broad retrieval
            pass

        return sub_queries

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        k: int = 60,
    ) -> List[Dict]:
        """
        Merge ranked lists using Reciprocal Rank Fusion (RRF).
        
        RRF Score = Σ (weight / (k + rank))
        """
        fused_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict] = {}

        # Score dense results
        for rank, result in enumerate(dense_results):
            key = result.get("chunk_id") or result.get("text", "")[:100]
            score = dense_weight / (k + rank + 1)
            fused_scores[key] = fused_scores.get(key, 0) + score
            if key not in result_map:
                result_map[key] = result

        # Score sparse results
        for rank, result in enumerate(sparse_results):
            key = result.get("chunk_id") or result.get("text", "")[:100]
            score = sparse_weight / (k + rank + 1)
            fused_scores[key] = fused_scores.get(key, 0) + score
            if key not in result_map:
                result_map[key] = result

        # Sort by fused score
        sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        results = []
        for key in sorted_keys:
            result = result_map[key].copy()
            result["rrf_score"] = fused_scores[key]
            results.append(result)

        return results

    def _rerank(self, query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank results using a cross-encoder model."""
        if not results:
            return results

        pairs = [(query, r["text"]) for r in results]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]

    def _mitigate_lost_in_middle(self, results: List[Dict]) -> List[Dict]:
        """
        Reorder results to counteract the 'Lost in the Middle' bias.
        
        Places the most relevant chunks at the beginning (primacy) and
        end (recency) of the context window, with less important chunks
        in the middle.
        
        Strategy: [1st, 3rd, 5th, ..., 6th, 4th, 2nd]
        """
        if len(results) <= 2:
            return results

        # Interleave: odd positions first (ascending), then even positions (descending)
        reordered = []
        odd_items = results[0::2]  # 1st, 3rd, 5th, ...
        even_items = results[1::2]  # 2nd, 4th, 6th, ...

        reordered.extend(odd_items)
        reordered.extend(reversed(even_items))

        return reordered
