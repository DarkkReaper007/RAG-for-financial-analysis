"""
RAG Pipeline Orchestrator

Connects all 6 stages of the RAG pipeline into a unified interface:
  Stage 1: PDF Parsing → Stage 2: Chunking → Stage 3: Embedding/Indexing
  Stage 4: Retrieval → Stage 5: Generation → Stage 6: Evaluation

Provides high-level methods: ingest(), query(), evaluate()
"""

import os
import time
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator

from src.ingestion.pdf_parser import PDFParser, ParsedDocument
from src.chunking.hierarchical_chunker import HierarchicalChunker, Chunk
from src.indexing.embedder import VectorStoreManager
from src.indexing.bm25_index import BM25Index
from src.retrieval.retriever import HybridRetriever
from src.generation.generator import RAGGenerator
from src.evaluation.evaluator import RAGEvaluator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for Indian IPO prospectuses.
    
    Orchestrates: PDF Parsing → Hierarchical Chunking → Embedding → 
    Hybrid Retrieval → Grounded Generation → Evaluation
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        use_qdrant: bool = True,
        ollama_model: str = "llama3.2:latest",
        ollama_url: str = "http://localhost:11434",
        data_dir: str = "./data/prospectuses",
        vectorstore_dir: str = "./vectorstore",
        use_reranker: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.bm25_path = self.vectorstore_dir / "bm25_index.pkl"

        # Initialize components
        logger.info("=" * 60)
        logger.info("Initializing RAG Pipeline")
        logger.info("=" * 60)

        # Stage 1 & 2
        self.parser = PDFParser()
        self.chunker = HierarchicalChunker(
            child_chunk_size=1000,
            child_overlap=200,
            min_chunk_size=100,
        )

        # Stage 3: Embedding & Indexing
        self.vector_store = VectorStoreManager(
            qdrant_url=qdrant_url,
            use_qdrant=use_qdrant,
            persist_dir=str(self.vectorstore_dir),
        )

        self.bm25_index = BM25Index()

        # Stage 4: Retrieval
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            use_reranker=use_reranker,
        )

        # Stage 5: Generation
        self.generator = RAGGenerator(
            model_name=ollama_model,
            ollama_base_url=ollama_url,
        )

        # Stage 6: Evaluation
        self.evaluator = RAGEvaluator()

        # State tracking
        self.ingested_documents: List[Dict] = []
        self.all_chunks: List[Chunk] = []
        self.is_indexed = False

        logger.info("Pipeline initialization complete")

    def ingest(
        self,
        pdf_paths: Optional[List[str]] = None,
        recreate_index: bool = True,
    ) -> Dict:
        """
        Execute Stages 1-3: Parse PDFs, chunk, embed, and index.
        
        Args:
            pdf_paths: List of PDF file paths. If None, scans data_dir.
            recreate_index: Whether to recreate the vector index.
            
        Returns:
            Dict with ingestion statistics.
        """
        total_start = time.time()
        stats = {"documents": [], "total_chunks": 0, "timing": {}}

        # Discover PDFs
        if pdf_paths is None:
            pdf_paths = sorted(str(p) for p in self.data_dir.glob("*.pdf"))

        if not pdf_paths:
            logger.warning(f"No PDFs found in {self.data_dir}")
            return {"error": "No PDFs found", "documents": []}

        logger.info(f"Ingesting {len(pdf_paths)} documents")

        # Create vector store collection
        self.vector_store.create_collection(recreate=recreate_index)

        self.all_chunks = []
        self.ingested_documents = []

        for pdf_path in pdf_paths:
            doc_start = time.time()
            path = Path(pdf_path)
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing: {path.name}")
            logger.info(f"{'='*40}")

            # Stage 1: Parse PDF
            t1 = time.time()
            parsed_doc = self.parser.parse(str(path))
            parse_time = time.time() - t1

            # Stage 2: Chunking
            t2 = time.time()
            chunks = self.chunker.chunk_document(parsed_doc)
            chunk_time = time.time() - t2

            self.all_chunks.extend(chunks)

            doc_info = {
                "filename": path.name,
                "company_name": parsed_doc.company_name,
                "total_pages": parsed_doc.total_pages,
                "sections": len(parsed_doc.sections),
                "tables": len(parsed_doc.tables),
                "chunks": {
                    "total": len(chunks),
                    "parent": sum(1 for c in chunks if c.chunk_type == "parent"),
                    "child": sum(1 for c in chunks if c.chunk_type == "child"),
                    "table": sum(1 for c in chunks if c.chunk_type == "table"),
                },
                "timing": {
                    "parse_seconds": round(parse_time, 2),
                    "chunk_seconds": round(chunk_time, 2),
                    "total_seconds": round(time.time() - doc_start, 2),
                },
            }
            stats["documents"].append(doc_info)
            self.ingested_documents.append(doc_info)

            logger.info(f"  → {doc_info['chunks']['total']} chunks created in {doc_info['timing']['total_seconds']}s")

        # Stage 3: Index all chunks
        t3 = time.time()
        self.vector_store.index_chunks(self.all_chunks)
        index_time = time.time() - t3

        # Build BM25 index
        t4 = time.time()
        self.bm25_index.build_index(self.all_chunks)
        bm25_time = time.time() - t4

        # Save BM25 index
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_index.save(str(self.bm25_path))

        self.is_indexed = True
        total_time = time.time() - total_start

        stats["total_chunks"] = len(self.all_chunks)
        stats["timing"] = {
            "embedding_seconds": round(index_time, 2),
            "bm25_seconds": round(bm25_time, 2),
            "total_seconds": round(total_time, 2),
        }
        stats["index_stats"] = self.vector_store.get_collection_stats()

        logger.info(f"\n{'='*60}")
        logger.info(f"Ingestion complete: {len(pdf_paths)} docs, "
                     f"{stats['total_chunks']} chunks, {total_time:.1f}s")
        logger.info(f"{'='*60}")

        return stats

    def query(
        self,
        question: str,
        top_k: int = 5,
        company_filter: Optional[str] = None,
        stream: bool = False,
    ) -> Dict:
        """
        Execute Stages 4-5: Retrieve and generate answer.
        
        Args:
            question: User question.
            top_k: Number of chunks to retrieve.
            company_filter: Optional company name filter.
            stream: Whether to stream the response.
            
        Returns:
            Dict with answer, sources, retrieved chunks, and pipeline info.
        """
        if not self.is_indexed:
            return {"error": "No documents ingested. Please run ingest() first."}

        total_start = time.time()

        # Stage 4: Retrieval
        t_ret = time.time()
        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            company_filter=company_filter,
        )
        retrieval_time = time.time() - t_ret

        chunks = retrieval_result["chunks"]
        pipeline_info = retrieval_result["pipeline_info"]

        # Stage 5: Generation
        t_gen = time.time()
        if stream:
            gen_result = self.generator.generate(question, chunks, stream=False)
        else:
            gen_result = self.generator.generate(question, chunks)
        generation_time = time.time() - t_gen

        total_time = time.time() - total_start

        return {
            "question": question,
            "answer": gen_result["answer"],
            "sources": gen_result["sources"],
            "retrieved_chunks": [
                {
                    "text": c.get("text", "")[:500] + ("..." if len(c.get("text", "")) > 500 else ""),
                    "company": c.get("company_name", ""),
                    "section": c.get("section_title", ""),
                    "page": c.get("page_start", "?"),
                    "score": round(c.get("rrf_score", c.get("score", 0)), 4),
                    "chunk_type": c.get("chunk_type", ""),
                }
                for c in chunks
            ],
            "pipeline_info": pipeline_info,
            "timing": {
                "retrieval_seconds": round(retrieval_time, 2),
                "generation_seconds": round(generation_time, 2),
                "total_seconds": round(total_time, 2),
            },
            "model_info": gen_result.get("model_info", {}),
        }

    def query_stream(
        self,
        question: str,
        top_k: int = 5,
        company_filter: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream a response token by token for the Gradio UI."""
        if not self.is_indexed:
            yield "Error: No documents ingested. Please click 'Ingest Documents' first."
            return

        # Retrieve
        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            company_filter=company_filter,
        )
        chunks = retrieval_result["chunks"]

        # Stream generation
        yield from self.generator.generate_stream(question, chunks)

    def evaluate(
        self,
        custom_questions: Optional[List[Dict]] = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Execute Stage 6: Evaluate the pipeline.
        
        Args:
            custom_questions: Optional list of custom QA test cases.
            top_k: Number of chunks to retrieve per question.
            
        Returns:
            Dict with per-question results and aggregate scores.
        """
        if not self.is_indexed:
            return {"error": "No documents ingested. Please run ingest() first."}

        questions = custom_questions or self.evaluator.get_sample_questions()
        logger.info(f"Evaluating pipeline on {len(questions)} questions")

        qa_pairs = []
        for q_info in questions:
            question = q_info["question"]
            expected = q_info.get("expected_sections", [])

            # Run the query pipeline
            result = self.query(question, top_k=top_k)

            if "error" in result:
                continue

            qa_pairs.append({
                "question": question,
                "answer": result["answer"],
                "contexts": [c["text"] for c in result["retrieved_chunks"]],
                "expected_sections": expected,
                "timing": result["timing"],
            })

        eval_results = self.evaluator.evaluate_batch(qa_pairs)

        # Add timing info
        if qa_pairs:
            avg_retrieval = sum(p["timing"]["retrieval_seconds"] for p in qa_pairs) / len(qa_pairs)
            avg_generation = sum(p["timing"]["generation_seconds"] for p in qa_pairs) / len(qa_pairs)
            eval_results["timing"] = {
                "avg_retrieval_seconds": round(avg_retrieval, 2),
                "avg_generation_seconds": round(avg_generation, 2),
            }

        return eval_results

    def get_status(self) -> Dict:
        """Get current pipeline status."""
        status = {
            "is_indexed": self.is_indexed,
            "documents_ingested": len(self.ingested_documents),
            "total_chunks": len(self.all_chunks),
            "documents": self.ingested_documents,
        }

        if self.is_indexed:
            status["index_stats"] = self.vector_store.get_collection_stats()

        return status

    def get_companies(self) -> List[str]:
        """Get list of ingested company names."""
        return list(set(d["company_name"] for d in self.ingested_documents))
