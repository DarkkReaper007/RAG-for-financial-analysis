"""
Stage 3: Embedding & Vector Store Indexing

Embeds chunks using BAAI/bge-small-en-v1.5 and stores them in Qdrant
vector database with HNSW indexing including metadata for filtered retrieval.
"""

import logging
from typing import List, Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Qdrant imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed, will use ChromaDB fallback")

from sentence_transformers import SentenceTransformer

from src.chunking.hierarchical_chunker import Chunk


class VectorStoreManager:
    """
    Manages embedding generation and vector storage using Qdrant (with ChromaDB fallback).
    
    Uses BAAI/bge-small-en-v1.5 for dense embeddings and stores vectors
    in Qdrant with HNSW indexing and metadata payload filtering.
    """

    COLLECTION_NAME = "ipo_prospectuses"
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM = 384  # bge-small output dimension

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        use_qdrant: bool = True,
        persist_dir: str = "./vectorstore",
    ):
        """
        Args:
            qdrant_url: URL of the Qdrant server.
            use_qdrant: Whether to use Qdrant (True) or ChromaDB fallback (False).
            persist_dir: Directory for ChromaDB persistence (fallback only).
        """
        self.use_qdrant = use_qdrant and QDRANT_AVAILABLE
        self.persist_dir = persist_dir
        self._chunks_store: Dict[str, Chunk] = {}  # In-memory chunk lookup

        # Load embedding model
        logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(self.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")

        # Initialize vector store
        if self.use_qdrant:
            try:
                self.qdrant = QdrantClient(url=qdrant_url)
                # Test connection
                self.qdrant.get_collections()
                logger.info(f"Connected to Qdrant at {qdrant_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Qdrant: {e}. Falling back to ChromaDB.")
                self.use_qdrant = False

        if not self.use_qdrant:
            import chromadb
            self.chroma = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"Using ChromaDB at {persist_dir}")

    def create_collection(self, recreate: bool = False):
        """Create or recreate the vector collection."""
        if self.use_qdrant:
            collections = [c.name for c in self.qdrant.get_collections().collections]
            if self.COLLECTION_NAME in collections:
                if recreate:
                    self.qdrant.delete_collection(self.COLLECTION_NAME)
                    logger.info(f"Deleted existing Qdrant collection: {self.COLLECTION_NAME}")
                else:
                    logger.info(f"Using existing Qdrant collection: {self.COLLECTION_NAME}")
                    return

            self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                    # HNSW parameters tuned for corpus of 100k+ chunks
                    hnsw_config={
                        "m": 16,
                        "ef_construct": 200,
                    } if False else None,  # Use defaults for prototype
                ),
            )
            logger.info(f"Created Qdrant collection: {self.COLLECTION_NAME}")
        else:
            if recreate:
                try:
                    self.chroma.delete_collection(self.COLLECTION_NAME)
                except Exception:
                    pass
            self.collection = self.chroma.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Created ChromaDB collection: {self.COLLECTION_NAME}")

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 64):
        """
        Embed and index chunks into the vector store.
        
        Args:
            chunks: List of Chunk objects to index.
            batch_size: Number of chunks to process per batch.
        """
        # Filter to only child and table chunks for retrieval
        # (parent chunks are stored separately for context expansion)
        retrievable_chunks = [c for c in chunks if c.chunk_type in ("child", "table")]
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]

        # Store all chunks in memory for lookup (including parents)
        for chunk in chunks:
            self._chunks_store[chunk.chunk_id] = chunk

        logger.info(f"Indexing {len(retrievable_chunks)} retrievable chunks "
                     f"({len(parent_chunks)} parent chunks stored for context)")

        # Prepare texts for embedding with BGE instruction prefix
        texts = [f"Represent this document for retrieval: {c.text}" for c in retrievable_chunks]

        # Batch embed
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.embed_model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.extend(embeddings)

        # Index into vector store
        if self.use_qdrant:
            self._index_qdrant(retrievable_chunks, all_embeddings, batch_size)
        else:
            self._index_chroma(retrievable_chunks, all_embeddings, batch_size)

        logger.info("Indexing complete")

    def _index_qdrant(self, chunks: List[Chunk], embeddings, batch_size: int):
        """Index chunks into Qdrant."""
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = chunk.to_dict()
            # Remove the text from payload to save space (stored separately)
            points.append(PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=payload,
            ))

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=batch,
            )

    def _index_chroma(self, chunks: List[Chunk], embeddings, batch_size: int):
        """Index chunks into ChromaDB."""
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            self.collection.add(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=[e.tolist() for e in batch_embeddings],
                documents=[c.text for c in batch_chunks],
                metadatas=[{
                    k: str(v) for k, v in c.to_dict().items()
                    if k != "text" and v is not None
                } for c in batch_chunks],
            )

    def search(
        self,
        query: str,
        top_k: int = 10,
        company_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search for relevant chunks using dense vector similarity.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            company_filter: Optional company name filter.
            
        Returns:
            List of dicts with 'chunk', 'score', and metadata.
        """
        # Encode query with BGE query instruction prefix
        query_embedding = self.embed_model.encode(
            f"Represent this sentence for searching relevant passages: {query}"
        ).tolist()

        if self.use_qdrant:
            return self._search_qdrant(query_embedding, top_k, company_filter)
        else:
            return self._search_chroma(query_embedding, top_k, company_filter)

    def _search_qdrant(
        self, query_embedding, top_k: int, company_filter: Optional[str]
    ) -> List[Dict]:
        """Search Qdrant with optional metadata filtering."""
        query_filter = None
        if company_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="company_name",
                        match=MatchValue(value=company_filter),
                    )
                ]
            )

        results = self.qdrant.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "chunk_id": hit.payload.get("chunk_id", ""),
                "parent_id": hit.payload.get("parent_id", ""),
                "company_name": hit.payload.get("company_name", ""),
                "section_title": hit.payload.get("section_title", ""),
                "page_start": hit.payload.get("page_start", 0),
                "chunk_type": hit.payload.get("chunk_type", ""),
                "filename": hit.payload.get("filename", ""),
            }
            for hit in results
        ]

    def _search_chroma(
        self, query_embedding, top_k: int, company_filter: Optional[str]
    ) -> List[Dict]:
        """Search ChromaDB with optional metadata filtering."""
        where_filter = None
        if company_filter:
            where_filter = {"company_name": company_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                search_results.append({
                    "text": doc,
                    "score": 1 - distance,  # Convert distance to similarity
                    "chunk_id": meta.get("chunk_id", ""),
                    "parent_id": meta.get("parent_id", ""),
                    "company_name": meta.get("company_name", ""),
                    "section_title": meta.get("section_title", ""),
                    "page_start": int(meta.get("page_start", 0)),
                    "chunk_type": meta.get("chunk_type", ""),
                    "filename": meta.get("filename", ""),
                })

        return search_results

    def get_parent_chunk(self, parent_id: str) -> Optional[Chunk]:
        """Retrieve a parent chunk by ID for context expansion."""
        return self._chunks_store.get(parent_id)

    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed collection."""
        if self.use_qdrant:
            info = self.qdrant.get_collection(self.COLLECTION_NAME)
            return {
                "backend": "Qdrant",
                "total_vectors": info.points_count,
                "status": info.status.name if hasattr(info.status, 'name') else str(info.status),
            }
        else:
            count = self.collection.count()
            return {
                "backend": "ChromaDB",
                "total_vectors": count,
                "status": "ready",
            }
