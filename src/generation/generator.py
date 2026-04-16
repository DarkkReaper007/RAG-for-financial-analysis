"""
Stage 5: LLM Generation

Connects to Ollama (Llama 3.2 3B) for grounded generation with
structured prompts, Chain-of-Thought reasoning, and citation formatting.
"""

import logging
import time
import json
import requests
from typing import List, Dict, Optional, Generator

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    LLM-based generation engine using Ollama for grounded financial QA.
    
    Features:
    - Structured prompts with retrieved context and citation instructions
    - Chain-of-Thought (CoT) reasoning for numerical consistency
    - Parent chunk expansion for broader context
    - Streaming response support
    """

    SYSTEM_PROMPT = """You are an expert financial analyst specializing in Indian IPO prospectuses filed with SEBI (Securities and Exchange Board of India).

Your task is to answer questions accurately based ONLY on the provided context from IPO prospectus documents. Follow these rules strictly:

1. **Ground all answers in the provided context.** Do not use any external knowledge.
2. **Cite your sources** using the format: [Source: Company, Section: SectionName, Page: PageNumber]
3. **For numerical questions**, show your reasoning step-by-step (Chain-of-Thought):
   - Extract the relevant numbers from the context
   - Show the calculation
   - State the result
4. **If the context does not contain enough information** to answer the question, say: "Based on the available context, I cannot find sufficient information to answer this question."
5. **Never fabricate financial figures, dates, or regulatory references.**
6. Use clear, professional language appropriate for financial analysis.
7. When multiple documents are relevant, compare and contrast the information.

Remember: Accuracy and faithfulness to the source documents are paramount."""

    CONTEXT_TEMPLATE = """--- Retrieved Context ---
{context_blocks}
--- End of Context ---

Question: {question}

Based on the context above, provide a detailed and accurate answer with proper citations:"""

    def __init__(
        self,
        model_name: str = "llama3.2:latest",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Args:
            model_name: Ollama model name.
            ollama_base_url: Ollama API base URL.
            temperature: Generation temperature (low for factual accuracy).
            max_tokens: Maximum tokens in response.
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Verify Ollama is accessible
        try:
            resp = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            if model_name not in models and not any(model_name in m for m in models):
                logger.warning(f"Model '{model_name}' not found in Ollama. Available: {models}")
            else:
                logger.info(f"Ollama connected. Using model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        stream: bool = False,
    ) -> Dict:
        """
        Generate an answer using the LLM with retrieved context.
        
        Args:
            question: User question.
            retrieved_chunks: List of retrieved chunk dicts from the retriever.
            stream: Whether to stream the response.
            
        Returns:
            Dict with 'answer', 'sources', 'timing', and 'model_info'.
        """
        start_time = time.time()

        # Build context blocks with source annotations
        context_blocks = self._build_context(retrieved_chunks)

        # Build the full prompt
        user_prompt = self.CONTEXT_TEMPLATE.format(
            context_blocks=context_blocks,
            question=question,
        )

        # Generate response
        if stream:
            return self._generate_stream(user_prompt, start_time)
        else:
            return self._generate_sync(user_prompt, start_time, retrieved_chunks)

    def generate_stream(
        self,
        question: str,
        retrieved_chunks: List[Dict],
    ) -> Generator[str, None, None]:
        """
        Stream the generated response token by token.
        
        Yields:
            Individual tokens/chunks of the response.
        """
        context_blocks = self._build_context(retrieved_chunks)
        user_prompt = self.CONTEXT_TEMPLATE.format(
            context_blocks=context_blocks,
            question=question,
        )

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": user_prompt,
                    "system": self.SYSTEM_PROMPT,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break

        except Exception as e:
            yield f"\n\n[Error generating response: {e}]"

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build formatted context string from retrieved chunks."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            company = chunk.get("company_name", "Unknown")
            section = chunk.get("section_title", "Unknown Section")
            page = chunk.get("page_start", "?")
            chunk_type = chunk.get("chunk_type", "text")
            text = chunk.get("text", "")

            # Use parent text if available for richer context
            if chunk.get("parent_text") and len(chunk["parent_text"]) < 3000:
                text = chunk["parent_text"]

            header = f"[Context {i}] Company: {company} | Section: {section} | Page: {page}"
            if chunk_type == "table":
                header += " | Type: Financial Table"

            context_parts.append(f"{header}\n{text}")

        return "\n\n".join(context_parts)

    def _generate_sync(
        self,
        user_prompt: str,
        start_time: float,
        retrieved_chunks: List[Dict],
    ) -> Dict:
        """Synchronous generation via Ollama API."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": user_prompt,
                    "system": self.SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            answer = data.get("response", "")
            gen_time = time.time() - start_time

            # Extract source citations from the answer
            sources = self._extract_sources(retrieved_chunks)

            return {
                "answer": answer,
                "sources": sources,
                "timing": {
                    "total_seconds": round(gen_time, 2),
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration_ms": data.get("eval_duration", 0) / 1e6,
                },
                "model_info": {
                    "model": self.model_name,
                    "temperature": self.temperature,
                },
                "context_used": len(retrieved_chunks),
            }

        except requests.exceptions.ConnectionError:
            return {
                "answer": "Error: Could not connect to Ollama. Please ensure Ollama is running.",
                "sources": [],
                "timing": {"total_seconds": time.time() - start_time},
                "model_info": {"model": self.model_name},
                "context_used": 0,
            }
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "timing": {"total_seconds": time.time() - start_time},
                "model_info": {"model": self.model_name},
                "context_used": 0,
            }

    def _generate_stream(self, user_prompt: str, start_time: float) -> Dict:
        """Streaming generation (collects full response)."""
        tokens = list(self.generate_stream_raw(user_prompt))
        answer = "".join(tokens)
        return {
            "answer": answer,
            "sources": [],
            "timing": {"total_seconds": round(time.time() - start_time, 2)},
            "model_info": {"model": self.model_name},
        }

    def generate_stream_raw(self, user_prompt: str) -> Generator[str, None, None]:
        """Raw streaming from Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": user_prompt,
                    "system": self.SYSTEM_PROMPT,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                stream=True,
                timeout=120,
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")
                    if data.get("done"):
                        break
        except Exception as e:
            yield f"[Error: {e}]"

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract formatted source references from retrieved chunks."""
        sources = []
        seen = set()

        for chunk in chunks:
            company = chunk.get("company_name", "Unknown")
            section = chunk.get("section_title", "Unknown")
            page = chunk.get("page_start", "?")
            filename = chunk.get("filename", "")

            key = f"{company}-{section}-{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "company": company,
                    "section": section,
                    "page": page,
                    "filename": filename,
                })

        return sources
