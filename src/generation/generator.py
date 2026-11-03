"""
Stage 5: LLM Generation

Connects to Ollama for grounded generation with structured prompts,
Chain-of-Thought reasoning, and citation formatting. Model is configurable
(see src/config/loader.py) — defaults to Mistral 7B locally or on a
self-hosted free-tier cloud VM; also tested with Llama 3.2 3B.
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
        model_config=None,
    ):
        """
        Args:
            model_name: Ollama model name (used if model_config not provided).
            ollama_base_url: Ollama API base URL (used if model_config not provided).
            temperature: Generation temperature (low for factual accuracy).
            max_tokens: Maximum tokens in response.
            model_config: Optional ModelConfig object from config loader.
                         Takes precedence over individual parameters.
        """
        # Use config if provided, otherwise fall back to parameters
        if model_config:
            self.model_name = model_config.model
            self.ollama_base_url = model_config.url
            self.temperature = model_config.temperature
            self.max_tokens = model_config.max_tokens
            self.timeout = model_config.timeout
            logger.info(f"Loaded config from ModelConfig: {self.model_name}")
        else:
            self.model_name = model_name
            self.ollama_base_url = ollama_base_url
            self.temperature = temperature
            self.max_tokens = max_tokens
            # 180s for both — see config/models.yaml for measured evidence:
            # full-prospectus generation exceeded even a 120s timeout on a
            # cold-start 7B-parameter model without a dedicated GPU. Remote
            # endpoints would want extra headroom on top of this for network
            # latency, but we don't have a signal here to size that safely,
            # so local and cloud share the same conservative floor.
            self.timeout = 180

        # Verify Ollama/inference service is accessible
        self._verify_connection()

    def _verify_connection(self):
        """Verify connection to Ollama/inference service."""
        try:
            # Try to connect to the service
            resp = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=5
            )
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model_name not in models and not any(self.model_name in m for m in models):
                logger.warning(f"Model '{self.model_name}' not found. Available: {models}")
            else:
                logger.info(f"Service connected. Using model: {self.model_name}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {self.ollama_base_url}. "
                        "Make sure the service is running.")
        except Exception as e:
            logger.error(f"Error verifying connection: {e}")

    def raw_complete(self, prompt: str, system_prompt: str = "", temperature: Optional[float] = None) -> str:
        """
        Send a prompt directly to the model with no context-wrapping or
        financial-analyst system prompt applied. Used for auxiliary tasks
        like LLM-as-judge scoring where the standard citation-heavy
        SYSTEM_PROMPT / CONTEXT_TEMPLATE would corrupt the output.

        Args:
            prompt: The raw prompt to send as the user turn.
            system_prompt: Optional system prompt override (default: none).
            temperature: Optional temperature override.

        Returns:
            The model's raw text response (empty string on error).
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature if temperature is not None else self.temperature,
                        "num_predict": 32,  # scoring responses are short
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"raw_complete failed: {e}")
            return ""

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
                timeout=self.timeout,
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
                timeout=self.timeout,
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
                timeout=self.timeout,
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
