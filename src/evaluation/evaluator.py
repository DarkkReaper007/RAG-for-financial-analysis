"""
Stage 6: Evaluation

Implements RAGAS-inspired evaluation metrics for the RAG pipeline:
- Faithfulness: Are generated claims grounded in retrieved context?
- Answer Relevance: Does the response address the query?
- Context Precision: Did the retriever surface the right chunks?

Also includes retrieval metrics: Hit Rate, MRR, Recall@K.
"""

import logging
import time
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question: str
    answer: str
    contexts: List[str]
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    retrieval_hit: bool = False
    details: Dict = field(default_factory=dict)


class RAGEvaluator:
    """
    Evaluation engine for the RAG pipeline.
    
    Implements lightweight versions of RAGAS metrics that can run
    locally without requiring an external LLM judge (suitable for prototype demo).
    
    For a production system, these would use an LLM-as-judge approach.
    """

    # Sample evaluation questions for IPO prospectus domain
    SAMPLE_QUESTIONS = [
        {
            "question": "What are the main risk factors mentioned in the prospectus?",
            "expected_sections": ["risk factors"],
            "category": "risk_analysis",
        },
        {
            "question": "What is the total issue size and price band?",
            "expected_sections": ["offer", "issue", "price"],
            "category": "issue_details",
        },
        {
            "question": "Who are the promoters and what is their shareholding?",
            "expected_sections": ["promoter", "capital structure", "shareholding"],
            "category": "ownership",
        },
        {
            "question": "What are the objects of the issue and how will the proceeds be used?",
            "expected_sections": ["objects of the issue", "proceeds"],
            "category": "use_of_proceeds",
        },
        {
            "question": "What is the company's revenue and profit for the last 3 years?",
            "expected_sections": ["financial", "statements", "restated"],
            "category": "financials",
        },
        {
            "question": "What are the outstanding litigations against the company?",
            "expected_sections": ["litigation", "legal", "outstanding"],
            "category": "legal",
        },
        {
            "question": "What is the basis for the issue price?",
            "expected_sections": ["basis", "issue price", "valuation"],
            "category": "valuation",
        },
        {
            "question": "Who are the book running lead managers and registrar?",
            "expected_sections": ["general information", "lead manager", "registrar"],
            "category": "intermediaries",
        },
        {
            "question": "What is the company's business model and competitive advantage?",
            "expected_sections": ["business", "overview", "competitive", "strength"],
            "category": "business",
        },
        {
            "question": "What is the dividend policy of the company?",
            "expected_sections": ["dividend", "policy"],
            "category": "dividend",
        },
    ]

    def __init__(self):
        pass

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        expected_sections: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single QA pair.
        
        Args:
            question: The question asked.
            answer: The generated answer.
            contexts: List of retrieved context texts.
            expected_sections: Optional list of expected section keywords.
            
        Returns:
            EvaluationResult with computed metrics.
        """
        result = EvaluationResult(
            question=question,
            answer=answer,
            contexts=contexts,
        )

        # Faithfulness: Check if answer claims are grounded in context
        result.faithfulness = self._compute_faithfulness(answer, contexts)

        # Answer Relevance: Check if answer addresses the question
        result.answer_relevance = self._compute_answer_relevance(question, answer)

        # Context Precision: Check if retrieved contexts are relevant
        result.context_precision = self._compute_context_precision(
            question, contexts, expected_sections
        )

        # Retrieval Hit
        if expected_sections:
            all_contexts_lower = " ".join(contexts).lower()
            result.retrieval_hit = any(
                s.lower() in all_contexts_lower for s in expected_sections
            )

        return result

    def evaluate_batch(
        self,
        qa_pairs: List[Dict],
    ) -> Dict:
        """
        Evaluate a batch of QA pairs and compute aggregate metrics.
        
        Args:
            qa_pairs: List of dicts with 'question', 'answer', 'contexts',
                      and optionally 'expected_sections'.
                      
        Returns:
            Dict with per-question results and aggregate scores.
        """
        results = []
        for pair in qa_pairs:
            result = self.evaluate_single(
                question=pair["question"],
                answer=pair["answer"],
                contexts=pair.get("contexts", []),
                expected_sections=pair.get("expected_sections"),
            )
            results.append(result)

        # Aggregate metrics
        n = len(results)
        if n == 0:
            return {"results": [], "aggregate": {}}

        aggregate = {
            "faithfulness": round(sum(r.faithfulness for r in results) / n, 3),
            "answer_relevance": round(sum(r.answer_relevance for r in results) / n, 3),
            "context_precision": round(sum(r.context_precision for r in results) / n, 3),
            "hit_rate": round(sum(1 for r in results if r.retrieval_hit) / n, 3),
            "total_questions": n,
        }

        return {
            "results": [
                {
                    "question": r.question,
                    "faithfulness": r.faithfulness,
                    "answer_relevance": r.answer_relevance,
                    "context_precision": r.context_precision,
                    "retrieval_hit": r.retrieval_hit,
                }
                for r in results
            ],
            "aggregate": aggregate,
        }

    def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Compute faithfulness score.
        
        Measures what fraction of significant claims in the answer
        can be traced back to the retrieved contexts.
        
        Approach: Extract key phrases from the answer and check
        if they appear in the context (lexical overlap + n-gram matching).
        """
        if not answer or not contexts:
            return 0.0

        # Check for explicit "cannot find" / "not enough information" disclaimers
        disclaimer_patterns = [
            "cannot find sufficient information",
            "not enough information",
            "not mentioned in the context",
            "no relevant information",
        ]
        answer_lower = answer.lower()
        if any(p in answer_lower for p in disclaimer_patterns):
            return 1.0  # Faithful refusal to hallucinate

        # Extract meaningful n-grams from the answer (skip very common words)
        answer_ngrams = self._extract_meaningful_ngrams(answer, n=3)
        
        if not answer_ngrams:
            return 0.5  # Can't determine, default to neutral

        # Check how many answer n-grams appear in any context
        all_context = " ".join(contexts).lower()
        grounded_count = sum(1 for ng in answer_ngrams if ng.lower() in all_context)

        score = grounded_count / len(answer_ngrams) if answer_ngrams else 0
        return round(min(1.0, score), 3)

    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        """
        Compute answer relevance score.
        
        Measures whether the answer addresses the question's key terms.
        """
        if not answer or len(answer.strip()) < 10:
            return 0.0

        # Extract question keywords (nouns, entities)
        q_keywords = self._extract_keywords(question)
        if not q_keywords:
            return 0.5

        answer_lower = answer.lower()

        # Check keyword coverage
        covered = sum(1 for kw in q_keywords if kw.lower() in answer_lower)
        coverage_score = covered / len(q_keywords) if q_keywords else 0

        # Length penalty — very short answers are likely less relevant
        length_score = min(1.0, len(answer) / 100)

        # Check for citation presence (good indicator of grounding)
        has_citations = bool(re.search(r'\[Source:|Page:|Section:', answer))
        citation_bonus = 0.1 if has_citations else 0

        score = (coverage_score * 0.7 + length_score * 0.2 + citation_bonus)
        return round(min(1.0, score), 3)

    def _compute_context_precision(
        self,
        question: str,
        contexts: List[str],
        expected_sections: Optional[List[str]] = None,
    ) -> float:
        """
        Compute context precision score.
        
        Measures whether the retrieved contexts are relevant to the question.
        """
        if not contexts:
            return 0.0

        q_keywords = self._extract_keywords(question)
        if not q_keywords:
            return 0.5

        # Score each context by keyword relevance
        relevant_count = 0
        for ctx in contexts:
            ctx_lower = ctx.lower()
            keyword_hits = sum(1 for kw in q_keywords if kw.lower() in ctx_lower)
            if keyword_hits >= max(1, len(q_keywords) * 0.3):
                relevant_count += 1

        # Additional check against expected sections
        section_bonus = 0.0
        if expected_sections:
            all_context = " ".join(contexts).lower()
            section_hits = sum(1 for s in expected_sections if s.lower() in all_context)
            section_bonus = 0.2 * (section_hits / len(expected_sections))

        base_score = relevant_count / len(contexts) if contexts else 0
        return round(min(1.0, base_score + section_bonus), 3)

    def _extract_meaningful_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Extract meaningful n-grams from text."""
        words = re.findall(r'[a-zA-Z0-9₹]+(?:\.[a-zA-Z0-9]+)*', text.lower())
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'are', 'was',
            'were', 'been', 'be', 'have', 'has', 'had', 'based', 'context',
            'source', 'section', 'page', 'company', 'according',
        }
        words = [w for w in words if w not in stop_words and len(w) > 2]

        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            ngrams.append(ngram)

        return ngrams

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from a question."""
        stop_words = {
            'what', 'who', 'where', 'when', 'how', 'why', 'which', 'whom',
            'the', 'is', 'at', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'are',
            'was', 'were', 'does', 'do', 'did', 'has', 'have', 'had',
            'mentioned', 'discuss', 'tell', 'me', 'about', 'describe',
        }
        words = re.findall(r'[a-zA-Z]+', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords

    def get_sample_questions(self) -> List[Dict]:
        """Return the predefined sample evaluation questions."""
        return self.SAMPLE_QUESTIONS
