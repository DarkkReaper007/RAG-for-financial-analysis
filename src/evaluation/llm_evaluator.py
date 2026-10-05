"""
Stage 6: LLM-Based Evaluation using RAGAS Framework

Implements production-grade evaluation using LLM-as-judge for:
- Faithfulness: Are generated claims grounded in retrieved context?
- Answer Relevance: Does the response address the query?
- Context Precision: Are retrieved chunks relevant to the question?
- Context Recall: Are all relevant chunks retrieved?
"""

import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvaluationScore:
    """Holds evaluation scores for a single QA pair."""
    question: str
    answer: str
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    retrieval_hit: bool = False
    details: Dict = field(default_factory=dict)


class LLMEvaluator:
    """
    LLM-based RAG evaluator using RAGAS framework.

    Uses an LLM (via Ollama) as a judge to evaluate:
    - Faithfulness of generated answers to context
    - Relevance of answers to questions
    - Precision of retrieved context
    - Recall of relevant context

    This is a semantic evaluation approach that better correlates
    with human judgment than rule-based heuristics.
    """

    # Evaluation prompts for LLM judge
    FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether statements in an answer are grounded in the provided context.

Question: {question}
Answer: {answer}
Context: {context}

Task: Evaluate if the claims made in the answer are directly supported by the context provided.
- If claims are paraphrased versions of context, they count as grounded.
- If claims are inferences based on context, they count as grounded.
- If claims contradict the context or go beyond it, they do not count.

Provide a faithfulness score from 0.0 to 1.0, where:
- 1.0 = All major claims are fully grounded in context
- 0.8 = Most claims grounded, minor details not in context
- 0.6 = About half the claims grounded
- 0.4 = Few claims grounded
- 0.2 = Minimal grounding
- 0.0 = No claims grounded / completely hallucinated

Respond with ONLY the score as a decimal (e.g., 0.85)"""

    RELEVANCE_PROMPT = """You are an expert evaluator assessing whether an answer adequately addresses a question.

Question: {question}
Answer: {answer}

Task: Evaluate if the answer directly and comprehensively addresses the question asked.
- Consider if key aspects of the question are covered.
- Consider if the answer is focused on the question (not off-topic).
- Consider if the answer provides sufficient detail.

Provide a relevance score from 0.0 to 1.0, where:
- 1.0 = Answer completely and thoroughly addresses the question
- 0.8 = Answer addresses most aspects of the question
- 0.6 = Answer addresses some aspects, missing some detail
- 0.4 = Answer partially addresses the question
- 0.2 = Answer barely touches the question
- 0.0 = Answer is completely irrelevant

Respond with ONLY the score as a decimal (e.g., 0.85)"""

    PRECISION_PROMPT = """You are an expert evaluator assessing if retrieved context is relevant to a question.

Question: {question}
Retrieved Context: {context}

Task: Evaluate if the retrieved context contains information relevant to answering the question.
- Each context chunk should be evaluated for relevance.
- Even partial relevance counts as precision.

Provide a precision score from 0.0 to 1.0, where:
- 1.0 = All context is highly relevant to the question
- 0.8 = Most context is relevant, some tangential information
- 0.6 = About half the context is relevant
- 0.4 = Some context is relevant, mostly irrelevant
- 0.2 = Minimal relevant context
- 0.0 = No relevant context

Respond with ONLY the score as a decimal (e.g., 0.85)"""

    def __init__(self, llm_judge=None):
        """
        Initialize LLM evaluator.

        Args:
            llm_judge: Optional LLM instance to use as judge.
                      If None, will attempt to import from generation module.
        """
        self.llm_judge = llm_judge
        self._init_judge()

    def _init_judge(self):
        """Initialize the LLM judge if not already provided."""
        if self.llm_judge is None:
            try:
                from src.generation.generator import RAGGenerator
                logger.info("Initializing LLM judge using RAGGenerator")
                self.llm_judge = RAGGenerator()
            except Exception as e:
                logger.warning(f"Could not initialize LLM judge: {e}. "
                              "Will use fallback evaluation.")
                self.llm_judge = None

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        expected_sections: Optional[List[str]] = None,
    ) -> EvaluationScore:
        """
        Evaluate a single QA pair using LLM judge.

        Args:
            question: The question asked.
            answer: The generated answer.
            contexts: List of retrieved context texts.
            expected_sections: Optional list of expected section keywords.

        Returns:
            EvaluationScore with LLM-based metrics.
        """
        score = EvaluationScore(question=question, answer=answer)

        if not self.llm_judge:
            logger.warning("LLM judge not available, using fallback scores")
            score.faithfulness = 0.5
            score.answer_relevance = 0.5
            score.context_precision = 0.5
            score.context_recall = 0.5
            return score

        context_text = " ".join(contexts) if contexts else ""

        try:
            # Faithfulness evaluation
            logger.debug("Evaluating faithfulness...")
            faith_prompt = self.FAITHFULNESS_PROMPT.format(
                question=question,
                answer=answer,
                context=context_text[:2000]  # Limit context length
            )
            faith_score = self._get_score_from_llm(faith_prompt)
            score.faithfulness = faith_score

            # Answer Relevance evaluation
            logger.debug("Evaluating answer relevance...")
            rel_prompt = self.RELEVANCE_PROMPT.format(
                question=question,
                answer=answer
            )
            rel_score = self._get_score_from_llm(rel_prompt)
            score.answer_relevance = rel_score

            # Context Precision evaluation
            logger.debug("Evaluating context precision...")
            prec_prompt = self.PRECISION_PROMPT.format(
                question=question,
                context=context_text[:2000]
            )
            prec_score = self._get_score_from_llm(prec_prompt)
            score.context_precision = prec_score

            # Context Recall (simple implementation)
            # In production, would need ground truth relevant chunks
            if expected_sections:
                all_contexts_lower = context_text.lower()
                hits = sum(1 for s in expected_sections if s.lower() in all_contexts_lower)
                score.context_recall = min(1.0, hits / len(expected_sections))
                score.retrieval_hit = hits > 0

        except Exception as e:
            logger.error(f"Error during LLM evaluation: {e}")
            # Return neutral scores on error
            score.faithfulness = 0.5
            score.answer_relevance = 0.5
            score.context_precision = 0.5

        return score

    def evaluate_batch(
        self,
        qa_pairs: List[Dict],
    ) -> Dict:
        """
        Evaluate a batch of QA pairs using LLM judge.

        Args:
            qa_pairs: List of dicts with 'question', 'answer', 'contexts',
                     and optionally 'expected_sections'.

        Returns:
            Dict with per-question results and aggregate scores.
        """
        logger.info(f"Running LLM-based evaluation on {len(qa_pairs)} questions")
        start_time = time.time()

        results = []
        for i, pair in enumerate(qa_pairs, 1):
            logger.info(f"  Evaluating question {i}/{len(qa_pairs)}...")

            score = self.evaluate_single(
                question=pair["question"],
                answer=pair["answer"],
                contexts=pair.get("contexts", []),
                expected_sections=pair.get("expected_sections"),
            )
            results.append(score)

        # Compute aggregate metrics
        n = len(results)
        if n == 0:
            return {"results": [], "aggregate": {}, "timing": {}}

        aggregate = {
            "faithfulness": round(sum(r.faithfulness for r in results) / n, 3),
            "answer_relevance": round(sum(r.answer_relevance for r in results) / n, 3),
            "context_precision": round(sum(r.context_precision for r in results) / n, 3),
            "context_recall": round(sum(r.context_recall for r in results) / n, 3),
            "hit_rate": round(sum(1 for r in results if r.retrieval_hit) / n, 3),
            "total_questions": n,
        }

        elapsed = time.time() - start_time

        return {
            "results": [
                {
                    "question": r.question,
                    "faithfulness": r.faithfulness,
                    "answer_relevance": r.answer_relevance,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "retrieval_hit": r.retrieval_hit,
                }
                for r in results
            ],
            "aggregate": aggregate,
            "timing": {
                "total_seconds": round(elapsed, 2),
                "avg_per_question": round(elapsed / n, 2),
            },
        }

    JUDGE_SYSTEM_PROMPT = (
        "You are a strict, impartial evaluator. Respond with ONLY a single "
        "decimal number between 0.0 and 1.0. No words, no explanation, no "
        "markdown — just the number."
    )

    def _get_score_from_llm(self, prompt: str) -> float:
        """
        Get a score from the LLM judge.

        Uses RAGGenerator.raw_complete() so the evaluation prompt is sent
        as-is, without being wrapped in the financial-analyst system prompt
        or citation-context template used for normal answer generation.

        Args:
            prompt: The evaluation prompt.

        Returns:
            Float score between 0.0 and 1.0.
        """
        import re

        if not self.llm_judge:
            return 0.5

        try:
            answer = self.llm_judge.raw_complete(
                prompt=prompt,
                system_prompt=self.JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
            ).strip()

            # Extract the first decimal number found in the response,
            # tolerating stray words/punctuation the model might add.
            match = re.search(r"(\d+(?:\.\d+)?)", answer)
            if match:
                score = float(match.group(1))
                # Model may occasionally answer on a 0-10 scale; normalize.
                if score > 1.0:
                    score = score / 10.0
                return min(1.0, max(0.0, score))

            logger.warning(f"Could not parse score from LLM response: {answer!r}")
            return 0.5

        except Exception as e:
            logger.error(f"Error getting score from LLM: {e}")
            return 0.5


class HybridEvaluator:
    """
    Hybrid evaluator that can switch between rule-based and LLM-based evaluation.
    Provides a unified interface for both approaches.
    """

    def __init__(self, use_llm_judge: bool = False, quick_mode: bool = False, llm_judge=None):
        """
        Initialize hybrid evaluator.

        Args:
            use_llm_judge: Whether to use LLM-based evaluation.
            quick_mode: Whether to use quick (rule-based) evaluation.
            llm_judge: Optional existing RAGGenerator to reuse as the judge
                      (avoids spinning up a second, differently-configured
                      model connection). If None, LLMEvaluator creates its
                      own default (local Ollama) instance.
        """
        self.use_llm_judge = use_llm_judge and not quick_mode
        self.quick_mode = quick_mode

        # The rule-based evaluator is always constructed — it's cheap (no
        # model loading) and is the sole source of the sample question bank,
        # regardless of which evaluator scores the answers.
        from .evaluator import RAGEvaluator
        self.rule_eval = RAGEvaluator()

        if self.use_llm_judge:
            self.llm_eval = LLMEvaluator(llm_judge=llm_judge)
            logger.info("Using LLM-based evaluator")
        else:
            self.llm_eval = None
            logger.info("Using rule-based evaluator")

    def evaluate_batch(self, qa_pairs: List[Dict]) -> Dict:
        """Evaluate batch of QA pairs."""
        if self.use_llm_judge:
            return self.llm_eval.evaluate_batch(qa_pairs)
        else:
            return self.rule_eval.evaluate_batch(qa_pairs)

    def get_sample_questions(self) -> List[Dict]:
        """Get sample evaluation questions."""
        return self.rule_eval.get_sample_questions()
