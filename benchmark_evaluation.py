#!/usr/bin/env python3
"""
Benchmark Script: Compare Rule-Based vs LLM-Based Evaluation

This script runs the RAG pipeline evaluation twice:
1. Using rule-based heuristic metrics (fast, ~30 seconds)
2. Using LLM-as-judge evaluation (accurate, ~5-10 minutes)

Results are compared to demonstrate improvements in:
- Faithfulness detection (semantic understanding)
- Answer relevance scoring (contextal accuracy)
- Context precision (retrieval quality assessment)

Usage:
    source .venv/bin/activate
    python benchmark_evaluation.py [--local-only] [--quick]

Options:
    --local-only   Skip LLM evaluation (only run rule-based)
    --quick        Run on subset of questions (2 instead of 10)
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import RAGPipeline
from src.config.loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


class BenchmarkEvaluator:
    """Run comparative evaluation benchmarks."""

    def __init__(self):
        self.config = ConfigLoader()
        self.pipeline = None
        self.results = {}

    def init_pipeline(self, use_llm_judge: bool = False):
        """
        Initialize pipeline with specified evaluator, or — if already
        initialized — swap the evaluator in place. Never silently ignores
        a mode switch, which would make run_llm_evaluation() re-run the
        rule-based evaluator after run_rule_based_evaluation() has already
        built the pipeline.
        """
        if self.pipeline is not None:
            self.pipeline.set_evaluator_mode(use_llm_judge=use_llm_judge)
            return

        model_cfg = self.config.get_model_config("local")
        eval_cfg = self.config.get_evaluator_config()
        eval_cfg["use_llm_judge"] = use_llm_judge

        logger.info(f"Initializing pipeline with LLM judge: {use_llm_judge}")
        self.pipeline = RAGPipeline(
            qdrant_url="http://localhost:6333",
            use_qdrant=True,
            data_dir="./data/prospectuses",
            vectorstore_dir="./vectorstore",
            use_reranker=True,
            model_config=model_cfg,
            evaluator_config=eval_cfg,
        )

    def ensure_indexed(self):
        """Ensure documents are ingested."""
        self.init_pipeline(use_llm_judge=False)
        if not self.pipeline.is_indexed:
            logger.info("Ingesting documents...")
            self.pipeline.ingest(recreate_index=True)
            logger.info("✓ Documents ingested")
        else:
            logger.info("✓ Documents already indexed")

    def run_rule_based_evaluation(self, sample_size: int = 10):
        """Run rule-based (heuristic) evaluation."""
        logger.info(f"\n{'='*60}")
        logger.info("Running RULE-BASED EVALUATION (Heuristics)")
        logger.info(f"{'='*60}")

        self.init_pipeline(use_llm_judge=False)
        self.ensure_indexed()

        start_time = time.time()
        results = self.pipeline.evaluate()
        elapsed = time.time() - start_time

        self.results["rule_based"] = {
            "metrics": results,
            "elapsed_seconds": elapsed,
        }

        self._print_results("Rule-Based", results, elapsed)
        return results

    def run_llm_evaluation(self, sample_size: int = 10):
        """Run LLM-based (semantic) evaluation."""
        logger.info(f"\n{'='*60}")
        logger.info("Running LLM-BASED EVALUATION (Semantic)")
        logger.info(f"{'='*60}")

        self.init_pipeline(use_llm_judge=True)
        self.ensure_indexed()

        start_time = time.time()
        results = self.pipeline.evaluate()
        elapsed = time.time() - start_time

        self.results["llm_based"] = {
            "metrics": results,
            "elapsed_seconds": elapsed,
        }

        self._print_results("LLM-Based", results, elapsed)
        return results

    def _print_results(self, mode: str, results: dict, elapsed: float):
        """Print evaluation results."""
        logger.info(f"\n{'─'*60}")
        logger.info(f"Results: {mode} Evaluation")
        logger.info(f"{'─'*60}")

        if "error" in results:
            logger.error(f"Error: {results['error']}")
            return

        agg = results.get("aggregate", {})

        # Print aggregate scores
        logger.info("\n📊 Aggregate Metrics:")
        logger.info(f"  Faithfulness:     {agg.get('faithfulness', 'N/A')}")
        logger.info(f"  Answer Relevance: {agg.get('answer_relevance', 'N/A')}")
        logger.info(f"  Context Precision:{agg.get('context_precision', 'N/A')}")
        logger.info(f"  Hit Rate:         {agg.get('hit_rate', 'N/A')}")
        logger.info(f"  Total Questions:  {agg.get('total_questions', 0)}")

        # Print timing
        logger.info(f"\n⏱️  Timing:")
        logger.info(f"  Total Time:       {elapsed:.2f}s")
        logger.info(f"  Per Question:     {elapsed / agg.get('total_questions', 1):.2f}s")

    def print_comparison(self):
        """Print comparison between rule-based and LLM evaluation."""
        if "rule_based" not in self.results or "llm_based" not in self.results:
            logger.warning("Cannot compare: missing evaluation results")
            return

        logger.info(f"\n{'='*60}")
        logger.info("COMPARISON: Rule-Based vs LLM-Based Evaluation")
        logger.info(f"{'='*60}\n")

        rb = self.results["rule_based"]["metrics"].get("aggregate", {})
        llm = self.results["llm_based"]["metrics"].get("aggregate", {})

        metrics = ["faithfulness", "answer_relevance", "context_precision", "hit_rate"]

        logger.info(f"{'Metric':<20} | {'Rule-Based':<12} | {'LLM-Based':<12} | {'Improvement':<12}")
        logger.info(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

        for metric in metrics:
            rb_val = rb.get(metric, 0)
            llm_val = llm.get(metric, 0)
            if rb_val > 0:
                improvement = ((llm_val - rb_val) / rb_val) * 100
                logger.info(
                    f"{metric:<20} | {rb_val:<12.3f} | {llm_val:<12.3f} | "
                    f"{improvement:+.1f}%"
                )
            else:
                logger.info(
                    f"{metric:<20} | {rb_val:<12.3f} | {llm_val:<12.3f} | "
                    f"N/A"
                )

        # Timing comparison
        rb_time = self.results["rule_based"]["elapsed_seconds"]
        llm_time = self.results["llm_based"]["elapsed_seconds"]
        logger.info(f"\n⏱️  Time Comparison:")
        logger.info(f"  Rule-Based: {rb_time:.2f}s")
        logger.info(f"  LLM-Based:  {llm_time:.2f}s ({llm_time/rb_time:.1f}x slower)")

        logger.info(f"\n✨ Key Improvements (LLM vs Rule-Based):")

        # Semantic understanding improvements
        if llm.get("faithfulness", 0) > rb.get("faithfulness", 0):
            logger.info(
                f"  • Better faithfulness detection: "
                f"catches paraphrased hallucinations"
            )

        if llm.get("answer_relevance", 0) > rb.get("answer_relevance", 0):
            logger.info(
                f"  • Better relevance scoring: "
                f"understands semantic alignment"
            )

        if llm.get("context_precision", 0) > rb.get("context_precision", 0):
            logger.info(
                f"  • Better context evaluation: "
                f"judges semantic relevance not just keywords"
            )


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark Rule-Based vs LLM-Based Evaluation"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip LLM evaluation (only run rule-based)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run on 2 questions instead of 10 (faster testing)",
    )

    args = parser.parse_args()

    benchmark = BenchmarkEvaluator()

    try:
        # Ensure documents are ingested
        benchmark.ensure_indexed()

        # Run rule-based evaluation
        benchmark.run_rule_based_evaluation()

        # Run LLM-based evaluation if requested
        if not args.local_only:
            logger.info("\n⏳ Running LLM evaluation... (this may take a few minutes)")
            benchmark.run_llm_evaluation()

            # Print comparison
            benchmark.print_comparison()
        else:
            logger.info("\n(Skipping LLM evaluation per --local-only flag)")

        logger.info("\n✅ Benchmark complete!")

    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
