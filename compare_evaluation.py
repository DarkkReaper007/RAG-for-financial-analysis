#!/usr/bin/env python3
"""Compare generator models with one shared LLM judge and actual RAGAS metrics.

The script saves each generated answer and retrieved context before scoring so
both evaluators score the exact same outputs. Run it once per generator model.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import RAGEvaluator
from src.pipeline import RAGPipeline

LOGGER = logging.getLogger("model_comparison")

FAITHFULNESS_PROMPT = """Score how much of the answer is supported by the supplied context.
Question: {question}
Answer: {answer}
Context: {context}
Return one number from 0.0 (unsupported) to 1.0 (all important claims supported)."""

RELEVANCE_PROMPT = """Score how directly and completely the answer addresses the question.
Question: {question}
Answer: {answer}
Return one number from 0.0 (irrelevant) to 1.0 (fully addresses it)."""

CONTEXT_PRECISION_PROMPT = """Score the relevance of these retrieved contexts to the question.
Question: {question}
Retrieved contexts, in rank order:
{context}
Return one number from 0.0 (irrelevant) to 1.0 (all highly relevant)."""


def write_json(path: Path, value: Any) -> None:
    def json_safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): json_safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(val) for val in item]
        if hasattr(item, "item"):
            item = item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def score_with_ollama(
    *, base_url: str, model: str, prompt: str, timeout: int
) -> tuple[float | None, str | None]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "system": "Return only one decimal score between 0 and 1.",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 32},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
        if not match:
            return None, f"Could not parse numeric score from {text!r}"
        return min(1.0, max(0.0, float(match.group(0)))), None
    except Exception as exc:  # Save failures rather than silently assigning neutral scores.
        return None, f"{type(exc).__name__}: {exc}"


def run_custom_judge(
    samples: list[dict[str, Any]], *, base_url: str, model: str, timeout: int
) -> dict[str, Any]:
    fields = {
        "faithfulness": FAITHFULNESS_PROMPT,
        "answer_relevance": RELEVANCE_PROMPT,
        "context_precision": CONTEXT_PRECISION_PROMPT,
    }
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, 1):
        LOGGER.info("Custom LLM judge: %s/%s", index, len(samples))
        context = "\n\n".join(sample["retrieved_contexts"])
        row: dict[str, Any] = {"question": sample["user_input"]}
        errors: dict[str, str] = {}
        for metric, template in fields.items():
            score, error = score_with_ollama(
                base_url=base_url,
                model=model,
                prompt=template.format(
                    question=sample["user_input"],
                    answer=sample["response"],
                    context=context[:12000],
                ),
                timeout=timeout,
            )
            row[metric] = score
            if error:
                errors[metric] = error
        if errors:
            row["errors"] = errors
        rows.append(row)

    aggregate: dict[str, float | None] = {}
    for metric in fields:
        values = [row[metric] for row in rows if row[metric] is not None]
        aggregate[metric] = round(sum(values) / len(values), 4) if values else None
    return {"judge_model": model, "results": rows, "aggregate": aggregate}


def run_ragas(
    samples: list[dict[str, Any]], *, base_url: str, model: str
) -> dict[str, Any]:
    """Run reference-free RAGAS metrics against saved answers and contexts."""
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
            num_ctx=8192,
            num_predict=512,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]
    dataset = Dataset.from_list(
        [
            {
                "user_input": sample["user_input"],
                "response": sample["response"],
                # Keep the full ranked list while bounding the prompt size for
                # local evaluator models with limited context/VRAM.
                "retrieved_contexts": [
                    context[:4000] for context in sample["retrieved_contexts"]
                ],
            }
            for sample in samples
        ]
    )
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=1, timeout=240, max_retries=2),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = results.to_pandas()
    records = frame.to_dict(orient="records")
    aggregate: dict[str, float | None] = {}
    input_columns = {"user_input", "response", "retrieved_contexts", "reference"}
    for metric in frame.columns:
        if metric in input_columns:
            continue
        values = [float(value) for value in frame[metric].dropna().tolist()]
        aggregate[str(metric)] = round(sum(values) / len(values), 4) if values else None
    return {
        "framework": "ragas",
        "version": "0.2.15",
        "judge_model": model,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "context_limit_chars_per_chunk": 4000,
        "results": records,
        "aggregate": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--judge-model", default="mistral:7b")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--use-config",
        action="store_true",
        help="Use the branch's ModelConfig path (for comparing feature branch behavior).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(run_dir / "benchmark.log", encoding="utf-8"), logging.StreamHandler()],
    )
    LOGGER.info("Starting %s with generator=%s, judge=%s", run_dir.name, args.generator_model, args.judge_model)

    pipeline_kwargs: dict[str, Any] = {}
    if args.use_config:
        from src.config.loader import ConfigLoader

        model_config = ConfigLoader().get_model_config("local")
        model_config.model = args.generator_model
        model_config.url = args.ollama_url
        pipeline_kwargs["model_config"] = model_config

    pipeline = RAGPipeline(
        ollama_model=args.generator_model,
        ollama_url=args.ollama_url,
        data_dir=str(REPO_ROOT / "data" / "prospectuses"),
        vectorstore_dir=str(run_dir / "vectorstore"),
        use_qdrant=False,
        use_reranker=True,
        **pipeline_kwargs,
    )
    # Align both model runs independently of branch-specific config defaults.
    pipeline.generator.temperature = args.temperature
    pipeline.generator.max_tokens = 2048
    ingestion = pipeline.ingest(recreate_index=True)
    if "error" in ingestion:
        raise RuntimeError(ingestion["error"])

    questions = RAGEvaluator.SAMPLE_QUESTIONS
    samples: list[dict[str, Any]] = []
    generation_started = time.time()
    for index, item in enumerate(questions, 1):
        question = item["question"]
        LOGGER.info("Generating answer %s/%s: %s", index, len(questions), question)
        result = pipeline.query(question, top_k=args.top_k)
        if "error" in result:
            raise RuntimeError(f"Query failed for {question!r}: {result['error']}")
        samples.append(
            {
                "user_input": question,
                "response": result["answer"],
                "retrieved_contexts": [chunk["text"] for chunk in result["retrieved_chunks"]],
                "expected_sections": item.get("expected_sections", []),
                "timing": result.get("timing", {}),
            }
        )
        write_json(run_dir / "samples.json", samples)

    rule_pairs = [
        {
            "question": sample["user_input"],
            "answer": sample["response"],
            "contexts": sample["retrieved_contexts"],
            "expected_sections": sample["expected_sections"],
        }
        for sample in samples
    ]
    rule_metrics = RAGEvaluator().evaluate_batch(rule_pairs)
    custom_judge = run_custom_judge(
        samples,
        base_url=args.ollama_url,
        model=args.judge_model,
        timeout=args.timeout,
    )
    ragas_result = run_ragas(samples, base_url=args.ollama_url, model=args.judge_model)

    metrics = {
        "branch": os.getenv("GIT_BRANCH", "unknown"),
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "judge_relationship": "same model as generator" if args.generator_model == args.judge_model else "independent model",
        "question_count": len(samples),
        "ollama_url": args.ollama_url,
        "generation_temperature": args.temperature,
        "top_k": args.top_k,
        "ingestion": {
            "documents": len(ingestion.get("documents", [])),
            "total_chunks": ingestion.get("total_chunks"),
            "timing": ingestion.get("timing"),
        },
        "generation_elapsed_seconds": round(time.time() - generation_started, 2),
        "native_rule_metrics": rule_metrics,
        "custom_llm_judge": custom_judge,
        "ragas": ragas_result,
    }
    write_json(run_dir / "metrics.json", metrics)
    LOGGER.info("Saved samples and metrics under %s", run_dir)
    LOGGER.info("Native rule aggregate: %s", rule_metrics.get("aggregate"))
    LOGGER.info("Custom judge aggregate: %s", custom_judge.get("aggregate"))
    LOGGER.info("RAGAS aggregate: %s", ragas_result.get("aggregate"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
