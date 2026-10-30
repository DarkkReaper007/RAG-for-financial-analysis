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
import types
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field

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
    samples: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    """Run reference-free RAGAS metrics directly on saved single-turn samples."""

    # RAGAS 0.2 imports its optional Hugging Face dataset and Vertex AI
    # integrations eagerly. This harness scores SingleTurnSample objects and
    # uses Ollama, so provide import-only placeholders when those extras are
    # absent instead of installing PyArrow or a Vertex AI SDK.
    try:
        from datasets import Dataset as _Dataset  # noqa: F401
    except ImportError:
        datasets_module = types.ModuleType("datasets")
        datasets_module.Dataset = type("Dataset", (), {})
        datasets_module.Sequence = type("Sequence", (), {})
        sys.modules["datasets"] = datasets_module
    try:
        from langchain_community.chat_models.vertexai import (  # noqa: F401
            ChatVertexAI as _ChatVertexAI,
        )
    except ImportError:
        vertex_module = types.ModuleType(
            "langchain_community.chat_models.vertexai"
        )
        vertex_module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[vertex_module.__name__] = vertex_module

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )
    from ragas.metrics._faithfulness import NLIStatementPrompt
    from ragas.run_config import RunConfig

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
            num_ctx=8192,
            # Faithfulness returns the original statements alongside verdicts;
            # long answers can require more than 512 tokens for valid JSON.
            num_predict=2048,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    # Keep RAGAS's faithfulness metric and scoring logic, but use a compact
    # output schema. The default schema/examples elicited long rationales that
    # exceeded Ollama's local generation cap and failed JSON parsing.
    class CompactStatementFaithfulnessAnswer(BaseModel):
        statement: str = Field(..., description="Original statement, unchanged")
        reason: str = Field(..., description="Reason in at most four words")
        verdict: int = Field(..., description="1 if supported, otherwise 0")

    class CompactNLIStatementOutput(BaseModel):
        statements: list[CompactStatementFaithfulnessAnswer]

    nli_prompt = NLIStatementPrompt()
    nli_prompt.instruction = (
        "Judge whether each statement is supported by the context. Preserve each "
        "statement exactly, set verdict to 1 if supported or 0 otherwise, and "
        "give a reason of no more than four words. Return only valid JSON matching "
        "the required schema. Do not add explanations outside the JSON."
    )
    nli_prompt.examples = []
    nli_prompt.output_model = CompactNLIStatementOutput

    metrics = [
        Faithfulness(llm=evaluator_llm, nli_statements_prompt=nli_prompt),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]
    run_config = RunConfig(max_workers=1, timeout=240, max_retries=2)
    for metric in metrics:
        metric.init(run_config)

    prior_records = []
    if progress_path and progress_path.exists():
        prior_records = json.loads(progress_path.read_text(encoding="utf-8")).get(
            "results", []
        )
    records_by_index = {
        row["sample_index"]: row
        for row in prior_records
        if row.get("complete") and isinstance(row.get("sample_index"), int)
    }
    for index, sample in enumerate(samples, 1):
        existing = records_by_index.get(index)
        if existing and all(existing.get(metric.name) is not None for metric in metrics):
            LOGGER.info("RAGAS checkpoint already complete: %s/%s", index, len(samples))
            continue
        LOGGER.info("RAGAS metrics: %s/%s", index, len(samples))
        ragas_sample = SingleTurnSample(
            user_input=sample["user_input"],
            response=sample["response"],
            # Keep ranked retrieval order while bounding local prompt size.
            retrieved_contexts=[
                context[:4000] for context in sample["retrieved_contexts"]
            ],
        )
        row: dict[str, Any] = existing or {
            "sample_index": index,
            "user_input": sample["user_input"],
        }
        errors: dict[str, str] = {}
        for metric in metrics:
            if row.get(metric.name) is not None:
                continue
            try:
                row[metric.name] = metric.single_turn_score(ragas_sample)
            except Exception as exc:
                row[metric.name] = None
                errors[metric.name] = f"{type(exc).__name__}: {exc}"
        if errors:
            row["errors"] = errors
        else:
            row.pop("errors", None)
        row["complete"] = True
        records_by_index[index] = row
        if progress_path:
            write_json(
                progress_path,
                {"framework": "ragas", "version": "0.2.15", "results": list(records_by_index.values())},
            )

    records = [records_by_index[index] for index in sorted(records_by_index)]
    aggregate: dict[str, float | None] = {}
    for metric in metrics:
        values = [
            float(row[metric.name])
            for row in records
            if row.get(metric.name) is not None
        ]
        aggregate[metric.name] = round(sum(values) / len(values), 4) if values else None
    clean_records = [
        {key: value for key, value in row.items() if key != "complete"}
        for row in records
    ]
    return {
        "framework": "ragas",
        "version": "0.2.15",
        "judge_model": model,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "context_limit_chars_per_chunk": 4000,
        "results": clean_records,
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
        "--score-only",
        action="store_true",
        help="Score existing samples.json without re-running ingestion or generation.",
    )
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

    ingestion: dict[str, Any] = {}
    generation_elapsed_seconds: float | None = None
    if args.score_only:
        samples = json.loads((run_dir / "samples.json").read_text(encoding="utf-8"))
        LOGGER.info("Loaded %s saved samples for scoring", len(samples))
    else:
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
        samples = []
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
        generation_elapsed_seconds = round(time.time() - generation_started, 2)

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
    ragas_result = run_ragas(
        samples,
        base_url=args.ollama_url,
        model=args.judge_model,
        progress_path=run_dir / "ragas_progress.json",
    )

    metrics = {
        "branch": os.getenv("GIT_BRANCH", "unknown"),
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "judge_relationship": "same model as generator" if args.generator_model == args.judge_model else "independent model",
        "scoring_resumed_from_saved_samples": args.score_only,
        "question_count": len(samples),
        "ollama_url": args.ollama_url,
        "generation_temperature": args.temperature,
        "top_k": args.top_k,
        "ingestion": {
            "documents": len(ingestion.get("documents", [])),
            "total_chunks": ingestion.get("total_chunks"),
            "timing": ingestion.get("timing"),
        },
        "generation_elapsed_seconds": generation_elapsed_seconds,
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
