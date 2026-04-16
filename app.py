"""
RAG System for Indian IPO Prospectuses — Gradio Web Interface

A polished, presentation-ready demo UI with:
- Document ingestion status dashboard
- Interactive query interface with streaming responses
- Pipeline visualization showing retrieval steps
- RAGAS evaluation dashboard with metrics
"""

import os
import sys
import time
import logging
import gradio as gr
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ─── Global Pipeline Instance ────────────────────────────────────────────
pipeline = None


def get_pipeline():
    """Lazy-initialize the pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline(
            qdrant_url="http://localhost:6333",
            use_qdrant=True,
            ollama_model="llama3.2:latest",
            ollama_url="http://localhost:11434",
            data_dir="./data/prospectuses",
            vectorstore_dir="./vectorstore",
            use_reranker=True,
        )
    return pipeline


# ─── Event Handlers ──────────────────────────────────────────────────────

def ingest_documents():
    """Ingest all PDFs from the data directory."""
    try:
        pipe = get_pipeline()
        stats = pipe.ingest(recreate_index=True)

        if "error" in stats:
            return f"❌ Error: {stats['error']}", ""

        # Build status report
        summary_lines = [
            f"## ✅ Ingestion Complete\n",
            f"**Total Documents:** {len(stats['documents'])}",
            f"**Total Chunks:** {stats['total_chunks']}",
            f"**Time Taken:** {stats['timing']['total_seconds']}s\n",
            f"### Vector Store",
            f"- Backend: {stats['index_stats'].get('backend', 'N/A')}",
            f"- Indexed Vectors: {stats['index_stats'].get('total_vectors', 'N/A')}",
            f"- Status: {stats['index_stats'].get('status', 'N/A')}\n",
        ]

        # Per-document details
        doc_table = "### Documents\n\n"
        doc_table += "| Document | Company | Pages | Sections | Chunks | Time |\n"
        doc_table += "|----------|---------|-------|----------|--------|------|\n"
        for doc in stats["documents"]:
            doc_table += (
                f"| {doc['filename'][:25]}... | {doc['company_name'][:20]}... | "
                f"{doc['total_pages']} | {doc['sections']} | "
                f"{doc['chunks']['total']} | {doc['timing']['total_seconds']}s |\n"
            )
        summary_lines.append(doc_table)

        return "\n".join(summary_lines), _get_status_html()

    except Exception as e:
        logger.exception("Ingestion failed")
        return f"❌ Ingestion failed: {str(e)}", ""


def query_rag(question, company_filter, top_k):
    """Query the RAG pipeline."""
    if not question.strip():
        return "Please enter a question.", "", ""

    try:
        pipe = get_pipeline()
        if not pipe.is_indexed:
            return "⚠️ Please ingest documents first.", "", ""

        company = company_filter if company_filter and company_filter != "All Companies" else None
        top_k = int(top_k) if top_k else 5

        result = pipe.query(question, top_k=top_k, company_filter=company)

        if "error" in result:
            return f"❌ {result['error']}", "", ""

        # Format answer
        answer_md = result["answer"]

        # Format sources
        sources_md = "### 📚 Sources\n\n"
        for src in result.get("sources", []):
            sources_md += (
                f"- **{src['company']}** — {src['section']} "
                f"(Page {src['page']}, File: {src['filename']})\n"
            )

        # Format retrieved chunks
        chunks_md = "### 🔍 Retrieved Chunks\n\n"
        for i, chunk in enumerate(result.get("retrieved_chunks", []), 1):
            score = chunk.get("score", 0)
            chunks_md += (
                f"**Chunk {i}** | Score: `{score}` | "
                f"Type: `{chunk['chunk_type']}` | "
                f"Section: _{chunk['section']}_ | Page: {chunk['page']}\n\n"
                f"```\n{chunk['text'][:300]}...\n```\n\n---\n\n"
            )

        # Format pipeline info
        pipeline_md = "### ⚙️ Pipeline Stages\n\n"
        pipeline_md += f"**Total Time:** {result['timing']['total_seconds']}s "
        pipeline_md += f"(Retrieval: {result['timing']['retrieval_seconds']}s, "
        pipeline_md += f"Generation: {result['timing']['generation_seconds']}s)\n\n"

        for stage in result.get("pipeline_info", {}).get("stages", []):
            name = stage.get("name", "")
            detail = ""
            if "results_count" in stage:
                detail = f" → {stage['results_count']} results"
            elif "output" in stage:
                detail = f" → {stage['output']}"
            elif "strategy" in stage:
                detail = f" → {stage['strategy']}"
            pipeline_md += f"- ✓ **{name}**{detail}\n"

        return answer_md, sources_md + "\n\n" + chunks_md, pipeline_md

    except Exception as e:
        logger.exception("Query failed")
        return f"❌ Error: {str(e)}", "", ""


def stream_query(question, company_filter, top_k):
    """Stream a query response for the Gradio UI."""
    if not question.strip():
        yield "Please enter a question."
        return

    try:
        pipe = get_pipeline()
        if not pipe.is_indexed:
            yield "⚠️ Please ingest documents first."
            return

        company = company_filter if company_filter and company_filter != "All Companies" else None
        top_k_val = int(top_k) if top_k else 5

        accumulated = ""
        for token in pipe.query_stream(question, top_k=top_k_val, company_filter=company):
            accumulated += token
            yield accumulated

    except Exception as e:
        yield f"❌ Error: {str(e)}"


def run_evaluation():
    """Run RAGAS evaluation on sample questions."""
    try:
        pipe = get_pipeline()
        if not pipe.is_indexed:
            return "⚠️ Please ingest documents first.", ""

        eval_results = pipe.evaluate()

        if "error" in eval_results:
            return f"❌ {eval_results['error']}", ""

        agg = eval_results.get("aggregate", {})

        # Aggregate metrics display
        metrics_md = "## 📊 Evaluation Results\n\n"
        metrics_md += "### Aggregate Scores\n\n"
        metrics_md += "| Metric | Score |\n"
        metrics_md += "|--------|-------|\n"
        metrics_md += f"| **Faithfulness** | {agg.get('faithfulness', 'N/A')} |\n"
        metrics_md += f"| **Answer Relevance** | {agg.get('answer_relevance', 'N/A')} |\n"
        metrics_md += f"| **Context Precision** | {agg.get('context_precision', 'N/A')} |\n"
        metrics_md += f"| **Hit Rate** | {agg.get('hit_rate', 'N/A')} |\n"
        metrics_md += f"| **Total Questions** | {agg.get('total_questions', 0)} |\n\n"

        if "timing" in eval_results:
            metrics_md += "### Timing\n\n"
            metrics_md += f"- Avg Retrieval: {eval_results['timing']['avg_retrieval_seconds']}s\n"
            metrics_md += f"- Avg Generation: {eval_results['timing']['avg_generation_seconds']}s\n\n"

        # Per-question breakdown
        details_md = "### Per-Question Breakdown\n\n"
        details_md += "| Question | Faithfulness | Relevance | Precision | Hit |\n"
        details_md += "|----------|-------------|-----------|-----------|-----|\n"

        for r in eval_results.get("results", []):
            hit = "✅" if r.get("retrieval_hit") else "❌"
            details_md += (
                f"| {r['question'][:50]}... | {r['faithfulness']} | "
                f"{r['answer_relevance']} | {r['context_precision']} | {hit} |\n"
            )

        return metrics_md, details_md

    except Exception as e:
        logger.exception("Evaluation failed")
        return f"❌ Evaluation failed: {str(e)}", ""


def get_companies():
    """Get list of ingested companies for the filter dropdown."""
    try:
        pipe = get_pipeline()
        companies = pipe.get_companies()
        return gr.update(choices=["All Companies"] + companies, value="All Companies")
    except Exception:
        return gr.update(choices=["All Companies"], value="All Companies")


def _get_status_html():
    """Get pipeline status as formatted markdown."""
    try:
        pipe = get_pipeline()
        status = pipe.get_status()
        if not status["is_indexed"]:
            return "⏳ **Status:** Not indexed. Click 'Ingest Documents' to begin."

        return (
            f"✅ **Status:** Ready | "
            f"**Documents:** {status['documents_ingested']} | "
            f"**Chunks:** {status['total_chunks']} | "
            f"**Backend:** {status.get('index_stats', {}).get('backend', 'N/A')}"
        )
    except Exception:
        return "⏳ Pipeline not initialized"


# ─── Gradio App ──────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "What are the main risk factors mentioned in the prospectus?",
    "What is the total issue size and price band for the IPO?",
    "Who are the promoters and what is their shareholding percentage?",
    "What are the objects of the issue and how will the proceeds be used?",
    "What is the company's revenue and profit for the last 3 years?",
    "What are the outstanding litigations against the company?",
    "What is the basis for the issue price?",
    "What is the company's business model and competitive advantage?",
    "Who are the book running lead managers?",
    "What is the dividend policy of the company?",
]

CUSTOM_CSS = """
/* ─── Global Theme ─── */
.gradio-container {
    max-width: 1400px !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ─── Header Styling ─── */
.header-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.header-banner h1 {
    color: #e0e0ff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: -0.02em;
}

.header-banner p {
    color: #a8a8d0 !important;
    font-size: 0.95rem !important;
    margin: 0.3rem 0 0 0 !important;
    line-height: 1.4;
}

/* ─── Status Bar ─── */
.status-bar {
    background: linear-gradient(135deg, #1a1a3e, #2a2a5e);
    padding: 0.8rem 1.5rem;
    border-radius: 10px;
    border: 1px solid rgba(100, 100, 255, 0.15);
    font-size: 0.9rem;
}

/* ─── Tab Styling ─── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 8px 8px 0 0 !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #302b63, #24243e) !important;
    color: #c0c0ff !important;
    border-bottom: 3px solid #7c6cf0 !important;
}

/* ─── Input Styling ─── */
.query-input textarea {
    font-size: 1.05rem !important;
    border: 2px solid rgba(124, 108, 240, 0.3) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease;
}

.query-input textarea:focus {
    border-color: #7c6cf0 !important;
    box-shadow: 0 0 0 3px rgba(124, 108, 240, 0.15) !important;
}

/* ─── Button Styling ─── */
.primary-btn {
    background: linear-gradient(135deg, #7c6cf0, #5a4fcf) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    padding: 0.6rem 1.5rem !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 108, 240, 0.4) !important;
}

/* ─── Card Styling ─── */
.result-card {
    background: linear-gradient(135deg, #1e1e3f, #252547);
    border: 1px solid rgba(124, 108, 240, 0.2);
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

/* ─── Markdown Output ─── */
.markdown-output {
    line-height: 1.7 !important;
}

.markdown-output table {
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
}

.markdown-output th, .markdown-output td {
    padding: 0.5rem 0.8rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: left;
}

.markdown-output th {
    background: rgba(124, 108, 240, 0.15);
    font-weight: 600;
}

/* ─── Pipeline Architecture Diagram ─── */
.architecture-text {
    font-family: 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
}
"""

PIPELINE_DIAGRAM = """
```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐        │
│  │  Stage 1  │───▶│   Stage 2     │───▶│    Stage 3          │        │
│  │ PDF Parse │    │ Hierarchical │    │ BGE-small Embed    │        │
│  │ (PyMuPDF) │    │  Chunking    │    │ + Qdrant Index     │        │
│  └──────────┘    └──────────────┘    └────────────────────┘        │
│                                              │                      │
│                                              ▼                      │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐        │
│  │  Stage 6  │◀───│   Stage 5     │◀───│    Stage 4          │        │
│  │  RAGAS    │    │ Llama 3.2 3B │    │ Hybrid Retrieval   │        │
│  │  Eval     │    │ Generation   │    │ Dense+BM25+Rerank  │        │
│  └──────────┘    └──────────────┘    └────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```
"""

def build_app():
    """Build the Gradio application."""

    with gr.Blocks(
        title="RAG System for Indian IPO Prospectuses",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
    ) as app:

        # ─── Header ──────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-banner">
            <h1>🏦 RAG System for Indian IPO Prospectuses</h1>
            <p>Retrieval-Augmented Generation for SEBI Financial Documents • Capstone Project</p>
            <p style="font-size: 0.8rem; margin-top: 0.5rem; color: #8888bb;">
                Llama 3.2 3B • BGE-small-en-v1.5 • Qdrant • Hybrid Retrieval • RAGAS Evaluation
            </p>
        </div>
        """)

        status_display = gr.Markdown(
            value=_get_status_html(),
            elem_classes=["status-bar"],
        )

        # ─── Tabs ─────────────────────────────────────────────────────
        with gr.Tabs():

            # ─── Tab 1: Documents & Ingestion ──────────────────────────
            with gr.Tab("📁 Documents", id="documents"):
                gr.Markdown("### Document Ingestion\nParse, chunk, embed, and index IPO prospectus PDFs.")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(PIPELINE_DIAGRAM)

                    with gr.Column(scale=1):
                        ingest_btn = gr.Button(
                            "🚀 Ingest Documents",
                            variant="primary",
                            elem_classes=["primary-btn"],
                            size="lg",
                        )
                        gr.Markdown(
                            "*Processes all PDFs in `data/prospectuses/`. "
                            "This may take a few minutes for large documents.*"
                        )

                ingest_output = gr.Markdown(label="Ingestion Results")

                ingest_btn.click(
                    fn=ingest_documents,
                    outputs=[ingest_output, status_display],
                    show_progress="full",
                )

            # ─── Tab 2: Query Interface ────────────────────────────────
            with gr.Tab("💬 Query", id="query"):
                gr.Markdown("### Ask a Question\nQuery the ingested IPO prospectuses using the full RAG pipeline.")

                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What are the main risk factors mentioned in the prospectus?",
                            lines=2,
                            elem_classes=["query-input"],
                        )
                    with gr.Column(scale=1):
                        company_dropdown = gr.Dropdown(
                            label="Filter by Company",
                            choices=["All Companies"],
                            value="All Companies",
                            interactive=True,
                        )
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Chunks to Retrieve (top-k)",
                        )

                with gr.Row():
                    query_btn = gr.Button(
                        "🔍 Search & Generate",
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )
                    stream_btn = gr.Button(
                        "⚡ Stream Response",
                        variant="secondary",
                    )
                    refresh_btn = gr.Button("🔄 Refresh Companies", size="sm")

                gr.Markdown("### Example Questions")
                example_btns = gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUESTIONS],
                    inputs=[question_input],
                    label="",
                )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=2):
                        answer_output = gr.Markdown(
                            label="Answer",
                            elem_classes=["markdown-output"],
                        )
                    with gr.Column(scale=1):
                        with gr.Accordion("📚 Sources & Chunks", open=False):
                            sources_output = gr.Markdown(elem_classes=["markdown-output"])
                        with gr.Accordion("⚙️ Pipeline Details", open=False):
                            pipeline_output = gr.Markdown(elem_classes=["markdown-output"])

                # Event handlers
                query_btn.click(
                    fn=query_rag,
                    inputs=[question_input, company_dropdown, top_k_slider],
                    outputs=[answer_output, sources_output, pipeline_output],
                    show_progress="full",
                )

                stream_btn.click(
                    fn=stream_query,
                    inputs=[question_input, company_dropdown, top_k_slider],
                    outputs=[answer_output],
                    show_progress="minimal",
                )

                refresh_btn.click(
                    fn=get_companies,
                    outputs=[company_dropdown],
                )

            # ─── Tab 3: Evaluation ─────────────────────────────────────
            with gr.Tab("📊 Evaluation", id="evaluation"):
                gr.Markdown(
                    "### RAGAS Evaluation Framework\n"
                    "Run the evaluation suite to measure **Faithfulness**, "
                    "**Answer Relevance**, and **Context Precision** across "
                    "10 sample financial questions."
                )

                eval_btn = gr.Button(
                    "▶️ Run Evaluation",
                    variant="primary",
                    elem_classes=["primary-btn"],
                    size="lg",
                )
                gr.Markdown(
                    "*⏱️ This will query the pipeline 10 times — may take several minutes.*"
                )

                with gr.Row():
                    with gr.Column():
                        eval_metrics = gr.Markdown(
                            label="Aggregate Metrics",
                            elem_classes=["markdown-output"],
                        )
                    with gr.Column():
                        eval_details = gr.Markdown(
                            label="Per-Question Breakdown",
                            elem_classes=["markdown-output"],
                        )

                eval_btn.click(
                    fn=run_evaluation,
                    outputs=[eval_metrics, eval_details],
                    show_progress="full",
                )

            # ─── Tab 4: Architecture ──────────────────────────────────
            with gr.Tab("🏗️ Architecture", id="architecture"):
                gr.Markdown("""
### System Architecture

This prototype implements a **6-stage modular RAG pipeline** as described in the capstone synopsis.

---

#### Stage 1: Document Ingestion & PDF Parsing
- **Tool:** PyMuPDF (fitz)
- **Approach:** Structure-aware parsing with font-size heuristics for header detection
- **Output:** Hierarchical sections with metadata (page numbers, font info, table boundaries)

#### Stage 2: Hierarchical Chunking  
- **Strategy:** HiChunk Auto-Merge Pattern
- **Parent Chunks:** Full sections for context-rich generation
- **Child Chunks:** ~1000-char semantic sub-sections for precise retrieval
- **Table Chunks:** Separately extracted financial tables

#### Stage 3: Embedding & Indexing
- **Embedding Model:** BAAI/bge-small-en-v1.5 (384-dimensional)
- **Vector Store:** Qdrant with HNSW indexing (Docker)
- **Sparse Index:** BM25 for exact keyword matching
- **Metadata:** Company name, section title, page numbers for filtered search

#### Stage 4: Hybrid Retrieval
- **Dense Search:** BGE cosine similarity via Qdrant
- **Sparse Search:** BM25 keyword matching
- **Fusion:** Reciprocal Rank Fusion (RRF) merging
- **Reranking:** Cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Lost-in-the-Middle:** Primacy-recency chunk placement

#### Stage 5: Grounded Generation
- **LLM:** Llama 3.2 3B via Ollama (local inference)
- **Prompting:** Structured financial analyst prompt with CoT reasoning
- **Citations:** Inline source references [Company, Section, Page]
- **Parent Expansion:** Child chunks expanded to parent context for generation

#### Stage 6: RAGAS Evaluation
- **Faithfulness:** Grounding of claims in retrieved context
- **Answer Relevance:** Query-answer alignment 
- **Context Precision:** Retrieval quality measurement
- **Retrieval Metrics:** Hit Rate & MRR
                """)

    return app


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
