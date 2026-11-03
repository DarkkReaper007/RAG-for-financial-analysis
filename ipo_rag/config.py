"""Pipeline configuration. Every ablation factor is a named field; `ABLATION_LEVELS` lists the studied levels."""
import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class IngestConfig:
    """Everything that changes the built index. `key()` names the on-disk index, so each variant is built once."""
    table_mode: str = "markdown"      # inline | flat | markdown            (study 2: parsing & table representation)
    chunker: str = "section"          # fixed | section                     (study 1: chunking & hierarchy)
    hierarchy: str = "parent"         # flat | parent
    contextual_header: bool = False   # prepend "[company | chapter | page]" to chunk text before embedding/BM25
    child_chars: int = 1000           # same size for fixed and section so the study isolates boundaries, not size
    child_overlap: int = 150
    parent_chars: int = 3000
    embedder: str = "bge-small"       # bge-small | bge-large | e5-large    (study 5: embedder)

    def key(self):
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class QueryConfig:
    retrieval_mode: str = "hybrid"    # dense | bm25 | hybrid               (study 3: retrieval mode)
    top_k: int = 5
    candidate_depth: int = 2          # candidates per retriever = top_k * depth (held fixed)
    rrf_dense_weight: float = 1.0     # standard equal-weight RRF. (0.6/0.4 with k=60 made hybrid == dense: BM25-only hits could never
    rrf_sparse_weight: float = 1.0    #  outrank a dense hit within the candidate cut-off; found by ragbench, see tests)
    rrf_k: int = 60
    rerank: bool = True               # held fixed
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reorder: bool = True              # lost-in-the-middle ordering, held fixed
    parent_cap_chars: int = 4000      # parent text longer than this falls back to the child chunk
    verify: str = "off"               # off | on                            (study 4: numeric & citation verification)


@dataclass(frozen=True)
class GenConfig:
    backend: str = "openai"           # ollama | openai (any OpenAI-compatible endpoint, e.g. NVIDIA)
    model: str = "nvidia/nemotron-3.5-bolt-e4b"   # generator is decided later; this is a placeholder
    url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 240


@dataclass(frozen=True)
class PipelineConfig:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    gen: GenConfig = field(default_factory=GenConfig)


# The five studies. Each is varied one factor at a time against BASELINE; everything else stays fixed.
BASELINE = PipelineConfig()
ABLATION_LEVELS = {
    "1_chunking_hierarchy": {
        "fixed_flat": dict(chunker="fixed", hierarchy="flat"),
        "section_flat": dict(chunker="section", hierarchy="flat"),
        "section_parent": dict(chunker="section", hierarchy="parent"),
        "section_parent_ctx": dict(chunker="section", hierarchy="parent", contextual_header=True),
    },
    "2_parsing_tables": {
        "inline": dict(table_mode="inline"),
        "flat": dict(table_mode="flat"),
        "markdown": dict(table_mode="markdown"),
        # "markdown_summary": planned (needs an LLM table-summary step)
    },
    "3_retrieval_mode": {m: dict(retrieval_mode=m) for m in ("dense", "bm25", "hybrid")},
    "4_verification": {v: dict(verify=v) for v in ("off", "on")},
    "5_embedder": {e: dict(embedder=e) for e in ("bge-small", "bge-large", "e5-large")},
}


def with_changes(cfg: PipelineConfig, **kw) -> PipelineConfig:
    """Return cfg with fields replaced, routing each name to the sub-config that owns it."""
    from dataclasses import replace
    parts = {"ingest": {}, "query": {}, "gen": {}}
    for k, v in kw.items():
        owner = next(p for p, c in (("ingest", cfg.ingest), ("query", cfg.query), ("gen", cfg.gen)) if hasattr(c, k))
        parts[owner][k] = v
    return PipelineConfig(**{p: replace(getattr(cfg, p), **parts[p]) for p in parts})
