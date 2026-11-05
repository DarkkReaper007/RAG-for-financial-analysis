#!/bin/zsh
# FinanceBench with the US 10-K layout labels (RAG_STRUCTURE=sec) + oracle runs for the SEBI-label baseline. Retrieval only, sequential (memory).
cd "$(dirname "$0")/.."; source .venv/bin/activate
export RAG_CORPUS=financebench RAG_GOLD=fb1
R=(python -m ragbench.runner --split ext --mode retrieval)
RAG_STRUCTURE=sebi "${R[@]}" --name oracle_sebi --oracle
RAG_STRUCTURE=sec "${R[@]}" --name sec_hybrid
RAG_STRUCTURE=sec "${R[@]}" --name sec_hybrid --oracle
RAG_STRUCTURE=sec "${R[@]}" --name sec_dense --set retrieval_mode=dense
RAG_STRUCTURE=sec "${R[@]}" --name sec_bm25 --set retrieval_mode=bm25
RAG_STRUCTURE=sec "${R[@]}" --name sec_section_flat --set hierarchy=flat
RAG_STRUCTURE=sec "${R[@]}" --name sec_section_parent_ctx --set contextual_header=true
RAG_STRUCTURE=sec "${R[@]}" --name sec_fixed_flat --set chunker=fixed hierarchy=flat
echo ALLDONE
