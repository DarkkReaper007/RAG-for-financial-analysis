"""LLM backends (Ollama native or any OpenAI-compatible endpoint) and the answer prompt. Failures are returned, never disguised as answers."""
import os
import time

import requests

from ipo_rag.config import GenConfig

SYSTEM = """You are a financial analyst answering questions about Indian IPO offer documents (Draft Red Herring Prospectus / Red Herring Prospectus / Prospectus filed with SEBI).
Rules:
1. Use ONLY the numbered context passages. Never use outside knowledge.
2. Cite every fact as [Company, Page: N] using the page numbers shown in the context headers.
3. Quote figures exactly as written, with their unit (e.g. "₹ 1,470.30 million"); do not convert units or round.
4. A value shown as [●] has not been determined yet in a draft document; say so rather than guessing.
5. If the context does not contain the answer, reply exactly: "The provided documents do not contain this information."
6. Be concise."""


def format_context(contexts, names):
    blocks = []
    for i, c in enumerate(contexts, 1):
        pg = c["page_start"] if c["page_start"] == c["page_end"] else f"{c['page_start']}-{c['page_end']}"
        blocks.append(f"[Context {i}] Company: {names[c['doc']]} | Section: {(c['chapter'] or 'n/a').title()} | Page: {pg}\n{c['text']}")
    return "\n\n".join(blocks)


def generate(cfg: GenConfig, question, context_block, extra=None):
    """-> {"answer", "failed", "error", "seconds", "tokens"}"""
    user = f"{context_block}\n\nQuestion: {question}\n\nAnswer:" + (f"\n\n{extra}" if extra else "")
    t = time.time()
    try:
        if cfg.backend == "ollama":
            r = requests.post(f"{cfg.url}/api/generate", json={"model": cfg.model, "system": SYSTEM, "prompt": user, "stream": False,
                              "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens}}, timeout=cfg.timeout)
            r.raise_for_status()
            j = r.json()
            text, toks = j.get("response", ""), j.get("eval_count")
        else:
            r = requests.post(f"{cfg.url}/chat/completions", headers={"Authorization": "Bearer " + os.environ[cfg.api_key_env]},
                              json={"model": cfg.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                    "temperature": cfg.temperature, "max_tokens": cfg.max_tokens}, timeout=cfg.timeout)
            r.raise_for_status()
            j = r.json()
            text, toks = j["choices"][0]["message"].get("content") or "", (j.get("usage") or {}).get("completion_tokens")
        text = text.strip()
        return {"answer": text, "failed": not text, "error": None if text else "empty_reply", "seconds": round(time.time() - t, 2), "tokens": toks}
    except Exception as e:
        return {"answer": "", "failed": True, "error": f"{type(e).__name__}: {e}"[:200], "seconds": round(time.time() - t, 2), "tokens": None}
