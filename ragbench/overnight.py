"""Resumable overnight queue. Every stage is idempotent (finished runs are skipped), so a lane can simply be started again.
  lane A  local compute : study 5 (embedders, dev corpus)
  lane B  API           : generator comparison -> study 4 (verification) -> hierarchy at answer level
  lane C  final         : waits for A and B, then ragbench.final (decisions, leave-one-out, one-shot test run, report)
Keys are read from the environment (GROQ_API_KEYS / NVIDIA_API_KEYS); nothing is written to disk."""
import argparse
import os
import subprocess
import sys
import time

from ragbench.runner import RESULTS

PY = sys.executable
STAGES = {
    "A": [[PY, "-m", "ragbench.ablate", "--study", "5_embedder", "--split", "dev"]],
    "B": [[PY, "-m", "ragbench.ablate", "--generators", "bolt", "nano", "ultra", "--split", "dev"],
          [PY, "-m", "ragbench.ablate", "--study", "4_verification", "--mode", "generation", "--gen", "bolt"],
          [PY, "-m", "ragbench.ablate", "--study", "4_verification", "--mode", "generation", "--gen", "nano"],
          [PY, "-m", "ragbench.ablate", "--study", "1_chunking_hierarchy", "--mode", "generation", "--gen", "bolt"],
          [PY, "-m", "ragbench.ablate", "--study", "1_chunking_hierarchy", "--mode", "generation", "--gen", "nano"]],
    # lane D: confirmatory runs on the held-out test split. Decisions were taken on dev, so none of these can change the final configuration;
    # they add statistical power and replicate the dev findings.
    "D": [[PY, "-m", "ragbench.ablate", "--study", "5_embedder", "--split", "test"]]                       # long local-compute job first (no LLM)
         + [[PY, "-m", "ragbench.ablate", "--study", "1_chunking_hierarchy", "--mode", "generation", "--gen", "bolt", "--split", "test", "--no-judge"]]
         + [[PY, "-m", "ragbench.ceiling", "--gen", g, "--split", "test"] for g in ("nano", "bolt")],
    # FinanceBench (run with RAG_CORPUS=financebench RAG_GOLD=fb1): F = retrieval studies, G = answer-level runs (waits for the baseline index)
    "F": [[PY, "-m", "ragbench.ablate", "--study", st, "--split", "ext"] for st in ("3_retrieval_mode", "1_chunking_hierarchy", "2_parsing_tables")],
    "G": [[PY, "-m", "ragbench.ablate", "--generators", "bolt", "nano", "--split", "ext"]]
         + [[PY, "-m", "ragbench.ablate", "--study", "1_chunking_hierarchy", "--mode", "generation", "--gen", "nano", "--split", "ext"],
            [PY, "-m", "ragbench.ablate", "--study", "4_verification", "--mode", "generation", "--gen", "nano", "--split", "ext"],
            [PY, "-m", "ragbench.ablate", "--study", "1_chunking_hierarchy", "--mode", "generation", "--gen", "bolt", "--split", "ext"]]
         + [[PY, "-m", "ragbench.ceiling", "--gen", "nano", "--split", "ext"], [PY, "-m", "ragbench.judge", "--run", "ceiling__nano__ext__generation"]],
    # lane E: all pending judging + report regeneration. Run when the Groq daily token quota has recovered (rolling 24 h window); resumable via the cache.
    "E": [[PY, "-m", "ragbench.ablate", "--judge-pending"]]
         + [[PY, "-m", "ragbench.judge", "--run", f"ceiling__{g}__test__generation"] for g in ("nano", "bolt")]
         + [[PY, "-m", "ragbench.ablate", "--study", st, "--mode", "generation", "--gen", g, "--split", "test", "--report-only"] for st, g in (("4_verification", "nano"), ("1_chunking_hierarchy", "nano"), ("1_chunking_hierarchy", "bolt"))]
         + [[PY, "-m", "ragbench.pooled"], [PY, "-m", "ragbench.summary"]],
}


def log(lane, msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"[lane {lane}]", msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, choices=["A", "B", "C", "D", "E", "F", "G"])
    ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if a.lane == "G":                                   # needs the baseline index built by lane F's first study
        from ipo_rag.config import BASELINE
        from ipo_rag.corpus import manifest
        from ipo_rag.indexing import index_dir
        while not all((index_dir(BASELINE.ingest, d) / "meta.json").exists() for d in manifest()):
            time.sleep(30)
    if a.lane in STAGES:
        for cmd in STAGES[a.lane]:
            t = time.time()
            log(a.lane, "START " + " ".join(cmd[2:]))
            rc = subprocess.run(cmd, env=env).returncode
            log(a.lane, f"END rc={rc} {time.time() - t:.0f}s")
        (RESULTS / f".lane{a.lane}_done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        log(a.lane, "DONE")
    else:
        while not all((RESULTS / f".lane{l}_done").exists() for l in "AB"):
            time.sleep(60)
            if a.deadline and time.time() > a.deadline:
                log("C", "deadline reached before lanes A/B finished; running final stage with what exists")
                break
        cmd = [PY, "-m", "ragbench.final"] + (["--deadline", str(a.deadline)] if a.deadline else [])
        log("C", "START final")
        rc = subprocess.run(cmd, env=env).returncode
        log("C", f"END rc={rc}")
        (RESULTS / ".laneC_done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
