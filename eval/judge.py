"""LLM-judge saved generation results with ONE fixed judge (same prompts for every arm).

    python eval/judge.py eval/results/C_mistral.json --judge-model qwen2.5-coder:7b-instruct-q4_K_M

Metrics (0-1):
  faithfulness       claim-level: answer split into atomic claims, each verified against the context
  relevance          does the answer address the question
  correctness        vs reference_answer (only when labelled)
  context_relevance  per retrieved chunk yes/no (RAG-triad "context relevance")
Failed generations and (for faithfulness/correctness) abstentions on unanswerable questions are not scored as answers.
"""
import argparse, hashlib, json, os, re, requests, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLAIMS = ("List the distinct factual claims made in the ANSWER below, one per line, max 8, no numbering, "
          "no commentary. Ignore citations and hedging.\n\nANSWER:\n{ans}\n\nCLAIMS:")
VERIFY = ("CONTEXT:\n{ctx}\n\nCLAIM: {claim}\n\nIs the claim fully supported by the CONTEXT? Reply with exactly one word: yes or no.")
RELEV = ("QUESTION: {q}\n\nANSWER:\n{ans}\n\nOn a scale from 0 to 1, how directly does the ANSWER address the QUESTION? "
         "Reply with only a number.")
CORRECT = ("QUESTION: {q}\n\nREFERENCE:\n{ref}\n\nCANDIDATE:\n{ans}\n\nOn a scale from 0 to 1, how factually consistent is "
           "the CANDIDATE with the REFERENCE (ignore style)? Reply with only a number.")
CTXREL = ("QUESTION: {q}\n\nPASSAGE:\n{ctx}\n\nDoes the PASSAGE contain information useful for answering the QUESTION? "
          "Reply with exactly one word: yes or no.")


class Judge:
    def __init__(self, model, url, cache_path):
        self.model, self.url, self.cache_path = model, url, cache_path
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    def ask(self, prompt, n=64):
        key = hashlib.sha1(prompt.encode()).hexdigest()
        if key not in self.cache:
            r = requests.post(f"{self.url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False,
                              "options": {"temperature": 0, "num_predict": n}}, timeout=600)
            self.cache[key] = r.json().get("response", "").strip()
        return self.cache[key]

    def save(self):
        json.dump(self.cache, open(self.cache_path, "w"))

    def yes(self, prompt):
        return self.ask(prompt, 4).lower().startswith("y")

    def num(self, prompt):
        m = re.search(r"\d+(?:\.\d+)?", self.ask(prompt, 8))
        return min(1.0, float(m.group())) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--gold", default="eval/gold_questions.json")
    ap.add_argument("--ctx-chars", type=int, default=6000)
    a = ap.parse_args()
    d = json.load(open(a.results))
    if a.judge_model == d["model"]:
        print(f"WARNING: judge == generator ({d['model']}): self-grading bias.", file=sys.stderr)
    gold = {q["id"]: q for q in json.load(open(a.gold))["questions"]}
    J = Judge(a.judge_model, a.url, a.results.replace(".json", f".judgecache.{a.judge_model.replace(':', '_').replace('/', '_')}.json"))

    for row in d.get("generation", []):
        row["scores"] = None
        if row["failed"]:
            continue
        g = gold.get(row["id"], {})
        ctx = "\n---\n".join(row["contexts"])[: a.ctx_chars]
        s = {"relevance": J.num(RELEV.format(q=row["question"], ans=row["answer"][:3000]))}
        if row["answerable"]:
            claims = [c.strip("-*• \t") for c in J.ask(CLAIMS.format(ans=row["answer"][:3000]), 400).splitlines() if len(c.strip()) > 15][:8]
            v = [J.yes(VERIFY.format(ctx=ctx, claim=c)) for c in claims]
            s["faithfulness"] = sum(v) / len(v) if v else None
            s["n_claims"] = len(claims)
            if g.get("reference_answer"):
                s["correctness"] = J.num(CORRECT.format(q=row["question"], ref=g["reference_answer"], ans=row["answer"][:3000]))
            rel = [J.yes(CTXREL.format(q=row["question"], ctx=c[:1500])) for c in row["contexts"]]
            s["context_relevance"] = sum(rel) / len(rel) if rel else None
        row["scores"] = s
        print(row["id"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
        J.save()
    d["judge_model"] = a.judge_model
    out = a.results.replace(".json", ".judged.json")
    json.dump(d, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
