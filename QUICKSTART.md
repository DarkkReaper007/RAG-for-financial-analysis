# Quick Start: Running the Improved RAG System

## Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install new dependency (config support)
pip install PyYAML

# Update requirements
pip install -r requirements.txt
```

## Run Gradio UI (Recommended)

```bash
source .venv/bin/activate
python app.py
```

Then open http://localhost:7860 in your browser.

### In the UI:

1. **Documents Tab** - Ingest PDFs (same as before)
2. **Query Tab** - Ask questions (same as before)
3. **Evaluation Tab** - **NEW**: Choose evaluation mode
   - "Rule-Based (Fast)" - ~30 seconds, heuristic-based
   - "LLM-Judge (Accurate)" - measured 15-20+ minutes on this hardware
     for 10 questions (each question needs 1 generation call + 3 judge
     calls; a full-prospectus generation call alone took 60-100s+ here)
   - Click "▶️ Run Evaluation"

## Run Command Line Benchmark

Compare both evaluation modes:

```bash
source .venv/bin/activate
python benchmark_evaluation.py
```

This will:
1. Ingest documents if needed
2. Run rule-based evaluation (~30s)
3. Run LLM evaluation (~5-10 min)
4. Show comparison table with improvements

## Switch to Cloud Model (Optional)

Mistral 7B is already pulled and working locally (`ollama pull mistral:7b`,
confirmed with a smoke-test query).

**Real measurement, not a guess:** on this project's hardware (a Mac with
no dedicated GPU), Mistral 7B is noticeably slower than Llama 3.2 3B for
full-prospectus queries — Llama 3.2 completed in ~100s while Mistral 7B
timed out at both the 120s local default and briefly at 120s again before
completing with a 240s timeout. If you're running locally on similar
hardware, either raise `local.timeout` in `config/models.yaml` (180-240s)
or accept the tradeoff for the larger model's better reasoning quality.
This is exactly the kind of latency a real cloud VM with more CPU/RAM (or
a GPU instance) would improve — which is the point of the section below.

To move it to a free cloud VM instead of running on this machine:

1. **Provision a free-tier VM** (e.g. Oracle Cloud "Always Free" Ampere A1 —
   4 vCPU / 24GB RAM, no cost ever) and install Ollama on it:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull mistral:7b
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```
   (Restrict access with a firewall rule or a tunnel like Tailscale/cloudflared.)

2. **Update config:**
   ```bash
   # Edit config/models.yaml
   cloud:
     model: "mistral:7b"
     url: "http://YOUR_CLOUD_VM_IP:11434"  # same Ollama API, remote host
     temperature: 0.1
     max_tokens: 2048
     timeout: 180
   ```

3. **Use cloud model:**
   ```bash
   export OLLAMA_URL="http://YOUR_CLOUD_VM_IP:11434"
   export OLLAMA_MODEL="mistral:7b"
   python app.py
   ```

## Test 3 Critical Questions

After ingesting documents, test these questions:

### Q1: Numerical Accuracy
**Question:** "What is the total share capital as per the prospectus?"
**Expected:** Exact ₹ figure with [Source: Company, Section: ..., Page: XX]
**Tests:** Faithfulness (no hallucinated numbers)

### Q2: Risk Analysis  
**Question:** "What are the main operational risks mentioned in the prospectus?"
**Expected:** List 3-5 specific operational risks
**Tests:** Answer relevance + context precision

### Q3: Citation Format
**Question:** "Who are the book running lead managers and their roles?"
**Expected:** Proper format with citations
**Tests:** Instruction-following, citation accuracy

## What You'll Notice

### Rule-Based Evaluation (Fast)
- ✓ Quick results (~30 seconds)
- ✗ May miss paraphrased hallucinations
- ✗ Only counts keyword presence

### LLM-Based Evaluation (Accurate)
- ✓ Semantic understanding (catches paraphrasing)
- ✓ Better relevance scoring
- ✓ Deeper context evaluation
- ✗ Takes much longer — measured ~60-100s per model call on this hardware,
  and LLM-judge makes 3 extra calls per question on top of generation

## Confirming the Improvement

The specific score deltas depend on your corpus and model — this repo
doesn't ship pre-measured numbers to avoid presenting guesses as fact.
Run it yourself and read the actual comparison table it prints:
```bash
python benchmark_evaluation.py
```

## Configuration Examples

### Use Local Ollama (Default)
```bash
# config/models.yaml already set to local
python app.py
```

### Use Cloud Inference
```bash
export OLLAMA_URL="http://YOUR_CLOUD_VM_IP:11434"
python app.py
```

### Override Temperature (More Creative)
```bash
export OLLAMA_TEMPERATURE="0.5"
python app.py
```

### Use LLM Evaluation
```bash
export USE_LLM_JUDGE=true
python app.py
# Then select "LLM-Judge (Accurate)" in Evaluation tab
```

## Troubleshooting

**Q: "ModuleNotFoundError: yaml"**
```bash
pip install PyYAML
```

**Q: "Failed to connect to Ollama"**
- Make sure Ollama is running: `ollama serve`
- Check config URL matches: `http://localhost:11434`

**Q: "Evaluation hangs"**
- LLM evaluation is slow (5-10 min is normal)
- Use `--local-only` flag to skip LLM: `python benchmark_evaluation.py --local-only`

**Q: "Cloud connection errors"**
- No API key needed — it's your own Ollama instance, not a hosted API
- Verify the VM's URL/port is correct in config and reachable (firewall rules, tunnel status)
- Test with curl first: `curl http://YOUR_CLOUD_VM_IP:11434/api/tags`

## Files Created/Modified

**New Files:**
- ✨ `src/config/loader.py` - Configuration system
- ✨ `src/evaluation/llm_evaluator.py` - LLM-based evaluation
- ✨ `config/models.yaml` - Configuration file
- ✨ `benchmark_evaluation.py` - Evaluation benchmark script
- ✨ `IMPROVEMENTS.md` - Full documentation

**Modified Files:**
- 📝 `src/generation/generator.py` - Config support
- 📝 `src/pipeline.py` - Evaluator selection
- 📝 `app.py` - UI toggles + config loading
- 📝 `requirements.txt` - Added PyYAML

## Next Steps

1. ✅ Try the Gradio UI with both evaluation modes
2. ✅ Run the benchmark script to see improvements
3. ✅ Test the 3 critical questions
4. 🚀 Deploy cloud model for production use

## Support

For detailed documentation, see:
- `IMPROVEMENTS.md` - Full technical details
- `benchmark_evaluation.py` - Example usage

---

**Happy RAG-ing! 🚀**
