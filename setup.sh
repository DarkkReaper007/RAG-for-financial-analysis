#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# RAG System for Indian IPO Prospectuses — Setup Script
# ─────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   RAG System for Indian IPO Prospectuses — Setup         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ─── Step 1: Python Virtual Environment ──────────────────────────────
echo "→ [1/5] Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  ✓ Created .venv"
else
    echo "  ✓ .venv already exists"
fi

source .venv/bin/activate
pip install --upgrade pip -q

# ─── Step 2: Install Dependencies ────────────────────────────────────
echo "→ [2/5] Installing Python dependencies..."
pip install -r requirements.txt -q
pip install qdrant-client -q
echo "  ✓ Dependencies installed"

# ─── Step 3: Verify Ollama ───────────────────────────────────────────
echo "→ [3/5] Checking Ollama..."
if command -v ollama &>/dev/null || [ -f /usr/local/bin/ollama ]; then
    OLLAMA_CMD="${OLLAMA_CMD:-/usr/local/bin/ollama}"
    if $OLLAMA_CMD list 2>/dev/null | grep -q "llama3.2"; then
        echo "  ✓ Ollama running with Llama 3.2"
    else
        echo "  ⚠ Llama 3.2 not found. Pull it with: ollama pull llama3.2:latest"
    fi
else
    echo "  ⚠ Ollama not found. Install from: https://ollama.com"
fi

# ─── Step 4: Verify Docker & Qdrant ─────────────────────────────────
echo "→ [4/5] Checking Docker & Qdrant..."
if command -v docker &>/dev/null || [ -f /usr/local/bin/docker ]; then
    DOCKER_CMD="${DOCKER_CMD:-/usr/local/bin/docker}"
    if $DOCKER_CMD ps 2>/dev/null | grep -q "qdrant"; then
        echo "  ✓ Qdrant container running"
    else
        echo "  Starting Qdrant container..."
        $DOCKER_CMD run -d --name qdrant \
            -p 6333:6333 -p 6334:6334 \
            -v "$(pwd)/qdrant_storage:/qdrant/storage" \
            qdrant/qdrant:latest 2>/dev/null || \
        $DOCKER_CMD start qdrant 2>/dev/null || \
        echo "  ⚠ Could not start Qdrant. Run manually or app will use ChromaDB."
    fi
else
    echo "  ⚠ Docker not found. App will fall back to ChromaDB."
fi

# ─── Step 5: Verify PDF Data ────────────────────────────────────────
echo "→ [5/5] Checking data directory..."
PDF_COUNT=$(find ./data/prospectuses -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')
if [ "$PDF_COUNT" -gt 0 ]; then
    echo "  ✓ Found $PDF_COUNT PDF files in data/prospectuses/"
else
    echo "  ⚠ No PDFs found in data/prospectuses/. Place IPO prospectus PDFs there."
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Setup Complete!                                        ║"
echo "║                                                          ║"
echo "║   To run the application:                                ║"
echo "║     source .venv/bin/activate                            ║"
echo "║     python app.py                                        ║"
echo "║                                                          ║"
echo "║   Then open: http://localhost:7860                       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
