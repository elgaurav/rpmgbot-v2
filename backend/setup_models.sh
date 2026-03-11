#!/bin/bash

# RPMG RAG Assistant - Model Setup Script
# This script pulls optimized Ollama models for <30s response time

echo "============================================================"
echo "🚀 RPMG RAG Assistant - Model Setup"
echo "============================================================"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo "   Install from: https://ollama.ai"
    exit 1
fi

echo "✅ Ollama detected"
echo ""

# Function to pull model with error handling
pull_model() {
    local model=$1
    local description=$2
    
    echo "📥 Pulling $model..."
    echo "   Purpose: $description"
    
    if ollama pull "$model"; then
        echo "   ✅ $model ready"
    else
        echo "   ❌ Failed to pull $model"
        return 1
    fi
    echo ""
}

echo "============================================================"
echo "RECOMMENDED MODELS (choose based on your hardware)"
echo "============================================================"
echo ""
echo "1. llama3.2:3b (MINIMUM - 4GB RAM, CPU OK, ~10-20s response)"
echo "2. llama3.1:8b (RECOMMENDED - 8GB RAM, ~5-15s response)"  
echo "3. phi3:medium (ALTERNATIVE - 8GB RAM, faster than llama)"
echo "4. llama3.1:70b (BEST - 48GB+ VRAM, <5s response, requires GPU)"
echo ""

# Pull the minimum viable model
echo "============================================================"
echo "STEP 1: Installing MINIMUM viable model (llama3.2:3b)"
echo "============================================================"
echo ""
pull_model "llama3.2:3b" "Fast CPU-friendly model, much better than 1b"

# Ask about better models
echo "============================================================"
echo "STEP 2: Optional better models"
echo "============================================================"
echo ""
read -p "Do you have 8GB+ RAM and want better quality? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Choose your model:"
    echo "1) llama3.1:8b (Best balance of speed/quality)"
    echo "2) phi3:medium (Faster, slightly lower quality)"
    echo "3) Both"
    echo ""
    read -p "Enter choice (1/2/3): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            pull_model "llama3.1:8b" "High-quality responses, good speed"
            ;;
        2)
            pull_model "phi3:medium" "Very fast, good quality"
            ;;
        3)
            pull_model "llama3.1:8b" "High-quality responses, good speed"
            pull_model "phi3:medium" "Very fast, good quality"
            ;;
    esac
fi

echo ""
read -p "Do you have 48GB+ GPU VRAM for best performance? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    pull_model "llama3.1:70b" "Best quality, <5s responses with GPU"
fi

echo ""
echo "============================================================"
echo "✅ MODEL SETUP COMPLETE"
echo "============================================================"
echo ""
echo "📋 Installed models:"
ollama list
echo ""
echo "⚙️  Next steps:"
echo "   1. Edit config.py and set LLM_MODEL to your chosen model"
echo "   2. Run: python ingest_pro.py (to create vector database)"
echo "   3. Run: python main.py (to start server)"
echo ""
echo "💡 Recommended config.py settings:"
echo "   - For CPU: LLM_MODEL = 'llama3.2:3b'"
echo "   - For 8GB RAM: LLM_MODEL = 'llama3.1:8b' or 'phi3:medium'"
echo "   - For GPU 48GB+: LLM_MODEL = 'llama3.1:70b'"
echo ""
echo "============================================================"