#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-deterministic}"

if [ "$MODE" = "llm" ]; then
    echo "Launching LangGraph multi-agent UI (requires OPENAI_API_KEY)..."
    python -m src.ui_gradio_llm
else
    echo "Launching deterministic Ops Console (no LLM required)..."
    python -m src.ui_gradio
fi
