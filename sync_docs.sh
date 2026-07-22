#!/usr/bin/env bash
# Re-copies the runtime source files into docs/ so the GitHub Pages (stlite) build
# stays in sync after you edit app.py or any agents/database/utils file.
# Run this before committing, whenever you change app code.
set -euo pipefail
cd "$(dirname "$0")"

cp app.py docs/app.py
cp agents/__init__.py agents/cleaning_agent.py agents/sentiment_agent.py agents/embedding_agent.py \
   agents/theme_agent.py agents/trend_agent.py agents/rag_agent.py agents/qa_agent.py docs/agents/
cp database/__init__.py database/chroma_store.py docs/database/
cp utils/__init__.py utils/helpers.py docs/utils/
cp data/sample_reviews.csv docs/data/

echo "docs/ synced from source."
