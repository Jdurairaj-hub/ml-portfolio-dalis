#!/bin/bash

# Copy notebooks and documents to docs folder
cp prediction-model/ML_Simple_LongFlat_SPY.ipynb docs/ 2>/dev/null || true
cp sentiment-analysis/main.py docs/sentiment-analysis-main.py 2>/dev/null || true
cp volatility-forecasting/main.py docs/volatility-forecasting-main.py 2>/dev/null || true
cp asset-correlation-analysis/README.md docs/asset-correlation-readme.md 2>/dev/null || true

# Remove .ipynb_checkpoints from docs/
rm -rf docs/.ipynb_checkpoints

# Launch mkdocs server
mkdocs serve
