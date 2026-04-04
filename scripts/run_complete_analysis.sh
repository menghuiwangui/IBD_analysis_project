#!/bin/bash
echo "[INFO] Starting IBD Analysis Pipeline..."
cd "$(dirname "$0")/../src/data_collection"
python run_complete_collection.py --query "IBD"
cd ../ai_analysis
python model_training.py
echo "[SUCCESS] Analysis completed!"