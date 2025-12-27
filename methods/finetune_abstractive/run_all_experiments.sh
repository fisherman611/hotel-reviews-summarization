#!/bin/bash
# Run all 9 experiments (3 models × 3 recipes)
# Usage: ./run_all_experiments.sh [--dry-run]

set -e

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

MODELS=("gemma3" "qwen25" "llama32")
RECIPES=("synth_100" "human_25" "mixed")

cd "$(dirname "$0")/../.."

echo "========================================"
echo "Running all 9 finetuning experiments"
echo "========================================"

for model in "${MODELS[@]}"; do
    for recipe in "${RECIPES[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Training: model=${model}, recipe=${recipe}"
        echo "----------------------------------------"
        
        CMD="python methods/finetune_abstractive/train.py --model ${model} --recipe ${recipe}"
        
        if [ "$DRY_RUN" == true ]; then
            echo "[DRY RUN] Would run: $CMD"
        else
            $CMD
        fi
    done
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
