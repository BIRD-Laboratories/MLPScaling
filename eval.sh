#!/bin/bash

# Parse arguments
eval_last_only=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--last-only)
            eval_last_only=true
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Function to evaluate a model checkpoint
evaluate_model() {
    local model_dir=$1
    local epoch=$2
    local checkpoint_file="$model_dir/epoch_$epoch.pth"

    if [ -f "$checkpoint_file" ]; then
        echo "Evaluating model in $model_dir at epoch $epoch..."
        python eval_model.py --model_dir "$model_dir" --checkpoint "$checkpoint_file"
    else
        echo "Checkpoint file not found: $checkpoint_file"
    fi
}

# Find all model folders
model_folders=$(find models -maxdepth 1 -type d -name 'mlp_l*_w*')

# Evaluate each model folder
for model_dir in $model_folders; do
    echo "Processing model folder: $model_dir"

    if [ "$eval_last_only" = true ]; then
        # Find the last checkpoint
        last_epoch=$(ls "$model_dir" | grep 'epoch_' | sed 's/epoch_//;s/\.pth//' | sort -n | tail -n 1)
        if [ -n "$last_epoch" ]; then
            evaluate_model "$model_dir" "$last_epoch"
        else
            echo "No checkpoints found in $model_dir"
        fi
    else
        # Evaluate all checkpoints
        for checkpoint_file in "$model_dir"/epoch_*.pth; do
            epoch=$(basename "$checkpoint_file" | sed 's/epoch_//;s/\.pth//')
            evaluate_model "$model_dir" "$epoch"
        done
    fi
done
