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
    local checkpoint_file=$2

    if [ -f "$checkpoint_file" ]; then
        echo "Evaluating model in $model_dir using checkpoint $(basename "$checkpoint_file")..."
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
        # Check if last_checkpoint file exists
        if [ -f "$model_dir/last_checkpoint" ]; then
            # Read the checkpoint filename from last_checkpoint
            checkpoint_filename=$(cat "$model_dir/last_checkpoint")
            checkpoint_file="$model_dir/$checkpoint_filename"
            # Extract epoch number from the filename
            epoch=$(basename "$checkpoint_file" | sed 's/epoch_//;s/\.pth//')
            # Evaluate the model using the checkpoint file
            evaluate_model "$model_dir" "$checkpoint_file"
        else
            # Fallback to finding the last epoch from epoch_*.pth files
            echo "last_checkpoint file not found in $model_dir. Finding the last epoch..."
            last_epoch=$(ls "$model_dir"/epoch_*.pth 2>/dev/null | grep -o 'epoch_[0-9]*' | grep -o '[0-9]*' | sort -nr | head -n 1)
            if [ -n "$last_epoch" ]; then
                checkpoint_file="$model_dir/epoch_$last_epoch.pth"
                evaluate_model "$model_dir" "$checkpoint_file"
            else
                echo "No checkpoints found in $model_dir"
            fi
        fi
    else
        # Evaluate all checkpoints
        for checkpoint_file in "$model_dir"/epoch_*.pth; do
            evaluate_model "$model_dir" "$checkpoint_file"
        done
    fi
done
