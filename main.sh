#!/bin/bash

# Initialize variables
LAST_RUN=""

# Ensure models directory exists
mkdir -p models

# Find the most recent model directory
last_model_dir=$(ls -t models/mlp_l* 2>/dev/null | head -n 1)

# Determine the last run experiment ID
if [ -n "$last_model_dir" ]; then
    experiment_id=$(basename "$last_model_dir" | sed -e 's/^mlp_//')
    LAST_RUN=$experiment_id
    echo "Last run experiment: $LAST_RUN"
else
    echo "No previous experiments found."
fi

# Define CSV file variables
CSV_FILE1="experiments.csv"
CSV_FILE2="experiments_expanded.csv"
PYTHON_SCRIPT="train_mlp_batches.py"

# Function to process the CSV file
process_csv_file() {
    local csv_file=$1
    local skip=true
    tail -n +2 "$csv_file" | while IFS=, read -r layer_count width _ batch_size; do
        experiment_id="l$layer_count-w$width"
        model_folder="models/mlp_l$layer_count_w$width"
        if [ -d "$model_folder" ]; then
            echo "Experiment $experiment_id already completed. Skipping."
            continue  # Skip to the next experiment
        fi
        if [ "$experiment_id" = "$LAST_RUN" ]; then
            skip=false
            echo "Found last run experiment: $experiment_id. Continuing from next experiment."
            continue  # Skip the current experiment
        fi
        if [ "$skip" = false ]; then
            echo "Running experiment with layer_count=$layer_count, width=$width, and batch_size=$batch_size"
            python $PYTHON_SCRIPT --layer_count "$layer_count" --width "$width" --batch_size "$batch_size"
            # Mark the experiment as completed
            mkdir -p "$model_folder"
        fi
    done
}

# Process the CSV file
process_csv_file $CSV_FILE
