#!/bin/bash

# Check if access_token is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <access_token>"
  exit 1
fi

ACCESS_TOKEN=$1

# Create and activate the virtual environment, then install dependencies
(
python -m venv MLPScaling
source MLPScaling/bin/activate
pip install --upgrade pip
pip install modelscope datasets transformers git+https://github.com/open-mmlab/mmengine
)

# Generate the experiments.csv file
python create_experiments.py

# Path to the CSV file containing layer counts, widths, and batch sizes
CSV_FILE="experiments.csv"

# Path to the Python script
PYTHON_SCRIPT="train_mlp_batches.py"

# Read the CSV file line by line
(
  read  # Skip the header line
  while IFS=, read -r layer_count width batch_size
  do
    echo "Running experiment with layer_count=$layer_count, width=$width, and batch_size=$batch_size"
    python $PYTHON_SCRIPT --layer_count $layer_count --width $width --batch_size $batch_size --access_token $ACCESS_TOKEN
  done
) < $CSV_FILE