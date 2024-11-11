#!/bin/bash

# Path to the CSV file containing layer counts, widths, and batch sizes
CSV_FILE="experiments.csv"

# Path to the Python script
PYTHON_SCRIPT="train_mlp.py"

# Read the CSV file line by line
{
  read  # Skip the header line
  while IFS=, read -r layer_count width batch_size
  do
    echo "Running experiment with layer_count=$layer_count, width=$width, and batch_size=$batch_size"
    python $PYTHON_SCRIPT --layer_count $layer_count --width $width --batch_size $batch_size
  done
} < $CSV_FILE