#!/bin/bash

# Check if access_token is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <access_token>"
  exit 1
fi

ACCESS_TOKEN=$1

# Create and activate the conda environment, then install dependencies
(
  export HF_ENDPOINT=https://hf-mirror.com
  conda create -n MLPScaling python=3.8 -y
  conda activate MLPScaling
  pip install --upgrade pip
  pip install modelscope datasets transformers git+https://github.com/open-mmlab/mmengine
)

# Generate the experiments.csv file
python create_experiments.py

# Navigate to the models directory
cd models || exit

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
  git init
fi

# Configure git environment variables
export GIT_AUTHOR_NAME='BirdL Canary ARC'
export GIT_AUTHOR_EMAIL='puffywastaken310@gmail.com'

# Define model ID and git remote URL
model_id="puffy310/MLPScaling"
git_remote_url="https://oath2:${ACCESS_TOKEN}@modelscope.cn/git/${model_id}.git"

# Add the remote if not already added
if ! git remote | grep -q modelscope; then
  git remote add modelscope $git_remote_url
fi

# Perform "status" branch test
git checkout main
git pull modelscope main

# Create and switch to "status" branch
git checkout -b status

# Create a test file
touch test_status.txt
git add test_status.txt
git commit -m "Test commit for status branch"

# Push the "status" branch to remote
git push -u modelscope status

# Check if the push was successful
if [ $? -ne 0 ]; then
  echo "Status branch test failed. Exiting."
  exit 1
fi

# Clean up the test file and commit
git rm test_status.txt
git commit -m "Remove test file"
git push modelscope status

# Switch back to main branch and delete the status branch
git checkout main
git branch -d status

echo "Status branch test successful."

# Path to the CSV file containing experiment parameters
CSV_FILE="../experiments.csv"
PYTHON_SCRIPT="../train_mlp_batches.py"

# Read the CSV file line by line
(
  read  # Skip the header line
  while IFS=, read -r layer_count width batch_size _
  do
    echo "Running experiment with layer_count=$layer_count, width=$width, and batch_size=$batch_size"
    python $PYTHON_SCRIPT --layer_count $layer_count --width $width --batch_size $batch_size --access_token $ACCESS_TOKEN --upload_checkpoint --delete_checkpoint

    # Handle git operations for each experiment
    model_folder="mlp_l$layer_count_w$width"
    cd $model_folder || exit

    # Create branch name
    branch_name="l$layer_count-w$width"

    # Checkout or create branch
    git checkout -b $branch_name || git checkout $branch_name

    # Add all files to be tracked
    git add .

    # Commit the changes
    commit_message="Experiment $branch_name"
    git commit -m "$commit_message"

    # Push the commit to the remote
    git push -u modelscope $branch_name

    # Create and push a tag for the version
    version_tag="l$layer_count_w$width"
    git tag $version_tag
    git push modelscope $version_tag

    # Switch back to main branch
    git checkout main

    # Delete the local branch
    git branch -d $branch_name

    # Return to the models directory
    cd ..
  done
) < $CSV_FILE

# Return to the original directory
cd ..
