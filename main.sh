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

# Path to the CSV file containing layer counts, widths, and batch sizes
CSV_FILE="experiments.csv"

# Path to the Python script
PYTHON_SCRIPT="train_mlp_batches.py"

# Read the CSV file line by line
(
  read  # Skip the header line
  while IFS=, read -r layer_count width batch_size _
  do
    echo "Running experiment with layer_count=$layer_count, width=$width, and batch_size=$batch_size"
    python $PYTHON_SCRIPT --layer_count $layer_count --width $width --batch_size $batch_size --access_token $ACCESS_TOKEN --upload_checkpoint --delete_checkpoint

    # Perform git operations and directory deletion after each experiment
    model_folder="mlp_l$layer_count_w$width"
    model_folder_path="models/$model_folder"
    results_folder="results/results_l$layer_count_w$width.txt"

    # Set git environment variables
    export GIT_AUTHOR_NAME='BirdL Canary ARC'
    export GIT_AUTHOR_EMAIL='puffywastaken310@gmail.com'

    # Define model ID and git remote URL
    model_id="puffy310/MLPScaling"
    git_remote_url="https://oath2:${ACCESS_TOKEN}@modelscope.cn/git/${model_id}.git"

    # Change directory to model_folder_path
    cd $model_folder_path || exit

    # Initialize git repository if not already initialized
    if [ ! -d .git ]; then
      git init
    fi

    # Create .gitignore file if not exists
    if [ ! -f .gitignore ]; then
      echo '*' > .gitignore
      echo '!.gitignore' >> .gitignore
      echo 'model.yaml' >> .gitignore
      echo 'results.txt' >> .gitignore
      echo 'models/' >> .gitignore
    fi

    # Create branch name
    branch_name="l$layer_count-w$width"

    # Checkout or create branch
    git checkout -b $branch_name || git checkout $branch_name

    # Add all files to be tracked
    git add .

    # Commit the changes
    commit_message="Upload model l$layer_count w$width"
    git commit -m "$commit_message"

    # Add the remote if not already added
    if ! git remote | grep -q modelscope; then
      git remote add modelscope $git_remote_url
    fi

    # Push the commit to the remote
    git push modelscope $branch_name

    # Create and push a tag for the version
    version_tag="l$layer_count_w$width"
    git tag $version_tag
    git push modelscope $version_tag

    # Delete the local model directory if specified
    if [ "$delete_checkpoint" = true ]; then
      rm -rf $model_folder_path
      echo "Deleted local checkpoint folder: $model_folder_path"
    fi

    # Return to the original directory
    cd -

  done
) < $CSV_FILE
