#!/bin/bash

# Parse arguments
# Initialize variables with empty values
#!/bin/bash

# Initialize variables with empty values
ACCESS_TOKEN=""
LAST_RUN=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -a|--access-token)
            if [[ -n "$2" ]]; then
                ACCESS_TOKEN="$2"
                shift  # Skip the argument after -a|--access-token
            else
                echo "Error: -a or --access-token option requires an argument."
                exit 1
            fi
            ;;
        -l|--last-run)
            if [[ -n "$2" ]]; then
                LAST_RUN="$2"
                shift  # Skip the argument after -l|--last-run
            else
                echo "Error: -l or --last-run option requires an argument."
                exit 1
            fi
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift  # Move to the next option
done

# Function to set up git and perform status branch test
setup_git() {
    local access_token=$1

    # Configure git credential helper
    git config --global credential.helper cache
    git config --global credential.helper 'cache --timeout=999999'

    # Approve and cache the credentials
    git credential approve << EOF
protocol=https
host=modelscope.cn
username=puffy310
password=$access_token
EOF

    # Ensure models directory exists and is a Git repository
    if [ ! -d "models/.git" ]; then
        mkdir -p models
        cd models || exit
        git init
        cd ..
    fi

    # Configure git environment variables
    export GIT_AUTHOR_NAME='BirdL Canary ARC'
    export GIT_AUTHOR_EMAIL='puffywastaken310@gmail.com'

    # Define model ID and git remote URL
    model_id="puffy310/MLPScaling"
    git_remote_url="https://${access_token}@modelscope.cn/${model_id}.git"

    # Add the remote if not already added
    if ! git -C models remote | grep -q modelscope; then
        cd models || exit
        git remote add modelscope $git_remote_url
        cd ..
    fi

    # Perform "status" branch test
    (
        cd models || exit
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
    )
}

# Function to handle git operations for each experiment
git_operations() {
    local layer_count=$1
    local width=$2
    local model_folder="models/mlp_l$layer_count_w$width"
    local original_dir=$(pwd)

    cd "$model_folder" || exit

    branch_name="l$layer_count-w$width"
    git checkout -b $branch_name || git checkout $branch_name

    git add .
    commit_message="Experiment $branch_name"
    git commit -m "$commit_message"

    git push -u modelscope $branch_name
    if [ $? -ne 0 ]; then
        echo "Push of branch $branch_name failed. Exiting."
        exit 1
    fi

    version_tag="l$layer_count_w$width"
    git tag $version_tag
    git push modelscope $version_tag
    if [ $? -ne 0 ]; then
        echo "Push of tag $version_tag failed. Exiting."
        exit 1
    fi

    git checkout main
    git branch -d $branch_name

    cd "$original_dir"
}

# Function to process a CSV file
process_csv_file() {
    local csv_file=$1
    local skip
    if [ -z "$LAST_RUN" ]; then
        skip=false
    else
        skip=true
    fi
    tail -n +2 "$csv_file" | while IFS=, read -r layer_count width _ batch_size; do
        experiment_id="l$layer_count-w$width"
        if [ "$experiment_id" = "$LAST_RUN" ]; then
            # Set skip to false to start processing from the next experiment
            skip=false
            echo "Found last run experiment: $experiment_id. Continuing from next experiment."
            continue  # Skip the current experiment
        fi
        if [ "$skip" = false ]; then
            echo "Running experiment with layer_count=$layer_count, width=$width, and batch_size=$batch_size"
            python $PYTHON_SCRIPT --layer_count "$layer_count" --width "$width" --batch_size "$batch_size" --access_token "$ACCESS_TOKEN" --upload_checkpoint --delete_checkpoint

            if [ -n "$ACCESS_TOKEN" ]; then
                git_operations "$layer_count" "$width"
            fi
        fi
    done
}

# Check if access token is provided
if [ -z "$ACCESS_TOKEN" ]; then
    echo "Access token not provided. Skipping git operations."
else
    setup_git $ACCESS_TOKEN
fi

# Ensure models directory exists
mkdir -p models

(
    export HF_ENDPOINT=https://hf-mirror.com
    pip install --upgrade pip
    pip install modelscope datasets transformers git+https://github.com/open-mmlab/mmengine
)

# Generate the experiments.csv file
python create_experiments.py

# Define CSV file variables
#CSV_FILE1="experiments.csv"
CSV_FILE2="experiments_expanded.csv"
PYTHON_SCRIPT="train_mlp_batches.py"

# Temp solution until I can get last_experiment to work

# Process each CSV file
#process_csv_file $CSV_FILE1
process_csv_file $CSV_FILE2
