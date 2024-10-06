#! /bin/bash

echo "> example.sh"

# export OBJAVERSE_DATASETS_DIR="/Users/ebrown/datasets/mem/objaverse_vida/"

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "script_dir: $script_dir"

# export OBJAVERSE_DATASETS_DIR=$(realpath $script_dir/../procthor/databases)
export OBJAVERSE_DATASETS_DIR=$(realpath ~/datasets/mem/objaverse_vida/procthor_databases_2023_07_28)
export OBJAVERSE_ASSETS_DIR=$(realpath ~/datasets/mem/objaverse_vida/processed_2023_07_28)

# if doesnt exist, error
if [ ! -d "$OBJAVERSE_DATASETS_DIR" ]; then
    echo "Error: OBJAVERSE_DATASETS_DIR does not exist: $OBJAVERSE_DATASETS_DIR"
    exit 1
fi
echo "OBJAVERSE_DATASETS_DIR: $OBJAVERSE_DATASETS_DIR"

python scripts/example.py

echo "< example.sh"
