#!/bin/bash

# =============================================================================
# Script Name: run_training.sh
# Description: Sequentially runs training commands for different epochs and
#              modes (ramfs and cephfs), saving outputs to corresponding files.
# Usage:        sudo ./run_training.sh
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Define an array of epoch numbers
epochs=(2 4 8)

# Define paths and parameters
PYTHON_EXEC="/home/cc/.pyenv/versions/training_gpu/bin/python"  # Absolute path to Python executable
TRAIN_SCRIPT="/home/cc/MultiNodeEnergyTracker/train_ceph_ramdisk.py"  # Absolute path to the training script
JSON_FILE="/mnt/mycephfs/imagenet/selected_files_20GB_train.json"
THREADS=8
GROUP_SIZE=128

# Define output directory
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"

# Function to display usage information
usage() {
    echo "Usage: sudo ./run_training.sh"
    exit 1
}

# Check if the script is run as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run with sudo or as root."
    usage
fi

# Verify that PYTHON_EXEC exists
if [[ ! -x "$PYTHON_EXEC" ]]; then
    echo "Error: Python executable not found at $PYTHON_EXEC"
    exit 1
fi

# Verify that TRAIN_SCRIPT exists
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi

# Iterate over each epoch number
for epoch in "${epochs[@]}"; do
    echo "----------------------------------------"
    echo "Starting training for Epoch: $epoch in ramfs mode"
    echo "----------------------------------------"
    
    # Define output file for ramfs mode
    RAMFS_OUTPUT="${OUTPUT_DIR}/epoch_${epoch}_ramfs_result.txt"
    
    # Run the training script in ramfs mode and save output
    sudo -E "$PYTHON_EXEC" "$TRAIN_SCRIPT" ramfs \
        --json_file "$JSON_FILE" \
        --threads "$THREADS" \
        --group_size "$GROUP_SIZE" \
        --epoch "$epoch" | tee "$RAMFS_OUTPUT"
    
    echo "Completed training for Epoch: $epoch in ramfs mode. Output saved to $RAMFS_OUTPUT"
    
    echo "----------------------------------------"
    echo "Starting training for Epoch: $epoch in cephfs mode"
    echo "----------------------------------------"
    
    # Define output file for cephfs mode
    CEPHFS_OUTPUT="${OUTPUT_DIR}/epoch_${epoch}_cephfs_result.txt"
    
    # Run the training script in cephfs mode and save output
    sudo -E "$PYTHON_EXEC" "$TRAIN_SCRIPT" cephfs \
        --json_file "$JSON_FILE" \
        --threads "$THREADS" \
        --group_size "$GROUP_SIZE" \
        --epoch "$epoch" | tee "$CEPHFS_OUTPUT"
    
    echo "Completed training for Epoch: $epoch in cephfs mode. Output saved to $CEPHFS_OUTPUT"
    
    echo -e "\n"
done

echo "All training runs completed successfully."
