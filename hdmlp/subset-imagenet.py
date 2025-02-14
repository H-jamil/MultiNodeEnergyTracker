import os
import shutil
import random
from tqdm import tqdm  # Import tqdm for the progress bar

# Source directory with the full dataset.
source_dir = "/home/cc/train/ILSVRC/Data/CLS-LOC/train"

# Target directory where the subset will be copied.
target_dir = "/home/cc/train/subset"

# Define target subset size: 10GB.
target_size_bytes = 5 * 1024 * 1024 * 1024  # 10 GB in bytes

# Step 1: Walk through the source directory and collect file paths and their sizes.
all_files = []
for root, dirs, files in os.walk(source_dir):
    for filename in files:
        file_path = os.path.join(root, filename)
        try:
            size = os.path.getsize(file_path)
            all_files.append((file_path, size))
        except OSError as e:
            print(f"Could not get size for {file_path}: {e}")

# Step 2: Shuffle the list so the selection is random.
random.shuffle(all_files)

# Step 3: Select files until the cumulative size reaches or exceeds 10GB.
selected_files = []
cumulative_size = 0

for file_path, size in all_files:
    if cumulative_size >= target_size_bytes:
        break
    selected_files.append(file_path)
    cumulative_size += size

print(f"Selected {len(selected_files)} files with total size: {cumulative_size / (1024**3):.2f} GB")

# Step 4: Copy the selected files to the target directory while preserving the original directory structure.
for file_path in tqdm(selected_files, desc="Copying files"):
    # Determine the relative path with respect to the source directory.
    relative_path = os.path.relpath(file_path, source_dir)
    # Create the destination path.
    dest_path = os.path.join(target_dir, relative_path)
    # Ensure the destination directory exists.
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        shutil.copy2(file_path, dest_path)
    except Exception as e:
        print(f"Failed to copy {file_path} to {dest_path}: {e}")

print("Subset creation completed.")
